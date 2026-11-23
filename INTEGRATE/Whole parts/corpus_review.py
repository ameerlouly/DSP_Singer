"""Lossless handoff, final annotation edits, and portable corpus export."""
import copy
import hashlib
import json
import math
from pathlib import Path
import uuid
from datetime import datetime, timezone
import numpy as np
import soundfile as sf
import asr_pipeline as asr
import step4_review as review
import step5_snr as snr

RATE=16000
VOICE=('speech','mixed')


def stamp():return datetime.now(timezone.utc).isoformat(timespec='seconds')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def source_signature(processing):
    if not processing.audio or not processing.session:return None
    return digest({'audio':str(processing.audio),'size':processing.audio.stat().st_size,
                   'mtime':processing.audio.stat().st_mtime_ns,'step4':processing.session,
                   'snr':processing.snr_results})


def from_processing(processing):
    """Handoff the exact reviewed continuous original source interval."""
    processing.require_approved()
    queue=review.approved_playback_queue(processing.session,processing.audio)
    by_id={s['id']:s for s in processing.session['segments']}
    qualities={s['id']:s for s in processing.snr_results} if processing.snr_snapshot==processing.session else {}
    records=[]
    for path,a,b,label,ident in queue:
        item=by_id[ident];original=item.get('original') or {}
        a,b=review.require_pdf_review(processing.session,item)
        path=str(Path(processing.audio).resolve())
        spans=[dict(start=a,end=b,clip_start=0.,clip_end=b-a)]
        quality=qualities.get(ident,{})
        raw=dict(id=ident,label=label,source_spans=spans,
                 snr_db=quality.get('snr_db'),quality=quality.get('quality','unavailable'))
        record=asr._make_segment(raw,len(records)+1,processing.duration)
        pad_a=pad_b=0.
        record.update(start=spans[0]['start'],end=spans[-1]['end'],duration=b-a,source_spans=copy.deepcopy(spans),
                      audio_pieces=[dict(path=str(Path(path).resolve()),start=a,end=b,synthetic_start=pad_a,synthetic_end=pad_b)],
                      asr_a_done=False,asr_b_done=False,source_mode='continuous_source',
                      source_metadata=copy.deepcopy(item),quality_history=copy.deepcopy(quality.get('quality_changes',[])),
                      human_changes=copy.deepcopy([x for x in processing.session.get('changes',[]) if x.get('item_id')==ident]+quality.get('quality_changes',[])),
                      source_origin=str(processing.audio),original_approval='approved')
        record.update(snr.corpus_fields(quality))
        record['audio_signature']=audio_signature(record)
        record['snr_audio_signature']=record['audio_signature'] if quality.get('analysis_version')==2 else None
        if label not in VOICE:record.update(status='not_applicable',transcript_status='not_applicable')
        records.append(record)
    return records


def audio_signature(record):
    pieces=[]
    for piece in record['audio_pieces']:
        path=Path(piece['path']);stat=path.stat()
        pieces.append(dict(piece,size=stat.st_size,mtime=stat.st_mtime_ns))
    return digest(dict(pieces=pieces,label=record['label']))


def read_audio(record,for_snr=False):
    arrays=[]
    for piece in record['audio_pieces']:
        start=float(piece['start']);end=float(piece['end'])
        if for_snr:
            start+=float(piece.get('synthetic_start',0));end-=float(piece.get('synthetic_end',0))
        if end<=start:continue
        with sf.SoundFile(piece['path']) as f:
            if (f.samplerate,f.channels,f.subtype)!=(RATE,1,'PCM_16'):
                raise ValueError('Expected 16 kHz mono PCM-16 clip audio.')
            a,b=round(start*RATE),round(end*RATE)
            if not 0<=a<b<=len(f):raise ValueError('Audio clip boundaries are outside the saved file.')
            f.seek(a);samples=f.read(b-a,dtype='float32')
            if len(samples)!=b-a:raise ValueError('A clip audio file is truncated.')
            arrays.append(samples)
    if not arrays:raise ValueError('No usable clip audio.')
    return np.concatenate(arrays)


def compare(record):
    record.update(normalized_transcript_a=asr.normalize_arabic(record.get('transcript_a','')),
                  normalized_transcript_b=asr.normalize_arabic(record.get('transcript_b','')),
                  normalization_version=asr.NORMALIZATION_VERSION,
                  agreement_rule_version=asr.AGREEMENT_RULE_VERSION)
    if record['label'] not in VOICE:
        record.update(status='not_applicable',transcript_status='not_applicable',
                      automatic_status='not_applicable',agreement_reason='Music/silence is not sent to ASR.',
                      cer=None,wer=None,word_difference=None,word_substitutions=None,
                      word_insertions=None,word_deletions=None);return
    if not (record.get('asr_a_done') and record.get('asr_b_done')):
        record.update(cer=None,wer=None,word_difference=None,word_substitutions=None,
                      word_insertions=None,word_deletions=None,automatic_status='needs_review',
                      agreement_reason='Waiting for both ASR attempts.')
        if not record.get('human_override'):
            record.update(status='needs_review',transcript_status='needs_review',final_transcript='',selected_transcript='')
        return
    details=asr.comparison_details(record['transcript_a'],record['transcript_b'])
    record.update(details);status=details['automatic_status']
    if not record.get('human_override'):
        final=record['transcript_a'] if status=='accepted' else ''
        record.update(status=status,transcript_status=status,final_transcript=final,selected_transcript=final)


def reset_audio_results(record):
    record.update(transcript_a='',transcript_b='',final_transcript='',selected_transcript='',
                  confidence_a=None,confidence_b=None,asr_a_done=False,asr_b_done=False,
                  cer=None,wer=None,word_difference=None,snr_db=None,clean_noisy='unavailable',
                  human_override=False,status='needs_review',transcript_status='needs_review')
    record.pop('asr_errors',None)
    for key in asr.COMPARISON_FIELDS:record.pop(key,None)
    for key in ('energy_snr_db','wada_snr_db','snr_analysis','snr_audio_signature','snr_decision_policy','snr_threshold_justification','snr_note','snr_override_by','snr_override_reason'):
        record.pop(key,None)
    record.update(snr_review_required=True,snr_review_status='Recalculate after audio edit')
    record['audio_signature']=audio_signature(record)
    if record['label'] not in VOICE:record.update(status='not_applicable',transcript_status='not_applicable')


def audit(log,action,who,before,after,reason=''):
    if not who.strip():raise ValueError('Enter the reviewer name first.')
    entry=dict(time_utc=stamp(),reviewer=who.strip(),action=action,reason=reason,
               before=copy.deepcopy(before),after=copy.deepcopy(after))
    log.append(entry)
    if isinstance(after,dict):
        after.update(updated_by=who.strip(),updated_at=entry['time_utc'])
        after.setdefault('human_changes',[]).append({k:v for k,v in entry.items() if k not in ('before','after')})


def validate_range(start,end,duration):
    start,end=float(start),float(end)
    if not math.isfinite(start+end) or not 0<=start<end<=duration:raise ValueError('Choose source times within the recording.')
    a,b=round(start*RATE),round(end*RATE)
    if not 5*RATE<=b-a<=15*RATE:raise ValueError('Final clips must be 5–15 seconds long.')
    return a/RATE,b/RATE


def edit(record,source,duration,start,end,label,quality,final,status,who,log,reason):
    if not who.strip():raise ValueError('Enter the reviewer name first.')
    if label not in ('speech','mixed','music','silence'):raise ValueError('Invalid content label.')
    if quality not in ('clean','noisy','unavailable'):raise ValueError('Invalid quality label.')
    if status not in ('accepted','needs_review','difficult','not_applicable'):raise ValueError('Invalid transcript status.')
    before=copy.deepcopy(record);draft=copy.deepcopy(record)
    changed_time=abs(start-record['start'])>1/RATE or abs(end-record['end'])>1/RATE
    if changed_time:
        a,b=validate_range(start,end,duration)
        draft.update(start=a,end=b,duration=b-a,source_mode='continuous_manual',
                     source_spans=[dict(start=a,end=b,clip_start=0.,clip_end=b-a)],
                     audio_pieces=[dict(path=str(source),start=a,end=b)])
        reset_audio_results(draft)
        # Previous transcripts/quality refer to different samples and cannot be
        # reapplied by the editor's stale fields during the boundary change.
        final='';status='needs_review';quality='unavailable'
    if quality!=draft.get('clean_noisy') and not reason.strip():raise ValueError('Enter a reason for a manual quality change.')
    draft.update(label=label,clean_noisy=quality,final_transcript=final.strip(),selected_transcript=final.strip())
    if not changed_time and quality in ('clean','noisy') and reason.strip():
        draft.update(snr_review_required=False,snr_review_status='Human reviewed',snr_override_by=who,snr_override_reason=reason)
        if (draft.get('snr_analysis') or {}).get('analysis_version')==2:
            analysis=copy.deepcopy(draft['snr_analysis']);snr.set_quality(analysis,quality,who,reason)
            draft.update(snr.corpus_fields(analysis))
    if label not in VOICE:status='not_applicable'
    elif status=='accepted' and not asr.normalize_arabic(final):raise ValueError('Accepted speech needs a non-empty final transcript after normalization.')
    draft.update(status=status,transcript_status=status,human_override=True)
    draft['audio_signature']=audio_signature(draft)
    record.clear();record.update(draft)
    audit(log,'edit_final_segment',who,before,record,reason)


def new_clip(source,start,end,label,duration):
    a,b=validate_range(start,end,duration)
    record=asr._make_segment(dict(id='manual-'+uuid.uuid4().hex[:10],start=a,end=b,label=label),1,duration)
    record.update(duration=b-a,audio_pieces=[dict(path=str(source),start=a,end=b)],
                  source_spans=[dict(start=a,end=b,clip_start=0.,clip_end=b-a)],source_mode='continuous_manual',
                  source_origin=str(source),original_approval='manual',asr_a_done=False,asr_b_done=False)
    reset_audio_results(record);return record


def slice_record(record,first,last):
    """Clip-local slicing retains joins, exact PCM sample ranges and padding."""
    result=copy.deepcopy(record);pieces=[];offset=0.
    for p in record['audio_pieces']:
        length=p['end']-p['start'];a,b=max(first,offset),min(last,offset+length)
        if b>a:
            q=dict(p,start=p['start']+a-offset,end=p['start']+b-offset)
            q['synthetic_start']=max(0.,min(b-offset,float(p.get('synthetic_start',0)))-(a-offset))
            q['synthetic_end']=max(0.,(b-offset)-max(a-offset,length-float(p.get('synthetic_end',0))))
            pieces.append(q)
        offset+=length
    spans=[]
    for p in record['source_spans']:
        a,b=max(first,p['clip_start']),min(last,p['clip_end'])
        if b>a:spans.append(dict(start=p['start']+a-p['clip_start'],end=p['start']+b-p['clip_start'],clip_start=a-first,clip_end=b-first))
    if not spans:raise ValueError('The selected interval has no mapped source audio.')
    result.update(segment_id='split-'+uuid.uuid4().hex[:10],audio_pieces=pieces,source_spans=spans,
                  start=spans[0]['start'],end=spans[-1]['end'],duration=last-first)
    reset_audio_results(result);return result


def split(record,position):
    cut=round(float(position)*RATE)/RATE;duration=record['duration']
    if not 5<=cut<=duration-5:raise ValueError('Both resulting clips must be at least 5 seconds long.')
    return [slice_record(record,0,cut),slice_record(record,cut,duration)]


def overlaps(records):
    spans=sorted((s['start'],s['end'],r['segment_id']) for r in records for s in r['source_spans'])
    active=[];result=[]
    for a,b,ident in spans:
        active=[s for s in active if s[1]>a+1/RATE]
        for left,right,other in active:
            if min(right,b)-a>1/RATE:result.append((other,ident))
        active.append((a,b,ident))
    return result


def merge(records):
    records=sorted(records,key=lambda r:r['start'])
    duration=sum(r['duration'] for r in records)
    if len(records)<2 or not 5<=duration<=15:raise ValueError('Select at least two clips with a combined length of 5–15 seconds.')
    if overlaps(records):raise ValueError('Selected clips overlap in the original source. Correct boundaries before merging.')
    result=copy.deepcopy(records[0]);spans=[];pieces=[];offset=0.
    for r in records:
        pieces.extend(copy.deepcopy(r['audio_pieces']))
        spans.extend(dict(p,clip_start=p['clip_start']+offset,clip_end=p['clip_end']+offset) for p in r['source_spans'])
        offset+=r['duration']
    result.update(segment_id='merged-'+uuid.uuid4().hex[:10],audio_pieces=pieces,source_spans=spans,
                  start=spans[0]['start'],end=spans[-1]['end'],duration=duration,
                  label='mixed' if any(r['label']=='mixed' for r in records) else records[0]['label'])
    reset_audio_results(result);return result


def recalculate_quality(records,threshold,policy='conservative',justification=''):
    result=copy.deepcopy(records)
    for r in result:
        require_continuous_pdf_audio(r)
        audio=read_audio(r,for_snr=True);duration=len(audio)/RATE
        previous=r.get('snr_analysis') or {}
        pauses=previous.get('pause_intervals') if r.get('snr_audio_signature')==r['audio_signature'] else None
        if pauses is None:pauses=[[0.,.1],[duration-.1,duration]]
        analysis=snr.analyze_samples(audio,pause_intervals=pauses,content_label=r['label'],threshold=threshold,policy=policy,justification=justification)
        analysis['pause_intervals']=copy.deepcopy(pauses)
        snr.preserve_override(analysis,previous)
        r.update(snr.corpus_fields(analysis),snr_audio_signature=r['audio_signature'])
    return result


def require_continuous_pdf_audio(record):
    """Later structural edits cannot silently reintroduce artificial/joined audio."""
    source=record.get('source_origin')
    pieces=record.get('audio_pieces') or []
    if not source or not pieces:raise ValueError('PDF export needs original-source provenance; reload the reviewed Step 4 clips.')
    source=Path(source).resolve()
    cursor=round(float(pieces[0]['start'])*RATE)
    for piece in pieces:
        a,b=round(float(piece['start'])*RATE),round(float(piece['end'])*RATE)
        if Path(piece['path']).resolve()!=source or a!=cursor or b<=a or piece.get('synthetic_start',0) or piece.get('synthetic_end',0):
            raise ValueError('PDF export cannot contain non-contiguous joins or synthetic silence. Select a continuous original interval and review its pauses.')
        cursor=b


def export_corpus(records,manifest,audit_log,destination,allow_partial=False):
    if not records:raise ValueError('No final review clips are loaded.')
    records=copy.deepcopy(records)
    for record in records:compare(record)
    if overlaps(records):raise ValueError('Final source spans overlap; correct them before corpus export.')
    pending=[]
    for r in records:
        if audio_signature(r)!=r['audio_signature']:raise ValueError('Source audio files changed; reload/review the clips.')
        if r['label'] in VOICE and (not r.get('asr_a_done') or not r.get('asr_b_done') or r['status']=='needs_review'):
            pending.append(r['segment_id'])
        if r['label'] in VOICE and r['status']=='accepted' and not asr.normalize_arabic(r['final_transcript']):
            raise ValueError('An accepted speech clip has no final transcript.')
        if r['clean_noisy'] not in ('clean','noisy'):pending.append(r['segment_id'])
        if r.get('snr_db') is None or (r.get('snr_analysis') or {}).get('analysis_version')!=2 or r.get('snr_audio_signature')!=r['audio_signature'] or r.get('snr_review_required',True) or not r.get('snr_threshold_justification','').strip():
            pending.append(r['segment_id'])
    source_pending=bool(manifest.get('pending_source_candidates') or manifest.get('unresolved_source_spans'))
    if source_pending and not allow_partial:
        raise ValueError('Some source candidates or unresolved spans are still pending. Finish source review or explicitly enable partial export.')
    if pending and not allow_partial:raise ValueError(f'{len(set(pending))} clips need ASR/quality review, current two-method SNR or a threshold justification. Complete them or explicitly enable partial export.')
    for record in records:require_continuous_pdf_audio(record)
    out=Path(destination)/('corpus_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
    exported=[]
    for i,record in enumerate(sorted(records,key=lambda r:r['start']),1):
        r=copy.deepcopy(record);audio=read_audio(r)
        if not 5*RATE<=len(audio)<=15*RATE:raise ValueError(f"Invalid clip length: {r['segment_id']}")
        name=f'segment_{i:05d}'
        sf.write(out/(name+'.wav'),audio,RATE,subtype='PCM_16')
        (out/(name+'.txt')).write_text(r.get('final_transcript',''),encoding='utf-8')
        r.update(file=name+'.wav',transcript_file=name+'.txt',sample_count=len(audio),sample_rate_hz=RATE,
                 sha256=hashlib.sha256((out/(name+'.wav')).read_bytes()).hexdigest())
        exported.append(r)
    metadata=dict(manifest,exported_utc=stamp(),partial=bool(pending) or source_pending,pending_segment_ids=sorted(set(pending)),
                  asr_comparison_policy=asr.comparison_policy(),
                  snr_methods=list(snr.METHODS),second_snr_method='wada_snr',
                  snr_analysis_complete=all((r.get('snr_analysis') or {}).get('analysis_version')==2 and r.get('snr_audio_signature')==r['audio_signature'] for r in records))
    if manifest.get('source_review') is not None:
        (out/'source_review.json').write_text(json.dumps(manifest['source_review'],ensure_ascii=False,indent=2),encoding='utf-8')
        metadata['source_review_file']='source_review.json'
    asr.export_csv(out/'corpus.csv',metadata,exported,audit_log)
    (out/'corpus.json').write_text(json.dumps(dict(metadata=metadata,segments=exported,audit_log=audit_log),ensure_ascii=False,indent=2),encoding='utf-8')
    import html
    esc=lambda value:html.escape(str(value),quote=True)
    counts={status:sum(r['status']==status for r in exported) for status in ('accepted','needs_review','difficult','not_applicable')}
    rows=''.join(f"<tr><td>{esc(r['segment_id'])}</td><td>{r['start']:.4f}</td><td>{r['end']:.4f}</td><td>{r['duration']:.4f}</td><td>{esc(r['label'])}</td><td>{esc(r.get('energy_snr_db'))}</td><td>{esc(r.get('wada_snr_db'))}</td><td>{esc(r.get('snr_db'))}</td><td>{esc(r['clean_noisy'])}</td><td>{esc(r['status'])}</td><td dir='auto'>{esc(r.get('final_transcript',''))}</td><td><audio controls preload='none' src='{esc(r['file'])}'></audio></td></tr>" for r in exported)
    report=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Speech corpus export</title>
<style>body{{font:15px/1.6 'Segoe UI',Arial;background:#f0f4f5;color:#173b40;margin:0}}header{{background:#173b40;color:white;padding:24px}}main{{padding:20px}}section{{background:white;border:1px solid #d7e3e6;border-radius:8px;padding:18px;margin-bottom:16px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #d7e3e6;text-align:left;padding:8px}}th{{background:#eaf2f2}}.scroll{{overflow:auto}}audio{{width:210px}}</style>
<header><h1>Speech corpus export</h1><p>Clips: {len(exported)} · Partial review: {metadata['partial']}</p></header><main><section><h2>Review summary</h2><p>{esc(counts)}</p><p>Both Arabic transcripts and agreement scores are in corpus.csv and corpus.json. Final text is paired with each WAV in a UTF-8 TXT file. Full audit data are saved alongside the corpus.</p><p>SNR: pause-energy and WADA-SNR estimates. Decision rule, threshold justification, range limits and review status are recorded per clip in CSV/JSON; incomplete or unreviewed SNR makes an export partial. Source candidates pending: {manifest.get('pending_source_candidates',0)}. Unresolved source spans: {manifest.get('unresolved_source_spans',0)}.</p></section><section class="scroll"><table><tr><th>ID</th><th>Source start</th><th>Source end</th><th>Clip length</th><th>Label</th><th>Energy SNR</th><th>WADA-SNR</th><th>Decision SNR</th><th>Quality</th><th>Status</th><th>Final text</th><th>Listen</th></tr>{rows}</table></section></main></html>"""
    (out/'corpus_report.html').write_text(report,encoding='utf-8')
    return dict(output_dir=str(out),clips=len(exported),partial=bool(pending) or source_pending)

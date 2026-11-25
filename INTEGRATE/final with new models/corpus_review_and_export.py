"""Build review records from source intervals, manage human transcription decisions and export the corpus.

Function guide
--------------
stamp, digest, source_signature, audio_signature
    Identify a source or audio revision and timestamp changes.
from_processing, is_preview, can_transcribe
    Convert the current segmentation and quality records into the ASR queue.
preview_audio, read_audio, transcribe_record
    Read the selected audio and transcribe eligible spans, including longer unresolved
    ones.
compare, reset_audio_results, audit
    Apply transcript-agreement rules, invalidate stale measurements and record edits.
validate_range, edit, new_clip, slice_record, split, overlaps, merge
    Validate and edit clip boundaries while preserving the source timeline.
recalculate_quality, require_continuous_pdf_audio
    Recompute quality from the clip and verify the audio mapping used for export.
process_summary, append_summary_workbook
    Compute per-video totals and append a named row to the summary workbook.
review_rows, review_html, write_review_table, export_review_csv, export_corpus
    Build review rows and write tables, paired audio/text files and corpus metadata.
"""
from output_folders import new_run_folder
import copy
import csv
import os
import tempfile
import html
import hashlib
import json
import math
from pathlib import Path
import uuid
from datetime import datetime, timezone
import numpy as np
import soundfile as sf
import transcript_agreement_and_io as asr
import segment_review as review
import clip_quality_assessment as snr

RATE=16000
VOICE=('speech','mixed')


def stamp():return datetime.now(timezone.utc).isoformat(timespec='seconds')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def source_signature(processing, snr_records=None):
    if not processing.audio or not processing.session:return None
    return digest({'audio':str(processing.audio),'size':processing.audio.stat().st_size,
                   'mtime':processing.audio.stat().st_mtime_ns,'step4':processing.session,
                   'snr':processing.snr_results if snr_records is None else snr_records})


def from_processing(processing):
    """Use the same source intervals and IDs as the SNR worker."""
    if not processing.audio or not processing.session:
        raise ValueError('Generate or open Part 4 segments first.')
    if any(item['status']=='approved' for item in processing.session['segments']):
        processing.require_approved()
    inputs=snr.session_records(processing.session,processing.audio,for_transcription=True)
    items={s['id']:s for s in processing.session['segments']+processing.session.get('unresolved',[])}
    qualities={s['id']:s for s in processing.snr_results} if review.same_snr_source(processing.snr_snapshot,processing.session) else {}
    changes_by_id={}
    for event in processing.session.get('changes',[]):
        changes_by_id.setdefault(event.get('item_id'),[]).append(event)
    source_path=str(Path(processing.audio).resolve())
    records=[]
    for row in inputs:
        a,b = row['source_start'],row['source_end']
        ident,label = row['id'],row['content_label']
        unresolved = row['segmentation_status']=='unresolved'
        pending = row['segmentation_status'] == 'pending'
        parent = row.get('parent_span_id',ident)
        item = items[parent]
        spans=[dict(start=a,end=b,clip_start=0.,clip_end=b-a)]
        quality=qualities.get(ident,{})
        raw=dict(id=ident,label=label,source_spans=spans,
                 snr_db=quality.get('snr_db'),quality=quality.get('quality','unavailable'))
        record=asr._make_segment(raw,len(records)+1,processing.duration,unresolved=unresolved)
        pad_a=pad_b=0.
        record.update(start=spans[0]['start'],end=spans[-1]['end'],duration=b-a,source_spans=copy.deepcopy(spans),
                      audio_pieces=[dict(path=source_path,start=a,end=b,synthetic_start=pad_a,synthetic_end=pad_b)],
                      asr_a_done=False,asr_b_done=False,source_mode='unresolved_preview' if unresolved else ('candidate_preview' if pending else 'continuous_source'),
                      segmentation_status=row['segmentation_status'],source_metadata=copy.deepcopy(item),quality_history=copy.deepcopy(quality.get('quality_changes',[])),
                      human_changes=copy.deepcopy(changes_by_id.get(parent,[])+quality.get('quality_changes',[])),
                      source_origin=str(processing.audio),original_approval=row['segmentation_status'] if unresolved or pending else 'approved')
        record.update(snr.corpus_fields(quality))
        record['audio_signature']=audio_signature(record)
        record['snr_audio_signature']=record['audio_signature'] if quality.get('analysis_version')==2 else None
        if not can_transcribe(record):record.update(status='not_applicable',transcript_status='not_applicable')
        records.append(record)
    return records


def is_preview(record):
    return record.get('source_mode') in ('unresolved_preview', 'candidate_preview')


def can_transcribe(record):
    return record['label'] in VOICE or (is_preview(record) and record['label'] == 'unclassified')


def preview_audio(record):
    """Read bounded, contiguous windows. Prefer a quiet 40 ms frame near each join."""
    piece, = record['audio_pieces']
    first, last = round(piece['start'] * RATE), round(piece['end'] * RATE)
    with sf.SoundFile(piece['path']) as wav:
        if (wav.samplerate, wav.channels, wav.subtype) != (RATE, 1, 'PCM_16') or not 0 <= first < last <= len(wav):
            raise ValueError('Unresolved preview requires a valid interval in the original mono PCM16 16 kHz WAV.')
        while first < last:
            wav.seek(first)
            size = min(last-first, 15 * RATE)
            audio = wav.read(size, dtype='float32')
            if len(audio) != size:
                raise ValueError('Could not read the complete unresolved interval.')
            if first + size < last:
                # Leave at least five seconds for the final decode. Fixed joins
                # here are inference-only and never become corpus cut points.
                high = min(size, last-first-5*RATE)
                low = high - 2*RATE
                frame = RATE // 25
                energy = np.square(audio[low:high]).reshape(-1, frame).mean(axis=1)
                audio = audio[:low + int(energy.argmin()) * frame + frame // 2]
            first += len(audio)
            if len(audio) < 5*RATE:
                audio = np.pad(audio, (0, 5*RATE-len(audio)))
            yield audio


def transcribe_record(record, key, model, stop_event):
    windows = preview_audio(record) if is_preview(record) else (read_audio(record),)
    parts = []
    scores = []
    for index, audio in enumerate(windows):
        if index and stop_event.is_set():
            raise InterruptedError('Stopped during unresolved span.')
        text, score = (asr.transcribe_qwen(model, audio) if key == 'a'
                       else asr.transcribe_cohere(model, audio, RATE))
        if text.strip():
            parts.append(text.strip())
        if score is not None:
            scores.append((score, len(audio)))
    score = sum(v*n for v,n in scores) / sum(n for _,n in scores) if scores else None
    return ' '.join(parts), score


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
    record.update(cer_a_reference='normalized_transcript_b',cer_b_reference='normalized_transcript_a',review_word_edit_limit=None)
    if not can_transcribe(record):
        record.update(status='not_applicable',transcript_status='not_applicable',
                      automatic_status='not_applicable',agreement_reason='Music/silence is not sent to ASR.',
                      cer=None,wer=None,word_difference=None,word_substitutions=None,
                      word_insertions=None,word_deletions=None,cer_a_vs_b=None,cer_b_vs_a=None,
                      automatic_choice=None,automatic_transcript='',
                      automatic_choice_reason='Music/silence is not sent to ASR.');return
    if not (record.get('asr_a_done') and record.get('asr_b_done')):
        record.update(cer=None,wer=None,word_difference=None,word_substitutions=None,
                      word_insertions=None,word_deletions=None,automatic_status='needs_review',
                      agreement_reason='Waiting for both ASR attempts.',
                      cer_a_vs_b=None,cer_b_vs_a=None,automatic_choice=None,automatic_transcript='',
                      automatic_choice_reason='Waiting for both ASR attempts; no automatic choice yet.')
        if not record.get('human_override'):
            record.update(status='needs_review',transcript_status='needs_review',final_transcript='',selected_transcript='')
        return
    duration = record['end'] - record['start'] if is_preview(record) else None
    details=asr.comparison_details(record['transcript_a'],record['transcript_b'],unresolved_duration=duration)
    record.update(details);status=details['automatic_status']
    if not record.get('human_override'):
        final=details['automatic_transcript']
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


SUMMARY_COLUMNS = (
    'Video', 'Duration', 'Speech', 'Music', 'Mixed', 'Segments',
    'Mean SNR', 'Accepted', 'Chosen by hand', 'Difficult',
)


def process_summary(duration, regions, records, audit_log):
    """Summarize source regions and current ASR outcomes."""
    duration_by_label = {label: 0.0 for label in ('speech', 'music', 'mixed')}
    for region in regions:
        label = region.get('label')
        if label in duration_by_label:
            duration_by_label[label] += float(region['end']) - float(region['start'])
    eligible = [r for r in records if can_transcribe(r)]
    manual_ids = set()
    for event in audit_log:
        before, after = event.get('before'), event.get('after')
        if not isinstance(after, dict):
            continue
        if event.get('action') == 'transcript_review' or (
            event.get('action') == 'edit_final_segment' and isinstance(before, dict)
            and any(before.get(k) != after.get(k) for k in ('final_transcript', 'status'))
        ):
            manual_ids.add(after.get('segment_id'))
    counts = {'accepted': 0, 'chosen': 0, 'difficult': 0}
    for record in eligible:
        if record.get('status') == 'difficult':
            counts['difficult'] += 1
        elif record.get('status') == 'accepted':
            if record['segment_id'] in manual_ids or record.get('automatic_status') != 'accepted':
                counts['chosen'] += 1
            else:
                counts['accepted'] += 1
    snr_values = [float(r['snr_db']) for r in records
                  if r.get('snr_db') is not None and math.isfinite(float(r['snr_db']))]
    total = len(eligible)
    return {
        'duration': float(duration), 'region_seconds': duration_by_label,
        'segments': len(records), 'unresolved': sum(r.get('source_mode') == 'unresolved_preview' for r in records),
        'mean_snr': sum(snr_values) / len(snr_values) if snr_values else None,
        'snr_count': len(snr_values), 'asr_total': total, 'counts': counts,
        'pending': total - sum(counts.values()),
    }


def append_summary_workbook(path, video_name, summary):
    """Append a numeric summary row, preserving the existing workbook."""
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    path = Path(path)
    if path.exists():
        workbook = load_workbook(path)
        if 'Process summary' not in workbook:
            raise ValueError('Choose a workbook previously created by Save summary, or a new filename.')
        sheet = workbook['Process summary']
        if tuple(sheet.cell(1, c).value for c in range(1, 11)) != SUMMARY_COLUMNS:
            raise ValueError('The summary workbook columns do not match the expected format.')
    else:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = 'Process summary'
        sheet.append(SUMMARY_COLUMNS)
        sheet.freeze_panes = 'B2'
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='173B40')
            cell.font = Font(bold=True, color='FFFFFF')
            cell.alignment = Alignment(wrap_text=True, vertical='center')
        for column in 'ABCDEFGHIJ':
            sheet.column_dimensions[column].width = 18
        sheet.column_dimensions['A'].width = 28
    n = summary['asr_total']
    regions = summary['region_seconds']
    counts = summary['counts']
    sheet.append([video_name, summary['duration']/60, regions['speech']/60,
                  regions['music']/60, regions['mixed']/60, summary['segments'],
                  summary['mean_snr'],
                  counts['accepted']/n if n else None,
                  counts['chosen']/n if n else None,
                  counts['difficult']/n if n else None])
    row = sheet.max_row
    sheet.cell(row, 1).data_type = 's'
    for col in range(2, 6):
        sheet.cell(row, col).number_format = '0.00 "min"'
    sheet.cell(row, 7).number_format = '0.00 "dB"'
    for col in range(8, 11):
        sheet.cell(row, col).number_format = '0.0%'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.xlsx', delete=False) as file:
            temporary = Path(file.name)
        workbook.save(temporary)
        os.replace(temporary, path)
    finally:
        workbook.close()
        if temporary:
            temporary.unlink(missing_ok=True)
    return path


REVIEW_COLUMNS = (
    'start', 'end', 'label', 'snr', 'quality', 'model_A_transcript',
    'model_B_transcript', 'final_transcript', 'status', 'review_status',
    'automatic_status', 'segment_id', 'audio_file', 'changed_by',
    'changed_at', 'change_log', 'segmentation_status',
)


def review_rows(records, audit_log):
    """Project records into the spreadsheet schema, preserving the caller's row order."""
    def segment_ids(snapshot):
        if isinstance(snapshot, dict):
            if snapshot.get('segment_id'):
                yield snapshot['segment_id']
            else:
                for value in snapshot.values():
                    yield from segment_ids(value)
        elif isinstance(snapshot, list):
            for value in snapshot:
                yield from segment_ids(value)

    history = {}
    for event in audit_log:
        for ident in set(segment_ids([event.get('before'), event.get('after')])):
            history.setdefault(ident, []).append(event)
    for record in records:
        ident = record['segment_id']
        if record.get('segmentation_status')=='unresolved' or record.get('source_mode')=='unresolved_preview':
            detailed = history.get(ident, [])
            # Final review stores a compact copy of each detailed audit event on the row.
            summaries = [{k:v for k,v in e.items() if k not in ('before','after')} for e in detailed]
            changes = []
            for event in [*record.get('human_changes', []), *record.get('quality_history', []), *detailed]:
                if event not in summaries and event not in changes:
                    changes.append(event)
            changes.sort(key=lambda e:e.get('time_utc','').split('+')[0])
        else:
            changes = history.get(ident, record.get('human_changes', []))
        latest = changes[-1] if changes else {}
        status = record.get('status', 'needs_review')
        raw_a = record.get('transcript_a') or ''
        raw_b = record.get('transcript_b') or ''
        norm_a = record.get('normalized_transcript_a')
        norm_b = record.get('normalized_transcript_b')
        if norm_a is None or norm_b is None:
            norm_a, norm_b, _ = asr.comparison_copies(raw_a, raw_b)
        final = record.get('final_transcript') or ''
        normalized_final = asr.normalize_arabic(final)
        if final and final == raw_b:
            normalized_final = norm_b
        elif final and final == raw_a:
            normalized_final = norm_a
        yield {
            'start': record['start'], 'end': record['end'],
            'label': record.get('label', ''), 'snr': record.get('snr_db'),
            'quality': record.get('clean_noisy', ''),
            'model_A_transcript': norm_a,
            'model_B_transcript': norm_b,
            'final_transcript': normalized_final,
            'status': status,
            'review_status': status if status in ('accepted', 'difficult', 'not_applicable') else 'pending',
            'automatic_status': record.get('automatic_status', ''),
            'segment_id': ident,
            'audio_file': record.get('file') or record.get('source_origin', ''),
            'changed_by': latest.get('reviewer') or record.get('updated_by', ''),
            'changed_at': latest.get('time_utc') or record.get('updated_at', ''),
            'change_log': json.dumps(changes, ensure_ascii=False),
            'segmentation_status': record.get('segmentation_status') or ('unresolved' if record.get('source_mode')=='unresolved_preview' else 'pending' if record.get('source_mode')=='candidate_preview' else 'resolved'),
        }


def review_html(rows, partial=None, playable=False):
    def cell(key, value):
        text = html.escape(str(value)) if value is not None else ''
        if key == 'audio_file' and playable:
            return f'<td>{text}<br><audio controls preload="metadata" src="{text}"></audio></td>'
        if key == 'change_log':
            details = text if value != '[]' else 'No changes recorded.'
            return f'<td><details><summary>View changes</summary>{details}</details></td>'
        return f'<td dir="auto">{text}</td>'

    headers = ''.join(f'<th scope="col">{key}</th>' for key in REVIEW_COLUMNS)
    body = ''.join('<tr>' + ''.join(cell(k, row[k]) for k in REVIEW_COLUMNS) + '</tr>' for row in rows)
    title = 'Speech corpus review' if partial is None else 'Speech corpus export'
    subtitle = ' · Partial export' if partial else ''
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{box-sizing:border-box;margin:0;padding:clamp(8px,2vw,20px);height:100vh;height:100dvh;display:flex;flex-direction:column;font:15px/1.5 'Segoe UI',sans-serif;color:#173b40}}
header{{flex:none}}h1{{margin:0}}p{{margin:4px 0 12px}}
section{{min-height:0;overflow-x:scroll;overflow-y:auto}}
section:focus-visible{{outline:2px solid #157a6e;outline-offset:2px}}
section::-webkit-scrollbar{{width:14px;height:14px}}
section::-webkit-scrollbar-track{{background:#eaf2f2}}
section::-webkit-scrollbar-thumb{{background:#527579;border:2px solid #eaf2f2;border-radius:7px}}
table{{border-collapse:separate;border-spacing:0}}th,td{{padding:8px;border-right:1px solid #d7e3e6;border-bottom:1px solid #d7e3e6;vertical-align:top}}
th{{position:sticky;top:0;background:#eaf2f2}}
td:nth-child(n+6):nth-child(-n+8){{min-width:22em}}details{{min-width:10em;max-width:30em;overflow-wrap:anywhere}}audio{{width:200px}}
@media print{{body{{display:block;height:auto;padding:0}}section{{overflow:visible}}th{{position:static}}}}
</style><header><h1 id="report-title">{title}</h1><p>{len(rows)} clips{subtitle}</p></header>
<section aria-labelledby="report-title" tabindex="0"><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></section></html>"""


def write_review_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as f:
        temporary = Path(f.name)
    try:
        suffix = path.suffix.lower()
        if suffix == '.csv':
            with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        elif suffix == '.html':
            temporary.write_text(review_html(rows), encoding='utf-8')
        elif suffix == '.json':
            temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        elif suffix == '.xlsx':
            from openpyxl import Workbook
            from openpyxl.utils import get_column_letter
            book = Workbook()
            try:
                sheet = book.active
                sheet.title = 'Review'
                sheet.append(REVIEW_COLUMNS)
                for index, row in enumerate(rows, start=2):
                    for col, key in enumerate(REVIEW_COLUMNS, start=1):
                        value = row[key]
                        if isinstance(value, str) and len(value) > 32767:
                            raise ValueError('A review cell exceeds the Excel limit. Save as JSON or CSV to keep the full text.')
                        cell = sheet.cell(index, col, value)
                        if isinstance(value, str):
                            cell.data_type = 's'
                sheet.freeze_panes = 'A2'
                sheet.auto_filter.ref = sheet.dimensions
                for i, key in enumerate(REVIEW_COLUMNS, 1):
                    sheet.column_dimensions[get_column_letter(i)].width = 45 if 'transcript' in key or key in ('audio_file','change_log') else max(12,len(key)+2)
                book.save(temporary)
            finally:
                book.close()
        else:
            raise ValueError('Choose CSV, Excel (.xlsx), HTML or JSON.')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def timeline_records(records, session=None, source='', qualities=()):
    """Include unresolved source time even when a span was skipped for transcription."""
    combined = list(records)
    if session and session.get('unresolved'):
        covered = []
        previews = {}
        for r in records:
            if r.get('source_mode') == 'unresolved_preview' or r.get('segmentation_status') == 'unresolved':
                parent = (r.get('source_metadata') or {}).get('id') or r['segment_id'].split('@')[0]
                previews.setdefault(parent, []).append((r['start'],r['end']))
            else:
                covered.extend((s['start'],s['end']) for s in r.get('source_spans',[]))
        for clip in session.get('segments',[]):
            if clip.get('status')=='approved':
                covered.extend(review.item_source_ranges(clip))
        quality_by_id = {q['id']:q for q in qualities}
        changes_by_parent = {}
        for event in session.get('changes',[]):
            ids = {event.get('item_id')}
            after = event.get('after')
            if isinstance(after,dict):
                ids.update(s['id'] for s in after.get('preserved_leftovers',[]))
            for parent_id in ids:
                changes_by_parent.setdefault(parent_id,[]).append(event)
        for item,a,b in review.downstream_unresolved(session,covered):
            parent = item['id']
            cursor = a
            for left,right in sorted(previews.get(parent,[])) + [(b,b)]:
                end = min(left,b)
                if cursor < end:
                    ident = parent if (cursor,end)==review.unresolved_bounds(item) else f'{parent}@{round(cursor*RATE)}-{round(end*RATE)}'
                    quality = quality_by_id.get(ident,{})
                    combined.append(dict(segment_id=ident,start=cursor,end=end,label=item.get('label'),
                        segmentation_status='unresolved',status='not_transcribed',source_origin=str(source),
                        snr_db=quality.get('snr_db'),clean_noisy=quality.get('quality','')))
                cursor = max(cursor,right)
                if cursor>=b:
                    break
        preferences = {}
        for event in session.get('asr_preference_history',[]):
            preferences.setdefault(event['item_id'],[]).append(event)
        for index,record in enumerate(combined):
            if record.get('segmentation_status')=='unresolved' or record.get('source_mode')=='unresolved_preview':
                ident = record['segment_id']
                parent = (record.get('source_metadata') or {}).get('id') or ident.split('@')[0]
                combined[index] = dict(record, human_changes=[*changes_by_parent.get(parent,[]),
                    *record.get('human_changes',[]), *(e for e in preferences.get(parent,[])
                    if e['start']<record['end'] and e['end']>record['start'])], quality_history=[
                    *record.get('quality_history',[]), *quality_by_id.get(ident,{}).get('quality_changes',[])])
    return sorted(combined,key=lambda r:(r['start'],r['end'],r['segment_id']))


def report_rows(records, audit_log, session=None, source='', qualities=()):
    ordered = timeline_records(records,session,source,qualities)
    return list(review_rows(ordered,audit_log))


def export_review_csv(path, records, audit_log, session=None, source='', qualities=()):
    return write_review_table(path,report_rows(records,audit_log,session,source,qualities))


def export_corpus(records,manifest,audit_log,destination,allow_partial=False):
    previews = [r for r in records if is_preview(r)]
    has_unresolved = bool(previews)
    records = [r for r in records if not is_preview(r)]
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
    source_review = manifest.get('source_review') or {}
    timeline = timeline_records([*records,*previews],source_review.get('step4'),
                                manifest.get('audio',''),source_review.get('snr') or [])
    def status(r):
        return r.get('segmentation_status') or ('unresolved' if r.get('source_mode')=='unresolved_preview'
                                                 else 'pending' if is_preview(r) else 'resolved')
    if not any(status(r) in ('unresolved','resolved') for r in timeline):
        raise ValueError('No resolved clips or unresolved spans to export.')
    source_pending=bool(any(status(r)=='unresolved' for r in timeline) or has_unresolved or manifest.get('pending_source_candidates') or manifest.get('unresolved_source_spans'))
    if source_pending and not allow_partial:
        raise ValueError('Some source candidates or unresolved spans are still pending. Finish source review or explicitly enable partial export.')
    if pending and not allow_partial:raise ValueError(f'{len(set(pending))} clips need ASR/quality review, current two-method SNR or a threshold justification. Complete them or explicitly enable partial export.')
    for record in records:require_continuous_pdf_audio(record)
    out=new_run_folder(destination)
    exported=[]
    report_records=[]
    for record in timeline:
        r=copy.deepcopy(record)
        state=status(r)
        if state=='pending':
            relative=Path('review_audio') / f'pending_{len(report_records)+1:05d}.wav'
            (out/relative.parent).mkdir(exist_ok=True)
            review.extract_wav(r['source_origin'],out/relative,r['start'],r['end'])
            r['file']=relative.as_posix()
        else:
            name=f'segment_{len(exported)+1:05d}' + ('_unresolved' if state=='unresolved' else '')
            wav=out/(name+'.wav')
            if state=='unresolved':
                if r.get('audio_signature') and audio_signature(r)!=r['audio_signature']:
                    raise ValueError('Unresolved source audio changed. Reload the clips before export.')
                review.extract_wav(r['source_origin'],wav,r['start'],r['end'])
                count=round(r['end']*RATE)-round(r['start']*RATE)
            else:
                audio=read_audio(r)
                count=len(audio)
                if not 5*RATE<=count<=15*RATE:raise ValueError(f"Invalid clip length: {r['segment_id']}")
                sf.write(wav,audio,RATE,subtype='PCM_16')
            (out/(name+'.txt')).write_text(r.get('final_transcript') or '',encoding='utf-8')
            r.update(file=wav.name,transcript_file=name+'.txt',sample_count=count,sample_rate_hz=RATE,
                     sha256=review.file_sha256(wav),segmentation_status=state)
            exported.append(r)
        report_records.append(r)
    metadata=dict(manifest,exported_utc=stamp(),partial=bool(pending) or source_pending,pending_segment_ids=sorted(set(pending)),
                  asr_comparison_policy=asr.comparison_policy(),
                  snr_methods=list(snr.METHODS),second_snr_method='wada_snr',
                  snr_analysis_complete=all((r.get('snr_analysis') or {}).get('analysis_version')==2 and r.get('snr_audio_signature')==r['audio_signature'] for r in exported))
    if manifest.get('source_review') is not None:
        (out/'source_review.json').write_text(json.dumps(manifest['source_review'],ensure_ascii=False,indent=2),encoding='utf-8')
        metadata['source_review_file']='source_review.json'
    rows = list(review_rows(exported, audit_log))
    write_review_table(out / 'corpus.csv', rows)
    report=list(review_rows(report_records,audit_log))
    write_review_table(out / 'review.csv', report)
    (out/'corpus.json').write_text(json.dumps(dict(metadata=metadata, segments=rows), ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'corpus_audit.json').write_text(json.dumps(dict(segments=exported, audit_log=audit_log), ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'corpus_report.html').write_text(review_html(report, partial=metadata['partial'], playable=True), encoding='utf-8')
    return dict(output_dir=str(out),clips=len(exported),partial=bool(pending) or source_pending)

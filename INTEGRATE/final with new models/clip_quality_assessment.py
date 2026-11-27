"""Turn two SNR measurements into a reviewable clean/noisy decision.

analyze_samples measures one clip, calculate works through a session, and reclassify
reuses the scores when the threshold changes. export_report saves the decision rule
and human corrections alongside the measurements.
"""

from output_folders import new_run_folder
import copy
import csv
import html
import hashlib
import json
import math
import wave
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import segment_review as review
from snr_estimators import energy_details,wada_details,METHODS

POLICIES=('conservative','energy','wada')
DISAGREEMENT_DB=6.


# Return the energy estimate together with any notes about the pause frames.
def estimate_energy(samples,rate=16000,pause_intervals=None):
    result=energy_details(samples,rate,pause_intervals)
    return result['value_db'],'; '.join(result['notes']) or 'Energy estimate uses pause-frame background noise and active-minus-noise power.'


# Return the WADA estimate and any caveats from the measurement.
def estimate_wada(samples,rate=16000):
    result=wada_details(samples,rate)
    return result['value_db'],'; '.join(result['notes'])


# Apply the selected threshold; warnings are informational, not approval gates.
def decision(item,threshold=20.,policy='conservative',justification=''):
    threshold=float(threshold)
    if not math.isfinite(threshold) or not -50<=threshold<=100:raise ValueError('Threshold must be finite and between -50 and 100 dB.')
    if policy not in POLICIES:raise ValueError('Unknown SNR decision rule.')
    energy=item.get('energy_snr_db');wada=item.get('wada_snr_db')
    for value in (energy,wada):
        if value is not None and not math.isfinite(float(value)):raise ValueError('Invalid non-finite SNR estimate.')
    energy=float(energy) if energy is not None else None
    wada=float(wada) if wada is not None else None
    item.update(energy_snr_db=energy,wada_snr_db=wada,
        energy_status=(item.get('energy_details') or {}).get('status','legacy'),
        wada_status=(item.get('wada_details') or {}).get('status','not_calculated'),
        wada_bound=(item.get('wada_details') or {}).get('bound'))
    wada_usable=wada is not None and (item.get('wada_details') or {}).get('status')=='ok'
    values=[float(v) for v in (energy,wada if wada_usable else None) if v is not None]
    selected=(min(values) if values else None) if policy=='conservative' else (energy if policy=='energy' else wada if wada_usable else None)
    auto='unavailable' if selected is None else 'clean' if selected>=threshold else 'noisy'
    energy_label='unavailable' if energy is None else 'clean' if energy>=threshold else 'noisy'
    wada_label='unavailable' if not wada_usable else 'clean' if wada>=threshold else 'noisy'
    difference=abs(energy-wada) if energy is not None and wada_usable else None
    e=item.get('energy_details') or {};reasons=[]
    if item.get('analysis_version')!=2:reasons.append('Legacy single-method result: recalculate both estimators.')
    if energy is None or not wada_usable:reasons.append('One or both methods unavailable or range-limited.')
    if energy_label!=wada_label:reasons.append('The estimators do not agree on clean/noisy.')
    if difference is not None and difference>DISAGREEMENT_DB:reasons.append(f'Estimates differ by more than {DISAGREEMENT_DB:g} dB.')
    if e.get('noise_source')=='quiet_frame_fallback':reasons.append('No trusted pause metadata; energy noise floor is a fallback.')
    if (e.get('noise_variation_db') or 0)>6:reasons.append('Nonstationary background noise; inspect the local noise profile.')
    if e.get('noise_power') and e.get('observed_power',0)/e['noise_power']<1.25:reasons.append('Weak active/noise contrast.')
    if item.get('content_label')=='unclassified':
        reasons.append('Content is unclassified; listen to confirm that a speech SNR estimate is meaningful.')
    elif item.get('content_label')!='speech':reasons.append('Speech-based assumptions are uncertain for music/mixed content.')
    if item.get('clipped_fraction',0)>.001:reasons.append('Clipped samples can bias both estimators.')
    manual=bool(item.get('override_by'))
    item.update(threshold_db=threshold,threshold_justification=str(justification).strip(),decision_policy=policy,
        snr_db=selected,energy_quality=energy_label,wada_quality=wada_label,method_difference_db=difference,
        automatic_quality=auto,automatic_review_required=False,quality_warning=bool(reasons),review_reasons=reasons)
    if not manual:item['quality']=auto
    item['review_required']=False
    item['review_status']='Manually corrected' if manual else 'Automatic (see notes)' if reasons else 'Automatic'
    notes=list(reasons)+list(e.get('notes',[]))+list((item.get('wada_details') or {}).get('notes',[]))
    item['note']='; '.join(dict.fromkeys(notes))
    return item


# Measure the same samples both ways before assigning a quality label.
def analyze_samples(samples,rate=16000,pause_intervals=None,*,content_label='speech',threshold=20.,policy='conservative',justification=''):
    energy=energy_details(samples,rate,pause_intervals);wada=wada_details(samples,rate)
    x=np.asarray(samples)
    clipped=float(np.mean(np.abs(x)>=32767/32768)) if x.ndim==1 and len(x) and np.isfinite(x).all() else 0.
    item=dict(analysis_version=2,energy_snr_db=energy['value_db'],wada_snr_db=wada['value_db'],
        energy_details=energy,wada_details=wada,content_label=content_label,clipped_fraction=clipped,
        noise_source=energy['noise_source'],noise_variation_db=energy['noise_variation_db'],override_by='',override_reason='',
        audio_sha256=hashlib.sha256(np.asarray(x,dtype='<f4').tobytes()).hexdigest() if x.ndim==1 and np.isfinite(x).all() else None)
    return decision(item,threshold,policy,justification)


def source_time_key(record):
    """Order resolved clips and unresolved spans on the same source timeline."""
    return float(record['source_start']), float(record['source_end']), record['id']


# Collect eligible clips and uncovered source intervals for the quality queue.
def session_records(data,source,*,include_unresolved=True,for_transcription=False):
    queue=review.approved_playback_queue(data,source);items={x['id']:x for x in data['segments']};out=[]
    for path,a,b,label,ident in queue:
        item=items[ident];source_a,source_b=review.require_pdf_review(data,item)
        duration=b-a;original=item.get('original') or {}
        unchanged=abs(source_a-float(original.get('start',-1)))<1/16000 and abs(source_b-float(original.get('end',-1)))<1/16000
        automated=(item.get('pdf_boundary_review') or {}).get('approval_method')=='auto_process'
        if automated:
            # Do not treat unverified edge context as measured quiet frames.
            left=float(original.get('left_padding',0.)) if unchanged else 0.
            right=float(original.get('right_padding',0.)) if unchanged else 0.
            left,right=min(.2,max(0.,left)),min(.2,max(0.,right))
            pauses=[]
            if left>=.1:pauses.append([0.,left])
            if right>=.1:pauses.append([duration-right,duration])
        else:
            left=float(original.get('left_padding',.1)) if unchanged else .1
            right=float(original.get('right_padding',.1)) if unchanged else .1
            left,right=min(.2,max(.1,left)),min(.2,max(.1,right))
            pauses=[[0.,left],[duration-right,duration]]
        for p in data.get('pause_intervals',[]):
            x,y=max(source_a,float(p['start'])),min(source_b,float(p['end']))
            if y>x:pauses.append([x-source_a,y-source_a])
        stat=Path(path).stat()
        out.append(dict(segmentation_status='resolved',id=ident,file=Path(path).name,path=path,read_start=a,read_end=b,
            audio_file_size=stat.st_size,audio_file_mtime_ns=stat.st_mtime_ns,
            source_start=source_a,source_end=source_b,content_label=label,synthetic_start=0.,synthetic_end=0.,
            pause_intervals=pauses or None, automatic_boundary_approval=automated,
            pause_basis=('Part 4 detector intervals (automatic approval; no listening confirmation)' if pauses else 'Quiet-frame fallback')
                        if automated else 'Part 4 detector intervals plus named reviewer boundary confirmation'))
    previews = []
    if data.get('analyze_pending_candidates'):
        for item in data['segments']:
            if item['status'] == 'pending':
                a, b = review.pdf_audio_mapping(data, item)
                previews.append((item, a, b, item['id'], 'pending'))
    if include_unresolved:
        covered = [(r['source_start'], r['source_end']) for r in out]
        covered.extend((a, b) for _, a, b, _, _ in previews)
        for item, a, b in review.downstream_unresolved(data, covered):
            ranges = review.transcription_ranges(data,item,a,b) if for_transcription else [(a,b)]
            for start,end in ranges:
                if review.under_one_second(start,end):
                    continue
                ident = item['id']
                original_a, original_b = review.unresolved_bounds(item)
                if (round(start*16000),round(end*16000)) != (round(original_a*16000),round(original_b*16000)):
                    ident = f"{ident}@{round(start*16000)}-{round(end*16000)}"
                previews.append((item, start, end, ident, 'unresolved'))
    if previews:
        source = Path(source).resolve()
        stat = source.stat()
        for item, a, b, ident, status in previews:
            pauses = []
            for pause in data.get('pause_intervals', []):
                left, right = max(a,pause['start']), min(b,pause['end'])
                if right > left:
                    pauses.append([left-a,right-a])
            out.append(dict(id=ident,parent_span_id=item['id'],segmentation_status=status,
                file=source.name,path=str(source),read_start=a,read_end=b,source_start=a,source_end=b,
                audio_file_size=stat.st_size,audio_file_mtime_ns=stat.st_mtime_ns,
                content_label=item.get('label') or 'unclassified',synthetic_start=0.,synthetic_end=0.,
                pause_intervals=pauses or None,pause_basis='Detector intervals inside the span' if pauses else 'Quiet-frame fallback'))
    return sorted(out, key=source_time_key)


# Measure each item and tell the caller how far through the queue we are.
def calculate(records,threshold,progress=None,*,policy='conservative',justification=''):
    if not records:raise ValueError('No approved clips or unresolved spans are available for SNR.')
    # Validate controls before file I/O, even if all WAVs are invalid.
    decision(dict(analysis_version=2),threshold,policy,justification)
    result=[]
    for index,record in enumerate(records):
        item=copy.deepcopy(record)
        try:
            with wave.open(str(item['path']),'rb') as w:
                if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())!=(16000,1,2,'NONE'):
                    raise ValueError('Expected 16 kHz, mono, 16-bit PCM WAV.')
                a=round((item['read_start']+item.get('synthetic_start',0))*16000)
                b=round((item['read_end']-item.get('synthetic_end',0))*16000)
                if not 0<=a<b<=w.getnframes():raise ValueError('Invalid audio interval.')
                w.setpos(a);samples=np.frombuffer(w.readframes(b-a),dtype='<i2').astype(np.float64)/32768
            stat=Path(item['path']).stat();item.update(audio_file_size=stat.st_size,audio_file_mtime_ns=stat.st_mtime_ns)
            analysis=analyze_samples(samples,pause_intervals=item.get('pause_intervals'),content_label=item.get('content_label','speech'),threshold=threshold,policy=policy,justification=justification)
        except (OSError,ValueError,wave.Error) as exc:
            analysis=decision(dict(analysis_version=2,energy_snr_db=None,wada_snr_db=None,
                energy_details=dict(notes=[str(exc)]),wada_details=dict(notes=[str(exc)]),content_label=item.get('content_label','speech')),threshold,policy,justification)
        item.update(analysis)
        if (item.get('automatic_boundary_approval') or item.get('segmentation_status') in ('unresolved', 'pending')) and item.get('noise_source') == 'reviewed_part4_pause_context':
            item['noise_source'] = item['energy_details']['noise_source'] = 'estimated_part4_pause'
        result.append(item)
        if progress:progress(index+1,len(records))
    return result


# Changing the decision rule only needs the stored measurements.
def reclassify(records,threshold,policy='conservative',justification=''):
    result=copy.deepcopy(records)
    for item in result:
        if 'energy_snr_db' not in item:item['energy_snr_db']=item.get('snr_db')
        decision(item,threshold,policy,justification)
    return result


# Keep the reviewer's correction if this is still the same audio.
def preserve_override(current,previous):
    if previous and previous.get('override_by') and current.get('audio_sha256') and current['audio_sha256']==previous.get('audio_sha256') and current.get('content_label')==previous.get('content_label'):
        for key in ('override_by','override_reason','quality','quality_changes'):
            if key in previous:current[key]=copy.deepcopy(previous[key])
        decision(current,current['threshold_db'],current['decision_policy'],current['threshold_justification'])
    return current


# Record who changed the quality label and why, or restore the automatic choice.
def set_quality(item,label,who,reason):
    if label not in (None,'clean','noisy') or not who.strip() or not reason.strip():
        raise ValueError('Choose clean/noisy and enter reviewer name and correction reason.')
    after=item['automatic_quality'] if label is None else label
    item.setdefault('quality_changes',[]).append(dict(time_utc=review.utc_now(),reviewer=who,reason=reason,before=item['quality'],after=after))
    item.update(quality=after,override_by=who if label is not None else '',override_reason=reason if label is not None else '')
    decision(item,item.get('threshold_db',20),item.get('decision_policy','conservative'),item.get('threshold_justification',''))
    return item


def quality_status(item):
    """Display provenance without requiring an approval, including legacy rows."""
    if not item:return 'Not calculated'
    if item.get('override_by'):return 'Manually corrected'
    return 'Automatic (see notes)' if item.get('review_reasons') or item.get('quality_warning') else 'Automatic'


def optional_quality_record(record):
    """Migrate saved ASR quality metadata without changing scores or labels."""
    analysis=record.get('snr_analysis')
    if isinstance(analysis,dict) and analysis:
        analysis=dict(analysis)
        analysis.update(review_required=False,automatic_review_required=False,
                        quality_warning=bool(analysis.get('review_reasons') or analysis.get('quality_warning')),
                        review_status=quality_status(analysis))
        record['snr_analysis']=analysis
        record['snr_review_status']=analysis['review_status']
    else:
        record['snr_review_status']='Not calculated' if record.get('snr_db') is None else ('Manually corrected' if record.get('snr_override_by') else 'Automatic')
    record['snr_review_required']=False
    return record


def corpus_fields(item):
    """Preserve both measurements and review provenance alongside transcripts."""
    return dict(snr_db=item.get('snr_db'),energy_snr_db=item.get('energy_snr_db'),wada_snr_db=item.get('wada_snr_db'),
        snr_threshold_db=item.get('threshold_db',20),snr_decision_policy=item.get('decision_policy','conservative'),
        snr_threshold_justification=item.get('threshold_justification',''),
        snr_review_required=False,snr_review_status=quality_status(item),
        snr_note=item.get('note',''),snr_analysis=copy.deepcopy(item),clean_noisy=item.get('quality','unavailable'))


# Save both scores and the reason the reviewer chose this threshold.
def export_report(records,threshold,directory,*,policy='conservative',justification='',provisional=False):
    # Saving is permitted before calibration; never invent a justification.
    records=reclassify(records,threshold,policy,justification)
    for row in records:row.setdefault('segmentation_status','resolved')
    base=new_run_folder(directory) / 'step5_snr'
    complete=all(r.get('analysis_version')==2 for r in records)
    payload=dict(format='speech_corpus_snr_v2',created_utc=datetime.now(timezone.utc).isoformat(),threshold_db=float(threshold),
        threshold_justification=justification.strip(),decision_policy=policy,methods=list(METHODS) if complete else ['legacy_energy_only'],
        second_method='wada_snr' if complete else 'not_calculated_for_legacy_data',
        threshold_justification_provided=bool(justification.strip()),
        assumptions='Blind SNR estimates, not clean-reference measurements. Music, changing noise, clipped audio and estimator disagreement are informational caveats; listening and correction are optional in the app.',segments=records)
    base.with_suffix('.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    fields=['id','segmentation_status','file','source_start','source_end','content_label','energy_snr_db','wada_snr_db','snr_db','decision_policy','threshold_db',
        'threshold_justification','energy_quality','wada_quality','automatic_quality','quality','review_status','review_required','method_difference_db',
        'noise_source','noise_variation_db','energy_status','wada_status','wada_bound','override_by','override_reason','note']
    with base.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    e=lambda x:html.escape(str(x),quote=True)
    columns=[('id','ID'),('segmentation_status','Segmentation'),('content_label','Content'),('energy_snr_db','Pause energy (dB)'),('wada_snr_db','WADA (dB)'),('wada_status','WADA status'),('snr_db','Decision (dB)'),
             ('automatic_quality','Automatic'),('quality','Quality'),('noise_source','Noise estimate'),('note','Notes')]
    rows=''.join('<tr>'+''.join('<td>'+e(r.get(key,''))+'</td>' for key,title in columns)+'</tr>' for r in records)
    counts={k:sum(r['quality']==k for r in records) for k in ('clean','noisy','unavailable')}
    warning_count=sum(bool(r.get('quality_warning')) for r in records)
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audio Quality SNR</title><style>body{{font:15px/1.5 Segoe UI,Arial;margin:0;background:#EFF4F5;color:#173B40}}header{{background:#173B40;color:white;padding:28px}}main{{padding:22px}}section{{background:white;border:1px solid #D5E3E5;border-radius:8px;padding:20px;margin-bottom:18px}}.cards{{display:flex;gap:15px;flex-wrap:wrap}}.cards div{{background:#E8F3F0;padding:14px;border-radius:6px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #D5E3E5;text-align:left;vertical-align:top}}th{{background:#EAF2F3;white-space:nowrap}}h1,h2{{margin-top:0}}</style>
<header><h1>Audio Quality SNR</h1><div>Two independent estimates · optional manual corrections</div></header><main>
<section><div class="cards"><div>Clean: {counts['clean']}</div><div>Noisy: {counts['noisy']}</div><div>Unavailable: {counts['unavailable']}</div><div>With estimator notes: {warning_count}</div></div></section>
<section><h2>Threshold and decision</h2><p>Threshold: <b>{float(threshold):g} dB</b> · Rule: <b>{e(policy)}</b></p><p><b>Justification:</b> {e(justification.strip() or 'Not provided; justify the final threshold in the project report')}</p><p>The initial threshold is 20 dB. This threshold does not follow signal loudness. Human quality overrides are retained when the threshold changes.</p></section>
<section><h2>Methods and limits</h2><p>Pause-energy: 20 ms frames, 10 ms hop; subtract background-noise power estimated from natural pause anchors. Noise power is interpolated between observed anchors and held constant beyond them; unobserved changes cannot be recovered.</p><p>WADA-SNR: Kim and Stern (2008), alpha=0.4 waveform-distribution model and original CMU lookup table (−20 to 100 dB, 1 dB grid). Balanced blocks are combined by estimated signal/noise energies. Range-limited values are flagged and excluded from automatic decision selection. The JSON records every block and boundary condition.</p><p>These estimates are not ground truth. Disagreement above 6 dB, different quality labels, mixed/music content, unavailable methods and variable noise are flagged. These notes do not require approval; listen and correct labels if needed. No method silently replaces a human correction.</p></section>
<section class="scroll"><table><tr>{''.join('<th>'+e(title)+'</th>' for key,title in columns)}</tr>{rows}</table></section></main></html>'''
    base.with_suffix('.html').write_text(doc,encoding='utf-8')
    return tuple(str(base.with_suffix(ext)) for ext in ('.html','.json','.csv'))

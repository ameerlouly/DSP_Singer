"""Energy SNR estimation and portable Step 5 reports (no GUI dependencies)."""
import csv
import html
import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import step4_review as review


def estimate_energy(samples, rate=16000):
    """Contrast high-energy frames with quiet frames; an estimate, not ground truth."""
    x=np.asarray(samples,dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        return None, 'Expected finite mono audio samples.'
    n=round(rate*.02); hop=round(rate*.01)
    if len(x)<n:
        return None,'Insufficient audio for frame analysis.'
    power=np.array([np.mean(x[i:i+n]**2) for i in range(0,len(x)-n+1,hop)])
    # Digital zeros cannot estimate the acoustic noise floor.
    power=power[power>1e-10]
    if len(power)<10:
        return None,'Too little non-zero audio to estimate SNR.'
    noise=float(np.percentile(power,10)); observed=float(np.percentile(power,80))
    if noise<=0 or observed<=noise:
        return None,'No usable contrast between quiet and active frames.'
    estimate=10*math.log10(max(observed-noise,1e-12)/noise)
    flag='Energy estimate assumes quiet frames contain background noise; verify music or continuous speech.'
    if observed/noise<4:
        flag='Low energy contrast; pause/noise estimate is uncertain.'
    return round(estimate,2),flag


def session_records(data,source):
    queue=review.approved_playback_queue(data,source)
    items={x['id']:x for x in data['segments']}
    out=[]
    for path,a,b,label,ident in queue:
        item=items[ident]; pad_a=pad_b=0.0
        if item['source_mode'] in ('original_candidate','trimmed_candidate'):
            orig=item.get('original') or {}; dur=float(orig.get('duration',b))
            pad_a=max(0.0,min(b,float(orig.get('synthetic_silence_start_seconds',0)))-a)
            pad_b=max(0.0,b-max(a,dur-float(orig.get('synthetic_silence_end_seconds',0))))
        out.append({'id':ident,'file':Path(path).name,'path':path,'read_start':a,'read_end':b,
                    'source_start':item['source_start'],'source_end':item['source_end'],
                    'content_label':label,'synthetic_start':pad_a,'synthetic_end':pad_b})
    return out


def calculate(records, threshold, progress=None):
    threshold=float(threshold)
    if not math.isfinite(threshold) or not -50<=threshold<=100:
        raise ValueError('Threshold must be a finite number from -50 to 100 dB.')
    if not records: raise ValueError('There are no approved clips to analyze.')
    result=[]
    for index,record in enumerate(records):
        item=dict(record)
        try:
            with wave.open(str(item['path']),'rb') as w:
                if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())!=(16000,1,2,'NONE'):
                    raise ValueError('Expected 16 kHz, mono, 16-bit PCM WAV.')
                start=round((item['read_start']+item['synthetic_start'])*16000)
                end=round((item['read_end']-item['synthetic_end'])*16000)
                if not 0<=start<end<=w.getnframes():raise ValueError('Invalid audio or padding boundaries.')
                w.setpos(start); audio=np.frombuffer(w.readframes(end-start),dtype='<i2').astype(np.float64)/32768
            score,note=estimate_energy(audio)
        except (OSError,ValueError,wave.Error) as exc:
            score,note=None,str(exc)
        auto='unavailable' if score is None else ('clean' if score>=threshold else 'noisy')
        item.update(snr_db=score,automatic_quality=auto,quality=auto,note=note,override_by='',override_reason='')
        result.append(item)
        if progress:progress(index+1,len(records))
    return result


def export_report(records,threshold,directory):
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    base=dest/('step5_snr_'+stamp)
    payload={'format':'speech_corpus_snr_v1','created_utc':datetime.now(timezone.utc).isoformat(),
             'threshold_db':float(threshold),'method':'energy frames: 10th and 80th percentile; signal power = active minus quiet',
             'threshold_justification':'20 dB is the project starting point; final value needs listening-based validation.',
             'second_method':'not implemented','segments':records}
    json_path=base.with_suffix('.json'); json_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    csv_path=base.with_suffix('.csv'); fields=['id','file','source_start','source_end','content_label','snr_db','automatic_quality','quality','override_by','override_reason','note']
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    esc=lambda v:html.escape(str(v),quote=True)
    rows=''.join('<tr>'+''.join('<td>'+esc(r.get(k,''))+'</td>' for k in fields)+'</tr>' for r in records)
    counts={k:sum(x['quality']==k for x in records) for k in ('clean','noisy','unavailable')}
    doc=f'''<!doctype html><meta charset="utf-8"><title>Step 5 SNR report</title>
<style>body{{font:15px Segoe UI,Arial;margin:2em;color:#173a40}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd;padding:5px;text-align:left}}th{{background:#eaf4f4}} .scroll{{overflow:auto}}</style>
<h1>Step 5 SNR report</h1><p>Threshold: {threshold:g} dB. Clean: {counts['clean']}; noisy: {counts['noisy']}; unavailable: {counts['unavailable']}.</p>
<p>20 dB is the project starting point, not a validated universal cutoff. Energy estimate uses 20 ms frames with 10 ms hop: quiet 10th percentile and active 80th percentile, subtracting estimated noise power. Synthetic padding and exact/near digital silence are excluded. Quiet frames may not be noise-only; music and continuous speech reduce reliability. WADA-SNR is not yet implemented. Labels and manual overrides are recorded separately.</p>
<div class="scroll"><table><tr>{''.join('<th>'+esc(k)+'</th>' for k in fields)}</tr>{rows}</table></div>'''
    report=base.with_suffix('.html');report.write_text(doc,encoding='utf-8')
    return str(report),str(json_path),str(csv_path)

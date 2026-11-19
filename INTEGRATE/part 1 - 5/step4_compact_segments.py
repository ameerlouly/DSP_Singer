"""Step 4: join short speech across removed non-speech, retaining source mapping.

All clips are candidates for listening review. The original WAV is unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import html
import json
import time
import numpy as np
import soundfile as sf
from step4_fallback import plan_low_energy_splits, PAD
from step4_cut_segments import (eligible_blocks, detect_speech, refine_long_islands,
    missing_intervals, write_unresolved_audio, MIN_SECONDS, MAX_SECONDS)

RATE = 16000
EDGE_PAD = .10
KEEP_PAUSE_MAX = .75  # Keep ordinary short pauses within a speech/mixed block.


def build_atoms(islands, block_start, block_end):
    """Keep at most 0.1 s of original pause on each side of each speech island.

    Neighboring atoms never overlap; their original sample locations are kept.
    """
    result = []
    for i, (start, end) in enumerate(islands):
        left = block_start if not i else (islands[i-1][1] + start) / 2
        right = block_end if i + 1 == len(islands) else (end + islands[i+1][0]) / 2
        a = max(block_start, left, start - EDGE_PAD)
        b = min(block_end, right, end + EDGE_PAD)
        first, last = max(0, round(a * RATE)), max(0, round(b * RATE))
        if last > first:
            result.append({'first': first, 'last': last,
                           'speech_start': start, 'speech_end': end,
                           'block_start': block_start, 'block_end': block_end})
    return result



def kept_gap_samples(left, right):
    """Preserve short gaps within a speech/mixed block; never retain music-only regions."""
    same_block = left['block_start'] == right['block_start'] and left['block_end'] == right['block_end']
    gap = right['first'] - left['last']
    return gap if same_block and 0 <= gap <= round(KEEP_PAUSE_MAX * RATE) else 0


def plan_groups(atoms):
    """Maximize VAD speech represented in valid compressed-duration clips.

    Skipped atoms remain unresolved; no arbitrary cut is made inside one.
    """
    n = len(atoms)
    score = [0.0] * (n+1)
    choices = [None] * n
    for i in range(n-1, -1, -1):
        best = score[i+1]
        sample_count = 0
        weight = 0.0
        for j in range(i, n):
            a = atoms[j]
            sample_count += a['last'] - a['first']
            if j > i:
                sample_count += kept_gap_samples(atoms[j-1], a)
            weight += a['speech_end'] - a['speech_start']
            if sample_count > int(MAX_SECONDS * RATE):
                break
            if sample_count >= int(MIN_SECONDS * RATE):
                if weight + score[j+1] > best + 1e-9:
                    best = weight + score[j+1]
                    choices[i] = j
        score[i] = best
    groups, skipped = [], []
    i = 0
    while i < n:
        if choices[i] is None:
            skipped.append(atoms[i])
            i += 1
        else:
            j = choices[i]
            groups.append(atoms[i:j+1])
            i = j+1
    return groups, skipped


def report_html(data):
    e = lambda x: html.escape(str(x), quote=True)
    rows=[]
    for x in data['segments']:
        parts = '<br>'.join(f"{p['start']:.3f}–{p['end']:.3f} s" for p in x['source_spans'])
        notes=e('; '.join(x['notes']))
        src=e(x['file'])
        kind = 'Low-energy fallback' if x.get('kind') == 'low_energy_fallback' else 'VAD candidate'
        rows.append(f"<tr><td>{src}</td><td>{kind}</td><td>{x['duration']:.3f}</td><td>{e(x['label'])}</td><td>{parts}</td>"
                    f"<td>{notes}</td><td><audio controls preload='none' src='{src}'></audio></td></tr>")
    pending=[]
    for x in data['unresolved']:
        f=e(x.get('review_file',''))
        sound=f"<a href='{f}'>Open WAV</a><br><audio controls preload='none' src='{f}'></audio>" if f else ''
        reason=x['reason']
        if x.get('candidate_files'):
            reason += f" ({len(x['candidate_files'])} low-energy fallback candidates; still requires review)"
        pending.append(f"<tr><td>{x['start']:.3f}</td><td>{x['end']:.3f}</td><td>{e(reason)}</td><td>{sound}</td></tr>")
    gaps=''.join(f"<tr><td>{x['start']:.3f}</td><td>{x['end']:.3f}</td><td>{x['end']-x['start']:.3f}</td></tr>" for x in data['not_in_clips_or_unresolved'])
    c=data['coverage_seconds']
    fallback_count=sum(x.get('kind')=='low_energy_fallback' for x in data['segments'])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Step 4 speech candidates</title>
<style>body{{font:15px/1.5 Segoe UI,Arial;margin:0;color:#173a40;background:#f1f6f7}}header{{background:#173a40;color:white;padding:24px}}main{{max-width:1400px;margin:auto;padding:18px}}section{{background:white;padding:18px;margin-bottom:15px;border:1px solid #d7e5e5;border-radius:10px}}h1,h2{{margin-top:0}}.scroll{{overflow:auto;max-height:650px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #dce6e8;padding:8px}}th{{background:#edf5f5}}audio{{width:220px}}.notice{{background:#fff5da}}</style></head>
<body><header><h1>Step 4: speech candidates</h1><div>{e(data['audio_name'])}</div></header><main>
<section class="notice"><b>Review every cut and join.</b> Pauses and labels are estimates; short pauses, music, or speech may be removed incorrectly. A candidate with several source spans is a concatenation, so the original start–end interval contains removed audio. Unresolved WAVs are not approved 5–15 s clips.</section>
<section><h2>Coverage</h2><p>Source: {data['duration_seconds']:.2f} s. Step 3 speech/mixed: {c['eligible_step3']:.2f} s. Candidate duration (may include padding): {sum(x['duration'] for x in data['segments']):.2f} s in {len(data['segments'])} clips, including {fallback_count} low-energy fallback clips. Unresolved: {c['unresolved']:.2f} s in {len(data['unresolved'])} spans. Other speech/mixed time outside clips and unresolved: {c['not_in_clips_or_unresolved']:.2f} s. Step 3 music/silence excluded: {c['excluded_step3_music_or_silence']:.2f} s.</p><p>Fallback clips cover portions of unresolved spans: those totals overlap and must not be added. Fallback boundaries are low-energy guesses, not verified pauses; their edge silence is synthesized. VAD can miss speech in excluded gaps. Review the original audio and Step 3 labels before final transcription.</p></section>
<section><h2>Candidate clips</h2><div class="scroll"><table><tr><th>File</th><th>Method</th><th>Duration (s)</th><th>Label</th><th>Source spans in order</th><th>Review notes</th><th>Listen</th></tr>{''.join(rows)}</table></div></section>
<section><h2>Unresolved spans</h2><p>These are saved intact with context in needs_manual_review, and may be shorter than 5 s or longer than 15 s. Long spans may also have low-energy fallback candidates above; they remain unresolved until their boundaries are manually verified.</p><div class="scroll"><table><tr><th>Start (s)</th><th>End (s)</th><th>Reason</th><th>Listen</th></tr>{''.join(pending)}</table></div></section>
<section><h2>Eligible audio outside candidate spans and unresolved audio</h2><p>These intervals may be silence or missed speech; listen in the original WAV.</p><details><summary>Show {len(data['not_in_clips_or_unresolved'])} intervals</summary><div class="scroll"><table><tr><th>Start (s)</th><th>End (s)</th><th>Duration (s)</th></tr>{gaps}</table></div></details></section>
<section>Complete ordered source spans, removed gaps, labels, and the Step 3 snapshot are available in step4_segments.json. Original WAV is unchanged.</section></main></body></html>'''


def export_candidates(audio_path, regions, output_parent, progress=lambda text: None):
    """Export joinable 5–15 s speech candidates and preserve unresolved audio."""
    import torch
    from silero_vad import load_silero_vad
    path=Path(audio_path)
    with sf.SoundFile(path) as audio:
        if (audio.samplerate,audio.channels,audio.subtype)!=(RATE,1,'PCM_16'):
            raise ValueError('Step 4 requires the Step 2 WAV format: 16 kHz, mono, 16-bit PCM.')
        duration=len(audio)/RATE
        previous=0.0
        if not regions:
            raise ValueError('Run or load Step 3 review first.')
        for r in regions:
            a,b=float(r['start']),float(r['end'])
            if r['label'] not in ('speech','mixed','music','silence') or abs(a-previous)>1e-4 or not a<b<=duration:
                raise ValueError('Step 3 regions do not form a valid continuous timeline.')
            previous=b
        if abs(previous-duration)>1e-4:
            raise ValueError('Step 3 regions do not cover the selected WAV.')
        blocks=eligible_blocks(regions)
        if not blocks:
            raise ValueError('No speech or mixed regions. Review Step 3 labels first.')
        torch.set_num_threads(1)
        progress('Loading Silero VAD...')
        model=load_silero_vad()
        output=Path(output_parent)/(path.stem+'_step4_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns())[-6:])
        output.mkdir(parents=True,exist_ok=False)
        data={'version':2,'audio_name':path.name,'audio_size':path.stat().st_size,'duration_seconds':duration,
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'method':'Silero 200/80 ms and speech joining; long unresolved speech has low-energy fallback cuts with synthetic silence; all need review',
              'eligible_block_count':len(blocks),'target_padding_seconds_per_island_edge':EDGE_PAD,
              'keep_original_short_pause_max_seconds':KEEP_PAUSE_MAX,
              'step3_regions':copy.deepcopy(regions),'segments':[],'unresolved':[]}
        atoms=[]
        # Runs are separated by unprocessable speech so no clip joins across it.
        runs=[]
        def flush():
            if atoms:
                runs.append(atoms.copy())
                atoms.clear()
        for number,(start,end) in enumerate(blocks,1):
            progress(f'Detecting speech in block {number}/{len(blocks)}...')
            islands=detect_speech(audio,start,end,model,progress)
            if not islands:
                flush()
                data['unresolved'].append({'start':start,'end':end,'reason':'No speech detected in this speech/mixed region; review entire region.'})
                continue
            islands=refine_long_islands(audio,islands,model,progress)
            for atom in build_atoms(islands,start,end):
                if (atom['last']-atom['first']) > int(MAX_SECONDS*RATE):
                    flush()
                    data['unresolved'].append({'start':atom['speech_start'],'end':atom['speech_end'],
                        'reason':'No suitable pause found inside this speech span; review manually.'})
                else:
                    atoms.append(atom)
        flush()
        for run in runs:
            groups, skipped=plan_groups(run)
            for atom in skipped:
                data['unresolved'].append({'start':atom['speech_start'],'end':atom['speech_end'],
                    'reason':'Could not combine this speech with nearby speech into a 5–15 s candidate.'})
            for group in groups:
                arrays=[]
                pieces=[]
                for index, atom in enumerate(group):
                    if index and kept_gap_samples(group[index-1], atom) == atom['first'] - group[index-1]['last'] and atom['block_start'] == group[index-1]['block_start']:
                        pieces[-1]['last'] = atom['last']
                    else:
                        pieces.append({'first':atom['first'], 'last':atom['last']})
                spans=[]
                offset=0
                for piece in pieces:
                    audio.seek(piece['first'])
                    arr=audio.read(piece['last']-piece['first'],dtype='float32',always_2d=True)
                    if len(arr)!=piece['last']-piece['first']:
                        raise RuntimeError('Unexpected end of WAV during segment export.')
                    arrays.append(arr)
                    spans.append({'start':piece['first']/RATE,'end':piece['last']/RATE,
                                  'clip_start':offset/RATE,'clip_end':(offset+len(arr))/RATE})
                    offset+=len(arr)
                samples=np.concatenate(arrays,axis=0)
                if not int(MIN_SECONDS*RATE)<=len(samples)<=int(MAX_SECONDS*RATE):
                    raise RuntimeError('Candidate length validation failed.')
                overlaps=[r for r in regions if r['label'] in ('speech','mixed') and
                          any(r['end']>p['start'] and r['start']<p['end'] for p in spans)]
                label='mixed' if any(r['label']=='mixed' for r in overlaps) else 'speech'
                removed=[{'start':a['end'],'end':b['start'],'duration':b['start']-a['end']}
                         for a,b in zip(spans,spans[1:]) if b['start']-a['end']>1/RATE]
                notes=['All boundaries and joins require listening review.']
                if group[0]['speech_start'] - spans[0]['start'] < .099 or spans[-1]['end'] - group[-1]['speech_end'] < .099:
                    notes.append('Less than 0.1 s estimated non-speech at one or both outer edges.')
                if removed:
                    notes.append(f'{len(removed)} original gap(s) removed; audio is non-contiguous.')
                if any(abs(a['first']/RATE-a['block_start']) < 0.002 or abs(a['last']/RATE-a['block_end']) < 0.002 for a in group):
                    notes.append('A piece touches a Step 3 boundary; a word could be clipped.')
                if label=='mixed': notes.append('Mixed assigned if any source piece overlaps a mixed region; verify music.')
                if any(r.get('needs_review') for r in overlaps): notes.append('A source Step 3 label requires review.')
                filename=f"segment_{len(data['segments'])+1:05d}.wav"
                sf.write(output/filename,samples,RATE,subtype='PCM_16')
                data['segments'].append({'kind':'vad_candidate','file':filename,'start':spans[0]['start'],'end':spans[-1]['end'],
                    'duration':len(samples)/RATE,'label':label,'source_spans':spans,'removed_gaps':removed,
                    'status':'needs_review','notes':notes})
        data['unresolved'].sort(key=lambda x:x['start'])
        progress('Proposing low-energy breaks for long unresolved speech...')
        for item in data['unresolved']:
            if item['reason'] != 'No suitable pause found inside this speech span; review manually.':
                continue
            first=max(0,round(item['start']*RATE))
            last=min(len(audio),round(item['end']*RATE))
            plans=plan_low_energy_splits(audio, first, last)
            if not plans:
                continue
            item['candidate_files']=[]
            item['review_status']='pending: candidate cuts are not verified pauses'
            for part in plans:
                a,b=part['start'],part['end']
                audio.seek(a)
                speech=audio.read(b-a,dtype='float32',always_2d=True)
                if len(speech)!=b-a:
                    raise RuntimeError('Unexpected end of WAV during fallback export.')
                # Artificial silence is separate from source audio and recorded.
                pad=np.zeros((PAD,1),dtype=np.float32)
                samples=np.concatenate((pad,speech,pad),axis=0)
                if not int(MIN_SECONDS*RATE)<=len(samples)<=int(MAX_SECONDS*RATE):
                    raise RuntimeError('Fallback candidate length validation failed.')
                overlaps=[r for r in regions if r['label'] in ('speech','mixed') and r['end']>a/RATE and r['start']<b/RATE]
                label='mixed' if any(r['label']=='mixed' for r in overlaps) else 'speech'
                filename=f"segment_{len(data['segments'])+1:05d}.wav"
                sf.write(output/filename,samples,RATE,subtype='PCM_16')
                notes=['Low-energy fallback: internal cut is NOT a verified word boundary.',
                       '0.1 s synthetic silence at each edge; it is not a detected natural pause.',
                       'Listen around both cuts in the original WAV before accepting.']
                if label=='mixed':
                    notes.append('Mixed assigned if any source part overlaps a mixed region; verify music.')
                data['segments'].append({'kind':'low_energy_fallback','file':filename,'start':a/RATE,'end':b/RATE,
                    'duration':len(samples)/RATE,'label':label,
                    'source_spans':[{'start':a/RATE,'end':b/RATE,'clip_start':PAD/RATE,'clip_end':(PAD+b-a)/RATE}],
                    'removed_gaps':[], 'synthetic_silence_start_seconds':PAD/RATE,
                    'synthetic_silence_end_seconds':PAD/RATE,
                    'candidate_left_rms':part['left_cut_rms'],'candidate_right_rms':part['right_cut_rms'],
                    'status':'needs_review_synthetic_break','notes':notes})
                item['candidate_files'].append(filename)
        # Fallback clips were generated after ordinary clips. Restore source
        # order for downstream transcription and update all cross-references.
        data['segments'].sort(key=lambda segment:segment['start'])
        renamed={}
        staged=[]
        for index, segment in enumerate(data['segments'],1):
            old=segment['file']
            new=f"segment_{index:05d}.wav"
            temporary=output/f".stage_{index:05d}.wav"
            (output/old).rename(temporary)
            staged.append((temporary,new))
            renamed[old]=new
            segment['file']=new
        for temporary,new in staged:
            temporary.rename(output/new)
        for item in data['unresolved']:
            if item.get('candidate_files'):
                item['candidate_files']=[renamed[name] for name in item['candidate_files']]
        progress('Saving unresolved speech for manual review...')
        for i,item in enumerate(data['unresolved'],1): write_unresolved_audio(audio,item,output,i)
        pieces=[piece for segment in data['segments'] for piece in segment['source_spans']]
        data['not_in_clips_or_unresolved']=missing_intervals(blocks,pieces,data['unresolved'])
        data['coverage_seconds']={'eligible_step3':sum(b-a for a,b in blocks),
            'unresolved':sum(x['end']-x['start'] for x in data['unresolved']),
            'not_in_clips_or_unresolved':sum(x['end']-x['start'] for x in data['not_in_clips_or_unresolved']),
            'excluded_step3_music_or_silence':sum(r['end']-r['start'] for r in regions if r['label'] in ('music','silence'))}
        (output/'step4_segments.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'step4_report.html').write_text(report_html(data),encoding='utf-8')
    return {'output_dir':str(output),'segments':len(data['segments']),'unresolved':len(data['unresolved'])}

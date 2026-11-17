"""Step 4: generate reviewable 5-15 second candidates at Silero VAD pauses.

The input region list is the same list shown by Step 3. Export never changes
that list or the source WAV. Regions with no safe-length grouping are recorded
for manual review rather than being forcibly cut through detected speech.
"""
from datetime import datetime, timezone
from pathlib import Path
import copy
import html
import json
import math
import time
import numpy as np
import soundfile as sf

MIN_SECONDS = 5.0
MAX_SECONDS = 15.0
EDGE_SECONDS = 0.15
MIN_PAUSE_MS = 200
REFINE_PAUSE_MS = 80
CHUNK_SECONDS = 120


def eligible_blocks(regions):
    """Combine adjacent speech and mixed labels, without crossing other labels."""
    blocks = []
    for r in regions:
        if r['label'] not in ('speech', 'mixed'):
            continue
        start, end = float(r['start']), float(r['end'])
        if blocks and abs(blocks[-1][1] - start) < 1e-5:
            blocks[-1][1] = end
        else:
            blocks.append([start, end])
    return blocks


def detect_speech(audio, start, end, model, progress, pause_ms=MIN_PAUSE_MS):
    """Find speech islands in an eligible block, reading at most 120 s at a time."""
    import torch
    from silero_vad import get_speech_timestamps
    rate = 16000
    start_sample, end_sample = round(start * rate), round(end * rate)
    ranges = []
    chunk_start = start_sample
    while chunk_start < end_sample:
        chunk_end = min(end_sample, chunk_start + CHUNK_SECONDS * rate)
        audio.seek(chunk_start)
        samples = audio.read(chunk_end - chunk_start, dtype='float32', always_2d=True)[:, 0]
        if not len(samples):
            raise RuntimeError('Unexpected end of WAV while reading audio.')
        detected = get_speech_timestamps(torch.from_numpy(samples), model, sampling_rate=rate,
                min_silence_duration_ms=pause_ms, min_speech_duration_ms=100,
                speech_pad_ms=0, return_seconds=False)
        for d in detected:
            a, b = (chunk_start + d['start']) / rate, (chunk_start + d['end']) / rate
            if b <= a:
                continue
            # If VAD produces adjacent islands at a processing boundary, do
            # not turn that artificial boundary into a candidate pause.
            if ranges and a - ranges[-1][1] < pause_ms / 1000:
                ranges[-1][1] = max(ranges[-1][1], b)
            else:
                ranges.append([a, b])
        chunk_start = chunk_end
        progress(f'Detecting pauses: {min(end, chunk_start / rate):.1f} / {end:.1f} seconds')
    return ranges



def refine_long_islands(audio, islands, model, progress):
    """Recheck long islands using shorter VAD pause candidates, without forced cuts."""
    result = []
    for start, end in islands:
        if end - start <= MAX_SECONDS:
            result.append([start, end])
            continue
        progress(f'Rechecking long speech span {start:.1f}–{end:.1f} s with {REFINE_PAUSE_MS} ms VAD pauses...')
        refined = detect_speech(audio, start, end, model, progress, pause_ms=REFINE_PAUSE_MS)
        # The second pass can miss speech at the edges. Keep the original
        # island boundaries so that no previously detected speech disappears.
        if not refined:
            result.append([start, end])
            continue
        refined[0][0] = start
        refined[-1][1] = end
        result.extend(refined)
    return result


def missing_intervals(blocks, segments, unresolved):
    """Exact eligible timeline not included in a candidate or an unresolved span."""
    covered = sorted((float(x['start']), float(x['end'])) for x in segments + unresolved)
    missing = []
    for block_start, block_end in blocks:
        cursor = block_start
        for a, b in covered:
            if b <= cursor:
                continue
            if a >= block_end:
                break
            if a > cursor + 1e-8:
                missing.append({'start': cursor, 'end': min(a, block_end),
                                'reason': 'Not in a candidate or unresolved span; listen for VAD-missed speech.'})
            cursor = max(cursor, min(b, block_end))
            if cursor >= block_end:
                break
        if cursor < block_end - 1e-8:
            missing.append({'start': cursor, 'end': block_end,
                            'reason': 'Not in a candidate or unresolved span; listen for VAD-missed speech.'})
    return missing


def write_unresolved_audio(audio, item, output, index):
    """Preserve an unresolved span intact, including up to 0.15 s context."""
    rate = audio.samplerate
    start = max(0, int(math.floor((item['start'] - EDGE_SECONDS) * rate)))
    end = min(len(audio), int(math.ceil((item['end'] + EDGE_SECONDS) * rate)))
    if end <= start:
        return
    folder = output / 'needs_manual_review'
    folder.mkdir(exist_ok=True)
    filename = f'unresolved_{index:05d}.wav'
    audio.seek(start)
    with sf.SoundFile(folder / filename, mode='w', samplerate=rate, channels=1,
                      format='WAV', subtype='PCM_16') as dst:
        remaining = end - start
        while remaining:
            samples = audio.read(min(rate * 30, remaining), dtype='float32', always_2d=True)
            if not len(samples):
                raise RuntimeError('Unexpected end of WAV while saving unresolved audio.')
            dst.write(samples)
            remaining -= len(samples)
    item['review_file'] = 'needs_manual_review/' + filename
    item['review_file_start'] = start / rate
    item['review_file_end'] = end / rate


def candidate_bounds(speech, i, j, block_start, block_end):
    left_limit = block_start if i == 0 else speech[i - 1][1]
    right_limit = block_end if j + 1 == len(speech) else speech[j + 1][0]
    start = max(left_limit, speech[i][0] - EDGE_SECONDS)
    end = min(right_limit, speech[j][1] + EDGE_SECONDS)
    return start, end


def group_at_pauses(speech, block_start, block_end):
    """Maximize detected speech covered by valid clips, skipping unsplittable spans."""
    n = len(speech)
    scores = [0.0] * (n + 1)
    choices = [None] * n
    for i in range(n - 1, -1, -1):
        best = scores[i + 1]
        speech_seconds = 0.0
        for j in range(i, n):
            speech_seconds += speech[j][1] - speech[j][0]
            start, end = candidate_bounds(speech, i, j, block_start, block_end)
            length = end - start
            if length > MAX_SECONDS + 1e-7:
                break
            if length + 1e-7 >= MIN_SECONDS:
                score = speech_seconds + scores[j + 1]
                if score > best + 1e-9:
                    best, choices[i] = score, (j, start, end)
        scores[i] = best
    groups, unresolved = [], []
    i = 0
    while i < n:
        if choices[i] is None:
            unresolved.append({'start': speech[i][0], 'end': speech[i][1],
                               'reason': 'No 5–15 s candidate could be formed at detected pauses.'})
            i += 1
        else:
            j, start, end = choices[i]
            groups.append({'start': start, 'end': end, 'speech_start': speech[i][0],
                           'speech_end': speech[j][1]})
            i = j + 1
    return groups, unresolved


def render_html(data):
    e = lambda x: html.escape(str(x), quote=True)
    rows = []
    for s in data['segments']:
        filename = e(s['file'])
        rows.append(f"<tr><td>{e(s['file'])}</td><td>{s['start']:.3f}</td><td>{s['end']:.3f}</td>"
                    f"<td>{s['duration']:.3f}</td><td>{e(s['label'])}</td>"
                    f"<td>{s['left_padding']:.3f}</td><td>{s['right_padding']:.3f}</td>"
                    f"<td>{e('; '.join(s['notes']))}</td><td><audio controls preload='none' src='{filename}'></audio></td></tr>")
    misses = []
    for u in data['unresolved']:
        audio = ''
        if u.get('review_file'):
            name = e(u['review_file'])
            audio = f"<a href='{name}'>Open WAV</a><br><audio controls preload='none' src='{name}'></audio>"
        misses.append(f"<tr><td>{u['start']:.3f}</td><td>{u['end']:.3f}</td><td>{e(u['reason'])}</td><td>{audio}</td></tr>")
    unchecked = data.get('not_in_clips_or_unresolved', [])
    unchecked_rows = ''.join(f"<tr><td>{item['start']:.3f}</td><td>{item['end']:.3f}</td><td>{item['end'] - item['start']:.3f}</td></tr>" for item in unchecked)
    coverage = data.get('coverage_seconds', {})
    eligible = coverage.get('eligible_step3', 0.0)
    unresolved_time = coverage.get('unresolved_detected_or_no_vad', 0.0)
    unchecked_time = coverage.get('not_in_clips_or_unresolved', 0.0)
    excluded = coverage.get('excluded_step3_music_or_silence', 0.0)
    clip_time = sum(clip['duration'] for clip in data['segments'])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step 4 candidate segments</title><style>body{{font:15px/1.5 Segoe UI,Arial;margin:0;color:#173a40;background:#f1f6f7}}header{{background:#173a40;color:white;padding:24px}}main{{max-width:1400px;margin:auto;padding:18px}}section{{background:white;padding:18px;margin-bottom:15px;border:1px solid #d7e5e5;border-radius:10px}}h1,h2{{margin-top:0}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #dce6e8;padding:8px}}th{{background:#edf5f5;white-space:nowrap}}audio{{width:220px}}.notice{{background:#fff5da}}</style></head><body>
<header><h1>Step 4: candidate segments</h1><div>{e(data['audio_name'])}</div></header><main>
<section class="notice"><b>Human review required.</b> Detected pauses are candidates, not proof that words are intact. Listen to every clip, especially mixed audio and cuts at Step 3 region edges. Padding describes VAD-estimated non-speech, which can include music or noise.</section>
<section><h2>Overview</h2><p><b>Candidate clips:</b> {len(data['segments'])} &nbsp; <b>Unresolved spans:</b> {len(data['unresolved'])} &nbsp; <b>Eligible blocks:</b> {data['eligible_block_count']}</p><p>Settings: 5–15 s; intended edge padding 0.15 s where available; Silero VAD first pass {MIN_PAUSE_MS} ms; long spans rechecked at {REFINE_PAUSE_MS} ms. A clip may have shorter padding at a region boundary or short pause.</p></section>
<section><h2>Time accounting</h2><p>Speech/mixed according to Step 3: {eligible:.2f} s. Total candidate duration: {clip_time:.2f} s (may include overlapping padding). Unresolved spans: {unresolved_time:.2f} s. Other eligible audio outside candidates and unresolved spans: {unchecked_time:.2f} s. Step 3 music/silence excluded from segmentation: {excluded:.2f} s.</p><p>Unchecked audio can include VAD-missed speech; the music/silence labels can also be wrong. Review them against the original audio. These categories are time-accounting aids, not a measurement of transcription completeness.</p></section>
<section><h2>Candidate clips</h2><div class="scroll"><table><tr><th>File</th><th>Start (s)</th><th>End (s)</th><th>Duration (s)</th><th>Label</th><th>Left padding (s)</th><th>Right padding (s)</th><th>Review notes</th><th>Listen</th></tr>{''.join(rows)}</table></div></section>
<section><h2>Unresolved spans</h2><p>Each span is also saved intact with a little context in needs_manual_review. These WAVs can be shorter than 5 seconds or longer than 15 seconds and are not accepted corpus clips. Inspect and select valid pause boundaries manually.</p><div class="scroll"><table><tr><th>Start (s)</th><th>End (s)</th><th>Reason</th><th>Review audio</th></tr>{''.join(misses)}</table></div></section>
<section><h2>Eligible audio outside clips and unresolved spans</h2><p>{len(unchecked)} intervals, total {unchecked_time:.2f} s. The original WAV remains unchanged. These VAD non-speech gaps may contain missed words.</p><details><summary>Show interval times</summary><div class="scroll"><table><tr><th>Start (s)</th><th>End (s)</th><th>Duration (s)</th></tr>{unchecked_rows}</table></div></details></section>
<section>Machine-readable results and the Step 3 snapshot are in step4_segments.json in this folder. Times refer to the original WAV.</section></main></body></html>'''


def export_candidates(audio_path, regions, output_parent, progress=lambda text: None):
    """Export WAV candidates plus HTML and JSON reports to a new subfolder."""
    from silero_vad import load_silero_vad
    import torch
    path = Path(audio_path)
    with sf.SoundFile(path) as audio:
        if (audio.samplerate, audio.channels, audio.subtype) != (16000, 1, 'PCM_16'):
            raise ValueError('Step 4 requires the Step 2 WAV format: 16 kHz, mono, 16-bit PCM.')
        duration = len(audio) / audio.samplerate
        if not regions or abs(float(regions[0]['start'])) > 1e-4 or abs(float(regions[-1]['end']) - duration) > 1e-4:
            raise ValueError('The Step 3 regions must cover the selected WAV.')
        previous = 0.0
        for r in regions:
            a, b = float(r['start']), float(r['end'])
            if r['label'] not in ('speech', 'music', 'mixed', 'silence') or abs(a - previous) > 1e-4 or not a < b <= duration:
                raise ValueError('Invalid or non-contiguous Step 3 regions.')
            previous = b
        blocks = eligible_blocks(regions)
        if not blocks:
            raise ValueError('No regions are labeled speech or mixed. Review Step 3 labels first.')
        torch.set_num_threads(1)
        progress('Loading Silero VAD...')
        model = load_silero_vad()
        output = Path(output_parent) / (path.stem + '_step4_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(time.time_ns())[-6:])
        output.mkdir(parents=True, exist_ok=False)
        data = {'version': 1, 'audio_name': path.name, 'audio_size': path.stat().st_size,
                'duration_seconds': duration, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'method': 'Silero VAD: 200 ms pauses, long islands rechecked at 80 ms, 120 s inference chunks',
                'min_duration': MIN_SECONDS, 'max_duration': MAX_SECONDS, 'target_edge_seconds': EDGE_SECONDS,
                'eligible_block_count': len(blocks), 'step3_regions': copy.deepcopy(regions),
                'segments': [], 'unresolved': []}
        for number, (block_start, block_end) in enumerate(blocks, 1):
            progress(f'Analyzing speech/mixed block {number} of {len(blocks)}...')
            islands = detect_speech(audio, block_start, block_end, model, progress)
            if not islands:
                data['unresolved'].append({'start': block_start, 'end': block_end,
                                           'reason': 'VAD detected no speech; listen and review the entire block.'})
                continue
            islands = refine_long_islands(audio, islands, model, progress)
            groups, unresolved = group_at_pauses(islands, block_start, block_end)
            data['unresolved'].extend(unresolved)
            for segment in groups:
                a, b = round(segment['start'] * 16000), round(segment['end'] * 16000)
                if (b - a) < MIN_SECONDS * 16000 or (b - a) > MAX_SECONDS * 16000:
                    data['unresolved'].append({'start': segment['speech_start'], 'end': segment['speech_end'],
                                               'reason': 'Duration fell outside 5–15 seconds after sample rounding.'})
                    continue
                overlapping = [r for r in regions if r['label'] in ('speech', 'mixed') and r['end'] > a / 16000 and r['start'] < b / 16000]
                label = 'mixed' if any(r['label'] == 'mixed' for r in overlapping) else 'speech'
                left_padding = max(0.0, segment['speech_start'] - a / 16000)
                right_padding = max(0.0, b / 16000 - segment['speech_end'])
                notes = ['Candidate only: listen to both edges and check the label.']
                label_overlap = {name: sum(max(0.0, min(float(r['end']), b / 16000) - max(float(r['start']), a / 16000))
                                           for r in overlapping if r['label'] == name)
                                 for name in ('speech', 'mixed')}
                if label_overlap['mixed'] > 0 and label_overlap['speech'] > 0:
                    notes.append(f"Step 3 overlaps: speech {label_overlap['speech']:.2f} s, mixed {label_overlap['mixed']:.2f} s; mixed label is conservative.")
                if left_padding < 0.1 or right_padding < 0.1:
                    notes.append('Less than 0.1 s detected non-speech at one or both edges.')
                if abs(a / 16000 - block_start) < 0.001 or abs(b / 16000 - block_end) < 0.001:
                    notes.append('Clip touches a Step 3 boundary; check for clipped words.')
                if label == 'mixed':
                    notes.append('Music can reduce VAD accuracy; padding may contain music.')
                if any(r.get('needs_review') for r in overlapping):
                    notes.append('At least one source region was flagged in Step 3.')
                filename = f"segment_{len(data['segments']) + 1:05d}.wav"
                audio.seek(a)
                samples = audio.read(b - a, dtype='float32', always_2d=True)
                if len(samples) != b - a:
                    raise RuntimeError('Unexpected end of WAV while exporting a candidate.')
                sf.write(output / filename, samples, 16000, subtype='PCM_16')
                data['segments'].append({'file': filename, 'start': a / 16000, 'end': b / 16000,
                    'duration': (b - a) / 16000, 'label': label, 'left_padding': left_padding,
                    'right_padding': right_padding, 'step3_label_overlap_seconds': label_overlap,
                    'status': 'needs_review', 'notes': notes})
        progress('Preserving unresolved spans for manual review...')
        for index, item in enumerate(data['unresolved'], 1):
            write_unresolved_audio(audio, item, output, index)
        data['not_in_clips_or_unresolved'] = missing_intervals(blocks, data['segments'], data['unresolved'])
        data['coverage_seconds'] = {
            'eligible_step3': sum(b - a for a, b in blocks),
            'unresolved_detected_or_no_vad': sum(u['end'] - u['start'] for u in data['unresolved']),
            'not_in_clips_or_unresolved': sum(u['end'] - u['start'] for u in data['not_in_clips_or_unresolved']),
            'excluded_step3_music_or_silence': sum(float(r['end']) - float(r['start']) for r in regions
                                                    if r['label'] in ('music', 'silence')),
        }
        (output / 'step4_segments.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        (output / 'step4_report.html').write_text(render_html(data), encoding='utf-8')
    return {'output_dir': str(output), 'segments': len(data['segments']), 'unresolved': len(data['unresolved'])}

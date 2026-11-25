"""Propose continuous speech clips at supported pauses and keep intervals that still need a reviewer.

Function guide
--------------
vocal_pause_candidates, split_at_vocal_pauses, vocal_cut_evidence,
vocal_pause_settings, mixed_pause_separator
    Locate and describe possible breaks in separated vocals.
merge_intervals, quiet_mixed_intervals, protect_mixed_continuity, explain_mixed_gaps,
eligible_blocks
    Combine source intervals and preserve possible sung words when accompaniment masks
    a pause.
detect_speech, recover_long_spans, adjacent_silence_limits
    Locate speech activity and examine long intervals for boundaries.
missing_intervals, write_unresolved_audio, subtract_intervals
    Find uncovered ranges and preserve their audio for review.
candidate_bounds, group_at_pauses, source_pause_intervals, pdf_pause_plan,
edge_pause_evidence
    Choose candidate endpoints using source pauses, padding and duration constraints.
render_html, export_candidates
    Write candidate audio, unresolved intervals and review metadata.

The input region list is the same list shown by Step 3. Export never changes
that list or the source WAV. Regions with no safe-length grouping are recorded
for manual review rather than being forcibly cut through detected speech.
"""
from output_folders import new_run_folder
from datetime import datetime, timezone
from pathlib import Path
import copy
import html
import json
import math
import numpy as np
import soundfile as sf

MIN_SECONDS = 5.0
MAX_SECONDS = 15.0
EDGE_SECONDS = 0.15
MIN_PAUSE_MS = 240
REFINE_PAUSE_MS = 200
CHUNK_SECONDS = 120
MIXED_QUIET_FRAME_SECONDS = 0.02
MIXED_QUIET_MIN_SECONDS = 0.20
MIXED_QUIET_MAX_DBFS = -55.0
MIXED_QUIET_DROP_DB = 24.0
MIXED_QUIET_FLOOR_DBFS = -80.0
VOCAL_QUIET_MAX_DBFS = -45.0
VOCAL_QUIET_DROP_DB = 18.0
VOCAL_PAUSE_DROP_DB = 12.0
VOCAL_PAUSE_MIN_SECONDS = 0.24
VOCAL_PAUSE_FLANK_SECONDS = 0.60
VOCAL_PAUSE_FLANK_DROP_DB = 12.0
MIN_EDGE_SECONDS = 0.10
MAX_EDGE_SECONDS = 0.20


def vocal_pause_candidates(starts, ends, levels, references, minimum=VOCAL_PAUSE_MIN_SECONDS):
    """Estimate internal phrase pauses from relative vocal energy, not silence.

    A singing pause can contain reverb or separator leakage, so an absolute
    -45 dBFS silence threshold is inappropriate for all boundary proposals.
    Require a sustained raw-energy dip and stronger vocals on BOTH sides. These are
    reviewable cut proposals; they are not removed source-audio intervals.
    """
    starts, ends, levels, references = [np.asarray(x, dtype=np.float64)
                                       for x in (starts, ends, levels, references)]
    if not len(levels):
        return []
    # Never let smoothing hide a brief consonant/onset inside the pause.
    # Each frame in the full interval must pass the raw-energy condition.
    low = (levels <= references - VOCAL_PAUSE_DROP_DB) & (references >= -70.)
    changes = np.diff(np.r_[False, low, False].astype(np.int8))
    proposals = []
    for i, j in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        a, b = float(starts[i]), float(ends[j - 1])
        if b - a + 1e-9 < minimum:
            continue
        left = levels[(ends <= a + 1e-9) & (ends > a - VOCAL_PAUSE_FLANK_SECONDS)]
        right = levels[(starts >= b - 1e-9) & (starts < b + VOCAL_PAUSE_FLANK_SECONDS)]
        at_start,at_end=i==0,j==len(levels)
        if at_start and at_end:
            continue
        if (not at_start and len(left)<6) or (not at_end and len(right)<6):continue
        quiet = float(np.percentile(levels[i:j], 75))
        flanks=[float(np.percentile(x,75)) for x in (left,right) if len(x)>=6]
        flank=min(flanks)
        if flank < -70. or flank - quiet < VOCAL_PAUSE_FLANK_DROP_DB:
            continue
        center = round((a + b) / 2 * 16000) / 16000
        proposals.append(dict(time=center, pause_start=a, pause_end=b, duration=b-a,
                              vocal_level_dbfs=quiet, flank_level_dbfs=flank,
                              contrast_db=flank-quiet, method='relative_vocal_energy_dip',
                              raw_energy_checked=True, minimum_edge_context_seconds=MIN_EDGE_SECONDS,
                              position='edge_context' if at_start or at_end else 'internal_pause',
                              status='needs_boundary_review'))
    return proposals


def split_at_vocal_pauses(islands, checks):
    """Keep every source sample while exposing vocal pauses to clip grouping.

    A source-mixture VAD island may span a vocal pause because music continues.
    Unioning preservation intervals must not erase a separately supported cut.
    """
    cuts = sorted({round(p['time'] * 16000) for c in checks
                   for p in c.get('boundary_candidates', []) if p.get('position')!='edge_context'})
    result = []
    for a, b in islands:
        first, last = round(a * 16000), round(b * 16000)
        points = [first] + [p for p in cuts if first < p < last] + [last]
        result.extend([[x / 16000, y / 16000] for x, y in zip(points, points[1:])])
    return result


def vocal_cut_evidence(start, end, checks):
    evidence = {}
    for side, time_value in (('start', start), ('end', end)):
        for check in checks:
            for point in check.get('boundary_candidates', []):
                if abs(point['time'] - time_value) <= 1 / 16000:
                    evidence[side] = copy.deepcopy(point)
    return evidence


def vocal_pause_settings(requested, separator):
    return dict(requested=requested, model='HDEMUCS_HIGH_MUSDB_PLUS' if separator else None,
                maximum_quiet_dbfs=VOCAL_QUIET_MAX_DBFS, relative_drop_db=VOCAL_QUIET_DROP_DB,
                analysis_block_seconds=8, context_seconds=1,
                cut_policy='continuous_source_5_15_with_measured_natural_edges',
                minimum_edge_seconds=MIN_EDGE_SECONDS,maximum_edge_seconds=MAX_EDGE_SECONDS,
                boundary_drop_db=VOCAL_PAUSE_DROP_DB, boundary_minimum_seconds=VOCAL_PAUSE_MIN_SECONDS,
                boundary_flank_seconds=VOCAL_PAUSE_FLANK_SECONDS,
                boundary_flank_drop_db=VOCAL_PAUSE_FLANK_DROP_DB)


def mixed_pause_separator(regions, progress):
    """Reuse Part 3's separator only for mixed audio; fail toward preservation."""
    if not any(r['label'] == 'mixed' for r in regions):
        return None
    try:
        from vocal_verification import VocalSeparator
        progress('Preparing vocal-pause checks for mixed audio (same model as Part 3)…')
        return VocalSeparator(progress=progress)
    except Exception as exc:
        progress('Vocal-pause model unavailable; preserving uncertain mixed audio using conservative mixture checks. '
                 'Long spans may require manual cuts. ' + str(exc))
        return None


def merge_intervals(intervals):
    """Union source intervals on the 16 kHz sample grid, without filling gaps."""
    merged = []
    for a, b in sorted((round(a * 16000), round(b * 16000)) for a, b in intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [[a / 16000, b / 16000] for a, b in merged]


def quiet_mixed_intervals(audio, start, end, *, separator=None, progress=lambda text: None,
                          return_boundaries=False, minimum_dip=VOCAL_PAUSE_MIN_SECONDS):
    """Conservative quiet-audio evidence, not a singing/word-boundary detector.

    Music can mask a real vocal pause. When uncertain we retain the recording.
    Read bounded, frame-aligned blocks and keep quiet runs across read boundaries.
    """
    rate = audio.samplerate
    first, last = round(start * rate), round(end * rate)
    frame = round(MIXED_QUIET_FRAME_SECONDS * rate)
    block = frame * int((8 if separator is not None else 30) / MIXED_QUIET_FRAME_SECONDS)
    minimum = round(MIXED_QUIET_MIN_SECONDS * rate)
    quiet, run = [], None
    analysis_starts, analysis_ends, analysis_levels, analysis_refs = [], [], [], []
    cursor = first
    while cursor < last:
        stop = min(last, cursor + block)
        # Context reduces separation artifacts at analysis-block boundaries;
        # only the exact central source samples contribute to quiet decisions.
        context = rate if separator is not None else 0
        read_start, read_end = max(0, cursor - context), min(len(audio), stop + context)
        audio.seek(read_start)
        mixture = audio.read(read_end - read_start, dtype='float32', always_2d=True)[:, 0]
        if len(mixture) != read_end - read_start or not np.isfinite(mixture).all():
            raise ValueError('Invalid or incomplete audio while checking mixed-region gaps.')
        if separator is not None:
            samples = np.asarray(separator(mixture, rate), dtype=np.float32)
            if samples.shape != mixture.shape or not np.isfinite(samples).all():
                raise ValueError('Vocal-pause separator returned invalid or misaligned audio.')
        else:
            samples = mixture
        samples = samples[cursor - read_start:stop - read_start]
        if len(samples) != stop - cursor or not np.isfinite(samples).all():
            raise ValueError('Invalid or incomplete audio while checking mixed-region gaps.')
        starts = np.arange(0, len(samples), frame)
        counts = np.minimum(frame, len(samples) - starts)
        energy = np.add.reduceat(samples.astype(np.float64) ** 2, starts) / counts
        db = 10 * np.log10(np.maximum(energy, 1e-24))
        reference = float(np.percentile(db, 75))
        if separator is not None and return_boundaries:
            analysis_starts.extend((cursor + starts) / rate)
            analysis_ends.extend((cursor + starts + counts) / rate)
            analysis_levels.extend(db)
            analysis_refs.extend([reference] * len(db))
        ceiling = VOCAL_QUIET_MAX_DBFS if separator is not None else MIXED_QUIET_MAX_DBFS
        drop = VOCAL_QUIET_DROP_DB if separator is not None else MIXED_QUIET_DROP_DB
        threshold = max(MIXED_QUIET_FLOOR_DBFS, min(ceiling, reference - drop))
        for offset, count, is_quiet in zip(starts, counts, db <= threshold):
            a, b = cursor + int(offset), cursor + int(offset + count)
            if is_quiet:
                if run is None:
                    run = a
            elif run is not None:
                if a - run >= minimum:
                    quiet.append([run / rate, a / rate])
                run = None
        cursor = stop
        if separator is not None:
            progress(f'Checking vocal pauses: {cursor / rate:.1f} / {end:.1f} seconds')
    if run is not None and last - run >= minimum:
        quiet.append([run / rate, last / rate])
    if return_boundaries:
        return quiet, vocal_pause_candidates(analysis_starts, analysis_ends, analysis_levels, analysis_refs, minimum_dip)
    return quiet


def protect_mixed_continuity(audio, islands, block_start, block_end, regions, progress=lambda text: None,
                             *, separator=None):
    """Retain VAD-missed mixed audio unless sustained quiet on separated vocals (or conservative mixture checks) supports a gap.

    VAD-negative does not mean vocal-free. This does not relabel audio as speech,
    prove the presence of singing, or manufacture pauses. It preserves uncertain
    source audio inside the user's existing Step 3 mixed labels for review.
    Keep the separated evidence for the long-span recovery pass.
    """
    original = merge_intervals(islands)
    additions, checks = [], []
    mixed = merge_intervals((max(block_start, float(r['start'])), min(block_end, float(r['end'])))
                           for r in regions if r['label'] == 'mixed'
                           and r['end'] > block_start and r['start'] < block_end)
    for start, end in mixed:
        progress(f'Protecting mixed/vocal continuity: {start:.1f}–{end:.1f} s…')
        method = 'original_mix_conservative'
        error = None
        boundaries = []
        short_boundaries = []
        if separator is not None:
            try:
                quiet, boundaries = quiet_mixed_intervals(audio, start, end, separator=separator,
                                                          progress=progress, return_boundaries=True,
                                                          minimum_dip=REFINE_PAUSE_MS / 1000)
                short_boundaries = [p for p in boundaries if p['duration'] + 1e-9 < VOCAL_PAUSE_MIN_SECONDS
                                    and p['position'] == 'internal_pause'
                                    and p['contrast_db'] >= VOCAL_QUIET_DROP_DB]
                boundaries = [p for p in boundaries if p['duration'] + 1e-9 >= VOCAL_PAUSE_MIN_SECONDS]
                method = 'separated_vocals_energy'
            except Exception as exc:
                error = str(exc)
                progress('Vocal-pause check failed; preserving uncertain audio using mixture checks: ' + error)
                quiet = quiet_mixed_intervals(audio, start, end)
        else:
            quiet = quiet_mixed_intervals(audio, start, end)
        active, cursor = [], start
        for a, b in quiet:
            if a > cursor:
                active.append([cursor, a])
            cursor = b
        if cursor < end:
            active.append([cursor, end])
        recovered = sum(b - a - sum(max(0., min(b, y) - max(a, x)) for x, y in original)
                        for a, b in active)
        checks.append(dict(start=start, end=end, quiet_intervals=quiet,
                           vad_missing_seconds_preserved=max(0., recovered),
                           quiet_evidence=method, separation_error=error,
                           boundary_candidates=boundaries, short_boundary_candidates=short_boundaries,
                           policy='retain_mixed_unless_sustained_quiet',
                           note='Quiet on a vocal stem is an estimated vocal pause, not a verified word boundary. Retained audio may include accompaniment.'))
        additions.extend(active)
    return split_at_vocal_pauses(merge_intervals(original + additions), checks), checks


def explain_mixed_gaps(gaps, checks):
    """Keep genuine estimated pauses visible, with their actual evidence in the UI."""
    for gap in gaps:
        for check in checks:
            if any(a - 1 / 16000 <= gap['start'] < gap['end'] <= b + 1 / 16000
                   for a, b in check['quiet_intervals']):
                gap['pause_evidence'] = check['quiet_evidence']
                gap['estimated_pause'] = True
                gap['reason'] = ('Estimated vocal pause: separated vocals were quiet; listen to verify the boundary.'
                                 if check['quiet_evidence'] == 'separated_vocals_energy' else
                                 'Estimated quiet-audio pause in a mixed region; listen to verify the boundary.')
                break
    return gaps


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
    left_gap = max(0., speech[i][0] - left_limit)
    right_gap = max(0., right_limit - speech[j][1])
    # Share short pauses rather than overlapping adjacent exported clips.
    left_pad = min(EDGE_SECONDS, left_gap if i == 0 else left_gap / 2)
    right_pad = min(EDGE_SECONDS, right_gap if j + 1 == len(speech) else right_gap / 2)
    start, end = speech[i][0] - left_pad, speech[j][1] + right_pad
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
            left_pad = speech[i][0] - start
            right_pad = end - speech[j][1]
            if min(left_pad, right_pad) < MIN_EDGE_SECONDS - 1e-8:
                continue
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
                               'reason': 'No 5–15 s candidate with 0.1–0.2 s natural pause at BOTH edges; select and review boundaries manually.'})
            i += 1
        else:
            j, start, end = choices[i]
            groups.append({'start': start, 'end': end, 'speech_start': speech[i][0],
                           'speech_end': speech[j][1]})
            i = j + 1
    return groups, unresolved


def group_short_unresolved(unresolved, candidates):
    """Group consecutive short spans within one eligible block, preserving source pauses."""
    timeline = sorted([(item['start'], False, item) for item in unresolved] +
                      [(item['start'], True, item) for item in candidates], key=lambda row: row[0])
    n = len(timeline)
    # Prefer covering more fragments, then use fewer review rows.
    scores = [(0, 0)] * (n + 1)
    ends = list(range(1, n + 1))
    for i in range(n - 1, -1, -1):
        scores[i] = scores[i + 1]
        start = timeline[i][0]
        previous_end = start
        for j in range(i, n):
            left, barrier, item = timeline[j]
            end = item['end']
            if (barrier or end - left >= MIN_SECONDS or left - previous_end > 1.0
                    or end - start > MAX_SECONDS):
                break
            previous_end = end
            if end - start >= MIN_SECONDS:
                count, rows = scores[j + 1]
                score = (count + j - i + 1, rows - 1)
                if score > scores[i]:
                    scores[i], ends[i] = score, j + 1
    result = []
    i = 0
    while i < n:
        j = ends[i]
        if j > i + 1:
            parts = [row[2] for row in timeline[i:j]]
            result.append(dict(start=parts[0]['start'], end=parts[-1]['end'],
                               parts=parts, reason='Consecutive short spans grouped with original audio between them. Listen to verify both outer boundaries.'))
        elif not timeline[i][1]:
            result.append(timeline[i][2])
        i = j
    return result


def source_pause_intervals(audio, islands, start, end, *, minimum=VOCAL_PAUSE_MIN_SECONDS, ceiling=0.):
    """VAD-negative AND low-energy source frames; active VAD misses are retained.

    A nonspeech probability alone is not evidence for discarding a quiet word.
    Read bounded chunks, and require a sustained raw-energy run. These remain
    detector proposals until a person checks complete words at both edges.
    """
    rate, frame = audio.samplerate, round(.02 * audio.samplerate)
    first, last = round(start*rate), round(end*rate)
    cursor, run, pauses = first, None, []
    detected = [(round(a*rate), round(b*rate)) for a,b in merge_intervals(islands)]
    island_index = 0
    while cursor < last:
        stop = min(last, cursor + 30*rate)
        read_start, read_end = max(0,cursor-rate), min(len(audio),stop+rate)
        audio.seek(read_start)
        data = audio.read(read_end-read_start,dtype='float32',always_2d=True)[:,0]
        if len(data)!=read_end-read_start or not np.isfinite(data).all():
            raise ValueError('Invalid audio while checking source pause context.')
        offsets = np.arange(0,len(data),frame)
        counts = np.minimum(frame,len(data)-offsets)
        energy = np.add.reduceat(data.astype(np.float64)**2,offsets)/counts
        reference = 10*math.log10(max(1e-24,float(np.percentile(energy,75))))
        threshold = min(ceiling, max(-80.,reference-12.))
        samples = data[cursor-read_start:stop-read_start]
        offsets = np.arange(0,len(samples),frame)
        counts = np.minimum(frame,len(samples)-offsets)
        levels = 10*np.log10(np.maximum(np.add.reduceat(samples.astype(np.float64)**2,offsets)/counts,1e-24))
        for offset,count,db in zip(offsets,counts,levels):
            a,b=cursor+int(offset),cursor+int(offset+count)
            while island_index<len(detected) and detected[island_index][1]<=a:
                island_index+=1
            vad_clear = island_index==len(detected) or detected[island_index][0]>=b
            quiet = vad_clear and db<=threshold
            if quiet:
                if run is None:run=a
            elif run is not None:
                if a-run>=round(minimum*rate):pauses.append([run/rate,a/rate])
                run=None
        cursor=stop
    if run is not None and last-run>=round(minimum*rate):
        pauses.append([run/rate,last/rate])
    return pauses


def subtract_intervals(intervals, excluded):
    result=[]
    for a,b in intervals:
        pieces=[[a,b]]
        for x,y in excluded:
            updated=[]
            for p,q in pieces:
                if y<=p or x>=q:updated.append([p,q]);continue
                if p<x:updated.append([p,x])
                if y<q:updated.append([y,q])
            pieces=updated
        result.extend(pieces)
    return result


def pdf_pause_plan(audio, vad_islands, start, end, regions, checks):
    """Separate estimated vocal pauses from audio coverage; never synthesize audio."""
    pauses=source_pause_intervals(audio,vad_islands,start,end)
    # Mixture VAD can be active throughout background music: use the vocal
    # track's supported pauses within each successfully separated mixed span.
    for check in checks:
        if check['quiet_evidence']!='separated_vocals_energy':continue
        pauses=subtract_intervals(pauses,[[check['start'],check['end']]])
        pauses.extend(check['quiet_intervals'])
        pauses.extend([[p['pause_start'],p['pause_end']] for p in check.get('boundary_candidates',[])])
    pauses=merge_intervals((max(start,a),min(end,b)) for a,b in pauses if b>start and a<end)
    pauses=[p for p in pauses if p[1]-p[0]>=2*MIN_EDGE_SECONDS-1e-8]
    active=subtract_intervals([[start,end]],pauses)
    return active,pauses


def adjacent_silence_limits(audio, vad, start, end, left_limit, right_limit):
    """Borrow only quiet, VAD-clear samples contiguous with the eligible region."""
    left, right = start, end
    if left_limit < start:
        for a, b in source_pause_intervals(audio, vad, left_limit, start, minimum=1/16000,
                                           ceiling=VOCAL_QUIET_MAX_DBFS):
            if b == start:
                left = a
    if end < right_limit:
        for a, b in source_pause_intervals(audio, vad, end, right_limit, minimum=1/16000,
                                           ceiling=VOCAL_QUIET_MAX_DBFS):
            if a == end:
                right = b
    return left, right


def lecture_noise_pauses(audio, vad):
    """Recover sustained VAD gaps with steady broadband noise below both speech flanks."""
    rate = audio.samplerate
    frame = round(.020 * rate)
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1 / rate)
    band = (freqs >= 150) & (freqs <= 4500)
    for (prev_start, lo), (hi, next_end) in zip(vad, vad[1:]):
        if not .32 <= hi - lo <= 3 or min(lo-prev_start, next_end-hi) < .2:
            continue
        first, last = round(lo * rate), round(hi * rate)
        flank = round(.2 * rate)
        audio.seek(first - flank)
        samples = audio.read(last-first+2*flank, dtype='float32', always_2d=True)[:, 0]
        gap = samples[flank:flank+last-first]
        # Check every complete frame, including a frame ending at the far edge.
        frames = np.lib.stride_tricks.sliding_window_view(gap, frame)
        frames = np.vstack((frames[::frame], frames[-1]))
        levels = 10*np.log10(np.mean(frames**2, axis=1) + 1e-12)
        flank_levels = [10*np.log10(np.median(np.mean(side.reshape(-1,frame)**2,axis=1)) + 1e-12)
                        for side in (samples[:flank],samples[-flank:])]
        if levels.max()-levels.min() > 5 or levels.max() > min(flank_levels)-6:
            continue
        power = abs(np.fft.rfft(frames*window, axis=1)[:,band])**2 + 1e-16
        flatness = np.exp(np.mean(np.log(power),axis=1)) / np.mean(power,axis=1)
        if np.min(flatness) >= .30:
            # Guard both transitions before the usual natural-padding checks.
            yield (first+2*frame)/rate, (last-2*frame)/rate


def recover_long_spans(audio, active, groups, unresolved, left, right, checks, model, progress):
    """Recheck only long unresolved spans. Reserve neighboring clips' existing padding."""
    recovered, remaining, used_pauses = [], [], []
    positions = {round(a*16000): i for i, (a, _) in enumerate(active)}
    for item in unresolved:
        a, b = item['start'], item['end']
        if b - a <= MAX_SECONDS:
            remaining.append(item)
            continue
        progress(f'Rechecking {a:.2f}–{b:.2f} s for supported pauses of at least {REFINE_PAUSE_MS} ms…')
        # Only separated-vocal evidence may introduce a new boundary in mixed audio.
        speech_ranges = subtract_intervals([[a,b]], [(c['start'], c['end']) for c in checks])
        proposals = []
        for start, end in speech_ranges:
            if end - start < REFINE_PAUSE_MS / 1000:
                continue
            vad = detect_speech(audio, start, end, model, progress, pause_ms=REFINE_PAUSE_MS)
            quiet = source_pause_intervals(audio, vad, start, end, minimum=REFINE_PAUSE_MS/1000)
            proposals.extend(lecture_noise_pauses(audio, vad))
            for x, y in quiet:
                # Require activity on both sides to reject fades, clipped edges and VAD dropouts.
                if (any(0 <= x-v <= VOCAL_PAUSE_FLANK_SECONDS for _,v in vad)
                        and any(0 <= u-y <= VOCAL_PAUSE_FLANK_SECONDS for u,_ in vad)):
                    proposals.append((x,y))
        for check in checks:
            proposals.extend((p['pause_start'],p['pause_end']) for p in check['short_boundary_candidates']
                             if a < p['pause_start'] < p['pause_end'] < b)
        proposals = merge_intervals((x,y) for x,y in proposals if a < x < y < b)
        if proposals:
            i = positions[round(a*16000)]
            low = left if i == 0 else (active[i-1][1]+a)/2
            high = right if i+1 == len(active) else (b+active[i+1][0])/2
            clips, leftovers = group_at_pauses(subtract_intervals([[a,b]], proposals), low, high)
            if clips:
                for clip in clips:
                    clip['recovered'] = True
                recovered.extend(clips)
                remaining.extend(leftovers)
                used_pauses.extend(proposals)
                continue
        remaining.append(item)
    return sorted(groups + recovered, key=lambda clip: clip['start']), remaining, used_pauses


def edge_pause_evidence(segment, pauses):
    evidence={}
    for side,edge,voice in [('start',segment['start'],segment['speech_start']),
                            ('end',segment['end'],segment['speech_end'])]:
        for a,b in pauses:
            if a-1e-8<=min(edge,voice) and max(edge,voice)<=b+1e-8:
                evidence[side]=dict(pause_start=a,pause_end=b,cut=edge,
                    retained_pause_seconds=abs(edge-voice),
                    status='estimated_pause_needs_human_word_boundary_review')
                break
    return evidence


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
<section class="notice"><b>Mixed/vocal continuity:</b> VAD-negative intervals in mixed regions are retained unless sustained quiet on the separated vocals (or conservative original-mixture checks) supports a gap. Sustained raw vocal-level drops provide pause proposals even when music keeps mixture VAD active. A short syllable dip is not a valid boundary. In mixed audio, background music can continue through a vocal pause. Remaining long spans stay unresolved when no supported 5–15 s grouping is possible. This is audio-preservation logic, not proof of vocals or a verified word boundary.</section>
<section><h2>Overview</h2><p><b>Candidate clips:</b> {len(data['segments'])} &nbsp; <b>Unresolved spans:</b> {len(data['unresolved'])} &nbsp; <b>Eligible blocks:</b> {data['eligible_block_count']}</p><p>Settings: 5–15 s; target edge padding 0.15 s with at least 0.1 s on both edges; Silero VAD first pass {MIN_PAUSE_MS} ms; both edges require 0.1–0.2 s detector-estimated natural pause; raw 240 ms vocal pauses may supplement speech VAD. Adjacent labelled silence can supply measured edge context. Long unresolved spans get a targeted {REFINE_PAUSE_MS} ms pause check. No artificial silence or joins are generated.</p></section>
<section><h2>Time accounting</h2><p>Speech/mixed according to Step 3: {eligible:.2f} s. Total candidate duration: {clip_time:.2f} s, including {coverage.get("adjacent_silence_in_clips",0.):.2f} s borrowed from adjacent Step 3 silence. Unresolved spans: {unresolved_time:.2f} s. Other eligible audio outside candidates and unresolved spans: {unchecked_time:.2f} s. Total labelled music/silence in Step 3: {excluded:.2f} s.</p><p>Unchecked audio can include VAD-missed speech; the music/silence labels can also be wrong. Review them against the original audio. These categories are time-accounting aids, not a measurement of transcription completeness.</p></section>
<section><h2>Candidate clips</h2><div class="scroll"><table><tr><th>File</th><th>Start (s)</th><th>End (s)</th><th>Duration (s)</th><th>Label</th><th>Left padding (s)</th><th>Right padding (s)</th><th>Review notes</th><th>Listen</th></tr>{''.join(rows)}</table></div></section>
<section><h2>Unresolved spans</h2><p>Each span is also saved intact with a little context in needs_manual_review. These WAVs can be shorter than 5 seconds or longer than 15 seconds and are not accepted corpus clips. Inspect and select valid pause boundaries manually.</p><div class="scroll"><table><tr><th>Start (s)</th><th>End (s)</th><th>Reason</th><th>Review audio</th></tr>{''.join(misses)}</table></div></section>
<section><h2>Eligible audio outside clips and unresolved spans</h2><p>{len(unchecked)} intervals, total {unchecked_time:.2f} s. The original WAV remains unchanged. These detector-estimated pauses can still contain quiet words; listen before marking them reviewed.</p><details><summary>Show interval times</summary><div class="scroll"><table><tr><th>Start (s)</th><th>End (s)</th><th>Duration (s)</th></tr>{unchecked_rows}</table></div></details></section>
<section>Machine-readable results and the Step 3 snapshot are in step4_segments.json in this folder. Times refer to the original WAV.</section></main></body></html>'''


def export_candidates(audio_path, regions, output_parent, progress=lambda text: None, *, use_vocal_pauses=True):
    """Propose continuous 5–15 s source clips with measured natural-pause edges.

    PDF Step 4 governs generation. No synthetic silence, stitching, forced
    word cuts or automatic approval. Every estimated boundary needs listening.
    """
    from silero_vad import load_silero_vad
    import torch
    path=Path(audio_path)
    with sf.SoundFile(path) as audio:
        if (audio.samplerate,audio.channels,audio.subtype)!=(16000,1,'PCM_16'):
            raise ValueError('Step 4 requires 16 kHz, mono, 16-bit PCM WAV.')
        duration=len(audio)/16000
        previous=0.
        for r in regions:
            a,b=float(r['start']),float(r['end'])
            if r['label'] not in ('speech','music','mixed','silence') or abs(a-previous)>1e-5 or not a<b<=duration:
                raise ValueError('Step 3 regions must form a continuous source timeline.')
            previous=b
        if not regions or abs(previous-duration)>1/16000:
            raise ValueError('Step 3 regions must cover the selected WAV.')
        blocks=eligible_blocks(regions)
        if not blocks:raise ValueError('No speech/mixed regions. Review Step 3 first.')
        torch.set_num_threads(1)
        progress('Loading Silero VAD for PDF pause checks…')
        model=load_silero_vad()
        separator=mixed_pause_separator(regions,progress) if use_vocal_pauses else None
        output=new_run_folder(output_parent)
        data=dict(version=3,audio_name=path.name,audio_size=path.stat().st_size,duration_seconds=duration,
            created_utc=datetime.now(timezone.utc).isoformat(),
            method='PDF Step 4: continuous source clips, sustained pause evidence, 0.1–0.2 s natural edge context; human review required',
            min_duration=MIN_SECONDS,max_duration=MAX_SECONDS,target_edge_seconds=EDGE_SECONDS,
            eligible_block_count=len(blocks),step3_regions=copy.deepcopy(regions),
            pdf_step4=dict(minimum_edge_seconds=MIN_EDGE_SECONDS,maximum_edge_seconds=MAX_EDGE_SECONDS,
                long_span_minimum_pause_ms=REFINE_PAUSE_MS,adjacent_silence_limit_seconds=EDGE_SECONDS,
                synthetic_padding=False,noncontiguous_joins=False,forced_cuts=False,
                word_boundary_validation='human listening required; energy/VAD do not prove complete words'),
            vocal_pause_settings=vocal_pause_settings(use_vocal_pauses,separator),
            mixed_continuity_settings=dict(frame_seconds=MIXED_QUIET_FRAME_SECONDS,
                minimum_quiet_seconds=MIXED_QUIET_MIN_SECONDS,maximum_quiet_dbfs=MIXED_QUIET_MAX_DBFS,
                relative_drop_db=MIXED_QUIET_DROP_DB,threshold_floor_dbfs=MIXED_QUIET_FLOOR_DBFS),
            mixed_continuity_checks=[],pause_intervals=[],segments=[],unresolved=[])
        silence = merge_intervals((r['start'],r['end']) for r in regions if r['label']=='silence')
        silence_before = {b:a for a,b in silence}
        silence_after = dict(silence)
        borrowed = 0.0
        for number,(start,end) in enumerate(blocks,1):
            progress(f'Checking real-pause candidates in speech/mixed block {number}/{len(blocks)}…')
            start, end = round(start*16000)/16000, round(end*16000)/16000
            # Divide any silence shared by two eligible blocks so clips cannot overlap.
            before = silence_before.get(start,start)
            after = silence_after.get(end,end)
            if number > 1 and abs(blocks[number-2][1]-before) < 1/16000:
                before = round((before+start)*8000)/16000
            if number < len(blocks) and abs(blocks[number][0]-after) < 1/16000:
                after = round((after+end)*8000)/16000
            lo, hi = max(before,start-EDGE_SECONDS), min(after,end+EDGE_SECONDS)
            vad=detect_speech(audio,lo,hi,model,progress)
            _,checks=protect_mixed_continuity(audio,vad,start,end,regions,progress,separator=separator)
            data['mixed_continuity_checks'].extend(checks)
            active,pauses=pdf_pause_plan(audio,vad,start,end,regions,checks)
            left,right=adjacent_silence_limits(audio,vad,start,end,lo,hi)
            # The borrowed samples can only supply padding. They never become active audio.
            pauses=merge_intervals(pauses + [(left,start),(end,right)])
            groups,unresolved=group_at_pauses(active,left,right)
            groups,unresolved,refined=recover_long_spans(
                audio,active,groups,unresolved,left,right,checks,model,progress)
            pauses=merge_intervals(pauses+refined)
            data['pause_intervals'].extend(dict(start=a,end=b,status='estimated_pause_review_required') for a,b in pauses)
            data['unresolved'].extend(group_short_unresolved(unresolved, groups))
            for segment in groups:
                a,b=round(segment['start']*16000),round(segment['end']*16000)
                segment['start'],segment['end']=a/16000,b/16000
                left=segment['speech_start']-a/16000
                right=b/16000-segment['speech_end']
                if not 5*16000<=b-a<=15*16000 or not all(MIN_EDGE_SECONDS-1/16000<=v<=MAX_EDGE_SECONDS+1/16000 for v in (left,right)):
                    raise RuntimeError('PDF duration/natural-edge validation failed; no candidate was accepted.')
                evidence=edge_pause_evidence(segment,pauses)
                if set(evidence)!={'start','end'}:raise RuntimeError('Missing natural-pause edge evidence.')
                overlap=[r for r in regions if r['label'] in ('speech','mixed') and r['end']>a/16000 and r['start']<b/16000]
                borrowed += max(0., start-a/16000) + max(0., b/16000-end)
                label='mixed' if any(r['label']=='mixed' for r in overlap) else 'speech'
                vocal=any(c['quiet_evidence']=='separated_vocals_energy' and c['end']>a/16000 and c['start']<b/16000 for c in checks)
                filename=f"segment_{len(data['segments'])+1:05d}.wav"
                audio.seek(a);samples=audio.read(b-a,dtype='float32',always_2d=True)
                if len(samples)!=b-a:raise RuntimeError('Incomplete source audio while exporting candidate.')
                sf.write(output/filename,samples,16000,subtype='PCM_16')
                data['segments'].append(dict(file=filename,start=a/16000,end=b/16000,duration=(b-a)/16000,
                    label=label,kind='vocal_pause_candidate' if vocal else 'vad_candidate',
                    left_padding=left,right_padding=right,boundary_evidence=evidence,
                    source_spans=[dict(start=a/16000,end=b/16000,clip_start=0.,clip_end=(b-a)/16000)],
                    removed_gaps=[],synthetic_silence_start_seconds=0.,synthetic_silence_end_seconds=0.,
                    status='needs_review',notes=(['Recovered during the shorter-pause review pass.'] if segment.get('recovered') else []) + ['Continuous original source audio; both edges have 0.1–0.2 s detector-estimated natural pause.',
                        'Listen before approval: check full words, especially quiet consonants and sung onsets. No energy detector guarantees word boundaries.']))
        for index,item in enumerate(data['unresolved'],1):
            item['mixed_continuity_protected']=any(r['label']=='mixed' and r['end']>item['start'] and r['start']<item['end'] for r in regions)
            write_unresolved_audio(audio,item,output,index)
        data['not_in_clips_or_unresolved']=missing_intervals(blocks,data['segments'],data['unresolved'])
        for gap in data['not_in_clips_or_unresolved']:
            gap['estimated_pause']=True
            gap['reason']='Estimated pause outside clips; retained in original WAV. Listen to verify no quiet word was missed.'
        data['coverage_seconds']=dict(adjacent_silence_in_clips=borrowed,eligible_step3=sum(b-a for a,b in blocks),
            unresolved_detected_or_no_vad=sum(x['end']-x['start'] for x in data['unresolved']),
            not_in_clips_or_unresolved=sum(x['end']-x['start'] for x in data['not_in_clips_or_unresolved']),
            excluded_step3_music_or_silence=sum(r['end']-r['start'] for r in regions if r['label'] in ('music','silence')))
        (output/'step4_segments.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'step4_report.html').write_text(render_html(data),encoding='utf-8')
    return dict(output_dir=str(output),segments=len(data['segments']),unresolved=len(data['unresolved']))

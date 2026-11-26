"""Cut speech and mixed audio at pauses in the original recording.

detect_speech and the vocal checks find possible breaks. group_at_pauses builds 5–15
second candidates. Short leftovers can join the following or preceding candidate if
their new outer edge still has enough quiet audio. recover_long_spans checks long
leftovers again, and export_candidates saves the clips and remaining intervals for
listening review.
"""

from output_folders import new_run_folder
from bisect import bisect_right
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
# Permit short natural breaks across eligible blocks. The full clip still
# includes every intervening sample and may not exceed MAX_SECONDS.
GLOBAL_MERGE_MAX_GAP_SECONDS = 2.0


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


# Find the vocal-pause measurement that supports each proposed cut.
def vocal_cut_evidence(start, end, checks):
    evidence = {}
    for side, time_value in (('start', start), ('end', end)):
        for check in checks:
            for point in check.get('boundary_candidates', []):
                if abs(point['time'] - time_value) <= 1 / 16000:
                    evidence[side] = copy.deepcopy(point)
    return evidence


# Keep the separator and pause settings alongside the candidates they produced.
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


# Leave quiet audio at each end and share a short pause with the neighboring clip.
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


def prepend_short_unresolved(audio, groups, unresolved, pauses, block_left, vad, mixed_checks):
    """Try adding each short leftover to the beginning of the clip that follows it."""
    return extend_candidates_with_short_spans(audio, groups, unresolved, pauses,
                                             block_left, vad, mixed_checks, prepend=True)


def extend_candidates_with_short_spans(audio, groups, unresolved, pauses, block_limit, vad,
                                       mixed_checks, *, prepend):
    """Try moving one end of a candidate to include the short span beside it.

    For example, a 2-second phrase can join a 10-second clip if the original
    gap and the silence at the edges still fit within 15 seconds. The far end
    of the clip stays where it was. Only the end we move needs a new pause check."""
    if not groups or not unresolved:
        return groups, unresolved, pauses
    timeline = sorted([(g['start'], True, g) for g in groups] +
                      [(r['start'], False, r) for r in unresolved], key=lambda row: row[0])
    pause_starts = [a for a, _ in pauses]
    vad_ends = [b for _, b in vad]
    segments, consumed, extra_pauses = [], set(), []
    rate = audio.samplerate
    for i, (_, candidate, item) in enumerate(timeline):
        if not candidate:
            continue
        segments.append(item)
        neighbor = i - 1 if prepend else i + 1
        if not 0 <= neighbor < len(timeline) or timeline[neighbor][1]:
            continue
        fragment = timeline[neighbor][2]
        first, last = fragment['start'], fragment['end']
        if last - first >= MIN_SECONDS:
            continue

        # The exported WAV includes everything between the two pieces. Count that
        # time toward 15 seconds and stop before we reach another clip.
        if prepend:
            if last > item['start']:
                continue
            limit = max(block_limit, item['end'] - MAX_SECONDS)
            if neighbor > 0:
                limit = max(limit, timeline[neighbor - 1][2]['end'])
            boundary = first
            if round(limit * rate) > round(first * rate):
                continue
            lo, hi = max(limit, first - MAX_EDGE_SECONDS), min(last, first + MAX_EDGE_SECONDS + 0.04)
        else:
            if first < item['end']:
                continue
            limit = min(block_limit, item['start'] + MAX_SECONDS)
            if neighbor + 1 < len(timeline):
                limit = min(limit, timeline[neighbor + 1][2]['start'])
            boundary = last
            if round(limit * rate) < round(last * rate):
                continue
            lo, hi = max(first, last - MAX_EDGE_SECONDS - 0.04), min(limit, last + MAX_EDGE_SECONDS)

        index = bisect_right(pause_starts, boundary) - 1
        pause = pauses[index] if index >= 0 and pauses[index][1] >= boundary else None
        if pause is None:
            # A short pause can leave enough silence for one clip even if it was
            # too short for two clips to share. Recheck that part of the WAV.
            # For a song, a dip in the backing track can happen mid-word, so
            # use only the pauses already found during mixed-audio analysis.
            if any(c['start'] < hi and c['end'] > lo for c in mixed_checks):
                continue
            k = bisect_right(vad_ends, lo)
            local_vad = []
            while k < len(vad) and vad[k][0] < hi:
                local_vad.append(vad[k])
                k += 1
            measured = source_pause_intervals(audio, local_vad, lo, hi,
                                             minimum=MIN_EDGE_SECONDS, ceiling=VOCAL_QUIET_MAX_DBFS)
            # If the quiet run ends at the search limit on the speech side, we
            # haven't observed where the voice begins or ends.
            pause = next(((a, b) for a, b in measured if a <= boundary <= b
                          and (b < hi if prepend else a > lo)), None)
            if pause is None:
                continue

        if prepend:
            voice = pause[1]
            edge = round(max(limit, pause[0], min(first, voice - EDGE_SECONDS)) * rate)
            if voice >= last or edge > round(first * rate):
                continue
            padding, frames = round(voice * rate) - edge, round(item['end'] * rate) - edge
            changes = dict(start=edge / rate, speech_start=voice,
                           prepended_unresolved=dict(start=first, end=last))
        else:
            voice = pause[0]
            edge = round(min(limit, pause[1], max(last, voice + EDGE_SECONDS)) * rate)
            if voice <= first or edge < round(last * rate):
                continue
            padding, frames = edge - round(voice * rate), edge - round(item['start'] * rate)
            changes = dict(end=edge / rate, speech_end=voice,
                           appended_unresolved=dict(start=first, end=last))
        if (round(MIN_EDGE_SECONDS * rate) <= padding <= round(MAX_EDGE_SECONDS * rate)
                and round(MIN_SECONDS * rate) <= frames <= round(MAX_SECONDS * rate)):
            segments[-1] = dict(item, **changes)
            consumed.add(neighbor)
            extra_pauses.append(pause)
    remaining = [item for i, (_, candidate, item) in enumerate(timeline)
                 if not candidate and i not in consumed]
    return segments, remaining, merge_intervals(pauses + extra_pauses)


def regroup_supported_pauses(groups, unresolved, pauses, start, end, left, right):
    """Replan an eligible block using measured pauses, never arbitrary cut points.

    Unlike the initial fixed-padding plan, try the full allowed padding range.
    Preserve all active samples already covered by candidates; only adopt a plan
    that covers additional active samples. Reconsider several short phrases and
    neighboring candidates together. Unchanged clips retain their metadata.
    """
    if not unresolved:
        return groups, unresolved
    rate = 16000
    minimum, maximum = round(MIN_SECONDS * rate), round(MAX_SECONDS * rate)
    min_pad, max_pad, target = (round(x * rate) for x in
                                (MIN_EDGE_SECONDS, MAX_EDGE_SECONDS, EDGE_SECONDS))
    active = [(round(a*rate), round(b*rate)) for a,b in
              subtract_intervals([[start,end]], merge_intervals(pauses))
              if round(b*rate)>round(a*rate)]
    n = len(active)
    if not n:
        return groups, unresolved
    old = sorted((round(g['start']*rate), round(g['end']*rate)) for g in groups)
    protected = []
    k = 0
    for a,b in active:
        while k < len(old) and old[k][1] <= a:
            k += 1
        j, covered = k, 0
        while j < len(old) and old[j][0] < b:
            covered += max(0, min(b,old[j][1])-max(a,old[j][0]))
            j += 1
        protected.append(covered)
    # Lexicographic objective: preserve previously covered audio, maximize new
    # active samples, then prefer padding close to the original 150 ms target.
    scores = [(0,0,0,0)]*(n+1)
    choices = [None]*n
    lo,hi=round(left*rate),round(right*rate)
    for i in range(n-1,-1,-1):
        best = scores[i+1]
        first = active[i][0]
        left_gap = first - (active[i-1][1] if i else lo)
        left_cap = min(max_pad, left_gap//2 if i else left_gap)
        if left_cap < min_pad:
            scores[i] = best
            continue
        total = keep = 0
        for j in range(i,n):
            a,b=active[j]
            total += b-a
            keep += protected[j]
            extent=b-first
            if extent+2*min_pad>maximum:
                break
            right_gap=(active[j+1][0] if j+1<n else hi)-b
            right_cap=min(max_pad, right_gap//2 if j+1<n else right_gap)
            if right_cap<min_pad or extent+left_cap+right_cap<minimum:
                continue
            lp,rp=min(target,left_cap),min(target,right_cap)
            # Grow or shrink only the quiet context, never speech, to meet 5–15 s.
            deficit=max(0,minimum-extent-lp-rp)
            delta=min(deficit,left_cap-lp);lp+=delta;rp+=deficit-delta
            excess=max(0,extent+lp+rp-maximum)
            delta=min(excess,lp-min_pad);lp-=delta;rp-=excess-delta
            if not (min_pad<=lp<=left_cap and min_pad<=rp<=right_cap):
                continue
            x,y=first-lp,b+rp
            tail=scores[j+1]
            score=(keep+tail[0],total+tail[1],tail[2]-1,
                   tail[3]-abs(lp-target)-abs(rp-target))
            if score>best:
                best,choices[i]=score,(j,x,y)
        scores[i]=best
    previous=sum(protected)
    if scores[0][0]!=previous or scores[0][1]<=previous:
        return groups, unresolved
    old_by_bounds={(round(g['start']*rate),round(g['end']*rate)):g for g in groups}
    proposed,remaining=[],[]
    i=0
    while i<n:
        if choices[i] is None:
            a,b=active[i]
            remaining.append(dict(start=a/rate,end=b/rate,
                reason='No 5–15 s candidate with measured natural pauses at both edges after regrouping; review manually.'))
            i+=1
            continue
        j,a,b=choices[i]
        clip=old_by_bounds.get((a,b))
        if clip is None:
            clip=dict(start=a/rate,end=b/rate,speech_start=active[i][0]/rate,
                      speech_end=active[j][1]/rate,regrouped=True)
        if set(edge_pause_evidence(clip,pauses))!={'start','end'}:
            return groups,unresolved
        proposed.append(clip)
        i=j+1
    return proposed,remaining


def group_short_unresolved(unresolved, candidates):
    """Group consecutive short spans within one eligible block, preserving source pauses."""
    timeline = sorted([(item['start'], False, item) for item in unresolved] +
                      [(item['start'], True, item) for item in candidates], key=lambda row: row[0])
    n = len(timeline)
    # Group as many short pieces as possible. If two plans cover the same number,
    # pick the one that leaves the reviewer fewer rows to listen to.
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



def merge_short_spans_across_blocks(unresolved, candidates, pauses, regions, duration):
    """Promote nearby short leftovers using source time, not Step 3 block IDs.

    Internal gaps up to GLOBAL_MERGE_MAX_GAP_SECONDS remain in the original WAV. A music
    label or existing candidate is a hard barrier. Only measured outer pause
    evidence can supply padding. Work on the sample grid and optimize covered
    unresolved audio, then the number of absorbed rows. No forced boundaries.
    """
    rate = 16000
    minimum, maximum = round(MIN_SECONDS*rate), round(MAX_SECONDS*rate)
    min_pad, max_pad, target = (round(x*rate) for x in
                                (MIN_EDGE_SECONDS, MAX_EDGE_SECONDS, EDGE_SECONDS))
    rows = sorted(unresolved, key=lambda u: (u['start'], u['end']))
    ranges = [(round(u['start']*rate),round(u['end']*rate)) for u in rows]
    n = len(rows)
    if not n:
        return [], rows
    evidence = [(round(a*rate),round(b*rate)) for a,b in merge_intervals(pauses)]
    ev_starts = [a for a,_ in evidence]
    barriers = [(round(a*rate),round(b*rate)) for a,b in merge_intervals(
        [(g['start'],g['end']) for g in candidates] +
        [(r['start'],r['end']) for r in regions if r['label']=='music'])]
    barrier_ends = [b for _,b in barriers]
    total_samples = round(duration*rate)

    def available_edge(t, side):
        # The pause must touch the exact active edge; a nearby pause is not enough.
        k = bisect_right(ev_starts, t if side=='right' else t-1)-1
        if k < 0:
            return 0
        a,b = evidence[k]
        if not a <= t <= b:
            return 0
        return max(0,min(max_pad, t-a if side=='left' else b-t))

    def crosses_barrier(a,b):
        k=bisect_right(barrier_ends,a)
        return k<len(barriers) and barriers[k][0]<b

    scores = [(0,0,0,0)]*(n+1)
    choices = [None]*n
    for i in range(n-1,-1,-1):
        best=scores[i+1]
        a,b=ranges[i]
        if not 0<b-a<minimum:
            scores[i]=best
            continue
        before=ranges[i-1][1] if i else 0
        lp_cap=min(available_edge(a,'left'), (a-before)//2 if i else a)
        k=bisect_right(barrier_ends,a)
        if k:
            lp_cap=min(lp_cap,a-barriers[k-1][1])
        if lp_cap<min_pad:
            scores[i]=best
            continue
        covered=0
        for j in range(i,n):
            x,y=ranges[j]
            if not 0<y-x<minimum or (j>i and not 0<=x-ranges[j-1][1]<=round(GLOBAL_MERGE_MAX_GAP_SECONDS*rate)):
                break
            if y-a+2*min_pad>maximum or crosses_barrier(a,y):
                break
            covered+=y-x
            after=ranges[j+1][0] if j+1<n else total_samples
            rp_cap=min(available_edge(y,'right'), (after-y)//2 if j+1<n else after-y)
            k=bisect_right(barrier_ends,y)
            if k<len(barriers):
                rp_cap=min(rp_cap,barriers[k][0]-y)
            extent=y-a
            if rp_cap<min_pad or extent+lp_cap+rp_cap<minimum:
                continue
            lp,rp=min(target,lp_cap),min(target,rp_cap)
            deficit=max(0,minimum-extent-lp-rp)
            delta=min(deficit,lp_cap-lp);lp+=delta;rp+=deficit-delta
            excess=max(0,extent+lp+rp-maximum)
            delta=min(excess,lp-min_pad);lp-=delta;rp-=excess-delta
            if not(min_pad<=lp<=lp_cap and min_pad<=rp<=rp_cap):
                continue
            if crosses_barrier(a-lp,y+rp):
                continue
            tail=scores[j+1]
            score=(covered+tail[0],j-i+1+tail[1],tail[2]-1,
                   tail[3]-abs(lp-target)-abs(rp-target))
            if score>best:
                best,choices[i]=score,(j,a-lp,y+rp)
        scores[i]=best
    merged,remaining=[],[]
    i=0
    while i<n:
        if choices[i] is None:
            remaining.append(rows[i]);i+=1;continue
        j,a,b=choices[i]
        merged.append(dict(start=a/rate,end=b/rate,speech_start=ranges[i][0]/rate,
                           speech_end=ranges[j][1]/rate,
                           merged_unresolved_parts=[dict(start=r['start'],end=r['end']) for r in rows[i:j+1]]))
        i=j+1
    return merged,remaining


def absorb_short_neighbors(unresolved, candidates, pauses, regions, duration):
    """Globally assign short unresolved spans to either adjacent candidate.

    Each option retains one complete existing candidate, any adjacent short
    fragments, and all intervening original samples. Optimize total recovered
    unresolved samples first, then count; never absorb the same row twice.
    """
    rate = 16000
    minimum, maximum = round(MIN_SECONDS*rate), round(MAX_SECONDS*rate)
    min_pad, max_pad, target = (round(v*rate) for v in
                                (MIN_EDGE_SECONDS, MAX_EDGE_SECONDS, EDGE_SECONDS))
    timeline = sorted([(round(c['start']*rate), round(c['end']*rate), True, c)
                       for c in candidates] +
                      [(round(u['start']*rate), round(u['end']*rate), False, u)
                       for u in unresolved], key=lambda r: (r[0],r[1]))
    n = len(timeline)
    if not n or not candidates or not unresolved:
        return [], list(unresolved)
    # Unexpected overlaps are not safe to repair by merging.
    if any(a[1]>b[0] for a,b in zip(timeline,timeline[1:])):
        return [], list(unresolved)
    evidence = [(round(a*rate),round(b*rate)) for a,b in merge_intervals(pauses)]
    ev_starts = [a for a,_ in evidence]
    barriers = [(round(a*rate),round(b*rate)) for a,b in merge_intervals(
        (r['start'],r['end']) for r in regions if r['label']=='music')]
    barrier_ends = [b for _,b in barriers]
    total = round(duration*rate)

    def crosses_music(a,b):
        k=bisect_right(barrier_ends,a)
        return k<len(barriers) and barriers[k][0]<b

    def cap(t, left, index):
        k=bisect_right(ev_starts,t-1 if left else t)-1
        if k<0 or not evidence[k][0]<=t<=evidence[k][1]:
            return 0
        a,b=evidence[k]
        limit = (t-a if left else b-t)
        # Reserve half the free gap if another unresolved row could need it.
        j=index-1 if left else index+1
        if 0<=j<n:
            neighbor=timeline[j]
            gap=t-neighbor[1] if left else neighbor[0]-t
            limit=min(limit,gap if neighbor[2] else gap//2)
        else:
            limit=min(limit,t if left else total-t)
        k=bisect_right(barrier_ends,t)
        if left and k:
            limit=min(limit,t-barriers[k-1][1])
        if not left and k<len(barriers):
            limit=min(limit,barriers[k][0]-t)
        return max(0,min(max_pad,limit))

    # Every range contains at most one existing candidate. Skipping a row
    # retains it unchanged. Disjoint DP choices cannot claim a row twice.
    scores=[(0,0,0,0)]*(n+1)
    choices=[None]*n
    for i in range(n-1,-1,-1):
        scores[i]=scores[i+1]
        a=timeline[i][0]
        base=None
        covered=count=0
        for j in range(i,n):
            x,y,is_candidate,item=timeline[j]
            if (y-a>maximum or (j>i and x-timeline[j-1][1]>round(GLOBAL_MERGE_MAX_GAP_SECONDS*rate))
                    or crosses_music(a,y)):
                break
            if is_candidate:
                if base is not None: break
                base=item
                base_index=j
            else:
                if not 0<y-x<minimum: break
                covered+=y-x;count+=1
            if base is None or not count:
                continue
            # The unchanged far edge keeps the original validated padding.
            old_a,old_b=timeline[base_index][:2]
            old_lp=round(base['left_padding']*rate)
            old_rp=round(base['right_padding']*rate)
            if not(min_pad<=old_lp<=max_pad and min_pad<=old_rp<=max_pad):
                continue
            left_new=i!=base_index;right_new=j!=base_index
            voice_a=a if left_new else old_a+old_lp
            voice_b=y if right_new else old_b-old_rp
            lp_cap=cap(a,True,i) if left_new else old_lp
            rp_cap=cap(y,False,j) if right_new else old_rp
            lp_min=min_pad if left_new else old_lp
            rp_min=min_pad if right_new else old_rp
            if lp_cap<lp_min or rp_cap<rp_min: continue
            lp=min(target,lp_cap) if left_new else old_lp
            rp=min(target,rp_cap) if right_new else old_rp
            excess=max(0,voice_b-voice_a+lp+rp-maximum)
            delta=min(excess,lp-lp_min);lp-=delta;rp-=excess-delta
            if rp<rp_min: continue
            start,end=voice_a-lp,voice_b+rp
            if not minimum<=end-start<=maximum or crosses_music(start,end): continue
            # Retain all of the old clip, even its original internal padding.
            if start>old_a or end<old_b: continue
            plan=dict(start=start/rate,end=end/rate,speech_start=voice_a/rate,
                      speech_end=voice_b/rate)
            if set(edge_pause_evidence(plan,pauses))!={'start','end'}: continue
            tail=scores[j+1]
            score=(covered+tail[0],count+tail[1],tail[2]-(end-start-(old_b-old_a)),tail[3]-1)
            if score>scores[i]:
                scores[i]=score
                choices[i]=(j,base,plan)
    replacements,remaining=[],[]
    i=0
    while i<n:
        choice=choices[i]
        if choice is None:
            if not timeline[i][2]: remaining.append(timeline[i][3])
            i+=1;continue
        j,base,plan=choice
        parts=[dict(start=r[3]['start'],end=r[3]['end']) for r in timeline[i:j+1] if not r[2]]
        replacements.append((base,dict(plan,merged_unresolved_parts=parts)))
        i=j+1
    return replacements,remaining


def propose_short_review_merges(unresolved, candidates, pauses, regions, duration):
    """Length-valid proposals whose outer edges still need explicit listening review.

    Keep the strict pass unchanged. Also allow a whole 5–15 s unresolved span
    as a pending candidate, with optional neighboring short fragments. At most
    one existing candidate or length-valid unresolved anchor per proposal. Never
    cross music or another clip, and retain all source samples. Unconfirmed
    context is not reported as detected silence.
    """
    rate=16000
    minimum,maximum=5*rate,15*rate
    min_pad,target,max_pad=(round(x*rate) for x in (MIN_EDGE_SECONDS,EDGE_SECONDS,MAX_EDGE_SECONDS))
    timeline=sorted([(round(x['start']*rate),round(x['end']*rate),True,x) for x in candidates]+
                    [(round(x['start']*rate),round(x['end']*rate),False,x) for x in unresolved],key=lambda x:(x[0],x[1]))
    n=len(timeline)
    if any(a[1]>b[0] for a,b in zip(timeline,timeline[1:])):
        return [],list(unresolved)
    evidence=[(round(a*rate),round(b*rate)) for a,b in merge_intervals(pauses)]
    music=[(round(r['start']*rate),round(r['end']*rate)) for r in regions if r['label']=='music']
    def crosses_music(a,b):
        return any(x<b and y>a for x,y in music)
    def context(t,left,i):
        neighbor=i-1 if left else i+1
        limit=(t-timeline[neighbor][1] if left else timeline[neighbor][0]-t) if 0<=neighbor<n else (t if left else round(duration*rate)-t)
        if 0<=neighbor<n and not timeline[neighbor][2]:limit//=2
        limit=max(0,min(target,limit))
        for a,b in music:
            if left and b<=t:limit=min(limit,t-b)
            if not left and a>=t:limit=min(limit,a-t)
        measured=max([0]+[min(limit,t-a if left else b-t) for a,b in evidence if a<=t<=b])
        return limit,measured
    scores=[(0,0,0,0)]*(n+1); choices=[None]*n
    for i in range(n-1,-1,-1):
        scores[i]=scores[i+1];a=timeline[i][0];base=None;full_span=False;count=covered=0
        for j in range(i,n):
            x,y,is_clip,row=timeline[j]
            if y-a>maximum or (j>i and x-timeline[j-1][1]>round(GLOBAL_MERGE_MAX_GAP_SECONDS*rate)) or crosses_music(a,y):break
            if is_clip:
                if base is not None or full_span:break
                base=row;base_i=j
            else:
                if not 0<y-x<=maximum:break
                if y-x>=minimum:
                    if base is not None or full_span:break
                    full_span=True
                count+=1;covered+=y-x
            if count<1 or (base is None and count<2 and not full_span):continue
            # Preserve established far edges of a retained candidate. New outer
            # edges get up to 150 ms source context, never fabricated silence.
            lcap,lmeas=context(a,True,i) if not (base is not None and i==base_i) else (0,0)
            rcap,rmeas=context(y,False,j) if not (base is not None and j==base_i) else (0,0)
            lp,rp=lcap,rcap
            excess=max(0,y-a+lp+rp-maximum);d=min(excess,lp);lp-=d;rp-=excess-d
            if not minimum<=y-a+lp+rp<=maximum or crosses_music(a-lp,y+rp):continue
            left_known=base is not None and i==base_i
            right_known=base is not None and j==base_i
            lverified=round(base['left_padding']*rate) if left_known else min(lp,lmeas)
            rverified=round(base['right_padding']*rate) if right_known else min(rp,rmeas)
            # Padding is confirmed only if the WHOLE context is measured quiet.
            lverified=lverified if left_known or lverified==lp and lp>=min_pad else 0
            rverified=rverified if right_known or rverified==rp and rp>=min_pad else 0
            start,end=(a-lp)/rate,(y+rp)/rate
            boundary={}
            if left_known and 'start' in base.get('boundary_evidence',{}):boundary['start']=base['boundary_evidence']['start']
            elif not left_known and lverified>=min_pad:boundary.update(edge_pause_evidence(dict(start=start,end=end,speech_start=a/rate,speech_end=y/rate),pauses))
            if right_known and 'end' in base.get('boundary_evidence',{}):boundary['end']=base['boundary_evidence']['end']
            elif not right_known and rverified>=min_pad:
                e=edge_pause_evidence(dict(start=start,end=end,speech_start=a/rate,speech_end=y/rate),pauses)
                if 'end' in e:boundary['end']=e['end']
            boundary={k:v for k,v in boundary.items() if (k=='start' and lverified>=min_pad) or (k=='end' and rverified>=min_pad)}
            tail=scores[j+1];score=(covered+tail[0],count+tail[1],len(boundary)+tail[2],tail[3]-1)
            if score>scores[i]:
                scores[i]=score
                choices[i]=(j,base,dict(start=start,end=end,left_padding=lverified/rate,right_padding=rverified/rate,
                    boundary_evidence=boundary,unverified_context_start_seconds=(0 if left_known or lverified else lp/rate),
                    unverified_context_end_seconds=(0 if right_known or rverified else rp/rate),
                    merged_unresolved_parts=[dict(start=t[3]['start'],end=t[3]['end']) for t in timeline[i:j+1] if not t[2]]))
    out=[];remaining=[];i=0
    while i<n:
        if choices[i] is None:
            if not timeline[i][2]:remaining.append(timeline[i][3])
            i+=1;continue
        j,base,plan=choices[i];out.append((base,plan));i=j+1
    return out,remaining


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


# Keep the source time left over after the supplied intervals have been taken out.
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
            # Keep supported internal pauses even when this isolated span cannot
            # form a clip. The final block-wide regrouping may combine a short
            # part with a neighboring candidate at one of these boundaries.
            used_pauses.extend(proposals)
            i = positions[round(a*16000)]
            low = left if i == 0 else (active[i-1][1]+a)/2
            high = right if i+1 == len(active) else (b+active[i+1][0])/2
            clips, leftovers = group_at_pauses(subtract_intervals([[a,b]], proposals), low, high)
            if clips:
                for clip in clips:
                    clip['recovered'] = True
                recovered.extend(clips)
                remaining.extend(leftovers)
                continue
        remaining.append(item)
    return sorted(groups + recovered, key=lambda clip: clip['start']), remaining, used_pauses


# Find the measured pause around each outer edge of the candidate.
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


# Show the proposed clips and the audio that still needs a listener.
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
<section><h2>Overview</h2><p><b>Candidate clips:</b> {len(data['segments'])} &nbsp; <b>Unresolved spans:</b> {len(data['unresolved'])} &nbsp; <b>Eligible blocks:</b> {data['eligible_block_count']}</p><p>Settings: 5–15 s; target edge padding 0.15 s with at least 0.1 s on both edges; Silero VAD first pass {MIN_PAUSE_MS} ms; both edges require 0.1–0.2 s detector-estimated natural pause; raw 240 ms vocal pauses may supplement speech VAD. Global short-span merging can preserve internal gaps up to {GLOBAL_MERGE_MAX_GAP_SECONDS:g} seconds across eligible blocks, provided no music-only region or other candidate is crossed. Length-valid unresolved spans and fallback merges with uncertain outer pauses appear as pending review candidates; measured padding is reported only where supported. Approve them individually after listening and correcting boundaries. Adjacent labelled silence can supply measured edge context. Long unresolved spans get a targeted {REFINE_PAUSE_MS} ms pause check. No artificial silence or joins are generated.</p></section>
<section><h2>Time accounting</h2><p>Speech/mixed according to Step 3: {eligible:.2f} s. Total candidate duration: {clip_time:.2f} s, including {coverage.get("adjacent_silence_in_clips",0.):.2f} s retained from Step 3 silence (edges and internal gaps). Unresolved spans: {unresolved_time:.2f} s. Other eligible audio outside candidates and unresolved spans: {unchecked_time:.2f} s. Total labelled music/silence in Step 3: {excluded:.2f} s.</p><p>Unchecked audio can include VAD-missed speech; the music/silence labels can also be wrong. Review them against the original audio. These categories are time-accounting aids, not a measurement of transcription completeness.</p></section>
<section><h2>Candidate clips</h2><div class="scroll"><table><tr><th>File</th><th>Start (s)</th><th>End (s)</th><th>Duration (s)</th><th>Label</th><th>Left padding (s)</th><th>Right padding (s)</th><th>Review notes</th><th>Listen</th></tr>{''.join(rows)}</table></div></section>
<section><h2>Unresolved spans</h2><p>Each span is also saved intact with a little context in needs_manual_review. These WAVs can be shorter than 5 seconds or longer than 15 seconds and are not accepted corpus clips. Inspect and select valid pause boundaries manually.</p><div class="scroll"><table><tr><th>Start (s)</th><th>End (s)</th><th>Reason</th><th>Review audio</th></tr>{''.join(misses)}</table></div></section>
<section><h2>Eligible audio outside clips and unresolved spans</h2><p>{len(unchecked)} intervals, total {unchecked_time:.2f} s. The original WAV remains unchanged. These detector-estimated pauses can still contain quiet words; listen before marking them reviewed.</p><details><summary>Show interval times</summary><div class="scroll"><table><tr><th>Start (s)</th><th>End (s)</th><th>Duration (s)</th></tr>{unchecked_rows}</table></div></details></section>
<section>Machine-readable results and the Step 3 snapshot are in step4_segments.json in this folder. Times refer to the original WAV.</section></main></body></html>'''


def export_candidates(audio_path, regions, output_parent, progress=lambda text: None, *, use_vocal_pauses=True):
    """Propose continuous 5–15 second clips with a measured pause at each end.

    Each candidate uses the original samples, including its quiet edges.
    Leave uncertain boundaries for a listener before approval."""
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
        data=dict(version=3,segmentation_revision='length_valid_review_v7',audio_name=path.name,audio_size=path.stat().st_size,duration_seconds=duration,
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
            # A short phrase between two clips gets one home. Try the following
            # clip first, then offer only the remaining phrases to the preceding clip.
            groups, unresolved, pauses = prepend_short_unresolved(
                audio, groups, unresolved, pauses, left, vad, checks)
            groups, unresolved, pauses = extend_candidates_with_short_spans(
                audio, groups, unresolved, pauses, right, vad, checks, prepend=False)
            # Reconsider the entire eligible block after both recovery and the
            # original one-sided merges. Do not sacrifice already covered speech.
            groups, unresolved = regroup_supported_pauses(
                groups, unresolved, pauses, start, end, left, right)
            data['pause_intervals'].extend(dict(start=a,end=b,status='estimated_pause_review_required') for a,b in pauses)
            data['unresolved'].extend(unresolved)
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
                if segment.get('regrouped'):
                    data['segments'][-1]['regrouped'] = True
                    data['segments'][-1]['notes'].append('Recovered by regrouping supported pauses with flexible natural edge padding.')
                for key, position in (('prepended_unresolved', 'preceding'), ('appended_unresolved', 'following')):
                    if key in segment:
                        fragment = segment[key]
                        row = data['segments'][-1]
                        row[key] = fragment
                        row['notes'].append(f"Includes {position} unresolved speech/mixed audio from "
                                            f"{fragment['start']:.3f} to {fragment['end']:.3f} s.")
        # Final global pass: Step 3 silence can separate short eligible blocks.
        # Keep the continuous original audio between them, not concatenated WAVs.
        progress('Combining neighboring short unresolved spans across eligible blocks…')
        all_pauses=merge_intervals((p['start'],p['end']) for p in data['pause_intervals'])
        merged,remaining=merge_short_spans_across_blocks(data['unresolved'],data['segments'],
                                                       all_pauses,regions,duration)
        for segment in merged:
            a,b=round(segment['start']*16000),round(segment['end']*16000)
            evidence=edge_pause_evidence(segment,all_pauses)
            lp=segment['speech_start']-a/16000
            rp=b/16000-segment['speech_end']
            if (not 5*16000<=b-a<=15*16000 or set(evidence)!={'start','end'} or
                    not all(MIN_EDGE_SECONDS-1/16000<=v<=MAX_EDGE_SECONDS+1/16000 for v in (lp,rp))):
                raise RuntimeError('Global short-span merge failed natural-pause validation.')
            overlap=[r for r in regions if r['end']>a/16000 and r['start']<b/16000]
            label='mixed' if any(r['label']=='mixed' for r in overlap) else 'speech'
            filename=f"segment_{len(data['segments'])+1:05d}.wav"
            audio.seek(a);samples=audio.read(b-a,dtype='int16',always_2d=True)
            if len(samples)!=b-a:raise RuntimeError('Incomplete source audio while exporting merged candidate.')
            sf.write(output/filename,samples,16000,subtype='PCM_16')
            data['segments'].append(dict(file=filename,start=a/16000,end=b/16000,duration=(b-a)/16000,
                label=label,kind='global_short_span_merge',left_padding=lp,right_padding=rp,
                boundary_evidence=evidence,source_spans=[dict(start=a/16000,end=b/16000,clip_start=0.,clip_end=(b-a)/16000)],
                removed_gaps=[],synthetic_silence_start_seconds=0.,synthetic_silence_end_seconds=0.,
                status='needs_review',merged_unresolved_parts=segment['merged_unresolved_parts'],
                notes=['Merged neighboring short unresolved spans using original continuous audio, including internal gaps.',
                       'Both outer edges have detector-estimated natural pause; listen and verify complete words before approval.']))
        # Absorb any remaining short fragments into either neighboring candidate,
        # even if Step 3 split them into different eligible blocks. Rewrite the
        # candidate WAV from the original source; do not concatenate clip files.
        progress('Absorbing short unresolved spans into neighboring candidates…')
        replacements,remaining=absorb_short_neighbors(remaining,data['segments'],all_pauses,regions,duration)
        for old, segment in replacements:
            a,b=round(segment['start']*16000),round(segment['end']*16000)
            evidence=edge_pause_evidence(segment,all_pauses)
            lp=segment['speech_start']-a/16000
            rp=b/16000-segment['speech_end']
            if (not 5*16000<=b-a<=15*16000 or set(evidence)!={'start','end'} or
                    not all(MIN_EDGE_SECONDS-1/16000<=v<=MAX_EDGE_SECONDS+1/16000 for v in (lp,rp))):
                raise RuntimeError('Neighbor absorption failed natural-pause validation.')
            audio.seek(a);samples=audio.read(b-a,dtype='int16',always_2d=True)
            if len(samples)!=b-a:raise RuntimeError('Incomplete source audio during neighbor absorption.')
            sf.write(output/old['file'],samples,16000,subtype='PCM_16')
            overlap=[r for r in regions if r['end']>a/16000 and r['start']<b/16000]
            old_bounds=dict(start=old['start'],end=old['end'],kind=old['kind'])
            old.update(start=a/16000,end=b/16000,duration=(b-a)/16000,
                label='mixed' if any(r['label']=='mixed' for r in overlap) else 'speech',
                kind='global_neighbor_absorption',left_padding=lp,right_padding=rp,
                boundary_evidence=evidence,source_spans=[dict(start=a/16000,end=b/16000,clip_start=0.,clip_end=(b-a)/16000)],
                status='needs_review',original_candidate_bounds=old_bounds,
                absorbed_neighbor_parts=segment['merged_unresolved_parts'])
            old['notes'].append('Extended existing candidate with adjacent short unresolved audio, preserving original samples and verified-duration outer pause evidence; listen before approval.')
        data['segments'].sort(key=lambda item:item['start'])
        data['neighbor_absorption']=dict(extended_candidates=len(replacements),
            absorbed_unresolved_parts=sum(len(s['merged_unresolved_parts']) for _,s in replacements),
            maximum_internal_gap_seconds=GLOBAL_MERGE_MAX_GAP_SECONDS,requires_natural_outer_pauses=True)
        data['global_short_merge']=dict(merged_candidates=len(merged),
            absorbed_unresolved_parts=sum(len(s['merged_unresolved_parts']) for s in merged),
            maximum_internal_gap_seconds=GLOBAL_MERGE_MAX_GAP_SECONDS,requires_natural_outer_pauses=True,
            crosses_music_regions=False)
        # A length-valid merge is useful as a review candidate even if its
        # new outer pause was not proven by the detector. Never auto-approve it.
        progress('Preparing length-valid spans and merges for boundary review…')
        proposed,remaining=propose_short_review_merges(remaining,data['segments'],all_pauses,regions,duration)
        for old,plan in proposed:
            a,b=round(plan['start']*16000),round(plan['end']*16000)
            if not 5*16000<=b-a<=15*16000:raise RuntimeError('Invalid review merge duration.')
            filename=old['file'] if old is not None else f"segment_{len(data['segments'])+1:05d}.wav"
            audio.seek(a);samples=audio.read(b-a,dtype='int16',always_2d=True)
            if len(samples)!=b-a:raise RuntimeError('Incomplete source audio during review merge.')
            sf.write(output/filename,samples,16000,subtype='PCM_16')
            overlap=[r for r in regions if r['end']>a/16000 and r['start']<b/16000]
            single=old is None and len(plan['merged_unresolved_parts'])==1
            row=dict(file=filename,start=a/16000,end=b/16000,duration=(b-a)/16000,
                label='mixed' if any(r['label']=='mixed' for r in overlap) else 'speech',
                kind='length_valid_needs_boundary_review' if single else 'merged_needs_boundary_review',status='needs_review',manual_boundary_review_required=True,
                left_padding=plan['left_padding'],right_padding=plan['right_padding'],boundary_evidence=plan['boundary_evidence'],
                unverified_context_start_seconds=plan['unverified_context_start_seconds'],unverified_context_end_seconds=plan['unverified_context_end_seconds'],
                merged_unresolved_parts=plan['merged_unresolved_parts'],
                source_spans=[dict(start=a/16000,end=b/16000,clip_start=0.,clip_end=(b-a)/16000)],
                removed_gaps=[],synthetic_silence_start_seconds=0.,synthetic_silence_end_seconds=0.,
                notes=[('LENGTH VALID' if single else 'MERGED') + ' — BOUNDARY REVIEW REQUIRED: valid 5–15 s continuous source interval. Outer natural pause(s) not fully verified. Listen and adjust source bounds if needed, then approve individually.'])
            if old is None:data['segments'].append(row)
            else:
                row['original_candidate_bounds']=dict(start=old['start'],end=old['end'],kind=old['kind'])
                old.update(row)
        data['segments'].sort(key=lambda item:item['start'])
        data['review_merges']=dict(candidates=len(proposed),single_length_valid_candidates=sum(old is None and len(p['merged_unresolved_parts'])==1 for old,p in proposed),absorbed_unresolved_parts=sum(len(p['merged_unresolved_parts']) for _,p in proposed),requires_individual_boundary_review=True)
        # Provide concrete diagnostics for still-unresolved spans. The lack of
        # pause evidence is different from a duration/gap/barrier rejection.
        # This does not fabricate or relax outer-edge evidence.
        ordered = sorted([(u['start'],u['end'],False,u) for u in remaining] +
                         [(c['start'],c['end'],True,c) for c in data['segments']],
                         key=lambda row:(row[0],row[1]))
        def edge_supported(t, left):
            a,b = (t-MIN_EDGE_SECONDS,t) if left else (t,t+MIN_EDGE_SECONDS)
            return any(x-1e-8<=a and b<=y+1e-8 for x,y in all_pauses)
        for i,(a,b,is_candidate,item) in enumerate(ordered):
            if is_candidate or b-a>=MIN_SECONDS:
                continue
            neighbors=[]
            for j,side in ((i-1,'previous'),(i+1,'next')):
                if 0<=j<len(ordered):
                    x,y,is_clip,_=ordered[j]
                    neighbors.append(dict(side=side,kind='candidate' if is_clip else 'unresolved',
                        gap_seconds=round(max(0.,a-y if side=='previous' else x-b),5),
                        combined_extent_seconds=round(max(b,y)-min(a,x),5)))
            detail=dict(start_pause_supported=edge_supported(a,True),
                        end_pause_supported=edge_supported(b,False),
                        max_internal_gap_seconds=GLOBAL_MERGE_MAX_GAP_SECONDS,neighbors=neighbors)
            item['merge_diagnostics']=detail
            missing=[side for side,ok in [('start',detail['start_pause_supported']),
                                          ('end',detail['end_pause_supported'])] if not ok]
            why=('; no measured 100 ms pause at own '+ ' and '.join(missing) if missing else '')
            neighbor_text=', '.join(f"{n['side']} {n['kind']} gap {n['gap_seconds']:.2f} s, combined extent {n['combined_extent_seconds']:.2f} s" for n in neighbors)
            item['reason']='No safe length-valid continuous merge within gap, music and overlap limits'+why+('. '+neighbor_text if neighbor_text else '')+'. Review boundaries manually.'
        # Preserve the old within-block grouping of leftovers for manual review.
        data['unresolved']=[]
        pending=iter(sorted(remaining,key=lambda u:u['start']))
        row=next(pending,None)
        for start,end in blocks:
            leftovers=[]
            while row is not None and round(row['start']*16000)<round(end*16000):
                # Do not silently drop unexpected ranges from the old detector.
                if (round(row['start']*16000)<round(start*16000) or
                        round(row['end']*16000)>round(end*16000)):
                    data['unresolved'].append(row)
                else:
                    leftovers.append(row)
                row=next(pending,None)
            if leftovers:
                local_candidates=[g for g in data['segments'] if g['end']>start and g['start']<end]
                data['unresolved'].extend(group_short_unresolved(leftovers,local_candidates))
        if row is not None:
            data['unresolved'].append(row)
            data['unresolved'].extend(pending)
        data['unresolved'].sort(key=lambda u:u['start'])
        # Account for silence retained anywhere in a candidate, including internal
        # bridges across Step 3 silence, without counting it as eligible voice.
        borrowed=sum(max(0.,min(c['end'],b)-max(c['start'],a))
                     for c in data['segments'] for a,b in silence)
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

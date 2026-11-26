"""Label a recording one analysis window at a time.

analysis_settings reads the window and hop. classify_file gathers the sound-event
scores, decide_content chooses a label, and merge_same_labeled_regions combines
neighboring windows with the same label. The original window scores stay available for
review.
"""

from pathlib import Path
from urllib.request import urlretrieve
import math
import copy
import os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL_URL = 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
LABELS_URL = 'https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
WINDOW_SECONDS = 2.0
HOP_SECONDS = 2.0
MIN_WINDOW_SECONDS = 2.0
MAX_WINDOW_SECONDS = 30.0
MIN_HOP_SECONDS = 0.1
SILENCE_DBFS = -45.0
SPEECH_THRESHOLD = 0.30
MUSIC_THRESHOLD = 0.30
VOCAL_THRESHOLD = 0.30
VOCAL_REVIEW_THRESHOLD = 0.15
# Resolve these by name: AudioSet indices are not hard-coded. Music genres,
# instruments, Synthetic singing, Humming and Beatboxing are not lyric evidence.
VOCAL_CLASSES = ('Singing', 'Male singing', 'Female singing', 'Child singing',
                 'Choir', 'Rapping', 'Vocal music', 'A capella')
SCORE_FIELDS = ('confidence', 'auto_confidence', 'speech_score', 'music_score',
                'vocal_score', 'voice_score', 'music_evidence_score', 'verified_voice_score')
LABELING_RULE = 'panns_speech_music_vocals_v2'


def analysis_settings(window_seconds=None, hop_seconds=None):
    """Validate optional settings; blank/None values use the existing defaults."""
    # An empty settings box means "use the default". A number entered by the user must be
    # valid.
    def number(value, default, name):
        if value is None or isinstance(value, str) and not value.strip():
            return default
        try:
            if isinstance(value, bool):
                raise ValueError()
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a number in seconds.') from exc
        if not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number in seconds.')
        return value
    window = number(window_seconds, WINDOW_SECONDS, 'Window length')
    hop = number(hop_seconds,  window * 0.5, 'Hop length')
    if not MIN_WINDOW_SECONDS <= window <= MAX_WINDOW_SECONDS:
        raise ValueError(f'Window length must be between {MIN_WINDOW_SECONDS:g} and {MAX_WINDOW_SECONDS:g} seconds.')
    if not MIN_HOP_SECONDS <= hop <= MAX_WINDOW_SECONDS:
        raise ValueError(f'Hop length must be between {MIN_HOP_SECONDS:g} and {MAX_WINDOW_SECONDS:g} seconds.')
    if hop > window:
        raise ValueError('Hop length must not exceed the window length, so no audio is skipped.')
    return window, hop


def recorded_analysis_settings(regions):
    """Return uniform settings from the results, not the next UI input values."""
    settings = None
    for region in regions:
        for window in region.get('analysis_windows', [region]):
            config = window.get('analysis_config')
            if config is None or settings is not None and config != settings:
                return None
            settings = config
    return copy.deepcopy(settings)


def decide_content(speech, music, vocal):
    """Map uncalibrated scores to the corpus's four-label policy.

    Vocal music (including a cappella) is proposed as mixed for this corpus.
    This is a labeling policy, not a claim that lyrics or accompaniment were
    transcribed/verified. Keep the raw Speech and Music scores separate.
    """
    voice = max(speech, vocal)
    music_evidence = max(music, vocal)
    if vocal >= VOCAL_THRESHOLD:
        return 'mixed', min(voice, music_evidence), True, (
            'Singing/vocal music detected; proposed mixed under corpus policy. '
            'Listen to confirm intelligible vocals and accompaniment; no lyrics were recognized.')
    has_speech, has_music = speech >= SPEECH_THRESHOLD, music >= MUSIC_THRESHOLD
    if has_speech and has_music:
        return 'mixed', min(speech, music), False, ''
    if has_speech:
        return 'speech', speech, False, ''
    if has_music:
        possible_vocals = vocal >= VOCAL_REVIEW_THRESHOLD
        return 'music', music, possible_vocals, (
            'Possible vocals below the decision threshold; listen before treating this as instrumental music.'
            if possible_vocals else '')
    label = 'speech' if speech >= music else 'music'
    return label, max(speech, music), True, 'Low speech/music scores; review background noise and possible vocals.'


# Download to a temporary file so an interrupted transfer can't leave a half-written model in
# use.
def download_if_missing(path, url, minimum_size):
    path = Path(path)
    if path.is_file() and path.stat().st_size >= minimum_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    try:
        urlretrieve(url, tmp)
        if tmp.stat().st_size < minimum_size:
            raise RuntimeError(f'Incomplete download: {path.name}')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def merge_same_labeled_regions(windows):
    """Merge touching equal labels, retaining each original inference record.

    Region scores are duration-weighted means of the original window scores,
    not a new model prediction. Any review flag remains set on the region.
    Gaps, overlaps and label changes always start a new region.
    """
    groups = []
    for window in windows:
        if (groups and groups[-1][-1]['label'] == window['label']
                and abs(float(groups[-1][-1]['end']) - float(window['start'])) <= 1e-9):
            groups[-1].append(window)
        else:
            groups.append([window])
    result = []
    for group in groups:
        original = copy.deepcopy(group)
        region = copy.deepcopy(group[0])
        region.update(start=group[0]['start'], end=group[-1]['end'],
                      analysis_windows=original, window_count=len(group),
                      needs_review=any(w.get('needs_review', False) for w in group))
        # Individual separation context/energy checks belong to their exact windows.
        region.pop('separation_check', None)
        methods = sorted({w['vocal_verification_method'] for w in group if w.get('vocal_verification_method')})
        if methods:
            region['vocal_verification_method'] = ', '.join(methods)
        # A window that covers more of the merged region contributes more to its score.
        def mean_score(key):
            valid = [(float(w['end']) - float(w['start']), float(w[key]))
                     for w in group if w.get(key) is not None]
            weight = sum(d for d, score in valid)
            return sum(d * score for d, score in valid) / weight if weight else None
        for key in SCORE_FIELDS:
            region[key] = mean_score(key)
        names = {name for w in group for name in w.get('vocal_class_scores', {})}
        if names:
            region['vocal_class_scores'] = {}
            for name in sorted(names):
                values = [(w['end'] - w['start'], w['vocal_class_scores'][name])
                          for w in group if name in w.get('vocal_class_scores', {})]
                region['vocal_class_scores'][name] = sum(d * v for d, v in values) / sum(d for d, v in values)
            region['vocal_evidence_class'] = max(region['vocal_class_scores'], key=region['vocal_class_scores'].get)
        # Average energy in linear units before converting it back to dBFS.
        energy = [(float(w['end']) - float(w['start']), 10 ** (float(w['rms_dbfs']) / 10))
                  for w in group if w.get('rms_dbfs') is not None]
        if energy:
            power = sum(d * e for d, e in energy) / sum(d for d, e in energy)
            region['rms_dbfs'] = 10 * math.log10(max(power, 1e-24))
        notes = list(dict.fromkeys(w['note'] for w in group if w.get('note')))
        if len(group) > 1:
            notes.insert(0, f"Merged {len(group)} adjacent {region['label']} windows; scores are duration-weighted averages. Original window scores are retained in analysis_windows.")
            region['score_aggregation'] = 'duration_weighted_mean'
        region['note'] = ' '.join(notes)
        result.append(region)
    return result


def classify_file(path, progress=lambda x: None, *, verify_vocals=True,
                  window_seconds=None, hop_seconds=None):
    """Tag overlapping input windows; assign each prediction to its leading hop.

    Input bounds and padding are retained separately from the non-overlapping
    decision intervals. Default 2/2 behavior matches the previous classifier.
    """
    window_seconds, hop_seconds = analysis_settings(window_seconds, hop_seconds)
    progress(f'PANNs settings: window {window_seconds:g} s; hop {hop_seconds:g} s.')
    progress('Preparing PANNs model; the first download is about 327 MB...')
    data = Path.home() / 'panns_data'
    download_if_missing(data / 'class_labels_indices.csv', LABELS_URL, 1000)
    checkpoint = data / 'Cnn14_mAP=0.431.pth'
    download_if_missing(checkpoint, MODEL_URL, 300_000_000)
    from panns_inference import AudioTagging, labels
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AudioTagging(checkpoint_path=str(checkpoint), device=device)
    speech_index = list(labels).index('Speech')
    music_index = list(labels).index('Music')
    vocal_indices = [(name, list(labels).index(name)) for name in VOCAL_CLASSES if name in labels]
    if not vocal_indices:
        raise RuntimeError('The PANNs label list has no singing/vocal classes; check the installed AudioSet labels.')
    # Run the voice tags on the separated track at the classifier's expected sample rate.
    def tag_voice(raw, rate):
        divisor = math.gcd(rate, 32000)
        x = resample_poly(raw, 32000 // divisor, rate // divisor).astype(np.float32)
        x = np.pad(x, (0, max(0, 64000 - len(x))))
        scores, _ = model.inference(x[None, :])
        voice = {'Speech': float(scores[0, speech_index])}
        voice.update({name: float(scores[0, index]) for name, index in vocal_indices})
        category = max(voice, key=voice.get)
        return dict(score=voice[category], category=category, class_scores=voice)
    windows = []
    with sf.SoundFile(path) as audio:
        rate, length = audio.samplerate, len(audio)
        if not length:
            raise ValueError('The selected audio is empty.')
        window_size = max(1, round(rate * window_seconds))
        step = max(1, round(rate * hop_seconds))
        config = dict(window_seconds=window_size / rate, hop_seconds=step / rate,
                      requested_window_seconds=window_seconds, requested_hop_seconds=hop_seconds,
                      coverage_policy='leading_hop_cell')
        total_windows = math.ceil(length / step)
        for number in range(total_windows):
            start_sample = number * step
            audio.seek(start_sample)
            raw = audio.read(window_size, dtype='float32', always_2d=True).mean(axis=1)
            input_end_sample = start_sample + len(raw)
            end_sample = min(length, start_sample + step)
            # Silence describes the decision interval, not a later sound that
            # happens to occur in the wider overlapping model context.
            local = raw[:end_sample - start_sample]
            rms = float(np.sqrt(np.mean(local.astype(np.float64) ** 2)))
            dbfs = 20 * math.log10(max(rms, 1e-12))
            start, end = start_sample / rate, end_sample / rate
            if dbfs < SILENCE_DBFS:
                record = dict(start=start, end=end, label='silence', confidence=min(1.0, max(0.0, (SILENCE_DBFS - dbfs) / 20.0)),
                              speech_score=None, music_score=None, rms_dbfs=dbfs,
                              vocal_score=None, voice_score=None, music_evidence_score=None,
                              vocal_class_scores={}, vocal_evidence_class=None,
                              needs_review=True, note='Energy threshold; review quiet speech')
            else:
                divisor = math.gcd(rate, 32000)
                x = resample_poly(raw, 32000 // divisor, rate // divisor).astype(np.float32)
                # PANNs requires a long enough input even for a short final window.
                padding = max(0, max(64000, round(window_seconds * 32000)) - len(x))
                x = np.pad(x, (0, padding))
                scores, _ = model.inference(x[None, :])
                speech, music = float(scores[0, speech_index]), float(scores[0, music_index])
                vocal_scores = {name: float(scores[0, index]) for name, index in vocal_indices}
                vocal_class = max(vocal_scores, key=vocal_scores.get)
                vocal = vocal_scores[vocal_class]
                label, confidence, needs_review, note = decide_content(speech, music, vocal)
                if vocal >= VOCAL_REVIEW_THRESHOLD:
                    note = (note + f' Strongest vocal category: {vocal_class} ({vocal:.3f}).').strip()
                record = dict(start=start, end=end, label=label, confidence=confidence,
                              speech_score=speech, music_score=music, rms_dbfs=dbfs,
                              vocal_score=vocal, voice_score=max(speech, vocal),
                              music_evidence_score=max(music, vocal),
                              vocal_class_scores=vocal_scores, vocal_evidence_class=vocal_class,
                              needs_review=needs_review,
                              note=note)
                record['inference_padding_seconds'] = padding / 32000
            record['analysis_config'] = config.copy()
            record['inference_start'] = start_sample / rate
            record['inference_end'] = input_end_sample / rate
            if step < window_size and record['label'] != 'silence':
                record['needs_review'] = True
                record['note'] = (record.get('note', '') +
                    ' Overlapping input context; this prediction applies to the leading hop interval. Review exact boundaries.').strip()
            record['labeling_rule'] = LABELING_RULE
            record['auto_label'] = record['label']
            record['auto_confidence'] = record['confidence']
            windows.append(record)
            if number % 5 == 0 or number + 1 == total_windows:
                progress(f'Analyzing: {number + 1} / {total_windows} windows')
    if verify_vocals and any(w['label'] == 'music' and (w.get('music_score') or 0) >= MUSIC_THRESHOLD for w in windows):
        from vocal_verification import VocalSeparator, verify_music_windows
        separator = VocalSeparator(device=device, progress=progress)
        windows = verify_music_windows(path, windows, tag_voice, separator, progress,
                                       MUSIC_THRESHOLD, VOCAL_THRESHOLD, interval_seconds=step / rate)
    # Simplify the timeline without discarding the original model windows.
    return merge_same_labeled_regions(windows)


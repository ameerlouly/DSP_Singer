"""Step 3: PANNs speech/music tagging plus an RMS silence detector."""
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
SILENCE_DBFS = -45.0
SPEECH_THRESHOLD = 0.30
MUSIC_THRESHOLD = 0.30


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
        def mean_score(key):
            valid = [(float(w['end']) - float(w['start']), float(w[key]))
                     for w in group if w.get(key) is not None]
            weight = sum(d for d, score in valid)
            return sum(d * score for d, score in valid) / weight if weight else None
        for key in ('confidence', 'auto_confidence', 'speech_score', 'music_score'):
            region[key] = mean_score(key)
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


def classify_file(path, progress=lambda x: None):
    """Return merged contiguous regions with traceable, uncalibrated scores."""
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
    windows = []
    with sf.SoundFile(path) as audio:
        rate, length = audio.samplerate, len(audio)
        if not length:
            raise ValueError('The selected audio is empty.')
        step = max(1, round(rate * WINDOW_SECONDS))
        total_windows = math.ceil(length / step)
        for number in range(total_windows):
            start_sample = audio.tell()
            raw = audio.read(step, dtype='float32', always_2d=True).mean(axis=1)
            end_sample = start_sample + len(raw)
            rms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)))
            dbfs = 20 * math.log10(max(rms, 1e-12))
            start, end = start_sample / rate, end_sample / rate
            if dbfs < SILENCE_DBFS:
                record = dict(start=start, end=end, label='silence', confidence=min(1.0, max(0.0, (SILENCE_DBFS - dbfs) / 20.0)),
                              speech_score=None, music_score=None, rms_dbfs=dbfs,
                              needs_review=True, note='Energy threshold; review quiet speech')
            else:
                divisor = math.gcd(rate, 32000)
                x = resample_poly(raw, 32000 // divisor, rate // divisor).astype(np.float32)
                # PANNs requires a long enough input even for a short final window.
                x = np.pad(x, (0, max(0, 64000 - len(x))))
                scores, _ = model.inference(x[None, :])
                speech, music = float(scores[0, speech_index]), float(scores[0, music_index])
                has_speech, has_music = speech >= SPEECH_THRESHOLD, music >= MUSIC_THRESHOLD
                if has_speech and has_music:
                    label, confidence, needs_review = 'mixed', min(speech, music), False
                elif has_speech:
                    label, confidence, needs_review = 'speech', speech, False
                elif has_music:
                    label, confidence, needs_review = 'music', music, False
                else:
                    # Non-silent sound with no confident speech/music detection.
                    label = 'speech' if speech >= music else 'music'
                    confidence, needs_review = max(speech, music), True
                record = dict(start=start, end=end, label=label, confidence=confidence,
                              speech_score=speech, music_score=music, rms_dbfs=dbfs,
                              needs_review=needs_review,
                              note='Low speech/music scores; review background noise' if needs_review else '')
            record['auto_label'] = record['label']
            record['auto_confidence'] = record['confidence']
            windows.append(record)
            if number % 5 == 0 or number + 1 == total_windows:
                progress(f'Analyzing: {number + 1} / {total_windows} windows')
    # Simplify the timeline without discarding the original model windows.
    return merge_same_labeled_regions(windows)


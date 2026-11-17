"""Step 3: PANNs speech/music tagging plus an RMS silence detector."""
from pathlib import Path
from urllib.request import urlretrieve
import math
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


def classify_file(path, progress=lambda x: None):
    """Return contiguous regions. Scores are heuristic, not calibrated probabilities."""
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
    # Keep windows separate so their actual model scores remain traceable.
    return windows


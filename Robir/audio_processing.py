"""Audio processing functions shared with the user interface."""
import math
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

def calculate_spectrogram(path, max_columns=4000):
    """Return time, frequency and relative-power dB arrays for a WAV file.

    The file is read in blocks. For long recordings, adjacent time columns are
    combined using the maximum power so the plot remains manageable.
    """
    window, hop = 512, 256
    with sf.SoundFile(path) as audio:
        rate, total = audio.samplerate, len(audio)
        if total == 0:
            raise ValueError('The selected WAV file is empty.')
        if rate <= 0:
            raise ValueError('The WAV file has an invalid sample rate.')
        duration = total / rate
        frame_count = max(1, 1 + max(0, total - window) // hop)
        stride = max(1, math.ceil(frame_count / max_columns))
        # Make each block a multiple of the reduction interval; frames align
        # across blocks, with the window overlap read again in the next block.
        frames_per_block = max(stride, (max(1, int(20 * rate / hop)) // stride) * stride)
        time_parts, power_parts = [], []
        first_frame = 0
        while first_frame < frame_count:
            count = min(frames_per_block, frame_count - first_frame)
            audio.seek(first_frame * hop)
            samples = audio.read((count - 1) * hop + window, dtype='float32', always_2d=True)
            samples = samples.mean(axis=1)
            if len(samples) < window:
                samples = np.pad(samples, (0, window - len(samples)))
            frequencies, times, power = spectrogram(
                samples, fs=rate, window='hann', nperseg=window,
                noverlap=window-hop, detrend=False, scaling='density', mode='psd')
            indexes = np.arange(0, power.shape[1], stride)
            reduced = np.maximum.reduceat(power, indexes, axis=1)
            time_parts.append((first_frame * hop / rate + times[indexes]).astype(np.float64))
            power_parts.append(reduced.astype(np.float32))
            first_frame += count
        times = np.concatenate(time_parts)
        power = np.concatenate(power_parts, axis=1)
        peak = float(np.max(power))
        # dB relative to the loudest spectrogram bin; silence stays at -80 dB.
        db = np.full_like(power, -80.0) if peak <= 0 else np.maximum(
            10 * np.log10(np.maximum(power / peak, 1e-8)), -80.0)
        return times, frequencies, db, rate, duration, stride





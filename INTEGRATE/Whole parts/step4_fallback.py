"""Review-only low-energy splits for long VAD intervals.

Low amplitude is not a verified pause. No source samples are omitted from a
fallback interval; artificial silence is explicitly recorded in metadata.
"""
import numpy as np

RATE = 16000
SILENCE_SECONDS = 0.1
PAD = int(SILENCE_SECONDS * RATE)
MIN_CONTENT = int(5.0 * RATE) - 2 * PAD
MAX_CONTENT = int(15.0 * RATE) - 2 * PAD
FRAME = int(.020 * RATE)
HOP = int(.010 * RATE)


def low_energy_boundary(audio, low, high):
    """Find the lowest RMS 20-ms window center in inclusive sample bounds."""
    if high < low:
        raise ValueError('No valid range for the candidate boundary.')
    start = max(0, low - FRAME // 2)
    stop = min(len(audio), high + FRAME // 2 + 1)
    audio.seek(start)
    x = audio.read(stop - start, dtype='float32', always_2d=True)[:, 0]
    if len(x) < FRAME:
        return low, None
    squared = np.asarray(x, dtype=np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    first = max(0, low - FRAME // 2 - start)
    last = min(len(x) - FRAME, high - FRAME // 2 - start)
    offsets = np.arange(first, last + 1, HOP, dtype=np.int64)
    if not len(offsets):
        return low, None
    energies = (cumulative[offsets + FRAME] - cumulative[offsets]) / FRAME
    minimum = float(np.min(energies))
    tied = np.flatnonzero(np.isclose(energies, minimum, rtol=1e-6, atol=1e-14))
    target = (low + high) / 2
    centers = start + offsets[tied] + FRAME // 2
    best = int(tied[np.argmin(np.abs(centers - target))])
    center = start + int(offsets[best]) + FRAME // 2
    return min(high, max(low, center)), float(np.sqrt(max(0.0, energies[best])))


def plan_low_energy_splits(audio, start_sample, end_sample):
    """Partition the complete interval; all clips include synthetic edge padding.

    Returns [] for intervals too short to make a 5-second clip. Each internal
    boundary is a candidate and must be checked by listening to the source.
    """
    if end_sample - start_sample < MIN_CONTENT:
        return []
    boundaries = [start_sample]
    rms_at_cut = []
    cursor = start_sample
    while end_sample - cursor > MAX_CONTENT:
        # Keep at least a legal minimum of source audio for the final clip.
        lo = cursor + MIN_CONTENT
        hi = min(cursor + MAX_CONTENT, end_sample - MIN_CONTENT)
        if hi < lo:
            raise ValueError('Cannot partition the interval into valid candidate lengths.')
        cut, rms = low_energy_boundary(audio, lo, hi)
        if not lo <= cut <= hi:
            raise RuntimeError('Invalid candidate boundary.')
        boundaries.append(cut)
        rms_at_cut.append(rms)
        cursor = cut
    boundaries.append(end_sample)
    return [{'start':a, 'end':b,
             'left_cut_rms': rms_at_cut[i-1] if i else None,
             'right_cut_rms': rms_at_cut[i] if i < len(rms_at_cut) else None}
            for i,(a,b) in enumerate(zip(boundaries,boundaries[1:]))]

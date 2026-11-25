"""Check whether a window labeled music also contains singing.

VocalSeparator extracts an estimated vocal track, and verify_music_windows checks it
against voice tags and local energy. Instruments can leak into the separated track, so
energy alone is weak evidence for a voice.
"""

import copy
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL_NAME = 'HDEMUCS_HIGH_MUSDB_PLUS'
BLOCK_SECONDS = 8.0
CONTEXT_SECONDS = 1.0
MIN_VOCAL_RATIO_DB = -30.0
MIN_VOCAL_DBFS = -60.0
RULE = 'panns_demucs_vocals_v3'


class VocalSeparator:
    # Load the separator and look up which output track contains the vocals.
    def __init__(self, device='cpu', progress=lambda text: None):
        try:
            import torch
            from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
        except (ImportError, OSError) as exc:
            raise RuntimeError('Improved singing detection needs compatible torch/torchaudio packages. '
                               'Install requirements.txt, or turn off Improve singing detection for the basic classifier.') from exc
        self.torch = torch
        self.device = device
        progress('Preparing vocal verification model; first use downloads about 319 MiB…')
        self.rate = HDEMUCS_HIGH_MUSDB_PLUS.sample_rate
        self.model = HDEMUCS_HIGH_MUSDB_PLUS.get_model().to(device).eval()
        self.index = self.model.sources.index('vocals')

    # Extract the vocal track and align it with the length of the original audio.
    def __call__(self, mono, rate):
        original_length = len(mono)
        if not original_length:
            return np.empty(0, dtype=np.float32)
        divisor = math.gcd(rate, self.rate)
        wave = resample_poly(mono, self.rate // divisor, rate // divisor).astype(np.float32)
        wave = np.pad(wave, (0, max(0, self.rate * 2 - len(wave))))
        mean, scale = float(wave.mean()), float(wave.std())
        if scale < 1e-8:
            return np.zeros(original_length, dtype=np.float32)
        mix = self.torch.from_numpy(np.stack([wave, wave])).to(self.device)
        with self.torch.inference_mode():
            sources = self.model(((mix - mean) / scale)[None])[0]
        # DC is not vocal evidence. Keep the original gain, without peak-normalizing
        # quiet leakage into an artificially loud signal.
        vocals = sources[self.index].mean(dim=0).detach().cpu().numpy() * scale
        divisor = math.gcd(rate, self.rate)
        vocals = resample_poly(vocals, rate // divisor, self.rate // divisor).astype(np.float32)
        return np.pad(vocals, (0, max(0, original_length - len(vocals))))[:original_length]


# Measure the RMS level in decibels, using the floor value for unusable input.
def level(audio):
    if not len(audio) or not np.isfinite(audio).all():
        return -240.0
    return 20 * math.log10(max(1e-12, float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))))


def verify_music_windows(path, windows, tag_voice, separator, progress=lambda text: None,
                         music_threshold=.30, voice_threshold=.30, *, interval_seconds=None):
    """Return reviewed proposals, preserving raw mixture scores and every decision.

    tag_voice returns an uncalibrated Speech/Singing evidence score plus its
    category. Context evidence supports a short window only if its local stem
    has sufficient energy. This can still blur a transition, so proposals need
    listening review. File names, source-specific times and lyrics are unused.
    """
    result = copy.deepcopy(windows)
    if interval_seconds is not None and (not math.isfinite(interval_seconds) or not 0 < interval_seconds <= 30):
        raise ValueError('Vocal verification interval must be between zero and 30 seconds.')
    interval_limit = interval_seconds if interval_seconds is not None else BLOCK_SECONDS
    block_seconds = max(BLOCK_SECONDS, interval_limit)
    blocks = {}
    for index, window in enumerate(result):
        if window['label'] == 'music' and (window.get('music_score') or 0) >= music_threshold:
            if not 0 < window['end'] - window['start'] <= interval_limit + 1e-6:
                raise ValueError('Verify original short analysis windows before merging regions.')
            block = math.floor(window['start'] / block_seconds + 1e-9)
            blocks.setdefault(block, []).append(index)
    with sf.SoundFile(path) as audio:
        rate, length = audio.samplerate, len(audio)
        for count, (block, indices) in enumerate(sorted(blocks.items()), 1):
            first = max(0, round((block * block_seconds - CONTEXT_SECONDS) * rate))
            end = max((block + 1) * block_seconds, max(result[i]['end'] for i in indices))
            last = min(length, round((end + CONTEXT_SECONDS) * rate))
            audio.seek(first)
            mixture = audio.read(last - first, dtype='float32', always_2d=True).mean(axis=1)
            vocals = np.asarray(separator(mixture, rate), dtype=np.float32)
            if len(vocals) != len(mixture) or not np.isfinite(vocals).all():
                raise ValueError('The vocal separator returned invalid or misaligned audio.')
            context_voice = tag_voice(vocals, rate) if level(vocals) >= MIN_VOCAL_DBFS else {'score': 0., 'category': ''}
            checks = []
            for index in indices:
                window = result[index]
                a = round(window['start'] * rate) - first
                b = round(window['end'] * rate) - first
                if not 0 <= a < b <= len(mixture):
                    raise ValueError('Vocal analysis window lies outside the source audio.')
                local = vocals[a:b]
                vocal_db = level(local)
                ratio_db = vocal_db - level(mixture[a:b])
                meaningful = vocal_db >= MIN_VOCAL_DBFS and ratio_db >= MIN_VOCAL_RATIO_DB
                local_voice = tag_voice(local, rate) if meaningful else {'score': 0., 'category': ''}
                checks.append((index, vocal_db, ratio_db, meaningful, local_voice))
            # A whole-context score can dilute alternating spoken/sung phrases.
            # Retain a strong short-window detection as contextual support, while
            # still requiring vocal-stem energy at EACH proposed time interval.
            support_voice = dict(context_voice, evidence_start=first / rate,
                                 evidence_end=last / rate, source='full_context')
            for index, _, _, meaningful, local_voice in checks:
                if meaningful and local_voice['score'] > support_voice['score']:
                    support_voice = dict(local_voice, evidence_start=result[index]['start'],
                                         evidence_end=result[index]['end'], source='nearby_window')
            for index, vocal_db, ratio_db, meaningful, local_voice in checks:
                window = result[index]
                best = max((local_voice, support_voice), key=lambda item: item['score'])
                confirmed = meaningful and best['score'] >= voice_threshold
                window['labeling_rule'] = RULE
                window['verified_voice_score'] = float(best['score']) if meaningful else 0.
                window['vocal_verification_method'] = MODEL_NAME
                window['separation_check'] = dict(
                    model=MODEL_NAME, context_start=first / rate, context_end=last / rate,
                    local_voice=local_voice, context_voice=context_voice, support_voice=support_voice,
                    vocal_dbfs=vocal_db, vocal_to_mix_db=ratio_db,
                    energy_gate_passed=meaningful, proposed_mixed=confirmed,
                    prior_label=window['label'], prior_confidence=window.get('confidence'),
                    voice_threshold=voice_threshold, minimum_vocal_ratio_db=MIN_VOCAL_RATIO_DB,
                    minimum_vocal_dbfs=MIN_VOCAL_DBFS,
                )
                if confirmed:
                    window.update(label='mixed', auto_label='mixed', needs_review=True,
                                  confidence=min(window['music_score'], float(best['score'])))
                    window['auto_confidence'] = window['confidence']
                    note = ('Vocal separation plus voice tagging supports mixed; '
                            'listen to confirm vocals and exact boundaries. No lyrics were recognized.')
                    if local_voice['score'] < voice_threshold:
                        note += ' Voice evidence uses the surrounding analysis context.'
                else:
                    # Retain uncertainty rather than declaring verified instrumental.
                    window['needs_review'] = True
                    note = 'Vocal check did not establish voice; music remains a proposal, not proof of absent vocals.'
                window['note'] = (str(window.get('note') or '') + ' ' + note).strip()
            progress(f'Checking vocals in music: {count} / {len(blocks)} blocks')
    return result

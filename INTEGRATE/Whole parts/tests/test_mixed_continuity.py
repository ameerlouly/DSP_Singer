"""Protect mixed audio from speech-only VAD misses, without declaring lyrics present."""
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import step4_cut_segments as natural
import step4_compact_segments as compact

RATE = 16000


def tone(seconds, amplitude=.15):
    t = np.arange(round(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class MixedContinuityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / 'source.wav'

    def tearDown(self):
        self.directory.cleanup()

    def check(self, samples, islands, regions=None):
        sf.write(self.path, samples, RATE, subtype='PCM_16')
        duration = len(samples) / RATE
        regions = regions or [dict(start=0., end=duration, label='mixed')]
        with sf.SoundFile(self.path) as audio:
            return natural.protect_mixed_continuity(audio, islands, 0., duration, regions)

    def test_active_mixed_preserves_internal_leading_trailing_misses(self):
        result, checks = self.check(tone(12), [[2.,4.],[6.,8.]])
        self.assertEqual(result, [[0.,12.]])
        self.assertAlmostEqual(checks[0]['vad_missing_seconds_preserved'], 8.)
        self.assertFalse(checks[0]['quiet_intervals'])

    def test_total_vad_miss_still_preserves_mixed_audio(self):
        result, checks = self.check(tone(12), [])
        self.assertEqual(result, [[0.,12.]])
        self.assertEqual(checks[0]['vad_missing_seconds_preserved'], 12.)

    def test_sustained_quiet_remains_available_as_pause(self):
        samples = tone(12)
        samples[round(5.5*RATE):6*RATE] = 0
        result, checks = self.check(samples, [[0.,3.],[8.,10.]])
        self.assertEqual(result, [[0.,5.5],[6.,12.]])
        self.assertEqual(checks[0]['quiet_intervals'], [[5.5,6.]])

    def test_short_note_dip_is_not_a_cut(self):
        samples = tone(12)
        samples[5*RATE:round(5.1*RATE)] = 0
        result, _ = self.check(samples, [[0.,5.],[5.1,12.]])
        self.assertEqual(result, [[0.,12.]])

    def test_quiet_singing_not_suppressed_by_absolute_threshold_alone(self):
        result, checks = self.check(tone(12, .0003), [])
        self.assertEqual(result, [[0.,12.]])
        self.assertFalse(checks[0]['quiet_intervals'])

    def test_mixed_protection_does_not_override_speech_music_silence_labels(self):
        regions = [dict(start=0.,end=4.,label='speech'),dict(start=4.,end=8.,label='mixed'),
                   dict(start=8.,end=10.,label='music'),dict(start=10.,end=12.,label='silence')]
        result, checks = self.check(tone(12), [[1.,2.]], regions)
        self.assertEqual(result, [[1.,2.],[4.,8.]])
        self.assertEqual(len(checks), 1)
        self.assertEqual(natural.eligible_blocks(regions), [[0.,8.]])

    def test_read_seam_and_partial_final_frame(self):
        samples = tone(31.013)
        samples[round(29.9*RATE):round(30.2*RATE)] = 0
        result, checks = self.check(samples, [])
        self.assertEqual(result, [[0.,29.9],[30.2,31.013]])
        self.assertEqual(checks[0]['quiet_intervals'], [[29.9,30.2]])

    def test_silence_is_not_promoted_to_active_audio(self):
        result, checks = self.check(np.zeros(12*RATE), [])
        self.assertEqual(result, [])
        self.assertEqual(checks[0]['quiet_intervals'], [[0.,12.]])

    def test_vocal_pause_can_be_found_while_backing_audio_continues(self):
        # Controlled separator: background is .1, background plus voice is .2.
        samples = np.full(12*RATE, .2, dtype=np.float32)
        samples[round(5.5*RATE):6*RATE] = .1
        sf.write(self.path, samples, RATE, subtype='PCM_16')
        separator = lambda wave, rate: np.where(wave > .15, .1, 0.).astype(np.float32)
        with sf.SoundFile(self.path) as audio:
            self.assertEqual(natural.quiet_mixed_intervals(audio, 0., 12.), [])
            islands, checks = natural.protect_mixed_continuity(audio, [[0.,3.],[8.,10.]],
                0., 12., [dict(start=0.,end=12.,label='mixed')], separator=separator)
        self.assertEqual(islands, [[0.,5.5],[6.,12.]])
        self.assertEqual(checks[0]['quiet_evidence'], 'separated_vocals_energy')
        self.assertIsNone(checks[0]['separation_error'])

    def test_failed_or_invalid_separator_falls_back_to_preserving_audio(self):
        sf.write(self.path, tone(12), RATE, subtype='PCM_16')
        def fail(wave, rate):
            raise RuntimeError('Controlled separator failure')
        for separator in (fail, lambda wave, rate: wave[:-1]):
            with self.subTest(separator=separator), sf.SoundFile(self.path) as audio:
                islands, checks = natural.protect_mixed_continuity(audio, [], 0., 12.,
                    [dict(start=0.,end=12.,label='mixed')], separator=separator)
            self.assertEqual(islands, [[0.,12.]])
            self.assertEqual(checks[0]['quiet_evidence'], 'original_mix_conservative')
            self.assertTrue(checks[0]['separation_error'])

    def test_separator_silence_cannot_erase_vad_positive_audio(self):
        sf.write(self.path, tone(12), RATE, subtype='PCM_16')
        with sf.SoundFile(self.path) as audio:
            islands, _ = natural.protect_mixed_continuity(audio, [[0.,12.]], 0., 12.,
                [dict(start=0.,end=12.,label='mixed')],
                separator=lambda wave, rate: np.zeros_like(wave))
        self.assertEqual(islands, [[0.,12.]])

    def export(self, module, seconds, detected):
        sf.write(self.path, tone(seconds), RATE, subtype='PCM_16')
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        fake_torch = types.ModuleType('torch')
        fake_torch.set_num_threads = lambda count: None
        fake_silero = types.ModuleType('silero_vad')
        fake_silero.load_silero_vad = lambda: object()
        # The second refinement pass also misses these vocals.
        with patch.dict(sys.modules, torch=fake_torch, silero_vad=fake_silero), \
                patch.object(natural, 'detect_speech', return_value=detected):
            output = module.export_candidates(self.path, [dict(start=0.,end=seconds,label='mixed')], self.root, use_vocal_pauses=False)
        data = json.loads((Path(output['output_dir'])/'step4_segments.json').read_text())
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(data['not_in_clips_or_unresolved'], [])
        self.assertTrue(data['mixed_continuity_checks'])
        self.assertTrue(all(s['status'].startswith('needs_review') for s in data['segments']))
        return data, Path(output['output_dir'])

    def test_both_exporters_preserve_missed_audio_when_natural_edges_are_missing(self):
        for module in (natural,compact):
            with self.subTest(module=module.__name__):
                data, output = self.export(module, 12., [[2.,4.],[6.,8.]])
                self.assertFalse(data['segments'])
                self.assertEqual([(u['start'],u['end']) for u in data['unresolved']], [(0.,12.)])

    def test_long_vocals_not_forced_into_fake_natural_pauses(self):
        data, _ = self.export(natural, 40., [])
        self.assertFalse(data['segments'])
        self.assertEqual([(u['start'],u['end']) for u in data['unresolved']], [(0.,40.)])
        self.assertTrue(data['unresolved'][0]['mixed_continuity_protected'])
        data, _ = self.export(compact, 40., [])
        self.assertFalse(data['segments'])
        self.assertEqual([(u['start'],u['end']) for u in data['unresolved']], [(0.,40.)])


if __name__ == '__main__':
    unittest.main()

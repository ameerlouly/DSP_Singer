"""Deterministic separation/voice-gating checks; no model download required."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vocal_verification import verify_music_windows
from step3_label_regions import merge_same_labeled_regions
from region_editing import relabel_range


def region(start=0., end=2., label='music'):
    return dict(start=start, end=end, label=label, auto_label=label, confidence=.9,
                auto_confidence=.9, speech_score=.02, music_score=.9,
                vocal_score=.05, note='', needs_review=False)


class Verification(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture.wav'
        self.rate = 16000
        time = np.arange(self.rate * 20) / self.rate
        sf.write(self.path, .2 * np.sin(2 * np.pi * 200 * time), self.rate, subtype='PCM_16')

    def test_voice_and_music_propose_mixed_without_replacing_raw_scores(self):
        windows = [region(), region(2, 4, 'speech')]
        original = copy.deepcopy(windows)
        result = verify_music_windows(self.path, windows,
                                      lambda x, rate: dict(score=.8, category='Speech'),
                                      lambda x, rate: x * .5)
        self.assertEqual(result[0]['label'], 'mixed')
        self.assertEqual(result[0]['auto_label'], 'mixed')
        self.assertAlmostEqual(result[0]['confidence'], .8)
        self.assertEqual(result[0]['vocal_score'], .05)
        self.assertTrue(result[0]['needs_review'])
        self.assertEqual(result[0]['separation_check']['prior_label'], 'music')
        self.assertEqual(result[1], windows[1])
        self.assertEqual(windows, original)

    def test_separator_energy_alone_is_not_voice_proof(self):
        result = verify_music_windows(self.path, [region()],
                                      lambda x, rate: dict(score=.1, category='Singing'),
                                      lambda x, rate: x * .8)
        self.assertEqual(result[0]['label'], 'music')
        self.assertTrue(result[0]['needs_review'])
        self.assertFalse(result[0]['separation_check']['proposed_mixed'])

    def test_tiny_leakage_does_not_become_loud_voice_evidence(self):
        calls = []
        def tag(x, rate):
            calls.append(len(x))
            return dict(score=.99, category='Singing')
        result = verify_music_windows(self.path, [region()], tag, lambda x, rate: x * 1e-5)
        self.assertEqual(result[0]['label'], 'music')
        self.assertFalse(result[0]['separation_check']['energy_gate_passed'])
        self.assertEqual(calls, [])

    def test_context_support_is_recorded_and_still_requires_local_energy(self):
        def tag(x, rate):
            return dict(score=.8 if len(x) > 2 * rate else .1, category='Singing')
        result = verify_music_windows(self.path, [region()], tag, lambda x, rate: x * .5)
        self.assertEqual(result[0]['label'], 'mixed')
        self.assertEqual(result[0]['separation_check']['local_voice']['score'], .1)
        self.assertIn('surrounding analysis context', result[0]['note'])
        def local_silence(x, rate):
            out=x.copy();out[:2 * rate]=0
            return out
        result = verify_music_windows(self.path, [region()], tag, local_silence)
        self.assertEqual(result[0]['label'], 'music')

    def test_blocks_are_bounded_and_only_music_candidates_are_checked(self):
        lengths = []
        def separate(x, rate):
            lengths.append(len(x))
            return x * .5
        windows = [region(), region(2, 4), region(8, 10), region(10, 12, 'speech')]
        result = verify_music_windows(self.path, windows,
                                      lambda x, rate: dict(score=.8, category='Singing'), separate)
        self.assertEqual(len(lengths), 2)
        self.assertTrue(all(n <= 10 * self.rate for n in lengths))
        self.assertEqual(result[-1], windows[-1])

    def test_nearby_strong_voice_can_support_an_energetic_weak_window(self):
        values=iter([.1, .8, .1])  # Full context, first local window, second local window.
        result=verify_music_windows(self.path,[region(),region(2,4)],
                                    lambda x,rate:dict(score=next(values),category='Speech'),
                                    lambda x,rate:x*.5)
        self.assertEqual([r['label'] for r in result],['mixed','mixed'])
        check=result[1]['separation_check']
        self.assertEqual(check['local_voice']['score'],.1)
        self.assertEqual(check['context_voice']['score'],.1)
        self.assertEqual(check['support_voice']['evidence_start'],0)
        self.assertEqual(check['support_voice']['evidence_end'],2)

    def test_invalid_separator_outputs_fail_without_partial_mutation(self):
        windows = [region()]
        original = copy.deepcopy(windows)
        for separate in (lambda x, rate: x[:-1], lambda x, rate: np.full_like(x, np.nan)):
            with self.assertRaises(ValueError):
                verify_music_windows(self.path, windows,
                                     lambda x, rate: dict(score=.8, category='Singing'), separate)
        self.assertEqual(windows, original)

    def test_long_merged_region_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'before merging'):
            verify_music_windows(self.path, [region(0, 18)],
                                 lambda x, rate: dict(score=.8, category='Singing'),
                                 lambda x, rate: x * .5)

    def test_merged_and_cropped_regions_keep_window_evidence(self):
        windows = [region(), region(2, 4)]
        verified = verify_music_windows(self.path, windows,
                                        lambda x, rate: dict(score=.8, category='Singing'),
                                        lambda x, rate: x * .5)
        merged = merge_same_labeled_regions(verified)[0]
        self.assertNotIn('separation_check', merged)
        self.assertEqual(len(merged['analysis_windows']), 2)
        self.assertAlmostEqual(merged['verified_voice_score'], .8)
        pieces, index = relabel_range(merged, 2, 4, 'speech')
        self.assertEqual(pieces[index]['analysis_windows'][0]['separation_check'], verified[1]['separation_check'])
        self.assertAlmostEqual(pieces[index]['verified_voice_score'], .8)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Regression checks for songs classified as music when Speech scores are low.

Inference outputs are controlled fixtures; this does not measure model accuracy.

Test guide
----------
VocalLabeling
    Check decisions, each vocal class is used with shuffled indices and related cases.
"""
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audio_region_classifier as tagging
from region_editing import relabel_range
from region_review_report import save_report, load_review


class VocalLabeling(unittest.TestCase):
    def test_decisions(self):
        cases = [
            ('instrumental', .02, .95, .01, 'music', False),
            ('sung lyrics with low speech', .02, .95, .8, 'mixed', True),
            ('vocal music with low parent music', .02, .05, .8, 'mixed', True),
            ('speech over music', .8, .6, .01, 'mixed', False),
            ('plain speech', .8, .02, .01, 'speech', False),
            ('uncertain vocals in music', .02, .95, .2, 'music', True),
            ('vocal threshold inclusive', .02, .95, .3, 'mixed', True),
            ('vocal threshold below', .02, .95, .299, 'music', True),
            ('weak background sound', .1, .2, .1, 'music', True),
        ]
        for name, speech, music, vocal, expected, review in cases:
            with self.subTest(name=name):
                label, score, needs_review, note = tagging.decide_content(speech, music, vocal)
                self.assertEqual((label, needs_review), (expected, review))
                self.assertTrue(0 <= score <= 1)
                if vocal >= tagging.VOCAL_THRESHOLD:
                    self.assertIn('no lyrics were recognized', note)

    def infer(self, names, vectors, silence=False):
        calls = []

        class Tagger:
            def __init__(self, **kwargs):
                self.index = 0

            def inference(self, audio):
                self_shape = audio.shape
                if self_shape != (1, 64000):
                    raise AssertionError(self_shape)
                calls.append(self.index)
                result = np.array([vectors[self.index]], dtype=np.float32)
                self.index += 1
                return result, None

        module = types.ModuleType('panns_inference')
        module.labels, module.AudioTagging = names, Tagger
        torch = types.ModuleType('torch')
        torch.Tensor = type('Tensor', (), {})
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / 'source.wav'
            time = np.arange(2 * 16000 * len(vectors)) / 16000
            audio = np.zeros_like(time) if silence else .2 * np.sin(2 * np.pi * 200 * time)
            sf.write(wav, audio, 16000, subtype='PCM_16')
            with patch.dict(sys.modules, panns_inference=module, torch=torch), \
                    patch.object(tagging, 'download_if_missing'):
                regions = tagging.classify_file(wav, verify_vocals=False)
        return regions, calls

    def test_each_vocal_class_is_used_with_shuffled_indices(self):
        # High genre/instrument/synthetic scores must not count as real vocal evidence.
        names = ['Guitar', 'Music', *reversed(tagging.VOCAL_CLASSES),
                 'Humming', 'Beatboxing', 'Synthetic singing', 'Song', 'Speech']
        for vocal_class in tagging.VOCAL_CLASSES:
            with self.subTest(vocal_class=vocal_class):
                vector = [.01] * len(names)
                for name in ('Guitar', 'Music', 'Humming', 'Beatboxing', 'Synthetic singing', 'Song'):
                    vector[names.index(name)] = .9
                vector[names.index(vocal_class)] = .8
                regions, calls = self.infer(names, [vector])
                self.assertEqual(regions[0]['label'], 'mixed')
                self.assertEqual(regions[0]['vocal_evidence_class'], vocal_class)
                self.assertTrue(regions[0]['needs_review'])
                self.assertLess(regions[0]['speech_score'], .02)
                self.assertAlmostEqual(regions[0]['vocal_score'], .8)
                self.assertEqual(regions[0]['labeling_rule'], tagging.LABELING_RULE)
                vector[names.index(vocal_class)] = .01
                instrumental, _ = self.infer(names, [vector])
                self.assertEqual(instrumental[0]['label'], 'music')
                self.assertFalse(instrumental[0]['needs_review'])

    def test_missing_vocal_categories_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError, 'no singing/vocal classes'):
            self.infer(['Speech', 'Music'], [[.01, .9]])

    def test_silence_bypasses_vocal_inference(self):
        regions, calls = self.infer(['Speech', 'Music', 'Singing'], [[.01, .9, .9]], silence=True)
        self.assertEqual(regions[0]['label'], 'silence')
        self.assertIsNone(regions[0]['vocal_score'])
        self.assertEqual(calls, [])

    def test_merge_edit_and_report_preserve_vocal_evidence(self):
        regions, calls = self.infer(['Rapping', 'Speech', 'Music', 'Singing'],
                                    [[.01, .02, .9, .8], [.7, .02, .9, .01]])
        self.assertEqual(len(regions), 1)
        merged = regions[0]
        original = copy.deepcopy(merged)
        self.assertEqual(merged['window_count'], 2)
        self.assertAlmostEqual(merged['vocal_score'], .75)
        self.assertTrue(merged['needs_review'])
        self.assertAlmostEqual(merged['vocal_class_scores']['Singing'], .405)
        pieces, selected = relabel_range(merged, 2, 4, 'speech')
        edited = pieces[selected]
        self.assertEqual(edited['label'], 'speech')
        self.assertEqual(edited['auto_label'], 'mixed')
        self.assertEqual(edited['vocal_evidence_class'], 'Rapping')
        self.assertAlmostEqual(edited['vocal_score'], .7)
        self.assertAlmostEqual(edited['vocal_class_scores']['Rapping'], .7)
        self.assertEqual(merged, original)
        payload = dict(audio_name='source.wav', audio_size=128044, duration_seconds=4,
                       regions=pieces, changes=[])
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'report.html'
            save_report(report, payload)
            self.assertEqual(load_review(report), payload)
            html = report.read_text(encoding='utf-8')
            self.assertIn('Vocal score', html)
            self.assertIn('Rapping', html)


if __name__ == '__main__':
    unittest.main(verbosity=2)

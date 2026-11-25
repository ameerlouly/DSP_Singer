"""Part 3 subrange edits: timeline invariants, Qt actions and saved reviews.

Test guide
----------
RangeEdits
    Check middle preserves both sides and original, edges and whole region and related
    cases.
check_qt_workflow
    Exercise range edits, playback, validation and session round-tripping through Qt.
"""
import copy
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_editing import RATE, relabel_range


class RangeEdits(unittest.TestCase):
    def setUp(self):
        self.region = dict(start=12., end=180., label='speech', auto_label='speech',
                           confidence=.8, auto_confidence=.8, needs_review=True)

    def test_middle_preserves_both_sides_and_original(self):
        original = copy.deepcopy(self.region)
        pieces, selected = relabel_range(self.region, 45, 60, 'music')
        self.assertEqual(selected, 1)
        self.assertEqual([(r['start'], r['end'], r['label']) for r in pieces],
                         [(12., 45., 'speech'), (45., 60., 'music'), (60., 180., 'speech')])
        self.assertTrue(pieces[0]['needs_review'])
        self.assertFalse(pieces[1]['needs_review'])
        self.assertTrue(pieces[1]['manual_override'])
        self.assertEqual(pieces[1]['auto_label'], 'speech')
        self.assertEqual(pieces[1]['auto_confidence'], .8)
        self.assertEqual(self.region, original)

    def test_edges_and_whole_region(self):
        for start, end, count, selected in [(12, 60, 2, 0), (45, 180, 2, 1), (12, 180, 1, 0)]:
            with self.subTest(start=start, end=end):
                pieces, index = relabel_range(self.region, start, end, 'silence')
                self.assertEqual((len(pieces), index), (count, selected))
                self.assertEqual(sum(r['end'] - r['start'] for r in pieces), 168)
                self.assertTrue(all(r['end'] > r['start'] for r in pieces))

    def test_invalid_ranges_rejected_without_mutating_original(self):
        original = copy.deepcopy(self.region)
        for start, end in [(11, 60), (45, 181), (60, 45), (45, 45), (45, 45 + .1 / RATE),
                           (float('nan'), 60), (45, float('inf')), ('bad', 60)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                relabel_range(self.region, start, end, 'music')
        with self.assertRaises(ValueError):
            relabel_range(self.region, 45, 60, 'unknown')
        self.assertEqual(self.region, original)

    def test_single_sample_and_sample_alignment(self):
        start = 45 + 1 / RATE
        pieces, selected = relabel_range(self.region, start, start + 1 / RATE, 'mixed')
        self.assertAlmostEqual((pieces[selected]['end'] - pieces[selected]['start']) * RATE, 1)
        for region in pieces:
            for key in ('start', 'end'):
                self.assertAlmostEqual(region[key] * RATE, round(region[key] * RATE))

    def test_original_window_scores_weighted_by_overlap(self):
        self.region['analysis_windows'] = [
            dict(start=12., end=50., confidence=.8, auto_confidence=.8, rms_dbfs=-20.),
            dict(start=50., end=180., confidence=.4, auto_confidence=.4, rms_dbfs=-40.)]
        pieces, index = relabel_range(self.region, 45, 60, 'music')
        edited = pieces[index]
        self.assertAlmostEqual(edited['auto_confidence'], (5 * .8 + 10 * .4) / 15)
        self.assertAlmostEqual(edited['rms_dbfs'], 10 * math.log10((5 * .01 + 10 * .0001) / 15))
        self.assertEqual(edited['analysis_windows'], self.region['analysis_windows'])
        self.assertEqual(pieces[0]['window_count'], 1)
        self.assertEqual(pieces[2]['window_count'], 1)
        self.assertEqual(edited['window_count'], 2)

    def test_repeated_edits_preserve_reviewed_outside_pieces(self):
        pieces, index = relabel_range(self.region, 45, 60, 'music')
        new, selected = relabel_range(pieces[index], 48, 52, 'mixed')
        self.assertEqual([r['label'] for r in new], ['music', 'mixed', 'music'])
        self.assertTrue(all(r['manual_override'] and not r['needs_review'] for r in new))
        self.assertEqual((new[selected]['start'], new[selected]['end']), (48, 52))


def check_qt_workflow():
    import time
    import numpy as np
    import soundfile as sf
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from speech_corpus_app import MainWindow
    from audio_review_controller import validate_regions
    from region_review_report import load_review

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    p = window.processing
    warnings = []
    with tempfile.TemporaryDirectory() as folder, \
            patch.object(QMessageBox, 'warning', side_effect=lambda *args: warnings.append(args)), \
            patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
        root = Path(folder)
        wav = root / 'source.wav'
        samples = np.zeros(190 * RATE)
        samples[RATE:2*RATE] = .2 * np.sin(2*np.pi*1000*np.arange(RATE)/RATE)
        sf.write(wav, samples, RATE, subtype='PCM_16')
        assert p.set_audio(wav)
        window.reviewer.setText('Range reviewer')
        p.regions = [dict(start=0., end=12., label='silence'),
                     dict(start=12., end=180., label='speech', auto_label='speech',
                          auto_confidence=.8, confidence=.8),
                     dict(start=180., end=190., label='music')]
        original = copy.deepcopy(p.regions)
        p.refresh_all()
        window.regionTable.selectRow(1)
        assert window.regionEditMode.currentIndex() == 0
        window.regionStart.setValue(45)
        window.regionEnd.setValue(60)
        window.regionLabel.setCurrentText('music')
        with patch.object(p, 'play') as playback:
            window.playEnteredRegionRange.click()
            playback.assert_called_once_with(wav, 45., 60.)
        # Existing downstream artifacts must be invalidated by any region edit.
        p.session = {'old': 'candidate session'}
        p.snr_results = [{'old': 'quality result'}]
        p.snr_report = root / 'old.html'
        with patch.object(p.asr, 'source_changed', wraps=p.asr.source_changed) as stale:
            window.applyRegion.click()
            stale.assert_called_once()
        assert not warnings, warnings
        assert p.session is None and p.snr_results == [] and p.snr_report is None
        assert p.regions[0] == original[0] and p.regions[-1] == original[-1]
        assert [(r['start'], r['end'], r['label']) for r in p.regions[1:4]] == [
            (12., 45., 'speech'), (45., 60., 'music'), (60., 180., 'speech')]
        validate_regions(p.regions, 190)
        assert window.regionTable.rowCount() == 5 and window.regionTable.currentRow() == 2
        assert p.changes[-1]['action'] == 'relabel_subrange'
        assert p.changes[-1]['annotator'] == 'Range reviewer'
        assert p.changes[-1]['before'] == original[1]
        assert p.dirty

        # Invalid edits warn and leave both timeline and audit untouched.
        before, history = copy.deepcopy(p.regions), copy.deepcopy(p.changes)
        window.regionStart.setValue(44)
        window.applyRegion.click()
        assert len(warnings) == 1 and p.regions == before and p.changes == history
        warnings.clear()

        def wait():
            deadline=time.monotonic()+20
            while p.task is not None and time.monotonic()<deadline:
                app.processEvents()
                time.sleep(.01)
            assert p.task is None

        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(root)):
            p.save_regions();wait()
        report, = (root / 'Labeled audio regions').glob('*/source_regions.html')
        assert load_review(report)['regions'] == p.regions
        project = root / 'session.json'
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(project), '')):
            p.save_project();wait()
        p.regions = []
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(project), '')):
            p.open_project();wait()
        def check_restored_plot():
            deadline = time.monotonic() + 10
            while p.task is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            assert p.task is None and p.regions == before and p.changes == history
            _, frequencies, db, *_ = p.plot_data
            assert frequencies[np.argmax(db.max(axis=1))] == 1000
            assert p.axes.get_ylim() == (0., RATE/2) and p.axes.get_ylabel() == 'Frequency (Hz)'
            assert p.figure.axes[1].get_ylabel() == 'Power relative to peak (dB)'
            assert len(p.axes.collections) == 2 and len(p.region_overlay.get_paths()) == len(before)

        check_restored_plot()
        # Open the HTML before displaying a spectrogram for this WAV.
        other = root / 'other.wav'
        sf.write(other, np.zeros(RATE), RATE, subtype='PCM_16')
        p.set_audio(other, ask=False)
        p.set_audio(wav, ask=False)
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(report), '')):
            p.load_regions();wait()
        check_restored_plot()
        # Opening the same report again reuses the mesh; overlays appear only once.
        existing = p.axes.collections[0]
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(report), '')):
            p.load_regions();wait()
        assert p.task is None and p.axes.collections[0] is existing
        check_restored_plot()

        # Legacy boundary mode still adjusts the neighbor rather than splitting.
        window.regionTable.selectRow(1)
        window.regionEditMode.setCurrentIndex(1)
        assert 'neighboring' in window.regionRangeHint.text()
        window.regionEnd.setValue(44)
        window.applyRegion.click()
        assert len(p.regions) == 5
        assert p.regions[1]['end'] == p.regions[2]['start'] == 44
        validate_regions(p.regions, 190)

        # An unchanged rounded boundary must not create a one-sample sliver.
        edge = 12 + 1 / RATE
        p.regions = [dict(start=0., end=edge, label='silence'),
                     dict(start=edge, end=190., label='speech')]
        p.refresh_all()
        window.regionEditMode.setCurrentIndex(0)
        window.regionTable.clearSelection()
        window.regionTable.selectRow(1)
        window.regionLabel.setCurrentText('mixed')
        window.applyRegion.click()
        assert len(p.regions) == 2 and p.regions[1]['start'] == edge
        assert p.regions[1]['label'] == 'mixed'
        validate_regions(p.regions, 190)
        assert not warnings, warnings
        p.dirty = False
        window.close()
        app.processEvents()
    print('PASS: Qt range playback/edit buttons, unchanged outside labels, invalid-range warnings, audit, downstream invalidation, HTML/session restore, legacy boundary mode and rounded sample edges.')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RangeEdits)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    check_qt_workflow()

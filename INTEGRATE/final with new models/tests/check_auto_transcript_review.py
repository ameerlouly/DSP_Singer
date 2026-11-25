"""Exercise one-click processing through an agreement and a disagreement.

The Qt button runs the real controllers, file I/O, SNR and comparison rules.
Controlled classifier, segmenter and recognizers supply repeatable inputs.
"""
import json
import os
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox
import corpus_review_and_export as corpus
import transcript_agreement_and_io as engines
import segment_review
from speech_corpus_app import MainWindow


def check():
    """Run a batch, inspect the pending choice, choose Qwen and approve its audio."""
    app = QApplication([])
    window = MainWindow(show_welcome=False)
    p = window.processing
    workspace = p.asr
    with tempfile.TemporaryDirectory() as directory, ExitStack() as mocks:
        root = Path(directory)
        source = root / 'source.wav'
        samples = np.random.default_rng(7).normal(0, .05, 16 * 16000)
        sf.write(source, samples, 16000, subtype='PCM_16')
        p.set_audio(source)
        segments = []
        for i, start in enumerate([1, 9]):
            name = f'clip{i}.wav'
            sf.write(root / name, samples[start*16000:(start+6)*16000], 16000, subtype='PCM_16')
            segments.append(dict(file=name, start=start, end=start+6, duration=6, label='speech'))
        (root / 'step4_segments.json').write_text(json.dumps(dict(
            version=2, audio_name=source.name, audio_size=source.stat().st_size,
            duration_seconds=16, segments=segments, unresolved=[])))
        errors = []
        mocks.enter_context(patch.object(QInputDialog, 'getText', return_value=('Test editor', True)))
        mocks.enter_context(patch.object(QMessageBox, 'warning', side_effect=lambda *args: errors.append(args[-1])))
        mocks.enter_context(patch('audio_review_controller.classify_file', return_value=[
            dict(start=0, end=16, label='speech', auto_label='speech', auto_confidence=.812, needs_review=True)]))
        mocks.enter_context(patch('pause_based_segmentation.export_candidates', return_value={'output_dir': root}))
        for key in ['load_qwen_model', 'load_cohere_model']:
            mocks.enter_context(patch.object(engines, key, return_value={}))
        mocks.enter_context(patch.object(engines, 'transcribe_qwen', side_effect=[('مَدْرَسَة', None), ('أهلاً بكم', None)]))
        mocks.enter_context(patch.object(engines, 'transcribe_cohere', side_effect=[('مدرسة', None), ('أهلاً بالجميع', None)]))
        window.autoProcessButton.click()
        deadline = time.monotonic() + 20
        while (p.task is not None or p.auto_stage) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert not errors, errors
        assert not p.task and not p.auto_stage
        assert window.regionTable.item(0, 4).text() == '0.812 (81.2%)'
        assert '0.812 (81.2%)' in next(root.rglob('*_regions.html')).read_text(encoding='utf-8')
        assert len(p.snr_results) == len(workspace.records) == 2
        first, second = workspace.records
        assert first['status'] == 'accepted' and first['final_transcript'] == 'مدرسة'
        assert second['status'] == 'needs_review' and not second['final_transcript'] and not second['automatic_choice']
        assert window.asrTable.currentRow() == 1 and not window.asrFinalText.toPlainText()
        assert all(s['status'] == 'pending' for s in p.session['segments'])
        rows = list(corpus.review_rows(workspace.records, []))
        assert rows[0]['model_A_transcript'] == rows[0]['model_B_transcript'] == rows[0]['final_transcript'] == 'مدرسه'
        assert rows[1]['final_transcript'] == ''
        window.asrUseA.click()
        assert second['final_transcript'] == 'أهلاً بكم'
        assert list(corpus.review_rows(workspace.records, []))[1]['final_transcript'] == 'اهلا بكم'
        segment_review.approve_segment(p.session, 1, 'Test editor')
        workspace.sync_current()
        chosen = workspace.records[1]
        assert chosen['human_override'] and chosen['segmentation_status'] == 'resolved' and not corpus.is_preview(chosen)
        assert chosen['final_transcript'] == 'أهلاً بكم'
        p.dirty = False
        window.close()
    print('PASS: automatic pipeline, manual transcript choice, region confidence, normalized reports and boundary approval.')


if __name__ == '__main__':
    check()

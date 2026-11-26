"""Run auto processing through clips whose model outputs agree and disagree.

The Qt controls and file operations are real. Controlled model outputs make it
possible to check exactly when the listener must choose the transcript.
"""

import json
import os
import subprocess
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

    def wait_for_idle():
        """Let Qt deliver worker results and start each queued stage."""
        deadline = time.monotonic() + 25
        while window.worker is not None or p.task is not None or p.auto_stage:
            assert time.monotonic() < deadline, f'Auto process stalled during {p.auto_stage}'
            app.processEvents()
            time.sleep(.01)

    with tempfile.TemporaryDirectory() as directory, ExitStack() as mocks:
        root = Path(directory)
        source = root / 'source.wav'
        samples = np.random.default_rng(7).normal(0, .05, 24 * 16000)
        sf.write(source, samples, 16000, subtype='PCM_16')
        video = root / 'lecture.mkv'
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(source), '-c:a', 'pcm_s16le', str(video)], check=True)
        window.channelBox.setCurrentText('Stereo')

        # Fix the proposed boundaries so one run exercises all three transcript decisions.
        def candidates(wav, regions, folder, log):
            from output_folders import new_run_folder
            target = new_run_folder(folder)
            segments = []
            for i, start in enumerate([1, 9]):
                name = f'clip{i}.wav'
                sf.write(target / name, samples[start*16000:(start+6)*16000], 16000, subtype='PCM_16')
                segments.append(dict(file=name, start=start, end=start+6, duration=6, label='speech'))
            (target / 'step4_segments.json').write_text(json.dumps(dict(
                version=2, audio_name=Path(wav).name, audio_size=Path(wav).stat().st_size,
                duration_seconds=24, segments=segments, unresolved=[dict(start=17, end=23), dict(start=23, end=23.5)])))
            return {'output_dir': target}
        errors = []
        mocks.enter_context(patch.object(QInputDialog, 'getText', return_value=('Test editor', True)))
        mocks.enter_context(patch.object(QMessageBox, 'warning', side_effect=lambda *args: errors.append(args[-1])))
        mocks.enter_context(patch('audio_review_controller.classify_file', return_value=[
            dict(start=0, end=24, label='speech', auto_label='speech', auto_confidence=.812, needs_review=True)]))
        mocks.enter_context(patch('pause_based_segmentation.export_candidates', side_effect=candidates))
        mocks.enter_context(patch.object(engines, 'load_qwen_model', return_value={}))
        mocks.enter_context(patch.object(engines, 'load_cohere_model', return_value=dict(
            engine=engines.COHERE_ENGINE, model_name=engines.DEFAULT_COHERE_MODEL)))
        mocks.enter_context(patch.object(engines, 'transcribe_qwen', side_effect=[('مَدْرَسَة', None), ('أهلاً بكم', None), ('واحد اثنان ثلاثة أربعة خمسة', None)]))
        mocks.enter_context(patch.object(engines, 'transcribe_cohere', side_effect=[('مدرسة', None), ('أهلاً بالجميع', None), ('شمس قمر بحر أرض نهر', None)]))
        # A bad source must end the chain before region labeling can start.
        broken = root / 'broken.mkv'
        broken.write_bytes(b'invalid video')
        window.videoPathInput.setText(str(broken))
        window.autoProcessButton.click()
        wait_for_idle()
        assert errors and p.audio is None and p.auto_stage is None and not p.task
        errors.clear()

        # Stub the network transfer only. FFmpeg extracts and verifies the returned file.
        window.videoPathInput.clear()
        window.urlInput.setText('https://example.org/lecture')
        window.outputFolderInput.setText(str(root))
        download = mocks.enter_context(patch('speech_corpus_app.download_video', return_value=video))
        window.autoProcessButton.click()
        wait_for_idle()
        assert not errors, errors
        download.assert_called_once()
        assert not p.task and not p.auto_stage
        assert window.regionTable.item(0, 4).text() == '0.812 (81.2%)'
        assert '0.812 (81.2%)' in next(root.rglob('*_regions.html')).read_text(encoding='utf-8')
        assert len(p.snr_results) == len(workspace.records) == 3
        assert p.audio.is_relative_to(root / 'Extract audio') and p.audio.with_suffix('.html').is_file()
        assert np.array_equal(sf.read(p.audio, dtype='int16')[0], sf.read(source, dtype='int16')[0])
        assert sf.info(p.audio).subtype == 'PCM_16' and sf.info(p.audio).samplerate == 16000
        assert p.snr_report.is_relative_to(root / 'Audio quality SNR')
        assert 'Pending listening review' in p.snr_report.read_text(encoding='utf-8')
        assert p.snr_report.with_suffix('.csv').is_file()
        quality = json.loads(p.snr_report.with_suffix('.json').read_text(encoding='utf-8'))
        saved = json.loads(workspace.autosave_path.read_text(encoding='utf-8'))['asr']['records']
        assert [r['id'] for r in quality['segments']] == [r['segment_id'] for r in saved]
        assert all(r['energy_snr_db'] is not None and r['wada_snr_db'] is not None for r in quality['segments'])
        assert [r['status'] for r in saved] == ['accepted', 'needs_review', 'difficult']
        assert all(r['asr_a_done'] and r['asr_b_done'] for r in saved)
        first, second, third = workspace.records
        assert third['segmentation_status'] == 'unresolved' and not third['final_transcript']
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
        # Reuse the extracted audio and model results, but save this run in its own folders.
        previous_checkpoint = workspace.autosave_path
        retained_text = second['final_transcript']
        mocks.enter_context(patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes))
        with patch('speech_corpus_app.extract_audio') as extract_again, patch.object(engines, 'load_qwen_model') as reload_a, patch.object(engines, 'load_cohere_model') as reload_b:
            window.autoProcessButton.click()
            wait_for_idle()
            assert not p.auto_stage and not p.task and not errors, errors
            extract_again.assert_not_called()
            reload_a.assert_not_called()
            reload_b.assert_not_called()
        assert workspace.autosave_path != previous_checkpoint
        saved_again = json.loads(workspace.autosave_path.read_text(encoding='utf-8'))
        assert saved_again['asr']['records'][1]['final_transcript'] == retained_text
        segment_review.approve_segment(p.session, 1, 'Test editor')
        workspace.sync_current()
        wait_for_idle()
        assert p.task is None
        chosen = workspace.records[1]
        assert chosen['human_override'] and chosen['segmentation_status'] == 'resolved' and not corpus.is_preview(chosen)
        assert chosen['final_transcript'] == 'أهلاً بكم'
        # A standalone WAV can start the same flow. Keep Qwen's work if Cohere cannot load.
        p.dirty = False
        window.videoPathInput.clear()
        p.set_audio(source)
        with patch.object(engines, 'transcribe_qwen', return_value=('مرحبا', None)), patch.object(
            engines, 'load_cohere_model', side_effect=RuntimeError('model unavailable')
        ), patch('speech_corpus_app.extract_audio') as extract_again:
            window.autoProcessButton.click()
            wait_for_idle()
            extract_again.assert_not_called()
        assert not errors and not p.auto_stage and not p.task
        partial = json.loads(workspace.autosave_path.read_text(encoding='utf-8'))
        assert len(partial['asr']['records']) == 3
        for row in partial['asr']['records']:
            assert row['asr_a_done'] and not row['asr_b_done'] and not row['final_transcript']
            assert row['asr_errors']['b'] == 'model unavailable'
        assert 'model unavailable' in window.activityLog.toPlainText() and p.snr_report.is_file()
        p.dirty = False
        window.close()
    print('PASS: automatic pipeline, manual transcript choice, region confidence, normalized reports and boundary approval.')


if __name__ == '__main__':
    check()

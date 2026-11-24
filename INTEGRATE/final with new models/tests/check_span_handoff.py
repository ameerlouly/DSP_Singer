"""Qt workflow with real PCM and controlled, independent Arabic recognizers.

Test guide
----------
wait
    Process Qt events until the active background task finishes.
check
    Drive the resolved and unresolved handoff through quality review, ASR and export.
"""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication, QMessageBox
import transcript_agreement_and_io as asr
import corpus_review_and_export as corpus
import clip_quality_assessment as snr
import segment_review as review
from speech_corpus_app import MainWindow


def wait(app, processing):
    deadline = time.monotonic() + 20
    while processing.task is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert processing.task is None


def check():
    app = QApplication([])
    window = MainWindow(show_welcome=False)
    processing = window.processing
    workspace = processing.asr
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / 'audio.wav'
        samples = np.random.default_rng(4).normal(0,.03,45*16000).astype('float32')
        sf.write(source, samples, 16000, subtype='PCM_16')
        pcm = sf.read(source, dtype='float32')[0]
        payload = dict(audio_name=source.name, audio_size=source.stat().st_size, duration_seconds=45,
                       segments=[], unresolved=[dict(start=0,end=22.000125),dict(start=35,end=42),dict(start=34,end=34.35)],
                       not_in_clips_or_unresolved=[dict(start=43,end=44)])
        path = root / 'segments.json'
        path.write_text(json.dumps(payload))
        processing.set_audio(source)
        processing.ui.reviewer.setText('Reviewer')
        processing.session = review.load_session(path, source)
        processing.session['unresolved'][-1].update(label='silence', status='reviewed')
        window.calculateSNR.click()
        wait(app,processing)
        rows=processing.snr_results
        assert len(rows)==3 and all(r['segmentation_status']=='unresolved' for r in rows)
        assert all(window.snrTable.item(i,3).text()=='Unresolved' for i in range(3))
        assert rows[0]['noise_source']=='quiet_frame_fallback'
        assert rows[0]['wada_snr_db'] is not None and rows[0]['energy_snr_db'] is not None
        report = snr.export_report(rows,20,root,justification='Controlled test')
        processing.match_snr(json.loads(Path(report[1]).read_text())['segments'])
        with Path(report[2]).open(encoding='utf-8-sig') as handle:
            assert all(r['segmentation_status']=='unresolved' for r in csv.DictReader(handle))
        window.asrSync.click()
        assert window.asrTable.rowCount() == 3 and window.finalTable.rowCount() == 3
        assert all(window.asrTable.item(i,4).text()=='Unresolved' for i in range(3))
        assert rows[0]['energy_snr_db']==workspace.records[0]['energy_snr_db']
        long, gap, short = workspace.records
        windows = list(corpus.preview_audio(long))
        assert len(windows) >= 2 and all(80000 <= len(w) <= 240000 for w in windows)
        np.testing.assert_array_equal(np.concatenate(windows), pcm[:round(22.000125*16000)])
        padded, = corpus.preview_audio(gap)
        raw = pcm[34*16000:round(34.35*16000)]
        np.testing.assert_array_equal(raw, padded[:len(raw)])
        assert len(padded) == 80000 and not padded[len(raw):].any()
        calls = []
        def run_a(model, audio):
            calls.append(('a', len(audio)))
            return 'اهلا', .8
        def run_b(model, audio, rate):
            calls.append(('b', len(audio)))
            return 'أهلا', None
        with patch.object(asr,'load_qwen_model',return_value=object()), patch.object(asr,'load_cohere_model',return_value=object()), patch.object(asr,'transcribe_qwen',side_effect=run_a), patch.object(asr,'transcribe_cohere',side_effect=run_b):
            window.asrPending.click()
            wait(app, processing)
        assert [n for k,n in calls if k=='a'] == [n for k,n in calls if k=='b']
        long, gap, short = workspace.records
        assert long['transcript_a'].split() == ['اهلا']*len(windows)
        assert all(r['status']=='accepted' and r['segmentation_status']=='unresolved' for r in workspace.records)
        window.asrTable.selectRow(2)
        window.asrUseB.click()
        snapshot = json.loads(json.dumps(workspace.payload()))
        workspace.restore(snapshot)
        assert workspace.records == snapshot['records'] and workspace.records[2]['human_override']
        assert window.finalTable.rowCount() == 3
        try:
            corpus.export_corpus(workspace.records, {}, [], root, True)
        except ValueError as exc:
            assert 'Approve a Part 4 clip' in str(exc)
        else:
            raise AssertionError('A preview entered the corpus')
        i = review.add_from_unresolved(processing.session,1,35,42,'mixed','Reviewer')
        review.approve_segment(processing.session,i,'Reviewer')
        window.calculateSNR.click()
        wait(app,processing)
        assert [r['segmentation_status'] for r in processing.snr_results]==['unresolved','unresolved','resolved']
        legacy=[dict(processing.snr_results[2])]
        legacy[0].pop('segmentation_status')
        processing.match_snr(legacy)
        with patch.object(QMessageBox,'question',side_effect=AssertionError('Exact promotion lost its preview')):
            workspace.sync_current()
        approved, = [r for r in workspace.final_records() if not corpus.is_preview(r)]
        assert approved['original_approval']=='approved' and approved['final_transcript']=='أهلا'
        assert approved['human_override'] and approved['asr_a_done'] and approved['asr_b_done']
        assert approved['transcribed_from_preview']==short['segment_id']
        assert len(workspace.records)==3 and window.finalTable.rowCount()==3
        window.finalTable.selectRow(workspace.final_records().index(approved))
        assert workspace.selected(window.finalTable)==[approved]
        result = corpus.export_corpus(workspace.records, {}, [], root, True)
        assert result['partial'] and result['clips']==1
        i = review.add_from_unresolved(processing.session,0,7,14,'speech','Reviewer')
        review.approve_segment(processing.session,i,'Reviewer')
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            workspace.sync_current()
        altered = [r for r in workspace.records if r['start']==7][0]
        assert not altered['asr_a_done'] and not altered['asr_b_done']
        leftovers = [(r['start'],r['end']) for r in workspace.records if corpus.is_preview(r)]
        assert (0,7) in leftovers and (14,22.000125) in leftovers
    print('PASS: approved and unresolved SNR + dual ASR, original PCM, fixed agreement, reports, session restore and corpus gates.')


if __name__ == '__main__':
    check()

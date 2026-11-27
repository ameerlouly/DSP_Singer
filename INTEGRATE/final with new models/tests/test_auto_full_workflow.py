"""Auto Process stage orchestration, approval provenance and export destination."""
import ast
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
import soundfile as sf
import segment_review as review
import clip_quality_assessment as snr
import corpus_review_and_export as corpus
from output_folders import step_folder
from test_batch_approval import candidate, session
from test_portable_review_html import AudioURLs

ROOT = Path(__file__).resolve().parents[1]

def method(file, name, env):
    tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[func],type_ignores=[]), file, 'exec'),env)
    return env[name]

class FullAutoTests(unittest.TestCase):
    def test_approve_valid_candidates_without_claiming_human_listening(self):
        data=session()
        before=copy.deepcopy(data)
        w=SimpleNamespace(session=data, reviewer=lambda:'Operator', modified_session=Mock(),
                          window=SimpleNamespace(append_log=Mock()), status=Mock())
        method('audio_review_controller.py','auto_approve_candidates',dict(review=review))(w)
        for i in (0,1,2,8):
            clip=data['segments'][i]
            self.assertEqual(clip['status'],'approved')
            self.assertEqual(clip['pdf_boundary_review']['approval_method'],'auto_process')
            self.assertFalse(clip['pdf_boundary_review']['human_listening_confirmed'])
            review.require_pdf_review(data,clip)
        self.assertEqual([data['segments'][i] for i in (3,4,5,6,7)],[before['segments'][i] for i in (3,4,5,6,7)])
        self.assertEqual(data['unresolved'],before['unresolved'])
        self.assertFalse(data['analyze_pending_candidates'])
        self.assertEqual(data['changes'], [])
        review.approve_segments(data,[0],'Human')
        self.assertTrue(data['segments'][0]['pdf_boundary_review']['human_listening_confirmed'])

    def test_overlap_rejects_atomically(self):
        data=dict(source=dict(duration_seconds=30),segments=[candidate(1,0,8),candidate(2,5,8)],changes=[])
        before=copy.deepcopy(data)
        w=SimpleNamespace(session=data,reviewer=lambda:'Operator')
        with self.assertRaisesRegex(ValueError,'overlap'):
            method('audio_review_controller.py','auto_approve_candidates',dict(review=review))(w)
        self.assertEqual(data,before)

    def test_stage_order_and_cancel(self):
        f=method('audio_review_controller.py','advance_auto_process',dict(exception_details=lambda e:(str(e),'')))
        for cancel in (False,True):
            with self.subTest(cancel=cancel):
                calls=[]
                def note(name):return lambda *a,**k:calls.append(name)
                a=SimpleNamespace(records=[dict(label='speech')],sync_current=note('sync'),transcribe=note('transcribe'),
                                  stop_event=SimpleNamespace(is_set=lambda:False),export_automatic=lambda:(calls.append('export') or not cancel))
                w=SimpleNamespace(auto_stage='download',audio='source.wav',reviewer=lambda:'Editor',source_folder=lambda:ROOT,task=object(),asr=a,
                    window=SimpleNamespace(start_operation=note('extract'),navigationList=SimpleNamespace(setCurrentRow=Mock())),
                    analyze=note('label'),generate_segments=note('segments'),auto_approve_candidates=note('approve'),calculate_snr=note('snr'),
                    save_snr=note('save_snr'),status=Mock(),task_error=Mock())
                for stage in ('extract','label','segments','snr','save_snr','sync','asr'):
                    f(w);self.assertEqual(w.auto_stage,stage)
                f(w)
                self.assertEqual(calls,['extract','label','segments','approve','snr','save_snr','sync','transcribe','export'])
                if cancel:
                    self.assertIsNone(w.auto_stage);self.assertIn('cancelled',w.status.call_args.args[0])
                else:
                    self.assertEqual(w.auto_stage,'export');f(w);self.assertIsNone(w.auto_stage)
                w.task_error.assert_not_called()

    def test_destination_selected_only_once_and_partial_forced(self):
        dialog=Mock(return_value='/chosen/drive')
        f=method('transcription_review_controller.py','export_automatic',dict(QFileDialog=SimpleNamespace(getExistingDirectory=dialog),step_folder=step_folder))
        w=SimpleNamespace(check_export_current=Mock(),p=SimpleNamespace(source_folder=lambda:ROOT),w=None,
                          ui=SimpleNamespace(allowPartialExport=Mock()),start_corpus_export=Mock())
        self.assertTrue(f(w))
        dialog.assert_called_once();w.ui.allowPartialExport.setChecked.assert_called_once_with(True)
        w.start_corpus_export.assert_called_once_with(Path('/chosen/drive/Review and export/Automatic partial export'),True,automatic=True)
        w.start_corpus_export.reset_mock();dialog.return_value=''
        self.assertFalse(f(w));w.start_corpus_export.assert_not_called()

    def test_snr_uses_only_measured_anchors_for_auto_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.wav'
            samples=np.random.default_rng(12).normal(0,.01,16*16000)
            sf.write(path,samples,16000,subtype='PCM_16')
            clip=candidate(1,0,8,flagged=True);clip['candidate_file']='clip.wav'
            clip['original'].update(left_padding=0.,right_padding=.15)
            sf.write(Path(tmp)/'clip.wav',samples[:8*16000],16000,subtype='PCM_16')
            data=dict(source=dict(duration_seconds=16,audio_name=path.name,audio_size=path.stat().st_size),candidate_root=tmp,segments=[clip],changes=[],unresolved=[],pause_intervals=[dict(start=3.,end=3.4)])
            review.approve_segments(data,[0],'Operator',approval_method='auto_process')
            rows=snr.session_records(data,path)
            self.assertEqual(rows[0]['pause_intervals'],[[7.85,8.],[3.,3.4]])
            self.assertTrue(rows[0]['automatic_boundary_approval'])
            results=snr.calculate(rows,20)
            self.assertEqual(results[0]['noise_source'],'estimated_part4_pause')
            self.assertEqual(results[0]['segmentation_status'],'resolved')
            reports=snr.export_report(results,20,Path(tmp)/'snr',provisional=True)
            self.assertTrue(all(Path(p).is_file() for p in reports))

    def test_auto_approved_and_unresolved_export_to_separate_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); project=root/'project';project.mkdir();source=project/'source.wav'
            samples=np.random.default_rng(13).integers(-2000,2000,26*16000,dtype=np.int16)
            sf.write(source,samples,16000,subtype='PCM_16')
            clip=candidate(1,0,8,flagged=True);clip['candidate_file']='clip.wav'
            sf.write(project/'clip.wav',samples[:8*16000],16000,subtype='PCM_16')
            data=dict(source=dict(duration_seconds=26,audio_name=source.name,audio_size=source.stat().st_size),candidate_root=str(project),segments=[clip],changes=[],pause_intervals=[],unresolved=[dict(id='unresolved-1',original=dict(start=8,end=26),label='unclassified',status='pending')])
            review.approve_segments(data,[0],'Operator',approval_method='auto_process')
            qualities=snr.calculate(snr.session_records(data,source),20)
            w=SimpleNamespace(audio=source,session=data,duration=26,snr_results=qualities,snr_snapshot=copy.deepcopy(data),require_approved=lambda:None)
            records=corpus.from_processing(w)
            self.assertEqual([r['segmentation_status'] for r in records],['resolved','unresolved'])
            self.assertTrue(all(corpus.can_transcribe(r) for r in records))
            # Pending ASR results are allowed in partial export without inventing final text.
            for r in records:r.update(transcript_a='مرحبا',asr_a_done=True)
            out=Path(corpus.export_corpus(records,dict(audio=str(source),source_review=dict(step4=data,snr=qualities)),[],root/'external'/'Review and export',True,automatic=True)['output_dir'])
            self.assertTrue(out.is_relative_to(root/'external'))
            self.assertTrue((out/'review_compatible.csv').is_file())
            self.assertEqual(json.loads((out/'corpus.json').read_text())['metadata']['export_mode'],'automatic_partial')
            self.assertEqual({p.name for p in out.glob('*.csv')},{'review_compatible.csv'})
            self.assertEqual([r['segmentation_status'] for r in json.loads((out/'corpus.json').read_text(encoding='utf-8'))['segments']],['resolved','unresolved'])
            urls=AudioURLs(out/'corpus_report.html').urls
            self.assertEqual(len(urls),2)
            source.unlink()
            for r,url in zip(records,urls):
                saved,rate=sf.read(out/url,dtype='int16');self.assertEqual(rate,16000)
                np.testing.assert_array_equal(saved,samples[int(r['start']*16000):int(r['end']*16000)])

if __name__=='__main__':unittest.main()

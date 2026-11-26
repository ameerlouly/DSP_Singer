"""Quality warnings never block otherwise complete export; missing SNR still does."""
import ast
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import soundfile as sf
import clip_quality_assessment as snr
import corpus_review_and_export as corpus

ROOT = Path(__file__).resolve().parents[1]

class OptionalQualityTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name); self.source=self.root/'audio.wav'
        sf.write(self.source,np.zeros(8*16000,dtype=np.int16),16000,subtype='PCM_16')
        self.analysis=snr.decision(dict(analysis_version=2,content_label='mixed',energy_snr_db=25.,wada_snr_db=15.,
            energy_details=dict(status='ok',noise_source='quiet_frame_fallback'),wada_details=dict(status='ok'),
            audio_sha256='controlled',override_by=''),20,'conservative','')
        self.r=dict(segment_id='s1',start=0,end=8,duration=8,label='mixed',source_mode='continuous_source',
            segmentation_status='resolved',source_origin=str(self.source),source_spans=[dict(start=0,end=8)],
            audio_pieces=[dict(path=str(self.source),start=0,end=8)],asr_a_done=True,asr_b_done=True,
            status='accepted',automatic_status='accepted',transcript_a='اهلا',transcript_b='اهلا',final_transcript='اهلا')
        self.r.update(snr.corpus_fields(self.analysis))
        self.r['audio_signature']=corpus.audio_signature(self.r);self.r['snr_audio_signature']=self.r['audio_signature']

    def export(self,partial=False):
        with patch.object(corpus,'compare'):
            return corpus.export_corpus([self.r],{},[],self.root/'out',partial)

    def test_warnings_and_missing_justification_do_not_require_partial_export(self):
        self.assertEqual(self.analysis['quality'],'noisy')
        self.assertTrue(self.analysis['quality_warning']);self.assertFalse(self.analysis['review_required'])
        # Saved V17 records can still have the old mandatory-review flag.
        self.r['snr_review_required']=True
        self.r['snr_analysis'].update(review_required=True,automatic_review_required=True,review_status='Review needed')
        before=copy.deepcopy(self.r)
        result=self.export();self.assertFalse(result['partial']);self.assertEqual(self.r,before)
        out=Path(result['output_dir'])
        meta=json.loads((out/'corpus.json').read_text())['metadata']
        self.assertTrue(meta['snr_quality_review_optional']);self.assertFalse(meta['snr_threshold_justification_provided'])
        record=json.loads((out/'corpus_audit.json').read_text())['segments'][0]
        self.assertFalse(record['snr_review_required']);self.assertTrue(record['snr_analysis']['quality_warning'])
        self.assertTrue(record['snr_analysis']['review_reasons']);self.assertEqual(record['clean_noisy'],'noisy')
        self.assertTrue((out/'corpus_report.html').exists());self.assertTrue((out/'review_compatible.csv').exists())

    def test_manual_correction_is_exported_without_review_gate(self):
        snr.set_quality(self.analysis,'clean','Editor','Listening correction')
        self.r.update(snr.corpus_fields(self.analysis))
        out=Path(self.export()['output_dir'])
        record=json.loads((out/'corpus_audit.json').read_text())['segments'][0]
        self.assertEqual(record['clean_noisy'],'clean');self.assertEqual(record['snr_analysis']['automatic_quality'],'noisy')
        self.assertEqual(record['snr_analysis']['override_by'],'Editor')
        with (out/'review_compatible.csv').open(encoding='utf-8-sig') as f:
            self.assertEqual(next(csv.DictReader(f))['quality'],'clean')

    def test_invalid_measurements_and_asr_still_require_partial(self):
        for changes in ({'snr_db':None},{'snr_audio_signature':'stale'},{'snr_analysis':{'analysis_version':1}},
                        {'clean_noisy':'unavailable'},{'status':'needs_review'}):
            with self.subTest(changes=changes), patch.dict(self.r,changes):
                with self.assertRaisesRegex(ValueError,'current usable SNR'):
                    self.export()
                self.assertTrue(self.export(True)['partial'])

    def test_reopen_legacy_session_keeps_transcripts_and_source_signature(self):
        # The actual restore method must not require ASR reruns solely because
        # the quality-review flags changed in this version.
        old=copy.deepcopy(self.analysis)
        old.update(id='s1',source_start=0,source_end=8,review_required=True,
                   automatic_review_required=True,review_status='Review needed')
        old.pop('quality_warning',None)
        saved=[old]
        processing=SimpleNamespace(audio=self.source,session={'segments':[]},
            applied_threshold=20,snr_policy='conservative',snr_justification='',
            snr_results=snr.reclassify(saved,20,'conservative',''))
        workspace=SimpleNamespace(clear=Mock(),p=processing,refresh=Mock(),
            ui=SimpleNamespace(qwenModel=Mock(),cohereModel=Mock()))
        record=copy.deepcopy(self.r);record['snr_review_required']=True
        state=dict(signature=corpus.source_signature(processing,saved),audit_log=[])
        tree=ast.parse((ROOT/'transcription_review_controller.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='restore')
        env=dict(snr=snr,corpus=corpus,engines=SimpleNamespace(DEFAULT_QWEN_MODEL='qwen',DEFAULT_COHERE_MODEL='cohere'))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<restore>','exec'),env)
        env['restore'](workspace,state,saved_snr=saved,prepared=([record],0,0))
        self.assertEqual(workspace.signature,corpus.source_signature(processing))
        self.assertEqual(workspace.records[0]['transcript_a'],self.r['transcript_a'])
        self.assertFalse(workspace.records[0]['snr_review_required'])

    def test_table_matches_rows_without_review_column(self):
        ui=ET.parse(ROOT/'main_window.ui').find('.//widget[@name="snrTable"]')
        columns=[c.find('property/string').text for c in ui.findall('column')]
        self.assertEqual(len(columns),12);self.assertNotIn('Review',columns);self.assertIn('Quality',columns)
        tree=ast.parse((ROOT/'audio_review_controller.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='snr_cells')
        method.decorator_list=[]; env={};exec(compile(ast.Module(body=[method],type_ignores=[]),'<table>','exec'),env)
        self.analysis.update(id='s1',source_start=0,source_end=8)
        row=env['snr_cells'](self.analysis);self.assertEqual(len(row),len(columns));self.assertEqual(row[columns.index('Quality')],'noisy')
        self.assertIn('do not agree',row[columns.index('Notes')])

if __name__=='__main__':unittest.main()

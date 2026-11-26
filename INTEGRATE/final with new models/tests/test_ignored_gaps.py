"""Estimated gaps stay in the audit without requiring approval or creating clips."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import segment_review as review
import corpus_review_and_export as corpus
from test_portable_review_html import AudioURLs


class GapExportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source.wav'
        self.samples = np.random.default_rng(3).integers(-32000, 32000, 30*16000, dtype=np.int16)
        sf.write(self.source, self.samples, 16000, subtype='PCM_16')
        payload = dict(audio_name=self.source.name, audio_size=self.source.stat().st_size,
                       duration_seconds=30, segments=[], unresolved=[],
                       not_in_clips_or_unresolved=[dict(start=8, end=10, estimated_pause=True)])
        path = self.root/'step4_segments.json'
        path.write_text(json.dumps(payload))
        self.session = review.load_session(path, self.source)
        self.gap = self.session['unresolved'][0]
        self.record = dict(segment_id='auto-00001',start=0,end=8,duration=8,label='speech',
                           source_mode='continuous_source',segmentation_status='resolved',
                           source_origin=str(self.source),source_spans=[dict(start=0,end=8)],
                           status='accepted',final_transcript='اهلا',transcript_a='اهلا',transcript_b='اهلا',
                           asr_a_done=True,asr_b_done=True,clean_noisy='clean',snr_db=24,
                           snr_analysis={'analysis_version':2},snr_review_required=False,
                           snr_threshold_justification='Controlled test',
                           audio_pieces=[dict(path=str(self.source),start=0,end=8)])
        self.record['audio_signature'] = corpus.audio_signature(self.record)
        self.record['snr_audio_signature'] = self.record['audio_signature']

    def manifest(self):
        return dict(audio=str(self.source),source_review=dict(step4=self.session),
                    pending_source_candidates=0,unresolved_source_spans=1)

    def test_gap_hidden_in_fresh_and_reopened_session(self):
        self.assertEqual(review.visible_unresolved(self.session), [])
        self.assertEqual(list(review.downstream_unresolved(self.session, [])), [])
        self.assertEqual(self.gap['status'], 'pending')
        saved=self.root/'old_review.json';saved.write_text(json.dumps(self.session))
        reopened=review.load_session(saved,self.source)
        self.assertEqual(review.visible_unresolved(reopened),[])
        self.assertEqual(reopened['unresolved'],self.session['unresolved'])

    def test_strict_export_no_gap_approval_even_with_legacy_count(self):
        before=copy.deepcopy(self.session)
        # ASR is already completed. Isolate the export gate from optional jiwer.
        with patch.object(corpus,'compare'):
            result=corpus.export_corpus([self.record],self.manifest(),[],self.root/'out')
        self.assertFalse(result['partial']);self.assertEqual(result['clips'],1)
        out=Path(result['output_dir']);meta=json.loads((out/'corpus.json').read_text())['metadata']
        self.assertEqual(meta['unresolved_source_spans'],0)
        self.assertEqual(meta['pending_boundary_segment_ids'],[])
        self.assertEqual(json.loads((out/'source_review.json').read_text())['step4'],before)
        self.assertEqual(self.session,before)
        urls=AudioURLs(out/'corpus_report.html').urls;self.assertEqual(len(urls),1)
        pcm,rate=sf.read(out/urls[0],dtype='int16');self.assertEqual(rate,16000)
        np.testing.assert_array_equal(pcm,self.samples[:8*16000])

    def test_genuine_unresolved_still_blocks_complete_export(self):
        self.session['unresolved'].append(dict(id='unresolved-00001', category='unresolved',
            label='unclassified',status='pending',original=dict(start=10,end=22)))
        self.assertEqual([r[1]['id'] for r in review.visible_unresolved(self.session)],['unresolved-00001'])
        with patch.object(corpus,'compare'):
            with self.assertRaisesRegex(ValueError,'source candidates or unresolved'):
                corpus.export_corpus([self.record],self.manifest(),[],self.root/'out')
            result=corpus.export_corpus([self.record],self.manifest(),[],self.root/'out',True)
        self.assertEqual(result['clips'],2);self.assertTrue(result['partial'])
        meta=json.loads((Path(result['output_dir'])/'corpus.json').read_text())['metadata']
        self.assertEqual(meta['pending_boundary_segment_ids'],['unresolved-00001'])

    def test_old_explicit_voice_gap_preserved(self):
        self.gap.update(label='mixed')
        self.assertEqual(len(review.visible_unresolved(self.session)),1)
        self.assertEqual(len(list(review.downstream_unresolved(self.session,[]))),1)
        with patch.object(corpus,'compare'):
            with self.assertRaisesRegex(ValueError,'pending'):
                corpus.export_corpus([self.record],self.manifest(),[],self.root/'out')

    def test_legacy_unclassified_gap_preview_filtered_from_report_and_export(self):
        preview=dict(segment_id='gap-00001',start=8,end=10,label='unclassified',
                     segmentation_status='unresolved',source_mode='unresolved_preview',
                     source_origin=str(self.source),source_metadata=copy.deepcopy(self.gap))
        self.assertEqual(corpus.timeline_records([preview],self.session,self.source),[])
        self.assertEqual(corpus.timeline_records([preview]),[])
        with patch.object(corpus,'compare'):
            result=corpus.export_corpus([self.record,preview],self.manifest(),[],self.root/'out')
        self.assertFalse(result['partial']);self.assertEqual(result['clips'],1)


if __name__=='__main__':unittest.main()

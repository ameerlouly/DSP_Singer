"""The additional export matches the external CSV header and preserves raw text."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import corpus_review_and_export as corpus
from test_portable_review_html import AudioURLs

EXPECTED = ['start','end','label','snr','quality','model_A_transcript',
            'model_B_transcript','final_transcript','status','review_status',
            'automatic_status','segment_id','audio_file','changed_by','changed_at','change_log']

class CompatibleReviewCSVTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.source = self.root/'source.wav'
        sf.write(self.source, np.zeros(30*16000, dtype=np.int16),16000,subtype='PCM_16')

    def record(self, ident, a, b, state='resolved'):
        record = dict(segment_id=ident,start=a,end=b,duration=b-a,label='speech',
                      source_mode={'resolved':'continuous_source','pending':'candidate_preview','unresolved':'unresolved_preview'}[state],
                      segmentation_status=state,source_origin=str(self.source),
                      source_spans=[dict(start=a,end=b)],audio_pieces=[dict(path=str(self.source),start=a,end=b)],
                      status='accepted' if state=='resolved' else 'needs_review',
                      automatic_status='needs_review',asr_a_done=True,asr_b_done=True,
                      transcript_a='أهلاً، يا فاطمة',transcript_b='أهلا يا فاطمة',
                      final_transcript='أهلاً، يا فاطمة' if state=='resolved' else '',
                      clean_noisy='clean',snr_db=23.5,snr_analysis={'analysis_version':2},
                      snr_review_required=False,snr_threshold_justification='Controlled test')
        record['audio_signature']=corpus.audio_signature(record)
        record['snr_audio_signature']=record['audio_signature']
        return record

    def verify(self, result, records):
        out=Path(result['output_dir']); path=out/corpus.COMPATIBLE_REVIEW_FILENAME
        self.assertTrue(path.read_bytes().startswith(b'\xef\xbb\xbf'))
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); self.assertEqual(reader.fieldnames,EXPECTED); rows=list(reader)
        self.assertFalse((out/'review.csv').exists())
        self.assertEqual({p.name for p in out.glob('*.csv')}, {'corpus.csv', corpus.COMPATIBLE_REVIEW_FILENAME})
        with (out/'corpus.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); self.assertEqual(reader.fieldnames,EXPECTED+['segmentation_status']); detailed=list(reader)
        self.assertEqual(len(rows),len(records))
        self.assertEqual([r['audio_file'] for r in rows],AudioURLs(out/'corpus_report.html').urls)
        self.assertEqual([r['audio_file'] for r in rows],[r['audio_file'] for r in detailed])
        for r,original,d in zip(rows,records,detailed):
            self.assertEqual(r['segment_id'],original['segment_id']); self.assertEqual(r['status'],original['status'])
            self.assertEqual(r['model_A_transcript'],original['transcript_a']); self.assertEqual(r['model_B_transcript'],original['transcript_b'])
            self.assertEqual(r['final_transcript'],original['final_transcript']); self.assertEqual(r['snr'],'23.5')
            self.assertEqual(d['segmentation_status'],original['segmentation_status'])
            self.assertFalse(Path(r['audio_file']).is_absolute()); self.assertTrue((out/r['audio_file']).is_file())
            self.assertEqual((out/r['audio_file']).with_suffix('.txt').read_text(encoding='utf-8'),r['final_transcript'])
            self.assertIsInstance(json.loads(r['change_log']),list)
        self.assertEqual(json.loads((out/'corpus.json').read_text())['metadata']['compatible_review_csv'],path.name)
        return rows

    def test_manual_complete(self):
        record=self.record('s1',0,8)
        event=dict(reviewer='Reviewer',time_utc='2026-11-26T20:00:00+00:00',action='transcript_review',before={'segment_id':'s1','final_transcript':''},after={'segment_id':'s1','final_transcript':record['final_transcript']})
        # Test the export of a previously reviewed record, independent of optional ASR libraries.
        with patch.object(corpus,'compare'):
            result=corpus.export_corpus([record],{},[event],self.root/'out')
        self.assertFalse(result['partial']); rows=self.verify(result,[record]); self.assertEqual(rows[0]['changed_by'],'Reviewer'); self.assertEqual(json.loads(rows[0]['change_log']),[event])

    def test_both_partial_paths(self):
        records=[self.record('p',0,8,'pending'),self.record('u',8,30,'unresolved')]
        for automatic in (False,True):
            with self.subTest(automatic=automatic):
                result=corpus.export_corpus(records,{},[],self.root/'out',True,automatic=automatic)
                self.assertTrue(result['partial']); rows=self.verify(result,records)
                self.assertTrue(all(r['review_status']=='pending' and not r['final_transcript'] for r in rows))

if __name__=='__main__':unittest.main()

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import corpus_review_and_export as corpus
from processing_errors import describe_exception, exception_details, save_failure_log
from test_portable_review_html import AudioURLs


class ExportFailureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source.wav'
        sf.write(self.source, np.zeros(16*16000,dtype=np.int16), 16000, subtype='PCM_16')
        self.record = dict(segment_id='auto-00042',start=0,end=8,label='speech',
                           source_mode='candidate_preview',segmentation_status='pending',
                           source_origin=str(self.source),status='needs_review',final_transcript='',
                           audio_pieces=[dict(path=str(self.source),start=0,end=8)])
        self.destination = self.root / 'Review and export'

    def test_blank_memory_error_reports_exact_export_stage_and_preserves_review(self):
        original=copy.deepcopy(self.record)
        with patch.object(corpus,'write_review_audio',side_effect=MemoryError()):
            try:
                corpus.export_corpus([self.record],{},[],self.destination,True,automatic=True)
            except MemoryError as error:
                message, trace = exception_details(error)
            else:
                self.fail('Expected simulated export failure')
        self.assertIn('MemoryError',message)
        self.assertIn('auto-00042',message)
        self.assertIn('Writing clip 1/1',message)
        self.assertIn(str(self.destination),message)
        self.assertIn('Traceback',trace)
        self.assertEqual(self.record,original)
        self.assertEqual(list(self.destination.iterdir()),[])
        path=save_failure_log(message,trace,self.destination,'export')
        self.assertTrue(path.is_file())
        self.assertIn('MemoryError',path.read_text(encoding='utf-8'))
        # Retry the existing review without any ASR or approval.
        result=corpus.export_corpus([self.record],{},[],self.destination,True,automatic=True)
        self.assertEqual(result['clips'],1)
        self.assertTrue(result['partial'])
        self.assertEqual(self.record,original)
        self.assertEqual(len(AudioURLs(Path(result['output_dir'])/'corpus_report.html').urls),1)

    def test_empty_assertion_has_nonempty_description(self):
        self.assertIn('AssertionError',describe_exception(AssertionError()))
        self.assertIn('without an error message',describe_exception(AssertionError()))

    def test_metadata_write_failure_keeps_previous_successful_export(self):
        first=corpus.export_corpus([self.record],{},[],self.destination,True,automatic=True)
        with patch.object(corpus,'write_json_file',side_effect=OSError('Disk full')):
            try:
                corpus.export_corpus([self.record],{},[],self.destination,True,automatic=True)
            except OSError as error:
                self.assertIn('Writing corpus metadata JSON',describe_exception(error))
                self.assertIn('Disk full',describe_exception(error))
            else:
                self.fail('Expected simulated write failure')
        self.assertEqual(list(self.destination.iterdir()),[Path(first['output_dir'])])

    def test_error_log_falls_back_when_destination_is_not_writable(self):
        blocked=self.root/'file'; blocked.write_text('not a directory')
        with patch('processing_errors.tempfile.gettempdir',return_value=str(self.root)):
            path=save_failure_log('error','trace',blocked)
        self.assertEqual(path.parent,self.root/'SpeechCorpusStudio')
        self.assertIn('trace',path.read_text(encoding='utf-8'))

    def test_streamed_html_matches_existing_output(self):
        rows=list(corpus.review_rows([self.record],[])); rows[0]['audio_file']='clip.wav'
        path=self.root/'review.html';corpus.write_review_html(path,rows,partial=True,playable=True)
        self.assertEqual(path.read_text(encoding='utf-8'),corpus.review_html(rows,partial=True,playable=True))

if __name__=='__main__':unittest.main()

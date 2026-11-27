"""The export selection follows the transcription queue, not uncovered time."""
import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import corpus_review_and_export as corpus
import segment_review as review
from test_ignored_gaps import GapExportTests as Fixture
from test_portable_review_html import AudioURLs


class TranscriptionExportSelectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.session = self.fixture.session
        self.source = self.fixture.source
        self.root = self.fixture.root
        self.session['unresolved'].extend([
            dict(id='wanted', category='unresolved', label='speech', status='reviewed', original=dict(start=10,end=22)),
            dict(id='skipped', category='unresolved', label='mixed', status='reviewed', original=dict(start=22,end=28)),
            dict(id='music', category='unresolved', label='music', status='reviewed', original=dict(start=28,end=30)),
        ])
        # Skip the middle of the wanted interval and all of another interval.
        review.toggle_span_transcription(self.session, self.session['unresolved'][1], 14, 18, 'Operator')
        review.toggle_span_transcription(self.session, self.session['unresolved'][2], 22, 28, 'Operator')
        def record(ident, parent, a, b, label='speech'):
            return dict(segment_id=ident, source_mode='unresolved_preview', segmentation_status='unresolved',
                        start=a, end=b, label=label, source_origin=str(self.source),
                        source_metadata={'id':parent}, audio_pieces=[dict(path=str(self.source),start=a,end=b)],
                        status='needs_review', transcript_a='أهلا', transcript_b='مرحبا', final_transcript='')
        self.records = [self.fixture.record, record('wanted@160000-224000','wanted',10,14),
                        record('wanted@288000-352000','wanted',18,22),
                        record('skipped','skipped',22,28,'mixed'), record('music','music',28,30,'music'),
                        record('wanted','wanted',10,22)]  # stale, partially skipped row
        self.expected = self.records[:3]

    def verify(self, result):
        out = Path(result['output_dir'])
        self.assertEqual(result['clips'], 3)
        expected = [r['segment_id'] for r in self.expected]
        for name in ('review.csv','review_compatible.csv','corpus.csv'):
            with (out/name).open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['segment_id'] for r in rows], expected)
        urls = AudioURLs(out/'corpus_report.html').urls
        self.assertEqual(len(urls), 3)
        self.assertEqual(len(list(out.glob('*.wav'))), 3)
        self.assertEqual(len(list(out.glob('*.txt'))), 3)
        for record,url in zip(self.expected,urls):
            pcm,rate=sf.read(out/url,dtype='int16')
            self.assertEqual(rate,16000)
            np.testing.assert_array_equal(pcm,self.fixture.samples[record['start']*16000:record['end']*16000])
        meta = json.loads((out/'corpus.json').read_text())['metadata']
        self.assertEqual(meta['export_scope'],'included_transcription_records_only')
        self.assertEqual(meta['unresolved_export_segment_ids'], sorted(expected[1:]))
        self.assertEqual(json.loads((out/'source_review.json').read_text())['step4'],self.session)

    def test_all_three_corpus_export_modes(self):
        for partial,automatic in ((False,False),(True,False),(True,True)):
            with self.subTest(partial=partial,automatic=automatic), patch.object(corpus,'compare'):
                result=corpus.export_corpus(self.records,self.fixture.manifest(),[],self.root/'out',partial,automatic=automatic)
                self.verify(result)

    def test_standalone_review_matches_export_selection(self):
        path=self.root/'report'/'review.html'
        corpus.export_review_csv(path,self.records,[],self.session,self.source)
        self.assertEqual(len(AudioURLs(path).urls),3)
        self.assertEqual(len(list(path.parent.glob('review_audio/*.wav'))),3)
        csv_path=path.with_suffix('.csv')
        corpus.export_review_csv(csv_path,self.records,[],self.session,self.source)
        with csv_path.open(encoding='utf-8-sig') as stream:
            self.assertEqual([r['segment_id'] for r in csv.DictReader(stream)],[r['segment_id'] for r in self.expected])

    def test_empty_queue_does_not_recreate_skipped_audio(self):
        with self.assertRaisesRegex(ValueError,'No included transcription clips'):
            corpus.export_corpus(self.records[3:],self.fixture.manifest(),[],self.root/'out',True)

if __name__ == '__main__':
    unittest.main()

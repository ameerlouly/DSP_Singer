"""Portable review report audio is a copy of the exact reviewed samples."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import shutil
import tempfile
import unittest
import sys
import numpy as np
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import corpus_review_and_export as corpus


class AudioURLs(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.urls = []
        self.feed(path.read_text(encoding='utf-8'))

    def handle_starttag(self, tag, attrs):
        if tag == 'audio':
            self.urls.append(dict(attrs)['src'])


class PortableReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'original & مصدر.wav'
        self.samples = np.random.default_rng(71).integers(-32768, 32768, 30*16000, dtype=np.int16)
        sf.write(self.source, self.samples, 16000, subtype='PCM_16')

    def record(self, ident, a, b, state='resolved', pieces=True):
        r = dict(segment_id=ident, start=a, end=b, label='speech',
                 segmentation_status=state, source_origin=str(self.source),
                 final_transcript='مرحبا', status='accepted')
        if pieces:
            r['audio_pieces'] = [dict(path=str(self.source), start=a, end=b)]
        return r

    def test_move_report_and_play_every_exact_interval(self):
        records = [self.record('approved', 1, 8), self.record('pending', 8, 13, 'pending'),
                   self.record('unresolved', 13, 29, 'unresolved', False)]
        report = self.root / 'report folder & تقرير' / 'review.html'
        corpus.export_review_csv(report, records, [], source=self.source)
        urls = AudioURLs(report).urls
        self.assertEqual(len(urls), 3)
        self.assertTrue(all(not urlsplit(u).scheme and not u.startswith('/') and '\\' not in u for u in urls))
        moved = self.root / 'moved folder'
        shutil.move(str(report.parent), moved)
        self.source.unlink()
        for record, url in zip(records, AudioURLs(moved/'review.html').urls):
            data, rate = sf.read(moved / unquote(url), dtype='int16')
            self.assertEqual(rate, 16000)
            np.testing.assert_array_equal(data, self.samples[record['start']*16000:record['end']*16000])

    def test_resaving_preserves_old_report_assets(self):
        report = self.root/'out'/'review.html'
        records = [self.record('one', 1, 8)]
        corpus.export_review_csv(report, records, [])
        first = AudioURLs(report).urls[0]
        corpus.export_review_csv(report, records, [])
        second = AudioURLs(report).urls[0]
        self.assertNotEqual(first, second)
        self.assertTrue((report.parent/first).is_file())
        self.assertTrue((report.parent/second).is_file())

    def test_invalid_interval_does_not_replace_existing_report(self):
        report = self.root/'out'/'review.html'
        corpus.export_review_csv(report, [self.record('one', 1, 8)], [])
        prior = report.read_bytes()
        before = sorted(report.parent.iterdir())
        with self.assertRaisesRegex(ValueError, 'Invalid review audio interval'):
            corpus.export_review_csv(report, [self.record('invalid', 26, 32)], [])
        self.assertEqual(prior, report.read_bytes())
        self.assertEqual(before, sorted(report.parent.iterdir()))

    def test_csv_save_does_not_create_audio_folder(self):
        path = self.root/'csv'/'review.csv'
        corpus.export_review_csv(path, [self.record('one', 1, 8)], [])
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_player_url_encoding(self):
        row = next(corpus.review_rows([self.record('one', 1, 8)], []))
        row['audio_file'] = 'a folder/clip #1.wav'
        path = self.root/'review.html'
        corpus.write_review_table(path, [row], playable=True)
        self.assertEqual(AudioURLs(path).urls, ['a%20folder/clip%20%231.wav'])


if __name__ == '__main__':
    unittest.main()

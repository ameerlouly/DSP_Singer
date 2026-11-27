"""Large history snapshots are preserved once rather than repeated per CSV row."""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import corpus_review_and_export as corpus
from compact_review_history import CompactReviewHistory, expand_change_log, INLINE_LIMIT
from restore_full_review_csv import restore


class CompactHistoryTests(unittest.TestCase):
    def test_small_history_unchanged(self):
        history = [{'time_utc': '2026-11-27', 'reviewer': 'مراجع', 'before': 'قديم', 'after': 'جديد'}]
        packer = CompactReviewHistory([], history)
        self.assertEqual(packer.encode(history), json.dumps(history, ensure_ascii=False))
        self.assertEqual(packer.index, [])

    def test_exact_restore_with_multiple_origins_and_duplicate_events(self):
        e1 = {'before': {'details': 'تفاصيل ' * 4000}, 'after': {'text': 'نص'}}
        e2 = {'action': 'quality', 'after': 'clean'}
        records = [{'human_changes': [e1], 'quality_history': [e2]}]
        audit_log = [{'action': 'transcript', 'before': {}, 'after': {}}]
        events = [e2, e1, audit_log[0], e2]
        packer = CompactReviewHistory(records, audit_log)
        cell = packer.encode(events)
        audit = dict(segments=records, audit_log=audit_log, review_history=packer.index)
        self.assertLess(len(cell), INLINE_LIMIT)
        self.assertEqual(expand_change_log(cell, json.loads(json.dumps(audit))),
                         json.dumps(events, ensure_ascii=False))
        self.assertEqual(len(packer.index), 1)

    def test_many_small_events_trigger_reference(self):
        events = [{'n': i, 'reason': 'a' * 300} for i in range(50)]
        packer = CompactReviewHistory([], events)
        cell = packer.encode(events)
        self.assertLess(len(cell), 300)
        self.assertEqual(expand_change_log(cell, {'audit_log':events, 'review_history':packer.index}),
                         json.dumps(events, ensure_ascii=False))

    def test_exact_compatible_projection_preserves_every_other_cell(self):
        event = {'time_utc': '2026-11-27T00:00:00+00:00', 'reviewer': 'Reviewer',
                 'action': 'transcript_review', 'reason': '',
                 'before': {'segment_id': 'auto-00007', 'text': 'نص', 'nested': {'payload': 'تفاصيل ' * 6000}},
                 'after': {'segment_id': 'auto-00007', 'selected_transcript': 'B', 'final_transcript': 'النص'}}
        record = {'segment_id': 'auto-00007', 'transcript_a': 'نص، أول',
                  'transcript_b': 'النص الثاني', 'final_transcript': 'النص الثاني'}
        packer = CompactReviewHistory([record], [event])
        marker = packer.encode([event])
        audit = {'segments': [record], 'audit_log': [event], 'review_history': packer.index}
        row = dict.fromkeys(corpus.REVIEW_COLUMNS, '')
        row.update(start=9.88, end=24.23, label='mixed', snr=8.74, quality='noisy',
                   model_A_transcript=record['transcript_a'], model_B_transcript=record['transcript_b'],
                   final_transcript=record['final_transcript'], status='accepted',
                   review_status='accepted', automatic_status='needs_review', segment_id=record['segment_id'],
                   audio_file='segment_00007.wav', changed_by='Reviewer', changed_at=event['time_utc'],
                   change_log=marker, segmentation_status='resolved')
        expected = {k: row[k] for k in corpus.COMPATIBLE_REVIEW_COLUMNS}
        expected['change_log'] = json.dumps([event], ensure_ascii=False)
        actual = next(corpus.compatible_review_rows([record], [row], audit=audit))
        self.assertEqual(actual, expected)
        self.assertEqual(list(actual), list(corpus.COMPATIBLE_REVIEW_COLUMNS))
        self.assertGreater(len(actual['change_log']), 32767)
        with self.assertRaisesRegex(ValueError, 'full activity history'):
            list(corpus.compatible_review_rows([record], [row]))
        # Normal inline values are preserved without changing their formatting.
        for inline in ('[]', '[ {"action": "transcript_review", "before": null} ]'):
            row['change_log'] = inline
            self.assertEqual(next(corpus.compatible_review_rows([record], [row], audit=audit))['change_log'], inline)

    def test_export_csv_full_details_html_remains_compact(self):
        original_limit = csv.field_size_limit(10_000_000)
        self.addCleanup(csv.field_size_limit, original_limit)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); wav = root / 'source.wav'
            sf.write(wav, np.zeros(8*16000*12, dtype=np.int16),16000,subtype='PCM_16')
            records = []
            for i in range(12):
                a,b=i*8,(i+1)*8
                records.append(dict(segment_id=f's{i}',start=a,end=b,label='speech',
                    source_mode='candidate_preview',segmentation_status='pending',source_origin=str(wav),
                    status='needs_review',transcript_a='أهلاً يا فاطمة',transcript_b='اهلا يا فاطمة',
                    final_transcript='',audio_pieces=[dict(path=str(wav),start=a,end=b)]))
            # A batch audit snapshot associated with all rows: this used to be
            # copied into every row of three different outputs.
            event = dict(time_utc='2026-11-27',reviewer='Reviewer',action='transcript_review',
                         before=[{'segment_id':r['segment_id']} for r in records],
                         after={'snapshot':'معلومات كاملة ' * 20000})
            audit_log = [event]
            for automatic in (False, True):
                with self.subTest(automatic=automatic):
                    result = corpus.export_corpus(records,{},audit_log,root/'out',True,automatic=automatic)
                    out = Path(result['output_dir'])
                    audit = json.loads((out/'corpus_audit.json').read_text(encoding='utf-8'))
                    self.assertEqual(audit['audit_log'],audit_log)
                    expected = list(corpus.compatible_review_rows(audit['segments'],
                                        corpus.review_rows(audit['segments'],audit_log)))
                    compact = out/'review_compatible.csv'
                    with compact.open(encoding='utf-8-sig',newline='') as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual(len(rows),12)
                    self.assertTrue(all(json.loads(r['change_log']) == audit_log for r in rows))
                    self.assertTrue(all(len(r['change_log']) > INLINE_LIMIT for r in rows))
                    self.assertTrue(all('event_refs_v1' not in r['change_log'] for r in rows))
                    data = json.loads((out/'corpus.json').read_text(encoding='utf-8'))
                    self.assertEqual(data['metadata']['compatible_review_csv_change_log_encoding'], 'inline_json')
                    self.assertTrue(all(len(r['change_log']) < 300 for r in data['segments']))
                    self.assertTrue(all(expand_change_log(r['change_log'], audit) == csv_row['change_log']
                                        for r, csv_row in zip(data['segments'], rows)))
                    self.assertIn('Open full audit',(out/'corpus_report.html').read_text(encoding='utf-8'))
                    self.assertEqual(len(audit['review_history']),12)
                    target = restore(out)
                    with target.open(encoding='utf-8-sig',newline='') as f:
                        restored = list(csv.DictReader(f))
                    # CSV values are strings; compare exactly with the old writer.
                    legacy = out/'expected.csv'
                    corpus.write_review_table(legacy, expected,csv_columns=corpus.COMPATIBLE_REVIEW_COLUMNS)
                    self.assertEqual(target.read_bytes(),legacy.read_bytes())
                    self.assertEqual(len(restored),12)
                    self.assertEqual(compact.read_bytes(), target.read_bytes())
                    self.assertLess((out/'corpus_report.html').stat().st_size, compact.stat().st_size/100)
                    archive = restore(out,compressed=True)
                    with gzip.open(archive,'rb') as f:
                        self.assertEqual(f.read(),target.read_bytes())
                    self.assertEqual(len(list(out.glob('*.wav'))),12)
                    print(f'Full CSV: {compact.stat().st_size:,} bytes; compact HTML: {(out/"corpus_report.html").stat().st_size:,} bytes')

if __name__ == '__main__': unittest.main()

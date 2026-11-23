"""PDF Step 6 examples and boundary cases; no model weights are downloaded."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import asr_pipeline as asr
import corpus_review as corpus


class PDFNormalizationTests(unittest.TestCase):
    def test_required_letter_table_and_arabic_punctuation(self):
        pairs = [
            ('إِنَّ السَّلَامَـ، عَلَيْكُمْ!', 'ان السلام عليكم'),
            ('أ إ آ ٱ ا', 'ا ا ا ا ا'),
            ('ه ة ي ى', 'ه ه ي ي'),
            ('مُدَرِّسَة هُدَى؛ كويسة؟', 'مدرسه هدي كويسه'),
            ('ازيك،عامل ايه؟', 'ازيك عامل ايه'),
        ]
        for raw, expected in pairs:
            with self.subTest(raw=raw):self.assertEqual(asr.normalize_arabic(raw),expected)

    def test_presentation_forms_decomposed_letters_and_invisible_controls(self):
        self.assertEqual(asr.normalize_arabic('\ufef7 \ufefb ا\u0654 ا\u0655'), 'لا لا ا ا')
        self.assertEqual(asr.normalize_arabic('ه\u200dدى\u200f'), 'هدي')
        self.assertEqual(asr.normalize_arabic('س\u08f0لام'), 'سلام')

    def test_preserve_dialect_words_and_other_hamza_letters(self):
        self.assertEqual(asr.normalize_arabic('إزّيك؟ مش عايزة أروح دلوقتي!'), 'ازيك مش عايزه اروح دلوقتي')
        self.assertEqual(asr.normalize_arabic('سؤال بيئة شيء'), 'سؤال بيئه شيء')

    def test_idempotent_without_mutating_the_original(self):
        raw='إِنَّ مُدَرِّسَةَ هُدَى، كويسة!'
        once=asr.normalize_arabic(raw)
        self.assertEqual(asr.normalize_arabic(once),once)
        self.assertEqual(raw,'إِنَّ مُدَرِّسَةَ هُدَى، كويسة!')


class PDFAgreementTests(unittest.TestCase):
    def test_agreement_after_normalization(self):
        r=asr.comparison_details('أهلاً، بكم!', 'اهلا بكم')
        self.assertEqual((r['automatic_status'],r['cer'],r['wer'],r['word_difference']),('accepted',0.,0.,0))
        self.assertEqual(r['comparison_engine'],'jiwer')

    def test_two_percent_boundary(self):
        self.assertEqual(asr.compare_transcripts('ا'*50,'ا'*49+'ب')[0],'accepted')
        self.assertEqual(asr.compare_transcripts('ا'*49,'ا'*48+'ب')[0],'needs_review')

    def test_three_word_edits_require_human_choice_four_are_difficult(self):
        reference='واحد اتنين تلاته اربعه'
        three=asr.comparison_details(reference,'خمسه سته سبعه اربعه')
        four=asr.comparison_details(reference,'خمسه سته سبعه تمانيه')
        self.assertEqual((three['automatic_status'],three['word_difference']),('needs_review',3))
        self.assertEqual((four['automatic_status'],four['word_difference']),('difficult',4))
        self.assertEqual(four['word_substitutions'],4)

    def test_insertions_and_deletions_are_counted_too(self):
        insertion=asr.comparison_details('اهلا بكم','اهلا وسهلا بكم هنا')
        deletion=asr.comparison_details('اهلا وسهلا بكم هنا','اهلا بكم')
        self.assertEqual(insertion['word_insertions'],2)
        self.assertEqual(deletion['word_deletions'],2)
        self.assertEqual(insertion['wer'],1.)
        self.assertEqual(deletion['wer'],.5)

    def test_empty_and_punctuation_only_outputs_are_never_accepted(self):
        for left,right in [('', ''),('؟!','ـَ'),('', 'اهلا بكم'),('اهلا بكم',''),('', 'واحد اتنين تلاته اربعه')]:
            with self.subTest(left=left,right=right):
                r=asr.comparison_details(left,right)
                self.assertNotEqual(r['automatic_status'],'accepted')
                self.assertIn('empty',r['agreement_reason'])
        self.assertEqual(asr.comparison_details('', 'اهلا بكم')['word_insertions'],2)

    def test_jiwer_is_required_not_silently_replaced(self):
        with patch.dict(sys.modules,{'jiwer':None}):
            with self.assertRaisesRegex(RuntimeError,'jiwer 4'):asr.compare_transcripts('اهلا','اهلا')


class ReviewAndExportTests(unittest.TestCase):
    def record(self):
        return dict(label='speech',transcript_a='مَدْرَسَة هدى',transcript_b='مدرسه هدي',
                    asr_a_done=True,asr_b_done=True,human_override=False,status='needs_review')

    def test_raw_text_preserved_automatic_final_and_human_override(self):
        r=self.record();corpus.compare(r)
        self.assertEqual(r['status'],'accepted')
        self.assertEqual(r['final_transcript'],'مَدْرَسَة هدى')
        self.assertEqual(r['normalized_transcript_a'],'مدرسه هدي')
        r.update(human_override=True,final_transcript='مدرسة هدى بعد المراجعة',selected_transcript='مدرسة هدى بعد المراجعة',status='accepted')
        r['transcript_b']='كلام مختلف تماما الان'
        corpus.compare(r)
        self.assertEqual(r['status'],'accepted')
        self.assertEqual(r['final_transcript'],'مدرسة هدى بعد المراجعة')
        self.assertNotEqual(r['automatic_status'],'accepted')

    def test_pending_model_clears_stale_automatic_metrics(self):
        r=self.record();corpus.compare(r);r['asr_b_done']=False;corpus.compare(r)
        self.assertIsNone(r['cer']);self.assertIsNone(r['wer'])
        self.assertEqual(r['status'],'needs_review');self.assertEqual(r['final_transcript'],'')

    def test_csv_has_raw_normalized_scores_and_policy(self):
        r=self.record();r['segment_id']='clip';corpus.compare(r)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'corpus.csv';asr.export_csv(p,{},[r])
            with p.open(encoding='utf-8-sig',newline='') as handle:row=next(csv.DictReader(handle))
        self.assertEqual(row['transcript_a'],'مَدْرَسَة هدى')
        self.assertEqual(row['normalized_transcript_a'],'مدرسه هدي')
        self.assertEqual(row['normalization_version'],asr.NORMALIZATION_VERSION)
        self.assertEqual(row['comparison_reference'],'transcript_a')
        self.assertEqual(float(row['wer']),0.)


if __name__=='__main__':unittest.main()

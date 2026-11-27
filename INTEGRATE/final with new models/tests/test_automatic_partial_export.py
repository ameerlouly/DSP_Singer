import copy, csv, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import corpus_review_and_export as corpus
from test_portable_review_html import AudioURLs

class AutoExportTests(unittest.TestCase):
    def test_pending_and_unresolved_export_without_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source.wav'
            samples=np.random.default_rng(5).integers(-32768,32767, 30*16000,dtype=np.int16)
            sf.write(source,samples,16000,subtype='PCM_16')
            def record(ident,a,b,state):
                return dict(segment_id=ident,start=a,end=b,label='speech',source_mode=state+'_preview' if state=='unresolved' else 'candidate_preview',segmentation_status=state,source_origin=str(source),status='needs_review',transcript_a='اهلا',transcript_b='مرحبا',final_transcript='',audio_pieces=[dict(path=str(source),start=a,end=b)])
            records=[record('p',0,8,'pending'),record('u',8,30,'unresolved')]; before=copy.deepcopy(records)
            with self.assertRaisesRegex(ValueError,'pending'):
                corpus.export_corpus(records,{},[],root)
            result=corpus.export_corpus(records,{},[],root,True,automatic=True)
            self.assertEqual(records,before); self.assertTrue(result['partial']);self.assertEqual(result['clips'],2)
            out=Path(result['output_dir']); meta=json.loads((out/'corpus.json').read_text())['metadata']
            self.assertEqual(meta['export_mode'],'automatic_partial');self.assertEqual(meta['pending_boundary_segment_ids'],['p','u'])
            urls=AudioURLs(out/'corpus_report.html').urls;self.assertEqual(len(urls),2)
            source.unlink()
            for r,url in zip(records,urls):
                data,rate=sf.read(out/url,dtype='int16'); self.assertEqual(rate,16000)
                np.testing.assert_array_equal(data,samples[r['start']*16000:r['end']*16000]);self.assertEqual((out/url).with_suffix('.txt').read_text(),'')
            with (out/'corpus.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            self.assertEqual([r['segmentation_status'] for r in rows],['pending','unresolved'])
    def test_complete_mode_allows_unresolved_without_approval_or_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source.wav'
            samples=np.random.default_rng(9).integers(-32000,32000,22*16000,dtype=np.int16)
            sf.write(source,samples,16000,subtype='PCM_16')
            # Legacy rows can carry explicit segmentation status without source_mode.
            record=dict(segment_id='u',start=0,end=22,label='speech',segmentation_status='unresolved',
                        source_origin=str(source),status='not_transcribed',final_transcript='',
                        audio_pieces=[dict(path=str(source),start=0,end=22)])
            before=copy.deepcopy(record)
            result=corpus.export_corpus([record],{},[],root,False)
            self.assertEqual(result['export_mode'],'complete_with_unresolved')
            self.assertEqual(record,before)
            out=Path(result['output_dir']); meta=json.loads((out/'corpus.json').read_text())['metadata']
            self.assertTrue(meta['partial']);self.assertEqual(meta['unresolved_export_segment_ids'],['u'])
            urls=AudioURLs(out/'corpus_report.html').urls
            self.assertEqual(len(urls),1);self.assertTrue(urls[0].endswith('_unresolved.wav'))
            pcm,rate=sf.read(out/urls[0],dtype='int16');self.assertEqual(rate,16000)
            np.testing.assert_array_equal(pcm,samples)
            self.assertEqual((out/urls[0]).with_suffix('.txt').read_text(),'')

    def test_automatic_requires_partial(self):
        with self.assertRaisesRegex(ValueError,'partial mode'):
            corpus.export_corpus([],{},[],'.',automatic=True)

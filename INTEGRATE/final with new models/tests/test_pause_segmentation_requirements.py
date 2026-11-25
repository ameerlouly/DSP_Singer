"""PDF Step 4: source continuity, pause context, complete coverage and review gates.

Test guide
----------
PDFPart4
    Check lecture clips keep real context and source, vad missed word is not discarded
    as silence and related cases.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pause_based_segmentation as cuts
import segment_review as review

RATE=16000

class PDFPart4(unittest.TestCase):
    def test_consecutive_short_unresolved(self):
        pieces = [dict(start=i*2,end=i*2+1.8) for i in range(10)]
        result = cuts.group_short_unresolved(pieces, [])
        self.assertEqual(sum(len(r['parts']) for r in result), 10)
        self.assertTrue(all(5 <= r['end']-r['start'] <= 15 for r in result))
        self.assertEqual(cuts.group_short_unresolved(pieces[:3],[dict(start=1.81,end=1.99)]),pieces[:3])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.path=self.root/'source.wav'
    def tearDown(self):self.tmp.cleanup()
    def wave(self,phrases,duration=40):
        t=np.arange(round(duration*RATE))/RATE
        samples=np.full(len(t),.0001,dtype=np.float32)
        for a,b in phrases:
            i,j=round(a*RATE),round(b*RATE)
            samples[i:j]+=.15*np.sin(2*np.pi*220*t[i:j])
        sf.write(self.path,samples,RATE,subtype='PCM_16')
    def export(self,module,phrases,label='speech',separator=None,regions=None):
        fake_torch=types.ModuleType('torch');fake_torch.set_num_threads=lambda n:None
        fake_vad=types.ModuleType('silero_vad');fake_vad.load_silero_vad=lambda:object()
        with patch.dict(sys.modules,torch=fake_torch,silero_vad=fake_vad), \
             patch.object(cuts,'detect_speech',side_effect=phrases if callable(phrases) else lambda *a,**kw:phrases), \
             patch.object(cuts,'mixed_pause_separator',return_value=separator):
            result=module.export_candidates(self.path,regions or [dict(start=0.,end=40.,label=label)],self.root)
        folder=Path(result['output_dir'])
        return json.loads((folder/'step4_segments.json').read_text()),folder
    def assert_contract(self,data,folder):
        source,_=sf.read(self.path,dtype='int16')
        for s in data['segments']:
            actual,_=sf.read(folder/s['file'],dtype='int16')
            np.testing.assert_array_equal(actual,source[round(s['start']*RATE):round(s['end']*RATE)])
            self.assertTrue(5<=len(actual)/RATE<=15)
            self.assertTrue(all(.1-1e-6<=s[k]<=.2+1e-6 for k in ('left_padding','right_padding')))
            self.assertEqual(set(s['boundary_evidence']),{'start','end'})
            self.assertEqual(s['source_spans'],[dict(start=s['start'],end=s['end'],clip_start=0.,clip_end=s['duration'])])
            self.assertEqual(s['removed_gaps'],[])
            self.assertEqual(s['synthetic_silence_start_seconds'],0.)
        total=sum(s['duration'] for s in data['segments'])+sum(u['end']-u['start'] for u in data['unresolved'])+sum(g['end']-g['start'] for g in data['not_in_clips_or_unresolved'])
        self.assertAlmostEqual(total,40.,places=5)
        ordered=sorted(data['segments'],key=lambda s:s['start'])
        self.assertTrue(all(a['end']<=b['start']+1/RATE for a,b in zip(ordered,ordered[1:])))
    def test_lecture_clips_keep_real_context_and_source(self):
        phrases=[[.5,9.5],[10.,19.],[19.5,29.],[29.5,39.5]];self.wave(phrases)
        digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        data,folder=self.export(cuts,phrases)
        self.assert_contract(data,folder)
        self.assertEqual(len(data['segments']),4)
        self.assertFalse(data['unresolved'])
        self.assertEqual(digest,hashlib.sha256(self.path.read_bytes()).hexdigest())
    def test_vad_missed_word_is_not_discarded_as_silence(self):
        self.wave([[.5,9.5],[10.,19.],[19.5,29.],[29.5,39.5]])
        detected=[[.5,4.],[5.,9.5],[10.,19.],[19.5,29.],[29.5,39.5]]
        data,folder=self.export(cuts,detected);self.assert_contract(data,folder)
        self.assertTrue(any(s['start']<4. and s['end']>5. for s in data['segments']))
        self.assertFalse(any(g['start']<5 and g['end']>4 for g in data['not_in_clips_or_unresolved']))
    def test_short_syllable_dip_and_brief_onset_inside_pause_are_rejected(self):
        starts=np.arange(1000)*.02;ends=starts+.02;reference=np.full(1000,-14.)
        levels=reference.copy();levels[400:410]=-35.  # 200 ms: not the conservative 240 ms rule.
        self.assertEqual(cuts.vocal_pause_candidates(starts,ends,levels,reference),[])
        levels=reference.copy();levels[400:415]=-35.;levels[407]=-14.  # Raw consonant/onset cannot be smoothed away.
        self.assertEqual(cuts.vocal_pause_candidates(starts,ends,levels,reference),[])
        levels=reference.copy();levels[400:420]=-35.
        points=cuts.vocal_pause_candidates(starts,ends,levels,reference)
        self.assertEqual(len(points),1);self.assertAlmostEqual(points[0]['time'],8.2)
        self.assertTrue(points[0]['raw_energy_checked'])
    def test_short_pause_is_shared_without_overlap(self):
        groups,missing=cuts.group_at_pauses([[.3,9.88],[10.12,19.7]],0.,20.)
        self.assertEqual(len(groups),2);self.assertFalse(missing)
        self.assertAlmostEqual(groups[0]['end'],groups[1]['start'])
        self.assertAlmostEqual(groups[0]['end']-groups[0]['speech_end'],.12)
    def test_missing_source_edge_and_unsplittable_singing_stay_unresolved(self):
        groups,missing=cuts.group_at_pauses([[0.,9.7],[10.3,19.7]],0.,20.)
        self.assertEqual(len(groups),1);self.assertEqual(missing[0]['start'],0.)
        self.wave([[0.,40.]])
        data,folder=self.export(cuts,[[0.,40.]],label='mixed')
        self.assertFalse(data['segments']);self.assertEqual([(u['start'],u['end']) for u in data['unresolved']],[(0.,40.)])
        self.assert_contract(data,folder)
    def test_singing_with_backing_uses_vocal_pauses_not_artificial_padding(self):
        samples=np.full(40*RATE,.2,dtype=np.float32)
        for a,b in [(0.,.4),(9.8,10.2),(19.8,20.2),(29.8,30.2),(39.6,40.)]:samples[round(a*RATE):round(b*RATE)]=.1
        sf.write(self.path,samples,RATE,subtype='PCM_16')
        sep=lambda x,rate:np.where(x>.15,.2,.02).astype(np.float32)
        data,folder=self.export(cuts,[[0.,40.]],label='mixed',separator=sep)
        self.assert_contract(data,folder);self.assertEqual(len(data['segments']),4)
        self.assertFalse(data['unresolved'])
        self.assertTrue(all(s['kind']=='vocal_pause_candidate' for s in data['segments']))
    def test_adjacent_silence_is_checked_and_shared_without_overlap(self):
        phrases = [[1., 9.], [9.22, 17.22]]
        regions = [dict(start=a,end=b,label=label) for a,b,label in [
            (0.,1.,'silence'),(1.,9.,'speech'),(9.,9.22,'silence'),
            (9.22,17.22,'speech'),(17.22,40.,'silence')]]
        self.wave(phrases)
        data, folder = self.export(cuts,phrases,regions=regions)
        self.assertEqual([(c['start'],c['end']) for c in data['segments']],
                         [(.85,9.11),(9.11,17.37)])
        self.assertFalse(data['unresolved'])
        self.assertAlmostEqual(data['coverage_seconds']['adjacent_silence_in_clips'],.52)
        for c in data['segments']:
            wave,_ = sf.read(self.path,dtype='int16')
            actual,_ = sf.read(folder/c['file'],dtype='int16')
            np.testing.assert_array_equal(actual,wave[round(c['start']*RATE):round(c['end']*RATE)])
        # Music and a mislabeled audible word at a block edge cannot supply padding.
        regions[0]['label'] = 'music'
        self.wave(phrases + [[17.22,17.37]])
        data,_ = self.export(cuts,phrases,regions=regions)
        self.assertEqual(len(data['unresolved']),2)
        self.assertFalse(data['segments'])

    def test_only_long_span_gets_second_pass_and_requires_quiet_source(self):
        first = [[.5,8.5],[9.,30.],[30.5,39.5]]
        refined = [[9.,19.],[19.22,30.]]
        calls = []
        def detect(audio,start,end,model,progress,pause_ms=240):
            calls.append((start,end,pause_ms))
            return first if pause_ms==240 else refined
        self.wave([first[0],*refined,first[-1]])
        data,folder = self.export(cuts,detect)
        self.assert_contract(data,folder)
        self.assertEqual(calls,[(0.,40.,240),(9.,30.,200)])
        self.assertEqual(len(data['segments']),4)
        self.assertFalse(data['unresolved'])
        # A quiet consonant that interrupts the 220 ms pause prevents the cut.
        self.wave([first[0],*refined,first[-1],[19.1,19.12]])
        data,folder = self.export(cuts,detect)
        self.assert_contract(data,folder)
        self.assertEqual(len(data['segments']),2)
        self.assertEqual([(u['start'],u['end']) for u in data['unresolved']],[(9.,30.)])

    def test_33_second_lecture_with_steady_room_noise(self):
        phrases = [[.5,11.],[11.5,22.],[22.5,33.5]]
        times = np.arange(40*RATE)/RATE
        samples = np.random.default_rng(123).normal(0,.035,len(times))
        for a,b in phrases:
            mask = (times >= a) & (times < b)
            samples[mask] += .15*np.sin(2*np.pi*220*times[mask])
        samples[times < .5] = samples[times >= 33.5] = 0
        sf.write(self.path,samples,RATE,subtype='PCM_16')
        regions = [dict(start=0.,end=40.,label='speech')]
        data,folder = self.export(cuts,phrases,regions=regions)
        self.assert_contract(data,folder)
        self.assertEqual(len(data['segments']),3)
        self.assertFalse(data['unresolved'])
        # Even if VAD misses a quiet voiced sound, reject the contaminated gap.
        mask = (times >= 11.15) & (times < 11.35)
        samples[mask] += .08*np.sin(2*np.pi*260*times[mask])
        sf.write(self.path,samples,RATE,subtype='PCM_16')
        data,folder = self.export(cuts,phrases,regions=regions)
        self.assertFalse(any(11 <= c['end'] <= 11.5 for c in data['segments']))
        self.assertTrue(data['unresolved'])

    def test_short_vocal_pause_requires_separated_vocals(self):
        times=np.arange(40*RATE)/RATE
        wave=np.full(len(times),.2,dtype=np.float32)
        for a,b in [(0.,.5),(9.9,10.12),(19.9,20.12),(29.9,30.12),(39.5,40.)]:
            wave[(times>=a)&(times<b)]=.1
        sf.write(self.path,wave,RATE,subtype='PCM_16')
        # Residual stem energy stays above the absolute quiet threshold.
        stem=lambda x,rate:np.where(x>.15,.2,.015).astype(np.float32)
        data,folder=self.export(cuts,[[0.,40.]],label='mixed',separator=stem)
        self.assert_contract(data,folder)
        self.assertEqual(len(data['segments']),4)
        self.assertFalse(data['unresolved'])
        data,_=self.export(cuts,[[0.,40.]],label='mixed')
        self.assertFalse(data['segments'])
        self.assertEqual(len(data['unresolved']),1)

    def make_session(self):
        self.wave([[.5,9.5],[10.,19.],[19.5,29.],[29.5,39.5]])
        data,folder=self.export(cuts,[[.5,9.5],[10.,19.],[19.5,29.],[29.5,39.5]])
        return review.load_session(folder/'step4_segments.json',self.path),folder
    def test_approval_records_human_boundary_review_and_exports_exact_source(self):
        session,folder=self.make_session();review.approve_segments(session,[0,1,2,3],'Reviewer')
        self.assertTrue(all(s['pdf_boundary_review']['reviewer']=='Reviewer' for s in session['segments']))
        out=review.save_and_export(session,self.path,self.root)
        manifest=json.loads((Path(out['output_dir'])/'step5_manifest.json').read_text())
        self.assertTrue(all(c['pdf_boundary_review'] and c['synthetic_silence']==dict(start_seconds=0.,end_seconds=0.) for c in manifest['clips']))
        self.assertEqual(out['exported'],4)
    def test_legacy_join_and_padding_cannot_be_approved_or_exported(self):
        session,_=self.make_session();s=session['segments'][0]
        s.update(original=dict(start=0.,end=20.,duration=10.,source_spans=[dict(start=0.,end=5.,clip_start=0.,clip_end=5.),dict(start=15.,end=20.,clip_start=5.,clip_end=10.)]),source_start=0.,source_end=20.)
        before=copy.deepcopy(session)
        with self.assertRaisesRegex(ValueError,'continuous original audio'):review.approve_segments(session,[0,1],'Reviewer')
        self.assertEqual(session,before)
        s['status']='approved'
        with self.assertRaisesRegex(ValueError,'continuous original audio'):review.save_and_export(session,self.path,self.root)
        s['original']=dict(start=0.,end=9.8,duration=10.,source_spans=[dict(start=0.,end=9.8,clip_start=.1,clip_end=9.9)])
        with self.assertRaisesRegex(ValueError,'continuous original audio'):review.approve_segment(session,0,'Reviewer')
        s.update(source_mode='continuous_source',source_start=.35,source_end=9.65)
        review.approve_segment(session,0,'Reviewer')
        self.assertEqual(s['status'],'approved')
    def test_changed_approved_boundaries_require_review_again(self):
        session,_=self.make_session();review.approve_segments(session,[0],'Reviewer')
        item=session['segments'][0];item.update(source_mode='continuous_source',source_start=1.,source_end=9.65)
        with self.assertRaisesRegex(ValueError,'Reapprove'):review.save_and_export(session,self.path,self.root)
        self.assertEqual(review.approve_segments(session,[0],'Reviewer'),1)
        self.assertEqual(review.approved_playback_queue(session,self.path)[0][0],str(self.path.resolve()))

    def test_asr_handoff_uses_source_coordinates_without_synthetic_trim(self):
        import corpus_review_and_export as corpus
        session,_=self.make_session();review.approve_segments(session,[0,1,2,3],'Reviewer')
        processing=types.SimpleNamespace(session=session,audio=self.path,duration=40.,snr_results=[],snr_snapshot=None,require_approved=lambda:None)
        records=corpus.from_processing(processing)
        self.assertEqual(len(records),4)
        for record in records:
            self.assertEqual(record['source_mode'],'continuous_source')
            self.assertTrue(all(p['synthetic_start']==p['synthetic_end']==0. for p in record['audio_pieces']))
            np.testing.assert_array_equal(corpus.read_audio(record),corpus.read_audio(record,for_snr=True))
            corpus.require_continuous_pdf_audio(record)
        changed=copy.deepcopy(records[0])
        changed['audio_pieces'].append(dict(changed['audio_pieces'][0],start=20.,end=26.))
        with self.assertRaisesRegex(ValueError,'non-contiguous'):corpus.require_continuous_pdf_audio(changed)
        changed=copy.deepcopy(records[0]);changed['audio_pieces'][0]['synthetic_start']=.1
        with self.assertRaisesRegex(ValueError,'synthetic'):corpus.require_continuous_pdf_audio(changed)

if __name__=='__main__':unittest.main()

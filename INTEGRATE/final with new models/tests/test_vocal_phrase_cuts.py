"""Estimated vocal stops must split preserved audio, even inside one VAD island.

Test guide
----------
envelope
    Build a controlled sequence of vocal energy and quiet intervals.
controlled_separator
    Supply known vocal samples to the pause detector.
VocalPhraseCuts
    Check relative dips are gain invariant, sustained note and short dip do not force
    cuts and related cases.
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pause_based_segmentation as natural

RATE=16000


def envelope(pauses, seconds=40.):
    signal=np.full(round(seconds*RATE), .2, dtype=np.float32)
    for a,b in pauses:
        signal[round(a*RATE):round(b*RATE)]=.1
    return signal


def controlled_separator(wave, rate):
    # Singing pauses have residual stem energy ABOVE the old -45 dBFS ceiling.
    return np.where(wave>.15,.2,.02).astype(np.float32)


class VocalPhraseCuts(unittest.TestCase):
    def test_relative_dips_are_gain_invariant(self):
        starts=np.arange(2000)*.02
        ends=starts+.02
        levels=np.full(len(starts), -14.)
        for t in (10,20,30):levels[(starts>=t-.2)&(starts<t+.2)]=-34.
        for gain_db in (0.,-25.,8.):
            points=natural.vocal_pause_candidates(starts,ends,levels+gain_db,np.full(len(starts),-14.+gain_db))
            self.assertEqual([p['time'] for p in points],[10.,20.,30.])
            self.assertTrue(all(p['status']=='needs_boundary_review' for p in points))

    def test_sustained_note_and_short_dip_do_not_force_cuts(self):
        starts=np.arange(2000)*.02; ends=starts+.02
        levels=np.full(len(starts),-14.)
        levels[500:503]=-40.  # 60 ms, too short.
        self.assertEqual(natural.vocal_pause_candidates(starts,ends,levels,np.full(len(starts),-14.)),[])
        # A fade without resumed vocals on the right is not an internal stop.
        levels[1000:]=-40.
        points=natural.vocal_pause_candidates(starts,ends,levels,np.full(len(starts),-14.))
        self.assertTrue(all(p['position']=='edge_context' for p in points))

    def test_source_positive_vad_does_not_erase_vocal_pause_proposals(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'song.wav'; samples=envelope([(9.8,10.2),(19.8,20.2),(29.8,30.2)])
            sf.write(path,samples,RATE,subtype='PCM_16')
            with sf.SoundFile(path) as audio:
                islands,checks=natural.protect_mixed_continuity(audio,[[0.,40.]],0.,40.,
                    [dict(start=0.,end=40.,label='mixed')],separator=controlled_separator)
            self.assertEqual(islands,[[0.,10.],[10.,20.],[20.,30.],[30.,40.]])
            self.assertEqual(checks[0]['quiet_intervals'],[])  # No audio was removed as a gap.
            self.assertEqual(len(checks[0]['boundary_candidates']),3)
            self.assertAlmostEqual(sum(b-a for a,b in islands),40.)

    def test_exporter_produces_continuous_review_clips(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'song.wav'
            sf.write(path,envelope([(0.,.4),(9.8,10.2),(19.8,20.2),(29.8,30.2),(39.6,40.)]),RATE,subtype='PCM_16')
            original_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            original_samples=sf.read(path,dtype='int16')[0]
            fake_torch=types.ModuleType('torch');fake_torch.set_num_threads=lambda count:None
            fake_silero=types.ModuleType('silero_vad');fake_silero.load_silero_vad=lambda:object()
            with self.subTest(mode=natural.__name__), \
                    patch.dict(sys.modules,torch=fake_torch,silero_vad=fake_silero), \
                    patch.object(natural,'detect_speech',return_value=[[0.,40.]]), \
                    patch.object(natural,'mixed_pause_separator',return_value=controlled_separator):
                output=natural.export_candidates(path,[dict(start=0.,end=40.,label='mixed')],root)
            folder=Path(output['output_dir']);data=json.loads((folder/'step4_segments.json').read_text())
            self.assertEqual(len(data['segments']),4)
            self.assertTrue(all(.1<=s['left_padding']+.000001<=.200001 and .1<=s['right_padding']+.000001<=.200001 for s in data['segments']))
            self.assertFalse(data['unresolved'])
            self.assertTrue(all(s['kind']=='vocal_pause_candidate' for s in data['segments']))
            self.assertTrue(all(s['status']=='needs_review' and s['boundary_evidence'] for s in data['segments']))
            for s in data['segments']:
                self.assertFalse(s.get('synthetic_silence_start_seconds'))
                self.assertFalse(s.get('removed_gaps'))
            for s in data['segments']:
                actual=sf.read(folder/s['file'],dtype='int16')[0]
                np.testing.assert_array_equal(actual,original_samples[round(s['start']*RATE):round(s['end']*RATE)])
            self.assertEqual(original_hash,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_cut_at_outer_quiet_gap_does_not_reintroduce_removed_audio(self):
        checks=[dict(boundary_candidates=[dict(time=10.),dict(time=20.)])]
        self.assertEqual(natural.split_at_vocal_pauses([[0.,9.8],[10.2,30.]],checks),
                         [[0.,9.8],[10.2,20.],[20.,30.]])


if __name__=='__main__':unittest.main()

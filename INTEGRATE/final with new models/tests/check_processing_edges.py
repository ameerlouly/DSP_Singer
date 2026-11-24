"""Test classification, spectrogram and SNR edge cases with synthetic audio.

Test guide
----------
FakeTagger
    Return fixed classifier scores for the decision checks.
"""
import sys
from pathlib import Path
from unittest.mock import patch
import tempfile
import types
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import audio_region_classifier as labels
import clip_quality_assessment as snr
from audio_spectrogram import calculate_spectrogram
from audio_review_controller import validate_regions

class FakeTagger:
 def __init__(self,**kw):self.i=0
 def inference(self,x):
  assert x.shape==(1,64000)
  scores=np.array([[.8,.05,.01],[.05,.8,.01],[.8,.7,.01]])[self.i:self.i+1];self.i+=1
  return scores,None
fake=types.ModuleType('panns_inference');fake.labels=['Speech','Music','Singing'];fake.AudioTagging=FakeTagger
torch=types.ModuleType('torch');torch.Tensor=type('Tensor',(),{});torch.cuda=types.SimpleNamespace(is_available=lambda:False)
with tempfile.TemporaryDirectory() as temp:
 path=Path(temp)/'labels.wav';rate=16000;t=np.arange(rate*8)/rate
 samples=.2*np.sin(2*np.pi*200*t);samples[6*rate:]=0
 sf.write(path,samples,rate,subtype='PCM_16')
 with patch.dict(sys.modules,panns_inference=fake,torch=torch),patch.object(labels,'download_if_missing'):
  regions=labels.classify_file(path,verify_vocals=False)
 assert [r['label'] for r in regions]==['speech','music','mixed','silence']
 validate_regions(regions,8)
 for length in [1,128,512,rate*60]:
  tiny=Path(temp)/'tiny.wav';sf.write(tiny,np.zeros(length),rate,subtype='PCM_16')
  times,freq,db,_,duration,stride=calculate_spectrogram(tiny,max_columns=300)
  assert np.isfinite(db).all() and db.min()==db.max()==-80
  assert db.shape[1]<=300 and db.shape[0]==len(freq)
 assert snr.estimate_energy(np.zeros(16000))[0] is None
 assert snr.estimate_energy(np.full(16000,np.nan))[0] is None
 assert snr.estimate_energy(np.array([0.1]))[0] is None
 # With explicit quiet frames, a higher true SNR produces a higher estimate.
 rng=np.random.default_rng(9);n=16000*10;t=np.arange(n)/16000
 speech=.1*np.sin(2*np.pi*200*t);speech[:16000*2]=0
 high,_=snr.estimate_energy(speech+rng.normal(0,.002,n))
 low,_=snr.estimate_energy(speech+rng.normal(0,.02,n))
 assert high>low+10,(high,low)
print('PASS: all four classifier branches (mock model), bounded silence spectrograms, invalid audio SNR, noise-order sanity check with pause frames.')

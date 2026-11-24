"""Exercise the region, segmentation and SNR pages together using controlled speech intervals.

Test guide
----------
wait
    Process Qt events until the active background task finishes.
select_all
    Select every candidate row for a batch review.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from unittest.mock import patch

import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt6.QtCore import QItemSelectionModel
from PyQt6.QtTest import QTest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speech_corpus_app import MainWindow
from audio_review_controller import validate_regions
from audio_spectrogram import calculate_spectrogram
import pause_based_segmentation as cuts
import segment_review as review
import clip_quality_assessment as snr

app=QApplication([])
w=MainWindow(show_welcome=False);p=w.processing;w.show();app.processEvents()
assert not w.outputFolderInput.text()
assert not w.downloadProgressBar.isTextVisible()
for index in (1,2,3):
 w.navigationList.setCurrentRow(index)
 assert w.pages.currentIndex()==index
 assert w.sharedControlsFrame.isVisible()

def wait():
 deadline=time.monotonic()+15
 while p.task is not None and time.monotonic()<deadline:
  app.processEvents();time.sleep(.01)
 assert p.task is None,'Background task timed out'

def select_all(table):
 table.clearSelection()
 for row in range(table.rowCount()):
  table.selectionModel().select(table.model().index(row,0),QItemSelectionModel.SelectionFlag.Select|QItemSelectionModel.SelectionFlag.Rows)

warnings=[]
with tempfile.TemporaryDirectory() as d, patch.object(QMessageBox,'warning',side_effect=lambda *a:warnings.append(a)),patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes),patch.object(w,'open_path'):
 root=Path(d);wav=root/'source.wav';rate=16000
 rng=np.random.default_rng(4);t=np.arange(40*rate)/rate
 x=rng.normal(0,.001,len(t))
 for a,b in [(0.5,6),(6.6,12.2),(20.5,26.5),(27,33),(33.6,39)]:
  mask=(t>=a)&(t<b)
  x[mask]+=.15*np.sin(2*np.pi*190*t[mask])*(.5+.5*np.sin(2*np.pi*2*t[mask])**2)
 sf.write(wav,x,rate,subtype='PCM_16')
 assert p.set_audio(wav)
 p.ui.reviewer.setText('Test reviewer')
 p.regions=[dict(start=0.,end=14.,label='speech',auto_label='speech',confidence=.9,auto_confidence=.9),dict(start=14.,end=20.,label='music',auto_label='music',confidence=.9,auto_confidence=.9),dict(start=20.,end=40.,label='mixed',auto_label='mixed',confidence=.8,auto_confidence=.8)]
 validate_regions(p.regions,40)
 for invalid in [[dict(start=0.,end=float('nan'),label='speech')],[dict(start=1.,end=40.,label='speech')]]:
  try:validate_regions(invalid,40)
  except ValueError:pass
  else:raise AssertionError('Invalid timeline accepted')
 p.refresh_all();p.ui.regionTable.selectRow(0);p.ui.regionEnd.setValue(13.5);p.edit_region()
 assert p.regions[0]['end']==p.regions[1]['start']==13.5
 assert p.changes and p.changes[-1]['annotator']=='Test reviewer'
 p.show_spectrogram();wait()
 assert p.plot_data is not None
 fake_torch=types.ModuleType('torch');fake_torch.set_num_threads=lambda n:None
 fake_vad=types.ModuleType('silero_vad');fake_vad.load_silero_vad=lambda:object()
 def islands(audio,start,end,model,log,pause_ms=200):
  return [[.5,6],[6.6,12.2]] if start<10 else [[20.5,26.5],[27,33],[33.6,39]]
 with patch.dict(sys.modules,torch=fake_torch,silero_vad=fake_vad),patch.object(cuts,'detect_speech',side_effect=islands),patch.object(QFileDialog,'getExistingDirectory',return_value=str(root)):
  p.generate_segments();wait()
 assert p.session and p.session['segments']
 for s in p.session['segments']:
  assert s['original'].get('kind')!='low_energy_fallback'
  assert 5<=s['original']['duration']<=15
  info=sf.info(Path(p.session['candidate_root'])/s['candidate_file'])
  assert (info.samplerate,info.channels,info.subtype)==(16000,1,'PCM_16')
 select_all(p.ui.segmentTable);p.approve()
 assert all(s['status']=='approved' for s in p.session['segments'])
 with patch.object(QFileDialog,'getExistingDirectory',return_value=str(root)):
  p.export_segments();wait()
 manifests=list(root.glob('Segmenting the speech/step4_review_*/step5_manifest.json'));assert len(manifests)==1
 from reviewed_clip_manifest import load_reviewed_clips
 loaded=load_reviewed_clips(manifests[0]);assert len(loaded)==len(p.session['segments'])
 p.calculate_snr();wait();assert len(p.snr_results)==len(loaded)
 p.ui.snrTable.selectRow(0);p.ui.overrideReason.setText('Listening test override');p.override_quality('noisy')
 assert p.snr_results[0]['quality']=='noisy' and p.snr_results[0]['quality_changes']
 p.ui.snrJustification.setText('20 dB retained as the project baseline for this controlled listening workflow.')
 with patch.object(QFileDialog,'getExistingDirectory',return_value=str(root)):p.save_snr()
 assert p.snr_report.is_file()
 result=json.loads(p.snr_report.with_suffix('.json').read_text())
 assert result['second_method']=='wada_snr' and result['methods']==['pause_energy','wada_snr']
 project=root/'session.json'
 with patch.object(QFileDialog,'getSaveFileName',return_value=(str(project),'')):p.save_project()
 expected=copy.deepcopy(p.session);expected_snr=copy.deepcopy(p.snr_results)
 p.session=None;p.snr_results=[]
 with patch.object(QFileDialog,'getOpenFileName',return_value=(str(project),'')):p.open_project()
 assert p.session==expected and p.snr_results==expected_snr and not p.dirty
 # A changed clip invalidates all old SNR results.
 p.ui.segmentTable.selectRow(0);p.ui.segmentLabel.setCurrentText('mixed');p.edit_segment()
 assert not p.snr_results and p.dirty
 assert not warnings,warnings
 p.dirty=False;w.close()
print('PASS: Qt navigation, background spectrogram, adjacent boundaries, strict segmentation exports, format checks, multi-approval, manifest hashes, SNR, manual overrides, reports, session resume, stale-result invalidation.')
print('VAD inference was stubbed with known speech intervals; no PANNs or Silero model accuracy is claimed.')

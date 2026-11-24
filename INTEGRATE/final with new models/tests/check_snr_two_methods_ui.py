"""Two-method Qt workflow with real numeric estimators, controlled source audio.

Test guide
----------
wait
    Process Qt events until the active background task finishes.
"""
import os,sys,tempfile,time,json,csv,copy
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication,QFileDialog,QMessageBox
from speech_corpus_app import MainWindow
import segment_review as review
import clip_quality_assessment as snr
import corpus_review_and_export as corpus
app=QApplication([]);w=MainWindow(show_welcome=False);p=w.processing;errors=[]
def wait():
 deadline=time.monotonic()+15
 while p.task is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
 assert p.task is None,'SNR worker timed out'
 app.processEvents()
with tempfile.TemporaryDirectory() as d,patch.object(QMessageBox,'warning',side_effect=lambda *args:errors.append(args)),patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
 root=Path(d);source=root/'source.wav';rng=np.random.default_rng(25);n=12*16000
 clean=rng.gamma(.4,1,n)*rng.choice([-1.,1.],n);clean*=.025/np.sqrt(np.mean(clean**2));clean[:3200]=0;clean[-3200:]=0
 sf.write(source,clean+rng.normal(0,.0025,n),16000,subtype='PCM_16')
 p.set_audio(source);w.reviewer.setText('Quality reviewer')
 metadata=dict(audio_name=source.name,audio_size=source.stat().st_size,duration_seconds=12.,pause_intervals=[dict(start=0.,end=.2),dict(start=11.8,end=12.)],segments=[dict(file='source.wav',start=0.,end=12.,duration=12.,label='speech',left_padding=.2,right_padding=.2)],unresolved=[])
 path=root/'step4_segments.json';path.write_text(json.dumps(metadata))
 p.set_session(review.load_session(path,source));review.approve_segments(p.session,[0],'Quality reviewer');p.refresh_all()
 w.navigationList.setCurrentRow(3)
 assert w.snrTable.columnCount()==13 and w.snrPolicy.count()==3
 rationale='20 dB starts from the project requirement; reviewed samples will justify the final threshold.'
 w.snrJustification.setText(rationale);w.calculateSNR.click();wait()
 assert len(p.snr_results)==1
 r=p.snr_results[0];assert r['energy_snr_db'] is not None and r['wada_snr_db'] is not None
 assert r['energy_details']['noise_frames']>=10
 assert w.snrTable.item(0,6).text()!='Unavailable' and w.snrTable.item(0,5).text()!='Unavailable'
 w.snrTable.selectRow(0);w.overrideReason.setText('Listened: acceptable speech quality for this test.');w.markClean.click()
 assert p.snr_results[0]['quality']=='clean' and p.snr_results[0]['review_status']=='Human reviewed'
 measurements=(p.snr_results[0]['energy_snr_db'],p.snr_results[0]['wada_snr_db'])
 with patch.object(snr,'calculate',side_effect=AssertionError('Threshold change must not re-read/re-estimate audio')):
  w.snrThreshold.setValue(30.);w.snrPolicy.setCurrentIndex(2)
 assert (p.snr_results[0]['energy_snr_db'],p.snr_results[0]['wada_snr_db'])==measurements
 assert p.snr_results[0]['quality']=='clean' and p.snr_policy=='wada'
 assert p.applied_threshold==30.
 # Explicit recomputation also keeps a named correction for identical samples.
 w.calculateSNR.click();wait();assert p.snr_results[0]['quality']=='clean' and p.snr_results[0]['override_by']=='Quality reviewer'
 with patch.object(QFileDialog,'getExistingDirectory',return_value=str(root)):w.saveSNR.click()
 assert p.snr_report.is_file()
 report=json.loads(p.snr_report.with_suffix('.json').read_text())
 assert report['format']=='speech_corpus_snr_v2' and report['second_method']=='wada_snr'
 assert report['decision_policy']=='wada' and report['threshold_justification']==rationale
 with p.snr_report.with_suffix('.csv').open(encoding='utf-8-sig') as f:row=next(csv.DictReader(f))
 assert row['energy_snr_db'] and row['wada_snr_db'] and row['quality']=='clean'
 saved=copy.deepcopy(p.snr_results);project=root/'session.json'
 with patch.object(QFileDialog,'getSaveFileName',return_value=(str(project),'')):p.save_project()
 p.snr_results=[]
 with patch.object(QFileDialog,'getOpenFileName',return_value=(str(project),'')):p.open_project()
 assert p.snr_results==saved and p.snr_policy=='wada' and p.snr_justification==rationale
 p.asr.sync_current();record=p.asr.records[0]
 assert record['energy_snr_db']==saved[0]['energy_snr_db'] and record['wada_snr_db']==saved[0]['wada_snr_db']
 assert not record['snr_review_required'] and record['snr_audio_signature']==record['audio_signature']
 out=corpus.export_corpus(p.asr.records,{},[],root,True)
 import json
 row=json.loads((Path(out['output_dir'])/'corpus_audit.json').read_text())['segments'][0]
 assert row['energy_snr_db'] is not None and row['wada_snr_db'] is not None and row['snr_threshold_justification']==rationale
 p.dirty=False;w.close();app.processEvents()
assert not errors,errors
print('PASS: real energy/WADA values, Qt worker, both table columns, live threshold/policy without recomputation, persistent human overrides, HTML/CSV/JSON, session restore, ASR handoff and final CSV fields.')
print('Synthetic speech/noise was used; real Windows device playback and listening-based threshold calibration were not exercised.')

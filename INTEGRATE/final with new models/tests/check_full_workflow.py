"""Real Qt/audio/filesystem integration with controlled ASR model outputs.

Test guide
----------
wait
    Process Qt events until the active background task finishes.
select
    Select a review table row before performing an action.
"""
import copy,csv,json,os,sys,tempfile,time,threading
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from PyQt6.QtCore import QItemSelectionModel,Qt
from PyQt6.QtWidgets import QApplication,QFileDialog,QMessageBox
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speech_corpus_app import MainWindow
import segment_review as review
import clip_quality_assessment as snr
import corpus_review_and_export as corpus
import transcript_agreement_and_io as engine

app=QApplication([]);w=MainWindow(show_welcome=False);p=w.processing;a=p.asr;w.show();app.processEvents()
errors=[];opened=[]
def wait():
 deadline=time.monotonic()+20
 while p.task is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
 assert p.task is None,'Task timed out'
 app.processEvents()
def select(table,indices):
 table.clearSelection()
 for i in indices:table.selectionModel().select(table.model().index(i,0),QItemSelectionModel.SelectionFlag.Select|QItemSelectionModel.SelectionFlag.Rows)

with tempfile.TemporaryDirectory() as d,patch.object(QMessageBox,'warning',side_effect=lambda *x:errors.append(x)),patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes),patch.object(w,'open_path',side_effect=opened.append):
 root=Path(d);source=root/'source.wav';rate=16000
 t=np.arange(40*rate)/rate;samples=(.12*np.sin(2*np.pi*(190*t+20*t*t))).astype('float32')
 
 for left,right in [(.5,.65),(6.35,6.5),(20,20.15),(25.85,26),(30,30.15),(35.85,36)]:samples[round(left*rate):round(right*rate)]=0
 samples+=np.random.default_rng(22).normal(0,.001,len(samples)).astype('float32')
 sf.write(source,samples,rate,subtype='PCM_16');samples=sf.read(source,dtype='float32')[0]
 sf.write(root/'first.wav',samples[8000:104000],rate,subtype='PCM_16')
 for name,start in [('second',20),('music',30)]:sf.write(root/(name+'.wav'),samples[start*rate:(start+6)*rate],rate,subtype='PCM_16')
 payload=dict(version=2,audio_name=source.name,audio_size=source.stat().st_size,duration_seconds=40,segments=[
  dict(file='first.wav',start=.5,end=6.5,duration=6.,label='speech',source_spans=[dict(start=.5,end=6.5,clip_start=0.,clip_end=6.)]),
  dict(file='second.wav',start=20.,end=26.,duration=6.,label='mixed'),
  dict(file='music.wav',start=30.,end=36.,duration=6.,label='music')],unresolved=[])
 meta=root/'step4_segments.json';meta.write_text(json.dumps(payload))
 p.set_audio(source);p.ui.reviewer.setText('Integration reviewer');p.regions=[dict(start=0.,end=40.,label='speech')]
 p.session=review.load_session(meta,source);review.approve_segments(p.session,[0,1,2],'Integration reviewer')
 p.ui.snrJustification.setText('20 dB is the baseline for this controlled workflow; quality confirmed by the test reviewer.');p.snr_controls_changed()
 p.snr_results=snr.calculate(snr.session_records(p.session,source),20,justification=p.snr_justification);p.snr_snapshot=copy.deepcopy(p.session)
 for r in p.snr_results:snr.set_quality(r,'noisy' if r['content_label']=='mixed' else 'clean','Integration reviewer','Controlled-source quality review')
 a.sync_current()
 w.navigationList.setCurrentRow(4)
 assert len(a.records)==3 and np.array_equal(corpus.read_audio(a.records[0]),sf.read(root/'first.wav',dtype='float32')[0])
 assert a.ui.asrFinalText.layoutDirection()==Qt.LayoutDirection.RightToLeft
 calls=[]
 def qwen(model,audio):
  calls.append(('a',len(audio)));return ('أهلاً بكم',.9) if len(calls)==1 else ('السلام عليكم',.8)
 def cohere(model,audio,rate):
  assert rate==16000;calls.append(('b',len(audio)))
  return ('اهلا بكم',.85) if sum(k=='b' for k,n in calls)==1 else ('السلام عليك',.75)
 model_a=object();model_b=object();phases=[]
 def release(model):phases.append('release_a' if model is model_a else 'release_b')
 def load_b(*args):
  assert phases==['release_a'],'Cohere loaded before Qwen was released'
  return model_b
 with patch.object(engine,'load_qwen_model',return_value=model_a),patch.object(engine,'load_cohere_model',side_effect=load_b),patch.object(engine,'release_asr_model',side_effect=release),patch.object(engine,'transcribe_qwen',side_effect=qwen),patch.object(engine,'transcribe_cohere',side_effect=cohere):
  a.transcribe(False);wait()
 assert phases==['release_a','release_b']
 assert a.records[0]['asr_b_engine']==engine.COHERE_ENGINE
 assert a.records[0]['models']['b']==engine.DEFAULT_COHERE_MODEL
 assert 'Cohere' in w.transcriptionPageTitle.text() and not hasattr(w,'mmsModel')
 assert len(calls)==4  # Music was excluded; both speech/mixed clips used both models.
 assert [r['clean_noisy'] for r in a.records[:2]]==['clean','noisy']
 assert a.records[0]['status']=='accepted'
 assert a.records[0]['transcript_a']=='أهلاً بكم'
 assert a.records[0]['normalized_transcript_a']==a.records[0]['normalized_transcript_b']=='اهلا بكم'
 assert a.records[0]['comparison_engine']=='jiwer'
 assert a.records[0]['automatic_choice']=='b' and a.records[0]['automatic_transcript']=='اهلا بكم'
 assert a.records[0]['cer_a_vs_b']==a.records[0]['cer_b_vs_a']==0.
 assert a.records[1]['status']=='needs_review'
 assert a.records[2]['status']=='not_applicable'
 autosave=json.loads(a.autosave_path.read_text());assert len(autosave['asr']['records'])==3
 assert autosave['asr']['records'][0]['asr_b_done']
 a.ui.asrTable.selectRow(0);app.processEvents()
 assert w.asrNormalizedA.isReadOnly() and w.asrNormalizedB.isReadOnly()
 assert w.asrTextA.toPlainText()=='أهلاً بكم' and w.asrNormalizedA.toPlainText()=='اهلا بكم'
 assert '0.00%' in w.asrComparison.text()
 assert w.asrTable.columnCount()==12
 assert '0.00%' in w.asrCerA.text() and '0.00%' in w.asrCerB.text()
 assert w.asrAutomaticChoice.text().endswith('accepted') and 'Identical' in w.asrAutomaticReason.text()
 a.ui.asrTable.selectRow(1);a.choose('b');assert a.records[1]['final_transcript']=='السلام عليك'
 a.ui.asrTable.selectRow(1);app.processEvents()
 assert a.records[1]['automatic_choice'] is None and a.records[1]['automatic_transcript']==''
 assert w.asrAutomaticChoice.text().endswith('needs_review')
 assert 'Current decision: accepted' in w.asrComparison.text()
 assert a.audit_log and a.records[1]['human_override']
 assert not list(root.glob('*_transcripts.csv'))
 # Rerun one clip, stop after Qwen, then resume only its missing Cohere result.
 a.ui.asrTable.selectRow(0)
 started=threading.Event();release=threading.Event()
 def stop_after_a(model,audio):started.set();release.wait(3);return 'اهلا بكم',.9
 with patch.object(engine,'load_qwen_model',return_value=object()),patch.object(engine,'transcribe_qwen',side_effect=stop_after_a),patch.object(engine,'load_cohere_model') as load_b:
  a.transcribe(True)
  deadline=time.monotonic()+3
  while not started.is_set() and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
  assert started.is_set() and w.asrStop.isVisible() and w.asrStop.isEnabled()
  assert not w.pages.isEnabled()
  w.asrStop.click();release.set();wait();load_b.assert_not_called()
 assert a.records[0]['asr_a_done'] and not a.records[0]['asr_b_done']
 with patch.object(engine,'load_qwen_model') as load_a,patch.object(engine,'load_cohere_model',return_value=object()),patch.object(engine,'transcribe_cohere',return_value=('اهلا بكم',.8)):
  a.transcribe(False);wait();load_a.assert_not_called()
 assert a.records[0]['asr_b_done']
 session=root/'complete.json'
 with patch.object(QFileDialog,'getSaveFileName',return_value=(str(session),'')):p.save_project()
 expected=copy.deepcopy(a.records)
 # Legacy cached comparison scores/text are refreshed from original outputs;
 # a named human transcript choice remains intact without model inference.
 stale=json.loads(session.read_text())
 for r in stale['asr']['records']:
  r.update(normalization_version='old',normalized_transcript_a='stale',normalized_transcript_b='stale',cer=99.,wer=99.)
 stale['asr']['records'][0].update(status='difficult',transcript_status='difficult',final_transcript='stale')
 session.write_text(json.dumps(stale,ensure_ascii=False))
 a.clear()
 with patch.object(QFileDialog,'getOpenFileName',return_value=(str(session),'')):p.open_project()
 assert a.records==expected and a.signature==corpus.source_signature(p)
 # Open an earlier MMS session: retain A/human edits, archive old B, and run
 # only Cohere. A legacy text must never be relabeled as Cohere output.
 old=json.loads(session.read_text());old['asr'].pop('cohere_model',None)
 old['asr']['mms_model']='facebook/mms-1b-all'
 for r in old['asr']['records'][:2]:
  r['models']['b']='facebook/mms-1b-all';r.pop('asr_b_engine',None)
 legacy=root/'legacy.json';legacy.write_text(json.dumps(old,ensure_ascii=False))
 with patch.object(QFileDialog,'getOpenFileName',return_value=(str(legacy),'')):p.open_project()
 assert all(r['asr_a_done'] and not r['asr_b_done'] for r in a.records[:2])
 assert all(r['legacy_asr_results'][-1]['model']=='facebook/mms-1b-all' for r in a.records[:2])
 assert a.records[1]['final_transcript']=='السلام عليك' and a.records[1]['human_override']
 with patch.object(engine,'load_qwen_model') as load_a,patch.object(engine,'load_cohere_model',return_value=object()),patch.object(engine,'transcribe_cohere',side_effect=[('اهلا بكم',None),('السلام عليك',None)]):
  a.transcribe(False);wait();load_a.assert_not_called()
 assert all(r['asr_b_done'] and r['asr_b_engine']==engine.COHERE_ENGINE for r in a.records[:2])
 # Full output includes exact approved continuous source audio plus UTF-8 text/CSV.
 with patch.object(QFileDialog,'getExistingDirectory',return_value=str(root)):a.export();wait()
 out=opened[-1];rows=list(csv.DictReader((out/'corpus.csv').open(encoding='utf-8-sig')))
 assert len(rows)==3 and rows[1]['final_transcript']=='السلام عليك'
 assert not (out/'approved_transcripts.csv').exists()
 assert tuple(rows[0])==corpus.REVIEW_COLUMNS
 assert rows[0]['model_A_transcript'] and rows[0]['model_B_transcript']
 assert list(json.loads((out/'corpus.json').read_text())['segments'][0])==list(corpus.REVIEW_COLUMNS)
 audit=json.loads((out/'corpus_audit.json').read_text())['segments'][0]
 assert audit['cer_a_vs_b']==0. and audit['models']['b']==engine.DEFAULT_COHERE_MODEL
 assert audit['legacy_asr_results'][0]['model']=='facebook/mms-1b-all'
 assert np.array_equal(sf.read(out/'segment_00001.wav',dtype='float32')[0],sf.read(root/'first.wav',dtype='float32')[0])
 assert (out/'corpus_report.html').is_file() and (out/'source_review.json').is_file()
 assert (out/'segment_00002.txt').read_text(encoding='utf-8')=='السلام عليك'
 assert json.loads((out/'corpus.json').read_text())['metadata']['partial'] is False
 # Structural changes preserve exact source samples and invalidate old annotations.
 select(a.ui.finalTable,[0,1]);before=np.concatenate([corpus.read_audio(a.records[0]),corpus.read_audio(a.records[1])]);a.merge()
 merged=a.records[0];assert merged['duration']==12 and not merged['asr_a_done'] and merged['snr_db'] is None
 assert np.array_equal(corpus.read_audio(merged),before)
 a.ui.finalTable.selectRow(0);a.ui.finalSplitPosition.setValue(6);a.split()
 assert len(a.records)==3 and a.records[0]['duration']==a.records[1]['duration']==6
 assert np.array_equal(corpus.read_audio(a.records[0]),before[:6*rate])
 a.ui.finalTable.selectRow(0);a.delete();assert len(a.records)==2
 a.ui.finalStart.setValue(0);a.ui.finalEnd.setValue(6);a.ui.finalLabel.setCurrentText('speech');a.add();assert len(a.records)==3
 try:corpus.export_corpus(a.records,{},a.audit_log,root)
 except ValueError:pass
 else:raise AssertionError('Incomplete annotations exported as complete')
 partial=corpus.export_corpus(a.records,{},a.audit_log,root,True);assert partial['partial']
 assert not errors,errors
 p.dirty=False;w.close()
print('PASS: approved continuous source audio/natural context, two-engine Arabic ASR orchestration, disagreement review, cancellation/resume, per-result autosave, combined-session restore, WAV/TXT/CSV export, structural edits, result invalidation and partial-export gating.')
print('ASR outputs were controlled test doubles. Actual Qwen/Cohere downloads and model accuracy were not exercised.')

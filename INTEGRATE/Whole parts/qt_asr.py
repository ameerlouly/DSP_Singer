"""Whisper/MMS and final annotation controllers for the unified main_window.ui."""
import copy
import gc
import json
from pathlib import Path
import tempfile
import threading
import uuid

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import soundfile as sf
import asr_pipeline as engines
import corpus_review as corpus
import step5_snr as snr


class ASRWorkspace(QObject):
    piece=pyqtSignal(object)
    progress=pyqtSignal(object)

    def __init__(self,processing):
        super().__init__(processing)
        self.p=processing;self.w=processing.window;self.ui=self.w
        self.records=[];self.audit_log=[];self.signature=None;self.source_metadata={}
        self.stop_event=threading.Event();self.asr_running=False;self.autosave_path=None
        self.current_asr=None;self.current_final=None
        self.transcript_draft=False;self.final_draft=False
        self.playback_dir=tempfile.TemporaryDirectory(prefix='speech_corpus_play_')
        self.p.table(self.ui.asrTable);self.p.table(self.ui.finalTable)
        for text in (self.ui.asrTextA,self.ui.asrTextB,self.ui.asrNormalizedA,self.ui.asrNormalizedB,self.ui.asrFinalText,self.ui.finalText):
            option=text.document().defaultTextOption();option.setAlignment(Qt.AlignmentFlag.AlignRight)
            option.setTextDirection(Qt.LayoutDirection.RightToLeft);text.document().setDefaultTextOption(option)
        self.ui.asrTable.itemSelectionChanged.connect(self.select_asr)
        self.ui.finalTable.itemSelectionChanged.connect(self.select_final)
        self.ui.asrFinalText.textChanged.connect(self.mark_transcript_draft)
        self.ui.finalText.textChanged.connect(self.mark_final_draft)
        for control in (self.ui.finalStart,self.ui.finalEnd):control.valueChanged.connect(self.mark_final_draft)
        for control in (self.ui.finalLabel,self.ui.finalQuality,self.ui.finalStatus):control.currentIndexChanged.connect(self.mark_final_draft)
        actions={
            'asrSync':self.sync_current,'asrSelected':lambda:self.transcribe(True),'asrPending':lambda:self.transcribe(False),
            'asrOpenSession':self.p.open_project,'asrPlay':lambda:self.play_table(self.ui.asrTable),
            'asrUseA':lambda:self.choose('a'),'asrUseB':lambda:self.choose('b'),
            'asrAccept':lambda:self.commit_transcript('accepted'),'asrDifficult':lambda:self.commit_transcript('difficult'),
            'asrGoReview':lambda:self.w.navigationList.setCurrentRow(5),'finalToASR':lambda:self.w.navigationList.setCurrentRow(4),
            'finalApply':self.commit_final,'finalPlay':lambda:self.play_table(self.ui.finalTable),
            'finalSplit':self.split,'finalMerge':self.merge,'finalAdd':self.add,'finalDelete':self.delete,
            'finalRecalculate':self.recalculate,'finalExport':self.export,
        }
        for name,action in actions.items():getattr(self.ui,name).clicked.connect(lambda checked=False,a=action:self.p.guard(a))
        self.ui.asrStop.clicked.connect(self.cancel)
        self.piece.connect(self.receive_piece);self.progress.connect(self.receive_progress)
        self.refresh()

    def find(self,ident):return next((r for r in self.records if r['segment_id']==ident),None)

    def mark_transcript_draft(self):self.transcript_draft=True;self.p.dirty=True
    def mark_final_draft(self,*args):self.final_draft=True;self.p.dirty=True

    def set_text(self,widget,text):
        widget.blockSignals(True);widget.setPlainText(text or '');widget.blockSignals(False)

    def clear(self):
        self.records=[];self.audit_log=[];self.signature=None;self.source_metadata={}
        self.current_asr=self.current_final=None;self.transcript_draft=self.final_draft=False
        for widget in (self.ui.asrTextA,self.ui.asrTextB,self.ui.asrNormalizedA,self.ui.asrNormalizedB,self.ui.asrFinalText,self.ui.finalText):self.set_text(widget,'')
        self.ui.asrComparison.setText('Select a clip to compare its transcripts.')
        self.refresh()

    def source_changed(self):
        if self.records:self.ui.asrInfo.setText('Earlier clips/quality changed. Use current approved clips again before ASR or export.')

    def check_current(self):
        if not self.records:raise ValueError('Use current approved clips on Transcription first.')
        if corpus.source_signature(self.p)!=self.signature:
            raise ValueError('Source clips or quality changed. Use current approved clips to synchronize before continuing.')

    def sync_current(self):
        self.commit_drafts()
        fresh=corpus.from_processing(self.p)
        fresh_by_id={r['segment_id']:r for r in fresh}
        replaced=[r for r in self.records if r['segment_id'] not in fresh_by_id or r.get('audio_signature')!=fresh_by_id[r['segment_id']]['audio_signature']]
        if replaced and QMessageBox.question(self.w,'Replace final clip edits',
            f'{len(replaced)} final-review clips differ from the earlier approved clips and will be replaced. Save a session first to keep those edits. Continue?',
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:
            return
        old={r['segment_id']:r for r in self.records};kept=0
        for i,r in enumerate(fresh):
            previous=old.get(r['segment_id'])
            if previous and previous.get('audio_signature')==r['audio_signature']:
                quality={key:copy.deepcopy(value) for key,value in r.items() if key.startswith('snr_') or key in ('energy_snr_db','wada_snr_db','clean_noisy','quality_history')}
                if previous.get('snr_override_by'):
                    analysis=copy.deepcopy(r.get('snr_analysis') or {})
                    snr.preserve_override(analysis,previous.get('snr_analysis'))
                    if analysis.get('override_by'):quality.update(snr.corpus_fields(analysis))
                fresh[i]=copy.deepcopy(previous);fresh[i].update(quality);kept+=1
        for record in fresh:corpus.compare(record)
        self.records=fresh;self.signature=corpus.source_signature(self.p)
        source=self.p.audio.parent/'metadata.json'
        self.source_metadata=json.loads(source.read_text(encoding='utf-8')) if source.is_file() else {}
        self.current_asr=self.current_final=None;self.transcript_draft=self.final_draft=False
        self.p.dirty=True;self.refresh()
        self.ui.asrInfo.setText(f'Loaded {len(fresh)} approved clips; retained {kept} unchanged reviews. Both clean and noisy speech/mixed clips are included.')

    def payload(self):
        return dict(records=copy.deepcopy(self.records),audit_log=copy.deepcopy(self.audit_log),signature=self.signature,
                    source_metadata=self.source_metadata,whisper_model=self.ui.whisperModel.text().strip(),mms_model=self.ui.mmsModel.text().strip(),
                    asr_comparison_policy=engines.comparison_policy())

    def restore(self,data):
        self.clear()
        if not data:return
        records=data.get('records',[])
        if not isinstance(records,list):raise ValueError('Saved ASR records are invalid.')
        ids=[]
        for r in records:
            ids.append(r['segment_id'])
            if not 5-1/16000<=r['duration']<=15+1/16000:raise ValueError('A saved final clip has invalid duration.')
            if corpus.audio_signature(r)!=r['audio_signature']:raise ValueError('An ASR source audio file changed or moved.')
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate final clip IDs in saved session.')
        for record in records:corpus.compare(record)
        self.records=records;self.audit_log=data.get('audit_log',[]);self.signature=data.get('signature')
        self.source_metadata=data.get('source_metadata',{})
        self.ui.whisperModel.setText(data.get('whisper_model','large-v3'));self.ui.mmsModel.setText(data.get('mms_model','facebook/mms-1b-all'))
        self.refresh()

    def refresh(self):
        self.p.populate(self.ui.asrTable,[[r['segment_id'],f"{r['start']:.3f}",f"{r['end']:.3f}",f"{r['duration']:.3f}",r['label'],r.get('clean_noisy',''),
            'Done' if r.get('asr_a_done') else 'Pending','Done' if r.get('asr_b_done') else 'Pending',r['status']] for r in self.records])
        self.p.populate(self.ui.finalTable,[[r['segment_id'],f"{r['start']:.3f}",f"{r['end']:.3f}",f"{r['duration']:.3f}",r['label'],
            '—' if r.get('snr_db') is None else f"{r['snr_db']:.2f}",r.get('clean_noisy',''),r['status'],r.get('final_transcript','')] for r in self.records])
        counts={s:sum(r['status']==s for r in self.records) for s in ['accepted','needs_review','difficult','not_applicable']}
        self.ui.finalSummary.setText('Clips: '+str(len(self.records))+' · '+' · '.join(f'{k}: {v}' for k,v in counts.items()))

    def selected(self,table):
        rows=self.p.selected(table)
        if not rows:raise ValueError('Select one or more clips.')
        return [self.records[i] for i in rows]

    def select_asr(self):
        rows=self.p.selected(self.ui.asrTable)
        if len(rows)!=1:return
        if self.transcript_draft and self.current_asr:
            try:self.commit_transcript('accepted' if self.ui.asrFinalText.toPlainText().strip() else 'needs_review',refresh=False)
            except Exception as exc:QMessageBox.warning(self.w,'Save transcript first',str(exc));return
        r=self.records[rows[0]];self.current_asr=r['segment_id']
        self.set_text(self.ui.asrTextA,r.get('transcript_a',''));self.set_text(self.ui.asrTextB,r.get('transcript_b',''));self.set_text(self.ui.asrFinalText,r.get('final_transcript',''))
        self.set_text(self.ui.asrNormalizedA,engines.normalize_arabic(r.get('transcript_a','')))
        self.set_text(self.ui.asrNormalizedB,engines.normalize_arabic(r.get('transcript_b','')))
        self.transcript_draft=False
        fmt=lambda v:'—' if v is None else f'{v:.3f}'
        percent=lambda v:'—' if v is None else f'{v:.2%}'
        self.ui.asrComparison.setText(f"Decision: {r['status']} · Normalized CER: {percent(r.get('cer'))} · WER: {percent(r.get('wer'))} · word edits: {r.get('word_difference')} · A score: {fmt(r.get('confidence_a'))} · B score: {fmt(r.get('confidence_b'))}\n{r.get('agreement_reason','Waiting for both ASR attempts.')} Scores compare A to B; they are not accuracy against a human transcript.")

    def select_final(self):
        rows=self.p.selected(self.ui.finalTable)
        if len(rows)!=1:return
        if self.final_draft and self.current_final:
            try:self.commit_final(refresh=False)
            except Exception as exc:QMessageBox.warning(self.w,'Save final edit first',str(exc));return
        r=self.records[rows[0]];self.current_final=r['segment_id']
        fields=[(self.ui.finalStart,r['start']),(self.ui.finalEnd,r['end'])]
        for field,value in fields:field.blockSignals(True);field.setValue(value);field.blockSignals(False)
        for field,value in [(self.ui.finalLabel,r['label']),(self.ui.finalQuality,r.get('clean_noisy','unavailable')),(self.ui.finalStatus,r['status'])]:
            field.blockSignals(True);field.setCurrentText(value);field.blockSignals(False)
        self.set_text(self.ui.finalText,r.get('final_transcript',''));self.final_draft=False
        self.ui.finalMapping.setText('Source mapping: '+', '.join(f"{p['start']:.3f}–{p['end']:.3f}s" for p in r['source_spans']))

    def changed(self,refresh=True):
        self.p.dirty=True
        if refresh:self.refresh()

    def choose(self,key):
        self.p.index(self.ui.asrTable)
        r=self.find(self.current_asr)
        if r is None:raise ValueError('Select one ASR row.')
        text=r.get('transcript_'+key,'')
        if not text:raise ValueError('That model produced no text for this clip.')
        self.set_text(self.ui.asrFinalText,text);self.commit_transcript('accepted')

    def commit_transcript(self,status,refresh=True):
        if refresh:self.p.index(self.ui.asrTable)
        r=self.find(self.current_asr)
        if r is None:raise ValueError('Select one ASR row.')
        who=self.p.reviewer();text=self.ui.asrFinalText.toPlainText().strip()
        if status=='accepted' and not engines.normalize_arabic(text):raise ValueError('Accepted speech needs final text, not only punctuation or diacritics.')
        before=copy.deepcopy(r)
        r.update(final_transcript=text,selected_transcript=text,status=status,transcript_status=status,human_override=True)
        corpus.audit(self.audit_log,'transcript_review',who,before,r)
        self.transcript_draft=False;self.changed(refresh)

    def commit_final(self,refresh=True):
        if refresh:self.p.index(self.ui.finalTable)
        r=self.find(self.current_final)
        if r is None:raise ValueError('Select one final-review row.')
        corpus.edit(r,self.p.audio,self.p.duration,self.ui.finalStart.value(),self.ui.finalEnd.value(),
            self.ui.finalLabel.currentText(),self.ui.finalQuality.currentText(),self.ui.finalText.toPlainText(),
            self.ui.finalStatus.currentText(),self.p.reviewer(),self.audit_log,self.ui.finalReason.text())
        self.final_draft=False;self.changed(refresh)
        if refresh:self.p.status('Final edit saved. If audio boundaries changed, recalculate SNR and rerun ASR.')

    def commit_drafts(self):
        if self.transcript_draft and self.current_asr:self.commit_transcript('accepted' if self.ui.asrFinalText.toPlainText().strip() else 'needs_review',False)
        if self.final_draft and self.current_final:self.commit_final(False)

    def transcribe(self,selected):
        self.commit_drafts()
        if not self.records:self.sync_current()
        self.check_current()
        chosen=self.selected(self.ui.asrTable) if selected else self.records
        ids={r['segment_id'] for r in chosen if r['label'] in corpus.VOICE}
        if not ids:raise ValueError('Select speech or mixed clips for transcription.')
        whisper=self.ui.whisperModel.text().strip();mms=self.ui.mmsModel.text().strip()
        if not whisper or not mms:raise ValueError('Enter both model names.')
        records=copy.deepcopy(self.records);jobs={}
        for r in records:
            if r['segment_id'] not in ids:continue
            keys=[key for key in ('a','b') if selected or not r.get('asr_'+key+'_done')]
            if keys:jobs[r['segment_id']]=keys
            if selected:
                r['asr_a_done']=r['asr_b_done']=False
        if not jobs:self.p.status('Both ASR engines have already run. Use Transcribe selected to rerun them.');return
        self.stop_event.clear();self.asr_running=True;self.ui.asrStop.setVisible(True)
        total=sum(map(len,jobs.values()));self.ui.asrProgress.setRange(0,total);self.ui.asrProgress.setValue(0)
        snapshot=self.p.project_payload()
        base_state=self.payload();base_state['records']=records
        self.autosave_path=self.p.audio.with_name(self.p.audio.stem+'_full_autosave.json')
        autosave=self.autosave_path
        def persist():
            from qt_processing import atomic_json
            snapshot['asr']=copy.deepcopy(base_state);atomic_json(autosave,snapshot)
        def action(log):
            completed=0;errors=[]
            persist()
            for key,name in [('a',whisper),('b',mms)]:
                work=[r for r in records if key in jobs.get(r['segment_id'],[])]
                if not work or self.stop_event.is_set():continue
                model=None
                try:
                    loader=engines.load_whisper_model if key=='a' else engines.load_mms_model
                    model=loader(name,log)
                    for r in work:
                        if self.stop_event.is_set():break
                        log(f"{name}: {r['segment_id']}")
                        try:
                            if corpus.audio_signature(r)!=r['audio_signature']:raise ValueError('Clip audio changed since review.')
                            audio=corpus.read_audio(r)
                            text,confidence=engines.transcribe_whisper(model,audio) if key=='a' else engines.transcribe_mms(model,audio,16000)
                            r.update(**{'transcript_'+key:text,'confidence_'+key:confidence,'asr_'+key+'_done':True})
                            r.setdefault('asr_errors',{}).pop(key,None)
                            r.setdefault('models',{})[key]=name
                            corpus.compare(r)
                        except Exception as exc:
                            r.setdefault('asr_errors',{})[key]=str(exc);errors.append(f"{r['segment_id']} {key}: {exc}")
                            r['asr_'+key+'_done']=False
                        # Save before beginning the next clip/model. Failure stops work.
                        persist();self.piece.emit(copy.deepcopy(r));completed+=1;self.progress.emit((completed,total))
                except Exception as exc:
                    # Persistence errors must stop the entire job instead of continuing.
                    if isinstance(exc,OSError):raise
                    errors.append(f'{name}: {exc}');log('ASR model error: '+str(exc))
                    for r in work:
                        if not r.get('asr_'+key+'_done'):r.setdefault('asr_errors',{})[key]=str(exc)
                    persist()
                finally:
                    if model is not None:del model
                    gc.collect()
            persist()
            return dict(records=records,errors=errors,cancelled=self.stop_event.is_set(),completed=completed,total=total)
        def done(result):
            self.records=result['records'];self.changed()
            state='Stopped after the current clip' if result['cancelled'] else 'Transcription finished'
            self.ui.asrInfo.setText(f"{state}: {result['completed']} / {result['total']} model/clip jobs. Errors: {len(result['errors'])}. Autosave: {autosave}")
            self.p.status(self.ui.asrInfo.text())
            if result['errors']:self.w.append_log('ASR errors: '+'; '.join(result['errors']))
        try:self.p.start('Transcribing approved speech/mixed clips…',action,done)
        except Exception:
            self.task_finished();raise

    def receive_piece(self,record):
        for i,r in enumerate(self.records):
            if r['segment_id']==record['segment_id']:self.records[i]=record;break
        self.p.dirty=True;self.refresh()

    def receive_progress(self,value):self.ui.asrProgress.setValue(value[0])

    def cancel(self):
        self.stop_event.set();self.ui.asrStop.setEnabled(False)
        self.p.status('Stopping after the current model/clip finishes; completed results remain autosaved.')

    def task_finished(self):
        self.asr_running=False;self.ui.asrStop.hide();self.ui.asrStop.setEnabled(True)

    def play_table(self,table):
        records=self.selected(table)
        if len(records)!=1:raise ValueError('Select one clip to play.')
        record=records[0];audio=corpus.read_audio(record)
        path=Path(self.playback_dir.name)/(uuid.uuid4().hex+'.wav')
        sf.write(path,audio,16000,subtype='PCM_16');self.p.play(path,0,len(audio)/16000)

    def split(self):
        self.commit_drafts();r=self.selected(self.ui.finalTable)
        if len(r)!=1:raise ValueError('Select one clip to split.')
        who=self.p.reviewer();parts=corpus.split(r[0],self.ui.finalSplitPosition.value())
        index=self.records.index(r[0]);before=copy.deepcopy(r[0]);self.records[index:index+1]=parts
        corpus.audit(self.audit_log,'split_final_clip',who,before,parts);self.current_final=None;self.changed()

    def merge(self):
        self.commit_drafts();selected=self.selected(self.ui.finalTable);who=self.p.reviewer();merged=corpus.merge(selected)
        before=copy.deepcopy(selected);ids={r['segment_id'] for r in selected}
        self.records=[r for r in self.records if r['segment_id'] not in ids]+[merged];self.records.sort(key=lambda r:r['start'])
        corpus.audit(self.audit_log,'merge_final_clips',who,before,merged);self.current_final=None;self.changed()

    def add(self):
        self.commit_drafts();self.p.require_audio();who=self.p.reviewer()
        record=corpus.new_clip(self.p.audio,self.ui.finalStart.value(),self.ui.finalEnd.value(),self.ui.finalLabel.currentText(),self.p.duration)
        self.records.append(record);self.records.sort(key=lambda r:r['start'])
        corpus.audit(self.audit_log,'add_final_clip',who,None,record);self.changed()

    def delete(self):
        self.commit_drafts();selected=self.selected(self.ui.finalTable);who=self.p.reviewer();ids={r['segment_id'] for r in selected}
        corpus.audit(self.audit_log,'delete_final_clips',who,selected,None)
        self.records=[r for r in self.records if r['segment_id'] not in ids];self.current_final=None;self.changed()

    def recalculate(self):
        self.commit_drafts();selected=self.selected(self.ui.finalTable);who=self.p.reviewer();threshold=self.ui.finalSNRThreshold.value()
        policy=self.p.snr_policy;justification=self.p.ui.snrJustification.text().strip()
        snapshot=copy.deepcopy(selected)
        def done(result):
            by_id={r['segment_id']:r for r in result}
            before=copy.deepcopy(self.records)
            self.records=[by_id.get(r['segment_id'],r) for r in self.records]
            corpus.audit(self.audit_log,'recalculate_final_snr',who,before,self.records,f'Threshold {threshold:g} dB; pause energy + WADA; rule {policy}')
            self.changed()
        self.p.start('Calculating pause energy + WADA-SNR for final-review clips…',lambda log:corpus.recalculate_quality(snapshot,threshold,policy,justification),done)

    def export(self):
        self.commit_drafts();self.check_current()
        directory=QFileDialog.getExistingDirectory(self.w,'Export final corpus',str(self.p.audio.parent))
        if not directory:return
        records=copy.deepcopy(self.records);audit=copy.deepcopy(self.audit_log)
        metadata=dict(source=self.source_metadata,audio=str(self.p.audio),
                      source_review=dict(region_changes=self.p.changes,step4=self.p.session,snr=self.p.snr_results),
                      whisper_model=self.ui.whisperModel.text(),mms_model=self.ui.mmsModel.text(),
                      pending_source_candidates=sum(r['status']=='pending' for r in self.p.session['segments']),
                      unresolved_source_spans=sum(r['status']!='reviewed' for r in self.p.session['unresolved']))
        partial=self.ui.allowPartialExport.isChecked()
        def done(result):
            self.p.status(f"Exported {result['clips']} clips to {result['output_dir']}. Partial: {result['partial']}.")
            self.w.open_path(Path(result['output_dir']))
        self.p.start('Writing corpus WAV, TXT, CSV and audit data…',lambda log:corpus.export_corpus(records,metadata,audit,directory,partial),done)

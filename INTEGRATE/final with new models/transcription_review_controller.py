"""Let the reviewer listen, compare model outputs and decide what text to keep.

ASRWorkspace starts transcription and updates each row as results arrive. Its commit
methods save typed edits before the user switches clips. Final review is read-only
and handles playback, summary workbooks and corpus export.
"""

from output_folders import metadata_folder, new_run_folder, save_dialog_folder, step_folder
import copy
import gc
import json
from pathlib import Path
import tempfile
import threading
import uuid

from PyQt6.QtCore import QObject, Qt, QSettings, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from PyQt6.QtMultimedia import QMediaPlayer
import soundfile as sf
import transcript_agreement_and_io as engines
import corpus_review_and_export as corpus
import clip_quality_assessment as snr


class ASRWorkspace(QObject):
    piece=pyqtSignal(object)
    progress=pyqtSignal(object)

    # Connect the transcript comparison and final review pages to the same clip records.
    def __init__(self,processing):
        super().__init__(processing)
        self.p=processing;self.w=processing.window;self.ui=self.w
        self.records=[];self.audit_log=[];self.signature=None;self.source_metadata={}
        self.transcribe_after_sync=None
        self.stop_event=threading.Event();self.asr_running=False;self.autosave_path=None
        self.current_asr=None;self.current_final=None
        self.transcript_draft=False
        self.playback_dir=tempfile.TemporaryDirectory(prefix='speech_corpus_play_')
        self.asr_preview = None
        self.asr_preview_file = None
        self.p.table(self.ui.asrTable);self.p.table(self.ui.finalTable);self.p.table(self.ui.processSummaryTable)
        for column, key in enumerate(corpus.REVIEW_COLUMNS):
            if 'transcript' in key or key in ('audio_file', 'change_log'):
                self.ui.finalTable._table_sizing.preferred[column] = 300 if 'transcript' in key else 200
        for text in (self.ui.asrTextA,self.ui.asrTextB,self.ui.asrNormalizedA,self.ui.asrNormalizedB,self.ui.asrFinalText,self.ui.finalText):
            option=text.document().defaultTextOption();option.setAlignment(Qt.AlignmentFlag.AlignRight)
            option.setTextDirection(Qt.LayoutDirection.RightToLeft);text.document().setDefaultTextOption(option)
        self.ui.asrTable.itemSelectionChanged.connect(self.select_asr)
        self.ui.finalTable.itemSelectionChanged.connect(self.select_final)
        self.ui.asrFinalText.textChanged.connect(self.mark_transcript_draft)
        actions={
            'asrSync':self.sync_current,'asrSelected':lambda:self.transcribe(True),'asrPending':lambda:self.transcribe(False),
            'asrOpenSession':self.p.open_project,'asrPlay':self.play_asr_cursor,
            'asrPausePreview':self.pause_asr_preview,'asrStopPreview':self.stop_asr_preview,
            'asrUseA':lambda:self.choose('a'),'asrUseB':lambda:self.choose('b'),
            'asrAccept':lambda:self.commit_transcript('accepted'),'asrDifficult':lambda:self.commit_transcript('difficult'),
            'asrGoReview':lambda:self.w.navigationList.setCurrentRow(5),'finalToASR':lambda:self.w.navigationList.setCurrentRow(4),
            'asrNextSegment':self.next_segment,'finalPlay':lambda:self.play_table(self.ui.finalTable),
            'finalExport':self.export,'finalSaveReview':self.save_review,
            'saveProcessSummary':self.save_process_summary,
        }
        for name,action in actions.items():getattr(self.ui,name).clicked.connect(lambda checked=False,a=action:self.p.guard(a))
        self.ui.asrSeekSlider.valueChanged.connect(self.asr_slider_changed)
        self.ui.asrSeekSlider.sliderReleased.connect(lambda: self.p.guard(self.seek_asr_preview))
        self.ui.asrStop.clicked.connect(self.cancel)
        self.piece.connect(self.receive_piece);self.progress.connect(self.receive_progress)
        self.refresh()

    # Look up a clip by its segment ID.
    def find(self,ident):return next((r for r in self.records if r['segment_id']==ident),None)

    # Remember that the text on the transcription page has unsaved edits.
    def mark_transcript_draft(self):self.transcript_draft=True;self.p.dirty=True

    # Set the displayed text without counting it as an edit by the reviewer.
    def set_text(self,widget,text):
        widget.blockSignals(True);widget.setPlainText(text or '');widget.blockSignals(False)

    # Drop the loaded records and reset both review pages.
    def clear(self):
        self.records=[];self.audit_log=[];self.signature=None;self.source_metadata={}
        self.current_asr=self.current_final=None;self.transcript_draft=False
        self.clear_editors()
        self.refresh()

    # Empty the text fields so they don't show words from the previous selection.
    def clear_editors(self):
        self.configure_asr_preview(None)
        for widget in (self.ui.asrTextA,self.ui.asrTextB,self.ui.asrNormalizedA,self.ui.asrNormalizedB,self.ui.asrFinalText,self.ui.finalText):self.set_text(widget,'')
        self.ui.asrSelectedStatus.setText('Segment — · Transcription status: —')
        self.ui.asrNextSegment.setEnabled(False)
        self.ui.finalMapping.setText('Select a segment to review its final transcript.')
        self.ui.finalPlay.setEnabled(False)
        self.ui.asrComparison.setText('Select a clip to compare its transcripts.')
        self.ui.asrCerA.setText('Qwen CER vs Cohere (normalized): —')
        self.ui.asrCerB.setText('Cohere CER vs Qwen (normalized): —')
        self.ui.asrAutomaticChoice.setText('Automatic decision before human editing: —')
        self.ui.asrAutomaticReason.setText('Select a clip to see the agreement result and reason.')

    # Tell the reviewer when upstream edits have changed the clip set.
    def source_changed(self):
        if self.records:self.ui.asrInfo.setText('Earlier clips/quality changed. Load clips and unresolved spans again before ASR or export.')

    # Check that the review still belongs to the current recording and clips.
    def check_current(self):
        if not self.records:raise ValueError('Load clips and unresolved spans on Transcription first.')
        if corpus.source_signature(self.p)!=self.signature:
            raise ValueError('Source clips or quality changed. Load clips and unresolved spans to synchronize before continuing.')

    # Save any typing before loading the current clips and unresolved spans.
    def sync_current(self, *, transcribe_after=None):
        self.commit_drafts()
        model_b=self.ui.cohereModel.text().strip()
        # Use the refreshed rows before starting a queued transcription.
        def done(result):
            if self.apply_sync(result):
                self.transcribe_after_sync=transcribe_after if self.records else None
            else:
                self.p.auto_stage=None
        self.p.start('Loading clips for transcription…',lambda log:self.prepare_sync(model_b),done)

    # Match incoming source intervals with existing transcripts so usable work survives a
    # refresh.
    def prepare_sync(self,model_b):
        # Source data is read-only while project controls are locked.
        fresh=corpus.from_processing(self.p)
        if not fresh and not self.p.session.get('unresolved'):
            raise ValueError('No approved clips or unresolved spans to load.')
        fresh_by_id={r['segment_id']:r for r in fresh}
        # Identify the same source interval by its file and sample-aligned endpoints.
        def interval(r):
            return (r['source_origin'], round(r['start']*corpus.RATE), round(r['end']*corpus.RATE))
        previews = {interval(r): r for r in self.records if corpus.is_preview(r)
                    and Path(r['source_origin']).is_file() and corpus.audio_signature(r) == r['audio_signature']}
        promotions = {r['segment_id']: previews[interval(r)] for r in fresh
                      if not corpus.is_preview(r) and r['label'] in corpus.VOICE and interval(r) in previews}
        promoted = {r['segment_id'] for r in promotions.values()}
        replaced = [r for r in self.records if r['segment_id'] not in promoted and
                    (not corpus.is_preview(r) or r.get('asr_a_done') or r.get('asr_b_done') or r.get('human_override')) and
                    (r['segment_id'] not in fresh_by_id or r.get('audio_signature') != fresh_by_id[r['segment_id']]['audio_signature'])]
        old={r['segment_id']:r for r in self.records};kept=0
        for i,r in enumerate(fresh):
            previous=old.get(r['segment_id'])
            preview = promotions.get(r['segment_id'])
            if preview and not previous:
                for key in ('transcript_a','transcript_b','confidence_a','confidence_b','asr_a_done','asr_b_done',
                            'models','asr_a_engine','asr_b_engine','asr_errors','legacy_asr_results','human_override','status',
                            'transcript_status','selected_transcript','final_transcript','updated_by','updated_at'):
                    if key in preview:
                        r[key] = copy.deepcopy(preview[key])
                r['human_changes'].extend(copy.deepcopy(preview.get('human_changes', [])))
                r['transcribed_from_preview'] = preview['segment_id']
            if previous and previous.get('audio_signature')==r['audio_signature']:
                quality={key:copy.deepcopy(value) for key,value in r.items() if key.startswith('snr_') or key in ('energy_snr_db','wada_snr_db','clean_noisy','quality_history')}
                if previous.get('snr_override_by'):
                    analysis=copy.deepcopy(r.get('snr_analysis') or {})
                    snr.preserve_override(analysis,previous.get('snr_analysis'))
                    if analysis.get('override_by'):quality.update(snr.corpus_fields(analysis))
                fresh[i]=copy.deepcopy(previous);fresh[i].update(quality)
                fresh[i].update({key: r[key] for key in ('source_metadata', 'segmentation_status', 'source_mode', 'original_approval')})
                kept+=1
        engines.migrate_second_engine(fresh,model_b)
        engines.migrate_first_engine(fresh,engines.DEFAULT_QWEN_MODEL)
        for record in fresh:corpus.compare(record)
        source=metadata_folder(self.p.source_folder())/'metadata.json'
        metadata=json.loads(source.read_text(encoding='utf-8')) if source.is_file() else {}
        return fresh,kept,len(replaced),corpus.source_signature(self.p),metadata

    # Put the prepared records into the active review session.
    def apply_sync(self,result):
        fresh,kept,replaced,signature,metadata=result
        if replaced and QMessageBox.question(self.w,'Replace changed clips or spans',
            f'{replaced} rows have changed or disappeared in Part 4. Save a session first to keep their previous reviews. Load the new rows?',
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:
            return False
        if not fresh:
            self.clear_editors()
        self.records=fresh;self.signature=signature;self.source_metadata=metadata
        self.current_asr=self.current_final=None;self.transcript_draft=False
        self.p.dirty=True;self.refresh()
        if self.ui.asrTable.rowCount():
            self.ui.asrTable.selectRow(0)
            self.select_asr()
        n = sum(corpus.is_preview(r) for r in fresh)
        self.ui.asrInfo.setText(f'Loaded {len(fresh)-n} approved clips and {n} previews awaiting boundary review. Retained {kept} unchanged rows. Preview text can be reviewed here before approving boundaries in Part 4.')

        if not fresh:
            self.ui.asrInfo.setText('No clips are queued for transcription. You can include a skipped span on the Segments page.')
        return True

    # Collect transcripts, audit history and source identity for a project save.
    def payload(self, *, snapshot=True):
        capture=copy.deepcopy if snapshot else lambda value:value
        return dict(records=capture(self.records),audit_log=capture(self.audit_log),signature=self.signature,
                    source_metadata=self.source_metadata,qwen_model=self.ui.qwenModel.text().strip(),cohere_model=self.ui.cohereModel.text().strip(),
                    second_asr_engine='cohere_arabic',
                    asr_comparison_policy=engines.comparison_policy())

    @staticmethod
    def prepare_records(data):
        """Validate saved audio and update cached comparisons without reading any UI widgets."""
        records=data.get('records',[])
        if not isinstance(records,list):raise ValueError('Saved ASR records are invalid.')
        records = [r for r in records if not (
            (r.get('segmentation_status') == 'unresolved' or r.get('source_mode') == 'unresolved_preview')
            and corpus.review.under_one_second(r['start'],r['end']))]
        ids=[]
        for r in records:
            ids.append(r['segment_id'])
            if corpus.is_preview(r):
                first, last = round(r['start']*corpus.RATE), round(r['end']*corpus.RATE)
                if not 0 <= first < last or round(r['duration']*corpus.RATE) != last-first:
                    raise ValueError('Invalid saved unresolved span.')
            elif not 5-1/16000<=r['duration']<=15+1/16000:raise ValueError('A saved final clip has invalid duration.')
            if corpus.audio_signature(r)!=r['audio_signature']:raise ValueError('An ASR source audio file changed or moved.')
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate final clip IDs in saved session.')
        migrated=engines.migrate_second_engine(records,data.get('cohere_model') or engines.DEFAULT_COHERE_MODEL)
        migrated_a=engines.migrate_first_engine(records,engines.DEFAULT_QWEN_MODEL)
        for record in records:corpus.compare(record)
        return records,migrated,migrated_a

    # Reopen the saved review and handle transcripts from an earlier engine.
    def restore(self,data,*,saved_snr=None,prepared=None):
        self.clear()
        if not data:return
        records,migrated,migrated_a=prepared if prepared is not None else self.prepare_records(data)
        self.ui.qwenModel.setText(engines.DEFAULT_QWEN_MODEL)
        self.ui.cohereModel.setText(data.get('cohere_model') or engines.DEFAULT_COHERE_MODEL)
        self.records=sorted((snr.optional_quality_record(r) for r in records),key=lambda r:(float(r['start']),float(r['end']),r['segment_id']))
        self.audit_log=data.get('audit_log',[])
        self.signature=data.get('signature')
        # Accept the deterministic SNR status migration only when the original
        # signature matches and current results are exactly the reclassified rows.
        if saved_snr and self.signature == corpus.source_signature(self.p,saved_snr):
            ordered=sorted(saved_snr,key=snr.source_time_key)
            migrated_snr=snr.reclassify(ordered,self.p.applied_threshold,self.p.snr_policy,self.p.snr_justification)
            if self.p.snr_results == ordered or self.p.snr_results == migrated_snr:
                self.signature=corpus.source_signature(self.p)
        self.source_metadata=data.get('source_metadata',{})
        self.refresh()
        if migrated_a:self.ui.asrInfo.setText(f'{migrated_a} older Model A outputs were archived. Run transcription for Qwen.')
        if migrated:self.ui.asrInfo.setText(f'{migrated} earlier system-B results were archived with their model names. Run Transcribe all pending for Cohere; Qwen and human transcript edits were preserved.')

    # Give the final table and its summary the same collection of records.
    def final_records(self):
        return self.records

    # Refresh both tables so they show the same clip decisions.
    def refresh(self):
        self.review_pending=set()
        self.row_by_id={r['segment_id']:i for i,r in enumerate(self.records)}
        self.p.populate(self.ui.asrTable,[self.asr_cells(r) for r in self.records])
        self.p.populate(self.ui.finalTable,[self.review_cells(row)
                        for row in corpus.review_rows(self.final_records(), self.audit_log)])
        self.refresh_summary()
        rows=self.p.selected(self.ui.asrTable)
        self.configure_asr_preview(self.records[rows[0]] if len(rows)==1 and rows[0]<len(self.records) else None)
        self.ui.asrNextSegment.setEnabled(bool(self.records) and (not rows or rows[-1] < len(self.records)-1))
        self.select_final()

    # Format times, completion flags and agreement scores for the transcription table.
    @staticmethod
    def asr_cells(r):
        percent=lambda value:'—' if value is None else f'{value:.2%}'
        return [r['segment_id'],f"{r['start']:.3f}",f"{r['end']:.3f}",f"{r['duration']:.3f}",
                r.get('segmentation_status','unresolved' if corpus.is_preview(r) else 'resolved').title(),
                r['label'],r.get('clean_noisy',''),'Done' if r.get('asr_a_done') else 'Pending',
                'Done' if r.get('asr_b_done') else 'Pending',percent(r.get('cer_a_vs_b')),
                percent(r.get('cer_b_vs_a')),r['status']]

    # Use the export column order in the on-screen final review.
    @staticmethod
    def review_cells(row):
        return ['' if row[k] is None else str(row[k]) for k in corpus.REVIEW_COLUMNS]

    # Mark the changed clip for a table update and the project for saving.
    def record_changed(self, row):
        self.p.dirty=True
        self.p.update_cells(self.ui.asrTable,row,self.asr_cells(self.records[row]))
        self.review_pending.add(row)
        if self.ui.pages.currentIndex()==5:
            self.flush_review_rows()
            self.select_final()

    # Paint the rows that changed, then refresh the totals once.
    def flush_review_rows(self):
        if not self.review_pending:
            return
        indices=sorted(self.review_pending)
        projected=corpus.review_rows([self.records[i] for i in indices],self.audit_log)
        for i,record in zip(indices,projected):
            self.p.update_cells(self.ui.finalTable,i,self.review_cells(record))
        self.review_pending.clear()
        self.refresh_summary()

    # Show the time spent in each content label and the outcomes of transcript review.
    def refresh_summary(self):
        final_rows=self.final_records()
        counts={s:sum(r['status']==s for r in final_rows) for s in ['accepted','needs_review','difficult','not_applicable']}
        self.ui.finalSummary.setText('Clips: '+str(len(final_rows))+' · '+' · '.join(f'{k}: {v}' for k,v in counts.items()))
        summary = corpus.process_summary(self.p.duration or 0, self.p.regions, self.records, self.audit_log, session=self.p.session)
        regions = summary['region_seconds']
        coverage = summary['segmentation_coverage']['voice']
        speech = summary['segmentation_coverage']['speech_only']
        # Show source time in minutes with two decimal places.
        def minute(value): return f'{value / 60:.2f} min'
        # Use the eligible ASR clips as the denominator for this percentage.
        def percent_count(value):
            return f"{100 * value / summary['asr_total']:.1f}%" if summary['asr_total'] else '—'
        snr_value = '—' if summary['mean_snr'] is None else f"{summary['mean_snr']:.2f} dB"
        video_name = self.source_metadata.get('title') or (self.p.source_folder().name if self.p.audio else '—')
        values = [video_name, minute(summary['duration']), minute(regions['speech']), minute(regions['music']),
                  minute(regions['mixed']), str(summary['segments']), snr_value,
                  percent_count(summary['counts']['accepted']), percent_count(summary['counts']['chosen']),
                  percent_count(summary['counts']['difficult']),
                  f"{coverage['segmented']:.2f} s", f"{coverage['not_segmented']:.2f} s"]
        self.p.populate(self.ui.processSummaryTable, [values])
        coverage_percent = '—' if coverage['fraction'] is None else f"{coverage['fraction'] * 100:.1f}%"
        self.ui.processSummaryNote.setText(
            f"Part 4 coverage: {coverage['segmented']:.2f} / {coverage['total']:.2f} s of speech + mixed "
            f"({coverage_percent}); {coverage['not_segmented']:.2f} s not segmented. "
            f"Speech only: {speech['segmented']:.2f} / {speech['total']:.2f} s segmented; "
            f"{speech['not_segmented']:.2f} s not segmented. Includes pending and approved candidates, "
            "not excluded clips; silence/music and duplicate coverage are not counted.\n"
            f"Includes {summary['unresolved']} unresolved spans. ASR outcomes use {summary['asr_total']} eligible items; "
            f"{summary['pending']} pending. Mean SNR uses {summary['snr_count']} available values.")


    # Find the records behind the highlighted table rows.
    def selected(self,table):
        rows=self.p.selected(table)
        if not rows:raise ValueError('Select one or more clips.')
        records = self.final_records() if table is self.ui.finalTable else self.records
        return [records[i] for i in rows]

    # Show the model texts for the clip the reviewer selected.
    def select_asr(self):
        rows=self.p.selected(self.ui.asrTable)
        self.ui.asrNextSegment.setEnabled(bool(self.records) and (not rows or rows[-1] < len(self.records)-1))
        if len(rows)!=1 or rows[0]>=len(self.records):
            self.configure_asr_preview(None)
            self.ui.asrSelectedStatus.setText('Select one segment to see its transcription status.')
            return
        if self.transcript_draft and self.current_asr:
            try:self.commit_transcript('accepted' if self.ui.asrFinalText.toPlainText().strip() else 'needs_review',refresh=False)
            except Exception as exc:
                self.configure_asr_preview(None)
                QMessageBox.warning(self.w,'Save transcript first',str(exc));return
        r=self.records[rows[0]];self.current_asr=r['segment_id']
        self.configure_asr_preview(r)
        self.set_text(self.ui.asrTextA,r.get('transcript_a',''));self.set_text(self.ui.asrTextB,r.get('transcript_b',''));self.set_text(self.ui.asrFinalText,r.get('final_transcript',''))
        self.set_text(self.ui.asrNormalizedA,r.get('normalized_transcript_a',''))
        self.set_text(self.ui.asrNormalizedB,r.get('normalized_transcript_b',''))
        self.transcript_draft=False
        self.show_comparison(r)

    def next_segment(self):
        """Advance in table order, preserving the existing draft-saving behavior."""
        table=self.ui.asrTable
        rows=self.p.selected(table)
        target=rows[-1]+1 if rows else 0
        if target >= len(self.records):
            self.p.status('Last segment reached.')
            return
        # Commit before moving so a failed draft save cannot change selection.
        self.commit_drafts()
        table.clearSelection()
        table.selectRow(target)
        cell=table.item(target,0)
        if cell is not None:table.scrollToItem(cell)

    # Display the normalized comparison beside the raw words and the reason for the decision.
    def show_comparison(self,r):
        fmt=lambda v:'—' if v is None else f'{v:.3f}'
        percent=lambda v:'—' if v is None else f'{v:.2%}'
        index=self.row_by_id.get(r['segment_id'])
        position=f'{index+1}/{len(self.records)}' if index is not None else '—'
        status=r.get('status','needs_review').replace('_',' ').title()
        boundary=r.get('segmentation_status','unresolved' if corpus.is_preview(r) else 'resolved').replace('_',' ').title()
        self.ui.asrSelectedStatus.setText(f"Segment {position} · {r['segment_id']} · Transcription status: {status} · Segmentation: {boundary}")
        rule_cer=r.get('cer') if r.get('normalized_transcript_a') and r.get('normalized_transcript_b') else None
        self.ui.asrComparison.setText(f"Current decision: {r['status']} · Rule CER (B vs A): {percent(rule_cer)} · WER: {percent(r.get('wer'))} · word edits: {r.get('word_difference')} · Review limit: {r.get('review_word_edit_limit') or '—'} word edits · A score: {fmt(r.get('confidence_a'))} · B score: {fmt(r.get('confidence_b'))}")
        self.ui.asrCerA.setText('Qwen CER vs Cohere (normalized): '+percent(r.get('cer_a_vs_b')))
        self.ui.asrCerB.setText('Cohere CER vs Qwen (normalized): '+percent(r.get('cer_b_vs_a')))
        self.ui.asrAutomaticChoice.setText(f"Automatic decision before human editing: {r.get('automatic_status','needs_review')}")
        self.ui.asrAutomaticReason.setText(r.get('agreement_reason','Waiting for both ASR attempts.'))

    # Display final text only; selecting a row cannot modify the record.
    def select_final(self):
        rows=self.p.selected(self.ui.finalTable)
        if len(rows)!=1 or rows[0]>=len(self.final_records()):
            self.current_final=None
            self.set_text(self.ui.finalText,'')
            self.ui.finalMapping.setText('Select a segment to review its final transcript.')
            self.ui.finalPlay.setEnabled(False)
            return
        r=self.final_records()[rows[0]];self.current_final=r['segment_id']
        self.set_text(self.ui.finalText,r.get('final_transcript',''))
        status=r.get('status','needs_review').replace('_',' ').title()
        boundary=r.get('segmentation_status','unresolved' if corpus.is_preview(r) else 'resolved').replace('_',' ').title()
        self.ui.finalMapping.setText(f"{r['segment_id']} · {r['start']:.3f}–{r['end']:.3f} s · Transcription: {status} · Segmentation: {boundary}")
        self.ui.finalPlay.setEnabled(True)

    # Flag the review as edited and redraw the tables if requested.
    def changed(self,refresh=True):
        self.p.dirty=True
        if refresh:self.refresh()

    # Use the model text the reviewer chose as the accepted transcript.
    def choose(self,key):
        self.p.index(self.ui.asrTable)
        r=self.find(self.current_asr)
        if r is None:raise ValueError('Select one ASR row.')
        text=r.get('transcript_'+key,'')
        if not r.get('asr_'+key+'_done') or not text:raise ValueError('That model has no completed transcript for this clip.')
        self.set_text(self.ui.asrFinalText,text);self.commit_transcript('accepted')

    # Store the edited transcript and decision under the reviewer's name.
    def commit_transcript(self,status,refresh=True):
        if refresh:self.p.index(self.ui.asrTable)
        r=self.find(self.current_asr)
        if r is None:raise ValueError('Select one ASR row.')
        who=self.p.reviewer();text=self.ui.asrFinalText.toPlainText().strip()
        if status=='accepted' and not engines.normalize_arabic(text):raise ValueError('Accepted speech needs final text, not only punctuation or diacritics.')
        before=copy.deepcopy(r)
        r.update(final_transcript=text,selected_transcript=text,status=status,transcript_status=status,human_override=True)
        corpus.audit(self.audit_log,'transcript_review',who,before,r)
        self.show_comparison(r)
        self.transcript_draft=False;self.record_changed(self.row_by_id[r['segment_id']])

    # Save the transcript draft before moving on to another clip or export.
    def commit_drafts(self):
        if self.transcript_draft and self.current_asr:self.commit_transcript('accepted' if self.ui.asrFinalText.toPlainText().strip() else 'needs_review',False)

    # Start recognition for the models and clips requested by the user.
    def transcribe(self,selected):
        self.commit_drafts()
        if not self.records:
            self.sync_current(transcribe_after=selected)
            return
        self.check_current()
        chosen=self.selected(self.ui.asrTable) if selected else self.records
        ids={r['segment_id'] for r in chosen if corpus.can_transcribe(r)}
        if not ids:raise ValueError('Select speech, mixed audio or an unresolved preview for transcription.')
        qwen=self.ui.qwenModel.text().strip();cohere=self.ui.cohereModel.text().strip()
        if not qwen or not cohere:raise ValueError('Enter both model names.')
        records=copy.deepcopy(self.records);jobs={}
        engines.migrate_second_engine(records,cohere)
        engines.migrate_first_engine(records,qwen)
        for r in records:
            if r['segment_id'] not in ids:continue
            keys=[key for key in ('a','b') if selected or not r.get('asr_'+key+'_done')]
            if keys:jobs[r['segment_id']]=keys
            for key in keys:
                r['asr_'+key+'_done']=False
                r['transcript_'+key]=''
                r['confidence_'+key]=None
            corpus.compare(r)
        # A repeat Auto process run still needs a checkpoint with the new SNR and reused transcripts.
        if not jobs and self.p.auto_stage != 'asr':
            self.p.status('Both ASR engines have already run. Use Transcribe selected to rerun them.')
            return
        self.stop_event.clear();self.asr_running=True;self.ui.asrStop.setVisible(True)
        total=sum(map(len,jobs.values()));self.ui.asrProgress.setRange(0,total);self.ui.asrProgress.setValue(0)
        snapshot=self.p.project_payload(snapshot=False,include_asr=False)
        snapshot['asr']=self.payload(snapshot=False)
        snapshot['asr']['records']=records
        asr_folder = new_run_folder(self.p.source_folder() / 'Transcription')
        self.autosave_path=asr_folder / (self.p.audio.stem+'_full_autosave.json')
        autosave=self.autosave_path
        # Write a recoverable snapshot as recognition progresses.
        def persist():
            from audio_review_controller import atomic_json
            atomic_json(autosave,snapshot)
        # Work through the selected models and clips, checkpointing completed results.
        def action(log):
            completed=0;errors=[]
            persist()
            for key,name in [('a',qwen),('b',cohere)]:
                work=[r for r in records if key in jobs.get(r['segment_id'],[])]
                if not work or self.stop_event.is_set():continue
                model=None
                try:
                    loader=engines.load_qwen_model if key=='a' else engines.load_cohere_model
                    model = loader(name, log, stop_event=self.stop_event) if key == 'a' else loader(name, log)
                    for r in work:
                        if self.stop_event.is_set():break
                        log(f"{name}: {r['segment_id']}")
                        try:
                            if corpus.audio_signature(r)!=r['audio_signature']:raise ValueError('Clip audio changed since review.')
                            text,confidence=corpus.transcribe_record(r,key,model,self.stop_event)
                            r.update(**{'transcript_'+key:text,'confidence_'+key:confidence,'asr_'+key+'_done':True})
                            r.setdefault('asr_errors',{}).pop(key,None)
                            r.setdefault('models',{})[key]=name
                            if key=='a':r['asr_a_engine']='qwen_transformers'
                            if key=='b':
                                r['asr_b_engine']=model['engine']
                                r['models']['b']=model['model_name']
                            corpus.compare(r)
                        except InterruptedError:
                            if self.stop_event.is_set():break
                            raise
                        except Exception as exc:
                            r.setdefault('asr_errors',{})[key]=str(exc);errors.append(f"{r['segment_id']} {key}: {exc}")
                            r['asr_'+key+'_done']=False
                        # Save before beginning the next clip/model. Failure stops work.
                        persist();self.piece.emit(copy.deepcopy(r));completed+=1;self.progress.emit((completed,total))
                except InterruptedError:
                    if not self.stop_event.is_set():
                        raise
                    break
                except Exception as exc:
                    # Persistence errors must stop the entire job instead of continuing.
                    if isinstance(exc,OSError):raise
                    errors.append(f'{name}: {exc}');log('ASR model error: '+str(exc))
                    for r in work:
                        if not r.get('asr_'+key+'_done'):r.setdefault('asr_errors',{})[key]=str(exc)
                    persist()
                finally:
                    engines.release_asr_model(model)
                    model=None
                    gc.collect()
            persist()
            return dict(records=records,errors=errors,cancelled=self.stop_event.is_set(),completed=completed,total=total)
        # Bring the finished records, cancellations and errors back to the review page.
        def done(result):
            self.records=result['records'];self.changed()
            self.select_asr()
            self.select_final()
            state='Stopped after the current clip' if result['cancelled'] else 'Transcription finished'
            self.ui.asrInfo.setText(f"{state}: {result['completed']} / {result['total']} model/clip jobs. Errors: {len(result['errors'])}. Autosave: {autosave}")
            if not total:
                self.ui.asrInfo.setText(f'Existing transcripts reused. Autosave: {autosave}')
            self.p.status(self.ui.asrInfo.text())
            if result['errors']:self.w.append_log('ASR errors: '+'; '.join(result['errors']))
        try:self.p.start('Transcribing clips and unresolved spans…',action,done)
        except Exception:
            self.task_finished();raise

    # Refresh the matching row as soon as a single clip result arrives.
    def receive_piece(self,record):
        i=self.row_by_id.get(record['segment_id'])
        if i is not None:
            self.records[i]=record
            self.record_changed(i)

    # Show how far the recognition worker has got through its queue.
    def receive_progress(self,value):self.ui.asrProgress.setValue(value[0])

    # Ask the worker to stop after its current model or clip.
    def cancel(self):
        self.stop_event.set();self.ui.asrStop.setEnabled(False)
        self.p.status('Stopping after the current model/clip finishes; completed results remain autosaved.')

    # Release the transcription controls and start any work queued after a refresh.
    def task_finished(self):
        self.asr_running=False;self.ui.asrStop.hide();self.ui.asrStop.setEnabled(True)
        if self.transcribe_after_sync is not None:
            selected=self.transcribe_after_sync
            self.transcribe_after_sync=None
            self.p.guard(lambda:self.transcribe(selected))

    # The ASR listening cursor always uses clip-relative milliseconds, even for
    # long previews stored as an interval of the original source WAV.
    @staticmethod
    def asr_preview_spec(record):
        if not record:
            return None
        pieces = record.get('audio_pieces') or []
        ranges = tuple((str(p['path']), round(float(p['start'])*corpus.RATE),
                        round(float(p['end'])*corpus.RATE)) for p in pieces)
        if not ranges or any(a < 0 or b <= a for _, a, b in ranges):
            return None
        token = (record['segment_id'], record.get('audio_signature'), ranges)
        if len(ranges) == 1:
            path, first, last = ranges[0]
            offset = round(first*1000/corpus.RATE)
            duration = round(last*1000/corpus.RATE)-offset
        else:
            # Preserve legacy multi-piece audio mappings exactly as in play_table.
            path, offset = None, 0
            duration = round(sum(b-a for _, a, b in ranges)*1000/corpus.RATE)
        if duration <= 0:
            return None
        return dict(token=token, path=path, offset=offset, duration=duration)

    def configure_asr_preview(self, record):
        spec = self.asr_preview_spec(record)
        changed = spec != self.asr_preview
        if changed:
            if self.p.play_owner == 'asr':
                self.p.stop_playback()
            self.asr_preview_file = None
        self.asr_preview = spec
        slider = self.ui.asrSeekSlider
        for widget in (slider, self.ui.asrPlay, self.ui.asrPausePreview, self.ui.asrStopPreview):
            widget.setEnabled(spec is not None)
        slider.blockSignals(True)
        slider.setRange(0, spec['duration'] if spec else 0)
        if changed:
            slider.setValue(0)
        slider.blockSignals(False)
        self.ui.asrSeekEnd.setText(self.p.clock(slider.maximum()))
        self.update_asr_cursor_label()

    def update_asr_cursor_label(self):
        self.ui.asrCursorLabel.setText('Clip time ' + self.p.clock(self.ui.asrSeekSlider.value()))

    def asr_slider_changed(self, value):
        self.update_asr_cursor_label()
        if not self.ui.asrSeekSlider.isSliderDown():
            self.p.guard(self.seek_asr_preview)

    def play_asr_cursor(self):
        records = self.selected(self.ui.asrTable)
        if len(records) != 1:
            raise ValueError('Select one clip to play.')
        record = records[0]
        self.configure_asr_preview(record)
        spec = self.asr_preview
        if not spec:
            raise ValueError('No playable audio in this clip.')
        if spec['path'] is not None:
            path = Path(spec['path'])
        else:
            if self.asr_preview_file is None:
                audio = corpus.read_audio(record)
                path = Path(self.playback_dir.name)/(uuid.uuid4().hex+'.wav')
                sf.write(path, audio, corpus.RATE, subtype='PCM_16')
                self.asr_preview_file = path
            path = self.asr_preview_file
        slider = self.ui.asrSeekSlider
        if slider.value() >= spec['duration']:
            slider.setValue(0)
        self.p.play(path, (spec['offset']+slider.value())/1000,
                    (spec['offset']+spec['duration'])/1000, owner='asr')
        if self.p.play_owner == 'asr':
            self.ui.asrPausePreview.setText('Pause')

    def seek_asr_preview(self):
        p, spec = self.p, self.asr_preview
        if p.play_owner != 'asr' or not p.play_range or p.player is None or not spec:
            return
        position = self.ui.asrSeekSlider.value()
        if position >= spec['duration']:
            p.stop_playback()
            return
        p.pending_seek = spec['offset']+position
        p.play_range = (p.pending_seek, spec['offset']+spec['duration'])
        # While loading, media_ready will use the latest requested seek instead.
        if p.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            p.player.setPosition(p.pending_seek)

    def pause_asr_preview(self):
        p = self.p
        if p.play_owner != 'asr' or p.player is None or not p.play_range:
            self.play_asr_cursor()
            return
        p.play_paused = not p.play_paused
        if p.play_paused:
            p.player.pause()
            p.play_timer.stop()
        else:
            if p.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
                p.player.play()
            p.play_timer.start(40)
        self.ui.asrPausePreview.setText('Resume' if p.play_paused else 'Pause')

    def update_asr_playback_position(self, position):
        if self.p.play_owner != 'asr' or not self.asr_preview:
            return
        slider = self.ui.asrSeekSlider
        if slider.isSliderDown():
            return
        slider.blockSignals(True)
        slider.setValue(min(slider.maximum(), max(0, position-self.asr_preview['offset'])))
        slider.blockSignals(False)
        self.update_asr_cursor_label()

    def asr_playback_stopped(self):
        self.ui.asrPausePreview.setText('Pause / resume')

    def stop_asr_preview(self):
        if self.p.play_owner == 'asr':
            self.p.stop_playback()
        self.ui.asrSeekSlider.setValue(0)

    # Play the selected review record using the samples described by its mapping.
    def play_table(self,table):
        records=self.selected(table)
        if len(records)!=1:raise ValueError('Select one clip to play.')
        record=records[0]
        if corpus.is_preview(record):
            self.p.play(Path(record['source_origin']),record['start'],record['end'])
            return
        audio=corpus.read_audio(record)
        path=Path(self.playback_dir.name)/(uuid.uuid4().hex+'.wav')
        sf.write(path,audio,16000,subtype='PCM_16');self.p.play(path,0,len(audio)/16000)

    # Ask for the video name and append its numbers to the chosen summary workbook.
    def save_process_summary(self):
        self.commit_drafts()
        self.check_current()
        default_name = self.source_metadata.get('title') or self.p.source_folder().name
        name, accepted = QInputDialog.getText(self.w, 'Video name', 'Enter the video name', text=default_name)
        if not accepted:
            return
        name = name.strip()
        if not name:
            raise ValueError('Enter a video name to save the summary.')
        settings = QSettings('Speech Corpus Studio', 'Speech Corpus Studio')
        default = settings.value('summary_workbook', str(self.p.source_folder() / 'process_summary.xlsx'))
        filename, _ = QFileDialog.getSaveFileName(
            self.w, 'Select summary workbook to create or append', str(default),
            'Excel workbook (*.xlsx)', options=QFileDialog.Option.DontConfirmOverwrite)
        if not filename:
            return
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix('.xlsx')
        if path.suffix.lower() != '.xlsx':
            raise ValueError('Choose an .xlsx workbook.')
        summary = corpus.process_summary(self.p.duration, self.p.regions, self.records, self.audit_log, session=self.p.session)
        # Remember the workbook for the next video once the row has been saved.
        def done(_):
            settings.setValue('summary_workbook',str(path))
            self.p.status(f'Summary appended: {path}')
        self.p.start('Saving summary workbook…',lambda log:corpus.append_summary_workbook(path,name,summary),done)

    # Write the reviewed timeline in the format selected in the save dialog.
    def save_review(self):
        self.commit_drafts()
        if self.records or not (self.p.session or {}).get('unresolved'):
            self.check_current()
        elif any(s['status']=='approved' for s in self.p.session.get('segments',[])):
            self.check_current()
        with save_dialog_folder(self.p.source_folder() / 'Review and export') as folder:
            formats = {'CSV (*.csv)': '.csv', 'Excel (*.xlsx)': '.xlsx', 'HTML (*.html)': '.html', 'JSON (*.json)': '.json'}
            path, selected = QFileDialog.getSaveFileName(
                self.w, 'Save review table', str(folder / 'review'), ';;'.join(formats))
            if not path:
                return
            path = Path(path)
            if not path.suffix:
                path = path.with_suffix(formats.get(selected, '.csv'))
        records,audit=self.records,self.audit_log
        session,source,qualities=self.p.session,self.p.audio,self.p.snr_results
        self.p.start('Saving review table…',lambda log:corpus.export_review_csv(path,records,audit,session,source,qualities),
                     lambda _:self.p.status(f'Review table saved: {path}'))

    # Start writing the corpus using the decisions currently on the review page.
    def export(self):
        self.commit_drafts()
        self.check_export_current()
        directory=QFileDialog.getExistingDirectory(self.w,'Export final corpus',str(self.p.source_folder()))
        if not directory:return
        self.start_corpus_export(step_folder(directory, 'Review and export'),
                                 self.ui.allowPartialExport.isChecked())

    def check_export_current(self):
        if self.records or not (self.p.session or {}).get('unresolved') or any(
                s['status']=='approved' for s in self.p.session.get('segments',[])):
            self.check_current()

    def export_automatic(self):
        """Choose a destination after ASR, then export a partial snapshot."""
        self.check_export_current()
        directory = QFileDialog.getExistingDirectory(self.w, 'Choose automatic partial corpus export location',
                                                      str(self.p.source_folder()))
        if not directory:
            return False
        self.ui.allowPartialExport.setChecked(True)
        directory = step_folder(directory, 'Review and export') / 'Automatic partial export'
        self.start_corpus_export(directory, True, automatic=True)
        return True

    def start_corpus_export(self, directory, partial, *, automatic=False):
        # Export runs with editing controls disabled and treats nested review data as read-only.
        # Avoid duplicating an entire long recording review before starting the worker.
        records=list(self.records);audit=list(self.audit_log)
        metadata=dict(source=self.source_metadata,audio=str(self.p.audio),
                      source_review=dict(region_changes=self.p.changes,step4=self.p.session,snr=self.p.snr_results),
                      qwen_model=self.ui.qwenModel.text(),cohere_model=self.ui.cohereModel.text(),second_asr_engine='cohere_arabic',
                      pending_source_candidates=sum(r['status']=='pending' for r in self.p.session['segments']),
                      unresolved_source_spans=sum(r['status']!='reviewed' for _,r,_,_ in corpus.review.visible_unresolved(self.p.session)))
        def done(result):
            if automatic:
                report = Path(result['output_dir']) / 'corpus_report.html'
                message = (f"Automatic processing finished. Partial corpus: {result['clips']} clips. "
                           f"Playable HTML: {report}. Compatible CSV: {corpus.COMPATIBLE_REVIEW_FILENAME}. "
                           f"Pending boundaries and transcripts remain unapproved.")
                self.w.append_log(message)
                self.p.status(message)
            else:
                self.p.status(f"Exported {result['clips']} clips to {result['output_dir']}. Mode: {result.get('export_mode', 'partial' if result['partial'] else 'complete')}." )
                self.w.open_path(Path(result['output_dir']))
        self.p.start('Writing corpus WAV, TXT, CSV and playable HTML…',
                     lambda log:corpus.export_corpus(records,metadata,audit,directory,partial,automatic=automatic,log=log),done)

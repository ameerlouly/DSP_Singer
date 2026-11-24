"""Connect the Qt region, segment and audio-quality pages to the processing pipeline.

Function guide
--------------
atomic_json, validate_regions
    Save session JSON atomically and check that region boundaries form a valid
    timeline.
Task.run, start, task_result, task_error, task_finished
    Run processing away from the UI thread and deliver results, errors and completion
    to the page.
eventFilter, fit_page_height, navigate
    Fit the scroll area to the visible page and switch workspaces.
table, populate, selected, index, reviewer, guard, status, confirm_discard
    Configure tables, identify selections and the reviewer, and report validation or
    replacement decisions.
start_auto_process, advance_auto_process, set_shared_controls_enabled
    Sequence automated stages and manage shared controls while work is active.
require_audio, choose_audio, set_audio, source_folder
    Select the source WAV and locate its working directory.
show_spectrogram, plot_spectrogram, analyze, redraw_regions, refresh_all
    Compute and draw the spectrogram, run labeling and refresh the review tables.
region_selected, clock, configure_region_preview, update_region_cursor_label,
update_region_range_summary
    Display the selected interval, source position and listening range.
region_slider_changed, region_cursor_interval, seek_region_preview, mark_region_time,
play_region_cursor, pause_region_preview
    Seek within the region and mark or audition a subrange.
region_mode_changed, entered_region_range, edit_region, apply_region_correction,
split_region, merge_region, plot_press, plot_release
    Edit labels and boundaries using fields, the plot or split/merge controls.
clear_snr, invalidate_segments
    Discard downstream results when their source changes.
write_region_report, region_payload, save_regions, load_regions, check_region_payload
    Build, save and verify region review data before restoring it.
generate_segments, set_session, load_candidates, segment_selected, segment_mode_changed
    Generate or reopen pause-based candidates and show their editable bounds.
part4_spec, configure_part4_preview, update_part4_summary, part4_slider_changed,
part4_interval, seek_part4, mark_part4, play_part4_cursor, pause_part4,
update_part4_playback_position
    Map the segment or unresolved-span slider to the source or candidate being
    auditioned.
modified_session, edit_segment, approve, exclude, resolve_overlaps
    Record reviewer actions and maintain consistent approved intervals.
unresolved_selected, add_manual, mark_unresolved, require_approved, export_segments
    Review uncovered audio, create clips and export approved audio with its source
    mapping.
require_analysis_source, calculate_snr, snr_controls_changed, refresh_snr, check_snr,
override_quality
    Estimate clip quality, apply the chosen decision rule and retain reviewer
    corrections.
save_snr, open_snr_report, load_snr, match_snr, set_snr_controls
    Save or reopen SNR results and match them to the current source intervals.
save_project, project_payload, open_project
    Serialize and restore the processing and transcription workspaces together.
play, media_ready, update_region_playback_position, playback_position,
enforce_playback_end, stop_playback
    Coordinate the media player, follow playback and stop at the requested endpoint.
play_entered_region_range, play_region, play_region_edge, play_segment,
play_segment_edge, play_unresolved, play_quality
    Audition the selected source range, boundary, candidate or quality-review clip.

All widgets and page layouts are editable in main_window.ui. Model inference and file-heavy
work run on a QThread; all widget updates run on the main Qt thread.
"""
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, QThread, QTimer, QUrl, QLocale, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QHeaderView,
                            QMessageBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import soundfile as sf

from audio_spectrogram import calculate_spectrogram
from region_review_report import load_review, save_report
from audio_region_classifier import classify_file, analysis_settings, recorded_analysis_settings
from region_editing import relabel_range
import segment_review as review
import clip_quality_assessment as snr

LABELS = ('speech', 'music', 'mixed', 'silence')
COLORS = dict(speech='#268b80', music='#a77cbc', mixed='#e6ab54', silence='#a1afb4')


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def validate_regions(regions, duration):
    if not isinstance(regions, list) or not regions:
        raise ValueError('No region labels are available.')
    previous = 0.0
    for r in regions:
        a, b = float(r['start']), float(r['end'])
        if r['label'] not in LABELS or not math.isfinite(a + b) or abs(a - previous) > 1e-6 or not 0 <= a < b <= duration + 1e-6:
            raise ValueError('Regions must cover the audio in order without gaps or overlaps.')
        for key in ('confidence', 'auto_confidence'):
            value = r.get(key)
            if value is not None and (not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
                raise ValueError('Automatic confidence scores must be between 0 and 1.')
        previous = b
    if abs(previous - duration) > 1e-6:
        raise ValueError('The region timeline does not cover the whole WAV.')


class Task(QThread):
    message = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, action, parent):
        super().__init__(parent)
        self.action = action

    def run(self):
        try:
            self.result.emit(self.action(self.message.emit))
        except Exception as exc:
            self.error.emit(str(exc))


class ProcessingWorkspace(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.ui = window
        self.audio = None
        self.duration = 0.0
        self.regions = []
        self.changes = []
        self.session = None
        self.snr_results = []
        self.snr_snapshot = None
        self.applied_threshold = 20.0
        self.snr_policy = 'conservative'
        self.snr_justification = ''
        self.snr_report = None
        self.task = None
        self.auto_stage = None
        self.auto_waiting = False
        self.auto_source = None
        self.dirty = False
        self.play_range = None
        self.play_path = None
        self.pending_seek = None
        self.play_paused = False
        self.preview_bounds = None
        self.part4_previews = {'segment': None, 'unresolved': None}
        self.play_owner = None
        self.plot_data = None
        self.patches = []
        self.drag_boundary = None
        self.finished_callback = None


        self.figure = Figure(figsize=(9, 3), constrained_layout=True, facecolor='#F8FAFB')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(290)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title('Choose a WAV and display its spectrogram', color='#173B40')
        self.ui.plotLayout.addWidget(NavigationToolbar2QT(self.canvas, self.ui))
        self.ui.plotLayout.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self.plot_press)
        self.canvas.mpl_connect('button_release_event', self.plot_release)
        self.histogram = Figure(figsize=(8, 2), constrained_layout=True, facecolor='#F8FAFB')
        self.hist_canvas = FigureCanvasQTAgg(self.histogram)
        self.hist_canvas.setMinimumHeight(240)
        self.ui.histogramLayout.addWidget(self.hist_canvas)
        # Open the audio device only when playback is requested.
        self.player = None
        self.audio_output = None
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.enforce_playback_end)

        self.table(self.ui.regionTable)
        self.table(self.ui.segmentTable)
        self.table(self.ui.unresolvedTable)
        self.table(self.ui.snrTable)
        self.ui.regionTable.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ui.unresolvedTable.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ui.regionTable.itemSelectionChanged.connect(self.region_selected)
        self.ui.regionEditMode.currentIndexChanged.connect(self.region_mode_changed)
        self.ui.regionStart.valueChanged.connect(self.update_region_range_summary)
        self.ui.regionEnd.valueChanged.connect(self.update_region_range_summary)
        self.ui.regionSeekSlider.valueChanged.connect(self.region_slider_changed)
        self.ui.regionSeekSlider.sliderReleased.connect(lambda: self.guard(self.seek_region_preview))
        self.ui.segmentTable.itemSelectionChanged.connect(self.segment_selected)
        self.ui.unresolvedTable.itemSelectionChanged.connect(self.unresolved_selected)
        self.ui.snrThreshold.valueChanged.connect(self.snr_controls_changed)
        self.ui.snrPolicy.currentIndexChanged.connect(self.snr_controls_changed)
        self.ui.snrJustification.editingFinished.connect(self.snr_controls_changed)
        self.ui.boundaryMode.currentIndexChanged.connect(lambda: self.guard(self.segment_mode_changed))
        for kind in self.part4_previews:
            slider = getattr(self.ui, kind + 'SeekSlider')
            slider.valueChanged.connect(lambda value, k=kind: self.part4_slider_changed(k))
            slider.sliderReleased.connect(lambda k=kind: self.guard(lambda: self.seek_part4(k)))
            for suffix in ('Start', 'End'):
                getattr(self.ui, kind + suffix).valueChanged.connect(
                    lambda value, k=kind: self.update_part4_summary(k))
            for name, action in (
                ('play' + kind.title() + 'Cursor', lambda k=kind: self.play_part4_cursor(k)),
                ('pause' + kind.title() + 'Preview', lambda k=kind: self.pause_part4(k)),
                ('mark' + kind.title() + 'Start', lambda k=kind: self.mark_part4(k, False)),
                ('mark' + kind.title() + 'End', lambda k=kind: self.mark_part4(k, True)),
            ):
                getattr(self.ui, name).clicked.connect(lambda checked=False, a=action: self.guard(a))
        self.ui.wavPath.setReadOnly(True)
        for widget, minimum in ((self.ui.pannsWindowInput, 2.), (self.ui.pannsHopInput, .1)):
            validator = QDoubleValidator(minimum, 30., 3, widget)
            validator.setLocale(QLocale.c())
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            widget.setValidator(validator)
        actions = {
            'chooseWav': self.choose_audio, 'showSpectrogram': self.show_spectrogram,
            'analyzeRegions': self.analyze, 'loadRegions': self.load_regions,
            'saveRegions': self.save_regions, 'applyRegion': self.apply_region_correction,
            'playEnteredRegionRange': self.play_entered_region_range,
            'playRegionCursor': self.play_region_cursor, 'pauseRegionPreview': self.pause_region_preview,
            'markRegionStart': lambda: self.mark_region_time(False),
            'markRegionEnd': lambda: self.mark_region_time(True),
            'splitRegion': self.split_region, 'mergeRegion': self.merge_region,
            'playRegion': self.play_region, 'playRegionStart': lambda: self.play_region_edge(False),
            'playRegionEnd': lambda: self.play_region_edge(True),
            'generateSegments': self.generate_segments, 'loadCandidates': self.load_candidates,
            'applySegment': self.edit_segment, 'approveSegments': self.approve,
            'excludeSegments': self.exclude, 'resolveOverlaps': self.resolve_overlaps,
            'exportSegments': self.export_segments, 'playSegment': self.play_segment,
            'playCutStart': lambda: self.play_segment_edge(False),
            'playCutEnd': lambda: self.play_segment_edge(True),
            'playUnresolved': self.play_unresolved, 'addManualClip': self.add_manual,
            'markUnresolved': self.mark_unresolved, 'calculateSNR': self.calculate_snr,
            'saveSNR': self.save_snr, 'openSNR': self.load_snr,
            'openSNRReport': self.open_snr_report,
            'markClean': lambda: self.override_quality('clean'),
            'markNoisy': lambda: self.override_quality('noisy'),
            'restoreQuality': lambda: self.override_quality(None), 'playQuality': self.play_quality,
            'saveProject': self.save_project, 'openProject': self.open_project,
            'stopPlayback': self.stop_playback,
            'goSegments': lambda: window.navigationList.setCurrentRow(2),
            'goQuality': lambda: window.navigationList.setCurrentRow(3),
        }
        for name, action in actions.items():
            getattr(self.ui, name).clicked.connect(lambda checked=False, a=action: self.guard(a))
        window.navigationList.currentRowChanged.disconnect()
        window.navigationList.currentRowChanged.connect(self.navigate)
        self.page_size_timer = QTimer(self)
        self.page_size_timer.setSingleShot(True)
        self.page_size_timer.timeout.connect(self.fit_page_height)
        window.pages.installEventFilter(self)
        for index in range(window.pages.count()):
            window.pages.widget(index).installEventFilter(self)
        self.navigate(0)
        self.refresh_all()
        from transcription_review_controller import ASRWorkspace
        self.asr = ASRWorkspace(self)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.LayoutRequest, QEvent.Type.Resize, QEvent.Type.Show):
            self.page_size_timer.start(0)
        return super().eventFilter(watched, event)

    def fit_page_height(self):
        pages = self.window.pages
        page = pages.currentWidget()
        if page is None:
            return
        height = max(page.minimumSizeHint().height(), page.minimumHeight(),
                     page.heightForWidth(pages.width()) if page.hasHeightForWidth() else page.sizeHint().height())
        if pages.minimumHeight() != height or pages.maximumHeight() != height:
            pages.setFixedHeight(height)

    def navigate(self, index):
        self.window.pages.setCurrentIndex(index)
        processing_page = 1 <= index <= 5
        self.ui.sharedControlsFrame.setVisible(processing_page)
        self.ui.processingStatusFrame.setVisible(processing_page)
        self.fit_page_height()
        self.window.contentScrollArea.verticalScrollBar().setValue(0)

    @staticmethod
    def table(table):
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().hide()

    @staticmethod
    def populate(table, rows):
        from PyQt6.QtWidgets import QTableWidgetItem
        table.blockSignals(True)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(value)))
        table.blockSignals(False)

    @staticmethod
    def selected(table):
        return sorted({x.row() for x in table.selectionModel().selectedRows()})

    def index(self, table):
        rows = self.selected(table)
        if len(rows) != 1:
            raise ValueError('Select one row for this action.')
        return rows[0]

    def reviewer(self):
        name = self.ui.reviewer.text().strip()
        if not name:
            raise ValueError('Enter your reviewer name first.')
        return name

    def guard(self, action):
        if self.task is not None or self.window.worker is not None:
            self.status('Wait for the running operation to finish.')
            return
        try:
            action()
        except Exception as exc:
            QMessageBox.warning(self.window, 'Cannot complete action', str(exc))

    def status(self, text):
        self.ui.processingStatus.setText(text)
        self.window.statusbar.showMessage(text)

    def confirm_discard(self):
        return not self.dirty or QMessageBox.question(self.window, 'Unsaved session',
            'Continue and discard unsaved changes? Save the session first to keep them.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def start(self, description, action, done):
        if self.task is not None or self.window.worker is not None:
            raise ValueError('An operation is already running.')
        self.stop_playback()
        self.status(description)
        self.ui.processingProgress.setRange(0, 0)
        self.ui.pages.setEnabled(False)
        self.set_shared_controls_enabled(False)
        for control in self.window.controls:
            control.setEnabled(False)
        self.task = Task(action, self)
        self.finished_callback = done
        self.task.result.connect(self.task_result)
        self.task.error.connect(self.task_error)
        self.task.message.connect(self.status)
        self.task.finished.connect(self.task_finished)
        self.task.finished.connect(self.task.deleteLater)
        self.task.start()

    def task_result(self, value):
        try:
            self.finished_callback(value)
        except Exception as exc:
            self.task_error(str(exc))

    def task_error(self, text):
        self.auto_stage = None
        self.auto_waiting = False
        self.status('Failed: ' + text)
        self.window.append_log('Parts 3–5: ' + text)
        QMessageBox.warning(self.window, 'Processing failed', text)

    def task_finished(self):
        self.task = None
        self.finished_callback = None
        self.ui.pages.setEnabled(True)
        self.set_shared_controls_enabled(True)
        for control in self.window.controls:
            control.setEnabled(True)
        self.ui.processingProgress.setRange(0, 100)
        self.ui.processingProgress.setValue(0)
        if hasattr(self, 'asr'):
            self.asr.task_finished()
        if self.auto_stage:
            QTimer.singleShot(0, self.advance_auto_process)

    def start_auto_process(self):
        self.require_audio()
        if self.task is not None or self.window.worker is not None:
            raise ValueError('Wait for the current operation to finish.')
        if self.auto_waiting and self.auto_source == self.audio:
            self.require_analysis_source()
            if not snr.session_records(self.session, self.audio):
                raise ValueError('Approve at least one candidate or keep an unresolved span before continuing.')
            self.auto_waiting = False
            self.auto_stage = 'snr'
            try:
                self.calculate_snr()
            except Exception:
                self.auto_stage = None
                self.auto_waiting = True
                raise
            return
        if not self.confirm_discard():
            return
        self.auto_source = self.audio
        self.auto_waiting = False
        self.auto_stage = 'label'
        try:
            self.analyze(confirmed=True)
        except Exception:
            self.auto_stage = None
            raise

    def advance_auto_process(self):
        try:
            if self.auto_stage == 'label':
                self.auto_stage = 'segments'
                self.generate_segments(output_folder=self.source_folder(), confirmed=True)
            elif self.auto_stage == 'segments':
                self.auto_stage = None
                self.auto_waiting = True
                self.status('Review the segment boundaries, then return to Source & Audio and click Auto process again.')
                QMessageBox.information(self.window, 'Review segment boundaries',
                    'Automatic labels and segment candidates are ready. Review the labels and listen to the segment boundaries. '
                    'Approve suitable segments, then click Auto process again to calculate SNR and transcribe.')
            elif self.auto_stage == 'snr':
                self.asr.sync_current()
                self.auto_stage = 'asr'
                self.asr.transcribe(False)
                if self.task is None:
                    self.auto_stage = None
            elif self.auto_stage == 'asr':
                self.auto_stage = None
                self.window.navigationList.setCurrentRow(4)
                errors = any(r.get('asr_errors') for r in self.asr.records)
                if self.asr.stop_event.is_set() or errors:
                    self.status('Transcription stopped or had errors. Check Transcription and Activity before continuing.')
                else:
                    self.window.append_log('Automatic processing finished. Review ASR results, quality and any unresolved spans.')
                    self.status('Automatic processing finished. Review the transcripts and audio quality.')
            elif self.auto_stage is not None:
                raise ValueError('Unknown automatic processing stage.')
        except Exception as error:
            self.auto_stage = None
            self.auto_waiting = False
            self.task_error(str(error))

    def set_shared_controls_enabled(self, enabled):
        """Support Designer rows with or without intermediate QWidget containers."""
        def visit(layout):
            for index in range(layout.count()):
                item = layout.itemAt(index)
                if item.widget() is not None:
                    item.widget().setEnabled(enabled)
                elif item.layout() is not None:
                    visit(item.layout())
        visit(self.ui.sharedSourceRowLayout)
        visit(self.ui.sharedReviewerRowLayout)

    def require_audio(self):
        if self.audio is None or not self.audio.is_file():
            raise ValueError('Choose a WAV or extract audio on Source & Audio first.')

    def choose_audio(self):
        path, _ = QFileDialog.getOpenFileName(self.window, 'Choose original WAV', '', 'Audio WAV (*.wav)')
        if path:
            self.set_audio(path)

    def set_audio(self, path, ask=True):
        path = Path(path).resolve()
        if path == self.audio:
            return True
        frames, rate = review.inspect_wav(path)
        if frames == 0:
            raise ValueError('The selected WAV is empty.')
        if ask and not self.confirm_discard():
            return False
        self.stop_playback()
        if hasattr(self, 'asr'):
            self.asr.clear()
        self.audio, self.duration = path, frames / rate
        self.ui.wavPath.setText(str(path))
        self.regions, self.changes = [], []
        self.session = None
        self.clear_snr()
        self.plot_data = None
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title('Click Show spectrogram to display this recording')
        self.canvas.draw_idle()
        self.dirty = False
        self.refresh_all()
        self.status(f'Ready: {path.name} · {self.duration:.2f} seconds')
        return True

    def show_spectrogram(self):
        self.require_audio()
        path = str(self.audio)
        self.start('Building spectrogram…', lambda log: calculate_spectrogram(path), self.plot_spectrogram)

    def plot_spectrogram(self, data):
        self.plot_data = data
        times, frequencies, db, rate, duration, stride = data
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        image = self.axes.pcolormesh(times, frequencies, db, shading='auto', cmap='magma', vmin=-80, vmax=0)
        self.figure.colorbar(image, ax=self.axes, label='Power relative to peak (dB)')
        self.axes.set(xlabel='Time (s)', ylabel='Frequency (Hz)', xlim=(0, duration), ylim=(0, rate/2))
        self.patches = []
        self.redraw_regions()
        self.status('Spectrogram ready. Zoom with the toolbar; drag a shared boundary to adjust it.')

    def analyze(self, confirmed=False):
        self.require_audio()
        window_seconds, hop_seconds = analysis_settings(
            self.ui.pannsWindowInput.text(), self.ui.pannsHopInput.text())
        if not confirmed and not self.confirm_discard():
            return
        path = str(self.audio)
        verify_vocals = self.ui.improveSinging.isChecked()
        def action(log):
            regions = classify_file(path, log, verify_vocals=verify_vocals,
                                    window_seconds=window_seconds, hop_seconds=hop_seconds)
            return regions, calculate_spectrogram(path)
        def done(result):
            regions, plot = result
            validate_regions(regions, self.duration)
            self.regions, self.changes = regions, []
            self.session = None
            self.clear_snr()
            self.dirty = True
            self.refresh_all()
            self.plot_spectrogram(plot)
            report = self.write_region_report()
            self.window.append_log(f'Region report saved: {report}')
            self.status(f'{len(regions)} regions labeled. Review flagged and mixed regions before cutting.')
        self.start('Loading PANNs; first run downloads its model…', action, done)

    def redraw_regions(self):
        for artist in self.patches:
            try: artist.remove()
            except ValueError: pass
        self.patches = []
        for r in self.regions:
            self.patches.append(self.axes.axvspan(r['start'], r['end'], color=COLORS[r['label']], alpha=.23))
        # The editable Qt color key sits below the plot, outside the audio data.
        legend = self.axes.get_legend()
        if legend is not None:
            legend.remove()
        self.canvas.draw_idle()

    def refresh_all(self):
        self.populate(self.ui.regionTable, [
            [f"{r['start']:.4f}", f"{r['end']:.4f}", r['label'], r.get('auto_label',''),
             '—' if r.get('auto_confidence') is None else f"{r['auto_confidence']:.3f}",
             'Needs review' if r.get('needs_review') else ('Edited' if r.get('manual_override') else '')]
            for r in self.regions])
        for row, region in enumerate(self.regions):
            def score(key):
                return '—' if region.get(key) is None else f'{region[key]:.3f}'
            details = (f'Speech: {score("speech_score")} | Music: {score("music_score")} | '
                       f'Vocals: {score("vocal_score")} ({region.get("vocal_evidence_class") or "not recorded"})'
                       + f' | Separated voice evidence: {score("verified_voice_score")}'
                       + '\n' + str(region.get('note') or 'Automatic scores are uncalibrated.'))
            for column in range(self.ui.regionTable.columnCount()):
                self.ui.regionTable.item(row, column).setToolTip(details)
        segments = self.session['segments'] if self.session else []
        rows = []
        for s in segments:
            if s['source_mode'] == 'continuous_source': duration = s['source_end'] - s['source_start']
            else:
                a,b = review.candidate_trim_bounds(s)
                duration = b-a
            original = s.get('original') or {}
            rows.append([s['id'],f"{s['source_start']:.4f}",f"{s['source_end']:.4f}",f'{duration:.4f}',s['label'],s['status'],
                         original.get('kind','vad_candidate') + ' · ' + '; '.join(original.get('notes',[]))])
        self.populate(self.ui.segmentTable, rows)
        self.visible_spans = review.visible_unresolved(self.session) if self.session else []
        self.populate(self.ui.unresolvedTable, [[item['id'], f'{a:.4f}', f'{b:.4f}', item['label'], item['status'], item['original'].get('reason','')] for _,item,a,b in self.visible_spans])
        self.ui.unresolvedTable.clearSelection()
        self.refresh_snr()
        self.configure_region_preview()
        for kind in self.part4_previews:
            self.configure_part4_preview(kind)

    def region_selected(self):
        rows = self.selected(self.ui.regionTable)
        if rows:
            r = self.regions[rows[0]]
            self.ui.regionLabel.setCurrentText(r['label'])
            self.ui.regionStart.setValue(r['start'])
            self.ui.regionEnd.setValue(r['end'])
        self.configure_region_preview(reset=True)

    @staticmethod
    def clock(milliseconds):
        seconds, ms = divmod(max(0, int(milliseconds)), 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return (f'{hours:02d}:' if hours else '') + f'{minutes:02d}:{seconds:02d}.{ms:03d}'

    def configure_region_preview(self, reset=False):
        rows = self.selected(self.ui.regionTable)
        valid = self.audio is not None and len(rows) == 1 and rows[0] < len(self.regions)
        bounds = None
        if valid:
            r = self.regions[rows[0]]
            bounds = (round(r['start'] * 1000), round(r['end'] * 1000))
        slider = self.ui.regionSeekSlider
        enabled = bounds is not None and bounds[1] > bounds[0]
        for widget in (slider, self.ui.playRegionCursor, self.ui.pauseRegionPreview,
                       self.ui.markRegionStart, self.ui.markRegionEnd):
            widget.setEnabled(enabled)
        changed = bounds != self.preview_bounds
        if changed and self.play_path is not None and self.play_path == self.audio:
            self.stop_playback()
        self.preview_bounds = bounds
        slider.blockSignals(True)
        slider.setRange(*(bounds or (0, 0)))
        if reset or changed:
            position = bounds[0] if bounds else 0
            if bounds and self.play_path == self.audio and self.player is not None:
                position = min(bounds[1], max(bounds[0], self.player.position()))
            slider.setValue(position)
        slider.blockSignals(False)
        self.ui.regionSeekEnd.setText(self.clock(bounds[1] if bounds else 0))
        self.update_region_cursor_label()
        self.update_region_range_summary()

    def update_region_cursor_label(self):
        self.ui.regionCursorLabel.setText('Position ' + self.clock(self.ui.regionSeekSlider.value()))

    def update_region_range_summary(self, *args):
        rows = self.selected(self.ui.regionTable)
        if self.audio is None or len(rows) != 1 or rows[0] >= len(self.regions):
            self.ui.regionRangeSummary.setText('Select a region to preview its audio.')
            return
        region = self.regions[rows[0]]
        a, b = self.entered_region_range(region)
        if not region['start'] <= a < b <= region['end']:
            self.ui.regionRangeSummary.setText('Choose a start before the end, both inside the selected region, to preview a range.')
            return
        self.ui.regionRangeSummary.setText(
            f'Region {self.clock(round(region["start"] * 1000))} – {self.clock(round(region["end"] * 1000))}'
            f'  |  Listening range {self.clock(round(a * 1000))} – {self.clock(round(b * 1000))}'
            f'  |  Duration {b - a:.3f} s'
        )

    def region_slider_changed(self, value):
        self.update_region_cursor_label()
        if not self.ui.regionSeekSlider.isSliderDown():
            self.guard(self.seek_region_preview)

    def region_cursor_interval(self):
        self.require_audio()
        region = self.regions[self.index(self.ui.regionTable)]
        cursor = self.ui.regionSeekSlider.value() / 1000
        a, b = self.entered_region_range(region)
        end = b if region['start'] <= a <= cursor < b <= region['end'] else region['end']
        return cursor, end

    def seek_region_preview(self):
        if self.preview_bounds is None:
            return
        cursor, end = self.region_cursor_interval()
        if self.play_owner not in self.part4_previews and self.player is not None and self.play_path == self.audio and self.play_range:
            if cursor >= end:
                self.stop_playback()
                return
            # An active preview keeps its pause state when the user seeks.
            self.play_range = (round(cursor * 1000), round(end * 1000))
            self.pending_seek = self.play_range[0]
            self.player.setPosition(self.pending_seek)

    def mark_region_time(self, end):
        self.require_audio()
        self.index(self.ui.regionTable)
        widget = self.ui.regionEnd if end else self.ui.regionStart
        widget.setValue(self.ui.regionSeekSlider.value() / 1000)
        self.update_region_range_summary()

    def play_region_cursor(self):
        cursor, end = self.region_cursor_interval()
        if cursor >= end:
            raise ValueError('Move the playback slider before the end of the region.')
        self.play(self.audio, cursor, end)

    def pause_region_preview(self):
        if self.play_owner in self.part4_previews or self.player is None or self.play_path != self.audio or not self.play_range:
            self.play_region_cursor()
            return
        self.play_paused = not self.play_paused
        if self.play_paused:
            self.player.pause()
            self.play_timer.stop()
        else:
            if self.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
                self.player.play()
            self.play_timer.start(40)
        self.ui.pauseRegionPreview.setText('Resume' if self.play_paused else 'Pause')

    def region_mode_changed(self, index):
        self.ui.regionRangeHint.setText(
            'Enter source start/end times for the section you want to relabel. The surrounding audio keeps its current label.'
            if index == 0 else
            'Edit the entire selected region. Moving a shared boundary also adjusts the neighboring region.'
        )

    def entered_region_range(self, region):
        # The four-decimal display may round a sample-aligned boundary. Restore
        # that exact edge when the user leaves its displayed value unchanged.
        values = []
        for widget, key in ((self.ui.regionStart, 'start'), (self.ui.regionEnd, 'end')):
            value = widget.value()
            if value == round(region[key], widget.decimals()):
                value = region[key]
            values.append(value)
        return values

    def clear_snr(self):
        self.snr_results = []
        self.snr_snapshot = None
        self.snr_report = None
        self.refresh_snr()

    def invalidate_segments(self):
        if hasattr(self, 'asr'):
            self.asr.source_changed()
        self.session = None
        self.clear_snr()

    def edit_region(self):
        i = self.index(self.ui.regionTable)
        name = self.reviewer()
        a, b = self.ui.regionStart.value(), self.ui.regionEnd.value()
        # Snap to the original audio sample grid.
        a,b = round(a*16000)/16000, round(b*16000)/16000
        if not 0 <= a < b <= self.duration:
            raise ValueError('Start and end must lie inside the audio.')
        if (i == 0 and a != 0) or (i and a <= self.regions[i-1]['start']):
            raise ValueError('Keep the first start at zero and stay after the previous region start.')
        if (i == len(self.regions)-1 and abs(b-self.duration)>1/16000) or (i+1<len(self.regions) and b>=self.regions[i+1]['end']):
            raise ValueError('Keep the last end at audio duration and stay before the next region end.')
        before = copy.deepcopy(self.regions[max(0,i-1):i+2])
        r = self.regions[i]
        r.update(start=a,end=b,label=self.ui.regionLabel.currentText(),manual_override=True,needs_review=False)
        if i:
            if self.regions[i-1]['end'] != a: self.regions[i-1]['needs_review'] = True
            self.regions[i-1]['end'] = a
        if i+1 < len(self.regions):
            if self.regions[i+1]['start'] != b: self.regions[i+1]['needs_review'] = True
            self.regions[i+1]['start'] = b
        self.changes.append(dict(time_utc=review.utc_now(),annotator=name,region_index=i,action='edit',before=before,after=copy.deepcopy(self.regions[max(0,i-1):i+2])))
        self.invalidate_segments()
        self.dirty = True
        self.refresh_all()
        self.ui.regionTable.selectRow(i)
        self.redraw_regions()
        self.status('Region corrected. Earlier segment/SNR results cleared; regenerate from these labels.')

    def apply_region_correction(self):
        if self.ui.regionEditMode.currentIndex() == 1:
            self.edit_region()
            return
        i = self.index(self.ui.regionTable)
        name = self.reviewer()
        before = copy.deepcopy(self.regions[i])
        start, end = self.entered_region_range(before)
        pieces, selected = relabel_range(
            before, start, end,
            self.ui.regionLabel.currentText(),
        )
        draft = self.regions[:i] + pieces + self.regions[i + 1:]
        validate_regions(draft, self.duration)
        self.regions = draft
        self.changes.append(dict(
            time_utc=review.utc_now(), annotator=name, region_index=i,
            action='relabel_subrange', before=before, after=copy.deepcopy(pieces),
        ))
        self.invalidate_segments()
        self.dirty = True
        self.refresh_all()
        self.ui.regionTable.selectRow(i + selected)
        self.redraw_regions()
        self.status('Selected range relabeled; surrounding regions preserved. Regenerate segments and SNR before continuing to ASR.')

    def play_entered_region_range(self):
        self.require_audio()
        i = self.index(self.ui.regionTable)
        start, end = self.ui.regionStart.value(), self.ui.regionEnd.value()
        if self.ui.regionEditMode.currentIndex() == 0:
            start, end = self.entered_region_range(self.regions[i])
            # Validate the same interval that Apply correction will split.
            relabel_range(self.regions[i], start, end, self.ui.regionLabel.currentText())
        if not 0 <= start < end <= self.duration:
            raise ValueError('Choose start/end times within the source WAV.')
        self.play(self.audio, start, end)

    def split_region(self):
        i = self.index(self.ui.regionTable);name = self.reviewer()
        old = copy.deepcopy(self.regions[i]);mid = round((old['start']+old['end'])*8000)/16000
        if not old['start'] < mid < old['end']: raise ValueError('This region is too short to split.')
        left,right = copy.deepcopy(old),copy.deepcopy(old)
        left['end'],right['start'] = mid,mid
        for x in (left,right): x.update(manual_override=True,needs_review=True)
        self.regions[i:i+1] = [left,right]
        self.changes.append(dict(time_utc=review.utc_now(),annotator=name,region_index=i,action='split',before=old,after=[left.copy(),right.copy()]))
        self.invalidate_segments();self.dirty=True;self.refresh_all();self.redraw_regions()

    def merge_region(self):
        i = self.index(self.ui.regionTable);name = self.reviewer()
        if i+1>=len(self.regions): raise ValueError('There is no following region to merge.')
        before=copy.deepcopy(self.regions[i:i+2]);merged=copy.deepcopy(before[0])
        merged.update(end=before[1]['end'],label=self.ui.regionLabel.currentText(),manual_override=True,needs_review=True,
                      confidence=None,auto_confidence=None,note='Merged manually; original scores retained in correction history.')
        self.regions[i:i+2]=[merged]
        self.changes.append(dict(time_utc=review.utc_now(),annotator=name,region_index=i,action='merge',before=before,after=merged.copy()))
        self.invalidate_segments();self.dirty=True;self.refresh_all();self.redraw_regions()

    def plot_press(self, event):
        if event.inaxes != self.axes or event.xdata is None or not self.regions or self.task or self.window.worker:
            return
        if self.canvas.toolbar and self.canvas.toolbar.mode:
            return
        tolerance = (self.axes.get_xlim()[1]-self.axes.get_xlim()[0])*.008
        edges = [(abs(r['end']-event.xdata),i) for i,r in enumerate(self.regions[:-1])]
        if edges and min(edges)[0] < tolerance:
            self.drag_boundary=min(edges)[1]
            self.ui.regionTable.selectRow(self.drag_boundary)
        else:
            for i,r in enumerate(self.regions):
                if r['start']<=event.xdata<=r['end']:
                    self.ui.regionTable.selectRow(i);break

    def plot_release(self, event):
        i,self.drag_boundary=self.drag_boundary,None
        if i is not None and event.inaxes == self.axes and event.xdata is not None:
            self.ui.regionTable.selectRow(i)
            self.ui.regionEditMode.setCurrentIndex(1)
            self.ui.regionEnd.setValue(event.xdata)
            self.guard(self.edit_region)

    def source_folder(self):
        folder = self.audio.parent
        return folder.parent if folder.name == 'Extract audio' else folder

    def write_region_report(self):
        folder = self.source_folder() / 'Labeled audio regions'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (self.audio.stem + '_regions.html')
        save_report(path, self.region_payload())
        return path

    def region_payload(self):
        self.require_audio();validate_regions(self.regions,self.duration)
        return dict(version=1,audio_name=self.audio.name,audio_size=self.audio.stat().st_size,duration_seconds=self.duration,
                    method='PANNs adjustable windows/hops and RMS silence threshold; optional Hybrid Demucs vocal verification; input bounds and rules recorded per prediction',
                    analysis_settings=recorded_analysis_settings(self.regions),
                    score_note='Uncalibrated automatic scores; manual boundaries may differ from inference windows. Vocal-music proposals are mixed under corpus policy and require listening review; context-based evidence may blur boundaries and does not confirm lyrics.',
                    regions=self.regions,changes=self.changes)

    def save_regions(self):
        self.require_audio()
        folder = QFileDialog.getExistingDirectory(self.window, 'Choose folder for region report', str(self.source_folder()))
        if not folder:
            return
        target = Path(folder) / 'Labeled audio regions'
        target.mkdir(parents=True, exist_ok=True)
        path = target / (self.audio.stem + '_regions.html')
        save_report(path, self.region_payload())
        self.status('Region report saved: ' + str(path))

    def load_regions(self):
        self.require_audio()
        path,_=QFileDialog.getOpenFileName(self.window,'Open region review','','Region report (*.html *.htm *.json)')
        if not path:return
        data=load_review(path)
        self.check_region_payload(data)
        if not self.confirm_discard():return
        self.regions=copy.deepcopy(data['regions']);self.changes=data.get('changes',[])
        config=recorded_analysis_settings(self.regions)
        if config:
            self.ui.pannsWindowInput.setText(f'{config["window_seconds"]:g}')
            self.ui.pannsHopInput.setText(f'{config["hop_seconds"]:g}')
        self.invalidate_segments();self.dirty=True;self.refresh_all();self.redraw_regions()
        self.status('Region review loaded. Show spectrogram if needed.')

    def check_region_payload(self,data):
        if data['audio_name']!=self.audio.name or data['audio_size']!=self.audio.stat().st_size or abs(data['duration_seconds']-self.duration)>1/16000:
            raise ValueError('Region report does not match the selected audio.')
        validate_regions(data['regions'],self.duration)

    def generate_segments(self, output_folder=None, confirmed=False):
        self.require_audio();validate_regions(self.regions,self.duration)
        if self.session and not confirmed and not self.confirm_discard():return
        folder = str(output_folder) if output_folder is not None else QFileDialog.getExistingDirectory(self.window,'Choose output folder',str(self.source_folder()))
        if not folder:return
        folder = str(Path(folder) / 'Segmenting the speech')
        regions=copy.deepcopy(self.regions);path=str(self.audio)
        def action(log):
            from pause_based_segmentation import export_candidates
            result=export_candidates(path,regions,folder,log)
            return review.load_session(Path(result['output_dir'])/'step4_segments.json',path)
        self.start('Finding speech and natural pause candidates…',action,self.set_session)

    def set_session(self,data):
        self.stop_playback()
        if hasattr(self,'asr'):
            self.asr.source_changed()
        self.session=data;self.clear_snr();self.dirty=True;self.refresh_all()
        self.status(f"{len(data['segments'])} candidates · {len(data['unresolved'])} unresolved spans/gaps. Listen before approving.")
        self.window.navigationList.setCurrentRow(2)

    def load_candidates(self):
        self.require_audio()
        path,_=QFileDialog.getOpenFileName(self.window,'Open candidates or review_session.json','','Step 4 JSON (*.json)')
        if not path:return
        data=review.load_session(path,self.audio)
        if self.session and not self.confirm_discard():return
        self.set_session(data)

    def segment_selected(self):
        rows=self.selected(self.ui.segmentTable)
        if len(rows) != 1 or not self.session:
            self.configure_part4_preview('segment')
            return
        item=self.session['segments'][rows[0]]
        self.ui.segmentLabel.setCurrentText(item['label'])
        self.ui.boundaryMode.blockSignals(True)
        if item['source_mode']=='continuous_source':
            self.ui.boundaryMode.setCurrentIndex(1);a,b=item['source_start'],item['source_end']
        else:
            self.ui.boundaryMode.setCurrentIndex(0);a,b=review.candidate_trim_bounds(item)
        self.ui.boundaryMode.blockSignals(False)
        self.ui.segmentStart.setValue(a);self.ui.segmentEnd.setValue(b)
        self.configure_part4_preview('segment', reset=True)

    def segment_mode_changed(self):
        rows = self.selected(self.ui.segmentTable)
        if not self.session or len(rows) != 1:
            return
        item = self.session['segments'][rows[0]]
        if self.ui.boundaryMode.currentIndex() == 0:
            if item['source_mode'] == 'continuous_source':
                self.ui.boundaryMode.blockSignals(True)
                self.ui.boundaryMode.setCurrentIndex(1)
                self.ui.boundaryMode.blockSignals(False)
                raise ValueError('This clip uses source times. Keep source boundary mode for this clip.')
            a, b = review.candidate_trim_bounds(item)
        else:
            a, b = item['source_start'], item['source_end']
        self.ui.segmentStart.setValue(a)
        self.ui.segmentEnd.setValue(b)
        self.configure_part4_preview('segment', reset=True)

    def part4_spec(self, kind):
        """Resolve the audible file and its time system without flattening compact joins."""
        table = self.ui.segmentTable if kind == 'segment' else self.ui.unresolvedTable
        rows = self.selected(table)
        if self.audio is None or not self.session or len(rows) != 1:
            return None
        if kind == 'segment':
            items = self.session['segments']
            if rows[0] >= len(items):
                return None
            item = items[rows[0]]
        else:
            if rows[0] >= len(self.visible_spans):
                return None
            _, item, visible_start, visible_end = self.visible_spans[rows[0]]
        if kind == 'segment' and self.ui.boundaryMode.currentIndex() == 0:
            if not item.get('candidate_file') or item['source_mode'] == 'continuous_source':
                return None
            path = review.safe_candidate_path(self.session['candidate_root'], item['candidate_file'])
            a, b = 0., float(item['original']['duration'])
            clock = 'Clip time (joins preserved)'
        else:
            path, clock = self.audio, 'Source time'
            if kind == 'segment':
                # A little surrounding audio makes source-boundary corrections possible.
                a, b = max(0., item['source_start'] - 2.), min(self.duration, item['source_end'] + 2.)
            else:
                a, b = visible_start, visible_end
        return dict(path=Path(path).resolve(), bounds=(round(a * 1000), round(b * 1000)),
                    token=(item['id'], clock, a, b), clock=clock)

    def configure_part4_preview(self, kind, reset=False):
        spec = self.part4_spec(kind)
        old = self.part4_previews[kind]
        changed = spec != old
        if changed and self.play_owner == kind:
            self.stop_playback()
        self.part4_previews[kind] = spec
        slider = getattr(self.ui, kind + 'SeekSlider')
        enabled = spec is not None and spec['bounds'][1] > spec['bounds'][0]
        for widget in (slider, getattr(self.ui, 'play' + kind.title() + 'Cursor'),
                       getattr(self.ui, 'pause' + kind.title() + 'Preview'),
                       getattr(self.ui, 'mark' + kind.title() + 'Start'),
                       getattr(self.ui, 'mark' + kind.title() + 'End')):
            widget.setEnabled(enabled)
        slider.blockSignals(True)
        slider.setRange(*(spec['bounds'] if spec else (0, 0)))
        if changed or reset:
            a = getattr(self.ui, kind + 'Start').value()
            slider.setValue(round(a * 1000) if spec else 0)
        slider.blockSignals(False)
        getattr(self.ui, kind + 'SeekEnd').setText(self.clock(slider.maximum()))
        self.update_part4_summary(kind)

    def update_part4_summary(self, kind):
        slider = getattr(self.ui, kind + 'SeekSlider')
        getattr(self.ui, kind + 'CursorLabel').setText('Position ' + self.clock(slider.value()))
        spec = self.part4_previews[kind]
        text = 'Select one candidate to listen and mark a range.' if kind == 'segment' else 'Select an unresolved span to listen and mark a range.'
        if spec:
            a, b = getattr(self.ui, kind + 'Start').value(), getattr(self.ui, kind + 'End').value()
            lo, hi = (v / 1000 for v in spec['bounds'])
            text = f'{spec["clock"]}  |  Marked range {self.clock(round(a * 1000))} – {self.clock(round(b * 1000))}  |  Duration {b-a:.3f} s'
            if not lo <= a < b <= hi:
                text += '  |  Enter start before end inside the listening range.'
            elif kind == 'segment' and not 5 <= b-a <= 15:
                text += '  |  Saved clips must be 5–15 s.'
        getattr(self.ui, kind + 'RangeSummary').setText(text)

    def part4_slider_changed(self, kind):
        self.update_part4_summary(kind)
        if not getattr(self.ui, kind + 'SeekSlider').isSliderDown():
            self.guard(lambda: self.seek_part4(kind))

    def part4_interval(self, kind, cursor=False):
        spec = self.part4_previews[kind]
        if spec is None:
            raise ValueError('Select one item in this panel first.')
        lo, hi = (v / 1000 for v in spec['bounds'])
        a, b = getattr(self.ui, kind + 'Start').value(), getattr(self.ui, kind + 'End').value()
        if cursor:
            position = getattr(self.ui, kind + 'SeekSlider').value() / 1000
            end = b if lo <= a <= position < b <= hi else hi
            a, b = position, end
        if not lo <= a < b <= hi:
            raise ValueError('Choose start before end inside the listening range; move the slider before its end.')
        return spec['path'], a, b

    def seek_part4(self, kind):
        if self.play_owner != kind or not self.play_range or self.player is None:
            return
        spec = self.part4_previews[kind]
        if spec is None:
            return
        position = getattr(self.ui, kind + 'SeekSlider').value()
        if position >= spec['bounds'][1]:
            self.stop_playback()
            return
        path, a, b = self.part4_interval(kind, cursor=True)
        self.play_range = (round(a * 1000), round(b * 1000))
        self.pending_seek = self.play_range[0]
        self.player.setPosition(self.pending_seek)

    def mark_part4(self, kind, end):
        if self.part4_previews[kind] is None:
            raise ValueError('Select one item in this panel first.')
        getattr(self.ui, kind + ('End' if end else 'Start')).setValue(
            getattr(self.ui, kind + 'SeekSlider').value() / 1000)

    def play_part4_cursor(self, kind):
        path, a, b = self.part4_interval(kind, cursor=True)
        self.play(path, a, b, owner=kind)

    def pause_part4(self, kind):
        if self.play_owner != kind or self.player is None or not self.play_range:
            self.play_part4_cursor(kind)
            return
        self.play_paused = not self.play_paused
        if self.play_paused:
            self.player.pause()
            self.play_timer.stop()
        else:
            if self.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
                self.player.play()
            self.play_timer.start(40)
        getattr(self.ui, 'pause' + kind.title() + 'Preview').setText('Resume' if self.play_paused else 'Pause')

    def update_part4_playback_position(self, position):
        kind = self.play_owner
        if kind not in self.part4_previews:
            return
        slider = getattr(self.ui, kind + 'SeekSlider')
        if slider.isSliderDown():
            return
        slider.blockSignals(True)
        slider.setValue(position)
        slider.blockSignals(False)
        self.update_part4_summary(kind)

    def modified_session(self):
        if hasattr(self, 'asr'):
            self.asr.source_changed()
        self.clear_snr();self.dirty=True;self.refresh_all()
        self.status('Part 4 changed. Recalculate SNR to update the resolved clips and unresolved spans.')

    def edit_segment(self):
        i=self.index(self.ui.segmentTable);name=self.reviewer()
        a,b=self.ui.segmentStart.value(),self.ui.segmentEnd.value()
        if self.ui.boundaryMode.currentIndex()==0:
            review.set_candidate_trim(self.session,i,a,b,name)
            item=self.session['segments'][i]
            review.set_segment(self.session,i,self.ui.segmentLabel.currentText(),item['source_start'],item['source_end'],name)
        else:
            # Explicitly convert compact candidates to a continuous source interval.
            review.validate_bounds(a,b,self.duration)
            review.set_segment(self.session,i,self.ui.segmentLabel.currentText(),a,b,name)
            self.session['segments'][i]['source_mode'] = 'continuous_source'
            # Keep the existing edit audit in sync even when only the mode changes.
            self.session['changes'][-1]['after'] = copy.deepcopy(self.session['segments'][i])
        self.modified_session();self.ui.segmentTable.selectRow(i)

    def approve(self):
        if not self.session:raise ValueError('Generate or open candidates first.')
        count=review.approve_segments(self.session,self.selected(self.ui.segmentTable),self.reviewer())
        self.modified_session();self.status(f'{count} clips approved. Export or calculate SNR when ready.')

    def exclude(self):
        rows=self.selected(self.ui.segmentTable);name=self.reviewer()
        if not rows:raise ValueError('Select one or more clips.')
        for i in rows:review.exclude_segment(self.session,i,name)
        self.modified_session()

    def resolve_overlaps(self):
        if not self.session:raise ValueError('Open candidates first.')
        result=review.resolve_approved_overlaps(self.session,self.reviewer())
        self.modified_session();self.status(str(result))

    def unresolved_selected(self):
        rows=self.selected(self.ui.unresolvedTable)
        if not rows or not self.session:
            self.configure_part4_preview('unresolved')
            return
        _, item, a, b = self.visible_spans[rows[0]]
        self.ui.unresolvedStart.setValue(a);self.ui.unresolvedEnd.setValue(b)
        if item['label'] in LABELS:self.ui.unresolvedLabel.setCurrentText(item['label'])
        self.configure_part4_preview('unresolved', reset=True)

    def add_manual(self):
        row = self.index(self.ui.unresolvedTable)
        i, _, a, b = self.visible_spans[row]
        if not a - 1/16000 <= self.ui.unresolvedStart.value() < self.ui.unresolvedEnd.value() <= b + 1/16000:
            raise ValueError('Select a range inside the visible unresolved span.')
        review.add_from_unresolved(self.session,i,self.ui.unresolvedStart.value(),self.ui.unresolvedEnd.value(),self.ui.unresolvedLabel.currentText(),self.reviewer())
        self.modified_session()

    def mark_unresolved(self):
        row = self.index(self.ui.unresolvedTable)
        i, _, a, b = self.visible_spans[row]
        if not a - 1/16000 <= self.ui.unresolvedStart.value() < self.ui.unresolvedEnd.value() <= b + 1/16000:
            raise ValueError('Select a range inside the visible unresolved span.')
        name=self.reviewer()
        review.label_unresolved(self.session,i,self.ui.unresolvedLabel.currentText(),name,self.ui.unresolvedStart.value(),self.ui.unresolvedEnd.value())
        review.mark_unresolved_reviewed(self.session,i,name)
        self.modified_session()

    def require_approved(self):
        self.require_audio()
        if not self.session:raise ValueError('Generate or load candidates and approve clips first.')
        if review.overlapping_approved(self.session):
            raise ValueError('Approved clips overlap. Review the times or use Resolve approved overlaps before exporting or calculating SNR.')
        if not any(s['status']=='approved' for s in self.session['segments']):
            raise ValueError('Approve at least one clip first.')

    def export_segments(self):
        self.require_approved()
        folder=QFileDialog.getExistingDirectory(self.window,'Export approved WAV files',str(self.source_folder()))
        if not folder:return
        folder = str(Path(folder) / 'Segmenting the speech')
        snapshot=copy.deepcopy(self.session);path=str(self.audio)
        def done(result):
            self.status(f"Exported {result['exported']} approved clips to {result['output_dir']}. Pending spans: {result['pending_unresolved']}.")
            self.window.open_path(Path(result['output_dir']))
        self.start('Exporting reviewed clips and source mappings…',lambda log:review.save_and_export(snapshot,path,folder),done)

    def require_analysis_source(self):
        self.require_audio()
        if not self.session:
            raise ValueError('Generate or open Part 4 segments first.')
        if any(item['status']=='approved' for item in self.session['segments']):
            self.require_approved()

    def calculate_snr(self):
        self.require_analysis_source()
        snapshot=copy.deepcopy(self.session);path=str(self.audio);threshold=self.ui.snrThreshold.value()
        previous={r['id']:copy.deepcopy(r) for r in self.snr_results}
        policy=snr.POLICIES[self.ui.snrPolicy.currentIndex()]
        justification=self.ui.snrJustification.text().strip()
        def action(log):
            records=snr.session_records(snapshot,path)
            return snr.calculate(records,threshold,lambda n,total:log(f'Calculating pause-energy + WADA-SNR: {n} / {total}'),policy=policy,justification=justification)
        def done(records):
            for item in records:snr.preserve_override(item,previous.get(item['id']))
            self.snr_results=records;self.snr_snapshot=snapshot;self.applied_threshold=threshold
            self.snr_policy=policy;self.snr_justification=justification
            self.snr_report=None;self.dirty=True;self.refresh_snr()
            self.status(f'{len(records)} two-method SNR results ready; {sum(r.get("review_required",True) for r in records)} need quality review.')
        self.start('Calculating SNR for approved clips and unresolved spans…',action,done)

    def snr_controls_changed(self,*args):
        threshold=self.ui.snrThreshold.value()
        policy=snr.POLICIES[max(0,self.ui.snrPolicy.currentIndex())]
        justification=self.ui.snrJustification.text().strip()
        if (threshold,policy,justification)==(self.applied_threshold,self.snr_policy,self.snr_justification):return
        self.applied_threshold=threshold;self.snr_policy=policy;self.snr_justification=justification
        if self.audio is not None:self.dirty=True
        if self.snr_results:
            self.snr_results=snr.reclassify(self.snr_results,threshold,policy,justification)
            self.snr_report=None;self.dirty=True;self.refresh_snr()
            self.status('Cached SNR estimates reclassified. Human quality corrections were preserved.')

    def refresh_snr(self):
        if not hasattr(self,'histogram'):return
        def number(value):return 'Unavailable' if value is None else f'{value:.2f}'
        rows=[]
        for r in self.snr_results:
            bound=(r.get('wada_details') or {}).get('bound')
            wada=number(r.get('wada_snr_db'))
            if bound:wada+=' (range limit)'
            basis={'reviewed_part4_pause_context':'Part 4 pauses','estimated_part4_pause':'Estimated pauses','quiet_frame_fallback':'Quiet-frame fallback','unavailable':'Unavailable'}.get(r.get('noise_source'),'Legacy estimate')
            rows.append([r['id'],f"{r['source_start']:.3f}",f"{r['source_end']:.3f}",r.get('segmentation_status','resolved').title(),r['content_label'],
                number(r.get('energy_snr_db',r.get('snr_db'))),wada,number(r.get('snr_db')),
                r['automatic_quality'],r['quality'],r.get('review_status','Legacy: recalculate'),basis,r['note']])
        self.populate(self.ui.snrTable,rows)
        self.histogram.clear();a,b=self.histogram.subplots(1,2)
        for key,label,color in [('energy_snr_db','Pause energy','#157A6E'),('wada_snr_db','WADA-SNR','#7C3AED')]:
            scores=[r[key] for r in self.snr_results if r.get(key) is not None]
            if scores:a.hist(scores,bins=min(20,max(3,len(scores))),color=color,alpha=.6,label=label)
        a.axvline(self.applied_threshold,color='#d59a45',linestyle='--',label=f'{self.applied_threshold:g} dB')
        a.set(xlabel='Estimated SNR (dB)',ylabel='Clips',title='Two-method comparison');a.legend(fontsize=8)
        labels=['clean','noisy','unavailable'];counts=[sum(r['quality']==l for r in self.snr_results) for l in labels]
        b.bar(labels,counts,color=['#268b80','#e6ab54','#a1afb4']);b.set(title='Reviewed quality',ylabel='Clips')
        self.hist_canvas.draw_idle()

    def check_snr(self):
        if not self.snr_results or self.snr_snapshot!=self.session:
            raise ValueError('Clips changed or no results are loaded. Calculate SNR again.')

    def override_quality(self,label):
        self.check_snr();rows=self.selected(self.ui.snrTable);who=self.reviewer();why=self.ui.overrideReason.text().strip()
        if not rows or not why:raise ValueError('Select clips and enter the correction reason.')
        for i in rows:
            snr.set_quality(self.snr_results[i],label,who,why)
        self.dirty=True;self.refresh_snr();self.status('Quality corrections recorded. Save the session or updated report.')

    def save_snr(self):
        self.check_snr()
        self.snr_controls_changed()
        if not self.snr_justification:raise ValueError('Enter a threshold justification before saving the SNR report.')
        folder=QFileDialog.getExistingDirectory(self.window,'Save SNR HTML / CSV / JSON',str(self.source_folder()))
        if not folder:return
        folder = str(Path(folder) / 'Audio quality SNR')
        self.snr_report=Path(snr.export_report(self.snr_results,self.applied_threshold,folder,policy=self.snr_policy,justification=self.snr_justification)[0])
        self.status('Saved SNR reports: '+str(self.snr_report))

    def open_snr_report(self):
        if not self.snr_report:raise ValueError('Save the SNR report first.')
        self.window.open_path(self.snr_report)

    def load_snr(self):
        self.require_analysis_source()
        path,_=QFileDialog.getOpenFileName(self.window,'Open SNR review','','SNR JSON (*.json)')
        if not path:return
        payload=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if payload.get('format') not in ('speech_corpus_snr_v1','speech_corpus_snr_v2'):raise ValueError('Not a supported SNR report.')
        records=self.match_snr(payload['segments'])
        threshold=float(payload['threshold_db'])
        if not math.isfinite(threshold) or not -50<=threshold<=100:raise ValueError('Invalid SNR threshold.')
        policy=payload.get('decision_policy','energy');justification=payload.get('threshold_justification','')
        self.snr_results=snr.reclassify(records,threshold,policy,justification)
        self.snr_snapshot=copy.deepcopy(self.session);self.applied_threshold=threshold
        self.snr_policy=policy;self.snr_justification=justification
        self.set_snr_controls();self.snr_report=Path(path).with_suffix('.html')
        self.dirty=True;self.refresh_snr();self.status('SNR review loaded.')

    def match_snr(self,records):
        # Earlier reports contain only approved clips. They can still be resumed.
        includes_spans = any(r.get('segmentation_status')=='unresolved' for r in records)
        current=snr.session_records(self.session,str(self.audio),include_unresolved=includes_spans)
        records=sorted(records,key=snr.source_time_key)
        keys=('id','path','read_start','read_end','source_start','source_end','content_label','synthetic_start','synthetic_end')
        if len(records)!=len(current) or any(any(a.get(k)!=b.get(k) for k in keys) for a,b in zip(records,current)):
            raise ValueError('Saved SNR results do not match the current Part 4 intervals.')
        for a,b in zip(records,current):
            if a.get('analysis_version')==2 and any(a.get(k)!=b.get(k) for k in ('pause_intervals','audio_file_size','audio_file_mtime_ns')):
                raise ValueError('SNR audio or pause evidence changed. Recalculate both estimators.')
        for r in records:
            for key in ('snr_db','energy_snr_db','wada_snr_db'):
                if r.get(key) is not None and not math.isfinite(float(r[key])):raise ValueError('Invalid SNR result.')
            if r['quality'] not in ('clean','noisy','unavailable'):raise ValueError('Invalid quality label.')
        return records

    def set_snr_controls(self):
        for widget in (self.ui.snrThreshold,self.ui.snrPolicy,self.ui.snrJustification):widget.blockSignals(True)
        self.ui.snrThreshold.setValue(self.applied_threshold)
        self.ui.snrPolicy.setCurrentIndex(snr.POLICIES.index(self.snr_policy))
        self.ui.snrJustification.setText(self.snr_justification)
        for widget in (self.ui.snrThreshold,self.ui.snrPolicy,self.ui.snrJustification):widget.blockSignals(False)

    def save_project(self):
        self.require_audio()
        path,_=QFileDialog.getSaveFileName(self.window,'Save complete review session',str(self.audio.with_name(self.audio.stem+'_session.json')),'Review session (*.json)')
        if not path:return
        payload=self.project_payload()
        atomic_json(path,payload);self.dirty=False;self.status('Complete review session saved. Keep the source and candidate folders in place.')

    def project_payload(self):
        self.require_audio()
        if hasattr(self, 'asr'):
            self.asr.commit_drafts()
        return dict(format='speech_corpus_qt_session_v1',audio=str(self.audio),audio_size=self.audio.stat().st_size,duration=self.duration,
                    reviewer=self.ui.reviewer.text(),regions=copy.deepcopy(self.regions),changes=copy.deepcopy(self.changes),step4=copy.deepcopy(self.session),
                    snr=copy.deepcopy(self.snr_results),snr_threshold=self.applied_threshold,updated_utc=review.utc_now(),
                    snr_policy=self.snr_policy,snr_justification=self.ui.snrJustification.text().strip(),
                    improve_singing=self.ui.improveSinging.isChecked(),
                    panns_inputs=dict(window=self.ui.pannsWindowInput.text(), hop=self.ui.pannsHopInput.text()),
                    asr=self.asr.payload() if hasattr(self,'asr') else None)

    def open_project(self):
        path,_=QFileDialog.getOpenFileName(self.window,'Open complete review session','','Review session (*.json)')
        if not path:return
        data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if data.get('format')!='speech_corpus_qt_session_v1':raise ValueError('Not a combined PyQt session.')
        audio=Path(data['audio']);frames,rate=review.inspect_wav(audio)
        if data['audio_size']!=audio.stat().st_size or abs(data['duration']-frames/rate)>1/rate:
            raise ValueError('The saved source audio has changed.')
        if data['regions']:validate_regions(data['regions'],data['duration'])
        if data.get('step4'):
            source=data['step4']['source'];review.check_source(audio,source['audio_name'],source['audio_size'],source['duration_seconds'])
            if not Path(data['step4']['candidate_root']).is_dir():raise ValueError('Keep the original candidate folder to resume review.')
        if not self.confirm_discard():return
        self.set_audio(audio,ask=False)
        self.regions=data['regions'];self.changes=data['changes'];self.session=data.get('step4')
        self.snr_results=[];self.snr_snapshot=None;self.applied_threshold=float(data.get('snr_threshold',20))
        self.snr_policy=data.get('snr_policy','energy');self.snr_justification=data.get('snr_justification','')
        if self.snr_policy not in snr.POLICIES:raise ValueError('Invalid saved SNR decision rule.')
        self.ui.reviewer.setText(data.get('reviewer',''));self.set_snr_controls()
        if data.get('snr'):
            rows=self.match_snr(data['snr'])
            self.snr_results=snr.reclassify(rows,self.applied_threshold,self.snr_policy,self.snr_justification)
            self.snr_snapshot=copy.deepcopy(self.session)
        self.ui.improveSinging.setChecked(data.get('improve_singing', True))
        inputs=data.get('panns_inputs') or {}
        self.ui.pannsWindowInput.setText(str(inputs.get('window', '')))
        self.ui.pannsHopInput.setText(str(inputs.get('hop', '')))
        self.asr.restore(data.get('asr'), saved_snr=data.get('snr'))
        self.dirty=False;self.refresh_all();self.redraw_regions();self.status('Session restored. Click Show spectrogram to rebuild the display.')

    def play(self,path,a,b,owner=None):
        if b<=a:raise ValueError('No playable audio in the selected interval.')
        path = Path(path).resolve()
        frames,rate=review.inspect_wav(path)
        a,b=max(0,a),min(frames/rate,b)
        if b<=a:raise ValueError('No playable audio in the selected interval.')
        self.stop_playback()
        if self.player is None:
            self.player=QMediaPlayer(self)
            self.audio_output=QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
            self.player.mediaStatusChanged.connect(self.media_ready)
            self.player.positionChanged.connect(self.playback_position)
            self.player.errorOccurred.connect(lambda error,text:self.status('Playback: '+text))
        self.play_path = path
        self.play_owner = owner
        self.play_range=(round(a*1000),round(b*1000))
        self.pending_seek = self.play_range[0]
        self.update_part4_playback_position(self.pending_seek)
        if owner in self.part4_previews:
            getattr(self.ui, 'pause' + owner.title() + 'Preview').setText('Pause')
        if path == self.audio:
            self.update_region_playback_position(self.pending_seek)
            self.ui.pauseRegionPreview.setText('Pause')
        url=QUrl.fromLocalFile(str(Path(path).resolve()))
        if self.player.source()==url and self.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia,QMediaPlayer.MediaStatus.BufferedMedia):
            self.player.setPosition(self.play_range[0]);self.player.play()
        else:
            self.player.setSource(url)
        self.play_timer.start(40)

    def media_ready(self,status):
        if status==QMediaPlayer.MediaStatus.LoadedMedia and self.play_range:
            self.player.setPosition(self.pending_seek if self.pending_seek is not None else self.play_range[0])
            if not self.play_paused:self.player.play()
        elif status==QMediaPlayer.MediaStatus.EndOfMedia and self.play_range:
            self.update_region_playback_position(self.play_range[1])
            self.update_part4_playback_position(self.play_range[1])
            self.stop_playback()

    def update_region_playback_position(self, position):
        slider = self.ui.regionSeekSlider
        if self.play_owner in self.part4_previews or self.play_path != self.audio or self.preview_bounds is None or slider.isSliderDown():
            return
        slider.blockSignals(True)
        slider.setValue(position)
        slider.blockSignals(False)
        self.update_region_cursor_label()

    def playback_position(self,position):
        self.update_region_playback_position(position)
        self.update_part4_playback_position(position)
        if self.play_range and position>=self.play_range[1]:self.stop_playback()

    def enforce_playback_end(self):
        if self.play_range and self.player is not None:self.playback_position(self.player.position())

    def stop_playback(self):
        self.play_range=None
        self.play_path=None
        self.play_owner=None
        self.pending_seek=None
        self.play_paused=False
        if self.player is not None:self.player.stop()
        self.play_timer.stop()
        self.ui.pauseRegionPreview.setText('Pause / resume')
        for kind in self.part4_previews:
            getattr(self.ui, 'pause' + kind.title() + 'Preview').setText('Pause / resume')

    def play_region(self):
        r=self.regions[self.index(self.ui.regionTable)];self.play(self.audio,r['start'],r['end'])

    def play_region_edge(self,end):
        r=self.regions[self.index(self.ui.regionTable)];t=r['end' if end else 'start'];self.play(self.audio,t-1,t+1)

    def play_segment(self):
        path,a,b=self.part4_interval('segment')
        self.play(path,a,b,owner='segment')

    def play_segment_edge(self,end):
        item=self.session['segments'][self.index(self.ui.segmentTable)];t=item['source_end' if end else 'source_start']
        self.play(self.audio,t-1,t+1)

    def play_unresolved(self):
        path,a,b=self.part4_interval('unresolved')
        self.play(path,a,b,owner='unresolved')

    def play_quality(self):
        self.check_snr();r=self.snr_results[self.index(self.ui.snrTable)];self.play(r['path'],r['read_start'],r['read_end'])

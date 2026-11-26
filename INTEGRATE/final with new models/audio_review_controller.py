"""Connect the region, segment and quality pages to the current recording.

ProcessingWorkspace handles listening, boundary edits, approvals and session saves.
Task runs slow work off the GUI thread. start_auto_process moves between stages after
each result is ready.
"""

from output_folders import new_run_folder, project_folder, save_dialog_folder, step_folder
from processing_errors import exception_details, save_failure_log
from table_sizing import TableSizing
import copy
import json
import math
import tempfile
import wave
from contextlib import contextmanager
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, QThread, QTimer, QUrl, QLocale, QSignalBlocker, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QInputDialog,
                            QMessageBox, QTableWidgetItem)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
import soundfile as sf

from audio_spectrogram import calculate_spectrogram
from region_review_report import load_review, save_report, confidence_text
from audio_region_classifier import classify_file, analysis_settings, recorded_analysis_settings
from region_editing import relabel_range
import segment_review as review
import clip_quality_assessment as snr

LABELS = ('speech', 'music', 'mixed', 'silence')
COLORS = dict(speech='#268b80', music='#a77cbc', mixed='#e6ab54', silence='#a1afb4')


# Swap in the finished JSON file so an interrupted save leaves the last good session intact.
def atomic_json(path, data):
    path = Path(path)
    temp = None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,
                                         suffix='.tmp',delete=False) as stream:
            temp = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


# Check the region boundaries before trusting them as a timeline for this recording.
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

    # Keep the work and its result together so the window can pick them up when the thread
    # finishes.
    def __init__(self, action, parent):
        super().__init__(parent)
        self.action = action
        self.value = None
        self.failure = None
        self.failure_traceback = ''

    # Run the job here and let the window handle either the result or the error.
    def run(self):
        try:
            self.value = self.action(self.message.emit)
        except Exception as exc:
            self.failure, self.failure_traceback = exception_details(exc)


class ProcessingWorkspace(QObject):
    # Wire the region, segment and quality pages to the same recording and player.
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.ui = window
        self.audio = None
        self.duration = 0.0
        self.regions = []
        self.changes = []
        self.session = None
        self.snr_view_dirty = False
        self.snr_results = []
        self.snr_snapshot = None
        self.applied_threshold = 20.0
        self.snr_policy = 'conservative'
        self.snr_justification = ''
        self.snr_report = None
        self.task = None
        self.auto_stage = None
        self.dirty = False
        self.play_range = None
        self.play_path = None
        self.pending_seek = None
        self.play_paused = False
        self.preview_bounds = None
        self.part4_previews = {'segment': None, 'unresolved': None}
        self.play_owner = None
        self.plot_data = None
        self.region_overlay = None
        self.drag_boundary = None
        self.finished_callback = None


        self.figure = Figure(figsize=(9, 3), constrained_layout=True, facecolor='#F8FAFB')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(290)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title('Choose audio and display its spectrogram', color='#173B40')
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
        self.equal_columns(self.ui.regionTable)
        self.table(self.ui.segmentTable)
        self.table(self.ui.unresolvedTable)
        self.table(self.ui.snrTable)
        self.ui.regionTable.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ui.unresolvedTable.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.ui.regionTable.itemSelectionChanged.connect(self.region_selected)
        self.ui.regionEditMode.currentIndexChanged.connect(self.region_mode_changed)
        self.ui.regionStart.valueChanged.connect(self.update_region_range_summary)
        self.ui.regionEnd.valueChanged.connect(self.update_region_range_summary)
        self.ui.regionSeekSlider.valueChanged.connect(self.region_slider_changed)
        self.ui.regionSeekSlider.sliderReleased.connect(lambda: self.guard(self.seek_region_preview))
        self.ui.segmentTable.itemSelectionChanged.connect(self.segment_selected)
        self.ui.unresolvedTable.itemSelectionChanged.connect(self.unresolved_selected)
        self.ui.snrThreshold.setKeyboardTracking(False)
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
            'approveAllSegments': self.approve_all, 'toggleSpanAsr': self.toggle_span_asr,
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

    # A resize or layout change can change how far the active page should scroll.
    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.LayoutRequest, QEvent.Type.Resize, QEvent.Type.Show):
            self.page_size_timer.start(0)
        return super().eventFilter(watched, event)

    # Use the height of the visible page so the user can't scroll far past its last control.
    def fit_page_height(self):
        pages = self.window.pages
        page = pages.currentWidget()
        if page is None:
            return
        height = max(page.minimumSizeHint().height(), page.minimumHeight(),
                     page.heightForWidth(pages.width()) if page.hasHeightForWidth() else page.sizeHint().height())
        if pages.minimumHeight() != height or pages.maximumHeight() != height:
            pages.setFixedHeight(height)

    # Switch pages and bring the newly visible review controls up to date.
    def navigate(self, index):
        self.window.pages.setCurrentIndex(index)
        if hasattr(self,'asr'):
            if index == 5:
                self.asr.flush_review_rows()
                self.asr.refresh_summary()
                self.asr.select_final()
            elif index == 4 and not self.asr.transcript_draft:
                self.asr.select_asr()
        if index == 3 and self.snr_view_dirty:
            self.refresh_snr()
        processing_page = 1 <= index <= 5
        self.ui.sharedControlsFrame.setVisible(processing_page)
        self.ui.processingStatusFrame.setVisible(processing_page)
        self.fit_page_height()
        self.window.contentScrollArea.verticalScrollBar().setValue(0)

    # Give the review tables consistent headings, selection and row styling.
    @staticmethod
    def table(table):
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table._table_sizing = TableSizing(table)

    @staticmethod
    def equal_columns(table):
        """Start with equal widths, then permit manual resizing."""
        table._table_sizing.equal = True
        table._table_sizing.schedule()

    @staticmethod
    @contextmanager
    def table_batch(table):
        """Repaint once, keeping any user-adjusted column widths."""
        enabled = table.updatesEnabled()
        table.setUpdatesEnabled(False)
        try:
            with QSignalBlocker(table):
                yield
        finally:
            table.setUpdatesEnabled(enabled)
            table._table_sizing.schedule()

    # Fill the table in one batch to avoid a repaint for every cell.
    @staticmethod
    def populate(table, rows):
        with ProcessingWorkspace.table_batch(table):
            table.setRowCount(len(rows))
            for i,row in enumerate(rows):
                ProcessingWorkspace.update_cells(table,i,[str(value) for value in row])

    # Return selected rows in screen order, once each.
    @staticmethod
    def selected(table):
        return sorted({x.row() for x in table.selectionModel().selectedRows()})

    # This action needs exactly one row. Report an empty or multiple selection to the caller.
    def index(self, table):
        rows = self.selected(table)
        if len(rows) != 1:
            raise ValueError('Select one row for this action.')
        return rows[0]

    # Read the editor name before recording an edit.
    def reviewer(self):
        name = self.ui.reviewer.text().strip()
        if not name:
            raise ValueError('Enter your reviewer name first.')
        return name

    # Keep conflicting jobs from running together and show the user why an action failed.
    def guard(self, action):
        if self.task is not None or self.window.worker is not None:
            self.status('Wait for the running operation to finish.')
            return
        try:
            action()
        except Exception as exc:
            self.task_error(*exception_details(exc), operation='Requested action')

    # Put the latest message where the reviewer can see it.
    def status(self, text):
        self.ui.processingStatus.setText(text)
        self.window.statusbar.showMessage(text)

    # Give the reviewer a chance to keep unsaved work before replacing the session.
    def confirm_discard(self):
        return not self.dirty or QMessageBox.question(self.window, 'Unsaved session',
            'Continue and discard unsaved changes? Save the session first to keep them.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    # Send slow work to a thread and keep its completion callback for the GUI.
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
        self.task_description = description
        self.task = Task(action, self)
        self.finished_callback = done
        self.task.message.connect(self.status)
        self.task.finished.connect(self.task_finished)
        self.task.finished.connect(self.task.deleteLater)
        self.task.start()

    # Give the finished result to the callback that requested it.
    def task_result(self, value):
        try:
            self.finished_callback(value)
        except Exception as exc:
            self.task_error(*exception_details(exc))

    # Show the error and stop the automatic sequence at the stage that failed.
    def task_error(self, text, trace='', *, operation=None):
        stage = self.auto_stage
        self.auto_stage = None
        operation = operation or (getattr(self, 'task_description', None) if self.task is not None else None) or stage or 'Processing'
        text = str(text).strip() or 'Unknown processing error. See the diagnostic log.'
        try:
            directory = self.source_folder() / 'Review and export' if self.audio else None
        except (OSError, AttributeError, TypeError):
            directory = None
        path = save_failure_log(text, trace, directory, operation)
        self.status('Failed: ' + text)
        self.window.append_log(f'{operation}: {text}')
        if path:
            self.window.append_log(f'Error log: {path}')
        message = f'{operation}\n\n{text}'
        if path:
            message += f'\n\nFull error log: {path}'
        if trace:
            self.window.append_log(trace)
        QMessageBox.warning(self.window, 'Processing failed', message)

    # Release the finished task so the next queued stage can start.
    def task_finished(self):
        # Keep the project locked until any result dialogs and UI updates finish.
        if self.task.failure is None:
            self.task_result(self.task.value)
        else:
            self.task_error(self.task.failure, self.task.failure_traceback)
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

    # Get the editor name before the automatic sequence starts making review entries.
    def start_auto_process(self):
        video_text = self.window.videoPathInput.text().strip()
        video = Path(video_text).resolve() if video_text else None
        if video and not video.is_file():
            raise ValueError('The selected recording could not be found. Choose the file again.')
        download = not video and self.audio is None and bool(self.window.urlInput.text().strip())
        if download and not self.window.outputFolderInput.text().strip():
            raise ValueError('Choose a destination folder for the video first.')
        if not video and not download:
            self.require_audio()
        if self.auto_stage or self.task is not None or self.window.worker is not None:
            raise ValueError('Wait for the current operation to finish.')
        if not self.confirm_discard():
            return
        editor, accepted = QInputDialog.getText(self.window, 'Auto process', 'Editor name:', text=self.ui.reviewer.text().strip())
        if not accepted:
            return
        editor = editor.strip()
        if not editor:
            raise ValueError('Enter an editor name to start Auto process.')
        self.ui.reviewer.setText(editor)
        self.asr.stop_event.clear()
        try:
            if download:
                self.auto_stage = 'download'
                self.window.start_operation('download', automatic=True)
            elif video and not (self.audio and self.audio.is_file() and
                                self.window.extracted_pair == (video, video.stat().st_mtime_ns, self.audio)):
                self.auto_stage = 'extract'
                self.window.start_operation('extract', automatic=True)
            else:
                self.auto_stage = 'label'
                self.analyze(confirmed=True, approval_editor=editor)
        except Exception:
            self.auto_stage = None
            raise

    # Start the next stage only after the previous one has delivered its data.
    def advance_auto_process(self):
        try:
            if self.auto_stage == 'download':
                self.auto_stage = 'extract'
                self.window.start_operation('extract', automatic=True)
            elif self.auto_stage == 'extract':
                self.auto_stage = 'label'
                self.analyze(confirmed=True, approval_editor=self.reviewer())
            elif self.auto_stage == 'label':
                self.auto_stage = 'segments'
                self.generate_segments(output_folder=self.source_folder(), confirmed=True)
            elif self.auto_stage == 'segments':
                self.session['analyze_pending_candidates'] = True
                self.auto_stage = 'snr'
                self.calculate_snr()
            elif self.auto_stage == 'snr':
                self.auto_stage = 'save_snr'
                self.save_snr(automatic=True)
            elif self.auto_stage == 'save_snr':
                self.auto_stage = 'sync'
                self.asr.sync_current()
            elif self.auto_stage == 'sync':
                self.auto_stage = 'asr'
                from corpus_review_and_export import can_transcribe
                if any(can_transcribe(record) for record in self.asr.records):
                    self.asr.transcribe(False)
                if self.task is None:
                    self.advance_auto_process()
            elif self.auto_stage == 'asr':
                errors = any(r.get('asr_errors') for r in self.asr.records)
                if self.asr.stop_event.is_set() or errors:
                    self.auto_stage = None
                    self.window.navigationList.setCurrentRow(4)
                    self.status('Transcription stopped or had errors. Check Transcription and Activity before continuing. Automatic export was not started.')
                else:
                    self.auto_stage = 'export'
                    self.asr.export_automatic()
            elif self.auto_stage == 'export':
                # The export worker has successfully saved the corpus and playable HTML.
                self.auto_stage = None
                self.window.navigationList.setCurrentRow(5)
            elif self.auto_stage is not None:
                raise ValueError('Unknown automatic processing stage.')
        except Exception as error:
            self.task_error(*exception_details(error))

    def set_shared_controls_enabled(self, enabled):
        """Support Designer rows with or without intermediate QWidget containers."""
        # Walk child layouts too so nested controls follow the same enabled state.
        def visit(layout):
            for index in range(layout.count()):
                item = layout.itemAt(index)
                if item.widget() is not None:
                    item.widget().setEnabled(enabled)
                elif item.layout() is not None:
                    visit(item.layout())
        visit(self.ui.sharedSourceRowLayout)
        visit(self.ui.sharedReviewerRowLayout)

    # Catch a missing or moved WAV before starting work on it.
    def require_audio(self):
        if self.audio is None or not self.audio.is_file():
            raise ValueError('Choose an MP3 or WAV, or extract audio on Source & Audio first.')

    # Open a ready-to-use WAV directly so saved reviews still point to the same file.
    def choose_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window, 'Choose an audio recording', '', 'Audio files (*.mp3 *.wav)')
        if not path:
            return
        if Path(path).suffix.lower() == '.wav':
            try:
                review.inspect_wav(path)
            except (ValueError, wave.Error, EOFError):
                pass  # FFmpeg can read other WAV encodings, including floating-point PCM.
            else:
                if self.set_audio(path):
                    self.window.videoPathInput.clear()
                return
        self.window.start_operation('extract', audio_source=path)

    # Switch to the chosen recording and clear results tied to the previous one.
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
        self.region_overlay = None
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title('Click Show spectrogram to display this recording')
        self.canvas.draw_idle()
        self.dirty = False
        self.refresh_all()
        self.status(f'Ready: {path.name} · {self.duration:.2f} seconds')
        return True

    # Prepare the spectral data for the WAV currently open.
    def show_spectrogram(self):
        self.require_audio()
        path = str(self.audio)
        self.start('Building spectrogram…', lambda log: calculate_spectrogram(path), self.plot_spectrogram)

    # Draw the power values against time and frequency.
    def plot_spectrogram(self, data):
        self.plot_data = data
        times, frequencies, db, rate, duration, stride = data
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        image = self.axes.pcolormesh(times, frequencies, db, shading='auto', cmap='magma', vmin=-80, vmax=0)
        self.figure.colorbar(image, ax=self.axes, label='Power relative to peak (dB)')
        self.axes.set(xlabel='Time (s)', ylabel='Frequency (Hz)', xlim=(0, duration), ylim=(0, rate/2))
        self.region_overlay = None
        self.redraw_regions()
        self.status('Spectrogram ready. Zoom with the toolbar; drag a shared boundary to adjust it.')

    # Record who bulk-approved the flagged regions and when.
    @staticmethod
    def approve_flagged_regions(regions, changes, editor):
        stamp = review.utc_now()
        approved = 0
        for index, region in enumerate(regions):
            if not region.get('needs_review'):
                continue
            region.update(needs_review=False, auto_approved_by=editor)
            changes.append(dict(time_utc=stamp, annotator=editor, region_index=index,
                                     action='auto_approve_region', start=region['start'],
                                     end=region['end'], label=region['label']))
            approved += 1
        return approved

    # Use the settings visible when the user starts labeling.
    def analyze(self, confirmed=False, approval_editor=None):
        self.require_audio()
        window_seconds, hop_seconds = analysis_settings(
            self.ui.pannsWindowInput.text(), self.ui.pannsHopInput.text())
        if not confirmed and not self.confirm_discard():
            return
        path = str(self.audio)
        verify_vocals = self.ui.improveSinging.isChecked()
        duration,folder=self.duration,self.source_folder()
        # Run the classifier and prepare the plot and report away from the GUI thread.
        def action(log):
            regions = classify_file(path, log, verify_vocals=verify_vocals,
                                    window_seconds=window_seconds, hop_seconds=hop_seconds)
            validate_regions(regions,duration)
            changes=[]
            count=self.approve_flagged_regions(regions,changes,approval_editor) if approval_editor else 0
            plot=calculate_spectrogram(path)
            report=self.write_region_report(Path(path),duration,regions,changes,folder)
            return regions,changes,count,plot,report
        # Bring the completed regions and their review history back into the editor.
        def done(result):
            self.regions,self.changes,count,plot,report=result
            regions=self.regions
            if approval_editor:
                self.window.append_log(f'Part 3: {count} flagged regions auto-approved by {approval_editor}.')
            self.session = None
            self.clear_snr()
            self.dirty = True
            self.refresh_all()
            self.plot_spectrogram(plot)
            self.window.append_log(f'Region report saved: {report}')
            if approval_editor:
                self.status(f'Part 3 complete. {count} regions auto-approved by {approval_editor}.')
            else:
                self.status(f'{len(regions)} regions labeled. Review flagged and mixed regions before cutting.')
        self.start('Loading PANNs; first run downloads its model…', action, done)

    # Move the colored bands to the current region edges.
    def redraw_regions(self):
        # A collection paints all region bands in a single Matplotlib draw.
        vertices = [[(r['start'],0), (r['end'],0), (r['end'],1), (r['start'],1)] for r in self.regions]
        colors = [COLORS[r['label']] for r in self.regions]
        if self.region_overlay is None:
            self.region_overlay = PolyCollection(vertices, facecolors=colors, edgecolors='none',
                alpha=.23, transform=self.axes.get_xaxis_transform(), zorder=1.1)
            self.axes.add_collection(self.region_overlay, autolim=False)
            if self.plot_data is None and self.regions:
                self.axes.set_xlim(self.regions[0]['start'],self.regions[-1]['end'])
        else:
            self.region_overlay.set_verts(vertices)
            self.region_overlay.set_facecolors(colors)
        self.canvas.draw_idle()

    # Show the current and original labels alongside the model's score.
    def region_rows(self, first, count):
        table = self.ui.regionTable
        for row in range(first,first+count):
            r = self.regions[row]
            self.update_cells(table,row,[f"{r['start']:.4f}", f"{r['end']:.4f}", r['label'], r.get('auto_label',''),
                confidence_text(r)])
            # Use a dash for a missing score so it won't look like zero confidence.
            def score(key):
                return '—' if r.get(key) is None else f'{r[key]:.3f}'
            details = (f'Speech: {score("speech_score")} | Music: {score("music_score")} | '
                       f'Vocals: {score("vocal_score")} ({r.get("vocal_evidence_class") or "not recorded"})'
                       + f' | Separated voice evidence: {score("verified_voice_score")}'
                       + '\n' + str(r.get('note') or 'Automatic scores are uncalibrated.'))
            for col in range(table.columnCount()):
                table.item(row,col).setToolTip(details)

    # Region boundaries changed, so any segments cut from the old timeline are stale.
    def region_edited(self, first, removed, added, selected):
        self.invalidate_segments()
        self.dirty = True
        table = self.ui.regionTable
        with QSignalBlocker(table):
            for _ in range(removed-added):
                table.removeRow(first+added)
            for offset in range(removed,added):
                table.insertRow(first+offset)
            self.region_rows(first,added)
            table.selectRow(selected)
        self.region_selected()
        self.redraw_regions()

    # Bring the review tables back in sync with the session.
    def refresh_all(self):
        with self.table_batch(self.ui.regionTable):
            self.ui.regionTable.setRowCount(len(self.regions))
            self.region_rows(0,len(self.regions))
        segments = self.session['segments'] if self.session else []
        self.populate(self.ui.segmentTable, [self.segment_cells(item) for item in segments])
        self.visible_spans = review.visible_unresolved(self.session) if self.session else []
        self.populate(self.ui.unresolvedTable, [self.span_cells(span) for span in self.visible_spans])
        self.ui.unresolvedTable.clearSelection()
        self.ui.toggleSpanAsr.setText('Skip transcription')
        self.refresh_snr()
        self.configure_region_preview()
        for kind in self.part4_previews:
            self.configure_part4_preview(kind)

    # Show the selected region's label and limits in the editor.
    def region_selected(self):
        rows = self.selected(self.ui.regionTable)
        if rows:
            r = self.regions[rows[0]]
            self.ui.regionLabel.setCurrentText(r['label'])
            self.ui.regionStart.setValue(r['start'])
            self.ui.regionEnd.setValue(r['end'])
        self.configure_region_preview(reset=True)

    # Turn the player's millisecond position into a clock the listener can read.
    @staticmethod
    def clock(milliseconds):
        seconds, ms = divmod(max(0, int(milliseconds)), 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return (f'{hours:02d}:' if hours else '') + f'{minutes:02d}:{seconds:02d}.{ms:03d}'

    # Keep the listening slider inside the selected region.
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

    # Show where playback will begin if the user presses Play from cursor.
    def update_region_cursor_label(self):
        self.ui.regionCursorLabel.setText('Position ' + self.clock(self.ui.regionSeekSlider.value()))

    # Show how much source audio the two time fields currently select.
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

    # Keep the position label and active playback in step with the slider.
    def region_slider_changed(self, value):
        self.update_region_cursor_label()
        if not self.ui.regionSeekSlider.isSliderDown():
            self.guard(self.seek_region_preview)

    # Find the part of the selected region still ahead of the cursor.
    def region_cursor_interval(self):
        self.require_audio()
        region = self.regions[self.index(self.ui.regionTable)]
        cursor = self.ui.regionSeekSlider.value() / 1000
        a, b = self.entered_region_range(region)
        end = b if region['start'] <= a <= cursor < b <= region['end'] else region['end']
        return cursor, end

    # Move the current preview to the position the listener picked.
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

    # Use the listening cursor for the selected start or end field.
    def mark_region_time(self, end):
        self.require_audio()
        self.index(self.ui.regionTable)
        widget = self.ui.regionEnd if end else self.ui.regionStart
        widget.setValue(self.ui.regionSeekSlider.value() / 1000)
        self.update_region_range_summary()

    # Play from the cursor to the end of the selected listening range.
    def play_region_cursor(self):
        cursor, end = self.region_cursor_interval()
        if cursor >= end:
            raise ValueError('Move the playback slider before the end of the region.')
        self.play(self.audio, cursor, end)

    # Pause the active region preview or continue from the same place.
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

    # Explain which part of the timeline the selected correction mode will change.
    def region_mode_changed(self, index):
        self.ui.regionRangeHint.setText(
            'Enter source start/end times for the section you want to relabel. The surrounding audio keeps its current label.'
            if index == 0 else
            'Edit the entire selected region. Moving a shared boundary also adjusts the neighboring region.'
        )

    # Check the user's start and end times before editing or playing the range.
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

    # Clear quality scores measured on a previous set of clips.
    def clear_snr(self):
        if self.snr_results or self.snr_snapshot is not None:
            self.snr_view_dirty = True
        self.snr_results = []
        self.snr_snapshot = None
        self.snr_report = None
        if self.snr_view_dirty and self.ui.pages.currentIndex() == 3:
            self.refresh_snr()

    # A new region edit can change the audio used for segmentation, SNR and ASR.
    def invalidate_segments(self):
        if hasattr(self, 'asr'):
            self.asr.source_changed()
        if self.session is not None or self.ui.segmentTable.rowCount() or self.ui.unresolvedTable.rowCount():
            self.session = None
            self.visible_spans = []
            for table in (self.ui.segmentTable,self.ui.unresolvedTable):
                with QSignalBlocker(table):
                    table.setRowCount(0)
            for kind in self.part4_previews:
                self.configure_part4_preview(kind)
        self.clear_snr()

    # Apply the selected edit and keep its details in the region history.
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
        self.region_edited(max(0,i-1), len(before), len(before), i)
        self.status('Region corrected. Earlier segment/SNR results cleared; regenerate from these labels.')

    # Route the correction to the handler for the selected editing mode.
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
        self.region_edited(i, 1, len(pieces), i+selected)
        self.status('Selected range relabeled; surrounding regions preserved. Regenerate segments and SNR before continuing to ASR.')

    # Let the reviewer hear the exact time range before applying a change.
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

    # Make two regions by cutting the selected one at its midpoint.
    def split_region(self):
        i = self.index(self.ui.regionTable);name = self.reviewer()
        old = copy.deepcopy(self.regions[i]);mid = round((old['start']+old['end'])*8000)/16000
        if not old['start'] < mid < old['end']: raise ValueError('This region is too short to split.')
        left,right = copy.deepcopy(old),copy.deepcopy(old)
        left['end'],right['start'] = mid,mid
        for x in (left,right): x.update(manual_override=True,needs_review=True)
        self.regions[i:i+1] = [left,right]
        self.changes.append(dict(time_utc=review.utc_now(),annotator=name,region_index=i,action='split',before=old,after=[left.copy(),right.copy()]))
        self.region_edited(i, 1, 2, i)

    # Join the selected region to the one immediately after it.
    def merge_region(self):
        i = self.index(self.ui.regionTable);name = self.reviewer()
        if i+1>=len(self.regions): raise ValueError('There is no following region to merge.')
        before=copy.deepcopy(self.regions[i:i+2]);merged=copy.deepcopy(before[0])
        merged.update(end=before[1]['end'],label=self.ui.regionLabel.currentText(),manual_override=True,needs_review=True,
                      confidence=None,auto_confidence=None,note='Merged manually; original scores retained in correction history.')
        self.regions[i:i+2]=[merged]
        self.changes.append(dict(time_utc=review.utc_now(),annotator=name,region_index=i,action='merge',before=before,after=merged.copy()))
        self.region_edited(i, 2, 1, i)

    # Remember which shared boundary the user grabbed on the plot.
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

    # Apply the new shared boundary when the user releases the mouse.
    def plot_release(self, event):
        i,self.drag_boundary=self.drag_boundary,None
        if i is not None and event.inaxes == self.axes and event.xdata is not None:
            self.ui.regionTable.selectRow(i)
            self.ui.regionEditMode.setCurrentIndex(1)
            self.ui.regionEnd.setValue(event.xdata)
            self.guard(self.edit_region)

    # Find the output folder for the project that owns this WAV.
    def source_folder(self):
        return project_folder(self.audio.parent)

    # Put this version of the region report in its own dated folder.
    @staticmethod
    def write_region_report(audio,duration,regions,changes,destination):
        folder = new_run_folder(step_folder(destination, 'Labeled audio regions'))
        path = folder / (audio.stem + '_regions.html')
        payload=dict(version=1,audio_name=audio.name,audio_size=audio.stat().st_size,duration_seconds=duration,
                    method='PANNs adjustable windows/hops and RMS silence threshold; optional Hybrid Demucs vocal verification; input bounds and rules recorded per prediction',
                    analysis_settings=recorded_analysis_settings(regions),
                    score_note='Uncalibrated automatic scores; manual boundaries may differ from inference windows. Vocal-music proposals are mixed under corpus policy and require listening review; context-based evidence may blur boundaries and does not confirm lyrics.',
                    regions=regions,changes=changes)
        save_report(path,payload)
        return path

    # Take a snapshot of the timeline and edits before handing them to the save worker.
    def save_regions(self):
        self.require_audio()
        folder = QFileDialog.getExistingDirectory(self.window, 'Choose folder for region report', str(self.source_folder()))
        if not folder:
            return
        audio,duration,regions,changes=self.audio,self.duration,self.regions,self.changes
        # Check the captured timeline before it goes into the report.
        def write(log):
            validate_regions(regions,duration)
            return self.write_region_report(audio,duration,regions,changes,folder)
        self.start('Saving region report…',write,lambda path:self.status('Region report saved: '+str(path)))

    # Choose a saved region review to reopen against this WAV.
    def load_regions(self):
        self.require_audio()
        path,_=QFileDialog.getOpenFileName(self.window,'Open region review','','Region report (*.html *.htm *.json)')
        if not path:return
        if not self.confirm_discard():return
        audio=self.audio
        build_plot=self.plot_data is None
        # Read the labels and prepare the audio plot together for the restored view.
        def read(log):
            data=load_review(path)
            self.check_region_payload(data)
            plot=calculate_spectrogram(str(audio)) if build_plot and data['regions'] else None
            return data,plot
        self.start('Opening region review…',read,self.restore_regions)

    # Display the saved labels over the recording and restore the edit history.
    def restore_regions(self,result):
        data,plot=result
        self.regions=data['regions'];self.changes=data.get('changes',[])
        config=recorded_analysis_settings(self.regions)
        if config:
            self.ui.pannsWindowInput.setText(f'{config["window_seconds"]:g}')
            self.ui.pannsHopInput.setText(f'{config["hop_seconds"]:g}')
        self.invalidate_segments();self.dirty=True;self.refresh_all()
        if plot is None:self.redraw_regions()
        else:self.plot_spectrogram(plot)
        self.status('Region review loaded.')

    # Check the saved audio name, size and duration before placing labels on this recording.
    def check_region_payload(self,data):
        if data['audio_name']!=self.audio.name or data['audio_size']!=self.audio.stat().st_size or abs(data['duration_seconds']-self.duration)>1/16000:
            raise ValueError('Region report does not match the selected audio.')
        validate_regions(data['regions'],self.duration)

    # Generate candidate clips from the current region timeline.
    def generate_segments(self, output_folder=None, confirmed=False):
        self.require_audio();validate_regions(self.regions,self.duration)
        if self.session and not confirmed and not self.confirm_discard():return
        folder = str(output_folder) if output_folder is not None else QFileDialog.getExistingDirectory(self.window,'Choose output folder',str(self.source_folder()))
        if not folder:return
        folder = str(step_folder(folder, 'Segmenting the speech'))
        regions=copy.deepcopy(self.regions);path=str(self.audio)
        # Write the candidate set and load its review session for the segment page.
        def action(log):
            from pause_based_segmentation import export_candidates
            result=export_candidates(path,regions,folder,log)
            return review.load_session(Path(result['output_dir'])/'step4_segments.json',path)
        self.start('Finding speech and natural pause candidates…',action,self.set_session)

    # Stop the old preview before showing a different set of candidates.
    def set_session(self,data):
        self.stop_playback()
        if hasattr(self,'asr'):
            self.asr.source_changed()
        self.session=data;self.clear_snr();self.dirty=True;self.refresh_all()
        self.status(f"{len(data['segments'])} candidates · {len(data['unresolved'])} unresolved spans/gaps. Listen before approving.")
        self.window.navigationList.setCurrentRow(2)

    # Open a previously generated candidate set for this recording.
    def load_candidates(self):
        self.require_audio()
        path,_=QFileDialog.getOpenFileName(self.window,'Open candidates or review_session.json','','Step 4 JSON (*.json)')
        if not path:return
        if self.session and not self.confirm_discard():return
        audio=self.audio
        self.start('Opening segment candidates…',lambda log:review.load_session(path,audio),self.set_session)

    # Load the highlighted candidate into the segment editor.
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

    # Use the time limits appropriate to source editing or clip-local trimming.
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
        if self.audio is None or not self.session or not rows or (kind == 'segment' and len(rows) != 1):
            return None
        if kind == 'segment':
            items = self.session['segments']
            if rows[0] >= len(items):
                return None
            item = items[rows[0]]
        else:
            if rows[0] >= len(self.visible_spans):
                return None
            _, item, _, _ = self.visible_spans[rows[0]]
            visible_start = min(self.visible_spans[r][2] for r in rows)
            visible_end = max(self.visible_spans[r][3] for r in rows)
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

    # Set the listening range for the chosen candidate or unresolved interval.
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

    # Show the cursor, selected limits and listening duration together.
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

    # Update the display and move playback when the Part 4 slider changes.
    def part4_slider_changed(self, kind):
        self.update_part4_summary(kind)
        if not getattr(self.ui, kind + 'SeekSlider').isSliderDown():
            self.guard(lambda: self.seek_part4(kind))

    # Translate the current preview selection into a filename and playback times.
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

    # Seek only if this slider owns the active preview.
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

    # Copy the cursor to the chosen boundary field so the reviewer can mark a cut by ear.
    def mark_part4(self, kind, end):
        if self.part4_previews[kind] is None:
            raise ValueError('Select one item in this panel first.')
        getattr(self.ui, kind + ('End' if end else 'Start')).setValue(
            getattr(self.ui, kind + 'SeekSlider').value() / 1000)

    # Play the rest of the selected preview from its cursor.
    def play_part4_cursor(self, kind):
        path, a, b = self.part4_interval(kind, cursor=True)
        self.play(path, a, b, owner=kind)

    # Pause or continue the Part 4 preview that owns the player.
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

    # Follow playback on the slider belonging to the current clip or span.
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

    # Build the cells for one candidate using its source time and review status.
    @staticmethod
    def segment_cells(item):
        if item['source_mode'] == 'continuous_source':
            duration = item['source_end'] - item['source_start']
        else:
            start, end = review.candidate_trim_bounds(item)
            duration = end - start
        original = item.get('original') or {}
        return [item['id'], f"{item['source_start']:.4f}", f"{item['source_end']:.4f}",
                f'{duration:.4f}', item['label'], item['status'],
                original.get('kind','vad_candidate') + ' · ' + '; '.join(original.get('notes',[]))]

    # Build the cells for the uncovered part of a span that is still visible.
    def span_cells(self, span):
        _, item, start, end = span
        kept = list(review.transcription_ranges(self.session, item, start, end))
        setting = 'Skipped' if not kept else ('Default' if kept == [(start,end)] else 'Partly skipped')
        return [item['id'], f'{start:.4f}', f'{end:.4f}', item['label'], item['status'], setting,
                item['original'].get('reason','')]

    # Update a single row without rebuilding the whole table.
    @staticmethod
    def update_cells(table, row, values):
        for column, value in enumerate(values):
            cell = table.item(row, column)
            if cell is None:
                table.setItem(row, column, QTableWidgetItem(value))
            elif cell.text() != value:
                cell.setText(value)

    def update_visible_spans(self, focus=None):
        """Apply changed ranges only. Keep untouched Qt items and their selection."""
        table = self.ui.unresolvedTable
        spans = review.visible_unresolved(self.session)
        # Compare rows by interval and identity so unchanged spans can keep their place.
        def keys(rows):
            return [(i, a, b) for i, _, a, b in rows]
        changes = SequenceMatcher(None, keys(self.visible_spans), keys(spans),
                                  autojunk=False).get_opcodes()
        with QSignalBlocker(table):
            for operation, old, old_end, new, new_end in reversed(changes):
                if operation == 'equal':
                    continue
                common = min(old_end-old, new_end-new)
                for _ in range(old+common, old_end):
                    table.removeRow(old+common)
                for row in range(old+common, old+new_end-new):
                    table.insertRow(row)
                for offset, span in enumerate(spans[new:new_end]):
                    self.update_cells(table, old+offset, self.span_cells(span))
            self.visible_spans = spans
            if focus is not None:
                parent, lo, hi, cut_start, cut_end = focus
                ident = self.session['unresolved'][parent]['id']
                remainders = []
                for row, span in enumerate(spans):
                    i, item, a, b = span
                    if i == parent:
                        # A whole-span label edit can retain exactly the same range.
                        self.update_cells(table, row, self.span_cells(span))
                    if ((i == parent or item.get('split_from') == ident)
                            and lo <= a < b <= hi and (a >= cut_end or b <= cut_start)):
                        remainders.append((a < cut_end, abs(a-cut_end), row))
                table.clearSelection()
                if remainders:
                    table.selectRow(min(remainders)[2])
        if focus is not None or self.part4_spec('unresolved') != self.part4_previews['unresolved']:
            self.unresolved_selected()

    # Flag the session as edited and update the rows affected by the change.
    def modified_session(self, segment_rows=(), *, spans_changed=False, focus=None):
        if hasattr(self, 'asr'):
            self.asr.source_changed()
        # Invalidate the data now. Draw the empty histogram when its page opens.
        if self.snr_results or self.snr_snapshot is not None:
            self.snr_view_dirty = True
        self.snr_results = []
        self.snr_snapshot = self.snr_report = None
        if self.snr_view_dirty and self.ui.pages.currentIndex() == 3:
            self.refresh_snr()
        self.dirty = True
        table = self.ui.segmentTable
        with QSignalBlocker(table):
            if table.rowCount() != len(self.session['segments']):
                table.setRowCount(len(self.session['segments']))
            for row in segment_rows:
                self.update_cells(table, row, self.segment_cells(self.session['segments'][row]))
        if spans_changed:
            self.update_visible_spans(focus)
        self.status('Part 4 changed. Recalculate SNR to update the resolved clips and unresolved spans.')

    # Apply the candidate's label and time fields in the selected editing mode.
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
        self.modified_session([i], spans_changed=bool(self.session['segments'][i].get('parent_unresolved_id')))
        self.segment_selected()

    # Record approval for the candidates the reviewer highlighted.
    def approve(self):
        if not self.session:raise ValueError('Generate or open candidates first.')
        rows=self.selected(self.ui.segmentTable)
        if len(rows)>1 and any((self.session['segments'][i].get('original') or {}).get('manual_boundary_review_required') for i in rows):
            raise ValueError('Listen to each boundary-review candidate and approve it individually.')
        if len(rows)==1 and (self.session['segments'][rows[0]].get('original') or {}).get('manual_boundary_review_required'):
            answer=QMessageBox.question(self.window,'Confirm clip boundaries',
                'Have you listened to this clip and confirmed complete words and 0.1–0.2 seconds of natural pause at BOTH ends? Adjust the source bounds first if needed.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
            if answer!=QMessageBox.StandardButton.Yes:return
        count=review.approve_segments(self.session,rows,self.reviewer())
        self.modified_session(rows);self.status(f'{count} clips approved. Export or calculate SNR when ready.')

    # Approve all duration-valid pending candidates, including boundary-review candidates.
    def approve_all(self):
        if not self.session:
            raise ValueError('Generate or open candidates first.')
        rows, skipped = review.batch_approval_plan(self.session)
        if not rows:
            self.status(f'No eligible pending 5–15 second candidates. {len(skipped)} invalid candidates left pending.')
            return
        flagged = sum(bool((self.session['segments'][i].get('original') or {}).get('manual_boundary_review_required')) for i in rows)
        if flagged:
            answer = QMessageBox.question(self.window, 'Confirm batch boundary approval',
                f'Approve all {len(rows)} eligible 5–15 second candidates, including {flagged} with unverified natural pauses? '
                'Duration alone does not prove a safe cut. By selecting Yes, you confirm that the clips have complete words '
                'and about 0.1–0.2 seconds of natural pause at both edges. '
                'Invalid or out-of-range candidates and unresolved spans will remain pending.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        count = review.approve_segments(self.session, rows, self.reviewer(), approval_method='batch_approve_all')
        self.modified_session(rows)
        self.status(f'{count} candidates approved in one batch, including {flagged} boundary-review candidates. '
                    f'{len(skipped)} invalid or out-of-range candidates left pending.')

    # Set whether this unresolved interval goes on to recognition.
    def toggle_span_asr(self):
        row = self.index(self.ui.unresolvedTable)
        span = self.visible_spans[row]
        _,item,start,end = span
        skipped = review.toggle_span_transcription(self.session,item,start,end,self.ui.reviewer.text())
        self.update_cells(self.ui.unresolvedTable,row,self.span_cells(span))
        self.ui.toggleSpanAsr.setText('Include in transcription' if skipped else 'Skip transcription')
        self.dirty = True
        self.asr.source_changed()
        self.status('Transcription preference updated for this range. Reload clips on the Transcription page.')

    # Record which selected candidates the reviewer wants to leave out.
    def exclude(self):
        rows=self.selected(self.ui.segmentTable);name=self.reviewer()
        if not rows:raise ValueError('Select one or more clips.')
        for i in rows:review.exclude_segment(self.session,i,name)
        self.modified_session(rows, spans_changed=any(self.session['segments'][i].get('parent_unresolved_id') for i in rows))

    # Resolve conflicts where approved clips occupy the same source time.
    def resolve_overlaps(self):
        if not self.session:raise ValueError('Open candidates first.')
        result=review.resolve_approved_overlaps(self.session,self.reviewer())
        self.modified_session(range(len(self.session['segments'])), spans_changed=True)
        self.segment_selected()
        self.status(str(result))

    # Show the selected uncovered interval in the listening and editing controls.
    def unresolved_selected(self):
        rows=self.selected(self.ui.unresolvedTable)
        self.ui.toggleSpanAsr.setText('Skip transcription')
        if not rows or not self.session:
            self.configure_part4_preview('unresolved')
            return
        _, item, a, b = self.visible_spans[rows[0]]
        a = min(self.visible_spans[r][2] for r in rows)
        b = max(self.visible_spans[r][3] for r in rows)
        if len(rows)==1 and not list(review.transcription_ranges(self.session,item,a,b)):
            self.ui.toggleSpanAsr.setText('Include in transcription')
        self.ui.unresolvedStart.setValue(a);self.ui.unresolvedEnd.setValue(b)
        if item['label'] in LABELS:self.ui.unresolvedLabel.setCurrentText(item['label'])
        self.configure_part4_preview('unresolved', reset=True)

    # Turn the marked part of an unresolved interval into a candidate.
    def add_manual(self):
        rows = self.selected(self.ui.unresolvedTable)
        if len(rows)>1:
            spans = [self.visible_spans[r] for r in rows]
            index = review.add_combined_clip(self.session, spans, self.ui.unresolvedStart.value(),
                        self.ui.unresolvedEnd.value(),self.ui.unresolvedLabel.currentText(),self.reviewer())
            self.modified_session([index], spans_changed=True)
            self.ui.segmentTable.selectRow(index)
            self.status('Combined candidate added. Listen to both ends and approve it when ready.')
            return
        row = self.index(self.ui.unresolvedTable)
        i, _, a, b = self.visible_spans[row]
        if not a - 1/16000 <= self.ui.unresolvedStart.value() < self.ui.unresolvedEnd.value() <= b + 1/16000:
            raise ValueError('Select a range inside the visible unresolved span.')
        start,end=self.ui.unresolvedStart.value(),self.ui.unresolvedEnd.value()
        index=review.add_from_unresolved(self.session,i,start,end,self.ui.unresolvedLabel.currentText(),self.reviewer())
        clip=self.session['segments'][index]
        self.modified_session([index], spans_changed=True, focus=(i,a,b,clip['source_start'],clip['source_end']))

    # Record the chosen label and who reviewed this unresolved range.
    def mark_unresolved(self):
        row = self.index(self.ui.unresolvedTable)
        i, _, a, b = self.visible_spans[row]
        if not a - 1/16000 <= self.ui.unresolvedStart.value() < self.ui.unresolvedEnd.value() <= b + 1/16000:
            raise ValueError('Select a range inside the visible unresolved span.')
        name=self.reviewer()
        review.label_unresolved(self.session,i,self.ui.unresolvedLabel.currentText(),name,self.ui.unresolvedStart.value(),self.ui.unresolvedEnd.value())
        review.mark_unresolved_reviewed(self.session,i,name)
        start,end=review.unresolved_bounds(self.session['unresolved'][i])
        self.modified_session(spans_changed=True, focus=(i,a,b,start,end))

    # Require at least one approved clip and check that approvals don't overlap.
    def require_approved(self):
        self.require_audio()
        if not self.session:raise ValueError('Generate or load candidates and approve clips first.')
        if review.overlapping_approved(self.session):
            raise ValueError('Approved clips overlap. Review the times or use Resolve approved overlaps before exporting or calculating SNR.')
        if not any(s['status']=='approved' for s in self.session['segments']):
            raise ValueError('Approve at least one clip first.')

    # Write the reviewed clips with the information needed to find their original samples.
    def export_segments(self):
        self.require_approved()
        folder=QFileDialog.getExistingDirectory(self.window,'Export approved WAV files',str(self.source_folder()))
        if not folder:return
        folder = str(step_folder(folder, 'Segmenting the speech'))
        snapshot=copy.deepcopy(self.session);path=str(self.audio)
        # Show where the exported WAVs were saved and open the folder.
        def done(result):
            self.status(f"Exported {result['exported']} approved clips to {result['output_dir']}. Pending spans: {result['pending_unresolved']}.")
            self.window.open_path(Path(result['output_dir']))
        self.start('Exporting reviewed clips and source mappings…',lambda log:review.save_and_export(snapshot,path,folder),done)

    # Check that this session has audio that can go through quality analysis.
    def require_analysis_source(self):
        self.require_audio()
        if not self.session:
            raise ValueError('Generate or open Part 4 segments first.')
        if any(item['status']=='approved' for item in self.session['segments']):
            self.require_approved()

    # Run both SNR measurements on the current set of eligible intervals.
    def calculate_snr(self):
        self.require_analysis_source()
        snapshot=copy.deepcopy(self.session);path=str(self.audio);threshold=self.ui.snrThreshold.value()
        previous={r['id']:copy.deepcopy(r) for r in self.snr_results}
        policy=snr.POLICIES[self.ui.snrPolicy.currentIndex()]
        justification=self.ui.snrJustification.text().strip()
        # Build the analysis queue from the captured session and measure each item.
        def action(log):
            records=snr.session_records(snapshot,path)
            return snr.calculate(records,threshold,lambda n,total:log(f'Calculating pause-energy + WADA-SNR: {n} / {total}'),policy=policy,justification=justification)
        # Reapply any valid reviewer corrections before displaying the new measurements.
        def done(records):
            for item in records:snr.preserve_override(item,previous.get(item['id']))
            self.snr_results=records;self.snr_snapshot=snapshot;self.applied_threshold=threshold
            self.snr_policy=policy;self.snr_justification=justification
            self.snr_report=None;self.dirty=True;self.refresh_snr()
            self.status(f'{len(records)} two-method SNR results ready; {sum(bool(r.get("quality_warning")) for r in records)} have estimator notes; correction is optional.')
        self.start('Calculating SNR for approved clips and unresolved spans…',action,done)

    # Reuse the measured scores when the user changes the threshold or decision rule.
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

    # Show the latest quality decisions on the SNR page.
    def refresh_snr(self):
        self.snr_view_dirty = False
        if not hasattr(self,'histogram'):return
        self.populate(self.ui.snrTable,[self.snr_cells(r) for r in self.snr_results])
        self.draw_snr_chart()

    # Lay out the two estimates, decision and review details in one table row.
    @staticmethod
    def snr_cells(r):
        # Show a missing estimate explicitly so it can't be read as 0 dB.
        def number(value):return 'Unavailable' if value is None else f'{value:.2f}'
        bound=(r.get('wada_details') or {}).get('bound')
        wada=number(r.get('wada_snr_db'))
        if bound:wada+=' (range limit)'
        basis={'reviewed_part4_pause_context':'Part 4 pauses','estimated_part4_pause':'Estimated pauses','quiet_frame_fallback':'Quiet-frame fallback','unavailable':'Unavailable'}.get(r.get('noise_source'),'Legacy estimate')
        return [r['id'],f"{r['source_start']:.3f}",f"{r['source_end']:.3f}",r.get('segmentation_status','resolved').title(),r['content_label'],
            number(r.get('energy_snr_db',r.get('snr_db'))),wada,number(r.get('snr_db')),
            r['automatic_quality'],r['quality'],basis,r['note']]

    # Redraw the charts using the measurements currently on screen.
    def draw_snr_chart(self):
        self.histogram.clear();a,b=self.histogram.subplots(1,2)
        for key,label,color in [('energy_snr_db','Pause energy','#157A6E'),('wada_snr_db','WADA-SNR','#7C3AED')]:
            scores=[r[key] for r in self.snr_results if r.get(key) is not None]
            if scores:a.hist(scores,bins=min(20,max(3,len(scores))),color=color,alpha=.6,label=label)
        a.axvline(self.applied_threshold,color='#d59a45',linestyle='--',label=f'{self.applied_threshold:g} dB')
        a.set(xlabel='Estimated SNR (dB)',ylabel='Clips',title='Two-method comparison');a.legend(fontsize=8)
        labels=['clean','noisy','unavailable'];counts=[sum(r['quality']==l for r in self.snr_results) for l in labels]
        self.quality_bars=b.bar(labels,counts,color=['#268b80','#e6ab54','#a1afb4']);b.set(title='Quality',ylabel='Clips')
        self.hist_canvas.draw_idle()

    # Catch results that belong to an earlier version of the clip list.
    def check_snr(self):
        if not self.snr_results or not review.same_snr_source(self.snr_snapshot,self.session):
            raise ValueError('Clips changed or no results are loaded. Calculate SNR again.')

    # Save the reviewer's clean/noisy choice or restore the automatic label.
    def override_quality(self,label):
        self.check_snr();rows=self.selected(self.ui.snrTable);who=self.reviewer();why=self.ui.overrideReason.text().strip()
        if not rows or not why:raise ValueError('Select clips and enter the correction reason.')
        for i in rows:
            snr.set_quality(self.snr_results[i],label,who,why)
        for i in rows:
            self.update_cells(self.ui.snrTable,i,self.snr_cells(self.snr_results[i]))
        counts=[sum(r['quality']==label for r in self.snr_results) for label in ('clean','noisy','unavailable')]
        for bar,count in zip(self.quality_bars,counts):
            bar.set_height(count)
        axis=self.quality_bars[0].axes
        axis.relim();axis.autoscale_view()
        self.hist_canvas.draw_idle()
        self.dirty=True;self.status('Quality corrections recorded. Save the session or updated report.')

    # Save the SNR results together with the reason for the selected threshold.
    def save_snr(self, *, automatic=False):
        self.check_snr()
        self.snr_controls_changed()
        folder = self.source_folder()
        if not automatic:
            folder = QFileDialog.getExistingDirectory(self.window, 'Save SNR HTML / CSV / JSON', str(folder))
            if not folder:
                return
        folder = str(step_folder(folder, 'Audio quality SNR'))
        records,threshold,policy,reason=self.snr_results,self.applied_threshold,self.snr_policy,self.snr_justification
        # Remember which HTML report the Open report button should use.
        def done(paths):
            self.snr_report=Path(paths[0])
            self.window.append_log(f'Saved SNR reports: {self.snr_report}')
            self.status(f'Saved SNR reports: {self.snr_report}')
        self.start('Saving SNR reports…', lambda log: snr.export_report(
            records, threshold, folder, policy=policy, justification=reason, provisional=automatic), done)

    # Open the saved quality report for this session.
    def open_snr_report(self):
        if not self.snr_report:raise ValueError('Save the SNR report first.')
        self.window.open_path(self.snr_report)

    # Ask which saved quality results to reload for this audio.
    def load_snr(self):
        self.require_analysis_source()
        path,_=QFileDialog.getOpenFileName(self.window,'Open SNR results','','SNR JSON (*.json)')
        if not path:return
        session,audio=self.session,self.audio
        # Read the scores and apply the decision settings stored with them.
        def read(log):
            payload=json.loads(Path(path).read_text(encoding='utf-8-sig'))
            if payload.get('format') not in ('speech_corpus_snr_v1','speech_corpus_snr_v2'):raise ValueError('Not a supported SNR report.')
            records=self.validate_saved_snr(payload['segments'],session,audio)
            threshold=float(payload['threshold_db'])
            if not math.isfinite(threshold) or not -50<=threshold<=100:raise ValueError('Invalid SNR threshold.')
            policy=payload.get('decision_policy','energy');justification=payload.get('threshold_justification','')
            return snr.reclassify(records,threshold,policy,justification),threshold,policy,justification,copy.deepcopy(session)
        # Put the restored scores and threshold back on the quality page.
        def done(result):
            self.snr_results,self.applied_threshold,self.snr_policy,self.snr_justification,self.snr_snapshot=result
            self.set_snr_controls();self.snr_report=Path(path).with_suffix('.html')
            self.dirty=True;self.refresh_snr();self.status('SNR results loaded.')
        self.start('Opening SNR results…',read,done)

    # Check that the saved results belong to these clips and this recording.
    def match_snr(self,records):
        return self.validate_saved_snr(records,self.session,self.audio)

    # Match saved measurements to their source intervals before reusing them.
    @staticmethod
    def validate_saved_snr(records,session,audio):
        # Earlier reports contain only approved clips. They can still be resumed.
        includes_spans = any(r.get('segmentation_status')=='unresolved' for r in records)
        current=snr.session_records(session,str(audio),include_unresolved=includes_spans)
        records=sorted((r for r in records if not (r.get('segmentation_status')=='unresolved' and
                       review.under_one_second(r['source_start'],r['source_end']))),key=snr.source_time_key)
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

    # Restore the controls together so intermediate values don't trigger recalculation.
    def set_snr_controls(self):
        for widget in (self.ui.snrThreshold,self.ui.snrPolicy,self.ui.snrJustification):widget.blockSignals(True)
        self.ui.snrThreshold.setValue(self.applied_threshold)
        self.ui.snrPolicy.setCurrentIndex(snr.POLICIES.index(self.snr_policy))
        self.ui.snrJustification.setText(self.snr_justification)
        for widget in (self.ui.snrThreshold,self.ui.snrPolicy,self.ui.snrJustification):widget.blockSignals(False)

    # Capture the processing pages and transcript review in the same save.
    def save_project(self):
        self.require_audio()
        with save_dialog_folder(self.source_folder() / 'Sessions') as folder:
            path,_=QFileDialog.getSaveFileName(self.window,'Save complete review session',str(folder / (self.audio.stem+'_session.json')),'Review session (*.json)')
            if not path:return
        payload=self.project_payload(snapshot=False)
        # Create the folder and safely replace the session file with the captured data.
        def write(log):
            Path(path).parent.mkdir(parents=True,exist_ok=True)
            atomic_json(path,payload)
        # Mark the project saved after the write succeeds.
        def done(_):
            self.dirty=False
            self.status('Complete review session saved: '+path)
        self.start('Saving review session…',write,done)

    # Collect the state that needs to survive closing the app.
    def project_payload(self, *, snapshot=True, include_asr=True):
        capture=copy.deepcopy if snapshot else lambda value:value
        self.require_audio()
        if hasattr(self, 'asr'):
            self.asr.commit_drafts()
        return dict(format='speech_corpus_qt_session_v1',audio=str(self.audio),audio_size=self.audio.stat().st_size,duration=self.duration,
                    reviewer=self.ui.reviewer.text(),regions=capture(self.regions),changes=capture(self.changes),step4=capture(self.session),
                    snr=capture(self.snr_results),snr_threshold=self.applied_threshold,updated_utc=review.utc_now(),
                    snr_policy=self.snr_policy,snr_justification=self.ui.snrJustification.text().strip(),
                    improve_singing=self.ui.improveSinging.isChecked(),
                    panns_inputs=dict(window=self.ui.pannsWindowInput.text(), hop=self.ui.pannsHopInput.text()),
                    asr=self.asr.payload(snapshot=snapshot) if include_asr and hasattr(self,'asr') else None)

    # Choose a saved session to reopen.
    def open_project(self):
        path,_=QFileDialog.getOpenFileName(self.window,'Open complete review session','','Review session (*.json)')
        if not path:return
        if not self.confirm_discard():return
        cached_audio=self.audio if self.plot_data is not None else None
        # Prepare its recording, quality results and plot before updating the window.
        def read(log):
            data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
            if data.get('format')!='speech_corpus_qt_session_v1':raise ValueError('Not a combined PyQt session.')
            audio=Path(data['audio']);frames,rate=review.inspect_wav(audio)
            if data['audio_size']!=audio.stat().st_size or abs(data['duration']-frames/rate)>1/rate:
                raise ValueError('The saved source audio has changed.')
            if data['regions']:validate_regions(data['regions'],data['duration'])
            if data.get('step4'):
                source=data['step4']['source'];review.check_source(audio,source['audio_name'],source['audio_size'],source['duration_seconds'])
                if not Path(data['step4']['candidate_root']).is_dir():raise ValueError('Keep the original candidate folder to resume review.')
            if data.get('snr_policy','energy') not in snr.POLICIES:
                raise ValueError('Invalid saved SNR decision rule.')
            saved_asr=data.get('asr')
            prepared=self.asr.prepare_records(saved_asr) if saved_asr else None
            plot=calculate_spectrogram(str(audio)) if data['regions'] and audio!=cached_audio else None
            quality=None
            if data.get('snr'):
                rows=self.validate_saved_snr(data['snr'],data.get('step4'),audio)
                quality=snr.reclassify(rows,float(data.get('snr_threshold',20)),data.get('snr_policy','energy'),data.get('snr_justification',''))
            return data, audio, prepared, plot, quality
        self.start('Opening review session…',read,self.restore_project)

    # Restore both the processing pages and the transcript editor.
    def restore_project(self,result):
        data,audio,prepared,plot,quality=result
        self.set_audio(audio,ask=False)
        self.regions=data['regions'];self.changes=data['changes'];self.session=data.get('step4')
        self.snr_results=[];self.snr_snapshot=None;self.applied_threshold=float(data.get('snr_threshold',20))
        self.snr_policy=data.get('snr_policy','energy');self.snr_justification=data.get('snr_justification','')
        if self.snr_policy not in snr.POLICIES:raise ValueError('Invalid saved SNR decision rule.')
        self.ui.reviewer.setText(data.get('reviewer',''));self.set_snr_controls()
        if quality is not None:
            self.snr_results=quality
            self.snr_snapshot=copy.deepcopy(self.session)
        self.ui.improveSinging.setChecked(data.get('improve_singing', True))
        inputs=data.get('panns_inputs') or {}
        self.ui.pannsWindowInput.setText(str(inputs.get('window', '')))
        self.ui.pannsHopInput.setText(str(inputs.get('hop', '')))
        self.asr.restore(data.get('asr'), saved_snr=data.get('snr'),prepared=prepared)
        self.dirty=False;self.refresh_all()
        if plot is not None:self.plot_spectrogram(plot)
        else:self.redraw_regions()
        self.status('Session restored.')

    # Play just the requested part of a recording.
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

    # Wait for the media to load before seeking to the requested start time.
    def media_ready(self,status):
        if status==QMediaPlayer.MediaStatus.LoadedMedia and self.play_range:
            self.player.setPosition(self.pending_seek if self.pending_seek is not None else self.play_range[0])
            if not self.play_paused:self.player.play()
        elif status==QMediaPlayer.MediaStatus.EndOfMedia and self.play_range:
            self.update_region_playback_position(self.play_range[1])
            self.update_part4_playback_position(self.play_range[1])
            self.stop_playback()

    # Keep the region's listening cursor aligned with playback.
    def update_region_playback_position(self, position):
        slider = self.ui.regionSeekSlider
        if self.play_owner in self.part4_previews or self.play_path != self.audio or self.preview_bounds is None or slider.isSliderDown():
            return
        slider.blockSignals(True)
        slider.setValue(position)
        slider.blockSignals(False)
        self.update_region_cursor_label()

    # Send each position update to the preview controls that need it.
    def playback_position(self,position):
        self.update_region_playback_position(position)
        self.update_part4_playback_position(position)
        if self.play_range and position>=self.play_range[1]:self.stop_playback()

    # Stop at the marked endpoint so playback doesn't spill into the next clip.
    def enforce_playback_end(self):
        if self.play_range and self.player is not None:self.playback_position(self.player.position())

    # Release the current listening range and stop the player.
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

    # Play the highlighted region from start to finish.
    def play_region(self):
        r=self.regions[self.index(self.ui.regionTable)];self.play(self.audio,r['start'],r['end'])

    # Play a little source audio on either side of the chosen region edge.
    def play_region_edge(self,end):
        r=self.regions[self.index(self.ui.regionTable)];t=r['end' if end else 'start'];self.play(self.audio,t-1,t+1)

    # Play the candidate interval currently shown in the segment editor.
    def play_segment(self):
        path,a,b=self.part4_interval('segment')
        self.play(path,a,b,owner='segment')

    # Listen across the candidate's cut in the original recording.
    def play_segment_edge(self,end):
        item=self.session['segments'][self.index(self.ui.segmentTable)];t=item['source_end' if end else 'source_start']
        self.play(self.audio,t-1,t+1)

    # Play the highlighted unresolved span so the reviewer can decide what to keep.
    def play_unresolved(self):
        path,a,b=self.part4_interval('unresolved')
        self.play(path,a,b,owner='unresolved')

    # Play the audio behind the selected quality measurement.
    def play_quality(self):
        self.check_snr();r=self.snr_results[self.index(self.ui.snrTable)];self.play(r['path'],r['read_start'],r['read_end'])

"""PyQt6 adapter for the supplied Parts 3–5 processing functions.

All widgets and page layouts are editable in main_window.ui. Model inference and file-heavy
work run on a QThread; all widget updates run on the main Qt thread.
"""
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QHeaderView,
                            QMessageBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import soundfile as sf

from audio_processing import calculate_spectrogram
from review_report import load_review, save_report
from step3_label_regions import classify_file
import step4_review as review
import step5_snr as snr

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
        self.snr_report = None
        self.task = None
        self.dirty = False
        self.play_range = None
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
        self.ui.segmentTable.itemSelectionChanged.connect(self.segment_selected)
        self.ui.unresolvedTable.itemSelectionChanged.connect(self.unresolved_selected)
        self.ui.wavPath.setReadOnly(True)
        actions = {
            'chooseWav': self.choose_audio, 'showSpectrogram': self.show_spectrogram,
            'analyzeRegions': self.analyze, 'loadRegions': self.load_regions,
            'saveRegions': self.save_regions, 'applyRegion': self.edit_region,
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
        self.navigate(0)
        self.refresh_all()

    def navigate(self, index):
        self.window.pages.setCurrentIndex(index)
        processing_page = 1 <= index <= 3
        self.ui.sharedControlsFrame.setVisible(processing_page)
        self.ui.processingStatusFrame.setVisible(processing_page)
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

    def analyze(self):
        self.require_audio()
        if not self.confirm_discard():
            return
        path = str(self.audio)
        def action(log):
            regions = classify_file(path, log)
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
        unresolved = self.session['unresolved'] if self.session else []
        self.populate(self.ui.unresolvedTable, [[s['id'],*[f'{v:.4f}' for v in review.unresolved_bounds(s)],s['label'],s['status'],s['original'].get('reason','')] for s in unresolved])
        self.refresh_snr()

    def region_selected(self):
        rows = self.selected(self.ui.regionTable)
        if rows:
            r = self.regions[rows[0]]
            self.ui.regionLabel.setCurrentText(r['label'])
            self.ui.regionStart.setValue(r['start'])
            self.ui.regionEnd.setValue(r['end'])

    def clear_snr(self):
        self.snr_results = []
        self.snr_snapshot = None
        self.snr_report = None
        self.refresh_snr()

    def invalidate_segments(self):
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
            self.ui.regionEnd.setValue(event.xdata)
            self.guard(self.edit_region)

    def region_payload(self):
        self.require_audio();validate_regions(self.regions,self.duration)
        return dict(version=1,audio_name=self.audio.name,audio_size=self.audio.stat().st_size,duration_seconds=self.duration,
                    method='PANNs 2-second windows and RMS silence threshold',score_note='Uncalibrated automatic scores; manual boundaries may differ from inference windows.',regions=self.regions,changes=self.changes)

    def save_regions(self):
        payload=self.region_payload()
        path,_=QFileDialog.getSaveFileName(self.window,'Save region HTML report',str(self.audio.with_name(self.audio.stem+'_regions.html')),'HTML report (*.html)')
        if path:
            save_report(path,payload)
            self.status('Region report saved. Use Save session to preserve segment and SNR edits too.')

    def load_regions(self):
        self.require_audio()
        path,_=QFileDialog.getOpenFileName(self.window,'Open region review','','Region report (*.html *.htm *.json)')
        if not path:return
        data=load_review(path)
        self.check_region_payload(data)
        if not self.confirm_discard():return
        self.regions=copy.deepcopy(data['regions']);self.changes=data.get('changes',[])
        self.invalidate_segments();self.dirty=True;self.refresh_all();self.redraw_regions()
        self.status('Region review loaded. Show spectrogram if needed.')

    def check_region_payload(self,data):
        if data['audio_name']!=self.audio.name or data['audio_size']!=self.audio.stat().st_size or abs(data['duration_seconds']-self.duration)>1/16000:
            raise ValueError('Region report does not match the selected audio.')
        validate_regions(data['regions'],self.duration)

    def generate_segments(self):
        self.require_audio();validate_regions(self.regions,self.duration)
        if self.session and not self.confirm_discard():return
        folder=QFileDialog.getExistingDirectory(self.window,'Choose output folder',str(self.audio.parent))
        if not folder:return
        regions=copy.deepcopy(self.regions);path=str(self.audio)
        compact=self.ui.segmentationMode.currentIndex()==1
        def action(log):
            if compact:from step4_compact_segments import export_candidates
            else:from step4_cut_segments import export_candidates
            result=export_candidates(path,regions,folder,log)
            return review.load_session(Path(result['output_dir'])/'step4_segments.json',path)
        self.start('Finding speech and natural pause candidates…',action,self.set_session)

    def set_session(self,data):
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
        if not rows or not self.session:return
        item=self.session['segments'][rows[0]]
        self.ui.segmentLabel.setCurrentText(item['label'])
        if item['source_mode']=='continuous_source':
            self.ui.boundaryMode.setCurrentIndex(1);a,b=item['source_start'],item['source_end']
        else:
            self.ui.boundaryMode.setCurrentIndex(0);a,b=review.candidate_trim_bounds(item)
        self.ui.segmentStart.setValue(a);self.ui.segmentEnd.setValue(b)

    def modified_session(self):
        self.clear_snr();self.dirty=True;self.refresh_all()
        self.status('Review changed. Recalculate SNR after approving the updated clips.')

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
        if not rows or not self.session:return
        item=self.session['unresolved'][rows[0]];a,b=review.unresolved_bounds(item)
        self.ui.unresolvedStart.setValue(a);self.ui.unresolvedEnd.setValue(b)
        if item['label'] in LABELS:self.ui.unresolvedLabel.setCurrentText(item['label'])

    def add_manual(self):
        i=self.index(self.ui.unresolvedTable)
        review.add_from_unresolved(self.session,i,self.ui.unresolvedStart.value(),self.ui.unresolvedEnd.value(),self.ui.unresolvedLabel.currentText(),self.reviewer())
        self.modified_session()

    def mark_unresolved(self):
        i=self.index(self.ui.unresolvedTable);name=self.reviewer()
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
        folder=QFileDialog.getExistingDirectory(self.window,'Export approved WAV files',str(self.audio.parent))
        if not folder:return
        snapshot=copy.deepcopy(self.session);path=str(self.audio)
        def done(result):
            self.status(f"Exported {result['exported']} approved clips to {result['output_dir']}. Pending spans: {result['pending_unresolved']}.")
            self.window.open_path(Path(result['output_dir']))
        self.start('Exporting reviewed clips and source mappings…',lambda log:review.save_and_export(snapshot,path,folder),done)

    def calculate_snr(self):
        self.require_approved()
        snapshot=copy.deepcopy(self.session);path=str(self.audio);threshold=self.ui.snrThreshold.value()
        def action(log):
            records=snr.session_records(snapshot,path)
            return snr.calculate(records,threshold,lambda n,total:log(f'Calculating energy SNR: {n} / {total}'))
        def done(records):
            self.snr_results=records;self.snr_snapshot=snapshot;self.applied_threshold=threshold
            self.snr_report=None;self.dirty=True;self.refresh_snr()
            self.status(f'{len(records)} energy SNR estimates ready. Second method still required for Part 5.')
        self.start('Calculating SNR for the approved clips…',action,done)

    def refresh_snr(self):
        if not hasattr(self,'histogram'):return
        self.populate(self.ui.snrTable,[[r['id'],f"{r['source_start']:.3f}",f"{r['source_end']:.3f}",r['content_label'],
                       'Unavailable' if r['snr_db'] is None else f"{r['snr_db']:.2f}",r['automatic_quality'],r['quality'],r['note']] for r in self.snr_results])
        self.histogram.clear();a,b=self.histogram.subplots(1,2)
        scores=[r['snr_db'] for r in self.snr_results if r['snr_db'] is not None]
        if scores:a.hist(scores,bins=min(20,max(3,len(scores))),color='#268b80',edgecolor='white')
        a.axvline(self.applied_threshold,color='#d59a45',linestyle='--',label=f'{self.applied_threshold:g} dB')
        a.set(xlabel='Energy SNR (dB)',ylabel='Clips',title='SNR distribution');a.legend(fontsize=8)
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
            r=self.snr_results[i];after=r['automatic_quality'] if label is None else label
            r.setdefault('quality_changes',[]).append(dict(time_utc=review.utc_now(),reviewer=who,reason=why,before=r['quality'],after=after))
            r.update(quality=after,override_by=who,override_reason=why)
        self.dirty=True;self.refresh_snr();self.status('Quality corrections recorded. Save the session or updated report.')

    def save_snr(self):
        self.check_snr()
        folder=QFileDialog.getExistingDirectory(self.window,'Save SNR HTML / CSV / JSON',str(self.audio.parent))
        if not folder:return
        self.snr_report=Path(snr.export_report(self.snr_results,self.applied_threshold,folder)[0])
        self.status('Saved SNR reports: '+str(self.snr_report))

    def open_snr_report(self):
        if not self.snr_report:raise ValueError('Save the SNR report first.')
        self.window.open_path(self.snr_report)

    def load_snr(self):
        self.require_approved()
        path,_=QFileDialog.getOpenFileName(self.window,'Open SNR review','','SNR JSON (*.json)')
        if not path:return
        payload=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if payload.get('format')!='speech_corpus_snr_v1':raise ValueError('Not a supported SNR report.')
        records=payload['segments'];self.match_snr(records)
        threshold=float(payload['threshold_db'])
        if not math.isfinite(threshold) or not -50<=threshold<=100:raise ValueError('Invalid SNR threshold.')
        self.snr_results=records;self.snr_snapshot=copy.deepcopy(self.session);self.applied_threshold=threshold
        self.ui.snrThreshold.setValue(threshold);self.snr_report=Path(path).with_suffix('.html')
        self.dirty=True;self.refresh_snr();self.status('SNR review loaded.')

    def match_snr(self,records):
        current=snr.session_records(self.session,str(self.audio))
        keys=('id','path','read_start','read_end','source_start','source_end','content_label','synthetic_start','synthetic_end')
        if len(records)!=len(current) or any(any(a.get(k)!=b.get(k) for k in keys) for a,b in zip(records,current)):
            raise ValueError('Saved SNR results do not match the approved clips.')
        for r in records:
            if r['snr_db'] is not None and not math.isfinite(float(r['snr_db'])):raise ValueError('Invalid SNR result.')
            if r['quality'] not in ('clean','noisy','unavailable'):raise ValueError('Invalid quality label.')

    def save_project(self):
        self.require_audio()
        path,_=QFileDialog.getSaveFileName(self.window,'Save complete review session',str(self.audio.with_name(self.audio.stem+'_session.json')),'Review session (*.json)')
        if not path:return
        payload=dict(format='speech_corpus_qt_session_v1',audio=str(self.audio),audio_size=self.audio.stat().st_size,duration=self.duration,
                     reviewer=self.ui.reviewer.text(),regions=self.regions,changes=self.changes,step4=self.session,
                     snr=self.snr_results,snr_threshold=self.applied_threshold,updated_utc=review.utc_now())
        atomic_json(path,payload);self.dirty=False;self.status('Complete review session saved. Keep the source and candidate folders in place.')

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
        self.ui.reviewer.setText(data.get('reviewer',''));self.ui.snrThreshold.setValue(self.applied_threshold)
        if data.get('snr'):
            self.match_snr(data['snr']);self.snr_results=data['snr'];self.snr_snapshot=copy.deepcopy(self.session)
        self.dirty=False;self.refresh_all();self.redraw_regions();self.status('Session restored. Click Show spectrogram to rebuild the display.')

    def play(self,path,a,b):
        if b<=a:raise ValueError('No playable audio in the selected interval.')
        self.stop_playback()
        if self.player is None:
            self.player=QMediaPlayer(self)
            self.audio_output=QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
            self.player.mediaStatusChanged.connect(self.media_ready)
            self.player.positionChanged.connect(self.playback_position)
            self.player.errorOccurred.connect(lambda error,text:self.status('Playback: '+text))
        frames,rate=review.inspect_wav(path)
        a,b=max(0,a),min(frames/rate,b)
        self.play_range=(round(a*1000),round(b*1000))
        url=QUrl.fromLocalFile(str(Path(path).resolve()))
        if self.player.source()==url and self.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia,QMediaPlayer.MediaStatus.BufferedMedia):
            self.player.setPosition(self.play_range[0]);self.player.play()
        else:
            self.player.setSource(url)
        self.play_timer.start(40)

    def media_ready(self,status):
        if status==QMediaPlayer.MediaStatus.LoadedMedia and self.play_range:
            self.player.setPosition(self.play_range[0]);self.player.play()

    def playback_position(self,position):
        if self.play_range and position>=self.play_range[1]:self.stop_playback()

    def enforce_playback_end(self):
        if self.play_range and self.player.position()>=self.play_range[1]:self.stop_playback()

    def stop_playback(self):
        self.play_range=None
        if self.player is not None:self.player.stop()
        self.play_timer.stop()

    def play_region(self):
        r=self.regions[self.index(self.ui.regionTable)];self.play(self.audio,r['start'],r['end'])

    def play_region_edge(self,end):
        r=self.regions[self.index(self.ui.regionTable)];t=r['end' if end else 'start'];self.play(self.audio,t-1,t+1)

    def play_segment(self):
        i=self.index(self.ui.segmentTable);item=self.session['segments'][i]
        if item['source_mode']=='continuous_source':path=self.audio;a,b=item['source_start'],item['source_end']
        else:path=review.safe_candidate_path(self.session['candidate_root'],item['candidate_file']);a,b=review.candidate_trim_bounds(item)
        self.play(path,a,b)

    def play_segment_edge(self,end):
        item=self.session['segments'][self.index(self.ui.segmentTable)];t=item['source_end' if end else 'source_start']
        self.play(self.audio,t-1,t+1)

    def play_unresolved(self):
        item=self.session['unresolved'][self.index(self.ui.unresolvedTable)];a,b=review.unresolved_bounds(item);self.play(self.audio,a,b)

    def play_quality(self):
        self.check_snr();r=self.snr_results[self.index(self.ui.snrTable)];self.play(r['path'],r['read_start'],r['read_end'])

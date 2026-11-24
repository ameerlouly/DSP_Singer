"""Launch Speech Corpus Studio and connect the source, extraction and activity controls.

Function guide
--------------
PipelineWorker.run
    Perform the download or extraction outside the GUI thread.
MainWindow, enter_workspace, main
    Load the Designer UI, initialize the review pages and start the Qt event loop.
append_log, choose_output, choose_video, extraction_format, update_format_hint
    Collect source and format choices and show operation details.
start_operation, set_download_progress, set_download_details
    Start the worker and display its reported transfer progress.
operation_succeeded, operation_failed, operation_finished, update_elapsed
    Handle worker outcomes, timing and the availability of controls.
show_source_details, open_path, open_report, open_result_folder, open_project_folder,
rename_folder
    Display source metadata and manage the user's output folders.
closeEvent, stop_running_operation_and_exit
    Confirm closure during an operation and stop active work before exit.
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from PyQt6 import uic
from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMainWindow, QMessageBox

from video_download_and_audio_extraction import CORPUS_FORMAT, WavFormat, download_video, extract_audio, rename_video_folder


class PipelineWorker(QThread):
    message = pyqtSignal(str)
    progress = pyqtSignal(int)
    download_details = pyqtSignal(object)
    succeeded = pyqtSignal(str, str)
    failed = pyqtSignal(str)

    def __init__(self, operation, value, output_folder='', folder_name='', parent=None, *, wav_format=CORPUS_FORMAT):
        super().__init__(parent)
        self.operation = operation
        self.value = value
        self.output_folder = output_folder
        self.folder_name = folder_name
        self.wav_format = wav_format

    def run(self):
        try:
            if self.operation == 'download':
                result = download_video(
                    self.value,
                    output_root=self.output_folder,
                    folder_name=self.folder_name,
                    log=self.message.emit,
                    progress_callback=self.progress.emit,
                    progress_details_callback=self.download_details.emit,
                )
            else:
                result = extract_audio(self.value, log=self.message.emit, wav_format=self.wav_format)
            self.succeeded.emit(self.operation, str(result))
        except Exception as error:
            self.failed.emit(str(error))


class MainWindow(QMainWindow):
    def __init__(self, *, show_welcome=True):
        super().__init__()
        self.project_folder = Path(__file__).resolve().parent
        uic.loadUi(str(self.project_folder / 'main_window.ui'), self)
        self.welcomeBackground.setPixmap(QPixmap(str(self.project_folder / 'assets' / 'welcome_signal_background.svg')))
        self.enterStudioButton.clicked.connect(self.enter_workspace)
        self.appScreens.setCurrentIndex(0 if show_welcome else 1)
        self.statusbar.setVisible(not show_welcome)
        self.setWindowTitle('Speech Signal Processing' if show_welcome else 'Speech Corpus Studio')
        if show_welcome:
            QTimer.singleShot(0, self.enterStudioButton.setFocus)
        self.downloadProgressBar.setTextVisible(False)
        self.downloadProgressBar.setRange(0, 100)
        self.downloadProgressBar.setValue(0)
        self.download_animation = QPropertyAnimation(self.downloadProgressBar, b'value', self)
        self.download_animation.setEasingCurve(QEasingCurve.Type.Linear)
        self.download_target = 0
        self.worker = None
        self.last_result = None
        self.report_path = None
        self.started = None
        self.failed_operation = False
        self.active_operation = None

        self.outputFolderInput.clear()
        self.downloadButton.clicked.connect(lambda: self.start_operation('download'))
        self.extractButton.clicked.connect(lambda: self.start_operation('extract'))
        self.autoProcessButton.clicked.connect(lambda: self.processing.guard(self.processing.start_auto_process))
        self.browseOutputButton.clicked.connect(self.choose_output)
        self.chooseVideoButton.clicked.connect(self.choose_video)
        self.openReportButton.clicked.connect(self.open_report)
        self.openOutputFolderButton.clicked.connect(self.open_result_folder)
        self.openProjectFolderButton.clicked.connect(self.open_project_folder)
        self.renameFolderButton.clicked.connect(self.rename_folder)
        self.controls = [
            self.urlInput, self.outputFolderInput, self.folderNameInput,
            self.videoPathInput, self.downloadButton, self.extractButton, self.autoProcessButton,
            self.browseOutputButton, self.chooseVideoButton, self.renameFolderButton,
        ]
        for control in (self.channelBox, self.sampleRateBox, self.bitDepthBox):
            self.controls.append(control)
            control.currentIndexChanged.connect(self.update_format_hint)
        self.update_format_hint()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_elapsed)
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setFormat('%p%')
        # Drive the visible Activity bar from the same smoothed download value.
        self.downloadProgressBar.valueChanged.connect(self.progressBar.setValue)
        self.openReportButton.setEnabled(False)
        self.openOutputFolderButton.setEnabled(False)
        self.append_log('Ready. Download a video or choose a local video for audio extraction.')
        from audio_review_controller import ProcessingWorkspace
        self.processing = ProcessingWorkspace(self)

    def enter_workspace(self):
        self.appScreens.setCurrentIndex(1)
        self.statusbar.show()
        self.setWindowTitle('Speech Corpus Studio')
        self.urlInput.setFocus()

    def append_log(self, message):
        self.activityLog.appendPlainText(f'{datetime.now():%H:%M:%S}  {message}')
        scrollbar = self.activityLog.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def choose_output(self):
        chosen = QFileDialog.getExistingDirectory(self, 'Choose destination', self.outputFolderInput.text())
        if chosen:
            self.outputFolderInput.setText(chosen)

    def choose_video(self):
        chosen, _ = QFileDialog.getOpenFileName(
            self, 'Choose a video', str(self.project_folder),
            'Video files (*.mp4 *.mkv *.webm *.mov *.avi *.m4v *.ts *.flv);;All files (*)',
        )
        if chosen:
            self.videoPathInput.setText(chosen)
            self.show_source_details(Path(chosen).parent)

    def extraction_format(self):
        return WavFormat(sample_rate=int(self.sampleRateBox.currentText()),
                         channels={'Mono': 1, 'Stereo': 2}[self.channelBox.currentText()],
                         bit_depth=int(self.bitDepthBox.currentText()))

    def update_format_hint(self):
        if self.extraction_format() == CORPUS_FORMAT:
            self.audioFormatLabel.setText('Corpus format · Opens in Regions after extraction.')
        else:
            self.audioFormatLabel.setText('Custom WAV export. For Regions, select 16000 Hz, mono, 16-bit.')

    def start_operation(self, operation):
        if self.worker is not None or self.processing.task is not None:
            return
        value = self.urlInput.text().strip() if operation == 'download' else self.videoPathInput.text().strip()
        output_folder = self.outputFolderInput.text().strip()
        if operation == 'download' and not output_folder:
            QMessageBox.warning(self, 'Destination required',
                                'No destination selected. Click Browse and choose a folder before downloading.')
            self.outputFolderInput.setFocus()
            return
        if not value:
            message = 'Enter a video URL.' if operation == 'download' else 'Choose a local video.'
            QMessageBox.information(self, 'Missing input', message)
            return
        wav_format = self.extraction_format()
        if operation == 'extract' and wav_format == CORPUS_FORMAT and not self.processing.confirm_discard():
            return
        self.failed_operation = False
        self.active_operation = operation
        self.started = time.monotonic()
        self.elapsedLabel.setText('0s elapsed')
        self.timer.start(1000)
        for control in self.controls:
            control.setEnabled(False)
        self.openReportButton.setEnabled(False)
        self.openOutputFolderButton.setEnabled(False)
        if operation == 'download':
            self.download_animation.stop()
            self.download_target = 0
            self.downloadProgressBar.setRange(0, 100)
            self.downloadProgressBar.setValue(0)
            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(0)
            self.set_download_progress(-1)
            self.statusLabel.setText('Reading video information…')
        else:
            self.download_animation.stop()
            self.statusLabel.setText('Extracting and verifying audio…')
            self.progressBar.setTextVisible(False)
            self.progressBar.setRange(0, 0)
        self.worker = PipelineWorker(
            operation, value, output_folder, self.folderNameInput.text().strip(), self, wav_format=wav_format,
        )
        self.worker.message.connect(self.append_log)
        self.worker.progress.connect(self.set_download_progress)
        self.worker.download_details.connect(self.set_download_details)
        self.worker.succeeded.connect(self.operation_succeeded)
        self.worker.failed.connect(self.operation_failed)
        self.worker.finished.connect(self.operation_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def set_download_progress(self, value):
        self.downloadProgressBar.setTextVisible(False)
        self.progressBar.setTextVisible(True)
        if value < 0:
            # Keep the last measured fill while metadata/total size is unknown.
            # Range (0, 0) would display the misleading half-filled busy block.
            self.progressBar.setRange(0, 100)
            self.progressBar.setFormat('%p%' if self.download_target > 0 else '—%')
            self.progressBar.setToolTip('Waiting for a total size estimate; the percentage retains the highest progress reached.')
            if self.download_target >= 99:
                self.statusLabel.setText('Merging video/audio and saving reports…')
            elif self.download_target > 0:
                self.statusLabel.setText('Downloading or processing… waiting for size information.')
            return
        value = max(0, min(100, int(value)))
        # Size estimates and retried streams can lower the raw byte ratio. Keep
        # the highest target for this download, including one still animating.
        # The fill waits until measured progress catches up; a new download
        # resets download_target to zero in start_operation.
        if value < self.download_target:
            self.progressBar.setToolTip('Holding the highest percentage while the revised download estimate catches up.')
            value = self.download_target
        self.progressBar.setRange(0, 100)
        self.progressBar.setFormat('%p%')
        if value == self.download_target and self.downloadProgressBar.maximum() != 0:
            return
        start_value = max(0, self.downloadProgressBar.value())
        self.download_animation.stop()
        self.downloadProgressBar.setRange(0, 100)
        self.download_target = value
        if value == 0:
            self.downloadProgressBar.setValue(value)
        else:
            # Brief interpolation follows measured updates without a fixed fill rate.
            self.download_animation.setDuration(max(120, min(600, abs(value - start_value) * 12)))
            self.download_animation.setStartValue(start_value)
            self.download_animation.setEndValue(value)
            self.download_animation.start()
        if value == 99:
            self.statusLabel.setText('Finishing download and saving reports…')
        elif value < 99:
            self.statusLabel.setText('Downloading video and audio…')

    def set_download_details(self, details):
        state = details.get('state')
        if state == 'reading':
            self.statusLabel.setText('Reading video information…')
            return
        if state == 'finishing':
            text = 'Merging audio/video and verifying output…'
            self.progressBar.setFormat('%p%')
            self.progressBar.setToolTip('Finishing the download; 100% follows successful verification.')
        elif state == 'complete':
            text = 'Video downloaded successfully'
            self.progressBar.setFormat('%p%')
            self.progressBar.setToolTip('Download completed and verified.')
        elif details.get('total_bytes') is None:
            text = 'Downloading; waiting for a total size estimate…'
            self.progressBar.setFormat('%p%' if self.download_target > 0 else '—%')
            self.progressBar.setToolTip('Waiting for a total size estimate; the percentage retains the highest progress reached.')
        else:
            text = 'Downloading video and audio…'
            self.progressBar.setFormat('%p%')
            self.progressBar.setToolTip(
                ('Estimated download percentage. ' if details.get('total_is_estimate') else
                 'Download percentage from the combined video and audio transfer progress. ')
                + 'The display never decreases; it may pause while a revised size estimate catches up.'
            )
        self.statusLabel.setText(text)

    def operation_succeeded(self, operation, path):
        self.last_result = Path(path)
        if operation == 'download':
            self.videoPathInput.setText(path)
            self.folderNameInput.clear()
            self.report_path = self.last_result.parent / 'metadata.html'
            self.set_download_progress(100)
            self.statusLabel.setText('Video downloaded successfully')
        else:
            self.report_path = self.last_result.with_suffix('.html')
            self.statusLabel.setText('Audio extracted and verified')
            if self.worker.wav_format == CORPUS_FORMAT:
                # Re-extraction can replace the currently reviewed WAV.
                self.processing.audio = None
                self.processing.set_audio(path, ask=False)
            else:
                self.statusLabel.setText('Custom WAV exported and verified')
                self.append_log('Custom export saved. To use this video in Regions, extract it with the corpus settings.')
        self.show_source_details(self.last_result.parent if operation == 'download' else self.last_result.parent.parent)
        self.append_log(f'Completed: {path}')

    def operation_failed(self, message):
        self.failed_operation = True
        self.download_animation.stop()
        if self.active_operation == 'download':
            # Stop animation when metadata fails before a percentage is available.
            if self.downloadProgressBar.maximum() == 0:
                self.downloadProgressBar.setRange(0, 100)
                self.downloadProgressBar.setValue(0)
        self.statusLabel.setText('Operation failed — see Activity')
        self.append_log('ERROR: ' + message)
        QMessageBox.warning(self, 'Processing failed', message)

    def operation_finished(self):
        self.update_elapsed()
        self.timer.stop()
        for control in self.controls:
            control.setEnabled(True)
        self.progressBar.setRange(0, 100)
        if self.failed_operation:
            self.progressBar.setValue(0)
        elif self.active_operation != 'download':
            self.progressBar.setValue(100)
        self.openReportButton.setEnabled(bool(self.report_path and self.report_path.is_file()))
        self.openOutputFolderButton.setEnabled(bool(self.last_result and self.last_result.parent.is_dir()))
        self.worker = None
        self.active_operation = None

    def update_elapsed(self):
        if self.started is not None:
            self.elapsedLabel.setText(f'{int(time.monotonic() - self.started)}s elapsed')

    def show_source_details(self, folder):
        fields = {'title':self.titleValueLabel, 'channel':self.channelValueLabel,
                  'duration_seconds':self.durationValueLabel, 'license':self.licenseValueLabel}
        data = {}
        record = folder / 'metadata.json'
        if record.is_file():
            try:
                data = json.loads(record.read_text(encoding='utf-8'))
                if not isinstance(data, dict):
                    raise ValueError('Expected a metadata object.')
            except (OSError, ValueError) as error:
                self.append_log(f'Could not read source details: {error}')
                data = {}
        for key, label in fields.items():
            value = data.get(key)
            if key == 'duration_seconds' and value is not None:
                try:
                    seconds = max(0, int(float(value)))
                    hours, remainder = divmod(seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    value = f'{hours:02}:{minutes:02}:{seconds:02}'
                except (TypeError, ValueError, OverflowError):
                    value = str(value)
            label.setText(str(value) if value is not None and value != '' else 'Not available yet')

    def open_path(self, path):
        if not path.exists():
            QMessageBox.information(self, 'File unavailable', f'Could not find {path}')
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
            QMessageBox.warning(self, 'Cannot open', f'Windows could not open {path}')

    def open_report(self):
        if self.report_path:
            self.open_path(self.report_path)

    def open_result_folder(self):
        if self.last_result:
            self.open_path(self.last_result.parent)

    def open_project_folder(self):
        self.open_path(self.project_folder)

    def rename_folder(self):
        if self.processing.audio is not None:
            QMessageBox.information(self, 'Audio review is loaded',
                'Rename the video folder before loading it into Regions. Moving a loaded source breaks candidate paths in saved reviews.')
            return
        current_video = self.videoPathInput.text().strip()
        folder = Path(current_video).resolve().parent if current_video else None
        if folder is None or not (folder / 'metadata.html').is_file():
            chosen = QFileDialog.getExistingDirectory(self, 'Choose completed video folder', self.outputFolderInput.text())
            if not chosen:
                return
            folder = Path(chosen).resolve()
        name, accepted = QInputDialog.getText(self, 'Rename video folder', 'Folder name', text=folder.name)
        if not accepted:
            return
        try:
            new_folder = rename_video_folder(folder, name)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, 'Cannot rename folder', str(error))
            return
        if current_video and Path(current_video).resolve().parent == folder:
            self.videoPathInput.setText(str(new_folder / Path(current_video).name))
        if self.last_result and self.last_result.parent == folder:
            self.last_result = new_folder / self.last_result.name
        if self.report_path and self.report_path.parent == folder:
            self.report_path = new_folder / self.report_path.name
        self.append_log(f'Folder renamed: {new_folder}')

    def closeEvent(self, event):
        if self.worker is not None or self.processing.task is not None:
            event.ignore()
            answer = QMessageBox.warning(
                self, 'Operation running',
                'An operation is still running.\n\n'
                'OK will terminate the operation and close the app. '
                'Unsaved changes will be lost, and incomplete output files may remain.\n\n'
                'Cancel will keep the app open and let the operation continue.',
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Ok:
                self.stop_running_operation_and_exit()
        else:
            if self.processing.confirm_discard():
                self.processing.stop_playback()
                event.accept()
            else:
                event.ignore()

    def stop_running_operation_and_exit(self):
        """Stop this app's child processes, then exit without destroying live QThreads.

        A worker can be inside native inference or a blocking subprocess call.
        QThread.terminate() can leave Python/library locks corrupted, and ordinary
        Qt teardown destroys still-running threads. A confirmed application exit
        instead ends the process after its FFmpeg/model child processes stop.
        """
        try:
            import psutil
            children = psutil.Process(os.getpid()).children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(children, timeout=2)
            if alive:
                raise RuntimeError('Some processing subprocesses could not be stopped. Try closing again.')
        except Exception as error:
            QMessageBox.warning(
                self, 'Could not stop processing',
                'The app has stayed open because processing could not be stopped.\n\n'
                + str(error) + '\n\nIf psutil is missing, run run_app.bat to install the updated requirements.',
            )
            return
        self.processing.stop_playback()
        # This is only reached after the user explicitly confirms termination.
        # It ends active Python/native worker threads without Qt destructor races.
        for stream in (sys.stdout, sys.stderr):
            try:
                if stream is not None:
                    stream.flush()
            except (OSError, ValueError):
                pass
        os._exit(0)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

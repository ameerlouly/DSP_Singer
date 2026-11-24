"""Check Cancel/OK handling and confirmed exit with a real thread/child process.

The checks run in file order when this script is executed directly.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication, QMessageBox
from speech_corpus_app import MainWindow

app = QApplication([])
window = MainWindow()
for target in ('download', 'processing'):
    if target == 'download':
        window.worker = object()
    else:
        window.processing.task = object()
    with patch.object(QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Cancel) as dialog, patch.object(window, 'stop_running_operation_and_exit') as stop:
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()
        stop.assert_not_called()
        assert dialog.call_args.args[-1] == QMessageBox.StandardButton.Cancel
    with patch.object(QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Ok), patch.object(window, 'stop_running_operation_and_exit') as stop:
        event = QCloseEvent()
        window.closeEvent(event)
        stop.assert_called_once()
    window.worker = None
    window.processing.task = None
with patch('psutil.Process', side_effect=psutil.AccessDenied()), patch.object(QMessageBox, 'warning') as dialog, patch('speech_corpus_app.os._exit') as exit_app:
    window.stop_running_operation_and_exit()
    exit_app.assert_not_called()
    dialog.assert_called_once()
window.processing.dirty = False
window.close()

# A separate application process verifies the real forced-exit path, avoiding
# deliberate termination of the test runner itself.
with tempfile.TemporaryDirectory() as directory:
    pidfile = Path(directory) / 'child.pid'
    script = '''
import sys,subprocess,time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QThread,QTimer
from PyQt6.QtWidgets import QApplication,QMessageBox
sys.path.insert(0,sys.argv[1])
from speech_corpus_app import MainWindow
class Busy(QThread):
    def run(self):
        while True: time.sleep(.02)
app=QApplication([])
w=MainWindow()
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
Path(sys.argv[2]).write_text(str(child.pid))
w.worker=Busy(w);w.worker.start();w.show()
with patch.object(QMessageBox,'warning',return_value=QMessageBox.StandardButton.Ok):
    QTimer.singleShot(100,w.close)
    app.exec()
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(ROOT), str(pidfile)],
                               env=dict(os.environ, QT_QPA_PLATFORM='offscreen'),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_pid = None
    try:
        output, errors = process.communicate(timeout=20)
        assert process.returncode == 0, (process.returncode, output, errors)
        assert pidfile.is_file()
        child_pid = int(pidfile.read_text())
        assert not psutil.pid_exists(child_pid), 'Child processing process was left running'
    finally:
        if process.poll() is None:
            process.kill();process.wait()
        if pidfile.exists():
            child_pid=int(pidfile.read_text())
            try:psutil.Process(child_pid).kill()
            except psutil.NoSuchProcess:pass
print('PASS: Cancel continues, OK invokes shutdown for both worker types, stop failures keep app open, confirmed exit stops a real child process and exits cleanly with a live QThread.')

import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from qt_app import MainWindow
app=QApplication([]);w=MainWindow();w.show();app.processEvents()
shown=[]
w.progressBar.valueChanged.connect(shown.append)
w.set_download_progress(-1)
assert w.progressBar.maximum()==100 and w.progressBar.value()==0
assert w.progressBar.isTextVisible() and w.progressBar.text()=='—%'
w.set_download_progress(50)
assert w.download_animation.duration()==600
assert w.progressBar.value()==0
QTest.qWait(200)
assert 0<w.progressBar.value()<50  # A large update visibly fills between its endpoints.
w.set_download_progress(10)  # A lower update must not reverse an unfinished animation.
assert w.download_target==50
QTest.qWait(500)
assert w.downloadProgressBar.value()==w.progressBar.value()==50
assert w.progressBar.text()=='50%'
QTest.qWait(350)
assert w.progressBar.value()==50  # No new packet means no invented progress.
w.set_download_progress(-1)
assert w.progressBar.maximum()==100 and w.progressBar.value()==50
w.set_download_details(dict(state='transferring',downloaded_bytes=1024*1024,total_bytes=2*1024*1024,total_is_estimate=True))
assert w.progressBar.text()=='50%' and 'Estimated' in w.progressBar.toolTip()
assert not any(unit in w.statusLabel.text() for unit in ('MiB','KiB','bytes'))
# A revised raw estimate may decrease; the displayed percentage must hold.
w.set_download_progress(25);QTest.qWait(350)
assert w.progressBar.value()==50
w.set_download_progress(0);QTest.qWait(150)
assert w.progressBar.value()==50
w.set_download_details(dict(state='transferring',downloaded_bytes=1024,total_bytes=None,total_is_estimate=False))
assert w.progressBar.text()=='50%' and w.progressBar.value()==50
w.set_download_progress(49);QTest.qWait(150)
assert w.progressBar.value()==50
w.set_download_progress(60);QTest.qWait(250)
assert w.progressBar.text()=='60%'
w.set_download_progress(99);QTest.qWait(700)
w.set_download_details(dict(state='finishing',downloaded_bytes=1024,total_bytes=1024,total_is_estimate=False))
assert 'Merging' in w.statusLabel.text() and w.progressBar.text()=='99%'
w.set_download_progress(100);w.active_operation='download';w.operation_finished()
QTest.qWait(350);assert w.progressBar.value()==100
assert w.progressBar.text()=='100%'
w.set_download_details(dict(state='complete',downloaded_bytes=1024,total_bytes=1024,total_is_estimate=False))
assert w.statusLabel.text()=='Video downloaded successfully'
assert not w.downloadProgressBar.isTextVisible() and w.downloadProgressBar.isHidden()
assert shown==sorted(shown),shown
# Reset only for a separate download, without starting a network operation.
w.urlInput.setText('https://example.com/video')
w.outputFolderInput.setText(str(w.project_folder))
with patch('qt_app.PipelineWorker') as worker:
 w.start_operation('download')
 assert w.download_target==0 and w.progressBar.value()==0
 worker.return_value.start.assert_called_once()
 w.operation_finished()
w.close()
print('PASS: percentage-only gradual fill, no reversals during animation or size corrections, holds during unknown totals, resumes after catching up, resets for new downloads and completes at 100%.')

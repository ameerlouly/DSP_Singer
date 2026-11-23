"""Qt singing-option handoff and complete-session persistence without models."""
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from qt_app import MainWindow

app=QApplication([])
w=MainWindow();p=w.processing
with tempfile.TemporaryDirectory() as directory, \
        patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
    root=Path(directory);wav=root/'source.wav'
    sf.write(wav,np.zeros(4*16000),16000,subtype='PCM_16')
    p.set_audio(wav)
    assert w.improveSinging.isChecked()
    def proposals(path, progress, *, verify_vocals):
        return [dict(start=0.,end=4.,label='music',confidence=.8,auto_confidence=.8)]
    with patch('qt_processing.classify_file',side_effect=proposals) as classify:
        p.analyze()
        # Changing the control afterwards must not read a Qt widget on the worker.
        w.improveSinging.setChecked(False)
        deadline=time.monotonic()+10
        while p.task is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        assert p.task is None
        assert classify.call_args.kwargs['verify_vocals'] is True
        p.analyze()
        deadline=time.monotonic()+10
        while p.task is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        assert p.task is None
        assert classify.call_args.kwargs['verify_vocals'] is False
    project=root/'session.json'
    with patch.object(QFileDialog,'getSaveFileName',return_value=(str(project),'')):
        p.save_project()
    w.improveSinging.setChecked(True)
    with patch.object(QFileDialog,'getOpenFileName',return_value=(str(project),'')):
        p.open_project()
    assert not w.improveSinging.isChecked()
    p.dirty=False;w.close();app.processEvents()
print('PASS: singing verification enabled by default, checkbox captured on the Qt main thread, both modes passed to worker and setting restored from session.')

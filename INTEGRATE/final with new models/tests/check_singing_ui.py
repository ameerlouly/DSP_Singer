"""Follow the singing-detection checkbox from the Qt page into the worker and back through a saved session.

The proposals helper records which option the worker received.
"""

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
from speech_corpus_app import MainWindow

app=QApplication([])
w=MainWindow();p=w.processing
with tempfile.TemporaryDirectory() as directory, \
        patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
    root=Path(directory);wav=root/'source.wav'
    sf.write(wav,np.zeros(4*16000),16000,subtype='PCM_16')
    p.set_audio(wav)
    assert w.improveSinging.isChecked()
    # Remember the vocal setting passed to the worker and supply one music window.
    def proposals(path, progress, *, verify_vocals, window_seconds, hop_seconds):
        return [dict(start=0.,end=4.,label='music',confidence=.8,auto_confidence=.8)]
    with patch('audio_review_controller.classify_file',side_effect=proposals) as classify:
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
    # Process UI events until the analysis has finished.
    def wait():
        deadline=time.monotonic()+20
        while p.task is not None and time.monotonic()<deadline:
            app.processEvents()
            time.sleep(.01)
        assert p.task is None

    project=root/'session.json'
    with patch.object(QFileDialog,'getSaveFileName',return_value=(str(project),'')):
        p.save_project();wait()
    w.improveSinging.setChecked(True)
    with patch.object(QFileDialog,'getOpenFileName',return_value=(str(project),'')):
        p.open_project();wait()
    assert not w.improveSinging.isChecked()
    p.dirty=False;w.close();app.processEvents()
print('PASS: singing verification enabled by default, checkbox captured on the Qt main thread, both modes passed to worker and setting restored from session.')

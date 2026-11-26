"""Exercise the local MP3 and WAV pickers with real FFmpeg conversion and Qt workers."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
from speech_corpus_app import MainWindow


def check():
    """Open corpus and non-corpus recordings through both file pickers."""
    app = QApplication([])
    window = MainWindow(show_welcome=False)
    processing = window.processing

    def wait_for_worker():
        deadline = time.monotonic() + 20
        while window.worker is not None or processing.task is not None:
            assert time.monotonic() < deadline, 'Audio import timed out'
            app.processEvents()
            time.sleep(.01)

    def browse(button, path):
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(path), '')) as dialog:
            button.click()
        assert '*.mp3' in dialog.call_args.args[3] and '*.wav' in dialog.call_args.args[3]
        wait_for_worker()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stereo = root / 'محاضرة 1.WAV'
        mp3 = root / 'محاضرة 1.MP3'
        compatible = root / 'ready.wav'
        tone = .3 * np.sin(np.arange(44100 * 6) * (2 * np.pi * 440 / 44100))
        sf.write(stereo, np.column_stack((tone, tone)), 44100, subtype='FLOAT')
        sf.write(compatible, np.zeros(16000), 16000, subtype='PCM_16')
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(stereo), str(mp3)], check=True)
        for source in (mp3, stereo):
            original = source.read_bytes()
            window.channelBox.setCurrentText('Stereo')
            browse(window.chooseWav, source)
            wav = processing.audio
            assert wav.is_relative_to(root / 'Extract audio') and wav.with_suffix('.html').is_file()
            data, rate = sf.read(wav)
            assert (rate, data.shape, sf.info(wav).subtype) == (16000, (96000,), 'PCM_16')
            assert np.max(np.abs(data)) > .2 and source.read_bytes() == original
            assert Path(window.videoPathInput.text()) == source

        browse(window.chooseWav, compatible)
        assert processing.audio == compatible and not window.videoPathInput.text()
        processing.dirty = True
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            browse(window.chooseWav, mp3)
        assert processing.audio == compatible and not window.videoPathInput.text()
        processing.dirty = False
        damaged = root / 'damaged.mp3'
        damaged.write_bytes(b'invalid recording')
        with patch.object(QMessageBox, 'warning') as warning:
            browse(window.chooseWav, damaged)
        warning.assert_called_once()
        assert processing.audio == compatible and not window.videoPathInput.text()

        # Stop at labeling to check which recording Auto process hands to the next stage.
        for source in (mp3, stereo):
            browse(window.chooseVideoButton, source)
            with patch.object(QInputDialog, 'getText', return_value=('Reviewer', True)), patch.object(
                processing, 'analyze') as analyze:
                window.autoProcessButton.click()
                wait_for_worker()
            analyze.assert_called_once_with(confirmed=True, approval_editor='Reviewer')
            assert processing.audio == window.extracted_pair[2] and window.extracted_pair[0] == source
            assert sf.info(processing.audio).samplerate == 16000
            processing.auto_stage = None
        window.close()
    print('PASS: MP3, floating-point stereo WAV, direct PCM WAV, Auto process, cancellation and invalid audio.')


if __name__ == '__main__':
    check()

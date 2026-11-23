"""Slider/range preview workflow without requiring an audio device or codec."""
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtMultimedia import QMediaPlayer
from qt_app import MainWindow


class Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in self.callbacks:
            callback(*args)


class Player:
    MediaStatus = QMediaPlayer.MediaStatus

    def __init__(self, parent):
        self.mediaStatusChanged = Signal()
        self.positionChanged = Signal()
        self.errorOccurred = Signal()
        self._source = None
        self._position = 0
        self.status = self.MediaStatus.NoMedia
        self.state = 'stopped'

    def setAudioOutput(self, output):
        pass

    def source(self):
        return self._source

    def setSource(self, source):
        self._source = source
        self.status = self.MediaStatus.LoadingMedia
        self.setPosition(0)

    def loaded(self):
        self.status = self.MediaStatus.LoadedMedia
        self.mediaStatusChanged.emit(self.status)

    def mediaStatus(self):
        return self.status

    def position(self):
        return self._position

    def setPosition(self, value):
        self._position = value
        self.positionChanged.emit(value)

    def play(self):
        self.state = 'playing'

    def pause(self):
        self.state = 'paused'

    def stop(self):
        self.state = 'stopped'
        self.setPosition(0)


app = QApplication([])
window = MainWindow()
p = window.processing
warnings = []
with tempfile.TemporaryDirectory() as directory, \
        patch('qt_processing.QMediaPlayer', Player), \
        patch('qt_processing.QAudioOutput', lambda parent: object()), \
        patch.object(QMessageBox, 'warning', side_effect=lambda *args: warnings.append(args)):
    root = Path(directory)
    source = root / 'source.wav'
    other = root / 'candidate.wav'
    sf.write(source, np.zeros(190 * 16000), 16000, subtype='PCM_16')
    sf.write(other, np.zeros(10 * 16000), 16000, subtype='PCM_16')
    assert not window.regionSeekSlider.isEnabled()
    assert p.set_audio(source)
    p.regions = [dict(start=0., end=12., label='silence'),
                 dict(start=12., end=180., label='speech'),
                 dict(start=180., end=190., label='music')]
    p.refresh_all()
    window.regionTable.selectRow(1)
    slider = window.regionSeekSlider
    assert slider.isEnabled() and (slider.minimum(), slider.maximum()) == (12000, 180000)
    assert 'Duration 168.000 s' in window.regionRangeSummary.text()
    slider.setValue(45000)
    window.markRegionStart.click()
    slider.setValue(60000)
    window.markRegionEnd.click()
    assert (window.regionStart.value(), window.regionEnd.value()) == (45, 60)
    assert 'Listening range 00:45.000 – 01:00.000' in window.regionRangeSummary.text()
    assert 'Duration 15.000 s' in window.regionRangeSummary.text()
    assert len(p.regions) == 3 and not p.dirty  # Marking is a preview, not a label edit.

    window.playEnteredRegionRange.click()
    assert p.play_range == (45000, 60000)
    assert p.player.status == Player.MediaStatus.LoadingMedia
    # Seeking while the source is still loading must survive media_ready.
    slider.setValue(48000)
    p.player.loaded()
    assert p.player.position() == 48000 and p.player.state == 'playing'
    p.player.setPosition(50000)
    assert slider.value() == 50000 and window.regionCursorLabel.text() == 'Position 00:50.000'
    window.pauseRegionPreview.click()
    assert p.player.state == 'paused' and p.play_paused
    slider.setValue(55000)
    assert p.player.position() == 55000 and p.player.state == 'paused'
    window.pauseRegionPreview.click()
    assert p.player.state == 'playing' and not p.play_paused

    # Moving the handle previews the new time; release commits a seek.
    slider.setSliderDown(True)
    slider.setValue(56000)
    p.player.setPosition(55100)
    assert slider.value() == 56000 and p.player.position() == 55100
    slider.setSliderDown(False)
    assert p.player.position() == 56000
    p.player.setPosition(60000)
    assert p.play_range is None and p.player.state == 'stopped'
    assert slider.value() == 60000  # Stop must not snap the cursor to zero.

    slider.setValue(70000)
    window.playRegionCursor.click()
    assert p.play_range == (70000, 180000)
    assert p.player.state == 'playing'
    window.stopPlayback.click()
    assert p.play_range is None and slider.value() == 70000

    # Other pages use the same player; their clip-local times must not move this slider.
    p.play(other, 0, 10)
    p.player.loaded()
    p.player.setPosition(2000)
    assert slider.value() == 70000
    window.pauseRegionPreview.click()
    assert p.play_path == source and p.play_range == (70000, 180000)
    p.player.loaded()
    window.regionTable.selectRow(2)
    assert p.play_range is None
    assert (slider.minimum(), slider.maximum(), slider.value()) == (180000, 190000, 180000)

    # Invalid marked ranges are explained and rejected before starting playback.
    window.regionStart.setValue(185)
    window.regionEnd.setValue(184)
    assert 'start before the end' in window.regionRangeSummary.text()
    window.playEnteredRegionRange.click()
    assert len(warnings) == 1 and p.play_range is None
    warnings.clear()
    assert p.clock(3_661_250) == '01:01:01.250'
    p.dirty = False
    window.close()
    app.processEvents()
assert not warnings, warnings
print('PASS: Part 3 slider bounds, marked start/end and duration, asynchronous seek, follow-playback, pause/resume, drag/release, range-end stop, cursor retention, cross-page audio isolation and invalid-range warnings.')
print('The media player is a controlled test double; actual Windows audio-device playback was not exercised.')

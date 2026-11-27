"""Exercise the listening controls for candidates and unresolved spans.

A simulated player supplies position and load events so the test can seek, mark
boundaries and stop playback at a known time.
"""

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
from speech_corpus_app import MainWindow


class Signal:
    # Start with no listeners attached to this signal.
    def __init__(self):
        self.callbacks = []

    # Remember which callback should receive a player event.
    def connect(self, callback):
        self.callbacks.append(callback)

    # Deliver this event to all the connected callbacks.
    def emit(self, *args):
        for callback in self.callbacks:
            callback(*args)


class Player:
    MediaStatus = QMediaPlayer.MediaStatus

    # Set up a player with a known source, clock and playback state.
    def __init__(self, parent):
        self.mediaStatusChanged = Signal()
        self.positionChanged = Signal()
        self.errorOccurred = Signal()
        self._source = None
        self._position = 0
        self.status = self.MediaStatus.NoMedia
        self.state = 'stopped'

    # Accept the audio output connection without using a real device.
    def setAudioOutput(self, output):
        pass

    # Return the URL the simulated player is using.
    def source(self):
        return self._source

    # Changing the simulated source also moves its cursor to the beginning.
    def setSource(self, source):
        self._source = source
        self.status = self.MediaStatus.LoadingMedia
        self.setPosition(0)

    # Tell the controller that the simulated media has finished loading.
    def loaded(self):
        self.status = self.MediaStatus.LoadedMedia
        self.mediaStatusChanged.emit(self.status)

    # Report the simulated load status to the controller.
    def mediaStatus(self):
        return self.status

    # Read the test player's clock in milliseconds.
    def position(self):
        return self._position

    # Move the clock and notify the controller, as a seek would do.
    def setPosition(self, value):
        self._position = value
        self.positionChanged.emit(value)

    # Make the simulated player enter its playing state.
    def play(self):
        self.state = 'playing'

    # Leave the cursor in place while simulated playback is paused.
    def pause(self):
        self.state = 'paused'

    # Stop the simulated player and rewind its clock.
    def stop(self):
        self.state = 'stopped'
        self.setPosition(0)


app = QApplication([])
w = MainWindow(show_welcome=False)
p = w.processing
warnings = []
with tempfile.TemporaryDirectory() as directory, \
        patch('audio_review_controller.QMediaPlayer', Player), \
        patch('audio_review_controller.QAudioOutput', lambda parent: object()), \
        patch.object(QMessageBox, 'warning', side_effect=lambda *args: warnings.append(args)):
    root = Path(directory)
    source, clip = root / 'source.wav', root / 'compact.wav'
    sf.write(source, np.zeros(60 * 16000), 16000, subtype='PCM_16')
    sf.write(clip, np.zeros(10 * 16000), 16000, subtype='PCM_16')
    assert not w.segmentSeekSlider.isEnabled() and not w.unresolvedSeekSlider.isEnabled()
    p.set_audio(source)
    w.reviewer.setText('Slider reviewer')
    p.regions = [dict(start=0., end=60., label='speech')]
    p.refresh_all()
    w.regionTable.selectRow(0)
    original = dict(start=10., end=35., duration=10., label='speech', file='compact.wav',
                    source_spans=[dict(start=10.,end=15.,clip_start=0.,clip_end=5.),
                                  dict(start=30.,end=35.,clip_start=5.,clip_end=10.)])
    p.set_session(dict(source=dict(audio_name=source.name, audio_size=source.stat().st_size,
                                  duration_seconds=60.), candidate_root=str(root), changes=[],
                       segments=[dict(id='joined', original=original, candidate_file='compact.wav',
                                      source_start=10.,source_end=35.,label='speech',
                                      status='approved',source_mode='original_candidate', reviewed_by='Old')],
                       unresolved=[dict(id='long-gap',original=dict(start=35.,end=60.,reason='No real pause'),
                                        label='unclassified',status='pending',reviewed_by=None)]))
    w.segmentTable.selectRow(0)
    slider = w.segmentSeekSlider
    assert (slider.minimum(), slider.maximum(), slider.value()) == (0, 10000, 0)
    assert 'Clip time (joins preserved)' in w.segmentRangeSummary.text()
    # Marking is non-destructive until Apply edit.
    slider.setValue(2000); w.markSegmentStart.click()
    slider.setValue(8000); w.markSegmentEnd.click()
    assert (w.segmentStart.value(), w.segmentEnd.value()) == (2, 8)
    assert 'Duration 6.000 s' in w.segmentRangeSummary.text()
    assert p.session['segments'][0]['status'] == 'approved'
    w.playSegment.click()
    assert p.play_path == clip and p.play_range == (2000, 8000)
    slider.setValue(3000)
    p.player.loaded()
    assert p.player.position() == 3000 and p.player.state == 'playing'
    p.player.setPosition(4000)
    assert slider.value() == 4000 and w.regionSeekSlider.value() == 0
    w.pauseSegmentPreview.click()
    slider.setValue(5000)
    assert p.player.position() == 5000 and p.player.state == 'paused'
    w.pauseSegmentPreview.click()
    slider.setSliderDown(True); slider.setValue(6000)
    p.player.setPosition(5100)
    assert slider.value() == 6000
    slider.setSliderDown(False)
    assert p.player.position() == 6000
    p.player.setPosition(8000)
    assert p.play_range is None and slider.value() == 8000
    w.applySegment.click()
    item = p.session['segments'][0]
    assert (item['clip_trim_start'],item['clip_trim_end']) == (2,8)
    assert (item['source_start'],item['source_end']) == (12,33)
    assert item['source_mode'] == 'trimmed_candidate' and item['status'] == 'pending'
    from segment_review import candidate_source_spans
    assert [(s['start'],s['end']) for s in candidate_source_spans(item)] == [(12,15),(30,33)]
    # Switching modes resets the input values to the correct coordinate system.
    w.boundaryMode.setCurrentIndex(1)
    assert (w.segmentStart.value(), w.segmentEnd.value()) == (12,33)
    assert 'Source time' in w.segmentRangeSummary.text()
    assert (slider.minimum(), slider.maximum()) == (10000,35000)
    slider.setValue(12000); w.markSegmentStart.click()
    slider.setValue(18000); w.markSegmentEnd.click()
    w.playSegment.click()
    assert p.play_path == source and p.play_range == (12000,18000)
    p.player.loaded()
    p.player.setPosition(14000)
    assert slider.value() == 14000 and w.regionSeekSlider.value() == 0
    w.applySegment.click()
    item = p.session['segments'][0]
    assert item['source_mode'] == 'continuous_source'
    assert (item['source_start'],item['source_end']) == (12,18)
    assert not any(e['action']=='edit_segment' for e in p.session['changes'])
    # Unresolved spans can be much longer than clips and keep source coordinates.
    w.unresolvedTable.selectRow(0)
    gap = w.unresolvedSeekSlider
    assert (gap.minimum(),gap.maximum(),gap.value()) == (35000,60000,35000)
    gap.setValue(40000); w.markUnresolvedStart.click()
    gap.setValue(48000); w.markUnresolvedEnd.click()
    w.playUnresolved.click()
    p.player.loaded()
    assert p.play_path == source and p.play_range == (40000,48000)
    p.player.setPosition(42000)
    assert gap.value() == 42000 and slider.value() == 12000
    w.pauseUnresolvedPreview.click()
    gap.setValue(45000)
    assert p.player.state == 'paused' and p.player.position() == 45000
    w.pauseUnresolvedPreview.click()
    # A selected subrange creates a normal pending clip without losing its parent span.
    # A local edit must not redraw unrelated pages or replace untouched table items.
    original_cell = w.segmentTable.item(0, 0)
    original_region_cell = w.regionTable.item(0, 0)
    p.snr_snapshot = p.session.copy()
    with patch.object(p, 'refresh_all', side_effect=AssertionError('Whole workspace refresh')), \
            patch.object(p, 'refresh_snr', wraps=p.refresh_snr) as snr_draw:
        w.addManualClip.click()
        assert p.snr_snapshot is None and p.snr_view_dirty
        snr_draw.assert_not_called()
        w.navigationList.setCurrentRow(3)
        snr_draw.assert_called_once()
        assert not p.snr_view_dirty
        w.navigationList.setCurrentRow(2)
    assert w.segmentTable.item(0, 0) is original_cell and w.regionTable.item(0, 0) is original_region_cell
    assert (w.unresolvedStart.value(), w.unresolvedEnd.value()) == (48, 60)
    assert p.selected(w.unresolvedTable) == [1]
    assert (p.session['segments'][-1]['source_start'],p.session['segments'][-1]['source_end']) == (40,48)
    assert len(p.session['unresolved']) == 1
    assert w.unresolvedTable.rowCount() == 2
    assert [(w.unresolvedTable.item(i,1).text(),w.unresolvedTable.item(i,2).text()) for i in range(2)] == [('35.0000','40.0000'),('48.0000','60.0000')]
    w.unresolvedTable.selectRow(0)
    w.unresolvedLabel.setCurrentText('mixed')
    w.markUnresolved.click()
    assert p.session['unresolved'][0]['status'] == 'reviewed'
    assert p.session['unresolved'][0]['label'] == 'mixed'
    w.segmentTable.selectRow(len(p.session['segments'])-1)
    w.excludeSegments.click()
    assert [(w.unresolvedTable.item(i,1).text(),w.unresolvedTable.item(i,2).text()) for i in range(2)] == [('35.0000','40.0000'),('40.0000','60.0000')]
    # No active selection disables playback; invalid ranges do not start audio.
    w.unresolvedTable.selectRow(0)
    w.unresolvedStart.setValue(47); w.unresolvedEnd.setValue(46)
    w.playUnresolved.click()
    assert len(warnings) == 1
    warnings.clear()
    from PyQt6.QtCore import QItemSelectionModel
    w.segmentTable.selectRow(0)
    w.segmentTable.selectionModel().select(w.segmentTable.model().index(1,0),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    assert not slider.isEnabled()
    w.unresolvedTable.clearSelection()
    assert not gap.isEnabled()
    assert p.play_range is None
    p.dirty = False
    w.close(); app.processEvents()
assert not warnings, warnings
print('PASS: Part 4 candidate/source time mapping, compact joins, seek/loading, pause/resume, drag, endpoint marks, apply edit, unresolved subrange playback, manual creation, preserved leftovers, selection isolation, and invalid ranges.')
print('The player was controlled for repeatable checks; actual Windows audio playback was not exercised.')

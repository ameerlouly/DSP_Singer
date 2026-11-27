"""Exercise ASR playback and the shared player using a simulated media clock.

These tests execute the real controller methods without requiring Qt or audio devices.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import uuid
import xml.etree.ElementTree as ET
import numpy as np
import soundfile as sf
import corpus_review_and_export as corpus
import segment_review as review

ROOT = Path(__file__).resolve().parents[1]
STATUS = SimpleNamespace(LoadedMedia='loaded', BufferedMedia='buffered',
                         LoadingMedia='loading', EndOfMedia='end')


def methods(filename, cls, names, env):
    tree = ast.parse((ROOT/filename).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), env)
    return env[cls]


class Slider:
    def __init__(self):
        self.n = self.max = 0
        self.down = self.blocked = self.enabled = False
        self.changed = None
    def setRange(self, lo, hi):
        self.max = hi; self.setValue(self.n)
    def maximum(self): return self.max
    def value(self): return self.n
    def isSliderDown(self): return self.down
    def blockSignals(self, flag): self.blocked = flag
    def setEnabled(self, flag): self.enabled = flag
    def setValue(self, n):
        n = min(self.max, max(0, n))
        if n != self.n:
            self.n = n
            if not self.blocked and self.changed: self.changed(n)


class Signal:
    def __init__(self): self.callbacks = []
    def connect(self, cb): self.callbacks.append(cb)
    def emit(self, *args):
        for cb in self.callbacks: cb(*args)


class Player:
    MediaStatus = STATUS
    def __init__(self, parent):
        self.positionChanged = Signal(); self.mediaStatusChanged = Signal(); self.errorOccurred = Signal()
        self.status = STATUS.LoadingMedia; self.state = 'stopped'; self.pos = 0; self.url = None
    def setAudioOutput(self, output): pass
    def source(self): return self.url
    def mediaStatus(self): return self.status
    def position(self): return self.pos
    def setPosition(self, pos): self.pos = pos; self.positionChanged.emit(pos)
    def setSource(self, url): self.url = url; self.status = STATUS.LoadingMedia; self.setPosition(0)
    def loaded(self): self.status = STATUS.LoadedMedia; self.mediaStatusChanged.emit(self.status)
    def play(self): self.state = 'playing'
    def pause(self): self.state = 'paused'
    def stop(self): self.state = 'stopped'; self.setPosition(0)


ASR = methods('transcription_review_controller.py', 'ASRWorkspace', {
    'asr_preview_spec','configure_asr_preview','update_asr_cursor_label','asr_slider_changed',
    'play_asr_cursor','seek_asr_preview','pause_asr_preview','update_asr_playback_position',
    'asr_playback_stopped','stop_asr_preview'}, dict(corpus=corpus, QMediaPlayer=Player, Path=Path, sf=sf, uuid=uuid))
Processing = methods('audio_review_controller.py','ProcessingWorkspace', {
    'play','media_ready','playback_position','stop_playback','clock','enforce_playback_end'},
    dict(Path=Path,review=review,QMediaPlayer=Player,QAudioOutput=lambda p:None,
         QUrl=SimpleNamespace(fromLocalFile=lambda path:path)))


class TranscriptionPlaybackTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root/'source.wav'
        self.samples = np.arange(60*16000, dtype=np.int32).astype(np.int16)
        sf.write(self.path, self.samples, 16000, subtype='PCM_16')
        self.record = dict(segment_id='unresolved', audio_pieces=[dict(path=str(self.path),start=30,end=52)])
        self.slider = Slider()
        p = self.p = Processing()
        p.player=None; p.play_owner=None; p.play_range=None; p.play_timer=Mock(); p.audio=self.path
        p.part4_previews={'segment':None,'unresolved':None}; p.update_region_playback_position=Mock(); p.update_part4_playback_position=Mock()
        p.status=Mock(); p.guard=lambda f:f()
        a=self.a=ASR(); p.asr=a; a.p=p; a.asr_preview=None; a.asr_preview_file=None
        a.playback_dir=SimpleNamespace(name=tmp.name); a.selected=lambda table:[self.record]
        a.ui=p.ui=SimpleNamespace(asrSeekSlider=self.slider,asrPlay=Mock(),asrPausePreview=Mock(),asrStopPreview=Mock(),
                 asrSeekEnd=Mock(),asrCursorLabel=Mock(),asrTable=object(),pauseRegionPreview=Mock(),
                 pauseSegmentPreview=Mock(),pauseUnresolvedPreview=Mock())
        self.slider.changed=a.asr_slider_changed
        a.configure_asr_preview(self.record)

    def test_relative_seek_pause_resume_and_end(self):
        p,a,s=self.p,self.a,self.slider
        self.assertEqual((s.value(),s.maximum()),(0,22000))
        s.setValue(3000); a.play_asr_cursor()
        self.assertEqual(p.play_range,(33000,52000)); self.assertEqual(p.play_owner,'asr')
        self.assertEqual(s.value(),3000)  # ignore loading's position=0
        p.player.loaded(); self.assertEqual(p.player.state,'playing')
        self.assertEqual(p.player.position(),33000)
        p.player.setPosition(34500); self.assertEqual(s.value(),4500)
        a.pause_asr_preview(); self.assertEqual(p.player.state,'paused')
        s.setValue(6000); self.assertEqual(p.player.position(),36000); self.assertEqual(p.player.state,'paused')
        a.pause_asr_preview(); self.assertEqual(p.player.state,'playing')
        s.down=True; s.setValue(8000); p.player.setPosition(37000)
        self.assertEqual(s.value(),8000); s.down=False; a.seek_asr_preview()
        self.assertEqual(p.player.position(),38000)
        p.player.setPosition(52000); self.assertIsNone(p.play_range); self.assertEqual(s.value(),22000)
        self.assertEqual(p.player.state,'stopped')
        a.play_asr_cursor(); self.assertEqual(p.play_range,(30000,52000))  # replay from start
        a.stop_asr_preview(); self.assertEqual(s.value(),0); self.assertIsNone(p.play_owner)

    def test_selection_change_and_no_selection(self):
        self.a.play_asr_cursor(); self.p.player.loaded(); self.p.player.setPosition(35000)
        self.a.configure_asr_preview(self.record); self.assertEqual(self.slider.value(),5000)
        other=dict(segment_id='next',audio_pieces=[dict(path=str(self.path),start=10,end=18)])
        self.a.configure_asr_preview(other)
        self.assertIsNone(self.p.play_owner); self.assertEqual((self.slider.value(),self.slider.maximum()),(0,8000))
        self.a.configure_asr_preview(None); self.assertFalse(self.slider.enabled)
        self.assertEqual(self.slider.maximum(),0)

    def test_seek_and_pause_before_media_loaded(self):
        self.a.play_asr_cursor(); self.a.pause_asr_preview(); self.slider.setValue(12000)
        self.p.player.loaded(); self.assertEqual(self.p.player.position(),42000)
        self.assertNotEqual(self.p.player.state,'playing')
        self.a.pause_asr_preview(); self.assertEqual(self.p.player.state,'playing')

    def test_legacy_multi_piece_audio_exact_and_reused(self):
        self.record=dict(segment_id='joined',audio_pieces=[dict(path=str(self.path),start=3,end=5),dict(path=str(self.path),start=10,end=15)])
        self.a.configure_asr_preview(self.record); self.assertEqual(self.slider.maximum(),7000)
        self.a.play_asr_cursor(); self.p.player.loaded(); path=self.a.asr_preview_file
        actual,rate=sf.read(path,dtype='int16'); self.assertEqual(rate,16000)
        np.testing.assert_array_equal(actual,np.concatenate([self.samples[48000:80000],self.samples[160000:240000]]))
        self.a.stop_asr_preview(); self.a.play_asr_cursor(); self.assertEqual(path,self.a.asr_preview_file)

    def test_end_of_media_and_other_player_owner(self):
        self.a.play_asr_cursor(); self.p.player.loaded(); self.p.media_ready(STATUS.EndOfMedia)
        self.assertEqual(self.slider.value(),22000); self.assertIsNone(self.p.play_owner)
        self.p.play_owner='segment'; self.p.play_range=(0,9000)
        self.a.stop_asr_preview(); self.assertEqual(self.p.play_owner,'segment')
        self.assertEqual(self.slider.value(),0)

    def test_ui_uses_existing_asr_play_and_distinct_asr_stop(self):
        tree=ET.parse(ROOT/'main_window.ui')
        editor=tree.find('.//widget[@name="transcriptEditor"]')
        for name in ('asrSeekSlider','asrCursorLabel','asrSeekEnd','asrPlay','asrPausePreview','asrStopPreview'):
            self.assertEqual(len(tree.findall(f'.//widget[@name="{name}"]')),1)
            self.assertIsNotNone(editor.find(f'.//widget[@name="{name}"]'))
        self.assertIsNotNone(tree.find('.//widget[@name="asrStop"]'))
        text=(ROOT/'transcription_review_controller.py').read_text(encoding='utf-8')
        self.assertIn('self.ui.asrStop.clicked.connect(self.cancel)',text)
        self.assertIn('self.ui.asrSeekSlider.sliderReleased.connect',text)


if __name__=='__main__': unittest.main()

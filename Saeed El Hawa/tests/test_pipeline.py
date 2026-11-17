import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import wave
from pipeline import extract_audio, download_video


class PipelineTests(unittest.TestCase):
    def test_real_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / 'lecture.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'color=c=blue:s=160x120:r=10', '-f', 'lavfi', '-i',
                            'sine=frequency=440:sample_rate=48000', '-t', '2',
                            '-ac', '2', '-c:v', 'mpeg4', '-c:a', 'aac', str(video)], check=True)
            result = extract_audio(video, log=lambda _: None)
            with wave.open(str(result), 'rb') as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (16000, 1, 2))
                self.assertAlmostEqual(wav.getnframes()/16000, 2, delta=0.1)
                self.assertTrue(any(wav.readframes(16000)))
            self.assertEqual(json.loads(result.with_suffix('.json').read_text())['codec'], 'pcm_s16le')
            self.assertTrue(video.exists())
            self.assertFalse(list(Path(temp).glob('*.partial.wav')))

    def test_no_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / 'silent.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'color=s=160x120:r=10', '-t', '1', '-c:v', 'mpeg4', str(video)], check=True)
            with self.assertRaisesRegex(ValueError, 'no audio'):
                extract_audio(video)

    def test_missing_file(self):
        with self.assertRaisesRegex(ValueError, 'existing video'):
            extract_audio('nonexistent-video.mp4')

    def test_invalid_url(self):
        with self.assertRaisesRegex(ValueError, 'URL'):
            download_video('not a URL')


if __name__ == '__main__':
    unittest.main()

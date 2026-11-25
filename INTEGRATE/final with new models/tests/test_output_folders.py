"""Check timestamp collisions, repeated extractions and legacy project paths."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from output_folders import metadata_folder, new_run_folder, project_folder, save_dialog_folder
from video_download_and_audio_extraction import extract_audio, rename_video_folder


class OutputFolderTests(unittest.TestCase):
    def test_simultaneous_saves_and_cancelled_dialog(self):
        with tempfile.TemporaryDirectory() as directory, patch('output_folders.datetime') as clock:
            clock.now.return_value = datetime(2026, 11, 25, 14, 3, 7)
            with ThreadPoolExecutor(max_workers=4) as pool:
                folders = list(pool.map(new_run_folder, [directory] * 4))
            stamp = '2026-11-25_14-03-07'
            self.assertEqual({p.name for p in folders}, {stamp, stamp+'_02', stamp+'_03', stamp+'_04'})
            with save_dialog_folder(directory) as unused:
                self.assertTrue(unused.is_dir())
            self.assertFalse(unused.exists())
            self.assertTrue(all(p.is_dir() for p in folders))

    def test_legacy_and_dated_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'video1'
            run = new_run_folder(root / 'Download video')
            (run / 'metadata.html').write_text('metadata')
            self.assertEqual(metadata_folder(root), run)
            self.assertEqual(project_folder(run), root)
            self.assertEqual(project_folder(root / 'Extract audio'), root)
            self.assertEqual(project_folder(root), root)
            renamed = rename_video_folder(root, 'Lecture')
            self.assertTrue((metadata_folder(renamed) / 'metadata.html').is_file())
            (renamed / 'metadata.json').write_text('{}')
            self.assertEqual(metadata_folder(renamed), renamed)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires FFmpeg')
    def test_extract_twice_keeps_both_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'video1'
            source = new_run_folder(root / 'Download video') / 'source.mkv'
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i',
                            'sine=frequency=440:duration=0.5', '-c:a', 'pcm_s16le', str(source)], check=True)
            first = extract_audio(source, log=lambda _: None)
            first_stat = first.stat()
            second = extract_audio(source, log=lambda _: None)
            self.assertNotEqual(first.parent, second.parent)
            self.assertEqual(first.stat().st_mtime_ns, first_stat.st_mtime_ns)
            for wav in (first, second):
                self.assertEqual(wav.parent.parent, root / 'Extract audio')
                self.assertEqual(project_folder(wav.parent), root)
                self.assertTrue(wav.with_suffix('.html').is_file())

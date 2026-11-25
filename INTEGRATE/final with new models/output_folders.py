"""Organize video outputs by processing step and local save time.

step_folder locates a named step. new_run_folder reserves a unique timestamp.
save_dialog_folder offers a timestamped default for an explicit Save As dialog.
project_folder and metadata_folder locate the source in new and older layouts.
"""
from contextlib import contextmanager
from datetime import datetime
from itertools import count
from pathlib import Path
import re

RUN_NAME = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?')


def step_folder(parent, step):
    """Accept either a project destination or the step directory itself."""
    parent = Path(parent)
    return parent if parent.name == step else parent / step


def new_run_folder(parent):
    """Reserve a local timestamp directory without reusing an earlier save."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    for n in count(1):
        folder = parent / (stamp if n == 1 else f'{stamp}_{n:02d}')
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder


@contextmanager
def save_dialog_folder(parent):
    """Remove the unused timestamp if the user cancels or selects another path."""
    folder = new_run_folder(parent)
    try:
        yield folder
    finally:
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()


def project_folder(media_folder):
    """Locate the project for downloaded video or an extracted or standalone WAV."""
    folder = Path(media_folder)
    if RUN_NAME.fullmatch(folder.name) and folder.parent.name in ('Download video', 'Extract audio'):
        return folder.parent.parent
    return folder.parent if folder.name == 'Extract audio' else folder


def metadata_folder(folder):
    """Locate the completed download's metadata, including older project folders."""
    folder = Path(folder)
    if (folder / 'metadata.json').is_file() or (folder / 'metadata.html').is_file():
        return folder
    downloads = folder / 'Download video'
    if downloads.is_dir():
        for run in sorted(downloads.iterdir(), reverse=True):
            if RUN_NAME.fullmatch(run.name) and ((run / 'metadata.json').is_file() or (run / 'metadata.html').is_file()):
                return run
    return folder

"""Keep processing failures readable, including exceptions with no message."""
from datetime import datetime
from pathlib import Path
import tempfile
import traceback


def describe_exception(exc):
    detail = str(exc).strip()
    name = type(exc).__name__
    if not detail:
        if isinstance(exc, MemoryError):
            detail = 'Not enough memory for this operation. Close other applications and retry the export from the saved review.'
        else:
            detail = 'The operation raised an exception without an error message.'
    notes = getattr(exc, '__notes__', ())
    return '\n'.join([f'{name}: {detail}', *map(str, notes)])


def exception_details(exc):
    return describe_exception(exc), ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def save_failure_log(message, trace='', directory=None, operation='Processing'):
    """Use the project folder if possible, otherwise a writable temporary folder."""
    name = f'processing_error_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}.log'
    for root in (directory, Path(tempfile.gettempdir()) / 'SpeechCorpusStudio'):
        if root is None:
            continue
        try:
            root = Path(root)
            root.mkdir(parents=True, exist_ok=True)
            path = root / name
            path.write_text(f'{operation}\n{message}\n\n{trace}', encoding='utf-8')
            return path
        except OSError:
            continue
    return None

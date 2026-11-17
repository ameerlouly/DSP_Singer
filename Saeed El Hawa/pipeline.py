"""Parts 1 and 2: download a video, preserve metadata, standardize audio."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import re
import shutil
import subprocess
import wave
from urllib.parse import urlparse
from html import escape


def require_tools():
    for name in ('ffmpeg', 'ffprobe'):
        if not shutil.which(name):
            raise RuntimeError(f'{name} was not found. Install FFmpeg and add its bin folder to PATH; restart the terminal.')


def save_html(path, data):
    """Save metadata as a styled, offline HTML report."""

    path = Path(path)
    is_audio = 'audio_file' in data

    heading = 'Audio report' if is_audio else 'Video report'
    subtitle = (
        'Audio format and extraction details'
        if is_audio
        else 'Downloaded video and source information'
    )

    title = (
        data.get('title')
        or data.get('audio_file')
        or data.get('video_file')
        or 'Untitled'
    )

    # Convert values into safe HTML text.
    def safe(value):
        if value is None or value == '':
            return 'Not reported'
        return escape(str(value))

    # Display duration as hours, minutes and seconds.
    def format_duration(value):
        try:
            seconds = max(0, int(float(value)))
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        except (TypeError, ValueError, OverflowError):
            return 'Not reported'

    # Select the information shown in the summary cards.
    if is_audio:
        channels = data.get('channels')

        channel_text = (
            'Mono' if channels == 1
            else 'Stereo' if channels == 2
            else channels
        )

        rate = data.get('sample_rate_hz')
        bits = data.get('bits_per_sample')

        summaries = [
            ('Duration', format_duration(data.get('duration_seconds'))),
            ('Sample rate', f'{rate} Hz' if rate is not None else None),
            ('Channels', channel_text),
            ('Bit depth', f'{bits}-bit' if bits is not None else None),
        ]
    else:
        summaries = [
            ('Duration', format_duration(data.get('duration_seconds'))),
            ('Channel', data.get('channel')),
            ('Video ID', data.get('video_id')),
            ('License', data.get('license')),
        ]

    cards_html = ''.join(
        f"""
        <div class="stat">
            <div class="stat-label">{safe(label)}</div>
            <div class="stat-value" dir="auto">{safe(value)}</div>
        </div>
        """
        for label, value in summaries
    )

    labels = {
        'input_url': 'Submitted link',
        'source_url': 'Original video link',
        'video_id': 'Video ID',
        'title': 'Title',
        'channel': 'Channel',
        'duration_seconds': 'Duration (seconds)',
        'license': 'License',
        'downloaded_at_utc': 'Downloaded at (UTC)',
        'verified_at_utc': 'Verified at (UTC)',
        'video_file': 'Video file',
        'audio_file': 'Audio file',
        'source_file': 'Source file',
        'sample_rate_hz': 'Sample rate (Hz)',
        'channels': 'Channels',
        'bits_per_sample': 'Bits per sample',
        'codec': 'Audio codec',
        'size_bytes': 'File size (bytes)',
        'ffmpeg_command': 'FFmpeg arguments',
    }

    rows = []
    technical_rows = []

    for key, value in data.items():
        label = labels.get(
            key,
            key.replace('_', ' ').capitalize()
        )

        if isinstance(value, (list, dict)):
            content = (
                '<pre>'
                + escape(json.dumps(value, ensure_ascii=False, indent=2))
                + '</pre>'
            )
        else:
            content = safe(value)

        # Make source links clickable.
        if key in ('input_url', 'source_url') and isinstance(value, str):
            try:
                parsed = urlparse(value)
                valid_link = (
                    parsed.scheme in ('http', 'https')
                    and bool(parsed.netloc)
                )
            except ValueError:
                valid_link = False

            if valid_link:
                content = (
                    f'<a href="{escape(value, quote=True)}" '
                    f'target="_blank" rel="noopener noreferrer">'
                    f'{safe(value)}</a>'
                )

        row = (
            '<tr>'
            f'<th scope="row">{safe(label)}</th>'
            f'<td dir="auto">{content}</td>'
            '</tr>'
        )

        # Keep the long FFmpeg command in a collapsible section.
        if key == 'ffmpeg_command':
            technical_rows.append(row)
        else:
            rows.append(row)

    rows_html = ''.join(rows)

    technical_html = ''
    if technical_rows:
        technical_html = f"""
        <details class="technical">
            <summary>Technical details</summary>
            <table aria-label="Technical details">
                <tbody>{''.join(technical_rows)}</tbody>
            </table>
        </details>
        """

    css = """
        :root {
            --ink: #173b40;
            --muted: #60767a;
            --accent: #147d72;
            --background: #f0f4f5;
            --border: #dde6e8;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            background: var(--background);
            color: var(--ink);
            font: 16px/1.6 "Segoe UI", Arial, sans-serif;
        }

        .hero {
            background: var(--ink);
            color: white;
            padding: 42px 24px;
        }

        .container {
            max-width: 1100px;
            margin: auto;
        }

        .eyebrow {
            color: #94d9cb;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 2px;
        }

        h1 {
            margin: 10px 0 6px;
            font-size: clamp(30px, 5vw, 42px);
            line-height: 1.2;
        }

        .subtitle {
            margin: 0;
            color: #c5d7d9;
        }

        main {
            max-width: 1100px;
            margin: 28px auto;
            padding: 0 24px;
        }

        .file-heading {
            margin-bottom: 22px;
        }

        .file-heading small {
            color: var(--muted);
            font-weight: 700;
            letter-spacing: 1px;
        }

        .file-heading h2 {
            margin: 5px 0;
            font-size: 24px;
            overflow-wrap: anywhere;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }

        .stat {
            background: white;
            border: 1px solid var(--border);
            border-top: 3px solid var(--accent);
            border-radius: 12px;
            padding: 20px;
        }

        .stat-label {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 7px;
        }

        .stat-value {
            color: var(--ink);
            font-size: 20px;
            font-weight: 600;
            overflow-wrap: anywhere;
        }

        .panel, .technical {
            background: white;
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            margin-bottom: 20px;
        }

        .panel-heading {
            padding: 20px 24px;
            border-bottom: 1px solid var(--border);
        }

        .panel-heading h2 {
            margin: 0;
            font-size: 19px;
        }

        .panel-heading p {
            margin: 4px 0 0;
            color: var(--muted);
            font-size: 14px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }

        th, td {
            padding: 15px 24px;
            text-align: start;
            vertical-align: top;
            border-bottom: 1px solid #edf1f3;
            overflow-wrap: anywhere;
        }

        th {
            width: 29%;
            background: #f9fbfb;
            color: var(--muted);
            font-size: 14px;
            font-weight: 600;
        }

        tr:last-child > * { border-bottom: none; }

        a {
            color: var(--accent);
            text-underline-offset: 3px;
        }

        a:hover { color: #0d574f; }

        a:focus-visible, summary:focus-visible {
            outline: 3px solid #147d72;
            outline-offset: 4px;
        }

        summary {
            padding: 18px 24px;
            cursor: pointer;
            font-weight: 600;
        }

        pre {
            margin: 0;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            font: 13px/1.6 Consolas, monospace;
        }

        footer {
            padding: 0 0 24px;
            color: var(--muted);
            font-size: 13px;
        }

        @media (max-width: 850px) {
            .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 520px) {
            .hero { padding: 28px 18px; }
            main { padding: 0 14px; }
            .stats { grid-template-columns: 1fr; }
            th, td { padding: 12px; }
            th { width: 38%; }
        }

        @media print {
            body { background: white; }
            .hero {
                background: white;
                color: #173b40;
                padding: 12px 24px;
            }
            .eyebrow, .subtitle { color: #60767a; }
            main { margin: 12px auto; }
            .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            tr, .stat { break-inside: avoid; }
        }
    """

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{safe(heading)} | Speech Corpus Studio</title>
    <style>{css}</style>
</head>
<body>
    <header class="hero">
        <div class="container">
            <div class="eyebrow">SPEECH CORPUS STUDIO</div>
            <h1>{safe(heading)}</h1>
            <p class="subtitle">{safe(subtitle)}</p>
        </div>
    </header>

    <main>
        <section class="file-heading">
            <small>FILE OVERVIEW</small>
            <h2 dir="auto">{safe(title)}</h2>
        </section>

        <section class="stats" aria-label="Summary">
            {cards_html}
        </section>

        <section class="panel">
            <div class="panel-heading">
                <h2>Metadata details</h2>
                <p>Recorded information for this file.</p>
            </div>

            <table aria-label="File metadata">
                <tbody>{rows_html}</tbody>
            </table>
        </section>

        {technical_html}

        <footer>
            Speech Corpus Studio · Local metadata report ·
            Missing values are shown as “Not reported”.
        </footer>
    </main>
</body>
</html>
"""

    temp = path.with_suffix('.html.tmp')
    temp.write_text(page, encoding='utf-8')
    temp.replace(path)

def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                             '-of', 'json', str(path)], capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError('Cannot read media: ' + result.stderr[-2000:])
    return json.loads(result.stdout)


def validate_folder_name(name):
    name = name.strip()
    if (not name or name in ('.', '..') or name.endswith('.')
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or re.fullmatch(r'CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]',
                            name.partition('.')[0].rstrip(), re.IGNORECASE)):
        raise ValueError('Enter a valid single folder name, such as Egyptian_Interview.')
    return name


def video_number(name):
    match = re.fullmatch(r'video([1-9][0-9]*)', name, re.IGNORECASE)
    return int(match[1]) if match else 0


def create_video_folder(output_root, folder_name):
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if folder_name and folder_name.strip():
        folder = root / validate_folder_name(folder_name)
        try:
            folder.mkdir()
        except FileExistsError as exc:
            raise ValueError(f'“{folder.name}” already exists. Choose a different folder name.') from exc
        return folder

    last_number = 0
    for entry in root.iterdir():
        last_number = max(last_number, video_number(entry.name))
        if not entry.is_dir() or entry.is_symlink():
            continue
        try:
            # A renamed folder remembers its original number.
            previous = int((entry / '.video-number').read_text(encoding='ascii'))
        except (OSError, ValueError):
            continue
        last_number = max(last_number, previous)

    while True:
        last_number += 1
        folder = root / f'video{last_number}'
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder


def rename_video_folder(folder, new_name):
    folder = Path(folder)
    new_name = validate_folder_name(new_name)
    if not folder.is_dir() or not (folder / 'metadata.html').is_file():
        raise ValueError('Choose a completed download folder.')
    if folder.name == new_name:
        return folder
    destination = folder.with_name(new_name)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f'“{new_name}” already exists. Choose a different folder name.')
    number = video_number(folder.name)
    if number:
        marker = folder / '.video-number'
        try:
            number = max(number, int(marker.read_text(encoding='ascii')))
        except (OSError, ValueError):
            pass
        marker.write_text(str(number), encoding='ascii')
    return folder.rename(destination)


def download_video(url, output_root='data', log=print, *, folder_name=None):
    require_tools()
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('Paste a complete http:// or https:// video URL.')
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError('Install dependencies: python -m pip install -r requirements.txt') from exc
    folder = create_video_folder(output_root, folder_name)
    log(f'Output folder: {folder}')
    last_percent = [-1]

    def progress(event):
        if event['status'] == 'downloading':
            total = event.get('total_bytes') or event.get('total_bytes_estimate')
            if total:
                percent = int(100 * event.get('downloaded_bytes', 0) / total)
                if percent // 10 != last_percent[0] // 10:
                    log(f'Download: {percent}%')
                    last_percent[0] = percent
        elif event['status'] == 'finished':
            log('Stream downloaded; completing media processing...')

    options = {
        'format': 'bv*+ba/b',
        'outtmpl': str(folder / 'source.%(ext)s'),
        'merge_output_format': 'mkv',
        'noplaylist': True,
        'quiet': True,
        'progress_hooks': [progress],
        'retries': 3,
        'socket_timeout': 30,
        'match_filter': lambda info, *, incomplete=False: (
            'Live streams are not supported; choose a finished video.'
            if info.get('is_live') else None),
    }
    log('Downloading video (this can take several minutes)...')
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
    if not info or info.get('_type') in ('playlist', 'multi_video'):
        raise ValueError('Use a single, finished video URL, not a playlist.')
    candidates = [p for p in folder.glob('source.*')
                  if p.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov', '.flv', '.avi', '.ts', '.m4v')]
    valid = []
    for candidate in candidates:
        streams = probe(candidate).get('streams', [])
        if all(any(s.get('codec_type') == kind for s in streams) for kind in ('video', 'audio')):
            valid.append(candidate)
    if not valid:
        raise RuntimeError('No completed video with both video and audio streams was found.')
    video = max(valid, key=lambda p: p.stat().st_mtime)
    metadata = {
        'input_url': url, 'source_url': info.get('webpage_url') or url,
        'video_id': info.get('id'), 'title': info.get('title'),
        'channel': info.get('channel') or info.get('uploader'),
        'duration_seconds': info.get('duration'),
        'license': info.get('license') or 'Unknown / not reported by source',
        'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
        'video_file': video.name,
    }
    save_html(folder / 'metadata.html', metadata)
    log(f'Saved video: {video}')
    log(f'License: {metadata["license"]}')
    if (metadata['duration_seconds'] or 0) < 3600:
        log('Note: this is fine for testing; final project videos must each be at least one hour.')
    return video


def extract_audio(video_path, log=print):
    require_tools()
    video = Path(video_path).resolve()
    if not video.is_file():
        raise ValueError('Select an existing video file first.')
    source = probe(video)
    audio = next((s for s in source.get('streams', []) if s.get('codec_type') == 'audio'), None)
    if audio is None:
        raise ValueError('This file contains no audio stream.')
    output = video.with_name(video.stem + '_16k_mono.wav')
    temporary = output.with_suffix('.partial.wav')
    log('Extracting the first audio track as 16 kHz, mono, 16-bit PCM...')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-i', str(video), '-map', '0:a:0', '-vn', '-ar', '16000', '-ac', '1',
               '-c:a', 'pcm_s16le', str(temporary)]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError('Audio extraction failed: ' + result.stderr[-3000:])
        with wave.open(str(temporary), 'rb') as wav:
            rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
            duration = wav.getnframes() / rate
            if (rate, channels, width, wav.getcomptype()) != (16000, 1, 2, 'NONE') or duration <= 0:
                raise RuntimeError('Output WAV failed audio format validation.')
        expected = audio.get('duration') or source.get('format', {}).get('duration')
        if expected and abs(float(expected) - duration) > max(2.0, float(expected) * 0.01):
            raise RuntimeError('Output duration differs unexpectedly from the source; inspect its timestamps.')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {'audio_file': output.name, 'source_file': video.name, 'sample_rate_hz': rate,
              'channels': channels, 'bits_per_sample': width * 8, 'codec': 'pcm_s16le',
              'duration_seconds': duration, 'size_bytes': output.stat().st_size,
              'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'ffmpeg_command': command}
    save_html(output.with_suffix('.html'), report)
    log(f'Verified: {rate} Hz | {channels} channel | {width * 8}-bit | {duration:.2f} seconds')
    log(f'Saved audio: {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--url', help='Download a video and extract its audio')
    group.add_argument('--video', help='Extract audio from an existing local video')
    parser.add_argument('--output', default='data', help='Download folder')
    parser.add_argument('--folder-name', help='Custom subfolder name for a URL download')
    args = parser.parse_args()
    if args.video and args.folder_name is not None:
        parser.error('--folder-name applies only to --url')
    try:
        video = download_video(args.url, args.output, folder_name=args.folder_name) if args.url else args.video
        extract_audio(video)
    except Exception as exc:
        parser.exit(1, f'Error: {exc}\n')

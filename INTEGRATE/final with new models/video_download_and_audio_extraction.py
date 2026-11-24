"""Download source video with metadata and convert its audio to a configurable PCM WAV.

Function guide
--------------
require_tools, probe
    Check media tools and inspect the source streams.
save_html, save_metadata_json, license_fields, recover_legacy_html_metadata,
update_license
    Record source metadata and license evidence and reopen older saved details.
validate_folder_name, video_number, create_video_folder, rename_video_folder
    Create or rename per-video folders with valid names.
DownloadProgress, download_video
    Combine stream byte counts and publish progress during the download.
DownloadProgress.positive, report, publish
    Validate stream sizes and report byte totals and percentage estimates.
DownloadProgress.__call__, finishing, postprocess
    Consume downloader hooks and reserve completion for the final processing stage.
WavFormat, extract_audio
    Validate the requested PCM format and extract audio with FFmpeg.
"""
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime, timezone
import argparse
import json
import math
import re
import shutil
import subprocess
from urllib.parse import urlparse
from html import escape, unescape


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
        'license_status': 'License verification status',
        'license_evidence': 'License evidence / notes',
        'license_verified_at_utc': 'License checked at (UTC)',
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


def save_metadata_json(path, data):
    """Save a machine-readable copy so the license can be reviewed later."""
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def license_fields(info, manual_license=None, license_evidence=None):
    """Never infer reuse rights when the site's license field is missing."""
    if manual_license is not None:
        manual_license = manual_license.strip()
        if not manual_license:
            raise ValueError('The manually entered license cannot be empty.')
        return {
            'license': manual_license,
            'license_status': 'Manually entered after source review (not independently validated)',
            'license_evidence': (license_evidence or 'User checked the original video page').strip(),
            'license_verified_at_utc': datetime.now(timezone.utc).isoformat(),
        }

    extracted = info.get('license')
    if isinstance(extracted, str) and extracted.strip():
        return {
            'license': extracted.strip(),
            'license_status': 'Provided by the website through yt-dlp',
            'license_evidence': 'yt-dlp metadata field: license',
        }

    return {
        'license': 'Not reported — manual verification required',
        'license_status': 'Unverified',
        'license_evidence': 'Check the source video page or request permission from its owner',
    }


def recover_legacy_html_metadata(report_path):
    """Read the original pipeline's HTML table, so old downloads need no re-download."""
    page = Path(report_path).read_text(encoding='utf-8')
    key_by_label = {
        'Submitted link': 'input_url',
        'Original video link': 'source_url',
        'Video ID': 'video_id',
        'Title': 'title',
        'Channel': 'channel',
        'Duration (seconds)': 'duration_seconds',
        'License': 'license',
        'Downloaded at (UTC)': 'downloaded_at_utc',
        'Video file': 'video_file',
    }
    record = {}
    for label, raw_value in re.findall(
            r'<tr>\s*<th scope="row">(.*?)</th>\s*<td[^>]*>(.*?)</td>\s*</tr>',
            page, flags=re.DOTALL):
        label = unescape(re.sub(r'<[^>]*>', '', label)).strip()
        if label in key_by_label:
            value = unescape(re.sub(r'<[^>]*>', '', raw_value)).strip()
            record[key_by_label[label]] = value
    if not record.get('source_url'):
        raise ValueError('Cannot recover source information from the legacy metadata.html file.')
    return record


def update_license(video_folder, license_name, evidence_note=None, log=print):
    """Record a manual license review without downloading the video again."""
    folder = Path(video_folder)
    record_path = folder / 'metadata.json'
    if record_path.is_file():
        data = json.loads(record_path.read_text(encoding='utf-8'))
    elif (folder / 'metadata.html').is_file():
        data = recover_legacy_html_metadata(folder / 'metadata.html')
    else:
        raise FileNotFoundError(f'No metadata.json or metadata.html exists in {folder}')
    if not isinstance(data, dict) or 'source_url' not in data:
        raise ValueError('This folder has invalid video metadata.')
    data.update(license_fields({}, license_name, evidence_note))
    save_metadata_json(record_path, data)
    save_html(folder / 'metadata.html', data)
    log(f'License recorded: {data["license"]}')
    log(f'Regenerated: {folder / "metadata.html"}')
    return data


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


class DownloadProgress:
    """Track combined video/audio bytes, marking estimated totals explicitly."""

    def __init__(self, info, callback, details_callback=None):
        formats = info.get('requested_formats') or [info]
        self.totals = {}
        self.estimated = {}
        self.downloaded = {}
        self.finished = set()
        self.callback = callback
        self.details_callback = details_callback
        self.last_value = -1
        self.last_reported = None
        for index, stream in enumerate(formats):
            ident = str(stream.get('format_id', index))
            size = self.positive(stream.get('filesize'))
            estimated = False
            if size is None:
                size = self.positive(stream.get('filesize_approx'))
                estimated = size is not None
            if size is None:
                rate = self.positive(stream.get('tbr'))
                duration = self.positive(stream.get('duration') or info.get('duration'))
                if rate is not None and duration is not None:
                    size = rate * 1000 / 8 * duration
                    estimated = True
            self.totals[ident] = size
            self.estimated[ident] = estimated
            self.downloaded[ident] = 0

    @staticmethod
    def positive(value):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number > 0 else None

    def report(self, value):
        # Estimates can change as HTTP headers/fragment sizes become available.
        # A corrected percentage may decrease rather than hiding that correction.
        if value != self.last_reported:
            self.last_reported = value
            if value >= 0:
                self.last_value = value
            if self.callback is not None:
                self.callback(value)
        if value == 100:
            self.publish('complete')

    def publish(self, state):
        known = all(size is not None for size in self.totals.values())
        if self.details_callback is not None:
            self.details_callback({
                'state': state,
                'downloaded_bytes': int(sum(self.downloaded.values())),
                'total_bytes': int(sum(self.totals.values())) if known else None,
                'total_is_estimate': known and any(self.estimated.values()),
            })

    def finishing(self):
        self.report(-1)
        self.publish('finishing')

    def __call__(self, event):
        status = event.get('status')
        if status not in ('downloading', 'finished'):
            return
        info = event.get('info_dict', {})
        ident = str(info.get('format_id', ''))
        if ident not in self.totals:
            if len(self.totals) == 1:
                ident = next(iter(self.totals))
            else:
                self.report(-1)
                self.publish('transferring')
                return
        count = event.get('downloaded_bytes')
        if count == 0:
            self.downloaded[ident] = 0
        elif self.positive(count) is not None:
            self.downloaded[ident] = int(count)
        exact = self.positive(event.get('total_bytes'))
        estimate = self.positive(event.get('total_bytes_estimate'))
        if exact is not None:
            self.totals[ident], self.estimated[ident] = exact, False
        elif estimate is not None and (self.totals[ident] is None or self.estimated[ident]):
            self.totals[ident], self.estimated[ident] = estimate, True
        if status == 'finished':
            self.finished.add(ident)
            # Successful completion gives the actual final transfer size.
            size = self.positive(event.get('downloaded_bytes'))
            if size is None and event.get('filename'):
                try:
                    size = self.positive(Path(event['filename']).stat().st_size)
                except OSError:
                    pass
            if size is None and not self.estimated[ident]:
                size = self.totals[ident]
            if size is not None:
                self.totals[ident], self.downloaded[ident], self.estimated[ident] = size, int(size), False
        if all(size is not None for size in self.totals.values()):
            total = sum(self.totals.values())
            downloaded = sum(self.downloaded.values())
            # Completion also requires successful merging, validation and reports.
            self.report(min(99, int(100 * downloaded / total)))
        else:
            # No denominator means no honest overall percentage is available.
            self.report(-1)
        self.publish('transferring')
        if len(self.finished) == len(self.totals):
            self.finishing()

    def postprocess(self, event):
        if event.get('status') == 'started':
            self.finishing()


def download_video(url, output_root='data', log=print, *, folder_name=None,
                   manual_license=None, license_evidence=None, progress_callback=None,
                   progress_details_callback=None):
    """Download a source; report byte-weighted progress and optional size details.

    progress_callback receives -1 when an overall percentage is unavailable or
    processing continues, otherwise 0..99; 100 follows successful validation.
    progress_details_callback receives byte counts, estimate flag and stage.
    """
    require_tools()
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('Paste a complete http:// or https:// video URL.')
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError('Install dependencies: python -m pip install -r requirements.txt') from exc
    hooks = []
    options = {
        'format': 'bv*+ba/b',
        'outtmpl': 'source.%(ext)s',
        'merge_output_format': 'mkv',
        'noplaylist': True,
        'quiet': True,
        'progress_hooks': hooks,
        'noprogress': True,
        'retries': 3,
        'socket_timeout': 30,
        'match_filter': lambda info, *, incomplete=False: (
            'Live streams are not supported; choose a finished video.'
            if info.get('is_live') else None),
    }
    if progress_callback is not None:
        progress_callback(-1)
    if progress_details_callback is not None:
        progress_details_callback(dict(state='reading', downloaded_bytes=0, total_bytes=None, total_is_estimate=False))
    log('Reading video information...')
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
        if not info or info.get('_type') in ('playlist', 'multi_video'):
            raise ValueError('Use a single, finished video URL, not a playlist.')
        if info.get('is_live'):
            raise ValueError('Choose a finished video instead of a live stream.')
        folder = create_video_folder(output_root, folder_name)
        log(f'Output folder: {folder}')
        # Configure the destination only after metadata validation succeeds.
        downloader.params['outtmpl']['default'] = str(folder / 'source.%(ext)s')
        progress = DownloadProgress(info, progress_callback, progress_details_callback)
        downloader.add_progress_hook(progress)
        downloader.add_postprocessor_hook(progress.postprocess)
        progress.report(0)
        log('Downloading selected video and audio streams...')
        info = downloader.process_ie_result(info, download=True)
    progress.finishing()
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
        **license_fields(info, manual_license, license_evidence),
        'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
        'video_file': video.name,
    }
    save_metadata_json(folder / 'metadata.json', metadata)
    save_html(folder / 'metadata.html', metadata)
    log(f'Saved video: {video}')
    log(f'License: {metadata["license"]}')
    if metadata['license_status'] == 'Unverified':
        log('WARNING: The license was not provided by the source. Manually check the video before using it in the final corpus.')
    if (metadata['duration_seconds'] or 0) < 3600:
        log('Note: this is fine for testing; final project videos must each be at least one hour.')
    progress.report(100)
    return video


PCM_CODECS = {8: 'pcm_u8', 16: 'pcm_s16le', 24: 'pcm_s24le', 32: 'pcm_s32le'}


@dataclass(frozen=True)
class WavFormat:
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16

    def __post_init__(self):
        if self.channels not in (1, 2) or self.bit_depth not in PCM_CODECS:
            raise ValueError('Choose mono or stereo and 8, 16, 24 or 32 bits per sample.')
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError('The sample rate must be a positive integer in Hz.')


CORPUS_FORMAT = WavFormat()


def extract_audio(video_path, log=print, *, wav_format=CORPUS_FORMAT):
    require_tools()
    video = Path(video_path).resolve()
    if not video.is_file():
        raise ValueError('Select an existing video file first.')
    source = probe(video)
    audio = next((s for s in source.get('streams', []) if s.get('codec_type') == 'audio'), None)
    if audio is None:
        raise ValueError('This file contains no audio stream.')
    rate, channels, bits = wav_format.sample_rate, wav_format.channels, wav_format.bit_depth
    codec = PCM_CODECS[bits]
    layout = 'mono' if channels == 1 else 'stereo'
    suffix = '_16k_mono' if wav_format == CORPUS_FORMAT else f'_{rate}Hz_{layout}_{bits}bit'
    output_folder = video.parent / 'Extract audio'
    output_folder.mkdir(parents=True, exist_ok=True)
    output = output_folder / f'{video.stem}{suffix}.wav'
    log(f'Extracting the first audio track as {rate} Hz, {layout}, {bits}-bit PCM...')
    with NamedTemporaryFile(dir=output.parent, prefix='.extract-', suffix='.wav', delete=False) as tmp:
        temporary = Path(tmp.name)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-i', str(video), '-map', '0:a:0', '-vn', '-ar', str(rate), '-ac', str(channels),
               '-c:a', codec]
    if wav_format != CORPUS_FORMAT:
        command.extend(['-rf64', 'auto'])
    command.append(str(temporary))
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError('Audio extraction failed: ' + result.stderr[-3000:])
        encoded = probe(temporary)
        stream = encoded['streams'][0]
        actual = (int(stream['sample_rate']), stream['channels'], stream['bits_per_sample'],
                  stream['codec_name'])
        duration = float(encoded['format'].get('duration', 0))
        if actual != (rate, channels, bits, codec) or duration <= 0:
            raise RuntimeError('Output WAV failed audio format validation.')
        expected = audio.get('duration') or source.get('format', {}).get('duration')
        if expected and abs(float(expected) - duration) > max(2.0, float(expected) * 0.01):
            raise RuntimeError('Output duration differs unexpectedly from the source; inspect its timestamps.')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {'audio_file': output.name, 'source_file': video.name, 'sample_rate_hz': rate,
              'channels': channels, 'bits_per_sample': bits, 'codec': codec,
              'duration_seconds': duration, 'size_bytes': output.stat().st_size,
              'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'ffmpeg_command': command}
    save_html(output.with_suffix('.html'), report)
    log(f'Verified: {rate} Hz | {layout} | {bits}-bit | {duration:.2f} seconds')
    log(f'Saved audio: {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--url', help='Download a video and extract its audio')
    group.add_argument('--video', help='Extract audio from an existing local video')
    group.add_argument('--verify-license', metavar='FOLDER',
                       help='Record a manually checked license for an existing download folder')
    parser.add_argument('--output', default='data', help='Download folder')
    parser.add_argument('--folder-name', help='Custom subfolder name for a URL download')
    parser.add_argument('--license-name',
                        help='License copied after checking the original video page')
    parser.add_argument('--license-evidence',
                        help='Notes or a link showing where the license was checked')
    args = parser.parse_args()
    if not args.url and args.folder_name is not None:
        parser.error('--folder-name applies only to --url')
    if args.video and (args.license_name or args.license_evidence):
        parser.error('--license-name and --license-evidence apply only to --url or --verify-license')
    if args.verify_license and not args.license_name:
        parser.error('--verify-license requires --license-name')
    if args.license_evidence and not args.license_name:
        parser.error('--license-evidence requires --license-name')
    try:
        if args.verify_license:
            update_license(args.verify_license, args.license_name, args.license_evidence)
        else:
            video = (download_video(args.url, args.output, folder_name=args.folder_name,
                                   manual_license=args.license_name,
                                   license_evidence=args.license_evidence)
                     if args.url else args.video)
            extract_audio(video)
    except Exception as exc:
        parser.exit(1, f'Error: {exc}\n')

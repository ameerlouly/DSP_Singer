"""Save a region timeline that can be read in a browser and reopened in the editor.

render_report builds the human-readable rows. save_report writes the report, and
load_review reads back the saved labels and review history.
"""

import base64
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

LABELS = ('speech', 'music', 'mixed', 'silence')
COLORS = {'speech': '#16a5cf', 'music': '#c5428c', 'mixed': '#da8a14', 'silence': '#7b8790'}


# Escape text before putting it in the HTML report.
def esc(value):
    return html.escape(str(value), quote=True)


# Display a missing score as a dash so it is distinct from a measured zero.
def number(value, digits=3):
    return '—' if value is None else f'{float(value):.{digits}f}'


def confidence_text(region):
    """Format confidence in the original automatic label, including legacy scores."""
    value = region.get('auto_confidence', None if region.get('manual_override') else region.get('confidence'))
    return '—' if value is None else f'{float(value):.3f} ({float(value):.1%})'


def render_report(payload):
    """Build a self-contained UTF-8 report including machine-readable review data."""
    regions = payload['regions']
    duration = float(payload['duration_seconds'])
    config = payload.get('analysis_settings')
    settings_text = (f'Window {number(config["window_seconds"])} s · Hop {number(config["hop_seconds"])} s'
                     if config else 'Not recorded (legacy or imported data)')
    counts = {label: 0 for label in LABELS}
    seconds = {label: 0.0 for label in LABELS}
    flagged = 0
    corrected = 0
    rows = []
    for i, region in enumerate(regions):
        label = region['label']
        length = float(region['end']) - float(region['start'])
        counts[label] += 1
        seconds[label] += length
        flagged += bool(region.get('needs_review'))
        corrected += bool(region.get('manual_override'))
        values = [i + 1, number(region['start']), number(region['end']), number(length), label,
                  region.get('auto_label', ''), confidence_text(region),
                  number(region.get('speech_score')), number(region.get('music_score')),
                  number(region.get('vocal_score')), region.get('vocal_evidence_class') or '—',
                  number(region.get('verified_voice_score')),
                  number(region.get('rms_dbfs'), 1),
                  'Yes' if region.get('needs_review') else 'No',
                  'Yes' if region.get('manual_override') else 'No', region.get('note', '')]
        rows.append('<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in values) + '</tr>')
    summary = []
    for label in LABELS:
        share = seconds[label] / duration * 100 if duration > 0 else 0
        summary.append(f'<tr><th>{esc(label.title())}</th><td>{counts[label]}</td>'
                       f'<td>{seconds[label]:.2f}</td><td>{share:.1f}%</td>'
                       f'<td><div class="track"><div class="bar" style="width:{max(0.0,min(100.0,share)):.2f}%;background:{COLORS[label]}"></div></div></td></tr>')
    changes = payload.get('changes', [])
    audit = []
    for change in changes:
        raw = json.dumps(change, ensure_ascii=False, indent=2)
        audit.append('<details><summary>' + esc(change.get('time_utc', '')) + ' — ' +
                     esc(change.get('annotator', '')) + ' — region ' +
                     esc(int(change.get('region_index', 0)) + 1) + '</summary><pre>' +
                     esc(raw) + '</pre></details>')
    if not audit:
        audit.append('<p>No manual edits were recorded.</p>')
    embedded = base64.b64encode(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                             separators=(',', ':')).encode('utf-8')).decode('ascii')
    headings = ['#', 'Start (s)', 'End (s)', 'Duration (s)', 'Current label', 'Automatic label',
                'Auto confidence', 'Speech score', 'Music score', 'Vocal score', 'Vocal category', 'Separated voice evidence',
                'RMS (dBFS)', 'Needs review', 'Manual edit', 'Notes']
    header = ''.join(f'<th>{esc(name)}</th>' for name in headings)
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speech corpus review — {esc(payload['audio_name'])}</title>
<style>
:root{{color-scheme:light;--ink:#1c3440;--line:#d9e4e8;--muted:#526b76;--accent:#167d77}}
body{{font:15px/1.5 Segoe UI,Arial,sans-serif;color:var(--ink);background:#f2f6f7;margin:0}}
header{{background:#173c42;color:white;padding:26px max(20px,calc((100% - 1300px)/2))}}h1{{margin:0;font-size:28px}}header p{{margin:5px 0 0;color:#c7dedf}}
main{{max-width:1300px;margin:auto;padding:18px}}section{{background:white;border:1px solid var(--line);border-radius:10px;margin-bottom:16px;padding:18px}}
h2{{margin:0 0 12px;font-size:19px}}.cards{{display:flex;gap:10px;flex-wrap:wrap}}.card{{background:#f3f8f8;border-radius:8px;padding:10px 14px;min-width:135px;flex:1}}.card strong{{display:block;font-size:20px}}small,.muted{{color:var(--muted)}}
.tablewrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}.regions-table{{table-layout:fixed;min-width:1400px}}.regions-table th{{white-space:normal;overflow-wrap:break-word}}.regions-table td{{overflow-wrap:anywhere}}.regions-table th,.regions-table td{{padding:8px 5px}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}thead th{{background:#eaf3f3;white-space:nowrap}}.track{{height:12px;background:#edf2f3;min-width:90px;border-radius:8px;overflow:hidden}}.bar{{height:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f5f8f9;padding:12px}}details{{border-top:1px solid var(--line);padding:8px 0}}summary{{cursor:pointer}}.scroll{{max-height:650px;overflow:auto}}footer{{color:var(--muted);padding:5px}}
</style></head><body><header><h1>Speech corpus review</h1><p>{esc(payload['audio_name'])} · Generated {generated}</p></header>
<main><section><h2>Recording and review</h2><div class="cards">
<div class="card"><small>Duration</small><strong>{duration:.2f} s</strong></div>
<div class="card"><small>Regions</small><strong>{len(regions)}</strong></div>
<div class="card"><small>Flagged for review</small><strong>{flagged}</strong></div>
<div class="card"><small>Manually edited regions</small><strong>{corrected}</strong></div>
<div class="card"><small>Correction events</small><strong>{len(changes)}</strong></div></div>
<p><b>Audio file:</b> {esc(payload['audio_name'])} ({esc(payload['audio_size'])} bytes)<br>
<b>Method:</b> {esc(payload.get('method', ''))}<br><b>PANNs settings:</b> {esc(settings_text)}<br><b>Scores:</b> {esc(payload.get('score_note', ''))}</p></section>
<section><h2>Current labels by duration</h2><div class="tablewrap"><table><thead><tr><th>Label</th><th>Regions</th><th>Duration (s)</th><th>Recording share</th><th>Share</th></tr></thead><tbody>{''.join(summary)}</tbody></table></div>
<p class="muted">Durations use the current reviewed boundaries. Counts are regions, not words or speakers. Scores are uncalibrated automatic estimates.</p></section>
<section><h2>All regions</h2><p class="muted">Automatic scores refer to the original analysis windows and may not describe manually changed boundaries. Review flags and notes are recorded in the table.</p><div class="tablewrap scroll"><table class="regions-table"><colgroup><col style="width:40px"><col style="width:76px"><col style="width:76px"><col style="width:76px"><col style="width:86px"><col style="width:86px"><col style="width:68px"><col style="width:68px"><col style="width:68px"><col style="width:68px"><col style="width:90px"><col style="width:105px"><col style="width:72px"><col style="width:68px"><col style="width:68px"><col style="width:340px"></colgroup><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>Manual correction history</h2>{''.join(audit)}</section>
<section><h2>Complete saved data</h2><p>The report contains the complete review data so it can be reopened in the application.</p>
<details><summary>Show full review data (JSON)</summary><pre>{esc(json.dumps(payload, ensure_ascii=False, indent=2))}</pre></details></section>
<footer>The report does not contain the WAV audio. Keep the audio file to reopen and listen in the app.</footer></main>
<script id="speech-corpus-review-data" type="application/octet-stream">{embedded}</script>
</body></html>'''


class EmbeddedReviewParser(HTMLParser):
    # Prepare a buffer for the review data embedded in the HTML.
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.capture = False
        self.found = False
        self.parts = []

    # Start collecting text at the script element that contains the saved review.
    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'speech-corpus-review-data':
            if self.found:
                raise ValueError('The report contains multiple embedded reviews.')
            self.found = True
            self.capture = True

    # Keep only text inside the review-data element.
    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)

    # Finish collecting the review data when the script closes.
    def handle_endtag(self, tag):
        if tag == 'script' and self.capture:
            self.capture = False


# Read the saved timeline back so the reviewer can continue editing.
def load_review(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.json':
        return json.loads(text)
    parser = EmbeddedReviewParser()
    parser.feed(text)
    if not parser.found:
        raise ValueError('This HTML file does not contain a saved review from this application.')
    try:
        data = base64.b64decode(''.join(parser.parts).strip(), validate=True)
        return json.loads(data.decode('utf-8'))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('The embedded review data could not be read.') from exc


def save_report(path, payload):
    """Replace the target only after the complete report has been written."""
    path = Path(path)
    contents = render_report(payload)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.review-', suffix='.tmp', delete=False) as temp:
            temp_name = temp.name
            temp.write(contents)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)

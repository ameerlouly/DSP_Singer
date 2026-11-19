# Speech Corpus Studio — PyQt6 integration, Parts 1–5

This package combines the latest Parts 1–2 download/extraction pipeline with the supplied Parts 3–5 processing code in one PyQt6 application. The teal sidebar, white cards, scrolling layout, empty destination field and download bar without percentage text are retained.

**Part 5 remains incomplete against the project specification: the supplied code contains one energy SNR estimator, not two.** See REVIEW.md before describing all five parts as complete. This integration does not claim to validate automatic labels or word boundaries on real Arabic recordings.

## Windows setup

1. Extract the entire ZIP. Use a short path such as `C:\SpeechCorpus` to reduce Windows long-path installation errors. Do not run the app inside the ZIP.
2. Use Python 3.11, preferably 64-bit. The launcher requests Python 3.11 when the `py` launcher is available. An existing `.venv` is reused.
3. If FFmpeg is installed without a PATH entry, copy **both** `ffmpeg.exe` and `ffprobe.exe` into `tools\ffmpeg\bin` inside this project. Alternatively, add the installed FFmpeg bin folder to Windows PATH and reopen the launcher.
4. Double-click `run_app.bat`. It creates `.venv` on the first run and installs `requirements.txt`. Torch and related packages are large, so the first installation can take several minutes. Subsequent launches reuse the environment; dependencies are installed again when requirements.txt changes.
5. The app opens with an empty destination field. Choose a destination before downloading.

Manual setup in PowerShell, if needed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Optional, only if FFmpeg has not been added to Windows PATH:
$env:PATH = "C:\your\ffmpeg\bin;" + $env:PATH
.\.venv\Scripts\python.exe qt_app.py
```

FFmpeg itself is not bundled. For a shared repository, do not commit `.venv`, downloaded models or corpus audio. Requirements use compatibility ranges, not a fully tested lockfile. If you install CUDA builds manually, use a matching torch/torchaudio pair and follow PyTorch's installation guidance for your computer.

## Using the combined app

### Source & Audio

1. Paste a video URL.
2. Choose Destination. An empty destination produces an error before downloading.
3. Enter an optional folder name; leave it blank for automatic `video1`, `video2`, etc.
4. Download. The colored line fills without numeric text. Unknown-size downloads and merging use an activity indicator.
5. Extract audio. The verified 16 kHz mono 16-bit PCM WAV is automatically made available to Regions. You may also browse an existing WAV from any processing page.

### Regions — Part 3

1. Enter the reviewer name.
2. Choose **Show spectrogram**, or **Label automatically** to classify and build the display.
3. On its first run, PANNs downloads its approximately 327 MB checkpoint and label file into your user home `panns_data` folder. Internet access is required for that first download. The first Silero run also needs its model/package resources.
4. Adjacent 2-second analysis windows with the same label are automatically merged into longer regions. Speech, music, mixed and silence are shown as colored spans over the spectrogram and as rows with duration-weighted average automatic scores. Individual window scores and review flags are retained in the saved region data. Use the Matplotlib toolbar to zoom or pan.
5. Select a row or click its plot region. Set its label/start/end, then Apply correction. Moving a shared boundary adjusts the adjacent region too. You can also drag near a shared boundary and release it at a new position. Turn off toolbar zoom/pan before dragging boundaries.
6. Play the region or listen to one second on each side of its start/end. Split at midpoint and Merge with next are available; edit the new boundary precisely after splitting.
7. Save HTML report to create a readable report that can be reopened. **Save session** preserves the combined review state.

Scores describe the original inference windows and are uncalibrated. Changing boundaries does not rerun inference. Quiet speech may be labeled silence; low speech/music scores are flagged for review. Region changes clear downstream segment and SNR results so they must be regenerated.

### Segments — Part 4

1. Use **Natural pauses** for ordinary project segmentation. This calls `step4_cut_segments.export_candidates`, keeping source intervals continuous and using existing pause context up to 0.15 seconds at each edge.
2. Choose Generate candidates and an output parent folder. No source WAV is modified.
3. Listen before approving. Ctrl-click or Shift-click to select multiple rows for approval or exclusion.
4. For generated clips, **Clip-local trim** edits within the candidate audio and preserves any joins/source mappings. **Original source times** edits a continuous interval in the source WAV. Enter times in the correct coordinate system.
5. Inspect unresolved spans and uncovered gaps below the candidates table. You can label and mark a span reviewed, or choose a valid 5–15 second interval and add a manual clip. Added clips still require approval.
6. Export approved WAVs creates a fresh folder containing separate audio files, a review session, HTML report, playlists and `step5_manifest.json` with source mappings and file hashes. Pending/unresolved counts remain explicit; a partial export does not mean review is complete.
7. Overlapping approved source audio blocks export/SNR until corrected. The overlap tool can trim duplicate edges or exclude duplicates, records those changes, and reports cases needing manual attention. Listen to trimmed boundaries again.

The optional **Compact joins + fallback** mode preserves your team's original behavior: it can join non-contiguous speech and add artificial edge silence to fallback cuts. It is explicitly offered as a review mode, not proof of compliance with the real-pause requirement. Natural mode avoids those fallback cuts but still depends on VAD and human listening.

### Audio Quality — Part 5

1. Approve clips in Segments; there is no need to select the WAV files again.
2. Start with a clean threshold of 20 dB and click Calculate SNR.
3. The table and histogram show the supplied energy SNR estimate and clean/noisy/unavailable decisions. Insufficient audio produces unavailable; it is not forced to clean or noisy.
4. Select one or more clips, enter a reviewer and correction reason, then mark clean/noisy or restore the automatic label. Automatic results and a timestamped correction history remain separate.
5. Save HTML / CSV / JSON. Reopen the saved SNR JSON while the matching Step 4 review is loaded, or save/open the complete session.

The UI visibly states that a second SNR method is missing. The estimator uses the 10th and 80th percentiles of nonzero frame energies. Short edge padding may provide fewer quiet frames than the assumed percentile; continuous speech and music can make estimates inaccurate. These results need listening-based evaluation and a second estimator before Part 5 is complete.

### Save and continue later

Click Save session on a processing page. The JSON stores the selected source path, region labels/correction history, Step 4 review, SNR results/quality corrections, reviewer and threshold. Open session restores them. Rebuild the spectrogram with Show spectrogram.

Keep the original source and candidate folders at their saved locations. Session files reference audio; they do not embed or copy it. Save prompts appear before discarding a dirty review or closing while idle. If processing is running, closing shows OK / Cancel: Cancel keeps processing; OK terminates the app and its processing subprocesses. Unsaved changes are lost and incomplete output files may remain after confirmed termination. Rename a video folder before loading it into Regions.

Transcription and Review & Export sidebar pages are reserved for the team's later integration. Part 4 approved-audio exports and Part 5 report exports are already available on their respective pages. ASR and final corpus export are not implemented here.

## Editing the UI in Qt Designer

- `main_window.ui`: existing application shell, sidebar and Source & Audio page.
- `processing_pages.ui`: new editable processing controls, tables and layouts for Parts 3–5. Its `processingPages` stacked widget contains the three pages; shared WAV/reviewer/session controls sit above it.
- `qt_processing.py`: Qt signal connections, Matplotlib canvases, background jobs, media playback and shared state.
- `qt_app.py`: entry point and download/extraction controller. It inserts the processing form into the shell and routes the sidebar.

Open `.ui` files with Designer. Keep widget object names unchanged, because the controllers refer to them. No UI-to-Python conversion is required: `uic.loadUi` loads the files at runtime.

## Calling the processing functions

```python
from pathlib import Path
from step3_label_regions import classify_file
from step4_cut_segments import export_candidates
import step4_review as review
import step5_snr as snr

wav = Path('path/to/audio.wav')
regions = classify_file(wav, progress=print)
result = export_candidates(wav, regions, wav.parent, progress=print)
session = review.load_session(Path(result['output_dir']) / 'step4_segments.json', wav)

# Review/listen first; approve only the indices actually checked.
review.approve_segments(session, [0], reviewer='Reviewer name')
records = snr.session_records(session, wav)
results = snr.calculate(records, threshold=20)
html_path, json_path, csv_path = snr.export_report(results, 20, wav.parent)
```

Run file/model operations in a worker when calling them from Qt. `Task` in `qt_processing.py` shows the QThread/signals pattern. Widget access belongs on the Qt main thread.

## Verification included

`tests/check_parts35.py` exercises synthetic audio through the Qt workflow with a controlled VAD stub. `tests/check_processing_edges.py` checks classifier branches using a fake model and tests spectrogram/SNR edge cases. Run from the extracted project folder:

```powershell
.\.venv\Scripts\python.exe tests\check_parts35.py
.\.venv\Scripts\python.exe tests\check_processing_edges.py
```

Tests use synthetic fixtures and do not prove real PANNs/Silero accuracy or Windows audio-device playback. REVIEW.md records what was and was not checked. AI assistance was used for the Qt integration, documentation and checks; team members should understand and explain the code during the demo.

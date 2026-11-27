# Speech Corpus Studio

Speech Corpus Studio is a Windows desktop application for building a labeled Arabic speech corpus from video. Each review row follows an interval in the original audio through labeling, segmentation, quality measurement, transcription and export.

## Start on Windows

1. Install Python 3.11 and make FFmpeg and FFprobe available on PATH, or put both executables in `tools/ffmpeg/bin/`.
2. Double-click `run_app.bat` (or `run_intel_arc.bat` for Intel Arc). The launcher which creates the local Python environment, installs the requirements, checks PyTorch and opens the application. Subsequent launches reuse that environment.
3. In **Source & Audio**, select a destination and download a video, or choose a local video. Extract a mono, 16 kHz, 16-bit PCM WAV for the full pipeline. The extraction controls also support producing other formats.
4. In **Transcription**, start a transcription job to download the recognizers on first use. **Qwen** is system A and **Cohere Arabic INT8** is system B. They run sequentially. Cohere uses NVIDIA CUDA or Intel Arc XPU. For Intel Arc, launch `run_intel_arc.bat` (separate `.venv-intel` environment); see the Cohere setup guide. Model weights are cached locally.

**Auto process** asks for the editor name, downloads if needed, extracts corpus-format audio if needed, labels regions, generates candidates, automatically approves every valid continuous 5–15 s candidate, calculates and saves both SNR methods (HTML/CSV/JSON), and runs both recognizers on all eligible speech/mixed clips and unresolved previews. Automatic approvals are recorded separately from human listening confirmations; invalid candidates and unresolved spans are not approved. Hidden gaps and excluded/nonvoice transcription rules remain unchanged. After successful ASR, a folder-selection dialog asks where to store the explicitly partial corpus, which includes playable `corpus_report.html`, `review_compatible.csv`, WAV/TXT, CSV and JSON. Output is grouped under the selected destination's `Review and export/Automatic partial export/<timestamp>/`. Cancelling only skips export, preserving SNR and ASR autosaves. Unfinished transcription decisions remain unfinished and both model outputs are preserved. An interrupted or failed transcription does not trigger automatic export; its autosave is retained. Manual export enables partial mode by default. Uncheck Allow partial export to request strict export. Complete-mode export checks pending candidates, resolved-clip SNR and transcription, but permits genuine unresolved spans even without review. Only unresolved spans included in the Transcription queue are included as `_unresolved.wav` and remain explicitly unresolved; the report identifies complete-mode export with unresolved intervals and the JSON retains `partial: true` when source review is incomplete; automatic approval does not prove the natural pauses are correct. No quality approval is required.

If an export fails, the dialog and Activity panel identify the exception, export stage and (when applicable) clip ID. A full traceback is saved in `Review and export/processing_error_<timestamp>.log` (or a temporary folder if the project is not writable). Existing ASR autosaves remain usable: reopen the latest complete `*_full_autosave.json` from the Transcription page, enable **Allow partial export**, and choose **Export final corpus**. There is no need to run the recognizers again just to retry an export. JSON and HTML files are written incrementally to reduce peak memory usage.

**Review & Export** now contains a read-only final transcript viewer, playback, the review table, export and summary controls. Select a row to inspect the final text and its status. Make any corrections in **Transcription**, **Segmenting the Speech**, or **Audio Quality SNR** instead. In Transcription the selected segment's current decision and segmentation status are displayed above the model texts; **Next segment →** advances one row, retaining existing draft-saving behavior and leaving untouched decisions unchanged. Use the clip-relative playback slider above the model texts to seek, **Play from cursor**, **Pause / resume**, or **Stop playback**. Changing the selected segment stops playback and resets the cursor; playback cannot continue past the selected clip.

Part 3 displays the automatic label confidence on a 0–1 scale and as a percentage in its table and region report. Review & Export shows normalized model and final text in its table and saved reports. The transcription editor and saved session retain the original wording.

## Workflow and code map

| Stage | Main files | What they do |
|---|---|---|
| Application and layout | `speech_corpus_app.py`, `main_window.ui` | Launch Qt, load the Designer layout and connect the source controls. |
| Source and extraction | `video_download_and_audio_extraction.py` | Download video, record metadata and license evidence, track bytes and extract PCM audio. |
| Labeling regions | `audio_region_classifier.py`, `vocal_verification.py`, `audio_spectrogram.py` | Apply PANNs sound-event labels, optionally check singing with a separated vocal track, and plot the signal. |
| Region review | `region_editing.py`, `region_review_report.py` | Relabel a range inside a merged region and save or reopen its HTML report. |
| Segmenting speech | `pause_based_segmentation.py`, `segment_review.py`, `reviewed_clip_manifest.py` | Use Silero speech activity and pause evidence to propose 5–15 s clips. Review cuts and export approved clips with their source mapping. Unresolved spans remain available for review and downstream analysis. |
| Audio quality | `snr_estimators.py`, `clip_quality_assessment.py` | Estimate SNR from pause energy and WADA, then apply the selected threshold and reviewer corrections. |
| Transcription | `qwen_asr.py`, `cohere_asr.py`, `transcript_agreement_and_io.py` | Decode eligible intervals, compare normalized Arabic text and preserve raw model outputs. |
| Review and export | `corpus_review_and_export.py` | Read and validate clip records, write the review table, append per-video summaries and export the corpus. |
| Review controls | `audio_review_controller.py`, `transcription_review_controller.py` | Connect the processing functions to the corresponding Qt pages and save or restore review sessions. |
| Environment | `setup_pytorch_cuda.py`, `requirements.txt` | Check the Python environment and configure matching Torch and torchaudio dependencies. |

Each application module starts with its purpose and a guide to its functions. For the details behind individual decisions, see [ASR agreement](ASR_STEP6_RULES.md), [Cohere setup](COHERE_ASR_SETUP.md), [singing detection](SINGING_DETECTION_NOTES.md) and [SNR methods](SNR_METHODS.md). The WADA reference data and its redistribution notice are in `assets/wada/`.

## Saved runs

The video project keeps its chosen name. Each processing step groups output from one run in a local-time folder named `YYYY-MM-DD_HH-MM-SS`. For example, two extractions go into separate directories under `video1/Extract audio/`. Saves in the same second get suffixes such as `_02` and `_03`.

Download metadata stays beside its video. SNR writes its HTML, CSV and JSON together. Each transcription run updates a single autosave in its own folder. Save As suggests dated folders for review tables and complete sessions. The shared Excel summary still appends to the workbook you select. Existing project folders and saved paths can be opened as before.

## Check a checkout

From the project folder, after installing dependencies, run these commands in PowerShell.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe tests\check_full_workflow.py
```

The workflow check creates temporary audio and uses controlled recognizer outputs. The other `tests/check_*.py` scripts exercise individual UI and pipeline behaviors. The optional `check_cohere_sdk.py` downloads processor/configuration files and exercises the SDK with a tiny model.

Keep the source, UI, runtime assets and test fixtures in Git. Use a destination outside the checkout for downloaded videos, model caches and generated project outputs.

### Portable review HTML

In Review and Export, choose **Save review table → HTML**. The report includes a player for every included, ASR-eligible segment or unresolved span in the Transcription queue. Skipped, non-voice and sub-second unresolved spans are not exported as WAVs. Its sibling `review_audio` directory (or a uniquely named `review_audio_*` directory on repeated saves) contains exact PCM copies of the reviewed audio. Keep the HTML and that directory together when moving or sharing the report. Paths are relative to the HTML file; no original source file or internet connection is needed for playback. CSV, JSON and Excel saves are unchanged.

### Automatic uncovered gaps (V12)

Part 4 no longer lists unclassified or non-voice `gap-*` rows in the unresolved table. They are estimated pauses, not confirmed silence. They stay in the original WAV and saved Step 4 JSON/audit and still count toward uncovered voice time where applicable; no approval is fabricated. They do not block complete or partial corpus export, and are not exported as individual clips. Previously explicitly labelled speech/mixed gaps are retained as voice-bearing unresolved spans. Pending candidates and resolved-clip SNR/transcription checks still apply to complete export. Unresolved speech no longer blocks it. Only the included transcription rows are exported, with unresolved status intact; skipped spans remain in the source audit. Existing sessions can be reopened without rerunning segmentation or ASR.

### Approve all segments (V13)

Approve all segments now includes every pending candidate whose continuous original-source audio lasts 5–15 seconds inclusive, including automatically merged candidates and candidates flagged for boundary review. If any have unverified pauses, one batch confirmation replaces individual prompts; the reviewer confirms the natural boundary conditions for the group. This is a reviewer approval, not automatic evidence of natural pauses. Invalid mapping, out-of-range, excluded and unresolved intervals are not approved. Valid pending manual candidates remain eligible as before. The audit records `batch_approve_all` as the approval method.

### Compatible review CSV (V16)

Both **Export final corpus** and Auto Process's automatic partial export now also write `review_compatible.csv` next to `corpus_report.html`, `corpus.csv` and the other artifacts. Its 16 columns match the supplied review CSV, in the same order, with UTF-8 BOM for Arabic text in Excel. It stores original Qwen and Cohere text and the actual final transcript (empty when not chosen), timestamps, SNR, quality, decision, reviewer and JSON change history. Audio filenames are relative to the export folder and point to the same WAVs used by the HTML. The detailed 17-column `corpus.csv` and reports are unchanged; the identical `review.csv` is no longer generated during corpus export. Standalone review-table saves still work normally; consult `segmentation_status` in those files or the audit JSON for pending/unresolved boundaries, since the compatible schema has no such column. No additional approval is implied by a transcription decision.

### Optional audio quality correction

Audio Quality SNR assigns clean/noisy automatically. There is no mandatory quality-review column or approval; estimator disagreements remain in Notes and the audit. Mark clean/noisy or restore the automatic decision only if needed. Report saving and corpus export do not require a threshold justification to be entered, although the assignment still requires justification in the report. Missing/stale/unavailable measurements still require recalculation or partial export. Reopen saved V17 sessions without rerunning ASR; after changing quality labels, load clips in Transcription to synchronize updated metadata without recomputing unchanged transcripts.

### Resizable UI tables

All seven tables support dragging column dividers in the top header and row dividers in the left header. Double-click a column divider to fit its contents. Region columns initially share the available width equally; dragging a divider overrides that layout. Other tables fit their initial contents once. Manual column widths are preserved during refreshes for the current app session. Wide tables can be scrolled horizontally.

### Export inclusion
Both manual and automatic corpus export, and Save review table (HTML/CSV/Excel/JSON), use the included ASR-eligible rows from Transcription, not the complete source timeline. After toggling Skip/Include in transcription in Part 4, reload the transcription queue before exporting. Excluded and skipped intervals remain in the Step 4/source review audit; no WAV/TXT or exported review row is generated for them. Pending transcription results on included rows remain eligible for partial export.

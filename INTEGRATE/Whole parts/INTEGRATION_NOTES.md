> Current Part 4/5 behavior is documented in README.md and SNR_METHODS.md. Older implementation details below may describe the original integration; the current path uses reviewed continuous source audio and two SNR estimators.

# Full integration review — 22 November 2026

The supplied `last.zip` contains a functional baseline for ASR, transcript agreement, review persistence and CSV export. Its original desktop frontend combines Tkinter with a PySide6 Arabic editor. It is not a drop-in frontend for the existing PyQt6 application, and its `pipeline.py` name conflicts with the existing download module.

The integrated application keeps the previous project and single `main_window.ui`, adapts the teammate's core as `asr_pipeline.py`, and adds `qt_asr.py` plus `corpus_review.py`. The previous source-only Parts 1–5 package is preserved separately.

## Requirement status

| Area | Current integrated behavior | Remaining validation or work |
|---|---|---|
| Download and audio extraction | Existing metadata/license handling, destination validation, realistic byte-weighted download bar, 16 kHz mono PCM-16 extraction | Live YouTube availability/license metadata depend on source/extractor. No new live YouTube test was done for this integration. |
| Region labeling | Existing PANNs labels, merged adjacent label runs, spectrogram and human boundary edits | Validate accuracy on representative Arabic speech/music/silence/mixed recordings. |
| Segmentation | Existing natural-pause default and explicit compact/fallback review mode, unresolved spans, approval and exact sample mappings | Human review remains required for safe word boundaries, joins and synthetic silence. |
| SNR | Pause-frame energy + WADA-SNR, both GUI values, decision rule, threshold explanation, persistent quality overrides and final-review recalculation | Implemented; matched original CMU scores on 12 reference examples and checked injected-noise fixtures. Real Arabic-video calibration/listening still required. |
| Dual ASR | Whisper and MMS on approved speech/mixed clips, including noisy clips; music/silence excluded | Actual model downloads/inference and Windows CPU/GPU installation were not executed in this environment. |
| Arabic agreement | PDF Step 6 letter normalization; original + normalized GUI text, jiwer CER/WER/word edits, versioned rules, empty-output handling and preserved human decisions | Specification/integration checks pass. Validate actual transcripts against human Arabic references; normalization does not fix dialect recognition. See ASR_STEP6_RULES.md. |
| Human review | Native Qt RTL final editing, choose A/B, label/quality overrides, move/split/merge/add/delete, audit history | Inspect edited clips and rerun invalidated SNR/transcripts; listen to target-device audio. |
| Persistence | Every model/clip result atomically saved in a complete session before the next job; explicit session save/reopen | Keep source and candidate audio at saved paths. Session files reference rather than embed audio. |
| Corpus output | Per-clip WAV + UTF-8 TXT, CSV, JSON, readable HTML with audio controls and source-review provenance | Process/review the required three one-hour dialect videos. Partial exports are marked. |

## Changes that matter

1. **One Qt framework and UI file.** No Tkinter/PySide6 application is nested inside PyQt6. All editable controls are in `main_window.ui`; model work remains in worker threads.
2. **Exact approved audio.** The handoff uses the actual candidate WAV and clip-local trims. It preserves joined spans and artificial padding rather than playing a single bounding source range that can contain removed gaps.
3. **Arabic MMS configured explicitly.** Both tokenizer language and model adapter are set to `ara`, following the MMS model-card API. MMS transcription/confidence share one forward pass in the integrated loading path.
4. **Sequential model phases.** Whisper is released before MMS is loaded. Cancel stops remaining jobs after the current operation; already completed results remain saved.
5. **No stale annotation after audio edits.** Moving/splitting/merging clears old transcripts, confidence, SNR and quality. The new sample ranges must be assessed again.
6. **Explicit incomplete outputs.** Standard corpus export blocks unresolved decisions, absent ASR attempts, unknown quality and overlapping final source spans. Explicit partial export retains pending states and source review counts. Full export now also checks current two-method SNR, review flags and threshold justification; explicit partial outputs remain marked.
7. **Backward-compatible combined sessions.** Earlier `speech_corpus_qt_session_v1` files load with empty ASR state. New saves include the ASR/final review alongside Regions, Segments and SNR.
8. **Dependency files combined.** `requirements.txt` includes faster-whisper, jiwer, transformers and accelerate in addition to existing PyQt6/audio/Torch requirements. The launcher installs the combined requirements in one environment. Models are downloaded on first use, not bundled in the ZIP.

## Checks performed

- Loaded the single UI file with all six pages and inspected the two new screens.
- Ran ten adapted tests from the teammate's core: nine passed; the Torch-specific confidence test was skipped because Torch is not installed in the verification environment.
- Verified explicit MMS Arabic tokenizer/adapter initialization using mocked library objects matching the model-card API; no checkpoint was downloaded for that check.
- Ran a real Qt/audio/filesystem workflow with **controlled ASR model outputs**: source review -> exact joined clip including padding -> two engine jobs -> disagreement/human choice -> autosave -> stop button -> resume only missing jobs -> combined-session reload -> real WAV/TXT/CSV/JSON/HTML exports.
- Compared exported joined PCM samples with the approved candidate samples; they matched.
- Checked merge/split/add/delete, source mappings, length limits, annotation invalidation and complete/partial export gates.
- Re-ran the earlier Parts 3–5 integration checks, close confirmation with a real running thread/child process, and measured download-byte/fill checks.

These checks validate integration behavior, not the real recognition accuracy of large-v3 or MMS. Windows batch installation, actual ASR checkpoints, microphone/speaker devices, and full-length recordings remain target-machine tests. No three-video corpus or golden-reference accuracy result is claimed.

## First local run

1. Extract into a fresh short path, for example `C:\SpeechCorpusFull`.
2. Confirm Python 3.11 and FFmpeg are available; otherwise use `tools\ffmpeg\bin` for ffmpeg.exe/ffprobe.exe.
3. Double-click `run_app.bat` to create/update the shared environment.
4. Start with two approved short clips. Run SNR, then Transcription -> Use current approved clips -> Transcribe all pending.
5. Check both Arabic outputs, playback and a saved/reopened session before running an hour of audio.
6. Export a small corpus and check each TXT matches its WAV and that the CSV retains both model texts/statuses.

## Attribution and primary references

- Teammate's `last.zip`: original transcript normalization/comparison, model wrapper baseline, source-manifest persistence and CSV export helpers. Original README calls the standalone app “Ameer — Week 3 Speech Corpus Review”.
- Earlier supplied Parts 3–5 archive and our existing PyQt6/download integration were retained as the baseline.
- MMS model card and explicit language-adapter example: https://huggingface.co/facebook/mms-1b-all
- faster-whisper author implementation: https://github.com/SYSTRAN/faster-whisper
- AI assistance was used for the frontend integration, adapters, documentation and tests. Each team member should be able to explain the code used in the demo.

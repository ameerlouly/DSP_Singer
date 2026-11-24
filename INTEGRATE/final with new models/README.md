# Speech Corpus Studio

Speech Corpus Studio is a Windows desktop application for building a labeled Arabic speech corpus from video. Each review row follows an interval in the original audio through labeling, segmentation, quality measurement, transcription and export.

## Start on Windows

1. Install Python 3.11 and make FFmpeg and FFprobe available on PATH, or put both executables in `tools/ffmpeg/bin/`.
2. Double-click `start_app.bat`. It calls `run_app.bat`, which creates the local Python environment, installs the requirements, checks PyTorch and opens the application. Subsequent launches reuse that environment.
3. In **Source & Audio**, select a destination and download a video, or choose a local video. Extract a mono, 16 kHz, 16-bit PCM WAV for the full pipeline. The extraction controls also support producing other formats.
4. In **Transcription**, start a transcription job to download the recognizers on first use. **Qwen** is system A and **Cohere Arabic INT8** is system B. They run sequentially. Cohere requires a CUDA-capable NVIDIA setup. Model weights are cached locally.

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
| Review and export | `corpus_review_and_export.py` | Edit clips and final transcripts, write the review table, append per-video summaries and export the corpus. |
| Review controls | `audio_review_controller.py`, `transcription_review_controller.py` | Connect the processing functions to the corresponding Qt pages and save or restore review sessions. |
| Environment | `setup_pytorch_cuda.py`, `requirements.txt` | Check the Python environment and configure matching Torch and torchaudio dependencies. |

Each application module starts with its purpose and a guide to its functions. For the details behind individual decisions, see [ASR agreement](ASR_STEP6_RULES.md), [Cohere setup](COHERE_ASR_SETUP.md), [singing detection](SINGING_DETECTION_NOTES.md) and [SNR methods](SNR_METHODS.md). The WADA reference data and its redistribution notice are in `assets/wada/`.

## Check a checkout

From the project folder, after installing dependencies, run these commands in PowerShell.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe tests\check_full_workflow.py
```

The workflow check creates temporary audio and uses controlled recognizer outputs. The other `tests/check_*.py` scripts exercise individual UI and pipeline behaviors. The optional `check_cohere_sdk.py` downloads processor/configuration files and exercises the SDK with a tiny model.

Keep the source, UI, runtime assets and test fixtures in Git. Use a destination outside the checkout for downloaded videos, model caches and generated project outputs.

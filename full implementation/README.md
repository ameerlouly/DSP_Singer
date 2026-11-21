# Full DSP Speech Corpus Implementation

A native Windows desktop application that performs the complete corpus workflow:

1. Download a video from a URL.
2. Extract one 16 kHz mono 16-bit PCM WAV.
3. Label contiguous regions as speech, music, mixed or silence.
4. Split speech/mixed regions into 5–15 second segments at low-energy pauses.
5. Estimate SNR and classify clean/noisy.
6. Run faster-whisper and MMS Arabic ASR.
7. Compare normalized transcripts and assign accepted, needs_review or difficult.
8. Review the complete WAV in a waveform and spectrogram, drag segment boundaries, listen to segments and apply human overrides.
9. Export the WAV, segment CSV/JSON, individual segment WAV files and final TXT transcripts.

## Run

Use Python 3.10 or later on Windows. From this folder, create a local
environment and install the listed dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This environment is local and is not stored in Git. Recreate it with these
commands after cloning the repository.

```powershell
.\run_app.bat
```

Or launch the app directly:

```powershell
python app.py
```

## Workflow

- Paste a video URL on the Download tab.
- Choose an output folder.
- The app downloads the video, extracts the WAV, labels regions, creates segments, calculates SNR and runs ASR.
- The Review tab opens automatically when processing completes.
- Drag vertical dividers to adjust boundaries.
- Select a row and click **Human Override** to change label, SNR, quality, status or transcripts.
- **Export WAV + CSV** copies the project WAV and exports all deliverable files.

## Restoring source media

Downloaded videos and extracted `source_16k_mono.wav` files are excluded from
Git because they are large generated media. To recreate a project, install the
dependencies above, run the app, paste a video URL you are authorized to
download and use into the Download tab, and choose the `data` folder as the
output location. The app creates a new `video_<timestamp>` folder containing
the downloaded source, extracted WAV, and processing outputs. To reproduce an
existing project, use its `source_url` from `metadata.json` if it is still
available and you have permission to use it, or supply your own authorized
source file.

## Exported files

- Source WAV
- `speech_corpus_segments.csv`
- `speech_corpus_segments.json`
- `segment_audio/*.wav`
- `segment_audio/*.txt`

## Notes

- Whisper and MMS models are downloaded or loaded during first use.
- MMS requires `torch` and `transformers`.
- Automatic region and SNR labels are preliminary confidence scores and can be overridden.

# Ameer — Week 3 Speech Corpus Review

A Windows desktop app for reviewing Robir segment JSON against the complete source
WAV, transcribing speech/mixed segments with Arabic Whisper and MMS, resolving
transcript disagreements, and exporting a corpus CSV.

## Install

Use Python 3.10 or later. From this folder, create and activate a virtual environment,
then install the desktop and Whisper dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MMS is optional until you run that model. Install it separately when ready:

```powershell
python -m pip install -r requirements-mms.txt
```

The first ASR run downloads model weights. The default models are `large-v3` and
`facebook/mms-1b-all`; change either model name in the app if your computer needs
a smaller Whisper model. Model inference runs on the CPU for Whisper and uses a
CUDA device for MMS when PyTorch detects one.

## Run

```powershell
python app.py
```

You can also double-click `run_app.bat` after installing the dependencies. The
app initially looks for `step5_manifest.json` and `Chat GPT الدحيح.wav` two
folders above `Ameer`; use **Browse…** to select different files.

1. Load the segment JSON and the complete source WAV. Step 5 manifests and
   speech-corpus JSON files with a `segments` list are supported. The app reads
   audio on demand, so the complete recording is not copied into memory.
2. Select a row and play its mapped source audio. Multi-span Robir clips play
   their source spans in order. **Play edited range** auditions the current
   start/end values before applying a cut or move.
3. Use **Transcribe selected** to run both ASR systems on one speech/mixed clip,
   or **Transcribe all pending** to process clips missing either transcript.
   The progress bar shows the active model and segment. Music and silence are
   excluded from ASR. Each Whisper or MMS result is saved to both the source
   manifest and review JSON before the worker moves to the next segment. The
   source record includes `transcript_a`, `transcript_b`, `selected_transcript`,
   `confidence_a`, `confidence_b`, and `transcript_status`. Confidence values are
   normalized estimates, not calibrated probabilities. Use **Stop transcription**
   to cancel remaining work without clearing completed results.
4. Matching normalized transcripts are marked `accepted` (identical or CER at
   most 2%). Up to three word edits are marked `needs_review`; larger
   disagreements are `difficult`. Listen, choose A/B, or edit the final
   transcript; use **Edit RTL…** beside any transcript field to edit it in the
   Qt-based bidirectional Arabic editor. Accept a human correction with
   **Save review**.
5. Edit labels, clean/noisy quality, and boundaries; split, merge, add, or delete
   segments; then save or export the corpus CSV. Changes are timestamped with
   the current Windows user.

Arabic comparison normalization removes diacritics, tatweel and punctuation,
maps hamza/alif variants to alef, `ة` to `ه`, and `ى` to `ي`. The source JSON for
each clip and the manifest metadata are retained in the CSV, alongside source
time spans, both transcripts, final transcript, CER/WER agreement metrics,
status, and human-change history.

Review progress is autosaved to the source JSON and beside it as
`<json-name>_review.json`. Source fields such as Step 5's clip `status` remain
unchanged; comparison state is written separately as `transcript_status`. Load
the same JSON and WAV later and choose **Load / Resume** to continue.

## Tests

```powershell
python -m unittest test_pipeline.py
```

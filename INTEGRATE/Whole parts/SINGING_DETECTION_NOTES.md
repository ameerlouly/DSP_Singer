# Singing detection update — 22 November 2026

This update changes the general classifier, not the labels at hand-picked song timestamps. It uses pretrained models without training or fine-tuning on the supplied recording. There are no filename, artist, lyric or source-specific time checks in the classification code.

## Diagnosis and implementation

The supplied report had 84 two-second/final-short analysis windows over 167.149 seconds. Its non-silent median vocal score was approximately 0.106, below the 0.30 decision threshold. Only four seconds became mixed. Trying the same PANNs model on ten-second blocks did not fix the problem, so longer input alone was not adopted as the solution.

The enabled-by-default Improve singing detection option now checks music proposals with TorchAudio's pretrained Hybrid Demucs `HDEMUCS_HIGH_MUSDB_PLUS`. It separates a likely vocal signal, and the existing PANNs tagger checks Speech/Singing evidence in that signal. A music window becomes a mixed proposal only when its local separated signal passes absolute and relative energy gates and there is voice-tagging support locally or nearby in the analysis block. Energy alone is insufficient: orchestral instruments can leak into a separator's vocal output.

Strong short-window voice detections also provide nearby support because a full-context score can dilute alternating spoken/sung phrases. Each original window records which interval supplied that support. Every corrected proposal retains Needs review. Context may blur the exact start/end of vocals by several seconds; use playback/range editing before approving boundaries.

Code: `vocal_verification.py` and `step3_label_regions.py`. UI option: `improveSinging` in the single `main_window.ui`. The backend API accepts `classify_file(path, progress=print, verify_vocals=True)`; False selects the basic classifier.

## Real-model observations

The following runs used actual pretrained PANNs and Hybrid Demucs inference, not fabricated model scores. Public controls came from librosa's example collection. Files were converted to the project's 16 kHz mono PCM-16 input format. These are small observational checks, not a manually annotated or held-out accuracy benchmark; training-set overlap was not audited.

| Input | Scope | Result from enhanced classifier |
| --- | --- | --- |
| Supplied WAV | Entire 167.149-second file | 134 s mixed, 28 s music, 5.149 s silence; previous report proposed only 4 s mixed |
| Karissa Hobbs — Let's Go Fishin' | Source 20–28 s, an independent vocal-song excerpt | All 8 s proposed mixed |
| Brahms — Hungarian Dance #5 | Source 20–28 s orchestral excerpt | All 8 s remained music; no mixed proposals |
| Mihai Sorohan — trumpet loop | Entire 5.333-second example | 4 s music plus 1.333 s silence; no mixed proposals |
| LibriSpeech narration (`libri1`) | First 8 s | All 8 s remained speech |

Supplied-WAV proposals: 0–18 s music; 18–70 s mixed; 70–80 s music; 80–162 s mixed; 162–167.149 s silence. These are automatic proposals, **not human-verified boundaries**. A regenerated HTML report was delivered separately and can be reopened against the unchanged source WAV.

The supplied WAV took approximately 66 seconds after model caching on the validation Linux CPU setup (four Torch threads, Python 3.12, torch/torchaudio 2.5.1 CPU). This is one measurement, not a Windows runtime promise or a one-hour performance test. Real Windows model inference/audio-device playback and representative hour-long recordings still need validation.

## Setup and limits

- First enhanced use on music downloads about 319 MiB of additional model weights; subsequent runs reuse the Torch cache. Launching the app alone does not download this model.
- The separator uses the already-required torch/torchaudio packages. No separate `demucs` Python package is required. PANNs' `librosa`/`torchlibrosa` requirements are now explicit.
- No source audio is altered; stems remain temporary in memory. The app archive contains source code and tests, not the supplied/public audio or model weights.
- Original mixture scores remain available. Separation scores, energy gates, model identity, prior labels and supporting intervals remain in the per-window data; they are not calibrated probabilities or human verification.
- No lyric transcription is performed here. Wordless vocals, quiet voices, unusual processing, separator leakage and instrumental breaks can still confuse the models. This does not guarantee every song or every passage.
- Save the old session before automatic reanalysis, which replaces the region review. Regenerate segments/SNR and resynchronize transcription inputs after region changes.

## Automated checks

- `tests/test_vocal_verification.py`: energy and voice gates, nearby support, bounded processing, invalid output/long merged input rejection, original-score provenance, merge and range-edit preservation.
- `tests/check_singing_ui.py`: enabled default, main-thread option capture, worker handoff and session persistence.
- Existing vocal-label, range-edit, playback, Parts 3–5, full ASR/review/export, processing-edge and download-progress checks passed. ASR/VAD/audio-device checks retain their documented test-double limitations.

## Primary references

- PANNs authors' implementation: https://github.com/qiuqiangkong/panns_inference
- PANNs author discussion of ten-second training inputs: https://signalprocessingsociety.org/our-story/news-updates/2024/05/panns-large-scale-pretrained-audio-neural-networks-audio-pattern-recognition
- TorchAudio Hybrid Demucs tutorial: https://docs.pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html
- Pretrained bundle: https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.html
- Public example descriptions: https://librosa.org/doc/latest/recordings.html

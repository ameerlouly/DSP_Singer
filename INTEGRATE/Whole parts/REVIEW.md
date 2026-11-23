> Current SNR update: pause-frame energy and WADA-SNR are implemented, both are visible in the GUI, and manual review/threshold justification propagate to reports and corpus export. See SNR_METHODS.md for current behavior and numerical checks. The baseline findings below are historical and do not describe the current Part 4/5 implementation.

> Historical review of the Parts 3–5 baseline. See INTEGRATION_NOTES.md for the current full integration and ASR/final-review status.

# Review of the supplied Parts 3–5

Reviewed against the uploaded project description, especially Steps 3, 4 and 5. This is a source review plus controlled integration testing, not a measured accuracy evaluation on Arabic video.

## Findings

| Part | What is present | Finding and integration decision |
|---|---|---|
| 3 | PANNs speech/music tags, RMS silence rule, four labels, automatic scores, spectrogram and manual neighboring-boundary edits | A reasonable baseline, not validated accuracy. Two-second windows are coarse. Non-silent noise with low speech/music scores is forced into speech or music and flagged for review. Silence scores are heuristic. The new Qt UI retains scores, flags, manual editing, listening and audit history. |
| 4 | Strict VAD pause candidates, compact joined candidates, low-energy fallback cuts, unresolved spans and detailed review/export code | The submitted Tk app imports compact mode by default. Joining non-contiguous audio and inserting synthetic silence does not demonstrate a real pause between words. The Qt default now calls the existing strict pause-based implementation. Compact mode is optional and explicitly identified. All candidates remain pending until a human approves them. |
| 4 | Target edge context | Strict mode attempts 0.15 s and compact mode targets 0.10 s. Short pauses and region boundaries can provide less. Notes flag this; padding is VAD-estimated non-speech, not verified silence. Human listening remains required. |
| 4 | File lengths, format, source mapping, unresolved/gap records | Useful validation and provenance are present. The Qt integration retains them, supports multi-approval, manual clips and resumed reviews, and blocks export/SNR when approved source spans overlap. It does not assert that every VAD boundary is safe. |
| 5 | One energy-contrast estimator and clean/noisy threshold, reports, manual quality overrides | **Incomplete:** the specification requires two methods. `second_method: not implemented` is explicitly saved. No WADA-SNR or other second estimator was found. The UI makes this visible rather than presenting Part 5 as complete. |
| 5 | Energy estimate using 10th/80th percentile of frame powers | Quiet frames are assumed to be noise-only. Real edge pauses can occupy less than 10% of a clip; music/continuous speech can invalidate the assumption. Validate against listening and known-noise mixtures and justify the final threshold. A simple noise-order check is insufficient evidence of real-world accuracy. |
| Integration | Supplied GUI modules use Tkinter | They cannot be inserted as native PyQt widgets. Processing functions were reused; the frontend was implemented in PyQt6, with all editable controls in one `main_window.ui` and Qt worker signals. |
| Delivery | Original archive has no usable dependency manifest/README and contains compiled bytecode | Added a requirements manifest, launcher and setup/use instructions. Source-only package excludes bytecode and virtual environments. |

## Changes made while integrating

- Retained the existing Part 1–2 pipeline and visual style, with the empty destination validation and number-free animated download bar.
- Passed verified extracted audio to the processing workspace automatically.
- Added Regions, Segments and Audio Quality screens, embedded spectrogram/histograms, playback around cut points, shared reviewer identity, multi-selection and save/resume.
- Selected strict segmentation by default; compact/fallback is an explicit alternative.
- Clear downstream results after edits to prevent old SNR values describing changed clips.
- Prevent concurrent source/model jobs from mutating the same review.
- Added rejection of non-finite audio samples in the SNR estimator; unavailable values stay explicit.
- Preserved original backend files where no change was necessary. No PANNs, Silero or ASR training was performed.

## Verification performed

- Loaded the unified main_window.ui form and exercised all six sidebar pages.
- Tested Part 1–2 local HTTP download and real FFmpeg extraction through the combined Qt app, including empty destination state, hidden percentage text and valid WAV format.
- Exercised background spectrogram rendering on synthetic WAVs, colored regions, adjacent-boundary edits and correction history.
- Ran strict segment planning, real WAV output, 5–15 second/PCM format checks and unresolved/gap handling with predetermined VAD intervals. **Silero inference was stubbed**; this checks integration and export logic, not model detection quality.
- Exercised all four classification branches using controlled fake PANNs scores. **Actual model inference/download was not run.**
- Tested multi-approval, separate reviewed WAV export, manifest file hashes, SNR calculation, quality overrides, HTML/CSV/JSON output, complete session round trip and stale-result invalidation.
- Checked silence/short WAV spectrograms, invalid and silent SNR inputs, and a higher/lower noise sanity case with known quiet frames.

## Still to validate with your team

1. Run the full model/dependency installation on the target Windows machine. The Windows batch launcher was reviewed, not executed on Windows in this environment.
2. Check actual audio playback on the target Windows device; no audible playback was validated here.
3. Run PANNs and Silero on representative Arabic speech, music, silence, mixed audio and quiet speech; manually score labels and cuts, including short pauses and region edges.
4. Implement and validate a second SNR estimator (for example WADA-SNR), display both estimates, define the clean/noisy decision rule and justify the final threshold.
5. Evaluate all methods on the required long videos; measure runtime, memory, missed speech and word-interior cuts. No one-hour model run was performed here.
6. Integrate Part 6 ASR and later final corpus annotation/export. Existing placeholders do not implement those requirements.

## Primary references

- Uploaded `Course_Project_Building_a_Labelled_Speech_Corpus_V1(7).pdf`, Steps 3–5.
- PANNs author implementation: https://github.com/qiuqiangkong/panns_inference
- PANNs published package source, version 0.1.1: checked its API and metadata.
- Silero VAD: https://github.com/snakers4/silero-vad
- WADA-SNR reference listed in the assignment: Kim and Stern, Interspeech 2008. A reference is not an implementation; this package still lacks the second estimator.

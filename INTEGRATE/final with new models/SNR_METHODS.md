# Part 5 — two-method SNR implementation and validation

The authoritative project PDF (Step 5, page 2) requests two SNR methods and a clean/noisy decision starting at 20 dB with a justified final threshold. These are blind estimates: neither method measures ground-truth speech-to-noise ratio without an isolated clean reference.

## 1. Pause-frame energy estimate

- Measure the exact PCM-16, 16 kHz mono source interval. Approved clips retain reviewed edge context. Unresolved spans use detector pause intervals when available and otherwise use the quiet-frame fallback. The Segmentation field identifies them in the table and saved reports.
- Use the measured natural edge context from Part 4 and its saved pause intervals. For a manually selected/edited range, use the minimum 0.1 s edge context explicitly confirmed by the named reviewer. These interval labels are estimates and must be checked by listening.
- Use 20 ms frame energies with a 10 ms hop after removing the clip's DC mean. Whole digital-zero frames are excluded from acoustic-noise estimation. Require at least ten valid pause frames and ten valid active frames.
- Average background-noise power within contiguous pause runs, binned into at most 0.25 s anchors. Interpolate power between observed anchors; hold it constant outside the outer anchors. This follows observed noise changes, not unobserved changes during speech.
- Average observed power over non-pause frames, and compute `10 log10((P_active - P_noise) / P_noise)`. Active frames are the clip interior excluding marked pauses; unknown internal pauses or music can bias the estimate.
- If no pause metadata is supplied to the standalone API, a quiet-frame fallback is allowed and explicitly flagged as uncertain. The normal GUI uses the reviewed Part 4 context. Insufficient acoustic noise or nonpositive active-minus-noise power returns unavailable rather than an invented large SNR.

## 2. WADA-SNR

New Python implementation of Kim and Stern (2008), Eq. 8:

`G = log(mean(abs(z))) - mean(log(abs(z)))`

Use the original CMU alpha=0.4 lookup table, not the energy estimator. The assumptions are a symmetric Gamma model for clean speech and additive Gaussian noise; music, singing, interfering speakers and other departures still need evaluation. WADA does not consume or depend on the energy method's pause mask.

- The unchanged table is bundled in `assets/wada/Alpha0.400000.txt`; its CMU redistribution notice is in the same directory. There is no runtime table/model download or extra Python dependency.
- Remove whole digital-zero 20 ms blocks. Split remaining samples into balanced blocks no longer than 100,000 samples so a short last tail is not silently discarded. Remove each block's DC mean and peak-normalize before calculating the scale-invariant statistic. Values for the logarithm are floored at `1e-10`.
- Match the CMU implementation's **first table threshold above G**, preserving the table's original small low-SNR Monte Carlo irregularities. Resolution is 1 dB; table range is −20 to 100 dB.
- Combine blocks by their estimated signal and noise energies, not by averaging their dB values. Range-limited blocks are recorded; the combined result is excluded from automatic decision selection when range-limited. All block statistics, limits and the table checksum are preserved in JSON.

## Decision and review

The default rule selects the lower valid estimate. Alternative GUI rules select pause energy or WADA specifically. A selected unavailable/range-limited method stays unavailable; the application does not silently substitute another method in a specifically selected rule. A single valid method under the default rule retains an informational warning.

The user-adjustable clean/noisy threshold is initially 20 dB. It does **not** track instantaneous loudness. Threshold/rule changes reuse both saved estimates and preserve named human quality corrections. Recalculating identical samples also preserves corrections using the audio hash. Changed clips invalidate their old SNR.

Informational notes are recorded for method disagreement, a difference above 6 dB, unavailable/range-limited estimates, fallback noise estimation, noise variation above 6 dB, weak energy contrast, clipping and music/mixed content. The 6 dB checks are warning heuristics, not assignment-mandated values or confidence probabilities. Mark clean/noisy with a reviewer and reason, or restore automatic. The automatic decision and the correction audit remain separate.

Quality approval and threshold-justification entry are optional for saving/export. The team must still justify its final threshold based on real-video evaluation for the assignment; an omitted explanation is recorded as missing. Both values, decision rule, threshold, justification, noise evidence and human decisions persist in session/report files and the final corpus CSV/JSON. Full corpus export blocks missing/stale SNR or an unavailable decision SNR, but not estimator warnings or missing justification. Explicit partial export retains incomplete measurements.

Legacy energy-only reports remain readable and visibly marked for recalculation; reopening them does not manufacture WADA results.

## Validation performed

- Known injected noise: energy estimates tested against controlled Gaussian-active signals with measured natural pause noise; tests require error below 1 dB on those fixtures.
- Known WADA-model mixtures: symmetric Gamma alpha=0.4 speech plus white noise at −5, 0, 10 and 20 dB; tests require error below 2.5 dB. These are model-assumption checks, not real Arabic accuracy scores.
- Compiled the authors' original CMU C++ program outside the distributed project, removing only its obsolete 32-bit build flag. On the first 80,000 PCM samples of **all 12 official sample recordings**, Python and C++ produced identical table-grid scores (tolerance 0.02 dB). Results are in `tests/data/wada_reference_checks.json`. This verifies implementation compatibility on single blocks, not perfect SNR accuracy. No compiled program or sample recordings are distributed with this project.
- Tested time-varying measured pause noise, silence/DC/NaN/short inputs, raw digital zeros, table bounds, gain/DC invariance, balanced-block sample accounting, policy disagreement, human overrides, threshold changes, report escaping and legacy import flags.
- Exercised the real estimators through Qt workers, both table columns/histograms, report export, save/resume, ASR handoff and final CSV fields. ASR outputs in integration tests are controlled; Windows playback and threshold calibration on the three required Arabic videos remain team validation tasks.

## Primary sources

- Kim, C., and Stern, R. M. (2008), *Robust Signal-to-Noise Ratio Estimation Based on Waveform Amplitude Distribution Analysis*, Interspeech, pp. 2598–2601: https://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
- Authors' reference source/table: https://www.cs.cmu.edu/~robust/Algorithms/WADA_SNR.zip
- CMU Robust Speech Group source index: https://www.cs.cmu.edu/~robust/code.html
- Bundled notice source: https://www.cs.cmu.edu/~robust/Algorithms/RobustLicense.txt

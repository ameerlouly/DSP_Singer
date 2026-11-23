# Speech Corpus Studio — integrated download, audio, ASR and review application

This package combines the latest download/audio/segmentation/SNR project with the Whisper/MMS and final-review code supplied in `last.zip`. All six application pages now work in one PyQt6 application. The teal sidebar, white cards, scrolling layout and empty destination field are retained. The Part 1 download bar shows percentage-only progress.

**Part 5 now implements pause-frame energy and WADA-SNR, with both estimates in the GUI/reports and a justified clean/noisy threshold.** See SNR_METHODS.md for numerical/reference checks and remaining real-video validation. Automatic labels, SNR estimates and word boundaries still require human evaluation.

## Windows setup

1. Extract the entire ZIP. Use a short path such as `C:\SpeechCorpus` to reduce Windows long-path installation errors. Do not run the app inside the ZIP.
2. Use Python 3.11, preferably 64-bit. The launcher requests Python 3.11 when the `py` launcher is available. An existing `.venv` is reused.
3. If FFmpeg is installed without a PATH entry, copy **both** `ffmpeg.exe` and `ffprobe.exe` into `tools\ffmpeg\bin` inside this project. Alternatively, add the installed FFmpeg bin folder to Windows PATH and reopen the launcher.
4. Double-click `run_app.bat`. It creates `.venv` on the first run and installs `requirements.txt`. Torch and related packages are large, so the first installation can take several minutes. Subsequent launches reuse the environment; dependencies are installed again when requirements.txt changes.
5. The app opens on the **Speech Signal Processing** welcome screen with the supervisor and team credits. Click **Enter Studio →** (or press Enter on the focused button) to open the workspace. The destination starts empty; choose one before downloading.

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

## Quick start: complete workflow

1. Extract this ZIP into a **new short folder**, for example `C:\SpeechCorpusFull`. This avoids mixing old controllers with the new UI.
2. Double-click `run_app.bat`. It creates or updates one `.venv` using the combined `requirements.txt`. Do not run the teammate's original `app.py` for this integration.
3. Source & Audio: paste a URL, choose a destination, download, then extract audio. You can also select existing source video/audio.
4. Labeling Regions: run automatic labels, inspect the spectrogram and correct labels/boundaries.
5. Segmenting the Speech: generate candidates, listen, approve useful clips, and inspect unresolved spans. Existing approved clips are passed directly to the later pages.
6. Audio Quality SNR: calculate energy SNR, set the threshold and correct quality labels if needed.
7. Transcription: click **Use current approved clips**, then **Transcribe all pending**. The default models are Whisper `large-v3` and MMS `facebook/mms-1b-all`. First use downloads their weights. Whisper runs CPU int8; MMS uses CUDA if available, otherwise CPU.
8. Listen to disagreements. Choose A/B, edit the Arabic final text, accept it or mark the clip difficult. Then use Review & Export for final label/quality/boundary changes.
9. Export the corpus to a new destination. It contains WAV + matching UTF-8 TXT files, `corpus.csv`, `corpus.json`, a readable `corpus_report.html` with audio controls, and source-review provenance.

New dependencies from the teammate's part are `faster-whisper`, `jiwer`, `transformers` and `accelerate`; PyTorch is already part of the earlier requirements. The combined application uses **PyQt6 only**: PySide6, the Tkinter frontend and sounddevice are not needed. The separate `requirements-mms.txt` is a compatibility shortcut to the full requirements, not a second environment.

Test one or two reviewed clips with the real models on your Windows computer before processing an entire recording. Model inference and first-time downloads can take considerable time, especially on CPU. This package contains source code, not model checkpoints or FFmpeg binaries.

## Using the combined app

### Source & Audio

1. Paste a video URL.
2. Choose Destination. An empty destination produces an error before downloading.
3. Enter an optional folder name; leave it blank for automatic `video1`, `video2`, etc.
4. Download. The download-card progress line is hidden; the Activity section contains the visible bar. Its fill and centered text show only the download percentage, without transferred byte counts or sizes. Internally the calculation uses combined downloaded bytes divided by the selected video/audio total bytes, so large video and small audio streams are weighted by size rather than split 50/50. A brief 120–600 ms interpolation smooths measured updates, including larger changes, and stops between updates. Totals inferred from metadata or bitrate are estimates and may be corrected as real headers arrive. The displayed percentage never decreases within one download: it holds at its highest target when a corrected total or stream retry lowers the raw ratio, then continues once measured progress catches up. This also protects an animation still in progress. Hover over the bar for the estimation note. If no total has ever been available, it shows `—%`; after a percentage is known it keeps that percentage and fill through temporary unknown-total updates. A new download resets the bar to zero. Merging and verification keep the bar below full; 100% follows successful completion. Audio extraction keeps its indeterminate activity indicator without a percentage.
5. Extract audio. The verified 16 kHz mono 16-bit PCM WAV is automatically made available to Regions. You may also browse an existing WAV from any processing page.

### Regions — Part 3

1. Enter the reviewer name.
2. Set optional **Window (s)** and **Hop (s)** values, then choose **Label automatically** to classify and build the display. Empty fields use the existing 2-second window and 2-second hop. Show spectrogram only builds the display.
3. On its first run, PANNs downloads its approximately 327 MB checkpoint and label file into your user home `panns_data` folder. Internet access is required for that first download. The first Silero run also needs its model/package resources.
4. Adjacent decision intervals with the same label are automatically merged into longer regions. Speech, music, mixed and silence are shown as colored spans over the spectrogram and as rows with duration-weighted average automatic scores. Individual prediction scores, input bounds, settings and review flags are retained in the saved region data. Use the Matplotlib toolbar to zoom or pan.
5. Select a row or click its plot region. The default **Relabel a range (keep the rest)** mode lets you correct a section inside a merged region: enter its source Start/End in seconds, choose its new label, optionally **Play entered range**, then **Apply correction**. Only this interval receives the new label; the portions before and after keep their existing labels. Example: select speech from 12–180 s, enter 45–60 s and choose music. The result is speech 12–45 s, music 45–60 s, speech 60–180 s. You can repeat this on any resulting piece. The range must lie inside one selected region and have positive length.
6. To move an existing shared boundary, choose **Move whole-region boundaries** before applying a correction; this adjusts the adjacent region too. You can also drag near a shared boundary in the plot and release it at a new position, which selects boundary mode. Turn off toolbar zoom/pan before dragging. Play region still plays the whole selected row; Play entered range previews the times you entered. Split at midpoint and Merge with next remain available.
7. Save HTML report to create a readable report that can be reopened. **Save session** preserves the combined review state.

#### Adjustable PANNs window and hop

- **Window** is the amount of audio used by the main PANNs tagging pass for each prediction: 2–30 seconds.
- **Hop** is the time advanced before the next prediction: 0.1 seconds up to the window length. A larger hop than the window is rejected so no audio is skipped.
- Leave either input empty to use its individual **2.0-second default**. Clearing both restores the original 2/2 behavior. Inputs use decimal seconds, for example Window `6` and Hop `1`.
- With Window 6/Hop 1, model inputs overlap: 0–6 s, 1–7 s, 2–8 s, and so on. Each prediction is assigned to its **leading hop interval**: 0–1 s, 1–2 s, 2–3 s. The last interval is clipped to the source duration. This produces one continuous timeline without duplicate audio spans; it does not average all overlapping predictions.
- Silence is checked on the assigned hop interval, so a later sound inside the wider input does not turn an otherwise silent leading interval into speech/music. Short final model inputs are padded to the selected input duration, and padding is recorded.
- Smaller hops require more inference calls. Longer windows provide more context but may blur a transition; a smaller hop does not prove more accurate boundaries. Overlapping proposals retain review flags. The singing-verification stage keeps its own context and adapts its block length when a chosen hop exceeds eight seconds.
- Saved sessions retain the input fields, including blank defaults. The HTML report displays the settings **actually used by the saved results**, independently of later field edits. Older imported data without settings is marked as not recorded.
- Each original record stores `analysis_config`, `inference_start`, `inference_end` and, when inference runs, `inference_padding_seconds`. Its `start`/`end` describe the assigned hop interval. Region/subrange score summaries are weighted by those decision intervals; the wider model input bounds remain available for provenance.

The new settings apply on the next automatic analysis. Save the current session before reanalysis, which replaces the region review and requires downstream regeneration.

Real PANNs/Hybrid Demucs execution checks also completed with Window 4/Hop 1 and Window 12/Hop 10. They verified parameter handling and continuous sample-aligned coverage; they do not establish the best settings or classification accuracy.

#### Listening slider and range selection

1. Select a merged region. The playback slider covers that region, using absolute source timestamps. Its position follows playback; the display shows the selected listening range and its duration in seconds.
2. Move the slider to the desired beginning and click **Set start here**. Move it to the desired ending and click **Set end here**. You can also type precise Start/End values directly. Marking a range does not apply a label correction.
3. Click **Play entered range** to hear just that interval. **Pause / resume** pauses and resumes source playback. The shared **Stop playback** button stops it.
4. **Play from cursor** listens from the slider position to the entered range end when the cursor is inside that range; otherwise it listens to the selected region end. Dragging during playback seeks with the same end rule and preserves the paused/playing state. Keyboard arrows move by 0.1 s; Page Up/Down move by 1 s. Playback stops at the chosen end and keeps the cursor there.
5. To relabel this range, choose its label and click **Apply correction** in range mode. Selecting another region stops the current source preview and resets the slider to the new region. Preview positions are temporary; saved sessions retain the applied regions and audit history.

Scores describe the original inference windows and are uncalibrated. Subrange scores use overlap-duration weighting of those original windows; a human label does not become a new automatic prediction. Quiet speech may be labeled silence; low speech/music scores are flagged for review. These edits change the region timeline, not the source WAV. Save session and the HTML region report retain the split regions and correction history. Region changes clear downstream segment and SNR results so they must be regenerated; synchronize the new approved clips on Transcription before continuing to ASR/export.

#### Songs, singing and the mixed label

The earlier decision checked only PANNs' `Speech` and `Music` outputs. Singing can have a low Speech score, so a song could be called music even when it contained lyrics. The basic classifier now also reads `Singing`, `Male singing`, `Female singing`, `Child singing`, `Choir`, `Rapping`, `Vocal music` and `A capella` from the same model output. Those scores alone still miss masked vocals in some recordings. The additional verification below addresses that failure without training on the user's song or adding source-specific timestamps.

| Detected evidence | Proposed label |
| --- | --- |
| Speech without music/vocal-music evidence | speech |
| Music without detected speech/vocals | music |
| Speech over music | mixed |
| Strong singing/rap/choir/vocal-music evidence | mixed, needs listening review |
| Music with borderline vocal evidence | music, needs listening review |

This corpus deliberately treats vocal music, including a cappella, as a mixed proposal to preserve it for review and later segmentation. This convention does **not** prove intelligible lyrics, instrumental accompaniment or simultaneous speech and music throughout a window. Wordless vocalizations and false detections can occur. Synthetic singing, humming, beatboxing, genre and instrument scores are not independently used as lyric evidence, although the model can still confuse them with singing.

**Improve singing detection** is enabled by default beside the Part 3 actions. It uses TorchAudio's pretrained **Hybrid Demucs `HDEMUCS_HIGH_MUSDB_PLUS`** to isolate a likely vocal stem in music candidates, then applies the existing PANNs Speech/Singing tagger to that stem. A separator can leak instruments into its vocal output, so stem energy alone never establishes a mixed label. A proposed mixed window requires sufficient local stem energy plus strong voice-tagging evidence, either locally or from the nearby analysis block. Strong short-window detections can support neighboring energetic windows when a full-context score dilutes alternating spoken/sung phrases. All such proposals retain Needs review.

The check normally reads eight-second blocks with one-second edge context and increases the block length for a hop longer than eight seconds. It preserves the source sample grid and does not save or modify audio stems. It skips non-music windows and does not change the raw mixture Speech/Music/Vocal scores. Per-window `separation_check` data records local/full-context/support scores, the exact support interval, energy gates, prior label/confidence and model name. `verified_voice_score` is an **uncalibrated support score**, shown as Separated voice evidence in row tooltips and HTML reports. It does not mean human verification. Merged and cropped regions retain their original per-window evidence.

First use on music downloads an additional **319 MiB** model into the Torch cache; it is reused afterwards. It runs on CUDA when available, otherwise CPU, and increases analysis time. The separator uses the existing torch/torchaudio dependencies; this update also declares the previously implicit PANNs dependencies `librosa` and `torchlibrosa` explicitly. No `demucs` Python package is required. Turn the checkbox off for the faster basic classifier. The choice is preserved in saved sessions. Model download/load errors are reported; enhanced detection does not silently fall back to the basic method.

Local vocal evidence can still be uncertain, and contextual support can blur a transition by several seconds. Listen and use the range editor to correct boundaries. This is a generic improvement for recordings, not a guarantee for every song, singer, language or mix. No threshold was tuned to a particular timestamp or filename. See **SINGING_DETECTION_NOTES.md** for real-inference observations and limitations.

Raw Speech and Music scores remain separate. `vocal_score` is the maximum of the selected vocal class scores; `voice_score` and `music_evidence_score` combine the available evidence for this labeling policy. The original per-class scores and decision-rule identifiers (`panns_speech_music_vocals_v2` or `panns_demucs_vocals_v3`) are saved per analysis window. Merged/subrange scores are duration-weighted summaries, not fresh model predictions. Hover over a Regions table row for its scores and explanation; the HTML report includes Vocal score and Vocal category columns. Older reports remain readable and show unavailable vocal scores.

`SPEECH_THRESHOLD`, `MUSIC_THRESHOLD` and `VOCAL_THRESHOLD` are initially 0.30; the borderline-vocal review threshold is 0.15 in `step3_label_regions.py`. These are heuristic thresholds, not validated accuracy guarantees. Check real Arabic song/instrumental examples before tuning them.

Existing saved labels do not change merely by updating the app. **Save your current session first**, then run **Label automatically** again on the source WAV to obtain vocal-aware predictions. Automatic reanalysis replaces the region review; regenerate segments/SNR and resynchronize approved clips for transcription afterwards.

References: [AudioSet Singing](https://research.google.com/audioset/ontology/singing.html), [AudioSet Vocal music](https://research.google.com/audioset/ontology/vocal_music_1.html), [TorchAudio Hybrid Demucs](https://docs.pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html).

### Segmenting the Speech — Part 4 (PDF requirements)

The project PDF, Step 4 on page 2, requires 5–15 second segments, cuts at real pauses between complete words/sentences, and about 0.1–0.2 seconds of silence at **each edge**. It also requires human override and listening around cut points. These requirements take priority over producing a high automatic clip count.

1. Choose **Real pauses · continuous audio (PDF)** and Generate candidates. Normal generation cuts original source audio only. It does not stitch distant spans, force a cut to satisfy the length limit, or add synthetic silence.
2. Speech-only regions use Silero VAD with 240 ms pauses, corroborated by low source-frame energy. A VAD-negative interval that still contains active audio is retained for review rather than treated as missing speech.
3. Mixed regions reuse Part 3's **HDEMUCS_HIGH_MUSDB_PLUS** model to inspect vocals while backing music continues. It is loaded on generation, not app launch; first use downloads about 319 MiB if it is not already cached. No new Python dependency is needed.
4. Vocal-pause proposals require at least **240 ms of consecutive raw low-energy frames**, a 12 dB drop relative to local vocal activity and a 12 dB contrast against available vocal flanks. Raw consonant/onset spikes cannot be smoothed away. Levels below −70 dBFS do not establish an active vocal flank. Leading/trailing context may be proposed with evidence on the available side; it is still an estimate.
5. The planner keeps **0.1–0.2 s of original estimated pause** at both candidate edges, targeting 0.15 s. Short shared pauses are divided to prevent overlap; unnecessary long pause material remains in the source and review list. The 5–15 s limit includes these edges. A file/Step 3 boundary with insufficient context remains unresolved. In mixed audio, “pause” means estimated absence of speech/vocals; the music may continue, and the exported waveform is never falsely described as digital silence.
6. Long continuous singing, uncertain word boundaries, short remnants or sections without enough natural edge context remain intact under Unresolved. Use the listening sliders to mark source times, add a 5–15 s manual clip or correct the region. The original recording and unused intervals remain available. Review the intervals labeled estimated pause as well: a quiet word may still be missed by a detector.
7. **Approve selected** means the named reviewer has listened, confirmed complete words and checked about 0.1–0.2 s of natural pause at both edges. This is recorded as `pdf_boundary_review` with the current source range/label. Manual edits remain possible; the annotator is responsible for checking their edges. Changed approved ranges require approval again.
8. Old joined/artificially padded candidates remain readable for inspection. They cannot be approved, handed to ASR or exported through Part 4 until converted to a valid continuous original-source interval and reviewed. Choose **Original source times** in the boundary editor, correct the cuts, Apply edit, listen and approve again. Old saved approvals without the current boundary review require reapproval.
9. Export approved WAVs writes separate original-source clips plus a review session, HTML report, playlists and Step 5 manifest. Overlaps still block export/SNR. The old compact generator's public entry point now delegates to this same planner. Later final-review edits cannot export non-contiguous joins or synthetic edge samples as corpus clips.

Natural edge context is detector-estimated until the annotator listens. VAD and energy models do **not** prove word boundaries. For the supplied song and saved labels, the stricter planner initially formed three automatic candidates with 0.12–0.15 s edge context; other spans remained unresolved. That is an execution check, not a claim of accurate cuts in every word/song. The earlier 21-candidate result used the now-retired 80 ms dip rule and did not meet the intended edge-context checks.

The JSON stores the PDF rule, source spans, pause intervals, per-edge retained context and singing-check evidence. No source-specific lyrics, filenames or timestamps are built into the algorithm. If separation is unavailable/fails, the log reports a conservative source-audio fallback; uncertain sections remain for review. **Restart and regenerate candidates** after applying this update. Save an old review session separately if needed; new cuts invalidate old SNR/transcription results.

#### Listening sliders for candidates and unresolved spans

1. Select exactly one candidate or unresolved span. Its listening panel enables automatically. Multiple candidate selection remains available for approval/exclusion; the slider stays disabled until one candidate is selected.
2. **Clip time (joins preserved)** plays the generated candidate WAV. Its slider covers that complete candidate, so trimmed material can be restored. **Source time** plays the original recording: candidates include two seconds of boundary context on each side; unresolved spans cover their current full interval. Switching the boundary mode resets the inputs to that mode's coordinate system.
3. Move the slider and click **Set start here**, then move it and click **Set end here**. The summary displays the marked timestamps and duration. These marks do not save an edit.
4. **Play clip** or **Play span** previews the entered range. **Play from cursor** plays to the entered end when the cursor is within the valid marked range; otherwise it plays to the listening interval's end. Seeking during playback preserves the paused/playing state. The cursor follows playback and stays at its final position after stopping. Arrow keys move by 0.1 s and Page Up/Down by 1 s.
5. For a candidate, choose its label and click **Apply edit** to commit the trim or source-boundary edit. It returns to pending approval, and old SNR/transcription results are invalidated. Saved clips must remain 5–15 seconds.
6. For an unresolved span, click **Add 5–15 s clip** to create a pending manual candidate, or **Label + mark reviewed** to review the marked subsection. Reviewing a subsection preserves both remaining sides as pending spans. Long spans can be previewed without first cutting them into clips.

Playback positions and unsaved marks are temporary. Saved sessions retain committed boundaries, labels and review history.

### Audio Quality SNR — Part 5

The PDF requires **two methods** and a clean/noisy threshold starting at **20 dB**, with a justified final choice. The GUI now calculates **pause-frame energy + WADA-SNR** on the existing approved clips. No reselecting WAVs, additional model download or new dependency is needed. Keep the bundled `assets/wada` folder with the application.

1. Approve source clips after listening to their natural pauses. Open Audio Quality SNR and click **Calculate SNR**. Both estimators run in a worker thread. Pause metadata from Part 4 supports the energy noise estimate; digital zeros are not an acoustic noise floor.
2. Inspect **Energy SNR**, **WADA SNR**, **Decision SNR**, quality labels and **Review** status. The comparison histogram displays both methods. Silence, missing pause evidence, out-of-table WADA values, variable noise and other uncertain cases remain explicit. Mixed/music results are flagged because speech-based assumptions can fail.
3. Start with **20 dB**. The default decision rule uses the lower valid estimate. You may choose energy or WADA specifically; both measurements remain visible. Changing threshold/rule reclassifies cached estimates without rereading audio or changing a named human correction. The noise-power estimate may vary between observed pauses; the classification threshold does not follow loudness.
4. Select one or more rows, enter a correction reason and reviewer, and **Mark clean**, **Mark noisy** or **Restore automatic**. Human corrections, original automatic labels and the audit are preserved. Recomputing identical audio also keeps named corrections; editing samples invalidates old SNR.
5. Enter a **threshold justification** based on listening/evaluation, then **Save HTML / CSV / JSON**. Both estimates, range flags, chosen rule, threshold, explanation and reviewed quality are saved. Save/open the complete session to resume all pages. Legacy energy-only reports load with an explicit recalculation warning.
6. Use current approved clips on Transcription; both SNR values and quality-review metadata continue through ASR and final corpus CSV/JSON. Final-review SNR recalculation also runs both methods. Full corpus export requires current two-method analysis, a usable decision value, completed quality review and a threshold explanation. Intermediate partial export remains available and marked.

**These are blind estimates, not reference measurements.** A 20 dB threshold and agreement between estimators do not prove quality. The team must justify the final threshold and evaluate the required real videos. See **SNR_METHODS.md** for formulas, assumptions, exact WADA-table provenance/licensing and verification against all 12 CMU reference examples.

### Save and continue later

Click Save session on a processing page. The JSON stores the selected source path, region labels/correction history, Step 4 review, SNR results/quality corrections, reviewer and threshold. Open session restores them. Rebuild the spectrogram with Show spectrogram.

Keep the original source and candidate folders at their saved locations. Session files reference audio; they do not embed or copy it. Save prompts appear before discarding a dirty review or closing while idle. If processing is running, closing shows OK / Cancel: Cancel keeps processing; OK terminates the app and its processing subprocesses. Unsaved changes are lost and incomplete output files may remain after confirmed termination. Rename a video folder before loading it into Regions.

### Transcription — two independent Arabic ASR systems

- Use current approved clips transfers all approved clip records and current SNR/quality into Transcription. Both clean and noisy speech/mixed clips are transcribed; music/silence are retained for export but excluded from ASR.
- The exact reviewed continuous original-source ranges are used. Part 4 revalidates source continuity and the human boundary review before ASR handoff. Artificial padding and non-contiguous joins cannot bypass this check through an old session.
- Transcribe selected processes Ctrl/Shift-selected clips again with both models. Transcribe all pending runs only missing engine results. An empty model transcript is recorded as a completed attempt and remains for human review.
- Models run one phase at a time: Whisper jobs, then MMS jobs. Each model is released before loading the next. MMS explicitly sets both its tokenizer and adapter to Arabic `ara`.
- Step 6 normalization follows the project PDF (pages 2–3): remove Arabic diacritics/tatweel, fold أ/إ/آ/ٱ to ا, ة to ه and ى to ي, remove punctuation and normalize whitespace. Unicode presentation forms are converted and invisible formatting controls removed. ؤ/ئ/ء and dialect words are preserved. These are comparison copies; the original model outputs and chosen/edited final spelling are retained.
- The GUI shows both normalized copies under the original model outputs, plus normalized CER/WER, word edits and the automatic decision reason. `jiwer>=4,<5` performs the comparison, with A as reference and B as hypothesis. WER can exceed 100%; it measures inter-model disagreement, not accuracy against a human reference.
- Identical **nonempty** normalized text or CER <= 2% is accepted automatically. Otherwise at most 3 word edits (substitutions + insertions + deletions) requires listening and choosing A/B or editing before acceptance. More than 3 edits is difficult. Empty/punctuation-only outputs never become automatically accepted. Human decisions retain their reviewer/audit record.
- Opening older complete sessions or synchronizing current clips refreshes comparisons from their original transcripts without rerunning models. Human transcript decisions are preserved. Raw/normalized text, normalization/rule versions, jiwer counts and decision reason are saved in session/CSV/JSON. See **ASR_STEP6_RULES.md** for the PDF mapping and examples.
- Scores are heuristic confidence estimates, not calibrated probabilities. The model A/B text boxes are read-only to preserve original outputs; the final Arabic text is editable with native Qt right-to-left behavior.
- Stop transcription appears in the shared status area during work. It finishes the current model/clip, preserves its result, and cancels remaining jobs. Closing the app during work uses the existing OK/Cancel termination warning.
- After every model result, a full resumable session is atomically saved beside the source WAV as `<source>_full_autosave.json`, before moving to the next job. Use Open saved session or the shared Open session button to restore it. Previous source JSON approval statuses are not overwritten.

### Review & Export — final corpus annotation

- Edit final Arabic text, labels and clean/noisy decisions. Enter your reviewer name; manual quality changes also require a reason. Human changes and original automatic results are retained separately.
- Changing source start/end deliberately replaces that clip with a continuous source interval. A joined clip retains its source-span mapping when times stay unchanged. All resulting clips must be 5–15 seconds.
- Split uses a **clip-local** position and requires at least 5 seconds on both sides. Merge concatenates selected non-overlapping clips only if their combined audio is 5–15 seconds. Add creates a 5–15-second source range; Delete records the removal in the audit log.
- Moving/splitting/merging audio clears old transcripts, confidences and SNR because they describe different samples. Recalculate selected SNR in Review & Export, then return to Transcription for the affected clips. These final-review edits operate on the final corpus records, preserving the earlier Step 4 review as provenance.
- Export normally requires both ASR attempts, resolved transcript decisions and clean/noisy labels. Accepted speech must have non-empty final text. All final source spans must be non-overlapping. Remaining source candidates/unresolved spans also need review.
- To save intermediate results deliberately, enable **Allow partial export**. Pending decisions and source-review counts are explicitly stored. `partial=false` requires the current two-method SNR and quality review as well as transcript/source review. It does not certify real-world model accuracy.
- Final export creates one WAV and one TXT per clip, one CSV per current source recording, JSON with both transcripts and full mappings, an HTML report and `source_review.json` with earlier changes. Difficult/non-speech clips may have empty TXT files; their decisions and both ASR texts remain in CSV/JSON.
- Use the same workflow on the three required long videos and keep each source's project/export separately. This integration does not claim that those three videos have already been processed.

## Editing the UI in Qt Designer

**The welcome screen and all six workspace pages are in one file: `main_window.ui`.** Their layouts, text, processing theme, button colors and table headings remain editable in Qt Designer. The decorative background is the packaged `assets/welcome_signal_background.svg`; keep the assets folder beside the UI file. It contains an illustrative synthetic waveform and spectrum, not a user's recording.

1. Open `main_window.ui` in Designer.
2. In Object Inspector, find the outer `appScreens` QStackedWidget. `currentIndex=0` is the welcome screen and `currentIndex=1` is the existing workspace.
3. On the welcome screen, edit `welcomeTitle`, `welcomeSupervisor`, `welcomeTeamNames`, the surrounding captions and `enterStudioButton`. The translucent card and background are `welcomeCard` and `welcomeBackground`.
4. To edit a processing page, first select `appScreens` index 1, then find the inner `pages` QStackedWidget. Its indices remain: 0 Source & Audio, 1 Labeling Regions, 2 Segmenting the Speech, 3 Audio Quality SNR, 4 Transcription, 5 Review & Export.
5. Shared audio/reviewer/session controls are in `sharedControlsFrame`; shared operation status and the ASR stop button are in `processingStatusFrame`. They are shown on processing pages 1–5 at runtime. Their `visible` property is initially false so Source & Audio stays clean; select them in Object Inspector to edit them.
6. `spectrogramHost` and `snrChartHost` are the chart placeholders. Move/resize these in Designer; Python draws the data plots inside their layouts at runtime.
7. Save, then restart the app to see changes. Keep widget/layout object names unchanged because the controllers refer to them.

Welcome credits are **DR Mohsen Rashwan**, and **Ameer Louly, Robir Tamer and Saeed Maher**. Enter Studio switches pages in the same window and retains the existing controllers; it does not launch a second application. The background resolves relative to the project folder even when launched from another working directory. `MainWindow(show_welcome=False)` is available for direct integration/testing of the workspace; ordinary launches show the welcome screen.

`qt_app.py` loads this single UI file; `qt_processing.py` connects the earlier processing functions and `qt_asr.py` connects transcription/final review. The teammate's `pipeline.py` is named `asr_pipeline.py` here so it does not replace the download/extraction `pipeline.py`. `corpus_review.py` handles the exact-audio handoff, final edits and export. No UI-to-Python conversion is required. The former `processing_pages.ui` file is no longer used and can be deleted from an older project folder after applying this update. Python processing functions remain separate from the editable interface.

## Calling the processing functions

```python
from pathlib import Path
from step3_label_regions import classify_file
from step4_cut_segments import export_candidates
import step4_review as review
import step5_snr as snr

wav = Path('path/to/audio.wav')
regions = classify_file(wav, progress=print, window_seconds=2, hop_seconds=2)
result = export_candidates(wav, regions, wav.parent, progress=print)
session = review.load_session(Path(result['output_dir']) / 'step4_segments.json', wav)

# Review/listen first; approve only the indices actually checked.
review.approve_segments(session, [0], reviewer='Reviewer name')
records = snr.session_records(session, wav)
justification = "Your listening-based justification for the final threshold"
results = snr.calculate(records, threshold=20, policy="conservative", justification=justification)
html_path, json_path, csv_path = snr.export_report(results, 20, wav.parent,
                                              policy="conservative", justification=justification)
```

Run file/model operations in a worker when calling them from Qt. `Task` in `qt_processing.py` shows the QThread/signals pattern. Widget access belongs on the Qt main thread.

## Integration review and verification

See **INTEGRATION_NOTES.md** for the review of the teammate's archive, fixes, tests and remaining requirements. The included automated tests use controlled ASR outputs; they do not establish actual model accuracy or validate target-machine installation.

`tests/check_welcome.py` checks default welcome startup, SVG loading from another working directory, minimum-window-size fit, mouse/Enter entry and navigation through all six workspace pages. Existing workflow checks start directly in the workspace with `show_welcome=False`.

```powershell
.\.venv\Scripts\python.exe tests\test_asr_core.py
.\.venv\Scripts\python.exe tests\test_pdf_asr_rules.py
.\.venv\Scripts\python.exe tests\check_full_workflow.py
```

`check_full_workflow.py` covers exact continuous original-source audio/natural context, two-engine orchestration, Arabic disagreement review, stop/resume, per-result autosaving, complete session restoration, WAV/TXT/CSV export and structural edits.

`tests/check_region_ranges.py` checks Part 3 subrange edits, edge/full-region corrections, sample alignment, original-score preservation, repeated edits, invalid-range warnings, actual Qt button connections, downstream invalidation and saved HTML/session restoration:

```powershell
.\.venv\Scripts\python.exe tests\check_region_ranges.py
.\.venv\Scripts\python.exe tests\check_region_playback.py
```

`check_region_playback.py` checks the listening slider, marked times/duration, seek while loading, playback position updates, pause/resume, drag/release, stopping at the range end and isolation from clip-local playback on other pages. It uses a controlled media player; test actual audio-device playback on your Windows computer.

`tests/check_segment_playback.py` checks the Part 4 candidate/unresolved sliders, clip/source time switching, preserved compact joins, marked-range playback, asynchronous seek, pause/resume, drag/release, source-boundary edits, manual clip creation, preserved unresolved leftovers, multiple selection and invalid-range warnings. Playback uses a controlled player; actual Windows audio-device playback still needs a local check.

```powershell
.\.venv\Scripts\python.exe tests\check_segment_playback.py
```

`tests/test_mixed_continuity.py` checks sustained singing-like activity, complete VAD misses, quiet singing, actual quiet gaps, brief note dips, source-label isolation, analysis-block seams, vocal-pause detection under continuing backing audio, separator failure, both exporters, unchanged source WAVs and explicit fallback/unresolved handling. Synthetic signals and controlled model outputs test preservation behavior rather than model accuracy:

```powershell
.\.venv\Scripts\python.exe tests\test_mixed_continuity.py
```

`tests/test_vocal_phrase_cuts.py` checks sustained relative vocal pauses, gain invariance, raw onset rejection and original-source sample equality. `tests/test_pdf_part4.py` checks the PDF contract: 5–15 s including 0.1–0.2 s real source context at each estimated edge, no overlapping short-pause padding, preservation of VAD-missed words, no forced cuts, compatibility entry points, approval provenance, legacy join/padding rejection, reapproval after edits and exact source coordinates at ASR/SNR handoff. Actual word-boundary accuracy still requires listening; these tests use controlled audio/model outputs.

```powershell
.\.venv\Scripts\python.exe tests\test_pdf_part4.py
.\.venv\Scripts\python.exe tests\test_vocal_phrase_cuts.py
```

`tests/test_vocal_labeling.py` checks vocal-aware decisions for songs with low Speech scores, every supported vocal category with shuffled indices, instrumental controls, review flags, merged/subrange score provenance and HTML round trips. Model scores are controlled test inputs, not recordings of real songs:

```powershell
.\.venv\Scripts\python.exe tests\test_vocal_labeling.py
```

`tests/test_panns_windows.py` checks blank defaults, invalid settings, actual model input lengths with controlled scores, overlapping input bounds, gap-free hop coverage, final padding and local silence. `tests/test_vocal_verification.py` checks energy/voice gates, contextual support, bounded reads, long-hop handling, score provenance and invalid output rejection without downloading a model. `tests/check_singing_ui.py` checks the worker option and saved-session setting.

```powershell
.\.venv\Scripts\python.exe tests\test_panns_windows.py
.\.venv\Scripts\python.exe tests\test_vocal_verification.py
.\.venv\Scripts\python.exe tests\check_singing_ui.py
```

## Two-method SNR checks

```powershell
.\.venv\Scripts\python.exe tests\test_snr_two_methods.py
.\.venv\Scripts\python.exe tests\check_snr_two_methods_ui.py
```

These cover known-noise numerical behavior, WADA bounds, silent/invalid input, policy changes, persistent human corrections, both reports, session restore and final corpus metadata. The CMU single-block comparison is documented in SNR_METHODS.md; no external C++ compiler is required to run the app.

## Earlier processing checks

`tests/check_download_bytes.py` checks byte weighting, unknown totals, estimate corrections and resumed transfers. `tests/check_download_fill.py` checks that the visible bar follows the measured progress and stops between updates. `tests/check_parts35.py` exercises synthetic audio through the Qt workflow with a controlled VAD stub. `tests/check_processing_edges.py` checks classifier branches using a fake model and tests spectrogram/SNR edge cases. Run from the extracted project folder:

```powershell
.\.venv\Scripts\python.exe tests\check_parts35.py
.\.venv\Scripts\python.exe tests\check_processing_edges.py
```

Most tests use synthetic fixtures and do not prove model accuracy. The singing update also ran real PANNs/Hybrid Demucs inference on the supplied WAV and public control excerpts; these are small observational checks, not a labeled accuracy benchmark. See SINGING_DETECTION_NOTES.md. Windows audio-device playback and real Silero boundary accuracy remain unvalidated. REVIEW.md records what was and was not checked. AI assistance was used for the Qt integration, documentation and checks; team members should understand and explain the code during the demo.

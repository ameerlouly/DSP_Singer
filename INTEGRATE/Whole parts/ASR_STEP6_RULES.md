# Step 6 — ASR and Arabic comparison rules

Checked against **Course_Project_Building_a_Labelled_Speech_Corpus_V1(3).pdf**, Step 6 on pages 2–3. The PDF requires two different ASR systems, normalization before comparison, jiwer WER, the agreement/word-difference decision table, and human choice/correction for small disagreements. It allows other engines if the initial pair does not perform well.

## Two ASR systems

The existing integration runs Whisper large-v3 and Meta MMS on every current approved speech/mixed clip, including **both clean and noisy** clips. Music/silence records remain in the corpus but are excluded from transcription. Whisper explicitly uses Arabic `language="ar"` and `task="transcribe"`. MMS loads its `ara` tokenizer and adapter. Both read the exact reviewed 16 kHz original audio.

This update does not change the selected model weights, Whisper's existing VAD/beam configuration, or claim to fix Egyptian-dialect or singing accuracy. It makes the PDF comparison rules explicit and inspectable. A dialect-tuned replacement still needs comparison against human reference transcripts on representative speech. Raw outputs from the two systems remain independent; neither transcript is supplied to the other model.

## Letter normalization

| Input | Comparison copy |
|---|---|
| Arabic diacritics, including extended/Quranic combining marks | Removed |
| Tatweel `ـ` | Removed |
| Alef forms `أ إ آ ٱ` | `ا` |
| `ة ه` | `ه` |
| `ى ي` | `ي` |
| Arabic and other Unicode punctuation | Word boundary, then whitespace collapsed |
| Arabic presentation forms / lam-alef ligatures | Standard Unicode letter forms (NFKC) |
| Invisible formatting/direction controls | Removed |

The core letter/diacritic/punctuation rules are from the PDF. Unicode glyph/control handling is a technical extension so visually equivalent text compares consistently; it does not rewrite dialect vocabulary. `ؤ`, `ئ` and standalone `ء` remain distinct; the PDF's hamza rule concerns the specified alef forms. No stemming, translating dialect to formal Arabic, or replacing similar-sounding words is applied.

Examples:

- `إِنَّ السَّلَامَـ، عَلَيْكُمْ!` → `ان السلام عليكم`
- `مَدْرَسَة هدى` → `مدرسه هدي`
- `إزّيك؟ مش عايزة أروح دلوقتي!` → `ازيك مش عايزه اروح دلوقتي`

**Only comparison copies are normalized.** The original A/B text stays read-only and unchanged. The final chosen transcript uses its original spelling, or the annotator's edited spelling. Normalization does not correct misheard words or establish true recognition accuracy.

## Scores and status

`jiwer>=4,<5` calculates normalized CER, WER and the minimum word-edit alignment. A is the reference; B is the hypothesis. Substitutions, insertions and deletions all count as word edits. WER = (S + I + D) / reference word count when the reference has words. The score is directional and can exceed 100%. With an empty reference, jiwer 4 reports insertion counts using its documented empty-reference convention, rather than dividing by zero.

Apply the PDF decision table in this order:

1. Both normalized strings nonempty and either identical or CER ≤ 2%: **accepted** automatically, with A's original spelling as the initial final transcript.
2. Otherwise, at most 3 word edits: **needs_review**. The named annotator listens, chooses A/B or edits, then accepts. The app does not automatically accept a near match.
3. More than 3 word edits: **difficult**. The annotator can still override after listening.

Empty or punctuation/diacritic-only outputs never qualify for automatic acceptance, even when both normalized strings are empty and jiwer scores are zero. One-sided empty output follows the word-edit review/difficult rule. Pending model attempts have no comparison scores.

These are **agreement metrics between systems**, not WER/CER against a human gold transcript. Both systems can agree on the same wrong words. The PDF also requires human evaluation of accepted transcripts on the required videos.

## GUI, persistence and export

Select one clip on **Transcription**:

- Original A/B outputs appear above read-only **Normalized for comparison** copies.
- The comparison line shows CER/WER as percentages, word edits, the decision reason and heuristic model scores.
- Play the clip, then use **Use A**, **Use B** or edit final text and accept. Original outputs remain available.
- Save the complete session. Older complete sessions are recomputed from their original transcripts on open; named human decisions survive this refresh. Reopening does not download or rerun a model.

Sessions, final CSV/JSON and corpus metadata retain normalized copies, raw outputs, normalization/rule versions, jiwer scores, S/I/D counts and the decision reason. Corpus export refreshes automatic comparisons before checking review completion. Audio edits invalidate old comparison evidence together with transcripts/SNR. The final TXT retains the annotator's chosen spelling.

## Installation and verification

Replace the update files, including `main_window.ui` and `requirements.txt`, then run `start_app.bat`. The existing launcher notices changed requirements and installs jiwer 4 in the existing project environment. The new comparison fields do not require another ASR model download.

```powershell
.\.venv\Scripts\python.exe tests\test_pdf_asr_rules.py
.\.venv\Scripts\python.exe tests\test_asr_core.py
.\.venv\Scripts\python.exe tests\check_full_workflow.py
```

Tests cover the PDF normalization examples, presentation/decomposed letters, preserved Egyptian wording, punctuation-only output, CER exactly at/above 2%, three/four word edits, insertions/deletions, human overrides, both clean/noisy ASR jobs, read-only normalized GUI text, legacy-session refresh, raw/normalized CSV and existing ASR stop/resume/autosave/export behavior. The Qt ASR models are test doubles. These checks establish implementation behavior, not real model accuracy.

References: project PDF pages 2–3; [jiwer documentation](https://jitsi.github.io/jiwer/); [faster-whisper](https://github.com/SYSTRAN/faster-whisper); [MMS model card](https://huggingface.co/facebook/mms-1b-all).

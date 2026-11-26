# Arabic transcription and agreement

System A is Qwen and system B is Cohere Arabic INT8. The two backends decode the same eligible audio independently. Both clean and noisy speech or mixed clips can enter transcription. Unresolved speech, mixed and unclassified intervals retain their segmentation status. Long unresolved intervals are decoded in smaller windows and retain one source interval for review.

## Comparison

`transcript_agreement_and_io.py` makes a comparison copy of each transcript. It normalizes Unicode presentation forms, removes Arabic diacritics, tatweel and invisible direction controls, folds alef variants to ا, ة to ه and ى to ي, and normalizes punctuation and whitespace. Listed alternate spellings can reconcile equivalent renderings such as a term spelled in different scripts. The editable terms are in `asr_spellings.csv`. The raw transcripts retain their spelling.

The GUI shows two directional character error rates. Qwen's score compares Qwen against Cohere's normalized text as reference. Cohere's score uses Qwen's normalized text as reference. Their denominators can differ. These scores measure disagreement between recognizers. Measuring recognition accuracy requires a human reference.

## Automatic status and final text

| Status | Rule |
|---|---|
| Accepted | Both comparison texts contain content and are identical or the CER using Qwen as reference is at most 4%. The raw Cohere transcript becomes the automatic final text. |
| Needs review | The acceptance rule did not pass, and substitutions, insertions and deletions together fit the word-edit limit. Listen, select a model or edit the transcript. |
| Difficult | The word-edit count exceeds the review limit. A reviewer can revisit the decision after listening. |

The base limit is four word edits. For unresolved spans longer than 15 seconds, each additional complete five seconds adds one edit. A 25-second unresolved span therefore has a limit of six. Empty outputs cannot be accepted automatically.

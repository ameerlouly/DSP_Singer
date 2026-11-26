"""Compare Arabic transcripts and save decisions with their source times.

normalize_arabic makes a comparison copy, comparison_details scores the two models'
outputs, and the manifest helpers preserve the link to the audio. Project and CSV
writers keep the raw texts, final edit and review status together.
"""

from __future__ import annotations

import csv
import copy
import json
import math
import os
import re
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

NORMALIZATION_VERSION = "pdf_step6_arabic_known_spellings_v3"
AGREEMENT_RULE_VERSION = "cohere_agreement_v5"
MAX_ACCEPTED_CER = 0.04
BASE_REVIEW_WORD_EDITS = 4
BASE_CLIP_SECONDS = 15
EXTRA_WORD_SECONDS = 5
COMPARISON_FIELDS = (
    "normalized_transcript_a", "normalized_transcript_b", "normalization_version",
    "agreement_rule_version", "comparison_engine", "comparison_reference",
    "automatic_status", "agreement_reason", "cer", "wer", "word_difference",
    "word_substitutions", "word_insertions", "word_deletions",
    "cer_a_vs_b", "cer_b_vs_a", "cer_a_reference", "cer_b_reference",
    "automatic_choice", "automatic_transcript", "automatic_choice_reason", "review_word_edit_limit",
)
from qwen_asr import MODEL_ID as DEFAULT_QWEN_MODEL, load_qwen_model, transcribe_qwen
from cohere_asr import (DEFAULT_COHERE_MODEL, COHERE_ENGINE, load_cohere_model,
                        transcribe_cohere, release_asr_model)

ARABIC_LETTER_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي",
})
SUPPORTED_LABELS = {"speech", "music", "mixed", "silence"}
SUPPORTED_STATUSES = {"accepted", "needs_review", "difficult"}


def normalize_arabic(text: str) -> str:
    """Build a copy for comparing Arabic text while keeping the original transcript.

    Expand presentation forms such as lam-alef, then fold the alef variants.
    Keep standalone ء and the hamza in ؤ and ئ distinct."""
    normalized = unicodedata.normalize("NFKC", text or "")
    characters = []
    for character in normalized:
        category = unicodedata.category(character)
        arabic_mark = category.startswith("M") and "ARABIC" in unicodedata.name(character, "")
        if character == "\u0640" or arabic_mark or category == "Cf":
            continue
        # Use a boundary for punctuation so adjacent words do not get glued.
        characters.append(" " if category.startswith("P") else character)
    return " ".join("".join(characters).translate(ARABIC_LETTER_MAP).split())


# Explicit spellings only. Keep ambiguous standalone Arabic words out of this list.
TERM_SPELLINGS = {
    "COP28": ("كوب ثمانية وعشرين", "كوب ثمانية وعشرون", "كوب تمانية وعشرين",
              "كوب ٢٨", "كوب 28", "COP 28"),
    "YouTube": ("يوتيوب",),
    "WiFi": ("واي فاي", "Wi Fi"),
    "Bluetooth": ("بلوتوث",),
}


@lru_cache(maxsize=1)
def load_term_spellings():
    """Read optional team-supplied spellings once per app run."""
    terms = {term: list(aliases) for term, aliases in TERM_SPELLINGS.items()}
    path = Path(__file__).with_name("asr_spellings.csv")
    if not path.exists():
        return terms
    owners = {normalize_arabic(word).casefold(): term
              for term, aliases in terms.items() for word in [term, *aliases]}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["term", "spelling"]:
            raise ValueError("asr_spellings.csv needs the header term,spelling.")
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            if len(row) != 2:
                raise ValueError(f"asr_spellings.csv line {reader.line_num} needs two columns.")
            term, alias = (cell.strip() for cell in row)
            if not term or term != normalize_arabic(term) or not normalize_arabic(alias):
                raise ValueError(f"Invalid term or spelling on asr_spellings.csv line {reader.line_num}.")
            key = term.casefold()
            term = next((name for name in terms if name.casefold() == key), term)
            for spelling in (term, alias):
                owner = owners.setdefault(normalize_arabic(spelling).casefold(), term)
                if owner != term:
                    raise ValueError(f"Conflicting spelling on asr_spellings.csv line {reader.line_num}: {spelling}")
            terms.setdefault(term, []).append(alias)
    return terms


def comparison_copies(first, second):
    """Match listed spellings only when both complete texts agree after substitution."""
    left, right = normalize_arabic(first), normalize_arabic(second)
    original = left, right
    matched = []
    for term, variants in load_term_spellings().items():
        english = re.compile(r"(?<!\S)" + re.escape(term) + r"(?!\S)", re.I)
        alternatives = sorted({normalize_arabic(v) for v in variants}, key=len, reverse=True)
        alias = re.compile(r"(?<!\S)(?:" + "|".join(map(re.escape, alternatives)) + r")(?!\S)", re.I)
        if ((english.search(left) and alias.search(right)) or
                (english.search(right) and alias.search(left))):
            left = english.sub(term, alias.sub(term, left))
            right = english.sub(term, alias.sub(term, right))
            matched.append(term)
    if matched and left == right:
        return left, right, matched
    return original[0], original[1], []


# Keep the comparison settings with the scores so a saved decision can be explained later.
def comparison_policy() -> dict[str, Any]:
    return dict(normalization_version=NORMALIZATION_VERSION,
                agreement_rule_version=AGREEMENT_RULE_VERSION,
                engine="jiwer", reference="transcript_a",
                maximum_accepted_cer=MAX_ACCEPTED_CER, base_review_word_edits=BASE_REVIEW_WORD_EDITS,
                unresolved_base_duration_seconds=BASE_CLIP_SECONDS,
                unresolved_full_seconds_per_extra_word=EXTRA_WORD_SECONDS,
                letter_map={"أ/إ/آ/ٱ": "ا", "ة": "ه", "ى": "ي"},
                remove=["Arabic diacritics", "tatweel", "punctuation", "invisible format controls"],
                known_spellings=load_term_spellings(), spelling_match_requires_full_agreement=True,
                unicode_form="NFKC", preserve_raw_transcripts=True,
                cer_a_reference="normalized_transcript_b",
                cer_b_reference="normalized_transcript_a",
                automatic_agreement_tie_break="b",
                choice_basis="PDF agreement rule; directional CER does not rank model accuracy")


def migrate_second_engine(records: list[dict[str, Any]], desired_model: str) -> int:
    """Archive old B evidence before using Cohere; keep named human choices.

    A completed MMS result (or an unknown old engine) cannot become a Cohere
    result by renaming the GUI. The original remains in the saved history.
    """
    changed = 0
    for record in records:
        models = record.setdefault('models', {})
        previous_model = models.get('b')
        current = previous_model == desired_model and record.get('asr_b_engine') == COHERE_ENGINE
        if current:
            continue
        has_result = bool(record.get('asr_b_done') or record.get('transcript_b'))
        if has_result:
            evidence = {key: copy.deepcopy(record.get(key)) for key in (
                'transcript_b', 'confidence_b', 'asr_b_done', 'asr_b_engine',
                'status', 'automatic_status', 'automatic_choice', 'automatic_transcript',
                'final_transcript', 'human_override', 'updated_by', *COMPARISON_FIELDS,
            )}
            evidence.update(model=previous_model, archived_utc=datetime.now(timezone.utc).isoformat(),
                            reason='System B changed; the old transcript is not a Cohere result.')
            record.setdefault('legacy_asr_results', []).append(evidence)
            changed += 1
        record.update(asr_b_done=False, transcript_b='', confidence_b=None, asr_b_engine=None)
        models.pop('b', None)
        if isinstance(record.get('asr_errors'), dict):record['asr_errors'].pop('b', None)
    return changed


def migrate_first_engine(records, desired_model):
    """Archive previous Model A evidence before running Qwen, retaining human edits."""
    changed = 0
    for record in records:
        models = record.setdefault('models', {})
        if models.get('a') == desired_model and record.get('asr_a_engine') == 'qwen_transformers':
            continue
        if record.get('asr_a_done') or record.get('transcript_a'):
            evidence = {key: copy.deepcopy(record.get(key)) for key in
                        ('transcript_a', 'confidence_a', 'asr_a_done', 'status',
                         'final_transcript', 'human_override', 'automatic_status')}
            evidence.update(model=models.get('a'), engine=record.get('asr_a_engine') or 'whisper',
                            archived_utc=datetime.now(timezone.utc).isoformat())
            record.setdefault('legacy_asr_results', []).append(evidence)
            changed += 1
        record.update(asr_a_done=False, transcript_a='', confidence_a=None, asr_a_engine=None)
        models.pop('a', None)
        if isinstance(record.get('asr_errors'), dict):
            record['asr_errors'].pop('a', None)
    return changed


# Give longer unresolved clips the extra word differences allowed by the review rule.
def review_word_limit(unresolved_duration: float | None = None) -> int:
    if unresolved_duration is None:
        return BASE_REVIEW_WORD_EDITS
    # Subtract on the 16 kHz source sample grid to avoid float rounding at a boundary.
    samples = round(unresolved_duration * 16000)
    extra_samples = max(0, samples - BASE_CLIP_SECONDS * 16000)
    return BASE_REVIEW_WORD_EDITS + extra_samples // (EXTRA_WORD_SECONDS * 16000)


def comparison_details(first: str, second: str, *, unresolved_duration: float | None = None) -> dict[str, Any]:
    """Compare normalized A (reference) to B with jiwer; scores are agreement.

    They are not accuracy against a human transcript. Empty/punctuation-only
    outputs never become accepted, even when both normalized strings match.
    jiwer 4 defines finite insertion counts for empty references.
    """
    try:
        from jiwer import cer, process_words
    except ImportError as exc:
        raise RuntimeError("Step 6 requires jiwer 4. Run the launcher or install requirements.txt.") from exc
    left, right, matched = comparison_copies(first, second)
    alignment = process_words(left, right)
    word_edits = alignment.substitutions + alignment.insertions + alignment.deletions
    character_error_rate = float(cer(left, right))
    # A's score uses B as reference; B's score uses A as reference. With an
    # empty reference the displayed per-model percentage is undefined.
    cer_a_vs_b = float(cer(right, left)) if right else None
    cer_b_vs_a = character_error_rate if left else None
    word_limit = review_word_limit(unresolved_duration)
    if left and right and (left == right or character_error_rate <= MAX_ACCEPTED_CER):
        status = "accepted"
        reason = "Identical after normalization." if left == right else f"Normalized CER is at most {MAX_ACCEPTED_CER:.0%}."
    elif word_edits <= word_limit:
        status = "needs_review"
        reason = f"Within the review limit of {word_limit} word edits. Listen and choose A/B or correct the text."
    else:
        status = "difficult"
        reason = f"More than {word_limit} word edits."
    if not left or not right:
        reason = "One or both normalized outputs are empty; no automatic acceptance. " + reason
    if matched:
        literal_cer = cer(normalize_arabic(first), normalize_arabic(second))
        reason = (f"Identical after matching listed spellings for {', '.join(matched)}. "
                  f"Before spelling matching, CER (B vs A) was {literal_cer:.2%}.")
    choice = "b" if status == "accepted" else None
    choice_reason = (
        f"{reason} Rule CER (B compared with A): {character_error_rate:.2%}; "
        f"word edits: {word_edits} ({alignment.substitutions} substitutions, "
        f"{alignment.insertions} insertions, {alignment.deletions} deletions). "
        + ("B (Cohere) is the configured tie-break when the agreement rule passes; "
           "this is not a claim that its transcript is more accurate."
           if choice else "No model selected automatically; listen and choose or correct the text.")
    )
    return dict(normalized_transcript_a=left, normalized_transcript_b=right,
                normalization_version=NORMALIZATION_VERSION,
                agreement_rule_version=AGREEMENT_RULE_VERSION,
                comparison_engine="jiwer", comparison_reference="transcript_a",
                automatic_status=status, agreement_reason=reason, review_word_edit_limit=word_limit,
                cer_a_vs_b=cer_a_vs_b, cer_b_vs_a=cer_b_vs_a,
                cer_a_reference="normalized_transcript_b", cer_b_reference="normalized_transcript_a",
                automatic_choice=choice, automatic_transcript=second.strip() if choice else "",
                automatic_choice_reason=choice_reason,
                cer=character_error_rate, wer=float(alignment.wer), word_difference=int(word_edits),
                word_substitutions=int(alignment.substitutions), word_insertions=int(alignment.insertions),
                word_deletions=int(alignment.deletions))


def compare_transcripts(first: str, second: str) -> tuple[str, float, int, float]:
    """Return agreement status, normalized CER, word edits, and normalized WER."""
    result = comparison_details(first, second)
    return result["automatic_status"], result["cer"], result["word_difference"], result["wer"]


# Read a finite numeric field and check its lower bound when one is supplied.
def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number.") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} is outside the valid range.")
    return result


# Check that the source intervals are valid for the length of the recording.
def _validated_spans(raw_spans: Any, duration: float) -> list[dict[str, float]]:
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError("A segment must contain at least one source span.")
    spans = []
    for raw_span in raw_spans:
        if not isinstance(raw_span, dict):
            raise ValueError("Each source span must be an object.")
        start = _number(raw_span.get("start"), "span start", minimum=0)
        end = _number(raw_span.get("end"), "span end", minimum=0)
        if end <= start or end > duration + 0.05:
            raise ValueError("A source span falls outside the WAV duration.")
        spans.append({"start": start, "end": min(end, duration)})
    spans.sort(key=lambda span: (span["start"], span["end"]))
    return spans


# Bring the record's time fields back in sync with its source intervals.
def _refresh_segment_times(segment: dict[str, Any]) -> None:
    spans = segment["source_spans"]
    segment["start"] = round(spans[0]["start"], 3)
    segment["end"] = round(spans[-1]["end"], 3)
    segment["duration"] = round(sum(s["end"] - s["start"] for s in spans), 3)


# Turn a saved clip description into the record the transcription page uses.
def _make_segment(
    raw: dict[str, Any], index: int, duration: float, *, unresolved: bool = False
) -> dict[str, Any]:
    raw_spans = raw.get("source_spans")
    if raw_spans:
        spans = _validated_spans(raw_spans, duration)
    else:
        spans = _validated_spans(
            [{"start": raw.get("start"), "end": raw.get("end")}], duration
        )
    label = str(raw.get("label") or "speech").strip().lower()
    if label not in SUPPORTED_LABELS and not (unresolved and label == "unclassified"):
        raise ValueError("Unknown content label; review the audio before ASR.")
    segment_id = str(
        raw.get("segment_id") or raw.get("id") or raw.get("file")
        or f"segment_{index:04d}"
    )
    changes = raw.get("human_changes")
    raw_status = raw.get("status")
    transcript_status = raw.get("transcript_status")
    source_metadata = raw.get("source_metadata", raw)
    segment = {
        "segment_id": segment_id,
        "label": label,
        "confidence": raw.get("confidence"),
        "confidence_a": raw.get("confidence_a"),
        "confidence_b": raw.get("confidence_b"),
        "snr_db": raw.get("snr_db", raw.get("snr")),
        "clean_noisy": raw.get("clean_noisy", raw.get("quality", "")),
        "transcript_a": str(raw.get("transcript_a") or ""),
        "transcript_b": str(raw.get("transcript_b") or ""),
        "final_transcript": str(
            raw.get("selected_transcript") or raw.get("final_transcript") or ""
        ),
        "selected_transcript": str(
            raw.get("selected_transcript") or raw.get("final_transcript") or ""
        ),
        "transcript_status": str(
            transcript_status
            or (
                raw_status
                if isinstance(raw_status, str) and raw_status in SUPPORTED_STATUSES
                else "needs_review"
            )
        ),
        "status": (
            transcript_status
            if isinstance(transcript_status, str)
            and transcript_status in SUPPORTED_STATUSES
            else raw_status
            if isinstance(raw_status, str) and raw_status in SUPPORTED_STATUSES
            else "needs_review"
        ),
        "cer": raw.get("cer"),
        "wer": raw.get("wer"),
        "word_difference": raw.get("word_difference"),
        "human_override": bool(raw.get("human_override", False)),
        "updated_by": str(raw.get("updated_by") or ""),
        "updated_at": str(raw.get("updated_at") or ""),
        "human_changes": changes if isinstance(changes, list) else [],
        "source_spans": spans,
        "source_metadata": source_metadata if isinstance(source_metadata, dict) else raw,
    }
    _refresh_segment_times(segment)
    for key in COMPARISON_FIELDS:
        if key in raw:
            segment[key] = raw[key]
    return segment


def load_manifest_segments(
    manifest_path: Path, wav_duration: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read Step 5 manifests or the project's regular speech-corpus JSON."""
    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read the segment JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("The segment JSON must contain a JSON object.")

    raw_segments = payload.get("clips")
    if payload.get("format") == "speech_corpus_step5_inputs_v1":
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("The Step 5 manifest does not contain any clips.")
        raw_segments = [
            {
                **clip,
                "segment_id": clip.get("file") or f"segment_{index:04d}",
                "source_metadata": clip,
            }
            for index, clip in enumerate(raw_segments, 1)
            if isinstance(clip, dict)
            and clip.get("status", "approved") == "approved"
            and clip.get("speech_or_mixed", True)
            and str(clip.get("label", "speech")).lower() in {"speech", "mixed"}
        ]
    else:
        raw_segments = payload.get("segments")

    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(
            "The JSON must contain speech/mixed clips or a segments list."
        )
    segments = [
        _make_segment(item, index, wav_duration)
        for index, item in enumerate(raw_segments, 1)
        if isinstance(item, dict)
    ]
    if not segments:
        raise ValueError("The JSON contains no usable speech or mixed segments.")
    if len({item["segment_id"] for item in segments}) != len(segments):
        raise ValueError("The JSON contains duplicate segment IDs.")
    segments.sort(key=lambda item: (item["start"], item["end"], item["segment_id"]))
    return payload, segments


# Read the audio metadata needed to check the segment timestamps.
def get_audio_info(wav_path: Path) -> dict[str, Any]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Audio support is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    try:
        info = sf.info(str(wav_path))
    except Exception as exc:
        raise ValueError(f"Could not open the WAV file: {exc}") from exc
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError("The WAV file contains no readable audio.")
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration": info.frames / info.samplerate,
        "subtype": info.subtype,
    }


def read_segment_audio(
    wav_path: Path, segment: dict[str, Any], target_rate: int | None = None
) -> tuple[Any, int]:
    """Read only the mapped WAV ranges, optionally downmixing/resampling them."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Audio support is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    pieces = []
    sample_rate = 0
    for span in segment["source_spans"]:
        with sf.SoundFile(str(wav_path)) as audio_file:
            sample_rate = int(audio_file.samplerate)
            start_frame = max(0, int(round(span["start"] * sample_rate)))
            end_frame = min(len(audio_file), int(round(span["end"] * sample_rate)))
            if end_frame <= start_frame:
                continue
            audio_file.seek(start_frame)
            piece = audio_file.read(
                end_frame - start_frame, dtype="float32", always_2d=True
            )
        if len(piece):
            pieces.append(piece.mean(axis=1))
    if not pieces:
        raise ValueError(f"No audio frames found for {segment['segment_id']}.")
    audio = np.concatenate(pieces).astype(np.float32, copy=False)
    if target_rate and target_rate != sample_rate:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError(
                "Resampling support is not installed. Run: "
                "python -m pip install -r requirements.txt"
            ) from exc
        divisor = math.gcd(sample_rate, target_rate)
        audio = resample_poly(
            audio, target_rate // divisor, sample_rate // divisor
        ).astype(np.float32, copy=False)
        sample_rate = target_rate
    return audio, sample_rate


def slice_spans(
    spans: list[dict[str, float]], start: float, end: float
) -> list[dict[str, float]]:
    """Intersect source-time spans with an edited interval."""
    result = []
    for span in spans:
        left = max(span["start"], start)
        right = min(span["end"], end)
        if right > left:
            result.append({"start": left, "end": right})
    return result


# Save review progress with the manifest and recording it belongs to.
def save_project(
    path: Path,
    manifest_path: Path,
    wav_path: Path,
    manifest: dict[str, Any],
    segments: list[dict[str, Any]],
    audit_log: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "format": "ameer_week3_review_v1",
        "manifest_path": str(Path(manifest_path).resolve()),
        "wav_path": str(Path(wav_path).resolve()),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "segments": segments,
        "audit_log": audit_log or [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_path, path)
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise OSError(f"Could not save the review project: {exc}") from exc


def update_source_manifest(
    manifest: dict[str, Any], segments: list[dict[str, Any]]
) -> None:
    """Copy review fields into matching source records without replacing their status."""
    collection_name = (
        "clips" if manifest.get("format") == "speech_corpus_step5_inputs_v1"
        else "segments"
    )
    raw_segments = manifest.get(collection_name)
    if not isinstance(raw_segments, list):
        raise ValueError(f"The source JSON has no {collection_name} list.")

    segments_by_id = {segment["segment_id"]: segment for segment in segments}
    for index, raw in enumerate(raw_segments, 1):
        if not isinstance(raw, dict):
            continue
        segment_id = str(
            raw.get("segment_id") or raw.get("id") or raw.get("file")
            or f"segment_{index:04d}"
        )
        segment = segments_by_id.get(segment_id)
        if segment is None:
            continue
        review_fields = {
            "transcript_a": segment.get("transcript_a", ""),
            "transcript_b": segment.get("transcript_b", ""),
            "selected_transcript": (
                segment.get("final_transcript")
                or segment.get("selected_transcript", "")
            ),
            "confidence_a": segment.get("confidence_a"),
            "confidence_b": segment.get("confidence_b"),
            "transcript_status": (
                segment.get("transcript_status")
                or segment.get("status", "needs_review")
            ),
        }
        has_review_data = any(
            review_fields[key]
            for key in ("transcript_a", "transcript_b", "selected_transcript")
        ) or any(
            review_fields[key] is not None
            for key in ("confidence_a", "confidence_b")
        )
        if has_review_data:
            raw.update(review_fields)


# Write the revised source manifest back to disk.
def write_source_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_path, path)
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise OSError(f"Could not update the source JSON: {exc}") from exc


# Copy the current review decisions into the source manifest before saving it.
def save_source_manifest(
    path: Path, manifest: dict[str, Any], segments: list[dict[str, Any]]
) -> None:
    update_source_manifest(manifest, segments)
    write_source_manifest(path, manifest)


# Reload earlier review work when its recording and manifest still match.
def load_saved_project(
    path: Path, manifest_path: Path, wav_path: Path, wav_duration: float
) -> list[dict[str, Any]] | None:
    if not Path(path).is_file():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read saved review progress: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("The saved review file must contain a JSON object.")
    if payload.get("format") != "ameer_week3_review_v1":
        raise ValueError("The saved review file has an unsupported format.")
    saved_manifest = payload.get("manifest_path")
    saved_wav = payload.get("wav_path")
    if not isinstance(saved_manifest, str) or not isinstance(saved_wav, str):
        raise ValueError("The saved review file has invalid source paths.")
    if (Path(saved_manifest).resolve()
            != Path(manifest_path).resolve()
            or Path(saved_wav).resolve()
            != Path(wav_path).resolve()):
        return None
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("The saved review file has no segment list.")
    segments = [
        _make_segment(item, index, wav_duration)
        for index, item in enumerate(raw_segments, 1)
        if isinstance(item, dict)
    ]
    if len({item["segment_id"] for item in segments}) != len(segments):
        raise ValueError("The saved review file contains duplicate segment IDs.")
    return segments


def export_csv(
    path: Path,
    manifest: dict[str, Any],
    segments: list[dict[str, Any]],
    audit_log: list[dict[str, Any]] | None = None,
) -> Path:
    """Export a CSV preserving each original JSON record and review decision."""
    base_fields = [
        "segment_id", "start", "end", "duration", "label", "confidence",
        "confidence_a", "confidence_b", "asr_a_done", "asr_b_done", "snr_db", "snr_threshold_db", "clean_noisy",
        "energy_snr_db", "wada_snr_db", "snr_decision_policy", "snr_threshold_justification", "snr_review_required", "snr_review_status", "snr_note",
        "transcript_a", "transcript_b", "selected_transcript",
        "final_transcript", "status", "transcript_status", "cer", "word_difference",
        "wer",
        "model_a", "model_b", "asr_b_engine", "legacy_asr_results",
        "human_override", "updated_by", "updated_at", "file", "transcript_file",
        "sample_count", "sample_rate_hz", "sha256", "source_spans",
        "human_changes", "source_json", "manifest_json", "audit_log",
    ]
    base_fields += [key for key in COMPARISON_FIELDS if key not in base_fields]
    source_keys = sorted({
        key
        for segment in segments
        for key in (segment.get("source_metadata") or {})
        if isinstance(key, str)
    })
    source_fields = [f"json_{key}" for key in source_keys]
    fields = base_fields + source_fields
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            manifest_metadata = {
                key: value
                for key, value in manifest.items()
                if key not in {"clips", "segments", "source_review"}
            }
            manifest_json = json.dumps(
                manifest_metadata, ensure_ascii=False, separators=(",", ":")
            )
            for segment in segments:
                row = {
                    key: segment.get(key, "")
                    for key in base_fields
                    if key not in {"source_spans", "human_changes", "source_json",
                                   "manifest_json"}
                }
                row["normalized_transcript_a"], row["normalized_transcript_b"], _ = comparison_copies(
                    segment.get("transcript_a", ""), segment.get("transcript_b", ""))
                row["normalization_version"] = NORMALIZATION_VERSION
                row["model_a"] = (segment.get("models") or {}).get("a", "")
                row["model_b"] = (segment.get("models") or {}).get("b", "")
                row["legacy_asr_results"] = json.dumps(segment.get("legacy_asr_results", []), ensure_ascii=False)
                row["source_spans"] = json.dumps(
                    segment.get("source_spans", []), ensure_ascii=False
                )
                row["human_changes"] = json.dumps(
                    segment.get("human_changes", []), ensure_ascii=False
                )
                row["source_json"] = json.dumps(
                    segment.get("source_metadata", {}), ensure_ascii=False
                )
                row["manifest_json"] = manifest_json
                row["audit_log"] = json.dumps(audit_log or [], ensure_ascii=False)
                for key in source_keys:
                    value = (segment.get("source_metadata") or {}).get(key, "")
                    row[f"json_{key}"] = (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list)) else value
                    )
                writer.writerow(row)
    except OSError as exc:
        raise OSError(f"Could not export CSV: {exc}") from exc
    return path

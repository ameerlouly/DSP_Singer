"""Data, audio, transcript-agreement, persistence, and export helpers."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
SUPPORTED_LABELS = {"speech", "music", "mixed", "silence"}
SUPPORTED_STATUSES = {"accepted", "needs_review", "difficult"}


def normalize_arabic(text: str) -> str:
    """Apply the Arabic normalization required before transcript comparison."""
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = ARABIC_DIACRITICS.sub("", normalized).replace("\u0640", "")
    normalized = (normalized.replace("أ", "ا").replace("إ", "ا")
                  .replace("آ", "ا").replace("ٱ", "ا")
                  .replace("ة", "ه").replace("ى", "ي"))
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(normalized.split())


def _edit_distance(left: list[Any], right: list[Any]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for column, right_item in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def compare_transcripts(first: str, second: str) -> tuple[str, float, int, float]:
    """Return agreement status, normalized CER, word edits, and normalized WER."""
    left = normalize_arabic(first)
    right = normalize_arabic(second)
    if not left and not right:
        return "needs_review", 0.0, 0, 0.0
    try:
        from jiwer import cer as jiwer_cer
        from jiwer import process_words
    except ImportError:
        char_error_rate = _edit_distance(list(left), list(right)) / max(1, len(left))
        reference_words = left.split()
        hypothesis_words = right.split()
        word_difference = _edit_distance(reference_words, hypothesis_words)
        word_error_rate = word_difference / max(1, len(reference_words))
    else:
        char_error_rate = float(jiwer_cer(left, right))
        word_alignment = process_words(left, right)
        word_difference = (
            word_alignment.substitutions
            + word_alignment.insertions
            + word_alignment.deletions
        )
        word_error_rate = float(word_alignment.wer)
    if left == right or char_error_rate <= 0.02:
        return "accepted", char_error_rate, word_difference, word_error_rate
    if word_difference <= 3:
        return "needs_review", char_error_rate, word_difference, word_error_rate
    return "difficult", char_error_rate, word_difference, word_error_rate


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number.") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} is outside the valid range.")
    return result


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


def _refresh_segment_times(segment: dict[str, Any]) -> None:
    spans = segment["source_spans"]
    segment["start"] = round(spans[0]["start"], 3)
    segment["end"] = round(spans[-1]["end"], 3)
    segment["duration"] = round(sum(s["end"] - s["start"] for s in spans), 3)


def _make_segment(
    raw: dict[str, Any], index: int, duration: float
) -> dict[str, Any]:
    raw_spans = raw.get("source_spans")
    if raw_spans:
        spans = _validated_spans(raw_spans, duration)
    else:
        spans = _validated_spans(
            [{"start": raw.get("start"), "end": raw.get("end")}], duration
        )
    label = str(raw.get("label") or "speech").strip().lower()
    if label not in SUPPORTED_LABELS:
        label = "speech"
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


def load_whisper_model(model_name: str, progress: Callable[[str], None]) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Whisper is unavailable. Install faster-whisper from requirements.txt."
        ) from exc
    progress(f"Loading Whisper model {model_name}; the first load downloads model data.")
    return WhisperModel(
        model_name, device="cpu", compute_type="int8", cpu_threads=2, num_workers=1
    )


def transcribe_whisper(model: Any, audio: Any) -> tuple[str, float | None]:
    pieces, _ = model.transcribe(
        audio, language="ar", vad_filter=True, beam_size=1, best_of=1,
        temperature=0.0,
    )
    text_parts = []
    log_probabilities = []
    for piece in pieces:
        text = piece.text.strip()
        if text:
            text_parts.append(text)
        log_probability = getattr(piece, "avg_logprob", None)
        if isinstance(log_probability, (int, float)) and math.isfinite(log_probability):
            log_probabilities.append(float(log_probability))
    confidence = None
    if log_probabilities:
        mean_log_probability = sum(log_probabilities) / len(log_probabilities)
        confidence = min(1.0, max(0.0, math.exp(mean_log_probability)))
    return " ".join(text_parts).strip(), confidence


def load_mms_model(model_name: str, progress: Callable[[str], None]) -> Any:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "MMS is unavailable. Install PyTorch and transformers from "
            "requirements-mms.txt."
        ) from exc
    torch.set_num_threads(1)
    progress(f"Loading MMS model {model_name}; the first load downloads model data.")
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        model_kwargs={"target_lang": "ara", "ignore_mismatched_sizes": True},
        device=0 if torch.cuda.is_available() else -1,
    )


def transcribe_mms(
    model: Any, audio: Any, sample_rate: int
) -> tuple[str, float | None]:
    result = model({"raw": audio, "sampling_rate": sample_rate})
    text = result.get("text", "") if isinstance(result, dict) else ""
    if not isinstance(text, str):
        raise RuntimeError("The MMS pipeline returned no transcript text.")
    confidence = None
    acoustic_model = getattr(model, "model", None)
    feature_extractor = getattr(model, "feature_extractor", None)
    tokenizer = getattr(model, "tokenizer", None)
    blank_id = getattr(tokenizer, "pad_token_id", None)
    if (acoustic_model is not None and feature_extractor is not None
            and isinstance(blank_id, int)):
        import torch

        features = feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        device = next(acoustic_model.parameters()).device
        model_inputs = {
            key: value.to(device) for key, value in features.items()
        }
        with torch.inference_mode():
            logits = acoustic_model(**model_inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        frame_confidence, token_ids = probabilities.max(dim=-1)
        speech_frames = token_ids.ne(blank_id)
        if bool(speech_frames.any()):
            confidence = float(frame_confidence[speech_frames].mean().item())
    return text.strip(), confidence


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


def save_source_manifest(
    path: Path, manifest: dict[str, Any], segments: list[dict[str, Any]]
) -> None:
    update_source_manifest(manifest, segments)
    write_source_manifest(path, manifest)


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
        "confidence_a", "confidence_b", "snr_db", "clean_noisy",
        "transcript_a", "transcript_b", "selected_transcript",
        "final_transcript", "status", "transcript_status", "cer", "word_difference",
        "wer",
        "human_override", "updated_by", "updated_at", "source_spans",
        "human_changes", "source_json", "manifest_json", "audit_log",
    ]
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
                if key not in {"clips", "segments"}
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

"""End-to-end DSP corpus pipeline: download, extract, label, segment, SNR and ASR."""
from __future__ import annotations

import csv
import html
import json
import math
import re
import shutil
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data"


def log_progress(callback: Callable[[str], None], message: str):
    callback(message)


def require_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise RuntimeError("FFmpeg is unavailable. Install imageio-ffmpeg.")


def download_video(url: str, output_root: Path, callback: Callable[[str], None] = print) -> tuple[Path, dict]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed") from exc

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = output_root / f"video_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)

    def hook(d):
        status = d.get("status")
        if status == "downloading":
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                callback(f"Downloading video: {downloaded / total:.1%}")
        elif status == "finished":
            callback("Video download finished.")

    ffmpeg_exe = require_ffmpeg()

    ydl_opts = {
        "outtmpl": str(folder / "source.%(ext)s"),
        "progress_hooks": [hook],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": ffmpeg_exe,
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        metadata = {
            "input_url": url,
            "source_url": info.get("webpage_url") or url,
            "video_id": info.get("id"),
            "title": info.get("title", "Untitled"),
            "channel": info.get("channel") or info.get("uploader"),
            "duration_seconds": info.get("duration"),
            "license": info.get("license"),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    candidates = [p for p in folder.iterdir() if p.is_file() and p.name.startswith(
        "source.") and p.suffix.lower() not in (".part", ".wav")]
    if not candidates:
        raise RuntimeError("Downloaded video file was not found")
    return max(candidates, key=lambda p: p.stat().st_size), metadata


def extract_wav(video_path: Path, callback: Callable[[str], None] = print) -> tuple[Path, dict]:
    ffmpeg = require_ffmpeg()
    video_path = Path(video_path)
    wav_path = video_path.parent / "source_16k_mono.wav"
    callback("Extracting 16 kHz mono 16-bit PCM WAV...")
    command = [ffmpeg, "-y", "-v", "error", "-i",
               str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)]
    subprocess.run(command, check=True)
    data, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if rate != 16000 or not len(data):
        raise RuntimeError("WAV validation failed")
    return wav_path, {"sample_rate_hz": rate, "channels": 1, "bits_per_sample": 16,
                      "duration_seconds": len(data) / rate, "audio_file": wav_path.name,
                      "extracted_at_utc": datetime.now(timezone.utc).isoformat()}


def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return 20 * math.log10(max(rms, 1e-12))


def classify_regions(wav_path: Path, callback: Callable[[str], None] = print) -> list[dict]:
    """Label windows with Silero speech activity and an energy/music heuristic."""
    data, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    duration = len(mono) / rate
    if rate != 16000:
        raise RuntimeError("Region labeling requires a 16 kHz WAV")

    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
        torch.set_num_threads(1)
        callback("Loading Silero VAD...")
        model = load_silero_vad()
        callback("Detecting real speech pauses...")
        speech_ranges = get_speech_timestamps(
            torch.from_numpy(mono), model,
            sampling_rate=16000,
            threshold=0.45,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
            speech_pad_ms=120,
            return_seconds=True,
        )
    except Exception as exc:
        callback(f"Silero VAD unavailable; using energy fallback: {exc}")
        speech_ranges = []

    speech_mask = np.zeros(len(mono), dtype=bool)
    for item in speech_ranges:
        a = int(float(item["start"]) * rate)
        b = int(float(item["end"]) * rate)
        speech_mask[max(0, a):min(len(mono), b)] = True

    window = int(rate)
    step = int(rate)
    regions = []
    total = max(1, math.ceil(duration))
    for number, start_sample in enumerate(range(0, len(mono), step), 1):
        end_sample = min(len(mono), start_sample + window)
        chunk = mono[start_sample:end_sample]
        if len(chunk) < int(rate * .1):
            break
        start, end = start_sample / rate, end_sample / rate
        speech_fraction = float(np.mean(speech_mask[start_sample:end_sample]))
        db = _rms_db(chunk)

        frequencies, _, power = spectrogram(
            chunk, fs=rate, nperseg=512, noverlap=384)
        total_power = np.maximum(power.sum(axis=0), 1e-12)
        centroid = float(np.sum(frequencies[:, None] * power, axis=0).sum() / (
            total_power.sum() * max(frequencies[-1], 1)))
        low = float(np.mean(power[frequencies < 300]))
        mid = float(
            np.mean(power[(frequencies >= 300) & (frequencies < 3000)]))
        low_mid_ratio = low / max(mid, 1e-12)

        if db < -48:
            label, confidence = "silence", min(1.0, (-db - 48) / 18)
        elif speech_fraction >= 0.35:
            if low_mid_ratio > 0.65:
                label, confidence = "mixed", min(.95, .50 +
                                                 speech_fraction * .35)
            else:
                label, confidence = "speech", min(.95, .55 +
                                                  speech_fraction * .35)
        elif db > -25 and low_mid_ratio > 0.35:
            label, confidence = "music", min(.95, .55 +
                                             min(0.35, (db + 25) / 100))
        elif speech_fraction < 0.35:
            label, confidence = "silence", min(.75, max(.35, (-db - 35) / 30))
        else:
            label, confidence = "mixed", .55

        regions.append({"start": start, "end": end, "label": label,
                        "confidence": round(float(confidence), 3),
                        "auto_label": label, "auto_confidence": round(float(confidence), 3),
                        "rms_db": round(db, 2), "speech_fraction": round(speech_fraction, 3),
                        "human_override": False, "needs_review": confidence < .60})
        if number % 25 == 0:
            callback(f"Labeling windows: {number}/{total}")

    regions[0]["start"] = 0.0
    regions[-1]["end"] = duration
    merged = []
    for region in regions:
        if merged and region["label"] == merged[-1]["label"]:
            merged[-1]["end"] = region["end"]
            merged[-1]["confidence"] = max(merged[-1]
                                           ["confidence"], region["confidence"])
            merged[-1]["speech_fraction"] = max(merged[-1].get(
                "speech_fraction", 0), region.get("speech_fraction", 0))
        else:
            merged.append(region)
    merged[-1]["end"] = duration
    callback(
        f"Created {len(merged)} preliminary labeled regions using Silero VAD")
    return merged


def find_real_pause_boundaries(wav_path: Path, start: float, end: float) -> list[float]:
    data, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    frame = int(.02 * rate)
    hop = int(.01 * rate)
    start_i, end_i = int(start * rate), int(end * rate)
    chunk = mono[start_i:end_i]
    db = []
    for i in range(0, max(1, len(chunk) - frame), hop):
        db.append(_rms_db(chunk[i:i+frame]))
    if not db:
        return []
    threshold = max(-50.0, float(np.percentile(db, 20)))
    pauses = []
    for i, level in enumerate(db):
        if level < threshold:
            pauses.append(start + i * hop / rate)
    # Collapse pauses within 0.25 s and retain pauses between 5–15 s from boundaries.
    result = []
    for pause in pauses:
        if not result or pause - result[-1] > .25:
            result.append(pause)
    return result


def build_segments(wav_path: Path, regions: list[dict], callback: Callable[[str], None] = print) -> list[dict]:
    """Cut speech/mixed regions into 10–15 s segments at low-energy pauses."""
    data, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    duration = len(data) / rate
    segments = []
    eligible = [r for r in regions if r["label"] in ("speech", "mixed")]

    blocks = []
    max_gap = 10.0
    for region in eligible:
        start = float(region["start"])
        end = min(duration, float(region["end"]))
        if not blocks:
            blocks.append([start, end, region["label"],
                          float(region["confidence"])])
            continue
        previous = blocks[-1]
        if start - previous[1] <= max_gap:
            previous[1] = max(previous[1], end)
            if region["label"] == "mixed":
                previous[2] = "mixed"
            previous[3] = max(previous[3], float(region["confidence"]))
        else:
            blocks.append([start, end, region["label"],
                          float(region["confidence"])])
    for region_number, (start, end, label, confidence) in enumerate(blocks, 1):
        callback(
            f"Finding 10–15 s pauses in block {region_number}/{len(blocks)}")
        if end - start < 10:
            continue

        cursor = start
        while end - cursor >= 10:
            remaining = end - cursor
            if remaining <= 15:
                cut = end
            else:
                search_start = cursor + 10
                search_end = min(cursor + 15, end)
                target = cursor + 12.5
                pauses = find_real_pause_boundaries(
                    wav_path, search_start, search_end)
                candidates = [
                    p for p in pauses if search_start <= p <= search_end]
                cut = min(candidates, key=lambda p: abs(
                    p-target)) if candidates else target
                cut = max(search_start, min(search_end, cut))
            segments.append(_make_segment(
                cursor, cut, label, confidence, len(segments)))
            cursor = cut

    segments.sort(key=lambda s: s["start"])
    segments = [s for s in segments if 10 <= s["duration"] <= 15]
    callback(
        f"Built {len(segments)} speech/mixed segments (10–15 seconds each)")
    return segments


def _make_segment(start: float, end: float, label: str, confidence: float, number: int) -> dict:
    start = max(0.0, start)
    end = max(start + .001, end)
    return {"segment_id": f"seg_{number+1:04d}", "start": round(start, 3), "end": round(end, 3),
            "duration": round(end-start, 3), "label": label, "confidence": round(confidence, 3),
            "snr_db": None, "clean_noisy": "", "transcript_a": "", "transcript_b": "",
            "final_transcript": "", "status": "needs_review", "cer": None,
            "word_difference": None, "human_override": False, "human_changes": []}


ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
PUNCTUATION = re.compile(r"[^\w\s\u0600-\u06ff]", re.UNICODE)


def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = PUNCTUATION.sub(" ", text)
    return " ".join(text.split())


def cer(a: str, b: str) -> float:
    try:
        from jiwer import cer as jiwer_cer
        return float(jiwer_cer(normalize_arabic(a), normalize_arabic(b)))
    except Exception:
        a, b = normalize_arabic(a), normalize_arabic(b)
        if not a:
            return 0.0 if not b else 1.0
        previous = list(range(len(b)+1))
        for i, ca in enumerate(a, 1):
            current = [i]
            for j, cb in enumerate(b, 1):
                current.append(
                    min(current[-1]+1, previous[j]+1, previous[j-1]+(ca != cb)))
            previous = current
        return previous[-1] / max(1, len(a))


def word_difference(a: str, b: str) -> int:
    import difflib
    wa, wb = normalize_arabic(a).split(), normalize_arabic(b).split()
    sm = difflib.SequenceMatcher(None, wa, wb)
    return sum(max(i2-i1, j2-j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


def agreement_status(a: str, b: str) -> tuple[str, float, int]:
    score = cer(a, b)
    words = word_difference(a, b)
    if normalize_arabic(a) == normalize_arabic(b) or score <= .02:
        return "accepted", score, words
    if words <= 3:
        return "needs_review", score, words
    return "difficult", score, words


def estimate_snr(wav_path: Path, start: float, end: float) -> tuple[float | None, str]:
    data, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)[int(start*rate):int(end*rate)]
    frame = int(.02 * rate)
    hop = int(.01 * rate)
    power = np.array([np.mean(mono[i:i+frame].astype(np.float64)**2)
                      for i in range(0, max(1, len(mono)-frame+1), hop)])
    power = power[power > 1e-10]
    if len(power) < 10:
        return None, "Insufficient non-zero audio"
    noise = float(np.percentile(power, 10))
    signal = float(np.percentile(power, 80))
    if noise <= 0 or signal <= noise:
        return None, "No usable energy contrast"
    return round(10 * math.log10(max(signal-noise, 1e-12) / noise), 2), "Energy-frame estimate"


_WHISPER_MODELS = {}
_MMS_MODELS = {}


def transcribe_whisper(wav_path: Path, model_name: str = "base", callback: Callable[[str], None] = lambda _text: None) -> str:
    try:
        from faster_whisper import WhisperModel
        import scipy.signal
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed") from exc

    cache_key = model_name
    if cache_key not in _WHISPER_MODELS:
        callback("Loading Whisper model. First run may take several minutes...")
        _WHISPER_MODELS[cache_key] = WhisperModel(
            model_name, device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
        callback("Whisper model loaded.")
    model = _WHISPER_MODELS[cache_key]

    audio, rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / rate))

    pieces, _ = model.transcribe(
        np.asarray(audio, dtype=np.float32),
        language="ar",
        vad_filter=True,
        beam_size=1,
        best_of=1,
        temperature=0.0,
    )
    return " ".join(p.text.strip() for p in pieces).strip()


def transcribe_mms(wav_path: Path, model_name: str = "facebook/mms-1b-all", callback: Callable[[str], None] = lambda _text: None) -> str:
    try:
        import torch
        import scipy.signal
        from transformers import AutoModelForCTC, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "MMS requires torch, transformers and scipy") from exc

    torch.set_num_threads(1)
    cache_key = model_name
    if cache_key not in _MMS_MODELS:
        callback("Loading Arabic ASR model. First run may take several minutes...")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCTC.from_pretrained(model_name)
        model.eval()
        _MMS_MODELS[cache_key] = (processor, model)
        callback("Arabic ASR model loaded.")
    processor, model = _MMS_MODELS[cache_key]

    audio, rate = sf.read(str(wav_path), dtype="float32")
    if rate != 16000:
        audio = scipy.signal.resample(audio, int(len(audio)*16000/rate))
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0].strip()


def write_segment_wav(source: Path, segment: dict, directory: Path) -> Path:
    data, rate = sf.read(str(source), dtype="float32", always_2d=True)
    i1 = int(float(segment["start"])*rate)
    i2 = min(len(data), int(float(segment["end"])*rate))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{segment['segment_id']}.wav"
    sf.write(str(path), data[i1:i2], rate, subtype="PCM_16")
    return path


def export_csv(path: Path, wav_path: Path, regions: list[dict], segments: list[dict]) -> Path:
    fields = ["segment_id", "start", "end", "duration", "label", "confidence", "snr_db",
              "clean_noisy", "transcript_a", "transcript_b", "final_transcript", "status", "cer",
              "word_difference", "human_override", "updated_by", "updated_at"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(segments)

    payload = {"version": 1, "source_wav": str(wav_path), "exported_utc": datetime.now(timezone.utc).isoformat(),
               "regions": regions, "segments": segments}
    Path(path).with_suffix(".json").write_text(json.dumps(
        payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return Path(path)

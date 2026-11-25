"""Check the path from Arabic transcript comparison to a saved review.

The manifest and CSV tests verify that each clip keeps its own source intervals, raw
model outputs and final text when the project is reopened.
"""

import sys, csv, json, math, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import transcript_agreement_and_io as pipeline
class ArabicAgreementTests(unittest.TestCase):
    # Compare Arabic spelling variants after normalizing marks, letters and punctuation.
    def test_normalization_applies_arabic_rules_and_removes_punctuation(self):
        self.assertEqual(
            pipeline.normalize_arabic("إِنَّ السَّلَامَـ، عَلَيْكُمْ!"),
            "ان السلام عليكم",
        )
        self.assertEqual(
            pipeline.normalize_arabic("مدرسة هدى"),
            "مدرسه هدي",
        )

    # Use known word and character differences to check the resulting review status.
    def test_agreement_status_uses_normalized_cer_and_word_distance(self):
        self.assertEqual(
            pipeline.compare_transcripts("أهلاً، بكم!", "اهلا بكم")[0],
            "accepted",
        )
        self.assertEqual(
            pipeline.compare_transcripts("السلام عليكم", "السلام عليك")[0],
            "needs_review",
        )
        status, _cer, word_difference, wer = pipeline.compare_transcripts(
            "احمد ذهب الى السوق",
            "الولد يكتب كتاب جديد جدا",
        )
        self.assertEqual(status, "difficult")
        self.assertGreater(word_difference, 3)
        self.assertGreater(wer, 0)


class ManifestTests(unittest.TestCase):
    # Read a clip made from multiple intervals and check that every mapping survives.
    def test_step5_manifest_preserves_multiple_source_spans(self):
        payload = {
            "format": "speech_corpus_step5_inputs_v1",
            "source_audio": {"duration_seconds": 20},
            "clips": [
                {
                    "order": 1,
                    "file": "speech_001.wav",
                    "status": "approved",
                    "label": "speech",
                    "speech_or_mixed": True,
                    "source_spans": [
                        {"start": 2, "end": 3, "clip_start": 0, "clip_end": 1},
                        {"start": 7, "end": 8, "clip_start": 1, "clip_end": 2},
                    ],
                },
                {
                    "order": 2,
                    "file": "music_002.wav",
                    "status": "approved",
                    "label": "music",
                    "speech_or_mixed": False,
                    "source_spans": [{"start": 9, "end": 10}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded, segments = pipeline.load_manifest_segments(path, 20)
        self.assertEqual(loaded["format"], "speech_corpus_step5_inputs_v1")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["segment_id"], "speech_001.wav")
        self.assertEqual(segments[0]["duration"], 2)
        self.assertEqual(len(segments[0]["source_spans"]), 2)
        self.assertEqual(segments[0]["source_metadata"]["file"], "speech_001.wav")

    # An interval beyond the end of the WAV should make the manifest fail validation.
    def test_manifest_rejects_spans_outside_the_source_wav(self):
        payload = {
            "segments": [{"segment_id": "bad", "start": 1, "end": 11}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the WAV duration"):
                pipeline.load_manifest_segments(path, 10)


class PersistenceTests(unittest.TestCase):
    # Read the CSV back and check both the original data and the later review fields.
    def test_review_progress_and_original_json_are_saved_in_csv(self):
        manifest_path = Path("manifest.json")
        wav_path = Path("source.wav")
        manifest = {
            "format": "example",
            "description": "test corpus",
            "clips": [{"file": "speech_001.wav"}],
        }
        segment = {
            "segment_id": "speech_001.wav",
            "start": 1.0,
            "end": 2.0,
            "duration": 1.0,
            "label": "speech",
            "confidence": 0.9,
            "snr_db": 12,
            "clean_noisy": "noisy",
            "transcript_a": "مرحبا",
            "transcript_b": "مرحبا",
            "final_transcript": "مرحبا",
            "status": "accepted",
            "cer": 0.0,
            "word_difference": 0,
            "human_override": True,
            "updated_by": "annotator",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "source_spans": [{"start": 1.0, "end": 2.0}],
            "human_changes": [{"field": "final_transcript"}],
            "source_metadata": {"file": "speech_001.wav", "reviewer": "Robir"},
        }
        audit_log = [{"action": "delete", "removed_segment_ids": ["other"]}]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "corpus.csv"
            pipeline.export_csv(csv_path, manifest, [segment], audit_log=audit_log)
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["json_file"], "speech_001.wav")
        self.assertEqual(json.loads(row["source_json"])["reviewer"], "Robir")
        self.assertEqual(
            json.loads(row["manifest_json"]),
            {"format": "example", "description": "test corpus"},
        )
        self.assertEqual(json.loads(row["audit_log"]), audit_log)
        self.assertEqual(row["final_transcript"], "مرحبا")

    # Reopen a saved review using the same audio and manifest it was created with.
    def test_saved_review_can_be_reopened_for_the_same_input_pair(self):
        manifest = {
            "format": "example",
            "segments": [{"segment_id": "seg-1", "start": 2, "end": 4}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            wav_path = root / "source.wav"
            review_path = root / "manifest_review.json"
            segments = [{
                "segment_id": "seg-1",
                "start": 2,
                "end": 4,
                "duration": 2,
                "label": "speech",
                "source_spans": [{"start": 2, "end": 4}],
                "transcript_a": "مرحبا",
                "transcript_b": "مرحبا",
                "final_transcript": "مرحبا",
                "status": "accepted",
                "human_changes": [],
                "source_metadata": {"segment_id": "seg-1"},
            }]
            pipeline.save_project(
                review_path, manifest_path, wav_path, manifest, segments,
                audit_log=[{"action": "save"}],
            )
            restored = pipeline.load_saved_project(
                review_path, manifest_path, wav_path, 10
            )
            saved = json.loads(review_path.read_text(encoding="utf-8"))
        self.assertEqual(restored[0]["final_transcript"], "مرحبا")
        self.assertEqual(saved["audit_log"], [{"action": "save"}])

    # Give each clip different words so a mix-up between records shows up on reload.
    def test_multiple_segments_keep_individual_saved_transcripts(self):
        manifest = {"format": "example", "segments": []}
        segments = []
        for index, transcript in enumerate(("النص الاول", "النص الثاني"), 1):
            segments.append({
                "segment_id": f"seg-{index}",
                "start": float(index),
                "end": float(index + 1),
                "duration": 1,
                "label": "speech",
                "source_spans": [{"start": float(index), "end": float(index + 1)}],
                "transcript_a": transcript,
                "transcript_b": transcript,
                "final_transcript": transcript,
                "status": "accepted",
                "human_changes": [],
                "source_metadata": {"segment_id": f"seg-{index}"},
            })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            wav_path = root / "source.wav"
            review_path = root / "manifest_review.json"
            pipeline.save_project(
                review_path, manifest_path, wav_path, manifest, segments
            )
            restored = pipeline.load_saved_project(
                review_path, manifest_path, wav_path, 10
            )
        self.assertEqual(
            [item["transcript_a"] for item in restored],
            ["النص الاول", "النص الثاني"],
        )
        self.assertEqual(
            [item["transcript_b"] for item in restored],
            ["النص الاول", "النص الثاني"],
        )

    # Writing transcript decisions back should preserve the clip's segmentation status.
    def test_source_manifest_receives_review_fields_without_changing_clip_status(self):
        manifest = {
            "format": "speech_corpus_step5_inputs_v1",
            "clips": [
                {"file": "speech_001.wav", "status": "approved"},
                {"file": "speech_002.wav", "status": "approved"},
            ],
        }
        segments = [{
            "segment_id": "speech_001.wav",
            "transcript_a": "مرحبا",
            "transcript_b": "مرحبا",
            "final_transcript": "مرحبا",
            "confidence_a": 0.92,
            "confidence_b": 0.86,
            "status": "accepted",
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            pipeline.save_source_manifest(path, manifest, segments)
            saved = json.loads(path.read_text(encoding="utf-8"))
        first, second = saved["clips"]
        self.assertEqual(first["status"], "approved")
        self.assertEqual(first["transcript_a"], "مرحبا")
        self.assertEqual(first["transcript_b"], "مرحبا")
        self.assertEqual(first["selected_transcript"], "مرحبا")
        self.assertEqual(first["confidence_a"], 0.92)
        self.assertEqual(first["confidence_b"], 0.86)
        self.assertEqual(first["transcript_status"], "accepted")
        self.assertEqual(second, {"file": "speech_002.wav", "status": "approved"})



if __name__=="__main__": unittest.main()

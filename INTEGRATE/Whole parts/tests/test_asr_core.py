import sys, csv, json, math, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import asr_pipeline as pipeline
class ArabicAgreementTests(unittest.TestCase):
    def test_normalization_applies_arabic_rules_and_removes_punctuation(self):
        self.assertEqual(
            pipeline.normalize_arabic("إِنَّ السَّلَامَـ، عَلَيْكُمْ!"),
            "ان السلام عليكم",
        )
        self.assertEqual(
            pipeline.normalize_arabic("مدرسة هدى"),
            "مدرسه هدي",
        )

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


class TranscriptionConfidenceTests(unittest.TestCase):
    def test_whisper_confidence_is_normalized_from_average_log_probability(self):
        pieces = [
            SimpleNamespace(text="مرحبا", avg_logprob=math.log(0.81)),
            SimpleNamespace(text=" بكم", avg_logprob=math.log(0.49)),
        ]

        class FakeWhisperModel:
            def transcribe(self, _audio, **_kwargs):
                if _kwargs.get("language") != "ar" or _kwargs.get("task") != "transcribe":
                    raise AssertionError("Step 6 requires Arabic transcription.")
                return iter(pieces), None

        text, confidence = pipeline.transcribe_whisper(FakeWhisperModel(), None)
        self.assertEqual(text, "مرحبا بكم")
        self.assertAlmostEqual(confidence, math.sqrt(0.81 * 0.49))

    def test_mms_confidence_uses_nonblank_frame_probabilities(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional for the core application")

        class FakeAcousticModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_values):
                logits = torch.tensor(
                    [[[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]],
                    device=input_values.device,
                )
                return SimpleNamespace(logits=logits)

        class FakeFeatureExtractor:
            def __call__(self, _audio, *, sampling_rate, return_tensors):
                if sampling_rate != 16000 or return_tensors != "pt":
                    raise AssertionError("Unexpected feature extractor arguments")
                return {"input_values": torch.zeros((1, 3))}

        class FakeMmsPipeline:
            model = FakeAcousticModel()
            feature_extractor = FakeFeatureExtractor()
            tokenizer = SimpleNamespace(pad_token_id=0)

            def __call__(self, _audio):
                return {"text": "مرحبا"}

        text, confidence = pipeline.transcribe_mms(
            FakeMmsPipeline(), None, 16000
        )
        self.assertEqual(text, "مرحبا")
        expected = torch.softmax(torch.tensor([2.0, 0.0, 0.0]), dim=0)[0].item()
        self.assertAlmostEqual(confidence, expected)


class ManifestTests(unittest.TestCase):
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

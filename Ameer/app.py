"""Desktop transcript review and corpus export application."""
from __future__ import annotations

import getpass
import json
import queue
import threading
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import sounddevice as sd

import pipeline
from rtl_editor import edit_arabic_text


class CorpusReviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Ameer — Week 3 Speech Corpus Review")
        self.root.geometry("1480x920")
        self.root.minsize(1120, 720)

        self.worker_queue: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self.segments: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}
        self.manifest_path: Path | None = None
        self.wav_path: Path | None = None
        self.project_path: Path | None = None
        self.audio_info: dict[str, Any] | None = None
        self.audit_log: list[dict[str, Any]] = []
        self.current_id: str | None = None
        self.busy = False
        self.closing = False
        self.transcription_errors: list[str] = []
        self.transcription_cancel_event: threading.Event | None = None
        self.required_transcripts: dict[str, set[str]] = {}
        self.completed_transcripts: dict[str, set[str]] = {}

        project_root = Path(__file__).resolve().parents[2]
        default_manifest = project_root / "step5_manifest.json"
        default_wav = project_root / "Chat GPT الدحيح.wav"
        self.manifest_var = tk.StringVar(
            value=str(default_manifest) if default_manifest.is_file() else ""
        )
        self.wav_var = tk.StringVar(
            value=str(default_wav) if default_wav.is_file() else ""
        )
        self.whisper_model_var = tk.StringVar(value="large-v3")
        self.mms_model_var = tk.StringVar(value="facebook/mms-1b-all")
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.label_var = tk.StringVar()
        self.quality_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.progress_message = tk.StringVar(
            value="Choose a segment JSON and source WAV.")
        self.progress_var = tk.DoubleVar(value=0)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll_worker)

    def _build_ui(self) -> None:
        source = ttk.LabelFrame(self.root, text="Project files", padding=8)
        source.pack(fill="x", padx=10, pady=(10, 5))

        ttk.Label(source, text="Segment JSON").grid(
            row=0, column=0, sticky="w")
        self.manifest_entry = ttk.Entry(source, textvariable=self.manifest_var)
        self.manifest_entry.grid(row=0, column=1, sticky="ew", padx=6)
        self.manifest_button = ttk.Button(
            source, text="Browse…", command=self._browse_manifest
        )
        self.manifest_button.grid(row=0, column=2, padx=(0, 10))
        ttk.Label(source, text="Source WAV").grid(row=1, column=0, sticky="w")
        self.wav_entry = ttk.Entry(source, textvariable=self.wav_var)
        self.wav_entry.grid(row=1, column=1, sticky="ew", padx=6)
        self.wav_button = ttk.Button(
            source, text="Browse…", command=self._browse_wav)
        self.wav_button.grid(row=1, column=2, padx=(0, 10))
        self.load_button = ttk.Button(
            source, text="Load / Resume", command=self._load_project
        )
        self.load_button.grid(row=0, column=3, rowspan=2, sticky="ns")
        source.columnconfigure(1, weight=1)

        model_bar = ttk.LabelFrame(
            self.root, text="Arabic transcription models", padding=8)
        model_bar.pack(fill="x", padx=10, pady=5)
        ttk.Label(model_bar, text="Whisper").grid(row=0, column=0, sticky="w")
        self.whisper_entry = ttk.Entry(
            model_bar, textvariable=self.whisper_model_var, width=20
        )
        self.whisper_entry.grid(row=0, column=1, padx=(5, 14), sticky="w")
        ttk.Label(model_bar, text="MMS").grid(row=0, column=2, sticky="w")
        self.mms_entry = ttk.Entry(
            model_bar, textvariable=self.mms_model_var, width=29)
        self.mms_entry.grid(row=0, column=3, padx=5, sticky="w")
        self.transcribe_selected_button = ttk.Button(
            model_bar, text="Transcribe selected", command=self._transcribe_selected
        )
        self.transcribe_selected_button.grid(row=0, column=4, padx=(12, 4))
        self.transcribe_pending_button = ttk.Button(
            model_bar, text="Transcribe all pending", command=self._transcribe_pending
        )
        self.transcribe_pending_button.grid(row=0, column=5, padx=4)
        self.stop_transcription_button = ttk.Button(
            model_bar,
            text="Stop transcription",
            command=self._stop_transcription,
            state="disabled",
        )
        self.stop_transcription_button.grid(row=0, column=6, padx=(4, 0))
        model_bar.columnconfigure(7, weight=1)

        self.panes = ttk.Panedwindow(self.root, orient="horizontal")
        self.panes.pack(fill="both", expand=True, padx=10, pady=5)

        list_panel = ttk.Frame(self.panes, padding=(0, 0, 5, 0))
        detail_panel = ttk.Frame(self.panes, padding=(5, 0, 0, 0))
        self.panes.add(list_panel, weight=3)
        self.panes.add(detail_panel, weight=4)

        ttk.Label(
            list_panel, text="Speech and mixed segments", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 5))
        columns = ("id", "start", "end", "duration", "label", "snr", "status")
        self.table = ttk.Treeview(
            list_panel, columns=columns, show="headings", selectmode="extended"
        )
        headings = (
            ("id", "Segment", 130),
            ("start", "Start", 80),
            ("end", "End", 80),
            ("duration", "Duration", 82),
            ("label", "Label", 80),
            ("snr", "SNR dB", 72),
            ("status", "Status", 112),
        )
        for key, title, width in headings:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=55,
                              stretch=key in {"id", "status"})
        scrollbar = ttk.Scrollbar(
            list_panel, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.table.bind("<<TreeviewSelect>>", self._select_segment)

        ttk.Label(
            detail_panel, text="Review selected segment", font=("Segoe UI", 11, "bold")
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 5))

        ttk.Label(detail_panel, text="Start (seconds)").grid(
            row=1, column=0, sticky="w")
        self.start_entry = ttk.Entry(
            detail_panel, textvariable=self.start_var, width=12)
        self.start_entry.grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(detail_panel, text="End (seconds)").grid(
            row=1, column=2, sticky="w")
        self.end_entry = ttk.Entry(
            detail_panel, textvariable=self.end_var, width=12)
        self.end_entry.grid(row=1, column=3, sticky="w", padx=4)
        self.move_button = ttk.Button(
            detail_panel, text="Apply cut / move", command=self._move_segment
        )
        self.move_button.grid(row=1, column=4, padx=4)
        self.play_button = ttk.Button(
            detail_panel, text="Play segment", command=self._play_segment
        )
        self.play_button.grid(row=1, column=5, padx=4)
        self.play_range_button = ttk.Button(
            detail_panel, text="Play edited range", command=self._play_edited_range
        )
        self.play_range_button.grid(row=2, column=5, padx=4, pady=(4, 8))
        self.stop_button = ttk.Button(
            detail_panel, text="Stop audio", command=self._stop_audio
        )
        self.stop_button.grid(row=2, column=4, padx=4, pady=(4, 8))
        ttk.Label(detail_panel, text="Label").grid(row=2, column=0, sticky="w")
        self.label_box = ttk.Combobox(
            detail_panel,
            textvariable=self.label_var,
            values=("speech", "mixed", "music", "silence"),
            state="readonly",
            width=12,
        )
        self.label_box.grid(row=2, column=1, sticky="w", padx=4, pady=(4, 8))
        ttk.Label(detail_panel, text="Status").grid(
            row=2, column=2, sticky="w")
        self.status_box = ttk.Combobox(
            detail_panel,
            textvariable=self.status_var,
            values=("accepted", "needs_review", "difficult"),
            state="readonly",
            width=16,
        )
        self.status_box.grid(row=2, column=3, sticky="w", padx=4, pady=(4, 8))

        ttk.Label(detail_panel, text="SNR quality").grid(
            row=3, column=0, sticky="w")
        self.quality_box = ttk.Combobox(
            detail_panel,
            textvariable=self.quality_var,
            values=("", "clean", "noisy"),
            state="readonly",
            width=12,
        )
        self.quality_box.grid(row=3, column=1, sticky="w", padx=4, pady=(0, 5))

        ttk.Label(detail_panel, text="Whisper transcript (A)").grid(
            row=4, column=0, columnspan=2, sticky="w"
        )
        self.edit_a_button = ttk.Button(
            detail_panel,
            text="Edit RTL…",
            command=lambda: self._edit_rtl_field(
                self.transcript_a, "Whisper transcript (A)"
            ),
        )
        self.edit_a_button.grid(row=4, column=2, sticky="e")
        ttk.Label(detail_panel, text="MMS transcript (B)").grid(
            row=4, column=3, columnspan=2, sticky="w"
        )
        self.edit_b_button = ttk.Button(
            detail_panel,
            text="Edit RTL…",
            command=lambda: self._edit_rtl_field(
                self.transcript_b, "MMS transcript (B)"
            ),
        )
        self.edit_b_button.grid(row=4, column=5, sticky="e")
        self.transcript_a = tk.Text(
            detail_panel, height=5, wrap="word", undo=True)
        self.transcript_a.grid(
            row=5, column=0, columnspan=3, sticky="nsew", padx=(0, 4))
        self.transcript_b = tk.Text(
            detail_panel, height=5, wrap="word", undo=True)
        self.transcript_b.grid(
            row=5, column=3, columnspan=3, sticky="nsew", padx=(4, 0))

        ttk.Label(detail_panel, text="Final transcript").grid(
            row=6, column=0, columnspan=5, sticky="w", pady=(7, 0)
        )
        self.edit_final_button = ttk.Button(
            detail_panel,
            text="Edit RTL…",
            command=lambda: self._edit_rtl_field(
                self.final_transcript, "Final transcript"
            ),
        )
        self.edit_final_button.grid(row=6, column=5, sticky="e", pady=(7, 0))
        self.final_transcript = tk.Text(
            detail_panel, height=7, wrap="word", undo=True)
        self.final_transcript.grid(
            row=7, column=0, columnspan=6, sticky="nsew", pady=(3, 5)
        )
        for widget in (
            self.transcript_a,
            self.transcript_b,
            self.final_transcript,
        ):
            self._configure_rtl_text(widget)
            for sequence in (
                "<KeyRelease>",
                "<<Paste>>",
                "<<Cut>>",
                "<<Undo>>",
                "<<Redo>>",
            ):
                widget.bind(
                    sequence,
                    lambda _event, target=widget: self.root.after_idle(
                        self._apply_rtl_tag, target
                    ),
                    add="+",
                )

        action_bar = ttk.Frame(detail_panel)
        action_bar.grid(row=8, column=0, columnspan=6,
                        sticky="ew", pady=(3, 0))
        self.use_a_button = ttk.Button(
            action_bar, text="Accept A", command=lambda: self._choose_transcript("A")
        )
        self.use_a_button.pack(side="left", padx=(0, 4))
        self.use_b_button = ttk.Button(
            action_bar, text="Accept B", command=lambda: self._choose_transcript("B")
        )
        self.use_b_button.pack(side="left", padx=4)
        self.save_button = ttk.Button(
            action_bar, text="Save review", command=self._save_detail
        )
        self.save_button.pack(side="left", padx=4)
        self.split_button = ttk.Button(
            action_bar, text="Split at…", command=self._split_segment
        )
        self.split_button.pack(side="left", padx=4)
        self.merge_button = ttk.Button(
            action_bar, text="Merge selected", command=self._merge_segments
        )
        self.merge_button.pack(side="left", padx=4)
        self.add_button = ttk.Button(
            action_bar, text="Add segment…", command=self._add_segment
        )
        self.add_button.pack(side="left", padx=4)
        self.delete_button = ttk.Button(
            action_bar, text="Delete", command=self._delete_segments
        )
        self.delete_button.pack(side="right")

        for column in range(6):
            detail_panel.columnconfigure(column, weight=1)
        detail_panel.rowconfigure(5, weight=1)
        detail_panel.rowconfigure(7, weight=1)

        bottom = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        bottom.pack(fill="x")
        self.export_button = ttk.Button(
            bottom, text="Export corpus CSV…", command=self._export_csv
        )
        self.export_button.pack(side="right", padx=(6, 0))
        ttk.Button(bottom, text="Save project", command=self._save_project).pack(
            side="right"
        )
        self.progress_bar = ttk.Progressbar(
            bottom, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.pack(side="left", fill="x",
                               expand=True, padx=(0, 10))
        ttk.Label(bottom, textvariable=self.progress_message,
                  width=58).pack(side="left")

        self._set_editor_enabled(False)

    def _browse_manifest(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose the segment JSON",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if path:
            self.manifest_var.set(path)

    def _browse_wav(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose the complete source WAV",
            filetypes=(("WAV audio", "*.wav"), ("All files", "*.*")),
        )
        if path:
            self.wav_var.set(path)

    def _load_project(self) -> None:
        if self.busy:
            return
        manifest_path = Path(self.manifest_var.get().strip())
        wav_path = Path(self.wav_var.get().strip())
        if not manifest_path.is_file() or not wav_path.is_file():
            messagebox.showerror(
                "Project files required",
                "Choose an existing segment JSON and complete source WAV.",
                parent=self.root,
            )
            return
        try:
            audio_info = pipeline.get_audio_info(wav_path)
            manifest, segments = pipeline.load_manifest_segments(
                manifest_path, audio_info["duration"]
            )
            project_path = manifest_path.with_name(
                manifest_path.stem + "_review.json"
            )
            saved_segments = pipeline.load_saved_project(
                project_path, manifest_path, wav_path, audio_info["duration"]
            )
            if saved_segments is not None:
                segments = saved_segments
                saved_data = json.loads(
                    project_path.read_text(encoding="utf-8-sig"))
                self.audit_log = saved_data.get("audit_log", [])
                if not isinstance(self.audit_log, list):
                    self.audit_log = []
                detail = f"Resumed {len(segments)} segments with saved review progress."
            else:
                self.audit_log = []
                detail = f"Loaded {len(segments)} segments."
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Could not load project",
                                 str(exc), parent=self.root)
            return

        declared_audio = manifest.get("source_audio")
        if isinstance(declared_audio, dict):
            declared_duration = declared_audio.get("duration_seconds")
            try:
                differs = abs(float(declared_duration) -
                              audio_info["duration"]) > 1
            except (TypeError, ValueError):
                differs = False
            if differs:
                messagebox.showwarning(
                    "Audio duration differs",
                    "The selected WAV duration differs from the JSON source metadata. "
                    "Please confirm that these files belong together.",
                    parent=self.root,
                )

        self.manifest_path = manifest_path.resolve()
        self.wav_path = wav_path.resolve()
        self.project_path = project_path
        self.manifest = manifest
        self.audio_info = audio_info
        self.segments = segments
        self.current_id = None
        self._refresh_table()
        self._set_editor_enabled(True)
        self.progress_var.set(0)
        self.progress_message.set(
            f"{detail} WAV: {audio_info['duration'] / 60:.1f} min, "
            f"{audio_info['sample_rate']} Hz, {audio_info['channels']} channels."
        )
        self._save_project(silent=True)

    def _set_editor_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (
            self.start_entry,
            self.end_entry,
            self.move_button,
            self.play_button,
            self.play_range_button,
            self.stop_button,
            self.label_box,
            self.quality_box,
            self.status_box,
            self.edit_a_button,
            self.edit_b_button,
            self.edit_final_button,
            self.use_a_button,
            self.use_b_button,
            self.save_button,
            self.split_button,
            self.merge_button,
            self.add_button,
            self.delete_button,
            self.transcribe_selected_button,
            self.transcribe_pending_button,
            self.export_button,
        ):
            widget.configure(state=state)
        for widget in (
            self.transcript_a, self.transcript_b, self.final_transcript
        ):
            widget.configure(state="disabled")

    def _find_segment(self, segment_id: str | None) -> dict[str, Any] | None:
        if segment_id is None:
            return None
        return next(
            (segment for segment in self.segments
             if segment["segment_id"] == segment_id),
            None,
        )

    def _selected_segments(self) -> list[dict[str, Any]]:
        ids = self.table.selection()
        return [
            segment for segment in self.segments
            if segment["segment_id"] in ids
        ]

    def _refresh_table(self, select_id: str | None = None) -> None:
        self.table.delete(*self.table.get_children())
        for segment in self.segments:
            self.table.insert(
                "", "end", iid=segment["segment_id"], values=("",) * 7)
            self._update_table_row(segment)
        if select_id and self.table.exists(select_id):
            self.table.selection_set(select_id)
            self.table.focus(select_id)
            self.table.see(select_id)

    def _update_table_row(self, segment: dict[str, Any]) -> None:
        if not self.table.exists(segment["segment_id"]):
            return
        snr = segment.get("snr_db")
        self.table.item(
            segment["segment_id"],
            values=(
                segment["segment_id"],
                f"{segment['start']:.3f}",
                f"{segment['end']:.3f}",
                f"{segment['duration']:.3f}",
                segment["label"],
                "" if snr is None else snr,
                segment["status"],
            ),
        )

    def _fill_editor(self, segment: dict[str, Any]) -> None:
        self.current_id = segment["segment_id"]
        self.start_var.set(f"{segment['start']:.3f}")
        self.end_var.set(f"{segment['end']:.3f}")
        self.label_var.set(segment["label"])
        self.quality_var.set(segment.get("clean_noisy") or "")
        self.status_var.set(segment["status"])
        self._set_text(self.transcript_a, segment.get("transcript_a", ""))
        self._set_text(self.transcript_b, segment.get("transcript_b", ""))
        self._set_text(self.final_transcript,
                       segment.get("final_transcript", ""))

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        previous_state = widget.cget("state")
        if previous_state == "disabled":
            widget.configure(state="normal")
        try:
            widget.delete("1.0", "end")
            widget.insert("1.0", value, "arabic_rtl")
            CorpusReviewApp._apply_rtl_tag(widget)
        finally:
            if previous_state == "disabled":
                widget.configure(state="disabled")

    def _edit_rtl_field(self, widget: tk.Text, title: str) -> None:
        try:
            text = edit_arabic_text(title, self._get_text(widget))
        except RuntimeError as exc:
            messagebox.showerror("RTL editor failed", str(exc), parent=self.root)
            return
        if text is not None:
            self._set_text(widget, text)

    @staticmethod
    def _configure_rtl_text(widget: tk.Text) -> None:
        widget.tag_configure("arabic_rtl", justify="right")
        CorpusReviewApp._apply_rtl_tag(widget)

    @staticmethod
    def _apply_rtl_tag(widget: tk.Text) -> None:
        widget.tag_add("arabic_rtl", "1.0", "end")

    @staticmethod
    def _get_text(widget: tk.Text) -> str:
        return widget.get("1.0", "end-1c").strip()

    def _select_segment(self, _event: tk.Event) -> None:
        selected = self._selected_segments()
        if not selected:
            return
        if len(selected) > 1:
            return
        if self.current_id and self.current_id != selected[0]["segment_id"]:
            if not self._commit_detail():
                self.table.selection_set(self.current_id)
                return
        self._fill_editor(selected[0])

    def _record_change(
        self, segment: dict[str, Any], field: str, old: Any, new: Any
    ) -> None:
        if old == new:
            return
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "actor": getpass.getuser(),
            "segment_id": segment["segment_id"],
            "field": field,
            "old_value": old,
            "new_value": new,
        }
        segment.setdefault("human_changes", []).append(event)
        self.audit_log.append(event)

    def _commit_detail(self, show_error: bool = True) -> bool:
        segment = self._find_segment(self.current_id)
        if not segment:
            return True
        try:
            start = float(self.start_var.get().strip())
            end = float(self.end_var.get().strip())
            if start < 0 or end <= start or end > self.audio_info["duration"]:
                raise ValueError(
                    "Start/end must be valid times within the source WAV.")
        except (ValueError, TypeError) as exc:
            if show_error:
                messagebox.showerror(
                    "Invalid segment", str(exc), parent=self.root)
            return False

        label = self.label_var.get().strip().lower()
        status = self.status_var.get().strip()
        if label not in pipeline.SUPPORTED_LABELS or status not in pipeline.SUPPORTED_STATUSES:
            if show_error:
                messagebox.showerror(
                    "Invalid review values", "Choose a valid segment label and status.",
                    parent=self.root,
                )
            return False
        values = {
            "label": label,
            "clean_noisy": self.quality_var.get().strip().lower(),
            "transcript_a": self._get_text(self.transcript_a),
            "transcript_b": self._get_text(self.transcript_b),
            "final_transcript": self._get_text(self.final_transcript),
            "status": status,
        }
        old_times = (segment["start"], segment["end"])
        new_times = (round(start, 3), round(end, 3))
        changed = old_times != new_times
        if changed:
            self._record_change(segment, "time_range", old_times, new_times)
            segment["source_spans"] = [{"start": start, "end": end}]
            segment["start"] = round(start, 3)
            segment["end"] = round(end, 3)
            segment["duration"] = round(end - start, 3)

        old_status = segment["status"]
        status_changed_by_user = status != old_status
        was_human_override = segment["human_override"]
        transcript_changed = False
        final_changed = values["final_transcript"] != segment["final_transcript"]
        for key, value in values.items():
            old = segment.get(key, "")
            if old != value:
                changed = True
                self._record_change(segment, key, old, value)
                transcript_changed |= key in {"transcript_a", "transcript_b"}
                segment[key] = value

        if final_changed and values["final_transcript"] and not status_changed_by_user:
            self._record_change(segment, "status",
                                segment["status"], "accepted")
            segment["status"] = "accepted"
            self.status_var.set("accepted")
        elif transcript_changed:
            first = segment["transcript_a"]
            second = segment["transcript_b"]
            if first and second:
                new_status, score, word_difference, word_error_rate = pipeline.compare_transcripts(
                    first, second
                )
                segment["cer"] = round(score, 4)
                segment["word_difference"] = word_difference
                segment["wer"] = round(word_error_rate, 4)
                if not status_changed_by_user:
                    segment["status"] = new_status
                    self.status_var.set(new_status)
                    if not was_human_override:
                        final_text = first if new_status == "accepted" else ""
                        segment["final_transcript"] = final_text
                        self._set_text(self.final_transcript, final_text)
            elif not status_changed_by_user:
                segment["status"] = "needs_review"
                self.status_var.set("needs_review")
        elif final_changed and not values["final_transcript"] and not status_changed_by_user:
            segment["status"] = "needs_review"
            self.status_var.set("needs_review")

        segment["selected_transcript"] = segment.get("final_transcript", "")
        segment["transcript_status"] = segment["status"]
        if changed:
            segment["human_override"] = True
            segment["updated_by"] = getpass.getuser()
            segment["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._update_table_row(segment)
            self._save_project(silent=True)
        return True

    def _save_detail(self) -> None:
        if self._commit_detail():
            self._save_project()

    def _choose_transcript(self, choice: str) -> None:
        if not self._commit_detail():
            return
        segment = self._find_segment(self.current_id)
        if not segment:
            return
        key = "transcript_a" if choice == "A" else "transcript_b"
        text = segment.get(key, "")
        if not text:
            messagebox.showinfo(
                "No transcript", f"Transcript {choice} is empty.", parent=self.root
            )
            return
        old_text = segment.get("final_transcript", "")
        self._record_change(segment, "final_transcript", old_text, text)
        old_status = segment["status"]
        self._record_change(segment, "status", old_status, "accepted")
        segment["final_transcript"] = text
        segment["selected_transcript"] = text
        segment["status"] = "accepted"
        segment["transcript_status"] = "accepted"
        segment["human_override"] = True
        segment["updated_by"] = getpass.getuser()
        segment["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.status_var.set("accepted")
        self._set_text(self.final_transcript, text)
        self._refresh_table(select_id=segment["segment_id"])
        self._save_project(silent=True)

    def _apply_time_range(self, segment: dict[str, Any], start: float, end: float) -> None:
        if start < 0 or end <= start or end > self.audio_info["duration"]:
            raise ValueError(
                "The edited range must fit inside the source WAV.")
        old = (segment["start"], segment["end"])
        new = (round(start, 3), round(end, 3))
        self._record_change(segment, "time_range", old, new)
        segment["source_spans"] = [{"start": start, "end": end}]
        segment["start"] = new[0]
        segment["end"] = new[1]
        segment["duration"] = round(end - start, 3)
        segment["human_override"] = True
        segment["updated_by"] = getpass.getuser()
        segment["updated_at"] = datetime.now(timezone.utc).isoformat()

    def _move_segment(self) -> None:
        if not self._commit_detail():
            return
        segment = self._find_segment(self.current_id)
        if not segment:
            return
        try:
            self._apply_time_range(
                segment, float(self.start_var.get()), float(self.end_var.get())
            )
        except ValueError as exc:
            messagebox.showerror("Invalid cut", str(exc), parent=self.root)
            return
        self._refresh_table(select_id=segment["segment_id"])
        self._fill_editor(segment)
        self._save_project(silent=True)

    def _play_segment(self) -> None:
        segment = self._find_segment(self.current_id)
        if not segment:
            return
        self._play_audio(segment)

    def _play_edited_range(self) -> None:
        if not self._commit_detail():
            return
        segment = self._find_segment(self.current_id)
        if segment:
            self._play_audio(segment)

    def _play_audio(self, segment: dict[str, Any]) -> None:
        if not self.wav_path:
            return
        try:
            audio, rate = pipeline.read_segment_audio(self.wav_path, segment)
            self._stop_audio()
            sd.play(audio, rate, blocking=False)
            self.progress_message.set(
                f"Playing {segment['segment_id']} ({segment['duration']:.2f} sec)."
            )
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("Playback failed", str(exc), parent=self.root)

    def _stop_audio(self) -> None:
        try:
            sd.stop()
        except sd.PortAudioError as exc:
            self.progress_message.set(f"Could not stop audio playback: {exc}")

    def _split_segment(self) -> None:
        if not self._commit_detail():
            return
        segment = self._find_segment(self.current_id)
        if not segment:
            return
        split_at = simpledialog.askfloat(
            "Split segment",
            f"Split at source time in seconds ({segment['start']:.3f}–"
            f"{segment['end']:.3f}):",
            minvalue=segment["start"] + 0.001,
            maxvalue=segment["end"] - 0.001,
            parent=self.root,
        )
        if split_at is None:
            return
        left_spans = pipeline.slice_spans(
            segment["source_spans"], segment["start"], split_at
        )
        right_spans = pipeline.slice_spans(
            segment["source_spans"], split_at, segment["end"]
        )
        if not left_spans or not right_spans:
            messagebox.showerror(
                "Invalid split", "Both resulting segments must contain audio.",
                parent=self.root,
            )
            return
        old_id = segment["segment_id"]
        position = self.segments.index(segment)
        split_event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "actor": getpass.getuser(),
            "action": "split",
            "segment_id": old_id,
            "before": segment,
            "split_at": round(split_at, 3),
        }
        self.audit_log.append(split_event)
        self.segments.pop(position)
        halves = []
        for suffix, spans in (("a", left_spans), ("b", right_spans)):
            half = dict(segment)
            candidate = f"{old_id}_{suffix}"
            existing_ids = {item["segment_id"] for item in self.segments}
            existing_ids.update(item["segment_id"] for item in halves)
            version = 2
            while candidate in existing_ids:
                candidate = f"{old_id}_{suffix}{version}"
                version += 1
            half["segment_id"] = candidate
            half["source_spans"] = spans
            half["human_changes"] = list(segment.get("human_changes", []))
            half["transcript_a"] = ""
            half["transcript_b"] = ""
            half["final_transcript"] = ""
            half["status"] = "needs_review"
            half["cer"] = None
            half["wer"] = None
            half["word_difference"] = None
            half["human_override"] = True
            pipeline._refresh_segment_times(half)
            halves.append(half)
            half["human_changes"].append(split_event)
            split_event.setdefault("created_segment_ids", []).append(
                half["segment_id"]
            )
        self.segments[position:position] = halves
        self.current_id = None
        self._refresh_table(select_id=halves[0]["segment_id"])
        self._fill_editor(halves[0])
        self._save_project(silent=True)

    def _merge_segments(self) -> None:
        selected_ids = set(self.table.selection())
        selected = [
            segment for segment in self.segments
            if segment["segment_id"] in selected_ids
        ]
        if len(selected) < 2:
            messagebox.showinfo(
                "Select segments", "Select at least two rows to merge.", parent=self.root
            )
            return
        if not self._commit_detail():
            return
        selected = [
            segment for segment in self.segments
            if segment["segment_id"] in selected_ids
        ]
        selected.sort(key=lambda item: (item["start"], item["end"]))
        merged_spans = []
        for segment in selected:
            merged_spans.extend(segment["source_spans"])
        merged_spans.sort(key=lambda span: span["start"])
        compact_spans: list[dict[str, float]] = []
        for span in merged_spans:
            if compact_spans and span["start"] <= compact_spans[-1]["end"]:
                compact_spans[-1]["end"] = max(
                    compact_spans[-1]["end"], span["end"]
                )
            else:
                compact_spans.append(dict(span))
        labels = {segment["label"] for segment in selected}
        new_id = f"merged_{selected[0]['segment_id']}"
        if self._find_segment(new_id) and new_id not in {
            item["segment_id"] for item in selected
        }:
            messagebox.showerror(
                "Merge failed", "A segment with the merged ID already exists.",
                parent=self.root,
            )
            return
        created = {
            "segment_id": new_id,
            "label": labels.pop() if len(labels) == 1 else "mixed",
            "confidence": None,
            "snr_db": None,
            "clean_noisy": "",
            "transcript_a": "",
            "transcript_b": "",
            "final_transcript": "",
            "status": "needs_review",
            "cer": None,
            "wer": None,
            "word_difference": None,
            "human_override": True,
            "updated_by": getpass.getuser(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "human_changes": [],
            "source_spans": compact_spans,
            "source_metadata": {
                "manual_merge": [segment["segment_id"] for segment in selected]
            },
        }
        pipeline._refresh_segment_times(created)
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "actor": getpass.getuser(),
            "action": "merge",
            "removed_segment_ids": [segment["segment_id"] for segment in selected],
            "before": selected,
            "created_segment_id": new_id,
        }
        created["human_changes"].append(event)
        self.audit_log.append(event)
        selected_ids = {segment["segment_id"] for segment in selected}
        first_position = min(self.segments.index(segment)
                             for segment in selected)
        self.segments = [
            segment for segment in self.segments
            if segment["segment_id"] not in selected_ids
        ]
        self.segments.insert(first_position, created)
        self.segments.sort(key=lambda item: (item["start"], item["end"]))
        self.current_id = None
        self._refresh_table(select_id=new_id)
        self._fill_editor(created)
        self._save_project(silent=True)

    def _add_segment(self) -> None:
        if not self._commit_detail():
            return
        if not self.audio_info:
            return
        start = simpledialog.askfloat(
            "Add segment", "Start time in source seconds:",
            minvalue=0, maxvalue=self.audio_info["duration"], parent=self.root,
        )
        if start is None:
            return
        end = simpledialog.askfloat(
            "Add segment", "End time in source seconds:",
            minvalue=start + 0.001, maxvalue=self.audio_info["duration"],
            parent=self.root,
        )
        if end is None:
            return
        number = 1
        existing_ids = {segment["segment_id"] for segment in self.segments}
        while f"manual_{number:04d}" in existing_ids:
            number += 1
        segment = {
            "segment_id": f"manual_{number:04d}",
            "label": "speech",
            "confidence": None,
            "snr_db": None,
            "clean_noisy": "",
            "transcript_a": "",
            "transcript_b": "",
            "final_transcript": "",
            "status": "needs_review",
            "cer": None,
            "wer": None,
            "word_difference": None,
            "human_override": True,
            "updated_by": getpass.getuser(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "human_changes": [],
            "source_spans": [{"start": start, "end": end}],
            "source_metadata": {"manually_added": True},
        }
        pipeline._refresh_segment_times(segment)
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "actor": getpass.getuser(),
            "action": "add",
            "created_segment_id": segment["segment_id"],
            "start": round(start, 3),
            "end": round(end, 3),
        }
        segment["human_changes"].append(event)
        self.audit_log.append(event)
        self.segments.append(segment)
        self.segments.sort(key=lambda item: (item["start"], item["end"]))
        self.current_id = None
        self._refresh_table(select_id=segment["segment_id"])
        self._fill_editor(segment)
        self._save_project(silent=True)

    def _delete_segments(self) -> None:
        selected = self._selected_segments()
        if not selected:
            messagebox.showinfo(
                "Select segments", "Select one or more rows to delete.",
                parent=self.root,
            )
            return
        if not messagebox.askyesno(
            "Delete segments",
            f"Delete {len(selected)} selected segment(s)?",
            parent=self.root,
        ):
            return
        if not self._commit_detail():
            return
        selected = self._selected_segments()
        removed = [segment["segment_id"] for segment in selected]
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "actor": getpass.getuser(),
            "action": "delete",
            "removed_segment_ids": removed,
            "before": selected,
        }
        self.audit_log.append(event)
        removed_set = set(removed)
        self.segments = [
            segment for segment in self.segments
            if segment["segment_id"] not in removed_set
        ]
        self.current_id = None
        self._refresh_table()
        if self.segments:
            self._fill_editor(self.segments[0])
        self._save_project(silent=True)

    def _transcribe_selected(self) -> None:
        if not self._commit_detail():
            return
        segment = self._find_segment(self.current_id)
        if not segment or segment["label"] not in {"speech", "mixed"}:
            messagebox.showinfo(
                "Select speech", "Select a speech or mixed segment to transcribe.",
                parent=self.root,
            )
            return
        self._start_transcription([segment], force=True)

    def _transcribe_pending(self) -> None:
        if not self._commit_detail():
            return
        pending = [
            segment for segment in self.segments
            if segment["label"] in {"speech", "mixed"}
            and (not segment.get("transcript_a") or not segment.get("transcript_b"))
        ]
        if not pending:
            messagebox.showinfo(
                "Nothing pending", "All speech/mixed segments already have both transcripts.",
                parent=self.root,
            )
            return
        if not messagebox.askyesno(
            "Transcribe all pending",
            f"Run both ASR models on up to {len(pending)} segments? "
            "This can take a long time and will download model data on first use.",
            parent=self.root,
        ):
            return
        self._start_transcription(pending, force=False)

    def _start_transcription(
        self, segments: list[dict[str, Any]], *, force: bool
    ) -> None:
        if self.busy or not self.wav_path:
            return
        jobs = []
        for segment in segments:
            if segment["label"] not in {"speech", "mixed"}:
                continue
            keys = ("transcript_a", "transcript_b") if force else tuple(
                key for key in ("transcript_a", "transcript_b")
                if not segment.get(key)
            )
            if keys:
                jobs.append((dict(segment), keys))
        if not jobs:
            return
        self.required_transcripts = {
            segment["segment_id"]: set(keys) for segment, keys in jobs
        }
        self.completed_transcripts = {
            segment["segment_id"]: set() for segment, _keys in jobs
        }
        for segment, _keys in jobs:
            current = self._find_segment(segment["segment_id"])
            if current is not None and not current["human_override"]:
                current["status"] = "needs_review"
                current["final_transcript"] = ""
                current["selected_transcript"] = ""
                current["transcript_status"] = "needs_review"
                self._update_table_row(current)
                if self.current_id == current["segment_id"]:
                    self._fill_editor(current)
        self._save_project(silent=True)
        self.busy = True
        self.transcription_errors = []
        self.transcription_cancel_event = threading.Event()
        self._set_processing(True)
        self.progress_var.set(0)
        self.progress_message.set("Starting Arabic speech recognition…")
        worker = threading.Thread(
            target=self._transcription_worker,
            args=(jobs, self.wav_path, self.whisper_model_var.get().strip(),
                  self.mms_model_var.get().strip(),
                  self.transcription_cancel_event),
            daemon=True,
        )
        worker.start()

    def _stop_transcription(self) -> None:
        if not self.busy:
            return
        if self.transcription_cancel_event is not None:
            self.transcription_cancel_event.set()
        self.progress_message.set("Stopping transcription; keeping completed results…")
        self.stop_transcription_button.configure(state="disabled")

    def _transcription_worker(
        self,
        jobs: list[tuple[dict[str, Any], tuple[str, ...]]],
        wav_path: Path,
        whisper_name: str,
        mms_name: str,
        cancel_event: threading.Event,
    ) -> None:
        total = sum(len(keys) for _, keys in jobs)
        complete = 0
        persistence_failed = False
        for key, model_name in (
            ("transcript_a", whisper_name),
            ("transcript_b", mms_name),
        ):
            if cancel_event.is_set() or persistence_failed:
                break
            selected_jobs = [
                (segment, keys) for segment, keys in jobs if key in keys
            ]
            if not selected_jobs:
                continue
            model = None
            try:
                if not model_name:
                    raise ValueError(f"Enter a model name for {key}.")
                self.worker_queue.put(
                    ("progress", f"Loading {model_name}…", complete, total))
                if key == "transcript_a":
                    model = pipeline.load_whisper_model(
                        model_name, lambda text: self.worker_queue.put(
                            ("progress", text, complete, total)
                        )
                    )
                else:
                    model = pipeline.load_mms_model(
                        model_name, lambda text: self.worker_queue.put(
                            ("progress", text, complete, total)
                        )
                    )
                for segment, _keys in selected_jobs:
                    if cancel_event.is_set() or persistence_failed:
                        break
                    self.worker_queue.put((
                        "active_segment", segment["segment_id"]
                    ))
                    self.worker_queue.put((
                        "progress",
                        f"{model_name}: {segment['segment_id']}",
                        complete,
                        total,
                    ))
                    try:
                        audio, sample_rate = pipeline.read_segment_audio(
                            wav_path, segment, target_rate=16000
                        )
                        if key == "transcript_a":
                            transcript, confidence = pipeline.transcribe_whisper(
                                model, audio)
                        else:
                            transcript, confidence = pipeline.transcribe_mms(
                                model, audio, sample_rate
                            )
                        saved = queue.Queue(maxsize=1)
                        self.worker_queue.put((
                            "asr_result", segment["segment_id"], key, transcript,
                            confidence, saved,
                        ))
                        if not saved.get():
                            self.worker_queue.put((
                                "transcription_save_error",
                                segment["segment_id"],
                            ))
                            persistence_failed = True
                            break
                        complete += 1
                        self.worker_queue.put((
                            "progress",
                            f"Transcribed {segment['segment_id']} with {model_name}",
                            complete,
                            total,
                        ))
                        if cancel_event.is_set() or persistence_failed:
                            break
                    except Exception as exc:
                        self.worker_queue.put((
                            "asr_segment_error", key, segment["segment_id"], str(
                                exc)
                        ))
            except Exception as exc:
                self.worker_queue.put(("asr_error", key, str(exc)))
            finally:
                if model is not None:
                    del model
                import gc
                gc.collect()
            if cancel_event.is_set() or persistence_failed:
                break
        self.worker_queue.put(("transcription_done", cancel_event.is_set()))

    def _set_processing(self, active: bool) -> None:
        self.load_button.configure(state="disabled" if active else "normal")
        self.stop_transcription_button.configure(
            state="normal" if active else "disabled"
        )
        for widget in (
            self.manifest_entry, self.wav_entry, self.manifest_button, self.wav_button,
            self.whisper_entry, self.mms_entry, self.transcribe_selected_button,
            self.transcribe_pending_button, self.export_button, self.save_button,
            self.move_button, self.split_button, self.merge_button, self.add_button,
            self.delete_button, self.label_box, self.status_box, self.start_entry,
            self.quality_box,
            self.end_entry, self.edit_a_button, self.edit_b_button,
            self.edit_final_button, self.use_a_button, self.use_b_button,
            self.play_button, self.play_range_button, self.stop_button,
        ):
            widget.configure(state="disabled" if active else "normal")
        if not active and not self.segments:
            self._set_editor_enabled(False)
        self.busy = active

    def _poll_worker(self) -> None:
        try:
            while True:
                event = self.worker_queue.get_nowait()
                kind = event[0]
                if kind == "active_segment":
                    _, segment_id = event
                    self._show_transcription_segment(segment_id)
                elif kind == "progress":
                    _, text, complete, total = event
                    self.progress_message.set(text)
                    self.progress_var.set(
                        0 if not total else min(100, complete * 100 / total)
                    )
                elif kind == "asr_result":
                    _, segment_id, key, text, confidence, saved = event
                    segment = self._find_segment(segment_id)
                    if segment is not None:
                        # Apply worker output only to the segment identified in the event.
                        # Do not change the Treeview selection here: forced selection churn
                        # causes _commit_detail() to write the currently displayed editor
                        # values to the wrong segment during a batch run.
                        segment[key] = text
                        confidence_key = (
                            "confidence_a" if key == "transcript_a"
                            else "confidence_b"
                        )
                        segment[confidence_key] = confidence
                        self.completed_transcripts.setdefault(
                            segment_id, set()).add(key)
                        required = self.required_transcripts.get(
                            segment_id, set())
                        if (self.completed_transcripts[segment_id] >= required
                                and segment["transcript_a"] and segment["transcript_b"]):
                            status, score, difference, word_error_rate = pipeline.compare_transcripts(
                                segment["transcript_a"], segment["transcript_b"]
                            )
                            segment["cer"] = round(score, 4)
                            segment["wer"] = round(word_error_rate, 4)
                            segment["word_difference"] = difference
                            if not segment["human_override"]:
                                segment["status"] = status
                                segment["final_transcript"] = (
                                    segment["transcript_a"] if status == "accepted" else ""
                                )
                        elif (self.completed_transcripts[segment_id] >= required
                              and not segment["human_override"]):
                            segment["status"] = "needs_review"
                            segment["final_transcript"] = ""
                        segment["selected_transcript"] = segment.get(
                            "final_transcript", ""
                        )
                        segment["transcript_status"] = segment["status"]
                        self._update_table_row(segment)
                        persisted = self._save_project(silent=True)
                    else:
                        persisted = False
                    saved.put(persisted)
                    if segment is not None:
                        self._show_transcription_segment(segment_id)
                elif kind == "asr_segment_error":
                    _, key, segment_id, error = event
                    model_name = "Whisper" if key == "transcript_a" else "MMS"
                    message = f"{model_name} failed for {segment_id}: {error}"
                    self.transcription_errors.append(message)
                    segment = self._find_segment(segment_id)
                    if segment is not None and not segment["human_override"]:
                        segment["status"] = "needs_review"
                        segment["final_transcript"] = ""
                        self._update_table_row(segment)
                        self._show_transcription_segment(segment_id)
                        self._save_project(silent=True)
                elif kind == "asr_error":
                    _, key, error = event
                    model_name = "Whisper" if key == "transcript_a" else "MMS"
                    self.transcription_errors.append(f"{model_name}: {error}")
                    self.progress_message.set(f"{model_name} failed: {error}")
                elif kind == "transcription_save_error":
                    _, segment_id = event
                    self.transcription_errors.append(
                        f"Could not save the transcript for {segment_id}; "
                        "transcription stopped."
                    )
                elif kind == "transcription_done":
                    _, cancelled = event
                    self._set_processing(False)
                    self.progress_var.set(100)
                    self.transcription_cancel_event = None
                    if cancelled:
                        self.progress_message.set(
                            "Transcription stopped. Completed transcript text remains available."
                        )
                    elif self.transcription_errors:
                        self.progress_message.set(
                            "Transcription finished with errors: "
                            + "; ".join(self.transcription_errors)
                        )
                    else:
                        self.progress_message.set(
                            "Transcription finished. Review disagreements and save corrections."
                        )
                    self._save_project(silent=True)
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(100, self._poll_worker)

    def _save_project(self, silent: bool = False) -> bool:
        if not all((self.project_path, self.manifest_path, self.wav_path)):
            return False
        try:
            pipeline.update_source_manifest(self.manifest, self.segments)
            pipeline.save_project(
                self.project_path,
                self.manifest_path,
                self.wav_path,
                self.manifest,
                self.segments,
                audit_log=self.audit_log,
            )
            pipeline.write_source_manifest(self.manifest_path, self.manifest)
            if not silent:
                self.progress_message.set(
                    f"Saved review progress to {self.project_path.name}.")
            return True
        except (OSError, TypeError, ValueError) as exc:
            if not silent:
                messagebox.showerror(
                    "Could not save project", str(exc), parent=self.root)
            else:
                self.progress_message.set(f"Autosave failed: {exc}")
            return False

    def _export_csv(self) -> None:
        if not self.segments or not self.manifest:
            messagebox.showinfo(
                "No project", "Load a project before exporting.", parent=self.root
            )
            return
        if not self._commit_detail():
            return
        default_name = (
            self.wav_path.stem + "_corpus.csv"
            if self.wav_path else "speech_corpus.csv"
        )
        path = filedialog.asksaveasfilename(
            title="Export reviewed corpus",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"),),
        )
        if not path:
            return
        try:
            exported = pipeline.export_csv(
                Path(path), self.manifest, self.segments, audit_log=self.audit_log
            )
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("CSV export failed",
                                 str(exc), parent=self.root)
            return
        self.progress_message.set(
            f"Exported {len(self.segments)} segments to {exported.name}.")
        messagebox.showinfo(
            "Export complete", f"Corpus CSV written to:\n{exported}", parent=self.root
        )

    def _close(self) -> None:
        if self.busy:
            messagebox.showwarning(
                "Transcription running",
                "Wait for transcription to finish before closing. Completed segments "
                "are autosaved as they finish.",
                parent=self.root,
            )
            return
        self.closing = True
        self._stop_audio()
        self._save_project(silent=True)
        self.root.destroy()

    def _show_transcription_segment(self, segment_id: str) -> None:
        """Select and display the segment currently being transcribed."""
        segment = self._find_segment(segment_id)
        if segment is None:
            return
        self._fill_editor(segment)
        if self.table.exists(segment_id):
            self.table.selection_set(segment_id)
            self.table.focus(segment_id)
            self.table.see(segment_id)


def main() -> None:
    root = tk.Tk()
    CorpusReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

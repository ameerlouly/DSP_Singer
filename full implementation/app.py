"""Clean desktop GUI for the complete DSP labelled speech corpus pipeline."""
from __future__ import annotations

import copy
import csv
import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import pipeline

COLORS = {"speech": "#24b8f2", "music": "#e95aab",
          "mixed": "#f5a623", "silence": "#9da9ad"}


class CorpusApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("DSP Labelled Speech Corpus — Full Implementation")
        root.geometry("1380x950")
        self.queue = queue.Queue()
        self.busy = False
        self.wav_path: Path | None = None
        self.video_path: Path | None = None
        self.project_dir: Path | None = None
        self.metadata: dict = {}
        self.regions: list[dict] = []
        self.segments: list[dict] = []
        self.selected_index: int | None = None
        self.drag_line = None
        self.drag_segment_index = None

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.download_tab = ttk.Frame(self.notebook, padding=18)
        self.process_tab = ttk.Frame(self.notebook, padding=18)
        self.review_tab = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(self.download_tab, text="1. Download")
        self.notebook.add(self.process_tab, text="2. Process")
        self.notebook.add(self.review_tab, text="3. Review & Export")

        self.build_download_tab()
        self.build_process_tab()
        self.build_review_tab()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(250, self.tick)

    def build_download_tab(self):
        card = ttk.LabelFrame(
            self.download_tab, text="Download a video", padding=24)
        card.place(relx=.5, rely=.42, anchor="center", relwidth=.72)
        ttk.Label(card, text="Video URL", font=(
            "Segoe UI", 11, "bold")).pack(anchor="w")
        self.url = tk.StringVar()
        ttk.Entry(card, textvariable=self.url).pack(fill="x", pady=(5, 14))

        row = ttk.Frame(card)
        row.pack(fill="x")
        ttk.Label(row, text="Save project folder:").pack(side="left")
        self.output_dir = tk.StringVar(value=str(pipeline.DEFAULT_OUTPUT))
        ttk.Entry(row, textvariable=self.output_dir).pack(
            side="left", fill="x", expand=True, padx=8)
        ttk.Button(row, text="Browse",
                   command=self.choose_output).pack(side="left")

        self.download_button = ttk.Button(
            card, text="Download and start processing", command=self.start_full_pipeline)
        self.download_button.pack(pady=20)
        self.download_status = tk.StringVar(
            value="Paste a YouTube or supported video URL.")
        ttk.Label(card, textvariable=self.download_status,
                  wraplength=850).pack(anchor="w")

        self.download_progress = ttk.Progressbar(
            card, mode="determinate", maximum=100)
        self.download_progress["value"] = 0
        self.download_progress.pack(fill="x", pady=(12, 0))

    def choose_output(self):
        path = filedialog.askdirectory(title="Choose project output folder")
        if path:
            self.output_dir.set(path)

    def build_process_tab(self):
        title = ttk.Label(
            self.process_tab, text="Automatic pipeline", font=("Segoe UI", 17, "bold"))
        title.pack(anchor="w")
        self.process_status = tk.StringVar(
            value="Waiting for a video download.")
        ttk.Label(self.process_tab, textvariable=self.process_status,
                  wraplength=1100).pack(anchor="w", pady=(8, 12))

        self.process_progress = ttk.Progressbar(
            self.process_tab, mode="determinate", maximum=100)
        self.process_progress["value"] = 0
        self.process_progress.pack(fill="x")
        self.step_status = tk.Text(
            self.process_tab, height=18, wrap="word", state="disabled")
        self.step_status.pack(fill="both", expand=True, pady=12)

        controls = ttk.Frame(self.process_tab)
        controls.pack(fill="x")
        ttk.Button(controls, text="Open completed review", command=lambda: self.notebook.select(
            self.review_tab)).pack(side="right")

    def build_review_tab(self):
        top = ttk.Frame(self.review_tab)
        top.pack(fill="x")
        ttk.Button(top, text="Load WAV project",
                   command=self.load_project).pack(side="left")
        ttk.Button(top, text="Play selected segment",
                   command=self.play_segment).pack(side="left", padx=5)
        ttk.Button(top, text="Stop", command=self.stop_playback).pack(
            side="left")
        ttk.Button(top, text="Export WAV + CSV",
                   command=self.export_project).pack(side="right")

        info = ttk.Frame(self.review_tab)
        info.pack(fill="x", pady=6)
        self.video_title = tk.StringVar(value="No project loaded")
        ttk.Label(info, textvariable=self.video_title, font=(
            "Segoe UI", 11, "bold")).pack(side="left")
        self.review_status = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.review_status).pack(side="right")

        self.figure = Figure(figsize=(12, 6), constrained_layout=True)
        self.wave_ax = self.figure.add_subplot(211)
        self.spec_ax = self.figure.add_subplot(212, sharex=self.wave_ax)
        self.wave_ax.set_title("Waveform")
        self.spec_ax.set_title("Spectrogram")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.review_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_figure_press)
        self.canvas.mpl_connect("button_release_event", self.on_figure_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_figure_motion)

        bottom = ttk.Frame(self.review_tab)
        bottom.pack(fill="x", pady=(8, 0))

        table_frame = ttk.Frame(bottom)
        table_frame.pack(side="left", fill="both", expand=True)
        columns = ("id", "start", "end", "duration", "label",
                   "confidence", "snr", "quality", "status")
        self.table = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=7, selectmode="browse")
        headings = [("id", "Segment", 105), ("start", "Start", 75), ("end", "End", 75),
                    ("duration", "Duration", 75), ("label", "Label", 75),
                    ("confidence", "Confidence", 90), ("snr", "SNR dB", 75),
                    ("quality", "Quality", 75), ("status", "Status", 110)]
        for key, title, width in headings:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, stretch=True)
        table_scroll = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=table_scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        table_scroll.pack(side="left", fill="y")
        self.table.bind("<<TreeviewSelect>>", self.select_segment)

        detail = ttk.LabelFrame(
            bottom, text="Selected segment", width=420, padding=8)
        detail.pack(side="right", fill="y", padx=(8, 0))
        self.detail_label = tk.StringVar(value="No segment selected")
        ttk.Label(detail, textvariable=self.detail_label,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(detail, text="Transcripts").pack(anchor="w", pady=(6, 0))
        self.transcript_text = tk.Text(detail, height=8, width=48, wrap="word")
        self.transcript_text.pack(fill="both", expand=True)
        self.human_override_button = ttk.Button(
            detail, text="Human Override", command=self.open_override)
        self.human_override_button.pack(fill="x", pady=6)

    def choose_output(self):
        path = filedialog.askdirectory(title="Choose project output folder")
        if path:
            self.output_dir.set(path)

    def append_process(self, text: str):
        self.step_status.configure(state="normal")
        self.step_status.insert("end", text + "\n")
        self.step_status.see("end")
        self.step_status.configure(state="disabled")

    def start_full_pipeline(self):
        url = self.url.get().strip()
        if not url:
            messagebox.showinfo("URL required", "Paste the video link first.")
            return
        if self.busy:
            messagebox.showinfo("Processing", "A pipeline is already running.")
            return
        self.busy = True
        self.download_button.config(state="disabled")
        self.download_progress["value"] = 5
        self.process_progress["value"] = 0
        self.process_status.set("Starting full pipeline...")
        self.step_status.configure(state="normal")
        self.step_status.delete("1.0", "end")
        self.step_status.configure(state="disabled")
        self.notebook.select(self.process_tab)
        output = Path(self.output_dir.get().strip())

        def worker():
            try:
                def cb(text):
                    self.queue.put(("progress", text))
                video, metadata = pipeline.download_video(url, output, cb)
                wav, audio_metadata = pipeline.extract_wav(video, cb)
                metadata.update(audio_metadata)
                metadata["video_file"] = video.name
                metadata["wav_file"] = wav.name
                regions = pipeline.classify_regions(wav, cb)
                segments = pipeline.build_segments(wav, regions, cb)

                segment_dir = wav.parent / "segments"
                for i, segment in enumerate(segments):
                    cb(f"Calculating SNR: {i+1}/{len(segments)}")
                    snr, note = pipeline.estimate_snr(
                        wav, segment["start"], segment["end"])
                    segment["snr_db"] = snr
                    segment["clean_noisy"] = "clean" if snr is not None and snr >= 20 else (
                        "noisy" if snr is not None else "unavailable")
                    segment["snr_note"] = note

                whisper_model = "base"
                mms_model = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
                for i, segment in enumerate(segments):
                    cb(f"Transcribing segment {i+1}/{len(segments)} with Whisper + Arabic ASR")
                    segment_wav = pipeline.write_segment_wav(
                        wav, segment, segment_dir)
                    try:
                        a = pipeline.transcribe_whisper(
                            segment_wav, whisper_model, cb)
                    except Exception as exc:
                        a = f"[Whisper unavailable: {exc}]"
                    try:
                        b = pipeline.transcribe_mms(segment_wav, mms_model, cb)
                    except Exception as exc:
                        b = f"[MMS unavailable: {exc}]"
                    status, cer, words = pipeline.agreement_status(a, b)
                    if a.startswith("[Whisper") or b.startswith("[MMS"):
                        status = "needs_review"
                    segment.update({"transcript_a": a, "transcript_b": b,
                                    "cer": round(cer, 4), "word_difference": words,
                                    "status": status,
                                    "final_transcript": a if status == "accepted" else ""})

                project_dir = wav.parent
                csv_path = pipeline.export_csv(
                    project_dir / "speech_corpus_segments.csv", wav, regions, segments)
                (project_dir / "metadata.json").write_text(json.dumps(metadata,
                                                                      ensure_ascii=False, indent=2), encoding="utf-8")
                self.queue.put(("complete", {"video": video, "wav": wav, "metadata": metadata,
                                             "regions": regions, "segments": segments,
                                             "project_dir": project_dir, "csv": csv_path}))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def load_project(self):
        wav_path = filedialog.askopenfilename(
            title="Open source WAV", filetypes=[("WAV", "*.wav")])
        if not wav_path:
            return
        wav_path = Path(wav_path)
        json_path = wav_path.with_name("speech_corpus_segments.json")
        csv_path = wav_path.with_name("speech_corpus_segments.csv")
        try:
            if json_path.exists():
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                self.regions = payload.get("regions", [])
                self.segments = payload.get("segments", [])
                self.metadata = json.loads((wav_path.parent / "metadata.json").read_text(
                    encoding="utf-8")) if (wav_path.parent / "metadata.json").exists() else {}
            elif csv_path.exists():
                with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
                    self.segments = list(csv.DictReader(handle))
                self.regions = []
                self.metadata = {}
            else:
                raise ValueError(
                    "No exported project JSON or CSV is beside this WAV")
            self.setup_review(wav_path, self.metadata,
                              self.regions, self.segments, wav_path.parent)
        except Exception as exc:
            messagebox.showerror("Could not load project", str(exc))

    def setup_review(self, wav_path: Path, metadata: dict, regions: list[dict], segments: list[dict], project_dir: Path):
        self.wav_path = wav_path
        self.project_dir = project_dir
        self.metadata = metadata
        self.regions = regions
        self.segments = segments
        self.video_title.set(metadata.get("title", wav_path.name))
        self.review_status.set(f"{len(self.segments)} segments")
        self.refresh_table()
        self.plot_full_audio()
        self.notebook.select(self.review_tab)

    def plot_full_audio(self):
        if self.wav_path is None:
            return
        data, rate = sf.read(str(self.wav_path),
                             dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        duration = len(mono) / rate
        plot_step = max(1, len(mono) // 200000)
        plot_mono = mono[::plot_step]
        times = np.arange(len(plot_mono)) * plot_step / rate
        self.wave_ax.clear()
        self.wave_ax.plot(times, plot_mono, color="#1f4e5f", linewidth=.55)
        self.wave_ax.set_ylabel("Amplitude")
        self.wave_ax.set_xlim(0, duration)

        self.spec_ax.clear()
        from scipy.signal import spectrogram, resample_poly
        spec_mono = resample_poly(mono, 1, 4)
        spec_rate = rate // 4
        frequencies, spec_times, power = spectrogram(
            spec_mono, fs=spec_rate, nperseg=512, noverlap=384)
        db = 10 * np.log10(power + 1e-12)
        self.spec_ax.pcolormesh(
            spec_times, frequencies, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        self.spec_ax.set_ylabel("Frequency (Hz)")
        self.spec_ax.set_xlabel("Time (s)")
        self.spec_ax.set_xlim(0, duration)
        self.segment_lines = []
        self.redraw_segment_overlays()
        self.canvas.draw_idle()

    def redraw_segment_overlays(self):
        for ax in (self.wave_ax, self.spec_ax):
            for artist in getattr(ax, "corpus_segment_artists", []):
                try:
                    artist.remove()
                except ValueError:
                    pass
            ax.corpus_segment_artists = []
        self.segment_lines = []
        for i, segment in enumerate(self.segments):
            start = float(segment.get("start", 0))
            end = float(segment.get("end", 0))
            label = str(segment.get("label", "speech"))
            color = COLORS.get(label, "#9da9ad")
            for ax in (self.wave_ax, self.spec_ax):
                patch = ax.axvspan(start, end, color=color,
                                   alpha=.16, zorder=1)
                ax.corpus_segment_artists.append(patch)
            if i == 0:
                line = self.wave_ax.axvline(
                    start, color="#12343b", linewidth=1.4, picker=5)
                self.segment_lines.append(line)
            line = self.wave_ax.axvline(
                end, color="#12343b", linewidth=1.4, picker=5)
            self.segment_lines.append(line)

    def refresh_table(self):
        for item in self.table.get_children():
            self.table.delete(item)
        for i, segment in enumerate(self.segments):
            snr = segment.get("snr_db", "")
            snr = "" if snr in (None, "") else f"{float(snr):.2f}"
            values = (segment.get("segment_id", i), f"{float(segment.get('start', 0)):.3f}",
                      f"{float(segment.get('end', 0)):.3f}", f"{float(segment.get('duration', 0)):.3f}",
                      segment.get(
                          "label", ""), f"{float(segment.get('confidence', 0))*100:.1f}%",
                      snr, segment.get("clean_noisy", ""), segment.get("status", ""))
            self.table.insert("", "end", iid=str(i), values=values)

    def select_segment(self, _event=None):
        selected = self.table.selection()
        if not selected:
            return
        i = int(selected[0])
        self.selected_index = i
        segment = self.segments[i]
        text = (f"Segment {segment.get('segment_id', '')} | "
                f"{float(segment.get('start', 0)):.3f}–{float(segment.get('end', 0)):.3f}s | "
                f"Label: {segment.get('label', '')} | ASR agreement: {segment.get('status', '')}\n"
                f"A: {segment.get('transcript_a', '')}\n"
                f"B: {segment.get('transcript_b', '')}\n"
                f"Final: {segment.get('final_transcript', '')}")
        self.detail_label.set(text)
        self.transcript_text.delete("1.0", "end")
        self.transcript_text.insert("1.0", f"TRANSCRIPT A — Whisper\n{segment.get('transcript_a', '')}\n\n"
                                    f"TRANSCRIPT B — MMS\n{segment.get('transcript_b', '')}\n\n"
                                    f"FINAL\n{segment.get('final_transcript', '')}")

    def on_figure_press(self, event):
        if event.inaxes not in (self.wave_ax, self.spec_ax) or event.xdata is None:
            return
        if event.inaxes is not self.wave_ax:
            return
        x = float(event.xdata)
        nearest = min(range(len(self.segment_lines)), key=lambda i: abs(
            float(self.segment_lines[i].get_xdata()[0]) - x))
        line_x = float(self.segment_lines[nearest].get_xdata()[0])
        if abs(line_x - x) > max(.25, (self.wave_ax.get_xlim()[1]-self.wave_ax.get_xlim()[0])*.01):
            for i, segment in enumerate(self.segments):
                if float(segment["start"]) <= x <= float(segment["end"]):
                    self.selected_index = i
                    self.table.selection_set(str(i))
                    self.table.see(str(i))
                    self.select_segment()
            return
        if nearest == 0:
            self.drag_segment_index = 0
        else:
            self.drag_segment_index = nearest
        self.drag_line = nearest

    def on_figure_motion(self, event):
        if self.drag_line is None or event.inaxes not in (self.wave_ax, self.spec_ax) or event.xdata is None:
            return
        x = float(event.xdata)
        i = self.drag_segment_index
        if i is None:
            return
        segment = self.segments[i]
        if i == 0:
            if x >= 0:
                segment["start"] = round(x, 3)
        else:
            segment = self.segments[i-1]
            if x >= float(segment["start"]):
                segment["end"] = round(x, 3)
                if i < len(self.segments):
                    self.segments[i]["start"] = round(x, 3)
        self.refresh_table()
        self.segment_lines[self.drag_line].set_xdata([x, x])
        self.canvas.draw_idle()

    def on_figure_release(self, _event):
        if self.drag_line is not None:
            self.drag_line = None
            self.drag_segment_index = None
            self.redraw_segment_overlays()
            self.canvas.draw_idle()

    def open_override(self):
        if self.selected_index is None:
            messagebox.showinfo("Select a segment", "Select a segment first.")
            return
        segment = self.segments[self.selected_index]
        dialog = tk.Toplevel(self.root)
        dialog.title("Human Override")
        dialog.geometry("620x600")
        dialog.transient(self.root)

        fields = ttk.Frame(dialog, padding=12)
        fields.pack(fill="x")
        start = tk.StringVar(value=str(segment.get("start", 0)))
        end = tk.StringVar(value=str(segment.get("end", 0)))
        label = tk.StringVar(value=str(segment.get("label", "speech")))
        snr = tk.StringVar(value="" if segment.get("snr_db")
                           is None else str(segment.get("snr_db")))
        quality = tk.StringVar(value=str(segment.get("clean_noisy", "")))
        status = tk.StringVar(value=str(segment.get("status", "needs_review")))
        annotator = tk.StringVar(value="student")

        grid = ttk.Frame(fields)
        grid.pack(fill="x")
        for text, var, vals in (("Start", start, None), ("End", end, None), ("Label", label, ["speech", "music", "mixed", "silence"]),
                                ("SNR dB", snr, None), ("Quality", quality,
                                                        ["clean", "noisy", "unavailable"]),
                                ("Status", status, [
                                 "accepted", "needs_review", "difficult"]),
                                ("Annotator", annotator, None)):
            row = ttk.Frame(grid)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=text, width=12).pack(side="left")
            if vals:
                ttk.Combobox(row, textvariable=var, values=vals, state="readonly").pack(
                    side="left", fill="x", expand=True)
            else:
                ttk.Entry(row, textvariable=var).pack(
                    side="left", fill="x", expand=True)

        ttk.Label(dialog, text="Transcript A — Whisper",
                  padding=(12, 6, 12, 0)).pack(anchor="w")
        ta = tk.Text(dialog, height=4, wrap="word")
        ta.pack(fill="x", padx=12)
        ta.insert("1.0", str(segment.get("transcript_a", "")))
        ttk.Label(dialog, text="Transcript B — MMS",
                  padding=(12, 6, 12, 0)).pack(anchor="w")
        tb = tk.Text(dialog, height=4, wrap="word")
        tb.pack(fill="x", padx=12)
        tb.insert("1.0", str(segment.get("transcript_b", "")))
        ttk.Label(dialog, text="Final transcript",
                  padding=(12, 6, 12, 0)).pack(anchor="w")
        final = tk.Text(dialog, height=4, wrap="word")
        final.pack(fill="x", padx=12)
        final.insert("1.0", str(segment.get("final_transcript", "")))

        def choose(source):
            final.delete("1.0", "end")
            final.insert("1.0", ta.get("1.0", "end").strip()
                         if source == "A" else tb.get("1.0", "end").strip())

        choices = ttk.Frame(dialog, padding=12)
        choices.pack(fill="x")
        ttk.Button(choices, text="Choose A",
                   command=lambda: choose("A")).pack(side="left")
        ttk.Button(choices, text="Choose B", command=lambda: choose(
            "B")).pack(side="left", padx=5)

        def apply():
            try:
                new_start, new_end = float(start.get()), float(end.get())
                if not 0 <= new_start < new_end:
                    raise ValueError("Start must be less than end.")
                new_snr = float(snr.get()) if snr.get().strip() else None
            except ValueError as exc:
                messagebox.showerror("Invalid data", str(exc), parent=dialog)
                return
            before = copy.deepcopy(segment)
            segment.update({"start": round(new_start, 3), "end": round(new_end, 3),
                            "duration": round(new_end-new_start, 3), "label": label.get(),
                            "snr_db": new_snr, "clean_noisy": quality.get(), "status": status.get(),
                            "transcript_a": ta.get("1.0", "end").strip(),
                            "transcript_b": tb.get("1.0", "end").strip(),
                            "final_transcript": final.get("1.0", "end").strip(),
                            "human_override": True, "updated_by": annotator.get().strip(),
                            "updated_at": datetime.now(timezone.utc).isoformat()})
            segment.setdefault("human_changes", []).append({"time_utc": datetime.now(timezone.utc).isoformat(),
                                                            "annotator": annotator.get().strip(),
                                                            "before": before, "after": copy.deepcopy(segment)})
            self.refresh_table()
            self.redraw_segment_overlays()
            self.plot_full_audio()
            self.save_project(show_message=False)
            dialog.destroy()

        ttk.Button(dialog, text="Apply override", command=apply).pack(
            padx=12, pady=12, ipadx=12)

    def play_segment(self):
        if self.selected_index is None:
            messagebox.showinfo("Select a segment",
                                "Select a segment in the table first.")
            return
        if self.wav_path is None:
            return
        segment = self.segments[self.selected_index]
        data, rate = sf.read(str(self.wav_path),
                             dtype="float32", always_2d=True)
        i1 = int(float(segment["start"])*rate)
        i2 = int(float(segment["end"])*rate)
        sd.stop()
        sd.play(data[i1:i2], samplerate=rate)

    def stop_playback(self):
        sd.stop()

    def save_project(self, show_message=True):
        if self.wav_path is None:
            return
        csv_path = self.wav_path.with_name("speech_corpus_segments.csv")
        pipeline.export_csv(csv_path, self.wav_path,
                            self.regions, self.segments)
        if show_message:
            messagebox.showinfo("Saved", f"Saved project data to {csv_path}")

    def export_project(self):
        if self.wav_path is None:
            messagebox.showinfo(
                "No project", "Load or complete a project first.")
            return
        destination = filedialog.askdirectory(title="Choose export folder")
        if not destination:
            return
        destination = Path(destination)
        import shutil
        dest_wav = destination / self.wav_path.name
        shutil.copy2(self.wav_path, dest_wav)
        csv_path = pipeline.export_csv(
            destination / "speech_corpus_segments.csv", dest_wav, self.regions, self.segments)
        segment_dir = destination / "segment_audio"
        segment_dir.mkdir(parents=True, exist_ok=True)
        for segment in self.segments:
            pipeline.write_segment_wav(dest_wav, segment, segment_dir)
            text_path = segment_dir / f"{segment['segment_id']}.txt"
            text_path.write_text(
                str(segment.get("final_transcript", "")), encoding="utf-8")
        messagebox.showinfo(
            "Export complete", f"Exported WAV, CSV, JSON, and segment files to:\n{destination}")

    def tick(self):
        try:
            kind, value = self.queue.get_nowait()
            if kind == "progress":
                self.append_process(value)
                self.process_status.set(value)
                text = str(value).lower()
                if "downloading video" in text:
                    self.download_progress["value"] = 30
                    self.process_progress["value"] = 10
                elif "video download finished" in text:
                    self.download_progress["value"] = 100
                    self.process_progress["value"] = 15
                elif "extracting" in text:
                    self.process_progress["value"] = 25
                elif "labeling windows" in text:
                    self.process_progress["value"] = 35
                elif "finding 10–15" in text:
                    self.process_progress["value"] = 50
                elif "calculating snr" in text:
                    self.process_progress["value"] = 65
                elif "transcribing" in text:
                    self.process_progress["value"] = 75
            elif kind == "complete":
                self.busy = False
                self.download_progress["value"] = 100
                self.process_progress["value"] = 100
                self.download_button.config(state="normal")
                self.append_process(
                    "Pipeline complete. Opening review window.")
                self.process_status.set("Processing complete.")
                self.setup_review(value["wav"], value["metadata"],
                                  value["regions"], value["segments"], value["project_dir"])
            elif kind == "error":
                self.busy = False
                self.download_progress["value"] = 0
                self.process_progress["value"] = 0
                self.download_button.config(state="normal")
                self.process_status.set("Processing failed.")
                self.append_process(f"ERROR: {value}")
                messagebox.showerror("Pipeline error", value)
        except queue.Empty:
            pass
        self.root.after(250, self.tick)

    def close(self):
        if self.busy:
            if not messagebox.askyesno("Processing in progress", "Stop the app while processing?"):
                return
        sd.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    CorpusApp(root)
    root.mainloop()

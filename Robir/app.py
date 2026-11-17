"""Current Tk interface for the speech corpus application."""
import copy
import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import soundfile as sf
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from audio_processing import calculate_spectrogram
from step3_label_regions import classify_file
from review_report import save_report, load_review
from step4_compact_segments import export_candidates

class SpectrogramApp:
    def __init__(self, root, container=None):
        self.root = root
        container = container or root
        root.title('WAV Spectrogram Viewer')
        root.geometry('1050x700')
        self.path = tk.StringVar(value='No WAV file selected')
        self.status = tk.StringVar(value='Select a WAV file, then click Show spectrogram.')
        self.clock_text = tk.StringVar(value='00:00.0 / 00:00.0')
        self.results = queue.Queue()
        self.selected = None
        self.plotted = None
        self.duration = 0.0
        self.position = 0.0
        self.cursor = None
        self.stream = None
        self.audio = None
        self.audio_lock = threading.Lock()
        self.anchor = None
        self.frames_read = 0
        self.play_rate = 0
        bar = ttk.Frame(container, padding=10)
        bar.pack(fill='x')
        self.select_button = ttk.Button(bar, text='Select WAV file', command=self.select_file)
        self.select_button.pack(side='left')
        ttk.Label(bar, textvariable=self.path).pack(side='left', padx=10, fill='x', expand=True)
        self.plot_button = ttk.Button(bar, text='Show spectrogram', command=self.start)
        self.plot_button.pack(side='right')
        controls = ttk.Frame(container, padding=(10, 0, 10, 6))
        controls.pack(fill='x')
        self.play_button = ttk.Button(controls, text='Play', command=self.play, state='disabled')
        self.play_button.pack(side='left')
        self.pause_button = ttk.Button(controls, text='Pause', command=self.pause, state='disabled')
        self.pause_button.pack(side='left', padx=5)
        self.stop_button = ttk.Button(controls, text='Stop', command=self.stop, state='disabled')
        self.stop_button.pack(side='left')
        ttk.Label(controls, textvariable=self.clock_text).pack(side='left', padx=12)
        ttk.Label(controls, text='Click the spectrogram to choose a time.').pack(side='left')
        ttk.Label(container, textvariable=self.status, padding=(10, 0, 10, 8), wraplength=1000).pack(fill='x')
        self.figure = Figure(figsize=(9, 5), constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.ax.set(xlabel='Time (seconds)', ylabel='Frequency (Hz)', title='Spectrogram')
        self.canvas = FigureCanvasTkAgg(self.figure, master=container)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, container, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side='bottom', fill='x')
        self.canvas.mpl_connect('button_press_event', self.on_click)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(50, self.tick)

    @staticmethod
    def format_time(seconds):
        minutes = int(seconds // 60)
        return f'{minutes:02d}:{seconds - minutes * 60:04.1f}'

    def update_cursor(self):
        self.clock_text.set(f'{self.format_time(self.position)} / {self.format_time(self.duration)}')
        if self.cursor is not None:
            self.cursor.set_xdata([self.position, self.position])
            self.canvas.draw_idle()

    def select_file(self):
        path = filedialog.askopenfilename(title='Select WAV file', filetypes=[('WAV audio', '*.wav'), ('All files', '*.*')])
        if path:
            self.close_stream()
            self.selected = path
            self.plotted = None
            self.position = self.duration = 0.0
            self.cursor = None
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.set(xlabel='Time (seconds)', ylabel='Frequency (Hz)', title='Spectrogram')
            self.canvas.draw_idle()
            self.update_cursor()
            self.path.set(path)
            self.set_controls(False)
            self.status.set('File selected. Click Show spectrogram.')

    def set_controls(self, enabled):
        state = 'normal' if enabled else 'disabled'
        for button in (self.play_button, self.pause_button, self.stop_button):
            button.config(state=state)

    def start(self):
        if not self.selected:
            messagebox.showinfo('Select a file', 'Please select a WAV file first.')
            return
        self.close_stream()
        self.plotted = None
        self.set_controls(False)
        self.select_button.config(state='disabled')
        self.plot_button.config(state='disabled')
        self.status.set('Calculating spectrogram. Long recordings can take some time...')
        path = self.selected
        def worker():
            try:
                self.results.put(('plot', calculate_spectrogram(path)))
            except Exception as exc:
                self.results.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def playback_time(self):
        if self.stream is None:
            return self.position
        with self.audio_lock:
            anchor = self.anchor
            last_frame = self.frames_read
        if anchor is None:
            return self.position
        try:
            seconds = anchor[1] / self.play_rate + max(0.0, self.stream.time - anchor[0])
        except Exception:
            return self.position
        return min(self.duration, last_frame / self.play_rate, seconds)

    def close_stream(self):
        stream, audio = self.stream, self.audio
        self.stream = None
        self.audio = None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
            stream.close()
        if audio is not None:
            audio.close()
        with self.audio_lock:
            self.anchor = None

    def play(self):
        if not self.plotted or self.stream is not None:
            return
        if self.position >= self.duration:
            self.position = 0.0
        try:
            self.audio = sf.SoundFile(self.plotted)
            self.play_rate = self.audio.samplerate
            self.audio.seek(min(int(self.position * self.play_rate), len(self.audio)))
            self.frames_read = self.audio.tell()
            self.anchor = None
            def callback(outdata, frames, timing, status):
                try:
                    first = self.audio.tell()
                    data = self.audio.read(frames, dtype='float32', always_2d=True)
                    outdata.fill(0)
                    outdata[:len(data)] = data
                    with self.audio_lock:
                        self.anchor = (timing.outputBufferDacTime, first)
                        self.frames_read = first + len(data)
                    if len(data) < frames:
                        raise sd.CallbackStop
                except sd.CallbackStop:
                    raise
                except Exception as exc:
                    self.results.put(('play_error', str(exc)))
                    raise sd.CallbackAbort
            self.stream = sd.OutputStream(samplerate=self.play_rate, channels=self.audio.channels,
                                          dtype='float32', callback=callback)
            self.stream.start()
            self.status.set('Playing. Click the spectrogram to move to another time.')
        except Exception as exc:
            self.close_stream()
            messagebox.showerror('Playback error', str(exc))

    def pause(self):
        if self.stream is not None:
            self.position = self.playback_time()
            self.close_stream()
            self.update_cursor()
            self.status.set('Paused.')

    def stop(self):
        self.close_stream()
        self.position = 0.0
        self.update_cursor()
        self.status.set('Stopped.')

    def on_click(self, event):
        if not self.plotted or event.inaxes != self.ax or event.xdata is None or self.toolbar.mode:
            return
        playing = self.stream is not None
        self.close_stream()
        self.position = min(self.duration, max(0.0, float(event.xdata)))
        self.update_cursor()
        if playing:
            self.play()

    def close(self):
        self.close_stream()
        self.root.destroy()

    def tick(self):
        try:
            while True:
                kind, result = self.results.get_nowait()
                if kind == 'play_error':
                    self.close_stream()
                    messagebox.showerror('Playback error', result)
                    continue
                self.select_button.config(state='normal')
                self.plot_button.config(state='normal')
                if kind == 'error':
                    self.status.set('Could not read or process this file.')
                    messagebox.showerror('Error', result)
                    continue
                times, frequencies, db, rate, duration, stride = result
                self.duration, self.position = duration, 0.0
                self.plotted = self.selected
                self.figure.clear()
                self.ax = self.figure.add_subplot(111)
                image = self.ax.imshow(db, origin='lower', aspect='auto', interpolation='nearest',
                    extent=[0, duration, 0, rate / 2], cmap='magma', vmin=-80, vmax=0)
                self.ax.set(xlabel='Time (seconds)', ylabel='Frequency (Hz)', title='Spectrogram')
                self.figure.colorbar(image, ax=self.ax, label='Power relative to peak (dB)')
                self.cursor = self.ax.axvline(0, color='cyan', linewidth=1.5)
                self.set_controls(True)
                self.update_cursor()
                self.status.set(f'Duration: {duration:.2f} s | Sample rate: {rate} Hz | FFT: 512 samples | Display combines up to {stride} time frames per column.')
        except queue.Empty:
            pass
        if self.stream is not None:
            self.position = self.playback_time()
            if not self.stream.active:
                self.position = self.duration
                self.close_stream()
                self.status.set('Playback finished.')
            self.update_cursor()
        self.root.after(50, self.tick)



COLORS = {'speech': '#24b8f2', 'music': '#e95aab', 'mixed': '#f5a623', 'silence': '#9da9ad'}


class RegionViewer(SpectrogramApp):
    def __init__(self, root):
        self.notebook=ttk.Notebook(root);self.notebook.pack(fill='both',expand=True)
        self.region_tab=ttk.Frame(self.notebook);self.notebook.add(self.region_tab,text='Step 3: Spectrogram')
        self.step4_tab=ttk.Frame(self.notebook);self.notebook.add(self.step4_tab,text='Step 4: Review clips')
        self.step5_tab=ttk.Frame(self.notebook);self.notebook.add(self.step5_tab,text='Step 5: SNR')
        from step5_window import SNRPanel
        self.snr_panel=SNRPanel(self.step5_tab,lambda:self.step4_review_window)
        self.snr_panel.pack(fill='both',expand=True)
        self.regions = []
        self.changes = []
        self.analysis_queue = queue.Queue()
        self.step4_queue = queue.Queue()
        self.step4_busy = False
        self.step4_review_window = None
        self.analyzing = False
        self.overlays = []
        self.show_regions = None
        self.dirty = False
        super().__init__(root,self.region_tab)
        root.title('Speech Corpus App')
        root.geometry('1250x900')
        panel = ttk.LabelFrame(self.region_tab, text='Step 3: automatic labels and manual review', padding=6)
        panel.pack(fill='x', padx=10, before=self.canvas.get_tk_widget())
        row = ttk.Frame(panel)
        row.pack(fill='x')
        self.analyze_button = ttk.Button(row, text='Automatically label regions', command=self.analyze)
        self.analyze_button.pack(side='left')
        self.save_button = ttk.Button(row, text='Save review', command=self.save_review)
        self.save_button.pack(side='left', padx=4)
        self.load_button = ttk.Button(row, text='Open review', command=self.open_review)
        self.load_button.pack(side='left')
        self.show_regions = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text='Show labels on spectrogram', variable=self.show_regions,
                        command=self.toggle_regions).pack(side='left', padx=10)
        ttk.Label(row, text='Annotator:').pack(side='left', padx=(12, 2))
        self.annotator = tk.StringVar()
        ttk.Entry(row, textvariable=self.annotator, width=18).pack(side='left')
        ttk.Label(row, text='Enter your name before editing.').pack(side='left', padx=8)
        columns = ('start', 'end', 'label', 'auto', 'confidence', 'review')
        self.table = ttk.Treeview(panel, columns=columns, show='headings', height=5, selectmode='browse')
        for col, label, width in [('start','Start (s)',75), ('end','End (s)',75), ('label','Current label',105),
                                  ('auto','Automatic label',110), ('confidence','Auto score',90), ('review','Review',80)]:
            self.table.heading(col, text=label)
            self.table.column(col, width=width, stretch=True)
        scroll = ttk.Scrollbar(panel, orient='vertical', command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side='left', fill='x', expand=True, pady=5)
        scroll.pack(side='left', fill='y')
        self.table.bind('<ButtonRelease-1>', self.select_region)
        self.table.bind('<KeyRelease-Up>', self.select_region)
        self.table.bind('<KeyRelease-Down>', self.select_region)
        edit = ttk.Frame(self.region_tab, padding=(10, 0))
        edit.pack(fill='x', before=self.canvas.get_tk_widget())
        self.label_value = tk.StringVar(value='speech')
        self.start_value, self.end_value = tk.StringVar(), tk.StringVar()
        ttk.Label(edit, text='Selected label:').pack(side='left')
        ttk.Combobox(edit, textvariable=self.label_value, values=list(COLORS), state='readonly', width=9).pack(side='left', padx=3)
        ttk.Label(edit, text='Start (s):').pack(side='left', padx=(6,0))
        ttk.Entry(edit, textvariable=self.start_value, width=9).pack(side='left')
        ttk.Label(edit, text='End (s):').pack(side='left', padx=(6,0))
        ttk.Entry(edit, textvariable=self.end_value, width=9).pack(side='left')
        self.edit_button = ttk.Button(edit, text='Apply correction', command=self.apply_edit)
        self.edit_button.pack(side='left', padx=8)
        ttk.Label(edit, text='Editing a boundary also adjusts its neighboring region.').pack(side='left')
        cut_row = ttk.Frame(self.region_tab, padding=(10, 5))
        cut_row.pack(fill='x', before=self.canvas.get_tk_widget())
        self.cut_button = ttk.Button(cut_row, text='Step 4: Join speech and generate candidates', command=self.start_step4)
        self.cut_button.pack(side='left')
        ttk.Button(cut_row, text='Open Step 4 review', command=self.open_step4_review).pack(side='left', padx=8)
        ttk.Label(cut_row, text='Shortens non-speech and joins speech/mixed. Review all joins.').pack(side='left', padx=8)
        ttk.Button(cut_row,text='Calculate SNR',command=lambda:self.notebook.select(self.step5_tab)).pack(side='left',padx=8)

    def confirm_discard(self):
        return not self.dirty or messagebox.askyesno('Unsaved review', 'Discard unsaved region changes?')

    def select_file(self):
        if self.analyzing or self.step4_busy or not self.confirm_discard():
            return
        if self.step4_review_window is not None and self.step4_review_window.winfo_exists() and self.step4_review_window.dirty:
            if not messagebox.askyesno('Unsaved Step 4 review','Changing the source discards the current unsaved Step 4 review. Continue?'):return
        old = self.selected
        super().select_file()
        if self.selected != old or (self.plotted is None and self.regions):
            if self.selected != old and self.step4_review_window is not None and self.step4_review_window.winfo_exists():
                self.step4_review_window.stop_playback();self.step4_review_window.destroy();self.step4_review_window=None
                self.snr_panel.results=[];self.snr_panel.snapshot=None;self.snr_panel.refresh()
            self.regions, self.changes, self.overlays = [], [], []
            self.dirty = False
            self.refresh_table()

    def start(self):
        if self.analyzing or self.step4_busy or not self.confirm_discard():
            return
        self.regions, self.changes = [], []
        self.dirty = False
        self.clear_overlays()
        self.refresh_table()
        super().start()

    def analyze(self):
        if not self.plotted:
            messagebox.showinfo('First display the spectrogram', 'Select a WAV file and show its spectrogram first.')
            return
        if self.analyzing or self.step4_busy or not self.confirm_discard():
            return
        self.analyzing = True
        self.analyze_button.config(state='disabled')
        self.select_button.config(state='disabled')
        self.plot_button.config(state='disabled')
        self.status.set('Starting analysis; first run downloads a large model...')
        path = self.plotted
        def worker():
            try:
                regions = classify_file(path, lambda msg: self.analysis_queue.put(('progress', msg)))
                self.analysis_queue.put(('done', regions))
            except Exception as exc:
                self.analysis_queue.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def clear_overlays(self):
        for item in self.overlays:
            try:
                item.remove()
            except (ValueError, AttributeError):
                pass
        self.overlays = []

    def toggle_regions(self):
        """Show or hide the colored overlays without altering region data."""
        visible = self.show_regions.get()
        for overlay in self.overlays:
            overlay.set_visible(visible)
        legend = self.ax.get_legend()
        if legend is not None:
            legend.set_visible(visible)
        self.canvas.draw_idle()

    def redraw_regions(self):
        self.clear_overlays()
        for region in self.regions:
            self.overlays.append(self.ax.axvspan(region['start'], region['end'], color=COLORS[region['label']], alpha=0.24, zorder=2))
        if self.cursor is not None:
            self.cursor.set_zorder(5)
        old_legend = self.ax.get_legend()
        if old_legend:
            old_legend.remove()
        self.ax.legend(handles=[Patch(facecolor=color, label=label.title()) for label, color in COLORS.items()], loc='upper right', ncol=4, fontsize='small')
        self.toggle_regions()

    def refresh_table(self, selected=None):
        self.table.delete(*self.table.get_children())
        for i, region in enumerate(self.regions):
            score = region.get('auto_confidence')
            self.table.insert('', 'end', iid=str(i), values=(f"{region['start']:.3f}", f"{region['end']:.3f}",
                region['label'], region.get('auto_label', ''), 'N/A' if score is None else f'{score:.2f}',
                'Yes' if region.get('needs_review') else ''))
        if selected is not None and str(selected) in self.table.get_children():
            self.table.selection_set(str(selected))
            self.table.see(str(selected))

    def select_region(self, event=None):
        selection = self.table.selection()
        if not selection:
            return
        region = self.regions[int(selection[0])]
        self.label_value.set(region['label'])
        self.start_value.set(str(region['start']))
        self.end_value.set(str(region['end']))
        # Ignore clicks on table headings or blank rows.
        if event is not None and getattr(event, 'num', None) == 1 and not self.table.identify_row(event.y):
            return
        was_playing = self.stream is not None
        self.close_stream()
        self.position = region['start']
        left, right = self.ax.get_xlim()
        if not left <= self.position <= right:
            width = right - left
            new_left = min(max(0.0, self.position - width / 2), max(0.0, self.duration - width))
            self.ax.set_xlim(new_left, min(self.duration, new_left + width))
        self.update_cursor()
        if was_playing:
            self.play()
        self.status.set(f"Selected {region['label']}; speech score: {region.get('speech_score')}, music score: {region.get('music_score')}. {region.get('note', '')}")

    def apply_edit(self):
        selection = self.table.selection()
        name = self.annotator.get().strip()
        if not selection or not name:
            messagebox.showinfo('Select and identify', 'Select a region and enter the annotator name first.')
            return
        i = int(selection[0])
        try:
            start, end = float(self.start_value.get()), float(self.end_value.get())
            if not (0 <= start < end <= self.duration):
                raise ValueError('Start and end must be within the audio, with start less than end.')
            if i == 0 and abs(start) > 0.0001 or i > 0 and start <= self.regions[i-1]['start']:
                raise ValueError('Start must remain after the previous region start; the first start must be 0.')
            if i == len(self.regions)-1 and abs(end-self.duration) > 0.0001 or i < len(self.regions)-1 and end >= self.regions[i+1]['end']:
                raise ValueError('End must remain before the next region end; the last end must equal the duration.')
        except ValueError as exc:
            messagebox.showerror('Invalid boundary', str(exc))
            return
        before = copy.deepcopy(self.regions[max(0,i-1):min(len(self.regions),i+2)])
        r = self.regions[i]
        r['start'], r['end'], r['label'] = start, end, self.label_value.get()
        r['needs_review'] = False
        r['manual_override'] = True
        if i > 0:
            if self.regions[i-1]['end'] != start:
                self.regions[i-1]['needs_review'] = True
            self.regions[i-1]['end'] = start
        if i + 1 < len(self.regions):
            if self.regions[i+1]['start'] != end:
                self.regions[i+1]['needs_review'] = True
            self.regions[i+1]['start'] = end
        self.changes.append({'time_utc': datetime.now(timezone.utc).isoformat(), 'annotator': name,
                             'region_index': i, 'before': before,
                             'after': copy.deepcopy(self.regions[max(0,i-1):min(len(self.regions),i+2)])})
        self.dirty = True
        self.refresh_table(i)
        self.redraw_regions()

    def save_review(self):
        if not self.regions:
            messagebox.showinfo('Nothing to save', 'Run automatic labeling or open a review first.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.html', filetypes=[('HTML report','*.html')], initialfile=Path(self.plotted).stem + '_review.html')
        if not path:
            return
        if Path(path).suffix.lower() not in ('.html', '.htm'):
            messagebox.showerror('Invalid filename', 'Save the review with an .html or .htm extension.')
            return
        try:
            payload = {'version': 1, 'audio_name': Path(self.plotted).name, 'audio_size': Path(self.plotted).stat().st_size,
                       'duration_seconds': self.duration, 'method': 'PANNs 2-second windows and RMS silence threshold',
                       'score_note': 'Automatic scores are uncalibrated; silence confidence is an energy-based heuristic.',
                       'regions': self.regions, 'changes': self.changes}
            save_report(path, payload)
            self.dirty = False
            self.status.set('Review saved.')
        except Exception as exc:
            messagebox.showerror('Save failed', str(exc))

    def open_review(self):
        if not self.plotted:
            messagebox.showinfo('First display the spectrogram', 'Select and display the matching WAV file first.')
            return
        if self.analyzing or self.step4_busy or not self.confirm_discard():
            return
        path = filedialog.askopenfilename(filetypes=[('Review files', '*.html *.htm *.json'), ('HTML files', '*.html *.htm'), ('JSON files', '*.json')])
        if not path:
            return
        try:
            payload = load_review(path)
            if payload['audio_name'] != Path(self.plotted).name or payload['audio_size'] != Path(self.plotted).stat().st_size or abs(payload['duration_seconds']-self.duration) > 0.001:
                raise ValueError('This review does not match the selected WAV filename, size, and duration.')
            regions = payload['regions']
            if not regions or abs(regions[0]['start']) > 0.0001 or abs(regions[-1]['end']-self.duration) > 0.0001:
                raise ValueError('The review does not cover the whole audio.')
            for i, region in enumerate(regions):
                if region['label'] not in COLORS or not 0 <= region['start'] < region['end'] <= self.duration or (i and abs(region['start']-regions[i-1]['end']) > 0.0001):
                    raise ValueError('Invalid or non-contiguous regions in this file.')
            self.regions, self.changes = regions, payload.get('changes', [])
            self.dirty = False
            self.refresh_table()
            self.redraw_regions()
            self.status.set('Review loaded.')
        except Exception as exc:
            messagebox.showerror('Could not open review', str(exc))

    def start_step4(self):
        if self.analyzing or self.step4_busy:
            return
        if not self.plotted or not self.regions:
            messagebox.showinfo('Step 3 required', 'Display a WAV and label or open its Step 3 review first.')
            return
        output_parent = filedialog.askdirectory(title='Choose a folder for Step 4 results')
        if not output_parent:
            return
        path = self.plotted
        region_snapshot = copy.deepcopy(self.regions)
        self.step4_busy = True
        self.cut_button.config(state='disabled')
        self.edit_button.config(state='disabled')
        self.analyze_button.config(state='disabled')
        self.select_button.config(state='disabled')
        self.plot_button.config(state='disabled')
        self.status.set('Preparing Step 4...')
        def worker():
            try:
                result = export_candidates(path, region_snapshot, output_parent,
                                           lambda text: self.step4_queue.put(('progress', text)))
                self.step4_queue.put(('done', result))
            except Exception as exc:
                self.step4_queue.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def open_step4_review(self, metadata_path=None):
        if not self.plotted:
            messagebox.showinfo('Open the source WAV', 'Select and display the matching original WAV first.')
            return
        if self.step4_review_window is not None and self.step4_review_window.winfo_exists():
            if metadata_path is None:
                self.notebook.select(self.step4_tab)
                return
            if self.step4_review_window.dirty and not messagebox.askyesno('Unsaved review','Replace the current Step 4 review without saving?'):
                return
            self.step4_review_window.stop_playback();self.step4_review_window.destroy();self.step4_review_window=None
        if metadata_path is None:
            metadata_path=filedialog.askopenfilename(title='Open Step 4 metadata or saved review',
                filetypes=[('Step 4 JSON files','*.json')])
            if not metadata_path:
                return
        try:
            from step4_review_window import Step4ReviewPanel
            self.close_stream()
            self.step4_review_window=Step4ReviewPanel(self.step4_tab,self.plotted,metadata_path,self.annotator.get())
            self.step4_review_window.pack(fill='both',expand=True)
            self.notebook.select(self.step4_tab)
        except Exception as exc:
            messagebox.showerror('Could not open Step 4 review',str(exc))

    def close(self):
        if self.analyzing or self.step4_busy:
            messagebox.showinfo('Processing in progress', 'Please wait for processing to finish before closing.')
            return
        if self.step4_review_window is not None and self.step4_review_window.winfo_exists():
            if self.step4_review_window._reseg_busy:
                messagebox.showinfo('Processing','Please wait for re-segmentation to finish.');return
            if self.step4_review_window.dirty and not messagebox.askyesno('Unsaved Step 4 review','Close without saving the Step 4 review?'):return
            self.step4_review_window.stop_playback()
        if self.snr_panel.busy:
            messagebox.showinfo('Processing','Please wait for SNR calculation to finish.');return
        if self.confirm_discard():
            super().close()

    def tick(self):
        super().tick()
        try:
            while True:
                kind, result = self.analysis_queue.get_nowait()
                if kind == 'progress':
                    self.status.set(result)
                    continue
                self.analyzing = False
                self.analyze_button.config(state='normal')
                self.select_button.config(state='normal')
                self.plot_button.config(state='normal')
                if kind == 'error':
                    self.status.set('Automatic labeling failed.')
                    messagebox.showerror('Automatic labeling error', result)
                else:
                    self.regions, self.changes, self.dirty = result, [], True
                    self.refresh_table()
                    self.redraw_regions()
                    self.status.set('Automatic labels ready. Review the results and save your changes.')
        except queue.Empty:
            pass
        try:
            while True:
                kind, result = self.step4_queue.get_nowait()
                if kind == 'progress':
                    self.status.set(result)
                    continue
                self.step4_busy = False
                self.cut_button.config(state='normal')
                self.edit_button.config(state='normal')
                self.analyze_button.config(state='normal')
                self.select_button.config(state='normal')
                self.plot_button.config(state='normal')
                if kind == 'error':
                    self.status.set('Step 4 could not finish.')
                    messagebox.showerror('Step 4 error', result)
                else:
                    self.status.set(f"Step 4 generated {result['segments']} candidates and {result['unresolved']} unresolved spans.")
                    self.open_step4_review(str(Path(result['output_dir']) / 'step4_segments.json'))
        except queue.Empty:
            pass


if __name__ == '__main__':
    root = tk.Tk()
    RegionViewer(root)
    root.mainloop()









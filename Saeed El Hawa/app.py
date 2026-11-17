"""Speech Corpus desktop interface. Run with your existing virtual environment."""
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, simpledialog
from pipeline import download_video, extract_audio, rename_video_folder


class CorpusApp:
    BG = '#f0f4f5'
    INK = '#173b40'
    MUTED = '#60767a'
    ACCENT = '#147d72'

    def __init__(self, root):
        self.root = root
        root.title('Speech Corpus Studio')
        root.geometry('980x820')
        root.minsize(820, 740)
        root.configure(bg=self.BG)
        self.events = queue.Queue()
        self.busy = False
        self.started = 0
        self.last_result = None
        self.report_path = None
        self.buttons = []
        self.entries = []
        self.url = tk.StringVar()
        self.output = tk.StringVar(value=str(Path(__file__).resolve().parent / 'data'))
        self.folder_name = tk.StringVar()
        self.video = tk.StringVar()
        self.status = tk.StringVar(value='Ready to begin')
        self.elapsed = tk.StringVar(value='')
        self.result_text = tk.StringVar(value='Your completed file will appear here.')
        self.setup_styles()

        header = tk.Frame(root, bg=self.INK, padx=28, pady=22)
        header.pack(fill='x')
        tk.Label(header, text='SPEECH CORPUS / PARTS 01–02', bg=self.INK,
                 fg='#94d9cb', font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        tk.Label(header, text='From video to audio.', bg=self.INK, fg='white',
                 font=('Segoe UI', 26, 'bold')).pack(anchor='w', pady=(6, 2))
        tk.Label(header, text='Download your source. Extract audio. Build your corpus.',
                 bg=self.INK, fg='#c5d7d9', font=('Segoe UI', 11)).pack(anchor='w')

        main = tk.Frame(root, bg=self.BG, padx=24, pady=18)
        main.pack(fill='both', expand=True)
        step1 = self.card(main)
        self.section(step1, '01', 'Download a video', 'Paste a video link to save the source and its metadata.')
        self.entry(step1, self.url).pack(fill='x', pady=(12, 8))
        folder_row = tk.Frame(step1, bg='white')
        folder_row.pack(fill='x')
        tk.Label(folder_row, text='Save to', bg='white', fg=self.MUTED,
                 font=('Segoe UI', 10)).pack(side='left', padx=(0, 10))
        self.entry(folder_row, self.output).pack(side='left', fill='x', expand=True)
        self.button(folder_row, 'Browse…', self.choose_folder).pack(side='left', padx=(8, 0))
        self.button(folder_row, 'Download video', lambda: self.start('download'), True).pack(side='left', padx=(8, 0))

        name_row = tk.Frame(step1, bg='white')
        name_row.pack(fill='x', pady=(8, 0))
        tk.Label(name_row, text='Folder name (optional)', bg='white', fg=self.MUTED,
                 font=('Segoe UI', 10)).pack(side='left', padx=(0, 10))
        self.entry(name_row, self.folder_name).pack(side='left', fill='x', expand=True)
        tk.Label(name_row, text='Blank uses video1, video2, …', bg='white', fg=self.MUTED,
                 font=('Segoe UI', 10)).pack(side='left', padx=(10, 0))

        step2 = self.card(main)
        self.section(step2, '02', 'Extract audio', 'Use the downloaded video or select a file from your computer.')
        source_row = tk.Frame(step2, bg='white')
        source_row.pack(fill='x', pady=(12, 10))
        self.entry(source_row, self.video).pack(side='left', fill='x', expand=True)
        self.button(source_row, 'Choose video…', self.choose_video).pack(side='left', padx=(8, 0))
        self.button(source_row, 'Extract audio', lambda: self.start('extract'), True).pack(side='left', padx=(8, 0))
        tk.Label(step2, text='OUTPUT FORMAT     16,000 Hz   /   Mono   /   16-bit PCM WAV',
                 bg='#eaf6f1', fg='#266b5d', font=('Segoe UI', 10), padx=12, pady=7).pack(anchor='w')

        result_card = self.card(main)
        state_row = tk.Frame(result_card, bg='white')
        state_row.pack(fill='x')
        self.status_label = tk.Label(state_row, textvariable=self.status, bg='white',
                                     fg=self.ACCENT, font=('Segoe UI', 11, 'bold'))
        self.status_label.pack(side='left')
        tk.Label(state_row, textvariable=self.elapsed, bg='white', fg=self.MUTED).pack(side='right')
        self.progress = ttk.Progressbar(result_card, mode='indeterminate', style='Studio.Horizontal.TProgressbar')
        self.progress.pack(fill='x', pady=(10, 9))
        tk.Label(result_card, textvariable=self.result_text, bg='white', fg=self.MUTED,
                 anchor='w', wraplength=820, justify='left', font=('Segoe UI', 9)).pack(fill='x')
        links = tk.Frame(result_card, bg='white')
        links.pack(fill='x', pady=(10, 0))
        self.report_button = ttk.Button(links, text='Open HTML report', command=self.open_report, state='disabled')
        self.report_button.pack(side='left')
        self.folder_button = ttk.Button(links, text='Open output folder', command=self.open_folder, state='disabled')
        self.folder_button.pack(side='left', padx=8)
        self.button(links, 'Rename video folder', self.rename_folder).pack(side='left')

        log_head = tk.Frame(main, bg=self.BG)
        log_head.pack(fill='x', pady=(1, 5))
        tk.Label(log_head, text='ACTIVITY', bg=self.BG, fg=self.MUTED,
                 font=('Segoe UI', 9, 'bold')).pack(side='left')
        ttk.Button(log_head, text='Clear log', command=self.clear_log).pack(side='right')
        log_frame = tk.Frame(main, bg='#dbe5e7', padx=1, pady=1)
        log_frame.pack(fill='both', expand=True)
        self.logbox = tk.Text(log_frame, wrap='word', height=5, state='disabled', relief='flat',
                              bg='#faffff', fg=self.INK, font=('Consolas', 10), padx=12, pady=10)
        scrollbar = ttk.Scrollbar(log_frame, command=self.logbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.logbox.configure(yscrollcommand=scrollbar.set)
        self.logbox.pack(fill='both', expand=True)
        self.logbox.tag_configure('error', foreground='#b42336')
        self.logbox.tag_configure('normal', foreground=self.INK)
        self.log('Ready. Download a video or choose an existing file to extract audio.')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.poll()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 10), padding=(12, 8),
                        background='#edf3f4', foreground=self.INK, borderwidth=0)
        style.map('TButton', background=[('active', '#dce9e9')],
                  foreground=[('disabled', '#89999b')])
        style.configure('Accent.TButton', background=self.ACCENT, foreground='white')
        style.map('Accent.TButton', background=[('disabled', '#c5d4d2'), ('active', '#10665e')],
                  foreground=[('disabled', '#536e69')])
        style.configure('TEntry', padding=8, fieldbackground='#f8fafb', foreground=self.INK)
        style.configure('Studio.Horizontal.TProgressbar', background=self.ACCENT,
                        troughcolor='#e8efef', borderwidth=0, thickness=5)

    def card(self, parent):
        card = tk.Frame(parent, bg='white', padx=18, pady=14,
                        highlightbackground='#dde6e8', highlightthickness=1)
        card.pack(fill='x', pady=(0, 12))
        return card

    def section(self, parent, number, title, subtitle):
        tk.Label(parent, text=f'{number}   {title}', bg='white', fg=self.INK,
                 font=('Segoe UI', 14, 'bold')).pack(anchor='w')
        tk.Label(parent, text=subtitle, bg='white', fg=self.MUTED,
                 font=('Segoe UI', 10)).pack(anchor='w', pady=(3, 0))

    def entry(self, parent, variable):
        entry = ttk.Entry(parent, textvariable=variable, font=('Segoe UI', 10))
        self.entries.append(entry)
        return entry

    def button(self, parent, text, command, primary=False):
        button = ttk.Button(parent, text=text, command=command,
                            style='Accent.TButton' if primary else 'TButton')
        self.buttons.append(button)
        return button

    def choose_folder(self):
        selected = filedialog.askdirectory(parent=self.root, title='Choose download folder')
        if selected:
            self.output.set(selected)

    def choose_video(self):
        selected = filedialog.askopenfilename(parent=self.root, title='Choose a local video')
        if selected:
            self.video.set(selected)

    def rename_folder(self):
        if self.busy:
            return
        recent = self.last_result.parent if self.last_result else None
        selected = filedialog.askdirectory(parent=self.root, title='Choose a video folder to rename',
                                          initialdir=str(recent) if recent else self.output.get())
        if not selected:
            return
        folder = Path(selected).resolve()
        name = simpledialog.askstring('Rename video folder', f'New name for “{folder.name}”',
                                      initialvalue=folder.name, parent=self.root)
        if name is None:
            return
        current_video = self.video.get().strip()
        try:
            video_path = Path(current_video).resolve() if current_video else None
            renamed = rename_video_folder(folder, name)
        except (OSError, ValueError) as exc:
            messagebox.showerror('Cannot rename folder', str(exc), parent=self.root)
            return
        if renamed == folder:
            return
        if video_path and video_path.parent == folder:
            self.video.set(str(renamed / video_path.name))
        if self.last_result and self.last_result.parent == folder:
            self.last_result = renamed / self.last_result.name
            self.result_text.set(str(self.last_result))
        if self.report_path and self.report_path.parent == folder:
            self.report_path = renamed / self.report_path.name
        self.status.set('Video folder renamed')
        self.log(f'Renamed folder: {renamed}')

    def log(self, text, error=False):
        self.logbox.configure(state='normal')
        self.logbox.insert('end', f'{datetime.now():%H:%M:%S}  {text}\n', 'error' if error else 'normal')
        self.logbox.see('end')
        self.logbox.configure(state='disabled')

    def clear_log(self):
        self.logbox.configure(state='normal')
        self.logbox.delete('1.0', 'end')
        self.logbox.configure(state='disabled')

    def start(self, kind):
        if self.busy:
            return
        value = (self.url.get() if kind == 'download' else self.video.get()).strip()
        folder = self.output.get().strip()
        folder_name = self.folder_name.get().strip()
        if not value or (kind == 'download' and not folder):
            messagebox.showinfo('Missing input', 'Enter a video URL and download folder, or choose a local video.', parent=self.root)
            return
        self.busy = True
        self.started = time.monotonic()
        self.last_result = self.report_path = None
        self.report_button.configure(state='disabled')
        self.folder_button.configure(state='disabled')
        self.result_text.set('Processing… Your saved file will appear here when complete.')
        self.status.set('Downloading video…' if kind == 'download' else 'Extracting audio…')
        self.status_label.configure(fg=self.ACCENT)
        for widget in self.buttons + self.entries:
            widget.configure(state='disabled')
        self.progress.start()
        self.log(self.status.get())

        def worker():
            try:
                logger = lambda text: self.events.put(('log', text))
                if kind == 'download':
                    result = download_video(value, folder, logger, folder_name=folder_name)
                else:
                    result = extract_audio(value, logger)
                self.events.put(('success', (kind, str(result))))
            except Exception as exc:
                self.events.put(('error', str(exc)))
            finally:
                self.events.put(('done', None))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        # Bound each pass so a burst of progress messages cannot block the UI.
        for _ in range(100):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'log':
                self.log(value)
            elif kind == 'success':
                operation, path = value
                self.last_result = Path(path)
                if operation == 'download':
                    self.video.set(path)
                    self.folder_name.set('')
                self.report_path = (self.last_result.parent / 'metadata.html' if operation == 'download'
                                    else self.last_result.with_suffix('.html'))
                self.result_text.set(path)
                self.status.set('Video downloaded — ready to extract' if operation == 'download' else 'Audio extracted and verified')
                self.log('Completed successfully.')
            elif kind == 'error':
                self.status.set('Could not complete — see activity below')
                self.status_label.configure(fg='#b42336')
                self.result_text.set('Check the error message, adjust the input, and try again.')
                self.log('ERROR: ' + value, error=True)
            elif kind == 'done':
                self.elapsed.set(f'{int(time.monotonic() - self.started)}s elapsed')
                self.busy = False
                self.progress.stop()
                for widget in self.buttons + self.entries:
                    widget.configure(state='normal')
                self.report_button.configure(state='normal' if self.report_path and self.report_path.is_file() else 'disabled')
                self.folder_button.configure(state='normal' if self.last_result else 'disabled')
        if self.busy:
            self.elapsed.set(f'{int(time.monotonic() - self.started)}s elapsed')
        self.root.after(150, self.poll)

    def open_report(self):
        if not self.report_path or not self.report_path.is_file():
            messagebox.showinfo('Report unavailable', 'The HTML report could not be found.', parent=self.root)
            return
        webbrowser.open(self.report_path.resolve().as_uri())

    def open_folder(self):
        if not self.last_result:
            return
        folder = str(self.last_result.parent)
        try:
            if sys.platform == 'win32':
                os.startfile(folder)
            else:
                subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', folder])
        except OSError as exc:
            messagebox.showerror('Cannot open folder', str(exc), parent=self.root)

    def close(self):
        if self.busy:
            messagebox.showinfo('Processing', 'Please wait for the current operation to finish before closing.', parent=self.root)
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    CorpusApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()

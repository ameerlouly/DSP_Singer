# Speech Corpus Studio · Parts 1 and 2

Download one online video and its source metadata, then extract the first audio track as a **16 kHz, mono, 16-bit PCM WAV**. Use `app.py` to run the Tkinter demo and `pipeline.py` to integrate the same functions into the team’s PyQt app.

## Run the demo on Windows

### 1. Add the installed FFmpeg folder to PATH

1. In File Explorer, find the folder that contains **both** `ffmpeg.exe` and `ffprobe.exe`. It is usually the `bin` folder inside your FFmpeg installation. Copy its full path, for example `C:\Tools\ffmpeg\bin`.
2. Open the Start menu, search for **Edit environment variables for your account**, and open it.
3. Under **User variables**, select **Path → Edit → New**. Paste the folder path and click **OK** in each dialog. Add the folder itself without the `ffmpeg.exe` filename.
4. Close your current PowerShell window and open a new one. If you’re using the terminal inside VS Code, restart VS Code. Run:

   ```powershell
   ffmpeg -version
   ffprobe -version
   ```

   Continue once both commands print version information.

For a temporary setup in one PowerShell window, you can use the commands below instead of editing Windows settings. Replace the example directory with your own and run the demo from that same window.

```powershell
$ffmpegBin = 'C:\Tools\ffmpeg\bin'
$env:Path = "$ffmpegBin;$env:Path"
ffmpeg -version
ffprobe -version
```

### 2. Set up Python and open the demo

1. Put the matching versions of `app.py` and `pipeline.py` side by side in your project folder. Open PowerShell in that folder. You can open the folder in File Explorer, type `powershell` into the address bar, and press Enter.
2. If the project does not already have a `.venv` folder, create one with `py -m venv .venv`. This requires Python to be installed. If you already have a working `.venv` for this project, reuse it.
3. Install the downloader and start the demo using the project’s Python:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install "yt-dlp[default]"
   .\.venv\Scripts\python.exe app.py
   ```

   These commands run the virtual environment’s Python directly, so no PowerShell activation command is needed. If the demo reports that `tkinter` is missing, use a Windows Python installation with Tcl/Tk support and create the environment from that Python.

### 3. Download a video and extract the WAV

1. Paste the URL of one finished video into the first field. Use a short video for your first demo run. The one-hour minimum applies to the final corpus sources.
2. Leave **Save to** at the default `data` folder or choose another output directory with **Browse…**.
3. Leave **Folder name (optional)** blank for `video1`, `video2`, etc. To assign a name now, type a single folder name such as `Egyptian_Interview`.
4. Click **Download video**. When it finishes, the selected video field fills in automatically. You can view the source metadata with **Open HTML report**.
5. Click **Extract audio**. Wait for **Audio extracted and verified**, then choose **Open output folder**. It contains the downloaded `source.*` video, `metadata.html`, the `source_16k_mono.wav` file, and `source_16k_mono.html` with the audio format details. The video extension depends on what was downloaded.
6. To rename a completed folder, click **Rename video folder**, select the video’s subfolder, and enter its new name. The video, WAV, and reports move together. The app updates any selected video and displayed result paths for that folder.

If you already have a local video, click **Choose video…** and then **Extract audio**. The WAV and audio report are saved next to that video.

## Call the functions from PyQt

Copy `pipeline.py` into the team project and import the functions there. The PyQt app uses its own interface and only needs the pipeline module plus the same `yt-dlp`, `ffmpeg`, and `ffprobe` setup.

| Function | Arguments and return value |
|---|---|
| `download_video(url, output_root='data', log=print, *, folder_name=None)` | Takes the video URL and the parent output directory. A blank or omitted `folder_name` uses automatic numbering. Saves the source metadata as `metadata.html` and returns the downloaded video as a `pathlib.Path`. |
| `extract_audio(video_path, log=print)` | Takes a video path as a string or `Path`. Extracts and verifies the WAV and writes an HTML audio report beside it. Returns the WAV as a `Path`. |
| `rename_video_folder(folder, new_name)` | Takes the completed video folder containing `metadata.html` and the new folder name. Rejects a name already in use. Returns the folder’s new `Path`. |

### Download and extract

```python
from pathlib import Path

from pipeline import download_video, extract_audio

# Supply these from your project and the PyQt input fields.
output_root = Path(r"C:\SpeechCorpus\data")
url = "YOUR_VIDEO_URL"
folder_name = ""  # Or "Egyptian_Interview"

video_path = download_video(url, output_root=output_root, folder_name=folder_name)
wav_path = extract_audio(video_path)

video_report = video_path.parent / "metadata.html"
audio_report = wav_path.with_suffix(".html")
# Pass wav_path to Part 3.
```

Run download and extraction in a **background worker** so the PyQt window remains responsive. Both functions take a `log=` callback that receives one status string at a time. Inside a PyQt worker you can pass a signal’s `.emit` method, such as `log=self.status_message.emit`, and connect the signal to the UI. Catch errors in the worker and send the message to the UI as well.

### Rename a folder from PyQt

```python
from pipeline import rename_video_folder

new_folder = rename_video_folder(video_path.parent, "Egyptian_Interview")
video_path = new_folder / video_path.name
wav_path = new_folder / wav_path.name
video_report = new_folder / "metadata.html"
audio_report = wav_path.with_suffix(".html")
```

Use the updated paths in the interface and in any following parts of the pipeline. The example assumes extraction has already finished. If you rename before extraction, update `video_path` and pass it to `extract_audio()` afterward.

## Optional command line check

From the project folder, run both stages in one command. Replace the placeholder with a video URL and choose a folder name that does not already exist.

```powershell
.\.venv\Scripts\python.exe pipeline.py --url "YOUR_VIDEO_URL" --output data --folder-name quick_demo
```

To check the WAV, edit the path below if you used a different folder name. Expect `codec_name=pcm_s16le`, `sample_rate=16000`, and `channels=1`.

```powershell
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 ".\data\quick_demo\source_16k_mono.wav"
```

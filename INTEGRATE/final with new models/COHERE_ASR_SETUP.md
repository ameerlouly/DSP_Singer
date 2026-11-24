# Cohere Arabic INT8 setup

The Transcription page uses `Qwen/Qwen3-ASR-1.7B-hf` for system A and `NAMAA-Space/cohere-transcribe-arabic-07-2026-int8` for system B. The default Cohere checkpoint revision is pinned in `cohere_asr.py`.

1. Start the application using `start_app.bat`. The launcher checks the project environment and installs the packages in `requirements.txt`.
2. Select the source and prepare eligible clips or unresolved spans. Open **Transcription** and start the job. The first run needs Internet access to download the model and processor files. Subsequent runs reuse the Hugging Face cache.
3. For a CUDA setup issue, run the diagnostic below from the project folder. The INT8 Cohere loader checks for CUDA before opening its checkpoint. The two recognizers are released between batches to limit GPU memory use.

```powershell
.\.venv\Scripts\python.exe setup_pytorch_cuda.py --check-only
```

When restoring a session from an older recognizer, the application can archive previous model results with their provenance. Named human corrections are retained. A newly pending engine can then produce its own transcript.

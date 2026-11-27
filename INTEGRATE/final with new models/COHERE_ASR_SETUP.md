# Cohere Arabic INT8 setup

The Transcription page uses `Qwen/Qwen3-ASR-1.7B-hf` for system A and `NAMAA-Space/cohere-transcribe-arabic-07-2026-int8` for system B. The default Cohere checkpoint revision is pinned in `cohere_asr.py`.

1. Start the application using `run_app.bat` for NVIDIA, or `run_intel_arc.bat` for Intel Arc. The launcher checks the project environment and installs the packages in `requirements.txt`.
2. Select the source and prepare eligible clips or unresolved spans. Open **Transcription** and start the job. The first run needs Internet access to download the model and processor files. Subsequent runs reuse the Hugging Face cache.
3. For a CUDA setup issue, run the diagnostic below from the project folder. The INT8 Cohere loader checks the selected CUDA or XPU backend before opening its checkpoint. The two recognizers are released between batches to limit GPU memory use.

```powershell
.\.venv\Scripts\python.exe setup_pytorch_cuda.py --check-only
```

When restoring a session from an older recognizer, the application can archive previous model results with their provenance. Named human corrections are retained. A newly pending engine can then produce its own transcript.

## Intel Arc integrated graphics (Windows 11)

1. Update the Intel Arc graphics driver using Intel or your laptop manufacturer. Supported Core Ultra Arc graphics require a compatible driver and XPU wheel; see https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-11.html .
2. Run `run_intel_arc.bat`. It creates `.venv-intel`, installs the project's pinned PyTorch and torchaudio from the XPU wheel index and installs the other existing requirements. The original `.venv` is not modified. The first installation needs internet and disk space.
3. Optional diagnostic (PowerShell, after installing):

```powershell
.\.venv-intel\Scripts\python.exe setup_pytorch_cuda.py --backend xpu --check-only --check-int8
```

The real INT8 preflight exercises a small bitsandbytes linear layer on XPU; it also runs before Cohere loads. The transcription progress displays `xpu:0` and the GPU name. Cohere retains the same pinned Arabic INT8 checkpoint, FP32 normalization state and FP16 autocast. It does not silently fall back to CPU or a full-precision checkpoint. Qwen's device policy is unchanged (CUDA if available, otherwise CPU). Other processing and output formats are unchanged. Intel integrated graphics uses shared system RAM, so close other memory-intensive apps if memory is insufficient. Hardware-specific model inference is not guaranteed by a preflight or unit test; test one short clip before a full run. The model card documents CUDA, not this exact integrated-GPU configuration. See https://huggingface.co/docs/transformers/quantization/bitsandbytes and https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html .

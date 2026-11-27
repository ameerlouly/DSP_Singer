"""Transcribe one prepared clip with the Arabic INT8 Cohere model.

load_cohere_model prepares the checkpoint and its normalization layers.
transcribe_cohere checks the waveform and looks for an incomplete decode before
returning the text. release_asr_model frees the model between runs.
"""

from __future__ import annotations

import gc
import os
import sys
from contextlib import nullcontext
from typing import Any, Callable

DEFAULT_COHERE_MODEL = 'NAMAA-Space/cohere-transcribe-arabic-07-2026-int8'
DEFAULT_COHERE_REVISION = '88a45a10a7afb54aeeeb9d7a50f88334886f5e74'
COHERE_ENGINE = 'cohere_arabic_int8'
MAX_NEW_TOKENS = 256


def gpu_available(torch: Any, backend: str) -> bool:
    api = getattr(torch, backend, None)
    return bool(api is not None and api.is_available())


def select_cohere_device(torch: Any) -> str:
    preference = os.environ.get('SPEECH_CORPUS_GPU', 'auto').lower()
    if preference not in ('auto', 'cuda', 'xpu'):
        raise ValueError('SPEECH_CORPUS_GPU must be auto, cuda or xpu.')
    choices = ('cuda', 'xpu') if preference == 'auto' else (preference,)
    for kind in choices:
        if gpu_available(torch, kind):
            if kind == 'cuda' and torch.cuda.get_device_capability(0) < (7, 5):
                if preference == 'auto':
                    continue
                raise RuntimeError('Cohere LLM.int8 needs NVIDIA compute capability 7.5 or newer.')
            return kind + ':0'
    raise RuntimeError('Cohere INT8 needs CUDA PyTorch or Intel XPU PyTorch with a supported GPU. '
                       'For Intel Arc, update the graphics driver and use run_intel_arc.bat. '
                       'No CPU/full-precision fallback was used.')


def clear_gpu_cache(torch: Any, kind: str) -> None:
    if gpu_available(torch, kind):
        getattr(torch, kind).empty_cache()


def oom_error(torch: Any, kind: str):
    # Modern torch exposes a shared exception; older builds may expose a backend alias.
    return getattr(torch, 'OutOfMemoryError', getattr(getattr(torch, kind, None), 'OutOfMemoryError', ()))


def check_xpu_int8(torch: Any, bnb: Any, device: str = 'xpu:0') -> None:
    """Run a tiny real INT8 operation before opening multi-GB model files."""
    try:
        with torch.inference_mode():
            layer = bnb.nn.Linear8bitLt(64, 64, bias=False, has_fp16_weights=False).to(device)
            x = torch.ones((2, 64), device=device, dtype=torch.float16)
            y = layer(x)
            if y.device.type != 'xpu' or not bool(torch.isfinite(y).all().item()):
                raise RuntimeError('INT8 smoke test produced invalid results or left XPU.')
            torch.xpu.synchronize()
    except Exception as exc:
        raise RuntimeError(f'Intel Arc INT8 preflight failed: {type(exc).__name__}: {exc}. '
                           'Update the Intel graphics driver; check the XPU PyTorch and bitsandbytes installation. '
                           'No CPU fallback was used.') from exc


def stabilize_normalization(network: Any) -> int:
    """Keep norm weights AND running statistics in FP32, without touching INT8.

    The saved checkpoint retains BF16 BatchNorm parameters/buffers. GPU AMP
    can send FP32 activations to those layers; autocast alone does not repair
    the stored state and BatchNorm then rejects the mixed dtypes.
    """
    import torch
    nn = torch.nn
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
                  nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                  nn.LayerNorm, nn.GroupNorm)
    if hasattr(nn, 'RMSNorm'):
        norm_types += (nn.RMSNorm,)
    count = 0
    for _name, module in network.named_modules():
        if isinstance(module, norm_types):
            # Module.float also converts running_mean/running_var while leaving
            # integer counters intact. Never call float()/half() on the model:
            # quantized Linear8bitLt weights and scales must stay untouched.
            module.float()
            count += 1
    return count


# Check the rate, length and samples before passing the clip to Cohere.
def validate_audio(audio: Any, sample_rate: int):
    import numpy as np
    samples = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16000 or samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError('Cohere needs finite 16 kHz mono audio; resample the audio instead of changing its declared rate.')
    if not 5 * sample_rate <= len(samples) <= 15 * sample_rate:
        raise ValueError('PDF Step 6 uses the approved 5–15 second clips from Part 4.')
    return samples


def load_cohere_model(model_name: str, progress: Callable[[str], None]) -> dict[str, Any]:
    """Load one INT8 Arabic model on one CUDA or Intel XPU GPU; no full-precision fallback."""
    try:
        import torch
        import bitsandbytes
        from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, CohereAsrForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError('Cohere dependencies are missing. Close the app and run run_intel_arc.bat for Intel or run_app.bat for NVIDIA to update requirements.') from exc
    device = select_cohere_device(torch)
    kind = device.split(':', 1)[0]
    api = getattr(torch, kind)
    if kind == 'xpu':
        progress(f'Checking Intel Arc INT8 kernels on {api.get_device_name(0)}...')
        check_xpu_int8(torch, bitsandbytes, device)
        clear_gpu_cache(torch, kind)
    revision = DEFAULT_COHERE_REVISION if model_name == DEFAULT_COHERE_MODEL else None
    options = {'revision': revision} if revision else {}
    config = AutoConfig.from_pretrained(model_name, **options)
    if config.model_type != 'cohere_asr':
        raise ValueError('System B must be a Cohere ASR checkpoint. An old MMS name cannot be used with the Cohere loader.')
    if 'ar' not in getattr(config, 'supported_languages', ['ar']):
        raise ValueError('The selected Cohere checkpoint does not declare Arabic support.')
    saved_quant = getattr(config, 'quantization_config', None)
    if saved_quant:
        if saved_quant.get('quant_method') != 'bitsandbytes' or not saved_quant.get('load_in_8bit', saved_quant.get('_load_in_8bit', False)):
            raise ValueError('Use the recommended bitsandbytes INT8 checkpoint for GPU memory efficiency.')
        quant_options = {}  # Respect the saved pre-quantized checkpoint config.
    else:
        quant_options = {'quantization_config': BitsAndBytesConfig(load_in_8bit=True)}
    memory = ''
    try:
        free, total = api.mem_get_info(0)
        memory = f': {free/1024**3:.1f} / {total/1024**3:.1f} GiB free (driver estimate)'
    except (AttributeError, RuntimeError, NotImplementedError):
        pass
    progress(f'Cohere INT8 on {api.get_device_name(0)} [{device}]{memory}. One clip at a time.')
    progress('Loading general Arabic Cohere; first use downloads the checkpoint (approximately 2.35 GB) and processor files.')
    processor = AutoProcessor.from_pretrained(model_name, **options)
    try:
        network = CohereAsrForConditionalGeneration.from_pretrained(
            model_name, device_map={'': 0 if kind == 'cuda' else device}, dtype=torch.float16,
            low_cpu_mem_usage=True, **options, **quant_options,
        ).eval()
        norm_count = stabilize_normalization(network)
    except oom_error(torch, kind) as exc:
        clear_gpu_cache(torch, kind)
        raise RuntimeError(f'Not enough free {kind.upper()} GPU memory for Cohere INT8. Close other GPU apps and retry one 5–15 s clip. The app did not switch to a larger full-precision model.') from exc
    if not getattr(network, 'is_loaded_in_8bit', False):
        del network
        gc.collect(); clear_gpu_cache(torch, kind)
        raise RuntimeError('The checkpoint did not load in INT8; stopped to avoid unexpected GPU-memory use.')
    progress(f'Cohere precision ready: {norm_count} normalization layers use FP32; INT8 weights and FP16 {kind.upper()} autocasting are retained.')
    return dict(engine=COHERE_ENGINE, model_name=model_name, revision=revision,
                processor=processor, model=network, device=device, input_dtype=torch.float16)


def transcribe_cohere(bundle: dict[str, Any], audio: Any, sample_rate: int = 16000) -> tuple[str, None]:
    """Transcribe the complete approved clip in Arabic, independently of A."""
    samples = validate_audio(audio, sample_rate)
    import torch
    kind = str(bundle['device']).split(':', 1)[0]
    processor, network = bundle['processor'], bundle['model']
    inputs = processor(audio=samples, sampling_rate=16000, language='ar', return_tensors='pt')
    chunk_index = inputs.pop('audio_chunk_index', None)
    inputs = inputs.to(bundle['device'], dtype=bundle.get('input_dtype', torch.float16))
    try:
        # FP32 norm state was prepared by the loader. AMP handles convolution
        # and matrix operations while norms may compute in FP32.
        precision = torch.autocast(device_type=kind, dtype=torch.float16) if kind in ('cuda', 'xpu') else nullcontext()
        with torch.inference_mode(), precision:
            outputs = network.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
    except oom_error(torch, kind) as exc:
        clear_gpu_cache(torch, kind)
        raise RuntimeError('Cohere ran out of GPU memory for this clip. Close other GPU applications and retry; the clip was not marked completed.') from exc
    # A token-budget truncation must not silently become a completed transcript.
    eos = network.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    last_token = int(outputs[0, -1])
    if outputs.shape[-1] >= MAX_NEW_TOKENS and eos_ids and last_token not in eos_ids:
        raise RuntimeError('Cohere reached its output-token limit without an end token. Review the clip; the unfinished text was not accepted as a complete ASR result.')
    decode_options = {'skip_special_tokens': True}
    if chunk_index is not None:
        decode_options.update(audio_chunk_index=chunk_index, language='ar')
    text = processor.decode(outputs, **decode_options)
    if isinstance(text, list) and len(text) == 1:
        text = text[0]
    if not isinstance(text, str):
        raise RuntimeError('Cohere returned an unexpected transcript format for one clip.')
    # No invented confidence: unlike the old CTC backend, this returns no frame score.
    return text.strip(), None


def release_asr_model(bundle: Any) -> None:
    """Release the previous model before loading the other ASR engine."""
    if isinstance(bundle, dict):
        bundle.pop('model', None)
        bundle.pop('processor', None)
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None:
        for kind in ('cuda', 'xpu'):
            clear_gpu_cache(torch, kind)

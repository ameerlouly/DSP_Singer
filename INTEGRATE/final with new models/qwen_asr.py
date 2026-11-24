"""Load Qwen on first transcription and decode one prepared speech waveform.

Function guide
--------------
load_qwen_model
    Load the processor and recognizer on the available device with a compatible
    precision.
transcribe_qwen
    Generate the transcript and return only tokens produced for the audio.
"""
MODEL_ID = 'Qwen/Qwen3-ASR-1.7B-hf'


def load_qwen_model(model_name, progress, stop_event=None):
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    def check_stop():
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('Qwen loading stopped.')

    check_stop()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    if device != 'cpu':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    progress(f'Loading Qwen on {device}. First use downloads about 4.1 GB; later runs reuse the cache.')
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        check_stop()
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_name, device_map={'': device}, dtype=dtype).eval()
    except InterruptedError:
        raise
    except OSError as exc:
        raise RuntimeError(f'Could not load Qwen model files. Check internet access and free disk space, then retry. {exc}') from exc
    check_stop()
    return {'processor': processor, 'model': model}


def transcribe_qwen(bundle, audio):
    import torch
    processor = bundle['processor']
    model = bundle['model']
    with torch.inference_mode():
        inputs = processor.apply_transcription_request(audio=audio).to(model.device, model.dtype)
        output = model.generate(**inputs, do_sample=False, max_new_tokens=256)
        generated = output[:, inputs['input_ids'].shape[1]:]
        text = processor.decode(generated, return_format='transcription_only')[0]
    return text.strip(), None

"""Try the Cohere processor and decoder with a small randomly initialized model.

This checks whether the installed Transformers APIs work together without needing the
full recognition weights.
"""

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoProcessor,CohereAsrConfig,CohereAsrForConditionalGeneration
from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig
import cohere_asr as backend

torch.set_num_threads(2);torch.manual_seed(4)
processor=AutoProcessor.from_pretrained(backend.DEFAULT_COHERE_MODEL,revision=backend.DEFAULT_COHERE_REVISION)
wave=np.zeros(6*16000,dtype=np.float32)
features=processor(audio=wave,language='ar',sampling_rate=16000,return_tensors='pt')
assert features['input_features'].shape==(1,601,128)
assert features['decoder_input_ids'].dtype==torch.int64
encoder=ParakeetEncoderConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,
    num_key_value_heads=4,num_mel_bins=128,subsampling_conv_channels=8,subsampling_factor=8,conv_kernel_size=9,max_position_embeddings=1024)
config=CohereAsrConfig(encoder_config=encoder,vocab_size=16384,hidden_size=32,intermediate_size=64,
    num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4,head_dim=8,decoder_start_token_id=13764)
model=CohereAsrForConditionalGeneration(config).eval()
model.generation_config.forced_eos_token_id=3
text,confidence=backend.transcribe_cohere(dict(processor=processor,model=model,device='cpu',input_dtype=torch.float32),wave,16000)
assert isinstance(text,str) and confidence is None
print('PASS: real Cohere processor Arabic prompt, PCM feature shapes, integer token preservation, tiny-model generation and batch text decoding. No accuracy/GPU-fit claim.')

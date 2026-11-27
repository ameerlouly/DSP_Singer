"""Device routing and launcher regression tests. Mock models; no real GPU required."""
import contextlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cohere_asr as backend
import setup_pytorch_cuda as setup
import test_cohere_backend as fixtures


def xpu_torch(cuda=False, xpu=True):
    torch = fixtures.fake_torch(cuda)
    torch.xpu = NS(is_available=Mock(return_value=xpu),get_device_name=Mock(return_value='Intel Arc integrated'),
                   mem_get_info=Mock(return_value=(6*1024**3,8*1024**3)),empty_cache=Mock(),synchronize=Mock())
    return torch


class IntelRoutingTests(unittest.TestCase):
    def test_auto_selects_xpu_if_no_cuda(self):
        with patch.dict(os.environ,{'SPEECH_CORPUS_GPU':'auto'}):
            self.assertEqual(backend.select_cohere_device(xpu_torch()),'xpu:0')
            self.assertEqual(backend.select_cohere_device(xpu_torch(cuda=True)),'cuda:0')

    def test_explicit_intel_no_silent_cuda_fallback(self):
        with patch.dict(os.environ,{'SPEECH_CORPUS_GPU':'xpu'}):
            self.assertEqual(backend.select_cohere_device(xpu_torch(cuda=True)),'xpu:0')
            with self.assertRaisesRegex(RuntimeError,'No CPU/full-precision fallback'):
                backend.select_cohere_device(xpu_torch(cuda=True,xpu=False))

    def test_same_pinned_model_loaded_on_xpu(self):
        _,lib=fixtures.CohereLoadingTests().libraries()
        torch=xpu_torch();bnb=NS();messages=[]
        with patch.dict(os.environ,{'SPEECH_CORPUS_GPU':'xpu'}),patch.dict(sys.modules,{'torch':torch,'transformers':lib,'bitsandbytes':bnb}),patch.object(backend,'check_xpu_int8') as check,patch.object(backend,'stabilize_normalization',return_value=3):
            result=backend.load_cohere_model(backend.DEFAULT_COHERE_MODEL,messages.append)
        check.assert_called_once_with(torch,bnb,'xpu:0')
        kwargs=lib.CohereAsrForConditionalGeneration.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs['device_map'],{'':'xpu:0'})
        self.assertEqual(kwargs['revision'],backend.DEFAULT_COHERE_REVISION)
        self.assertNotIn('quantization_config',kwargs)
        self.assertEqual(result['device'],'xpu:0');self.assertEqual(result['input_dtype'],'float16')
        self.assertTrue(any('xpu:0' in text for text in messages))

    def test_failed_preflight_does_not_download_model(self):
        _,lib=fixtures.CohereLoadingTests().libraries()
        with patch.dict(os.environ,{'SPEECH_CORPUS_GPU':'xpu'}),patch.dict(sys.modules,{'torch':xpu_torch(),'transformers':lib,'bitsandbytes':NS()}),patch.object(backend,'check_xpu_int8',side_effect=RuntimeError('preflight failed')):
            with self.assertRaisesRegex(RuntimeError,'preflight failed'):
                backend.load_cohere_model(backend.DEFAULT_COHERE_MODEL,lambda _:None)
        lib.AutoConfig.from_pretrained.assert_not_called()

    def test_xpu_decode_and_release(self):
        bundle,batch,processor=fixtures.CohereDecodingTests().fixture()
        bundle['device']='xpu:0';torch=xpu_torch()
        with patch.dict(sys.modules,{'torch':torch}):
            text,confidence=backend.transcribe_cohere(bundle,np.zeros(96000),16000)
            backend.release_asr_model(bundle)
        self.assertEqual(batch.moved,('xpu:0',{'dtype':'float16'}))
        self.assertEqual(processor.call_args.kwargs['language'],'ar')
        torch.autocast.assert_called_once_with(device_type='xpu',dtype='float16')
        torch.xpu.empty_cache.assert_called_once();torch.cuda.empty_cache.assert_not_called()
        self.assertNotIn('model',bundle);self.assertTrue(text);self.assertIsNone(confidence)

    def test_preflight_checks_output_on_xpu(self):
        torch=xpu_torch();torch.ones=Mock(return_value=object());torch.isfinite=Mock(return_value=NS(all=lambda:NS(item=lambda:True)))
        output=NS(device=NS(type='xpu'));layer=Mock(return_value=output);layer.to.return_value=layer
        bnb=NS(nn=NS(Linear8bitLt=Mock(return_value=layer)))
        backend.check_xpu_int8(torch,bnb)
        layer.to.assert_called_once_with('xpu:0');torch.xpu.synchronize.assert_called_once()
        output.device.type='cpu'
        with self.assertRaisesRegex(RuntimeError,'preflight failed'):
            backend.check_xpu_int8(torch,bnb)


class IntelSetupTests(unittest.TestCase):
    def test_installs_xpu_instead_of_cuda_on_explicit_request(self):
        probes=[dict(installed=True,available=False,version='2.11.0',audio_version='2.11.0'),dict(xpu_available=True,xpu_name='Arc')]
        with patch.object(setup,'probe_runtime',side_effect=probes),patch.object(setup,'nvidia_present',return_value=True),patch.object(setup.subprocess,'run',return_value=NS(returncode=0)) as install:
            self.assertEqual(setup.main(['--backend','xpu']),0)
        args=install.call_args.args[0];self.assertIn(setup.XPU_INDEX,args);self.assertNotIn(setup.CUDA_INDEX,args)
        self.assertIn('torch=='+setup.TORCH_VERSION,args);self.assertIn('torchaudio=='+setup.TORCH_VERSION,args)

    def test_ready_and_broken_xpu_are_not_reinstalled(self):
        for available,expected in [(True,0),(False,1)]:
            runtime=dict(installed=True,version='2.11.0+xpu',audio_version='2.11.0+xpu',xpu_build='2026',xpu_available=available)
            with patch.object(setup,'probe_runtime',return_value=runtime),patch.object(setup.subprocess,'run') as install:
                self.assertEqual(setup.main(['--backend','xpu']),expected)
                install.assert_not_called()

    def test_intel_launcher_uses_separate_environment_for_all_paths(self):
        root=Path(__file__).resolve().parents[1]
        intel=(root/'run_intel_arc.bat').read_text();normal=(root/'run_app.bat').read_text()
        self.assertIn('SPEECH_CORPUS_VENV=.venv-intel',intel);self.assertIn('SPEECH_CORPUS_GPU=xpu',intel)
        self.assertIn('call "%~dp0run_app.bat"',intel)
        self.assertIn('%SPEECH_CORPUS_VENV%\\Scripts\\python.exe',normal)
        self.assertNotIn('.venv\\installed-requirements.txt',normal)
        self.assertEqual(normal.count('%SPEECH_CORPUS_VENV%\\installed-requirements.txt'),3)


if __name__=='__main__':unittest.main()

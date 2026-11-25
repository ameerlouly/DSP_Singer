"""Check how the Cohere backend loads, decodes and releases a clip using stand-in model libraries.

Other cases cover saved records from earlier engines and the Windows launcher's CUDA
setup decision.
"""

import contextlib
import copy
import io
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cohere_asr as backend
import transcript_agreement_and_io as asr
import corpus_review_and_export as corpus
import setup_pytorch_cuda as setup


# Provide just the CUDA and precision settings that the loader reads from torch.
def fake_torch(available=True):
    cuda=NS(is_available=Mock(return_value=available),get_device_capability=Mock(return_value=(8,6)),
            mem_get_info=Mock(return_value=(5*1024**3,6*1024**3)),
            get_device_name=Mock(return_value='RTX 3060 Laptop'),empty_cache=Mock(),
            OutOfMemoryError=type('FakeOOM',(RuntimeError,),{}))
    return NS(cuda=cuda,float16='float16',inference_mode=contextlib.nullcontext,
              autocast=Mock(side_effect=lambda **_:contextlib.nullcontext()))


class CohereLoadingTests(unittest.TestCase):
    # Set up the normalization stub for this case and remove it after the test.
    def setUp(self):
        normalization = patch.object(backend, "stabilize_normalization", return_value=0)
        normalization.start()
        self.addCleanup(normalization.stop)

    # Supply fake model libraries so each test can choose its device and checkpoint settings.
    def libraries(self,available=True,model_type='cohere_asr',quantized=True):
        torch=fake_torch(available)
        config=NS(model_type=model_type,supported_languages=['ar','en'],quantization_config={'quant_method':'bitsandbytes','load_in_8bit':True})
        net=NS(is_loaded_in_8bit=quantized);net.eval=Mock(return_value=net)
        transformer=NS(AutoConfig=NS(from_pretrained=Mock(return_value=config)),
                       AutoProcessor=NS(from_pretrained=Mock(return_value=object())),
                       BitsAndBytesConfig=Mock(),
                       CohereAsrForConditionalGeneration=NS(from_pretrained=Mock(return_value=net)))
        return torch,transformer

    # Inspect the load request to catch changes to the model, revision or device.
    def test_pinned_arabic_int8_checkpoint_is_loaded_on_one_cuda_device(self):
        torch,lib=self.libraries();log=[]
        with patch.dict(sys.modules,{'torch':torch,'transformers':lib,'bitsandbytes':NS()}):
            bundle=backend.load_cohere_model(backend.DEFAULT_COHERE_MODEL,log.append)
        kwargs=lib.CohereAsrForConditionalGeneration.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs['device_map'],{'':0});self.assertEqual(kwargs['dtype'],'float16')
        self.assertEqual(kwargs['revision'],backend.DEFAULT_COHERE_REVISION)
        self.assertNotIn('quantization_config',kwargs) # Keep saved INT8 metadata.
        self.assertEqual(bundle['engine'],backend.COHERE_ENGINE)
        self.assertEqual(bundle['device'],'cuda:0');self.assertTrue(log)

    # A machine without CUDA should reach the error path before starting a download.
    def test_no_cuda_stops_before_download_instead_of_full_precision_fallback(self):
        torch,lib=self.libraries(available=False)
        with patch.dict(sys.modules,{'torch':torch,'transformers':lib,'bitsandbytes':NS()}):
            with self.assertRaisesRegex(RuntimeError,'CUDA PyTorch'):backend.load_cohere_model(backend.DEFAULT_COHERE_MODEL,lambda _:None)
        lib.AutoConfig.from_pretrained.assert_not_called()

    # Try an old model name and a load that fails to return the expected quantized weights.
    def test_mms_name_and_failed_int8_load_are_rejected(self):
        torch,lib=self.libraries(model_type='wav2vec2')
        with patch.dict(sys.modules,{'torch':torch,'transformers':lib,'bitsandbytes':NS()}):
            with self.assertRaisesRegex(ValueError,'Cohere ASR checkpoint'):backend.load_cohere_model('facebook/mms-1b-all',lambda _:None)
        lib.CohereAsrForConditionalGeneration.from_pretrained.assert_not_called()
        torch,lib=self.libraries(quantized=False)
        with patch.dict(sys.modules,{'torch':torch,'transformers':lib,'bitsandbytes':NS()}):
            with self.assertRaisesRegex(RuntimeError,'did not load in INT8'):backend.load_cohere_model(backend.DEFAULT_COHERE_MODEL,lambda _:None)


class CohereDecodingTests(unittest.TestCase):
    # Build a predictable processor and token sequence for one decoding call.
    def fixture(self,decoded='أهلاً بكم',outputs=None):
        class Batch(dict):
            # Remember where the decoder asked to move the test inputs.
            def to(self,device,**kwargs):self.moved=(device,kwargs);return self
        batch=Batch(input_features='audio-features')
        processor=NS(decode=Mock(return_value=decoded))
        processor_call=Mock(return_value=batch)
        class Processor:
            decode=processor.decode
            # Send the inputs to the mock processor so the test can inspect them.
            def __call__(self,**kwargs):return processor_call(**kwargs)
        model=NS(generate=Mock(return_value=np.array([[4,7,3]]) if outputs is None else outputs),generation_config=NS(eos_token_id=3))
        return dict(processor=Processor(),model=model,device='cuda:0',input_dtype='float16'),batch,processor_call

    # Look at what reaches the decoder, including the clip and requested language.
    def test_full_clip_is_forced_to_arabic(self):
        bundle,batch,processor=self.fixture(decoded=['أهلاً بكم'])
        wave=np.zeros(6*16000,dtype=np.float32)
        torch=fake_torch()
        with patch.dict(sys.modules,{'torch':torch}):text,confidence=backend.transcribe_cohere(bundle,wave,16000)
        self.assertEqual(text,'أهلاً بكم');self.assertIsNone(confidence)
        args=processor.call_args.kwargs
        self.assertEqual(args['language'],'ar');self.assertEqual(args['sampling_rate'],16000)
        self.assertTrue(np.array_equal(args['audio'],wave));self.assertNotIn('text',args)
        self.assertEqual(batch.moved,('cuda:0',{'dtype':'float16'}))
        self.assertEqual(bundle['model'].generate.call_args.kwargs['num_beams'],1)
        torch.autocast.assert_called_once_with(device_type='cuda',dtype='float16')

    # Try sample rates, channel shapes, values and lengths that the decoder must reject.
    def test_wrong_rate_stereo_nonfinite_and_invalid_duration_are_rejected(self):
        for wave,rate in [(np.zeros(6*8000),8000),(np.zeros((96000,2)),16000),(np.full(96000,np.nan),16000),(np.zeros(4*16000),16000),(np.zeros(16*16000),16000)]:
            with self.subTest(rate=rate,shape=wave.shape):
                with self.assertRaises(ValueError):backend.validate_audio(wave,rate)

    # A full token budget without an ending token can mean the transcript was cut off.
    def test_missing_end_token_is_not_a_completed_transcript(self):
        bundle,_,_=self.fixture(outputs=np.ones((1,257),dtype=int))
        with patch.dict(sys.modules,{'torch':fake_torch()}):
            with self.assertRaisesRegex(RuntimeError,'token limit'):backend.transcribe_cohere(bundle,np.zeros(96000),16000)

    # After release, neither bundle should keep references to heavy model objects.
    def test_release_clears_both_engine_bundles(self):
        torch = fake_torch()
        with patch.dict(sys.modules, {'torch': torch}):
            for engine in ('qwen', 'cohere'):
                with self.subTest(engine=engine):
                    bundle = dict(model=object(), processor=object())
                    backend.release_asr_model(bundle)
                    self.assertEqual(bundle, {})
        self.assertEqual(torch.cuda.empty_cache.call_count, 2)


class EngineMigrationTests(unittest.TestCase):
    # Build a saved transcription record with the model and review history needed for
    # migration.
    def record(self,human=False):
        r=dict(segment_id='old',label='speech',models={'a':'large-v3','b':'facebook/mms-1b-all'},
               asr_a_done=True,asr_b_done=True,transcript_a='أهلاً بكم',transcript_b='اهلا بكم',
               confidence_b=.8,human_override=False,status='accepted')
        corpus.compare(r)
        if human:r.update(human_override=True,final_transcript='نص اختاره المراجع',selected_transcript='نص اختاره المراجع')
        return r

    # Opening an older session should preserve the retired engine's text for reference.
    def test_legacy_mms_text_is_archived_and_only_b_needs_rerun(self):
        r=self.record(True);human=r['final_transcript']
        self.assertEqual(asr.migrate_second_engine([r],backend.DEFAULT_COHERE_MODEL),1);corpus.compare(r)
        self.assertTrue(r['asr_a_done']);self.assertFalse(r['asr_b_done'])
        self.assertEqual(r['transcript_a'],'أهلاً بكم');self.assertEqual(r['transcript_b'],'')
        self.assertEqual(r['legacy_asr_results'][0]['model'],'facebook/mms-1b-all')
        self.assertEqual(r['legacy_asr_results'][0]['transcript_b'],'اهلا بكم')
        self.assertEqual(r['final_transcript'],human);self.assertIsNone(r['cer'])
        self.assertEqual(asr.migrate_second_engine([r],backend.DEFAULT_COHERE_MODEL),0)
        self.assertEqual(len(r['legacy_asr_results']),1)

    # Recheck the automatic choice while keeping any valid current-model output.
    def test_automatic_acceptance_is_invalidated_but_current_cohere_is_kept(self):
        r=self.record();asr.migrate_second_engine([r],backend.DEFAULT_COHERE_MODEL);corpus.compare(r)
        self.assertEqual(r['status'],'needs_review');self.assertEqual(r['final_transcript'],'')
        r.update(asr_b_done=True,asr_b_engine=backend.COHERE_ENGINE,transcript_b='اهلا بكم')
        r['models']['b']=backend.DEFAULT_COHERE_MODEL;before=copy.deepcopy(r)
        self.assertEqual(asr.migrate_second_engine([r],backend.DEFAULT_COHERE_MODEL),0)
        self.assertEqual(r,before)


class CUDASetupTests(unittest.TestCase):
    # Supply the hardware checks and capture the install decision without changing the
    # machine.
    def run_setup(self,probes,present=True):
        with patch.object(setup,'probe_runtime',side_effect=probes),patch.object(setup,'nvidia_present',return_value=present),patch.object(setup.subprocess,'run',return_value=NS(returncode=0)) as install,contextlib.redirect_stdout(io.StringIO()):
            result=setup.main([])
        return result,install

    # With NVIDIA present, a CPU-only install should be repaired in the project's interpreter.
    def test_cpu_torch_on_nvidia_is_repaired_in_the_project_python(self):
        result,install=self.run_setup([dict(installed=True,available=False,cuda_build=None),dict(available=True,name='RTX 3060')])
        self.assertEqual(result,0);command=install.call_args.args[0]
        self.assertEqual(command[0],sys.executable)
        self.assertIn('torch=='+setup.TORCH_VERSION,command)
        self.assertIn('torchaudio=='+setup.TORCH_VERSION,command)
        self.assertIn(setup.CUDA_INDEX,command)

    # A working CUDA installation should pass the probe as it is.
    def test_ready_cuda_is_not_reinstalled(self):
        result,install=self.run_setup([dict(available=True,name='RTX 3060',version=setup.TORCH_VERSION+'+cu126',audio_version=setup.TORCH_VERSION+'+cu126')])
        self.assertEqual(result,0);install.assert_not_called()

    # A missing GPU or a broken driver should not cause repeated installation attempts.
    def test_no_gpu_and_broken_cuda_driver_do_not_trigger_reinstall_loops(self):
        result,install=self.run_setup([dict(installed=True,available=False,cuda_build=None)],present=False)
        self.assertEqual(result,0);install.assert_not_called()
        result,install=self.run_setup([dict(installed=True,available=False,cuda_build='12.6')])
        self.assertEqual(result,1);install.assert_not_called()


if __name__=='__main__':unittest.main()

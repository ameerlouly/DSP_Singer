"""Check which PyTorch runtime the project Python can use.

probe_runtime tests it in a separate process so Windows can release the loaded
libraries. main decides whether a CUDA or XPU PyTorch install is needed.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TORCH_VERSION = '2.11.0'
XPU_INDEX = 'https://download.pytorch.org/whl/xpu'
CUDA_INDEX = 'https://download.pytorch.org/whl/cu126'
PROBE = '''import json, importlib.metadata
try:
 import torch
 available=torch.cuda.is_available()
 xpu=getattr(torch,'xpu',None)
 xpu_available=bool(xpu is not None and xpu.is_available())
 print(json.dumps(dict(installed=True,version=torch.__version__,cuda_build=torch.version.cuda,
   xpu_build=getattr(torch.version,'xpu',None),available=available,xpu_available=xpu_available,
   audio_version=importlib.metadata.version("torchaudio"),
   name=torch.cuda.get_device_name(0) if available else None,
   xpu_name=xpu.get_device_name(0) if xpu_available else None)))
except Exception as exc:
 print(json.dumps(dict(installed=False,available=False,error=f'{type(exc).__name__}: {exc}')))
'''


# Ask a separate Python process what this PyTorch installation can actually run.
def probe_runtime(python=None):
    result = subprocess.run([python or sys.executable, '-c', PROBE], capture_output=True, text=True, timeout=90)
    if result.returncode:
        return dict(installed=False, available=False, error=result.stderr[-1500:])
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return dict(installed=False, available=False, error='PyTorch probe returned no readable result.')


# Check whether the NVIDIA utility can see a device on this machine.
def nvidia_present():
    executable = shutil.which('nvidia-smi')
    if not executable:
        return False
    result = subprocess.run([executable, '--query-gpu=name', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=20)
    return result.returncode == 0 and bool(result.stdout.strip())


# Use the probe results to decide whether the project's PyTorch install needs changing.
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='Check without downloading/installing packages.')
    parser.add_argument('--backend', choices=('auto','cuda','xpu'),
                        default=os.environ.get('SPEECH_CORPUS_GPU','auto'),
                        help='Select NVIDIA CUDA or Intel Arc XPU explicitly.')
    parser.add_argument('--check-int8', action='store_true',
                        help='Check Intel INT8 kernels without downloading model weights.')
    args = parser.parse_args(argv)
    runtime = probe_runtime()
    paired = runtime.get('version','').split('+')[0] == TORCH_VERSION and runtime.get('audio_version','').split('+')[0] == TORCH_VERSION
    backend = args.backend
    if backend == 'auto':
        if runtime.get('available') or nvidia_present():
            backend = 'cuda'
        elif runtime.get('xpu_available') or runtime.get('xpu_build') or '+xpu' in runtime.get('version',''):
            backend = 'xpu'
        else:
            if args.check_only or args.check_int8:
                print('No usable CUDA or XPU PyTorch runtime. Use run_intel_arc.bat on Intel Arc.', flush=True)
                return 1
            print('No NVIDIA GPU or configured Intel XPU runtime detected. For Intel Arc, use run_intel_arc.bat.', flush=True)
            return 0
    flag = 'available' if backend == 'cuda' else 'xpu_available'
    name_key = 'name' if backend == 'cuda' else 'xpu_name'
    build = runtime.get('cuda_build') if backend == 'cuda' else (runtime.get('xpu_build') or '+xpu' in runtime.get('version',''))
    title = 'CUDA' if backend == 'cuda' else 'Intel XPU'
    if runtime.get(flag) and paired:
        print(title+' PyTorch ready: '+str(runtime.get(name_key) or backend), flush=True)
        if args.check_int8:
            if backend != 'xpu':
                print('--check-int8 is for Intel XPU; use --backend xpu.', flush=True)
                return 1
            try:
                import torch, bitsandbytes
                from cohere_asr import check_xpu_int8
                check_xpu_int8(torch,bitsandbytes)
            except Exception as exc:
                print(str(exc), flush=True)
                return 1
            print('Intel Arc INT8 kernel test passed (no model downloaded).', flush=True)
        return 0
    if args.check_only or args.check_int8:
        print(title+' PyTorch is not ready. '+runtime.get('error',''), flush=True)
        return 1
    if runtime.get('installed') and build and not runtime.get(flag):
        print(title+' PyTorch is installed but cannot access the GPU. Update the graphics driver and retry.\n'+runtime.get('error',''), flush=True)
        return 1
    index = CUDA_INDEX if backend == 'cuda' else XPU_INDEX
    print('Installing '+title+' PyTorch and matching torchaudio into the selected environment. First setup downloads large wheels.', flush=True)
    command = [sys.executable, '-m', 'pip', 'install', '--force-reinstall',
               'torch=='+TORCH_VERSION, 'torchaudio=='+TORCH_VERSION, '--index-url', index]
    result = subprocess.run(command)
    if result.returncode:
        print(title+' dependency installation failed; see the pip output.', flush=True)
        return result.returncode
    runtime = probe_runtime()
    if not runtime.get(flag):
        print(title+' is still unavailable after installation. Check your graphics driver.\n'+runtime.get('error',''), flush=True)
        return 1
    print(title+' PyTorch ready: '+str(runtime.get(name_key) or backend), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.TimeoutExpired) as exc:
        print('GPU setup could not finish: '+str(exc), file=sys.stderr)
        raise SystemExit(1)

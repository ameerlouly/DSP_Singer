"""Prepare the project's Windows PyTorch environment for NVIDIA inference.

Function guide
--------------
probe_runtime
    Check installed Torch and torchaudio in an isolated child process.
nvidia_present
    Query the driver for an available NVIDIA device.
main
    Check the runtime and install the matching CUDA wheels when setup requires it.

Torch is probed in a child so Windows DLLs are not held open during pip repair.
No NVIDIA driver or system-wide package is installed here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

TORCH_VERSION = '2.11.0'
CUDA_INDEX = 'https://download.pytorch.org/whl/cu126'
PROBE = '''import json, importlib.metadata
try:
 import torch
 available=torch.cuda.is_available()
 print(json.dumps(dict(installed=True,version=torch.__version__,cuda_build=torch.version.cuda,available=available,audio_version=importlib.metadata.version("torchaudio"),
   name=torch.cuda.get_device_name(0) if available else None)))
except Exception as exc:
 print(json.dumps(dict(installed=False,available=False,error=str(exc))))
'''


def probe_runtime(python=None):
    result = subprocess.run([python or sys.executable, '-c', PROBE], capture_output=True, text=True, timeout=90)
    if result.returncode:
        return dict(installed=False, available=False, error=result.stderr[-1500:])
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return dict(installed=False, available=False, error='PyTorch probe returned no readable result.')


def nvidia_present():
    executable = shutil.which('nvidia-smi')
    if not executable:
        return False
    result = subprocess.run([executable, '--query-gpu=name', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=20)
    return result.returncode == 0 and bool(result.stdout.strip())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='Check without downloading/installing packages.')
    args = parser.parse_args(argv)
    runtime = probe_runtime()
    paired = runtime.get('version','').split('+')[0] == TORCH_VERSION and runtime.get('audio_version','').split('+')[0] == TORCH_VERSION
    if runtime.get('available') and paired:
        print('CUDA PyTorch ready: '+runtime.get('name','NVIDIA GPU'), flush=True)
        return 0
    if args.check_only:
        print('CUDA PyTorch is not ready. '+runtime.get('error',''), flush=True)
        return 1
    if not nvidia_present():
        print('No NVIDIA GPU detected. The app can open; Cohere INT8 needs an NVIDIA CUDA GPU when ASR is used.', flush=True)
        return 0
    if runtime.get('installed') and runtime.get('cuda_build') and not runtime.get('available'):
        print('CUDA PyTorch is installed but cannot access the NVIDIA GPU. Check the NVIDIA driver/device and retry.\n'+runtime.get('error',''), flush=True)
        return 1
    print('Installing CUDA 12.6 PyTorch and matching torchaudio into the project environment. First setup downloads large wheels.', flush=True)
    command = [sys.executable, '-m', 'pip', 'install', '--force-reinstall',
               'torch=='+TORCH_VERSION, 'torchaudio=='+TORCH_VERSION, '--index-url', CUDA_INDEX]
    result = subprocess.run(command)
    if result.returncode:
        print('CUDA dependency installation failed; see the pip output. Run the launcher again after resolving the error.', flush=True)
        return result.returncode
    runtime = probe_runtime()
    if not runtime.get('available'):
        print('CUDA is still unavailable after installation. Check your NVIDIA driver.\n'+runtime.get('error',''), flush=True)
        return 1
    print('CUDA PyTorch ready: '+runtime.get('name','NVIDIA GPU'), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.TimeoutExpired) as exc:
        print('GPU setup could not finish: '+str(exc), file=sys.stderr)
        raise SystemExit(1)

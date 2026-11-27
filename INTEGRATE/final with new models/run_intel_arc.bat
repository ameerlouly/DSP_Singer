@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Separate Intel environment: do not replace the existing NVIDIA/CPU venv.
set "SPEECH_CORPUS_VENV=.venv-intel"
set "SPEECH_CORPUS_GPU=xpu"
call "%~dp0run_app.bat"
exit /b %errorlevel%

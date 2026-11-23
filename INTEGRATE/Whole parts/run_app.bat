@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Creating the Python 3.11 environment...
    where /q py.exe
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3.11 -m venv .venv
    )
    if errorlevel 1 goto :environment_failed
)
if not exist "%VENV_PY%" goto :environment_failed
if not exist ".venv\installed-requirements.txt" goto :install
fc /b requirements.txt .venv\installed-requirements.txt >nul 2>&1
if errorlevel 1 goto :install
goto :launch
:install
echo Installing dependencies. The first installation can take several minutes...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
copy /y requirements.txt .venv\installed-requirements.txt >nul
if errorlevel 1 goto :failed
:launch
rem Optional local FFmpeg installation: place both executables in tools\ffmpeg\bin.
if exist "tools\ffmpeg\bin\ffmpeg.exe" set "PATH=%CD%\tools\ffmpeg\bin;%PATH%"
where /q ffmpeg.exe
if errorlevel 1 goto :ffmpeg_failed
where /q ffprobe.exe
if errorlevel 1 goto :ffmpeg_failed
"%VENV_PY%" qt_app.py
if errorlevel 1 goto :failed
popd
exit /b 0
:environment_failed
echo Could not create the environment. Install Python 3.11 or create .venv manually.
goto :failed
:ffmpeg_failed
echo Put ffmpeg.exe and ffprobe.exe in tools\ffmpeg\bin, or add their installed folder to Windows PATH.
:failed
popd
pause
exit /b 1

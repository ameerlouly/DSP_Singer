@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Set this to your team's PyQt entry point when integrating the app.
set "APP_ENTRY=app.py"

pushd "%~dp0"
if errorlevel 1 (
    echo Could not open the launcher folder.
    pause
    exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"
set "REQUIREMENTS=requirements.txt"
set "INSTALLED_REQUIREMENTS=.venv\installed-requirements.txt"

if not exist "%APP_ENTRY%" (
    echo Cannot find "%APP_ENTRY%". Put this launcher beside the app or update APP_ENTRY.
    goto :failure
)
if not exist "%REQUIREMENTS%" (
    echo Cannot find requirements.txt. Put it beside the launcher.
    goto :failure
)

if not exist "%VENV_PY%" (
    echo Creating the project Python environment...
    where /q py.exe
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -m venv .venv
    )
    if errorlevel 1 goto :venv_failure
    if not exist "%VENV_PY%" goto :venv_failure
    goto :install
)

if not exist "%INSTALLED_REQUIREMENTS%" goto :install
fc /b "%REQUIREMENTS%" "%INSTALLED_REQUIREMENTS%" >nul 2>&1
if errorlevel 1 goto :install
goto :launch

:install
echo Installing project dependencies...
"%VENV_PY%" -m pip install -r "%REQUIREMENTS%"
if errorlevel 1 (
    echo Dependency installation failed. See the pip message above.
    goto :failure
)
copy /y "%REQUIREMENTS%" "%INSTALLED_REQUIREMENTS%" >nul
if errorlevel 1 (
    echo Dependencies installed, but the install state could not be saved.
    goto :failure
)

:launch
where /q ffmpeg.exe
if errorlevel 1 goto :ffmpeg_failure
where /q ffprobe.exe
if errorlevel 1 goto :ffmpeg_failure

"%VENV_PY%" "%APP_ENTRY%"
if errorlevel 1 (
    echo The app exited with an error. See the message above.
    goto :failure
)
popd
exit /b 0

:venv_failure
echo Could not create .venv. Install Python with the venv module and try again.
goto :failure

:ffmpeg_failure
echo Add the installed FFmpeg bin folder containing ffmpeg.exe and ffprobe.exe to Windows PATH.
echo Then close this window and run the launcher again.

:failure
popd
pause
exit /b 1

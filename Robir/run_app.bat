@echo off
setlocal
cd /d "%~dp0"
title Speech Corpus App

if not exist "app.py" (
    echo ERROR: app.py was not found in this folder.
    echo Extract all files from the ZIP into the same folder.
    goto :failed
)

if exist ".venv\Scripts\python.exe" goto :environment_ready

echo Creating the Python environment. This is only needed once...
where py >nul 2>&1
if errorlevel 1 goto :try_python
py -3 -c "import sys" >nul 2>&1
if errorlevel 1 goto :try_python
py -3 -m venv .venv
if errorlevel 1 goto :venv_failed
goto :environment_ready

:try_python
where python >nul 2>&1
if errorlevel 1 goto :python_missing
python -c "import sys" >nul 2>&1
if errorlevel 1 goto :python_missing
python -m venv .venv
if errorlevel 1 goto :venv_failed

:environment_ready
".venv\Scripts\python.exe" -c "import tkinter, numpy, scipy, matplotlib, soundfile, sounddevice, torch, importlib.util; assert importlib.util.find_spec('panns_inference') and importlib.util.find_spec('silero_vad')" >nul 2>&1
if not errorlevel 1 goto :launch

echo Checking and installing required packages. Internet may be needed...
".venv\Scripts\python.exe" -m pip install numpy scipy matplotlib soundfile sounddevice torch panns-inference silero-vad
if errorlevel 1 goto :install_failed
".venv\Scripts\python.exe" -c "import tkinter, numpy, scipy, matplotlib, soundfile, sounddevice, torch, importlib.util; assert importlib.util.find_spec('panns_inference') and importlib.util.find_spec('silero_vad')" >nul 2>&1
if errorlevel 1 goto :import_failed

:launch
echo Starting the application...
".venv\Scripts\python.exe" "app.py"
if errorlevel 1 goto :program_failed
exit /b 0

:python_missing
echo ERROR: Python 3 could not be started.
echo Install Python 3 from https://www.python.org/downloads/windows/
echo During installation, select "Add Python to PATH".
goto :failed

:venv_failed
echo ERROR: Could not create the .venv environment.
echo Check your Python installation and folder permissions.
goto :failed

:install_failed
echo ERROR: Package installation failed. Check the messages above and your internet connection.
goto :failed

:import_failed
echo ERROR: A required package or tkinter could not be imported.
echo Check that your Python installation includes Tcl/Tk and IDLE.
goto :failed

:program_failed
echo ERROR: The program stopped. Check the error message above.
goto :failed

:failed
echo.
pause
exit /b 1





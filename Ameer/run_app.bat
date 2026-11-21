@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0app.py"
) else (
    py -3 "%~dp0app.py"
)
if errorlevel 1 (
    echo.
    echo The app could not start. Install dependencies using:
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
)

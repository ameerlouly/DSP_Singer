@echo off
cd /d "%~dp0"
"%~dp0..\Ameer\myenv\Scripts\python.exe" app.py
if errorlevel 1 pause

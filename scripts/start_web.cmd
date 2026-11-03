@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Python is not installed in .venv. Run scripts\init_windows.cmd first.
    exit /b 1
)
set "PYTHONPATH=src"
set "PYTHONIOENCODING=utf-8"
".venv\Scripts\python.exe" scripts\daily_picks.py --saved-only
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" scripts\serve_daily.py

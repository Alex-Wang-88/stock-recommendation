@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\streamlit.exe" (
    echo Streamlit is not installed in .venv.
    echo Run: .venv\Scripts\python.exe -m pip install -e ".[web,free-data]"
    exit /b 1
)
set "PYTHONPATH=src"
".venv\Scripts\streamlit.exe" run app.py

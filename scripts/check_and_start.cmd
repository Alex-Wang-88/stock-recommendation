@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo Local environment is missing. Running initialization...
    call "%~dp0init_windows.cmd"
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Streamlit is missing. Running initialization...
    call "%~dp0init_windows.cmd"
    if errorlevel 1 exit /b 1
)

call "%~dp0start_web.cmd"
exit /b %errorlevel%

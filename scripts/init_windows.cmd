@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

echo [1/3] Checking Python 3.11 or newer...
set "PYTHON_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.11"
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install Python 3.11+ from https://www.python.org/downloads/windows/
        exit /b 1
    )
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo Python 3.11 or newer is required.
        python --version
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

echo [2/3] Creating or checking the local virtual environment...
if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv.
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo The local virtual environment is incomplete.
    exit /b 1
)

echo [3/3] Installing project dependencies...
".venv\Scripts\python.exe" -m pip install -e ".[web,free-data]"
if errorlevel 1 (
    echo Dependency installation failed. Check the network and try again.
    exit /b 1
)

echo.
echo Initialization completed.
echo Start the app with: scripts\start_web.cmd
exit /b 0

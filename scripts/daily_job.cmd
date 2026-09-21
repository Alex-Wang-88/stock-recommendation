@echo off
REM Daily archive job for the stock-recommendation project.
REM
REM Run this AFTER the market close (and after upstream publishes daily data).
REM Running it in the morning silently gets the PREVIOUS day's theme attribution
REM (the "rollback snapshot" trap) - the job guards against that, but you would
REM waste a run.
REM
REM Designed to be driven by Windows Task Scheduler without any agent in the loop.
REM ASCII-only on purpose: .cmd files with non-ASCII content get garbled on Windows.

setlocal
REM Force ONE encoding for the whole file. Without this, cmd.exe writes its own
REM lines in the OEM codepage (GBK here) while Python 3.13 writes UTF-8, and the
REM log ends up with two encodings mixed - unreadable by any single decoder.
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PROJECT=%~dp0.."
cd /d "%PROJECT%"

set "PYTHONPATH=%PROJECT%\src"
REM Single append-only log. Do NOT build the name from %DATE%: its format is
REM locale-dependent ("2026/09/19", "09/19/2026", "Sat 09/19/2026") and the
REM substring offsets silently produce garbage names on some locales.
set "LOG=%PROJECT%\logs\daily-job.log"

if not exist "%PROJECT%\logs" mkdir "%PROJECT%\logs"

echo. >> "%LOG%"
echo ==== run at %DATE% %TIME% ==== >> "%LOG%"

"%PROJECT%\.venv\Scripts\python.exe" -m recommend.cli daily-job --preset low-position >> "%LOG%" 2>&1
set "CODE=%ERRORLEVEL%"

echo exit=%CODE% >> "%LOG%"
echo exit=%CODE%
endlocal & exit /b %CODE%

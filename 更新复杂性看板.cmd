@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment missing: .venv\Scripts\python.exe
  pause
  popd
  exit /b 1
)
".venv\Scripts\python.exe" scripts\build_complexity_snapshot.py
if errorlevel 1 (
  echo Snapshot refresh failed. See the source status on the page.
) else (
  echo Snapshot updated. Reload site\complexity.html to view it.
)
pause
popd
endlocal

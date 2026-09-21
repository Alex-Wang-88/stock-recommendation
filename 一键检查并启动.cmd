@echo off
setlocal
cd /d "%~dp0"
call "%~dp0scripts\check_and_start.cmd"
exit /b %errorlevel%

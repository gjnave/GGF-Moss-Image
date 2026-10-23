@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
  echo Follow the manual setup in README.md or use the member quick installer.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u app.py %*
if errorlevel 1 pause

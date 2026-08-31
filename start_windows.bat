@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo TrafficVision AI is not set up yet.
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
start "" "http://localhost:8000"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

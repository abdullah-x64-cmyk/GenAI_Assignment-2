@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python 3 is required. Install Python 3.10 or newer, then double-click this file again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv
if errorlevel 1 (
  echo Could not create the application environment.
  pause
  exit /b 1
)
echo Installing or updating the application dependencies. This can take several minutes the first time.
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-app.txt
if errorlevel 1 (
  echo Dependency installation failed. Check your internet connection and try again.
  pause
  exit /b 1
)
echo Starting TrialLens. Your browser will open automatically.
".venv\Scripts\python.exe" -m src.web_app
pause

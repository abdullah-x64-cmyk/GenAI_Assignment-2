@echo off
setlocal
cd /d "%~dp0"
rem Some Windows installs have a working python.exe but an unregistered py launcher.
set "TRIAL_PYTHON="
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if not errorlevel 1 set "TRIAL_PYTHON=python"
if not defined TRIAL_PYTHON (
  py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
  if not errorlevel 1 set "TRIAL_PYTHON=py -3"
)
if not defined TRIAL_PYTHON (
  echo A working Python 3.10 or newer is required. Install Python with Add to PATH enabled, then try again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  %TRIAL_PYTHON% -m venv .venv
  if errorlevel 1 (
    echo Could not create the application environment.
    pause
    exit /b 1
  )
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

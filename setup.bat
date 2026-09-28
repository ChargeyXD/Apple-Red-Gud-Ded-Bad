@echo off
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during install, then run setup.bat again.
  pause
  exit /b 1
)
echo Creating virtual environment...
%PY% -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 ( echo Install failed. & pause & exit /b 1 )
echo.
echo Setup done. Next: put apple_model.onnx and model_info.json in the model folder, then run run_checker.bat
pause

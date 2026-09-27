@echo off
REM ============================================================
REM  setup.bat - One-time setup for Windows
REM  Creates a virtual environment, installs dependencies, builds
REM  the feature dataset, and trains the XGBoost model.
REM ============================================================

echo ===============================================
echo  Hybrid AI Traffic Incident Detection System
echo  Setup (Windows)
echo ===============================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Install Python 3.10-3.12 from https://www.python.org/downloads/
    echo and make sure to check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

echo.
echo [1/5] Creating virtual environment (.venv) ...
python -m venv .venv
if errorlevel 1 goto :error

echo.
echo [2/5] Activating virtual environment ...
call .venv\Scripts\activate.bat
if errorlevel 1 goto :error

echo.
echo [3/5] Installing Python dependencies (this can take a few minutes) ...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [4/5] Building the feature dataset from the Mobile Century data ...
cd backend\ml
python preprocess.py
if errorlevel 1 goto :error_ml

echo.
echo [5/5] Training the XGBoost incident-severity model ...
python train_model.py
if errorlevel 1 goto :error_ml
cd ..\..

echo.
echo ===============================================
echo  Setup complete!
echo  Next steps:
echo    1. Run start_backend.bat   (in a new window)
echo    2. Run start_dashboard.bat (in another new window)
echo    3. Open frontend\index.html in your browser (optional)
echo  Or just run run_all.bat to do all three at once.
echo ===============================================
pause
exit /b 0

:error_ml
cd ..\..
:error
echo.
echo [ERROR] Setup failed. See the message above for details.
pause
exit /b 1

@echo off
REM ============================================================
REM  start_dashboard.bat - Starts the Streamlit live dashboard
REM  NOTE: make sure start_backend.bat is already running in
REM  another window before (or after) launching this.
REM ============================================================
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting dashboard. It will open automatically in your browser at
echo http://localhost:8501
echo Press CTRL+C to stop.
streamlit run dashboard\app.py
pause

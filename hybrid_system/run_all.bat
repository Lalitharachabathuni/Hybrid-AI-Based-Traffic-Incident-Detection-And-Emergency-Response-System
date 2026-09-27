@echo off
REM ============================================================
REM  run_all.bat - Launches backend + dashboard + frontend
REM  Opens the backend and dashboard each in their own window so
REM  you can see their logs, and opens the frontend in your
REM  default browser.
REM ============================================================
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting backend (FastAPI) in a new window ...
start "Traffic Incident System - Backend (FastAPI)" cmd /k "call .venv\Scripts\activate.bat && cd backend && uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 4 /nobreak >nul

echo Starting dashboard (Streamlit) in a new window ...
start "Traffic Incident System - Dashboard (Streamlit)" cmd /k "call .venv\Scripts\activate.bat && streamlit run dashboard\app.py"

timeout /t 4 /nobreak >nul

echo Opening the React + Tailwind dashboard ...
start "" "http://127.0.0.1:8000/app/index.html"

timeout /t 1 /nobreak >nul

echo Opening the citizen accident reporting page ...
start "" "http://127.0.0.1:8000/app/report.html"

echo.
echo All components are starting up in separate windows:
echo   1. Backend        -  http://127.0.0.1:8000/docs
echo   2. Dashboard       -  http://127.0.0.1:8000/app/index.html
echo   3. Report a crash  -  http://127.0.0.1:8000/app/report.html
echo   4. Streamlit dash  -  http://localhost:8501
echo.
echo Close each window (or press CTRL+C inside it) to stop that component.
pause

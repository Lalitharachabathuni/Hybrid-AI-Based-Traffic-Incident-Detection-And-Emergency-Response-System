@echo off
REM ============================================================
REM  start_backend.bat - Starts the FastAPI backend
REM ============================================================
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

cd backend
echo Starting backend at http://127.0.0.1:8000  (API docs: http://127.0.0.1:8000/docs)
echo Press CTRL+C to stop.
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
cd ..
pause

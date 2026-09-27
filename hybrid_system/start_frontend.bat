@echo off
REM ============================================================
REM  start_frontend.bat - Opens the React + Tailwind web UI
REM  This is a static page (no build step needed); it just needs
REM  the backend (start_backend.bat) running at 127.0.0.1:8000.
REM ============================================================
start "" "http://127.0.0.1:8000/app/index.html"

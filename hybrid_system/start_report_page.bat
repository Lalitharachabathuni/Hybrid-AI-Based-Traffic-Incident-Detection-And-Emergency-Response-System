@echo off
REM ============================================================
REM  start_report_page.bat - Opens the public citizen accident
REM  reporting page (photo + GPS upload for rural areas).
REM  NOTE: make sure start_backend.bat is already running in
REM  another window first -- this page needs it at 127.0.0.1:8000.
REM ============================================================
start "" "http://127.0.0.1:8000/app/report.html"

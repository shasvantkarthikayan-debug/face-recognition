@echo off
REM Face Recognition System - Quick Start
echo Starting Face Recognition System...
echo.
echo Opening browser to http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.

REM Check if virtual environment exists and activate it
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Start the application
python app.py

pause

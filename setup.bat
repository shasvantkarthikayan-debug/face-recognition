@echo off
REM Face Recognition System - Quick Setup
REM This batch file runs the PowerShell setup script

echo ========================================
echo Face Recognition System - Quick Setup
echo ========================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell not found!
    echo Please run: python scripts\setup_environment.py
    pause
    exit /b 1
)

echo Running setup script...
echo.

REM Run PowerShell script
powershell -ExecutionPolicy Bypass -File "scripts\setup_environment.ps1"

echo.
echo Setup script completed.
pause

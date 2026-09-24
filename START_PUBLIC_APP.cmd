@echo off
title Cazette Dashboard - Public access
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_public_app.ps1"
if errorlevel 1 (
    echo.
    echo The dashboard could not be started. See the error above.
    pause
)

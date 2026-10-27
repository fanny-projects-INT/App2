@echo off
title Cazette Dashboard
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\start-public-app.ps1"
if errorlevel 1 (
    echo.
    echo The dashboard could not be started. See logs\streamlit.err.log and logs\cloudflared.log.
    pause
)

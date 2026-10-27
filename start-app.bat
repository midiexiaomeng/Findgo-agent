@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM Normal launch: no console window
if not exist "logs" mkdir logs

where pythonw >nul 2>nul
if errorlevel 1 (
    start "" python desktop_app.py
) else (
    start "" pythonw desktop_app.py
)

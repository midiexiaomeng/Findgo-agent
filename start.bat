@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)

if errorlevel 1 (
    echo.
    echo [ERROR] Startup failed. Run scripts\setup.ps1 first.
    pause
)

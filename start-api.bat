@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "FINDGO_CONFIG=%~dp0config.api.yaml"

if not exist ".venv-api\Scripts\pythonw.exe" (
    echo .venv-api not found. Run deploy-api.bat first.
    pause
    exit /b 1
)

start "" ".venv-api\Scripts\pythonw.exe" desktop.py --mode edge

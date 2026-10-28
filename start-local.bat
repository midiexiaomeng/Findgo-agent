@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "FINDGO_CONFIG=%~dp0config.local.yaml"

if not exist ".venv-local\Scripts\pythonw.exe" (
    echo .venv-local not found. Run deploy-local.bat first.
    pause
    exit /b 1
)

REM portable Ollama: start it if present and not already listening
if exist "%~dp0ollama\ollama.exe" (
    netstat -ano | findstr ":11434" | findstr "LISTENING" >nul
    if errorlevel 1 (
        start "" /min "%~dp0ollama\ollama.exe" serve
    )
)

start "" ".venv-local\Scripts\pythonw.exe" desktop.py --mode edge

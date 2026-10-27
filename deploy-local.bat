@echo off
chcp 65001 >nul
cd /d "%~dp0"
title imgsearch deploy - local model

echo.
echo ============================================
echo  imgsearch deployment: LOCAL MODEL
echo  Creates .venv-local, downloads CLIP/BLIP,
echo  sets up Ollama and pulls Qwen.
echo  First run may take 10-30 minutes.
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found. Install Python 3.9+ and add it to PATH.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\deploy-local.ps1" %*

echo.
echo ============================================
if errorlevel 1 (
    echo Deployment failed. Scroll up for details.
) else (
    echo Deployment finished. Double-click start-local.bat
)
echo ============================================
pause

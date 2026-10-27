@echo off
chcp 65001 >nul
cd /d "%~dp0"
title imgsearch deploy - API mode

echo.
echo ============================================
echo  imgsearch deployment: API MODE
echo  Creates .venv-api and downloads CLIP/BLIP.
echo  The LLM reranker calls a cloud API.
echo  CLIP and BLIP still run on this machine.
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found. Install Python 3.9+ and add it to PATH.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\deploy-api.ps1" %*

echo.
echo ============================================
if errorlevel 1 (
    echo Deployment failed. Scroll up for details.
) else (
    echo Deployment finished. Double-click start-api.bat
)
echo ============================================
pause

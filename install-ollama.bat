@echo off
chcp 65001 >nul
cd /d "%~dp0"
title findgo - portable ollama

echo.
echo ============================================
echo  Portable Ollama installer
echo  Installs to: %~dp0ollama
echo  Models in  : %~dp0ollama\models
echo  Downloads about 1.1GB on first run.
echo ============================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-ollama-portable.ps1" %*

echo.
echo ============================================
if errorlevel 1 (
    echo Install failed. Scroll up for details.
) else (
    echo Done. Start the app with start-local.bat
)
echo ============================================
pause

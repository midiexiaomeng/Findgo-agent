@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 正常启动：无控制台窗口
if not exist "logs" mkdir logs

where pythonw >nul 2>nul
if errorlevel 1 (
    start "" python desktop_app.py
) else (
    start "" pythonw desktop_app.py
)

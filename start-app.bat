@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 双击启动原生桌面窗口（无控制台、无端口）
where pythonw >nul 2>nul
if errorlevel 1 (
    start "" python desktop_app.py
) else (
    start "" pythonw desktop_app.py
)

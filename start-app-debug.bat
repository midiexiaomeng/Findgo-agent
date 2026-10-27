@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 调试启动：保留控制台，便于看报错和日志
echo ============================================
echo  imgsearch Agent - debug console
echo  Logs are also written to logs\app.log
echo ============================================
echo.

python desktop_app.py

echo.
echo Process exited with code %errorlevel%
pause

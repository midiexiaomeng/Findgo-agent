@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM Debug launch: keep console open for errors and logs
echo ============================================
echo  Findgo - debug console
echo  Logs are also written to logs\app.log
echo ============================================
echo.

python desktop_app.py

echo.
echo Process exited with code %errorlevel%
pause

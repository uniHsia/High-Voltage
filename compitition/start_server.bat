@echo off
chcp 65001 >nul
echo Starting Django server with UTF-8 support...
cd /d "%~dp0"
python run_with_utf8.py
pause
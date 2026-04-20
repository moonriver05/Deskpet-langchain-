@echo off
REM 这是注释
chcp 65001 >nul
cd /d "%~dp0"
call  .\venv\Scripts\activate.bat
python pet.py
pause


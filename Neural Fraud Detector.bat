@echo off
REM Neural Fraud Detector v3 - Double-click to run
REM Place this file next to launch.py and double-click to run

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv and run
echo Starting Neural Fraud Detector v3...
call venv\Scripts\activate.bat
python launch.py

REM Keep window open
pause
@echo off
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python is required. Download from https://python.org
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

echo.
echo Starting AI Label Studio at http://localhost:5000
echo Press Ctrl+C to stop.
echo.
python app.py

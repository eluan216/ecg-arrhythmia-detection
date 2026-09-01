@echo off
REM Setup script for ECG Arrhythmia Detection on Windows

echo.
echo ================================================================================
echo ECG ARRHYTHMIA DETECTION - WINDOWS SETUP
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Creating virtual environment...
if not exist venv (
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install -e ".[dev]" -q

if errorlevel 1 (
    echo Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SETUP COMPLETE
echo ================================================================================
echo.
echo To continue, run one of the following:
echo.
echo   1. Run project setup validation:
echo      python setup_project.py
echo.
echo   2. Start the API server:
echo      uvicorn api.main:app --reload
echo.
echo   3. View MLflow experiments:
echo      mlflow ui
echo.
echo   4. Run tests:
echo      pytest tests/ -v
echo.
echo   5. Train the baseline model (requires data):
echo      python -m src.train.train_baseline
echo.
echo Virtual environment is still active. To deactivate it, run:
echo   deactivate
echo.
pause

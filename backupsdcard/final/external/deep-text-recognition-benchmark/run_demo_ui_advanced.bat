@echo off
echo ========================================
echo Advanced OCR Demo UI with Text Detection
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Installing required packages...
pip install easyocr opencv-python-headless pillow torch torchvision

echo.
echo Starting Advanced OCR Demo UI...
python demo_ui_advanced.py

pause

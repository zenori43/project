@echo off
echo ========================================
echo CRAFT Line Detection Tool
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
pip install opencv-python-headless pillow torch torchvision numpy

echo.
echo Starting CRAFT Line Detection Tool...
python craft_line_detection.py

pause

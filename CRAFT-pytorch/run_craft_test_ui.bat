@echo off
title CRAFT Test UI - Three Lines Detection

echo Installing required packages...
pip install opencv-python-headless pillow torch torchvision numpy scikit-image

echo.
echo Starting CRAFT Test UI...
python craft_test_ui.py

pause

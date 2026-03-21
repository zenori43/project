@echo off
cd /d "%~dp0.."
echo Starting Cap Text Fade Generator...
python "program label\cap_text_fade_ui.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Program exited with code %ERRORLEVEL%
    pause
)

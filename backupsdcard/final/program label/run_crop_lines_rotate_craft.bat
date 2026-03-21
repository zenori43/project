@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo Usage: run_crop_lines_rotate_craft.bat ^<input_image_or_folder^> ^<output_folder^>
  pause
  exit /b 1
)

if "%~2"=="" (
  echo Usage: run_crop_lines_rotate_craft.bat ^<input_image_or_folder^> ^<output_folder^>
  pause
  exit /b 1
)

python "crop_lines_rotate_craft.py" --input "%~1" --output "%~2"
pause


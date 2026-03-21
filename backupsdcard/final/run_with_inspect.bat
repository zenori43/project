@echo off
REM รันแอปพร้อม PyQtInspect (Direct mode) - ใช้ดีบัก/ตรวจสอบ widget บน UI
REM ใช้ Python 3.10 (TensorFlow รองรับ)

cd /d "%~dp0"
echo ติดตั้ง PyQtInspect สำหรับ Python 3.10 (ถ้ายังไม่มี)...
py -3.10 -m pip install PyQtInspect -q
py -3.10 -m PyQtInspect --direct --file main.py %*
pause

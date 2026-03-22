@echo off
cd /d "%~dp0.."
REM ใช้ Python 3.11 โดยตรง (ถ้ามีหลายตัว) เพื่อให้ตรงกับที่ติดตั้ง pymodbus
py -3.11 tools\modbus_simulator.py
if errorlevel 1 python tools\modbus_simulator.py
pause

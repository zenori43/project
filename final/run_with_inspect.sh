#!/bin/bash
# รันแอปพร้อม PyQtInspect (Direct mode) - ใช้ดีบัก/ตรวจสอบ widget บน UI
# ต้องติดตั้งก่อน: pip install PyQtInspect

cd "$(dirname "$0")"
python -m PyQtInspect --direct --file main.py "$@"

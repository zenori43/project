#!/bin/bash
# -*- coding: utf-8 -*-
# Script สำหรับรันโปรแกรม Bottle Detection System
# ระบบตรวจจับขวดและ OCR - USB Camera + Modbus + Sentech Camera

# ตั้งค่าสีสำหรับ output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Bottle Detection System${NC}"
echo -e "${BLUE}  ระบบตรวจจับขวดและ OCR${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ตรวจสอบ Python version
echo -e "${YELLOW}[1/5] ตรวจสอบ Python version...${NC}"
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}❌ Python3 not found!${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}✅ Found: $PYTHON_VERSION${NC}"
echo ""

# ตรวจสอบว่า main.py มีอยู่
echo -e "${YELLOW}[2/5] ตรวจสอบไฟล์หลัก...${NC}"
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ main.py not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ main.py found${NC}"
echo ""

# ตรวจสอบโครงสร้างโฟลเดอร์
echo -e "${YELLOW}[3/5] ตรวจสอบโครงสร้างโปรเจกต์...${NC}"
REQUIRED_DIRS=("config" "core" "gui" "libs" "models" "external")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "${RED}❌ Directory not found: $dir${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✅ โครงสร้างโปรเจกต์ครบถ้วน${NC}"
echo ""

# ตรวจสอบโมเดล
echo -e "${YELLOW}[4/5] ตรวจสอบโมเดล...${NC}"
REQUIRED_MODELS=(
    "models/detection/cap.pt"
    "models/detection/bottle.pt"
    "models/rotation/text_rotation_model.pth"
    "models/ocr/fix10.pth"
)

MISSING_MODELS=0
for model in "${REQUIRED_MODELS[@]}"; do
    if [ ! -f "$model" ]; then
        echo -e "${RED}⚠️  Model not found: $model${NC}"
        MISSING_MODELS=$((MISSING_MODELS + 1))
    fi
done

if [ $MISSING_MODELS -eq 0 ]; then
    echo -e "${GREEN}✅ โมเดลทั้งหมดพร้อมใช้งาน${NC}"
else
    echo -e "${YELLOW}⚠️  พบโมเดลที่หายไป $MISSING_MODELS ไฟล์ (โปรแกรมอาจทำงานไม่เต็มประสิทธิภาพ)${NC}"
fi
echo ""

# ตรวจสอบ external dependencies
echo -e "${YELLOW}[5/5] ตรวจสอบ External Dependencies...${NC}"
if [ -L "external/CRAFT-pytorch" ] || [ -d "external/CRAFT-pytorch" ]; then
    echo -e "${GREEN}✅ CRAFT-pytorch found${NC}"
else
    echo -e "${RED}⚠️  CRAFT-pytorch not found (บางฟีเจอร์อาจไม่ทำงาน)${NC}"
fi

if [ -L "external/deep-text-recognition-benchmark" ] || [ -d "external/deep-text-recognition-benchmark" ]; then
    echo -e "${GREEN}✅ deep-text-recognition-benchmark found${NC}"
else
    echo -e "${RED}⚠️  deep-text-recognition-benchmark not found (OCR อาจไม่ทำงาน)${NC}"
fi
echo ""

# ตั้งค่า environment variables
export PYTHONUNBUFFERED=1
export QT_X11_NO_MITSHM=1

# ตั้งค่า CUDA environment variables ก่อนรันโปรแกรม
# เพื่อให้ PyTorch และ OpenCV detect CUDA ได้อย่างถูกต้อง
export CUDA_HOME=/usr/local/cuda-11.4
export CUDA_PATH=/usr/local/cuda-11.4

# เพิ่ม CUDA paths ไปยัง LD_LIBRARY_PATH
if [ -z "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/local/lib
else
    export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH
fi

# เพิ่ม OpenCV Python path ไปยัง PYTHONPATH
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=/usr/local/lib/python3.8/site-packages/
else
    export PYTHONPATH=/usr/local/lib/python3.8/site-packages/:$PYTHONPATH
fi

# เพิ่ม CUDA bin ไปยัง PATH
if [ -z "$PATH" ]; then
    export PATH=/usr/local/cuda-11.4/bin:$PATH
else
    export PATH=/usr/local/cuda-11.4/bin:$PATH
fi

# โหลด libgomp เฉพาะเมื่อยังใช้ TensorFlow CPU (tensorflow_cpu_aws) และไฟล์มีอยู่จริง
# ถ้าติดตั้ง TensorFlow GPU แล้ว ไฟล์นี้มักไม่มี — ไม่ set LD_PRELOAD (กัน error ใน log)
LIBGOMP="$HOME/.local/lib/python3.8/site-packages/tensorflow_cpu_aws.libs/libgomp-cc9055c7.so.1.0.0"
if [ -f "$LIBGOMP" ] && [ -r "$LIBGOMP" ]; then
    export LD_PRELOAD="$LIBGOMP${LD_PRELOAD:+:$LD_PRELOAD}"
else
    # ล้าง LD_PRELOAD ถ้าเคยชี้ไปที่ path นี้ (ไฟล์หายหรือใช้ TF GPU)
    case "${LD_PRELOAD-}" in
        *tensorflow_cpu_aws*libgomp*) export LD_PRELOAD= ;;
    esac
fi

echo -e "${GREEN}✅ CUDA environment variables set (OpenCV, PyTorch, TensorFlow ใช้ GPU)${NC}"
echo -e "${GREEN}   CUDA_HOME: $CUDA_HOME${NC}"
echo -e "${GREEN}   LD_LIBRARY_PATH includes: CUDA libraries${NC}"
echo -e "${GREEN}   Defect model (TensorFlow) จะใช้ GPU เมื่อรันผ่าน run.sh${NC}"
echo ""

# ตรวจสอบว่ามี arguments สำหรับ debug mode หรือไม่
DEBUG_MODE=false
if [ "$1" == "--debug" ] || [ "$1" == "-d" ]; then
    DEBUG_MODE=true
fi

# ตรวจสอบว่ามี terminal หรือไม่ (เมื่อรันจาก Desktop shortcut จะไม่มี terminal)
if [ -t 0 ] && [ -t 1 ]; then
    # มี terminal - แสดง output
    HAS_TERMINAL=true
else
    # ไม่มี terminal - redirect ไปยัง log file
    HAS_TERMINAL=false
fi

# รันโปรแกรม
LOG_FILE="app.log"

if [ "$DEBUG_MODE" = true ] && [ "$HAS_TERMINAL" = true ]; then
    # Debug mode + มี terminal - แสดง output ทั้งหมด
    echo -e "${BLUE}🐛 Debug mode enabled${NC}"
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  กำลังเริ่มโปรแกรม...${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    $PYTHON_CMD -u main.py "$@"
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ โปรแกรมปิดโดยปกติ${NC}"
    else
        echo -e "${RED}❌ โปรแกรมปิดด้วย error code: $EXIT_CODE${NC}"
    fi
elif [ "$HAS_TERMINAL" = true ]; then
    # Normal mode + มี terminal - แสดง output และ redirect ไปยัง log
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  กำลังเริ่มโปรแกรม...${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${YELLOW}📝 Log file: $LOG_FILE${NC}"
    echo ""
    $PYTHON_CMD -u main.py "$@" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ โปรแกรมปิดโดยปกติ${NC}"
    else
        echo -e "${RED}❌ โปรแกรมปิดด้วย error code: $EXIT_CODE${NC}"
    fi
else
    # ไม่มี terminal (รันจาก Desktop shortcut) - redirect ทั้งหมดไปยัง log file
    $PYTHON_CMD -u main.py "$@" >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?
fi

exit $EXIT_CODE


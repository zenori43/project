#!/bin/bash
# -*- coding: utf-8 -*-
# ติดตั้ง TensorFlow แบบ GPU บน Jetson (แทนตัว CPU จาก pip)
# ใช้ wheel จาก NVIDIA ที่รองรับ CUDA

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  ติดตั้ง TensorFlow GPU สำหรับ Jetson${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# ใช้ pip ของ user (--user) เพื่อไม่ต้องใช้ sudo
PYTHON="${PYTHON:-python3}"
PIP="$PYTHON -m pip"

# 1) ถอนการติดตั้ง TensorFlow / tf-keras แบบเดิม (CPU)
echo -e "${YELLOW}[1/4] ถอนการติดตั้ง TensorFlow (CPU) เดิม...${NC}"
$PIP uninstall -y tensorflow tf-keras 2>/dev/null || true
$PIP uninstall -y tensorflow-cpu-aws tensorflow-cpu 2>/dev/null || true
echo -e "${GREEN}   Done.${NC}"
echo ""

# 2) อัปเกรด pip และ dependencies ที่จำเป็น
echo -e "${YELLOW}[2/4] อัปเกรด pip และติดตั้ง dependencies...${NC}"
$PIP install --upgrade pip setuptools wheel
$PIP install numpy h5py pybind11
echo -e "${GREEN}   Done.${NC}"
echo ""

# 3) เลือก wheel ตาม JetPack (ลองตามลำดับ)
# v502 = JetPack 5.0.2, v512 = 5.1.2 | Python 3.8 = cp38
WHEEL_URL=""
# ลอง v502 ก่อน (มักมี 2.10.0)
url502="https://developer.download.nvidia.com/compute/redist/jp/v502/tensorflow/tensorflow-2.10.0+nv22.10-cp38-cp38-linux_aarch64.whl"
if curl -sI "$url502" | head -1 | grep -q 200; then
  WHEEL_URL="$url502"
  echo -e "${GREEN}   พบ wheel: JetPack v502 (TensorFlow 2.10.0)${NC}"
fi
if [ -z "$WHEEL_URL" ]; then
  for ver in v512 v51; do
    url="https://developer.download.nvidia.com/compute/redist/jp/${ver}/tensorflow/tensorflow-2.10.1+nv22.12-cp38-cp38-linux_aarch64.whl"
    if curl -sI "$url" | head -1 | grep -q 200; then
      WHEEL_URL="$url"
      echo -e "${GREEN}   พบ wheel: JetPack $ver${NC}"
      break
    fi
  done
fi

if [ -z "$WHEEL_URL" ]; then
  echo -e "${RED}ไม่พบ wheel อัตโนมัติ - ลองติดตั้งจาก NVIDIA ด้วยมือ:${NC}"
  echo "  https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html"
  echo "  หรือดาวน์โหลดจาก: https://developer.download.nvidia.com/compute/redist/jp/"
  echo ""
  echo "  ตัวอย่าง (แก้ JP version ตาม JetPack ของคุณ):"
  echo "  pip3 install https://developer.download.nvidia.com/compute/redist/jp/v512/tensorflow/tensorflow-2.10.1+nv22.12-cp38-cp38-linux_aarch64.whl"
  exit 1
fi

# 4) ดาวน์โหลด wheel ด้วย wget (timeout นาน + retry) แล้วค่อย pip install จากไฟล์
#    ไฟล์ ~460MB ถ้าใช้ pip ดาวน์โหลดตรงๆ มัก timeout
echo -e "${YELLOW}[3/4] ดาวน์โหลด TensorFlow GPU wheel จาก NVIDIA (~460MB)...${NC}"
WHEEL_NAME="${WHEEL_URL##*/}"
WHEEL_LOCAL="/tmp/$WHEEL_NAME"
if [ -f "$WHEEL_LOCAL" ]; then
  echo -e "${GREEN}   พบไฟล์ใน cache: $WHEEL_LOCAL (ข้ามการดาวน์โหลด)${NC}"
else
  if command -v wget &>/dev/null; then
    wget -c --timeout=3600 --tries=3 -q --show-progress -O "$WHEEL_LOCAL" "$WHEEL_URL" || {
      echo -e "${RED}   wget ล้มเหลว - ลองใช้ pip (timeout 30 นาที)...${NC}"
      rm -f "$WHEEL_LOCAL"
      WHEEL_LOCAL=""
    }
  else
    WHEEL_LOCAL=""
  fi
  if [ -z "$WHEEL_LOCAL" ] || [ ! -f "$WHEEL_LOCAL" ]; then
    echo "   ใช้ pip ดาวน์โหลด (--timeout 1800 วินาที)..."
    $PIP install --user --timeout 1800 "$WHEEL_URL"
    echo -e "${GREEN}   Done.${NC}"
    echo ""
    # ข้ามไปขั้นตอนตรวจสอบ
    WHEEL_LOCAL="SKIP"
  fi
fi
if [ -n "$WHEEL_LOCAL" ] && [ "$WHEEL_LOCAL" != "SKIP" ] && [ -f "$WHEEL_LOCAL" ]; then
  echo -e "${YELLOW}[4/4] ติดตั้งจากไฟล์ท้องถิ่น...${NC}"
  $PIP install --user "$WHEEL_LOCAL"
  echo -e "${GREEN}   Done.${NC}"
  echo ""
fi

# 5) ตั้งค่า CUDA environment (ให้ TensorFlow เห็น GPU เหมือนตอนรัน run.sh)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.4}"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-11.4}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH:-}"

# 6) ตรวจสอบด้วย test_cuda_complete.py (OpenCV, PyTorch, TensorFlow/Defect)
echo -e "${YELLOW}[4/4] ตรวจสอบ CUDA/GPU (OpenCV, PyTorch, TensorFlow Defect)...${NC}"
(cd "$SCRIPT_DIR" && $PYTHON test_cuda_complete.py) || true

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  เสร็จสิ้น${NC}"
echo -e "${GREEN}========================================${NC}"

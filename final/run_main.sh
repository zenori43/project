#!/bin/bash
# -*- coding: utf-8 -*-
# Wrapper script for running main.py directly
# Sets CUDA environment variables before running Python

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

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

# รัน main.py
python3 main.py "$@"


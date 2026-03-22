#!/bin/bash
# ตรวจสอบ TensorFlow + GPU (ใช้เมื่อติดตั้ง TensorFlow เสร็จแล้ว)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.4}"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-11.4}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH:-}"

python3 test_cuda_complete.py

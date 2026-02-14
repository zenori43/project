#!/bin/bash
# ตั้งค่า CUDA environment เหมือน run.sh
# ใช้: source set_cuda_env.sh   (จากโฟลเดอร์ final หรือ path ไปยัง final)
# หลัง source แล้วรัน python3 main.py หรือ python3 test_cuda_complete.py ได้

export CUDA_HOME=/usr/local/cuda-11.4
export CUDA_PATH=/usr/local/cuda-11.4

if [ -z "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/local/lib
else
    export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=/usr/local/lib/python3.8/site-packages/
else
    export PYTHONPATH=/usr/local/lib/python3.8/site-packages/:$PYTHONPATH
fi

if [[ ":$PATH:" != *":/usr/local/cuda-11.4/bin:"* ]]; then
    export PATH=/usr/local/cuda-11.4/bin:$PATH
fi

echo "✅ CUDA env set: CUDA_HOME=$CUDA_HOME"
echo "   Run: python3 main.py  or  python3 test_cuda_complete.py"

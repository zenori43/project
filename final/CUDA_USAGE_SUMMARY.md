# สรุปการใช้งาน CUDA ในโปรแกรม

## ✅ ส่วนที่ใช้ CUDA แล้ว

### 1. PyTorch CUDA
- ✅ `libs/processing/rotationCRAFT.py` - ใช้ `.cuda()` สำหรับ model และ data
- ✅ `libs/processing/craft_line_detection.py` - ใช้ `.cuda()` สำหรับ model และ data  
- ✅ `libs/processing/deep_ocr.py` - ใช้ `torch.device('cuda')`
- ✅ `libs/detection/bottledetect.py` - มี CUDA_AVAILABLE check

### 2. OpenCV CUDA Setup
- ✅ `core/cuda_setup.py` - Setup CUDA environment variables
- ✅ `main.py` - เรียก `cuda_setup` ก่อน import OpenCV

## ❌ ส่วนที่ยังไม่ใช้ CUDA (ควรแก้ไข)

### 1. OpenCV Image Processing
โปรแกรมใช้ OpenCV CPU functions หลายที่ ควรเปลี่ยนเป็น CUDA:

#### `core/image_processor.py`
- Line 102: `cv2.cvtColor()` → ควรใช้ `cv2.cuda.cvtColor()`
- Line 106: `cv2.cvtColor()` → ควรใช้ `cv2.cuda.cvtColor()`
- Line 119: `cv2.resize()` → ควรใช้ `cv2.cuda.resize()`

#### `gui/handlers/bottle_handlers.py`
- Line 250: `cv2.resize()` → ควรใช้ `cv2.cuda.resize()`
- Line 253: `cv2.cvtColor()` → ควรใช้ `cv2.cuda.cvtColor()`
- Line 308: `cv2.resize()` → ควรใช้ `cv2.cuda.resize()`
- Line 311: `cv2.cvtColor()` → ควรใช้ `cv2.cuda.cvtColor()`

#### `gui/main_window.py`
- Line 1189: `cv2.cvtColor()` → ควรใช้ `cv2.cuda.cvtColor()`
- Line 1200: `cv2.resize()` → ควรใช้ `cv2.cuda.resize()`
- Line 1203: `cv2.GaussianBlur()` → ควรใช้ `cv2.cuda.GaussianBlur()`
- Line 1216: `cv2.cvtColor()` → ควรใช้ `cv2.cuda.cvtColor()`

#### `gui/handlers/cap_handlers.py`
- Line 452: `cv2.cvtColor()` → ควรใช้ `cv2.cuda.cvtColor()`
- Line 455: `cv2.cvtColor()` → ควรใช้ `cv2.cuda.cvtColor()`

## 🔧 วิธีแก้ไข

### 1. ใช้ Helper Functions (แนะนำ)
ผมได้สร้าง `core/cuda_image_utils.py` ที่มี helper functions:
- `cuda_resize()` - Resize with CUDA fallback
- `cuda_cvtColor()` - Color conversion with CUDA fallback  
- `cuda_gaussianBlur()` - Gaussian blur with CUDA fallback
- `cuda_batch_operations()` - Batch operations with CUDA

**ตัวอย่างการใช้งาน:**
```python
from core.cuda_image_utils import cuda_resize, cuda_cvtColor, cuda_gaussianBlur

# แทนที่
image = cv2.resize(image, (320, 240))

# ด้วย
from core.cuda_image_utils import cuda_resize
image = cuda_resize(image, (320, 240))

# แทนที่
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ด้วย
from core.cuda_image_utils import cuda_cvtColor
rgb_image = cuda_cvtColor(image, cv2.COLOR_BGR2RGB)
```

### 2. ใช้ CUDA โดยตรง (Advanced)
```python
import cv2

# Check CUDA availability
if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Upload to GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    
    # Process on GPU
    gpu_resized = cv2.cuda.resize(gpu_img, (320, 240))
    gpu_rgb = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)
    
    # Download from GPU
    result = gpu_rgb.download()
else:
    # Fallback to CPU
    result = cv2.cvtColor(cv2.resize(image, (320, 240)), cv2.COLOR_BGR2RGB)
```

## 📊 ประสิทธิภาพ

### คาดหวัง
- **Resize**: เร็วขึ้น 2-5x
- **Color Conversion**: เร็วขึ้น 3-10x  
- **Gaussian Blur**: เร็วขึ้น 5-20x (ขึ้นกับ kernel size)

### ข้อจำกัด
- CUDA มี overhead สำหรับภาพเล็ก (< 100x100 pixels)
- ภาพเล็กอาจช้ากว่า CPU เนื่องจาก overhead
- ภาพใหญ่ (> 640x480) จะเห็นประโยชน์ชัดเจน

## ✅ สรุป

1. **PyTorch**: ใช้ CUDA แล้ว ✅
2. **OpenCV**: ยังไม่ใช้ CUDA ❌
3. **Helper Functions**: สร้างไว้แล้ว ✅ (`core/cuda_image_utils.py`)
4. **Next Step**: แก้ไขโค้ดให้ใช้ helper functions

## 🚀 ขั้นตอนต่อไป

1. Import `cuda_image_utils` ในไฟล์ที่ต้องการ
2. แทนที่ `cv2.resize()` → `cuda_resize()`
3. แทนที่ `cv2.cvtColor()` → `cuda_cvtColor()`
4. แทนที่ `cv2.GaussianBlur()` → `cuda_gaussianBlur()`
5. ทดสอบประสิทธิภาพ

---

## ⚠️ เคยใช้ CUDA ได้ แต่ตอนนี้ใช้ไม่ได้ — วิธีแก้

สาเหตุส่วนใหญ่คือ ** environment ไม่ถูก set** ตอนรัน (รันจาก IDE / ดับเบิลคลิก python โดยไม่ผ่าน `run.sh`)

### 1. รันผ่าน run.sh (แนะนำ)

```bash
cd /path/to/final
./run.sh
```

หรือ debug mode:

```bash
./run.sh --debug
```

`run.sh` จะ set `CUDA_HOME`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `PATH` ให้ PyTorch และ OpenCV เห็น CUDA

### 2. รันจาก Cursor/IDE ให้ CUDA ทำงาน

ต้องให้ shell ที่ใช้รันโปรแกรมมี env เดียวกับ `run.sh`:

**วิธีที่ 1: เปิด terminal แล้ว source ก่อนรัน**

```bash
cd /path/to/final
source set_cuda_env.sh   # สคริปต์ในโปรเจกต์ (สร้างไว้ให้แล้ว)
python3 main.py
```

**วิธีที่ 2: ใน Cursor ใช้ terminal ที่เคย source แล้ว**

- เปิด terminal ใน Cursor
- รัน `source final/set_cuda_env.sh` (หรือ `cd final && source set_cuda_env.sh`)
- จากนั้นรัน `python3 main.py` หรือกด Run ใน IDE (ถ้า IDE ใช้ shell เดียวกัน)

### 3. ตรวจสอบว่า CUDA ถูก set หรือยัง

```bash
cd final
source set_cuda_env.sh
python3 test_cuda_complete.py
```

ถ้าเห็น `PyTorch CUDA available: True` และ/หรือ `OpenCV CUDA devices > 0` แปลว่าพร้อมใช้

### 4. สิ่งที่ต้องมีบนเครื่อง (Jetson)

| สิ่งที่ต้องมี | หมายเหตุ |
|---------------|----------|
| `CUDA_HOME` / `LD_LIBRARY_PATH` | ต้องชี้ไปที่ cuda-11.4 และ tegra (Jetson) |
| PyTorch build รองรับ CUDA | ใช้เวอร์ชันที่ NVIDIA build สำหรับ Jetson |
| OpenCV build with CUDA | มักอยู่ที่ `/usr/local/lib/python3.8/site-packages/` และต้องมีใน `PYTHONPATH` |

ถ้ารันผ่าน `run.sh` แล้วยังใช้ CUDA ไม่ได้ ให้เช็ค:

- `ls /usr/local/cuda-11.4/lib64/libcudart*` ว่ามีไฟล์
- `python3 -c "import torch; print(torch.cuda.is_available())"` หลัง source env


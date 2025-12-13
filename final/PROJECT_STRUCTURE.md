# โครงสร้างโปรเจกต์ (Project Structure)

## 📁 โครงสร้างโฟลเดอร์

```
project_name/
├── config/                  # 1. ส่วนตั้งค่า (Configuration)
│   └── settings.py          # เก็บ paths, constants, configurations
│
├── core/                    # 2. ส่วนหลัก/ตรรกะทางธุรกิจ (Business Logic)
│   ├── __init__.py
│   ├── camera_manager.py    # จัดการกล้อง (USBCamera, SentechCamera)
│   ├── image_processor.py   # ฟังก์ชันประมวลผลภาพ, OCR, utilities
│   └── business_logic.py    # Thread classes และ business logic
│
├── gui/                     # 3. ส่วน GUI (View)
│   ├── __init__.py
│   └── main_window.py       # BottleDetectionGUI class - จัดการ UI
│
├── external/               # 4. External Dependencies (via symbolic links)
│   ├── README.md
│   ├── CRAFT-pytorch/       # CRAFT text detection library (symlink)
│   │   └── craft_mlt_25k.pth # CRAFT model file
│   │   └── craft_refiner_CTW1500.pth # CRAFT refiner model
│   └── deep-text-recognition-benchmark/  # Deep OCR library (symlink)
│       └── utils.py, model.py, dataset.py
│
├── models/                 # 5. Model Files
│   ├── README.md
│   ├── detection/           # Detection models (YOLO)
│   │   ├── bottle.pt       # Bottle detection model
│   │   └── cap.pt          # Cap detection model
│   ├── rotation/            # Rotation models
│   │   ├── text_rotation_model.pth  # AI text rotation model
│   │   └── best_accuracy.pth        # Best accuracy rotation model
│   └── ocr/                # OCR models
│       └── fix10.pth       # Deep OCR model
│
├── libs/                    # 6. External Libraries/Modules
│   ├── __init__.py
│   ├── camera/              # Camera modules
│   │   ├── __init__.py
│   │   └── usbcamra.py      # USB Camera management
│   │
│   ├── detection/           # Detection modules
│   │   ├── __init__.py
│   │   ├── bottledetect.py  # Bottle detection (YOLO)
│   │   └── capmodel.py      # Cap detection (YOLO)
│   │
│   └── processing/          # Processing modules
│       ├── __init__.py
│       ├── rotationmodel.py      # AI rotation model
│       ├── rotationCRAFT.py       # CRAFT text detection & rotation
│       ├── craft_line_detection.py # CRAFT line detection
│       └── deep_ocr.py            # Deep OCR model
│
├── tests/                   # 7. Test Cases (optional)
│
├── main.py                  # 8. Entry Point
└── requirements.txt         # 9. Dependencies
```

## 📋 การจัดหมวดหมู่ไฟล์

### 🔧 config/
- **settings.py**: เก็บการตั้งค่าทั้งหมด (paths, constants, configurations)

### 💼 core/
- **camera_manager.py**: จัดการกล้องทั้งสองตัว (USBCamera, SentechCamera)
- **image_processor.py**: ฟังก์ชันประมวลผลภาพ, OCR, และ utilities
- **business_logic.py**: Thread classes สำหรับ async processing

### 🖼️ gui/
- **main_window.py**: BottleDetectionGUI class - จัดการ UI และ event handlers

### 🔗 external/
- **CRAFT-pytorch/**: CRAFT text detection library (symbolic link)
- **deep-text-recognition-benchmark/**: Deep OCR library (symbolic link)

### 📦 models/
- **detection/**: Detection models (YOLO) - bottle.pt, cap.pt
- **rotation/**: Rotation models - text_rotation_model.pth, best_accuracy.pth
- **ocr/**: OCR models - fix10.pth

### 📚 libs/
- **camera/**: Modules สำหรับจัดการกล้อง
- **detection/**: Modules สำหรับการตรวจจับวัตถุ (bottle, cap)
- **processing/**: Modules สำหรับการประมวลผลภาพ (rotation, CRAFT, line detection, OCR)

## 🔄 การ Import

### จาก config:
```python
from config.settings import (
    CAP_MODEL_PATH,
    MODBUS_IP,
    CAP_DETECTION_AVAILABLE
)
```

### จาก core:
```python
from core.camera_manager import USBCamera, SentechCamera
from core.image_processor import perform_ocr_on_image
from core.business_logic import CapDetectionThread
```

### จาก external (auto-configured):
External dependencies จะถูกเพิ่มใน sys.path อัตโนมัติผ่าน `config/settings.py`

### จาก models:
Model paths ถูกกำหนดใน `config/settings.py`:
```python
from config.settings import (
    CAP_MODEL_PATH,
    BOTTLE_MODEL_PATH,
    ROTATION_MODEL_PATH,
    CRAFT_MODEL_PATH,
    OCR_MODEL_PATH
)
```

### จาก libs:
```python
from libs.camera.usbcamra import USBCamera
from libs.detection.bottledetect import process_bottle_image_simple
from libs.detection.capmodel import initialize_detector
from libs.processing.rotationmodel import initialize_model
from libs.processing.craft_line_detection import detect_lines_from_rotation_result
from libs.processing.deep_ocr import initialize_ocr_model
```

### จาก gui:
```python
from gui.main_window import BottleDetectionGUI
```

## ✅ ประโยชน์ของการจัดโครงสร้างแบบนี้

1. **แยกหน้าที่ชัดเจน**: แต่ละส่วนมีหน้าที่เฉพาะ
2. **ดูแลง่าย**: หาไฟล์ได้ง่าย แก้ไขสะดวก
3. **ขยายได้**: เพิ่มฟีเจอร์ใหม่ได้ง่าย
4. **ทดสอบง่าย**: แยกส่วนได้ชัดเจน
5. **อ่านง่าย**: โครงสร้างเข้าใจง่าย


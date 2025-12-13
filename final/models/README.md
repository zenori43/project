# Models Directory

โฟลเดอร์นี้เก็บไฟล์โมเดลทั้งหมดของโปรเจกต์

## โครงสร้าง

```
models/
├── detection/           # Detection models (YOLO)
│   ├── bottle.pt       # Bottle detection model
│   └── cap.pt          # Cap detection model
│
├── rotation/           # Rotation models
│   ├── text_rotation_model.pth   # AI text rotation model
│   └── best_accuracy.pth         # Best accuracy rotation model
│
└── ocr/                # OCR models
    └── fix10.pth       # Deep OCR model
```

## การใช้งาน

Paths ของโมเดลถูกกำหนดใน `config/settings.py` และใช้ในโค้ดผ่านการ import:

```python
from config.settings import (
    CAP_MODEL_PATH,
    BOTTLE_MODEL_PATH,
    ROTATION_MODEL_PATH,
    OCR_MODEL_PATH,
    BEST_ACCURACY_MODEL_PATH
)
```

## หมายเหตุ

- CRAFT models ยังอยู่ที่ external path (`CRAFT-pytorch/`) เพราะเป็น external dependency
- ทุกโมเดลจะถูก validate เมื่อ import `config.settings`
- ถ้าโมเดลไม่พบ จะแสดง warning แต่โปรแกรมยังทำงานได้


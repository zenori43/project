# External Dependencies

โฟลเดอร์นี้เก็บ external repositories ที่โปรเจกต์ใช้ (linked via symbolic links)

## โครงสร้าง

```
external/
├── CRAFT-pytorch/              # CRAFT text detection library (symbolic link)
│   └── craft_mlt_25k.pth       # CRAFT model file
│   └── craft_refiner_CTW1500.pth # CRAFT refiner model file
│
└── deep-text-recognition-benchmark/  # Deep OCR library (symbolic link)
    └── utils.py                 # OCR utilities
    └── model.py                  # OCR model
    └── dataset.py               # OCR dataset
```

## การใช้งาน

Paths ของ external dependencies ถูกกำหนดใน `config/settings.py`:

```python
from config.settings import (
    CRAFT_PYTORCH_DIR,
    DEEP_TEXT_RECOGNITION_DIR,
    CRAFT_MODEL_PATH,
    CRAFT_REFINER_PATH
)
```

## หมายเหตุ

- โฟลเดอร์เหล่านี้เป็น symbolic links ไปยัง repositories จริง
- Repository จริงอยู่ที่:
  - CRAFT-pytorch: `/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch`
  - deep-text-recognition-benchmark: `/home/nvidia/Desktop/final_boss/backupsdcard/deep-text-recognition-benchmark`
- หากต้องการ clone repositories ใหม่:
  ```bash
  # CRAFT-pytorch
  git clone <repository-url> /home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch
  
  # deep-text-recognition-benchmark
  git clone <repository-url> /home/nvidia/Desktop/final_boss/backupsdcard/deep-text-recognition-benchmark
  ```


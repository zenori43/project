# การรันโปรแกรม

## วิธีการรัน

### Linux/macOS:
```bash
./run.sh
```

### Windows:
```cmd
run.bat
```

### หรือรันโดยตรงด้วย Python:
```bash
python3 main.py
# หรือ
python main.py
```

## Options

### Debug Mode (Linux/macOS only):
```bash
./run.sh --debug
# หรือ
./run.sh -d
```

Debug mode จะแสดง output ทั้งหมดบน terminal โดยไม่บันทึกลง log file

## สิ่งที่ Script จะตรวจสอบ

1. **Python Version** - ตรวจสอบว่ามี Python3 หรือไม่
2. **ไฟล์หลัก** - ตรวจสอบว่า main.py มีอยู่
3. **โครงสร้างโปรเจกต์** - ตรวจสอบว่าโฟลเดอร์สำคัญมีอยู่:
   - config/
   - core/
   - gui/
   - libs/
   - models/
   - external/
4. **โมเดล** - ตรวจสอบว่าโมเดลที่จำเป็นมีอยู่:
   - models/detection/cap.pt
   - models/detection/bottle.pt
   - models/rotation/text_rotation_model.pth
   - models/ocr/fix10.pth
5. **External Dependencies** - ตรวจสอบว่า:
   - external/CRAFT-pytorch/ มีอยู่
   - external/deep-text-recognition-benchmark/ มีอยู่

## Log Files

เมื่อรันใน normal mode (ไม่ใช่ debug mode) จะสร้าง log file:
- `app.log` - เก็บ output ทั้งหมด

## Troubleshooting

### ถ้า Script ไม่สามารถ execute ได้:
```bash
chmod +x run.sh
```

### ถ้าโมเดลไม่พบ:
- ตรวจสอบว่าโมเดลอยู่ใน `models/` directory ที่ถูกต้อง
- ตรวจสอบ path ใน `config/settings.py`

### ถ้า External Dependencies ไม่พบ:
- ตรวจสอบว่า symbolic links ใน `external/` ยังทำงานอยู่
- ถ้า symbolic link หายไป ให้สร้างใหม่:
  ```bash
  cd external
  ln -sf /home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch CRAFT-pytorch
  ln -sf /home/nvidia/Desktop/final_boss/backupsdcard/deep-text-recognition-benchmark deep-text-recognition-benchmark
  ```

### ถ้า Python Import Error:
- ตรวจสอบว่า dependencies ติดตั้งครบ:
  ```bash
  pip install -r requirements.txt
  ```


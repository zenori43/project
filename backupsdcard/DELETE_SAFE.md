# ไฟล์/โฟลเดอร์ที่ลบได้และไม่ได้

## ❌ อย่าลบ (ยังใช้งานอยู่):

### 1. **final/** ✅ จำเป็นมาก!
- **เป็นโปรเจกต์หลัก** - เก็บโค้ดทั้งหมด
- **อย่าลบเด็ดขาด!**

### 2. **CRAFT-pytorch/** ⚠️ ยังใช้อยู่!
- **Linked ไว้ใน**: `final/external/CRAFT-pytorch`
- **ถูกใช้ใน**: 
  - `libs/processing/rotationCRAFT.py`
  - `libs/processing/craft_line_detection.py`
  - `config/settings.py`
- **Size**: ~117MB
- **ถ้าลบ**: โปรแกรมจะไม่สามารถใช้ CRAFT text detection ได้

### 3. **deep-text-recognition-benchmark/** ⚠️ ยังใช้อยู่!
- **Linked ไว้ใน**: `final/external/deep-text-recognition-benchmark`
- **ถูกใช้ใน**: 
  - `libs/processing/deep_ocr.py`
- **Size**: ~957MB
- **ถ้าลบ**: โปรแกรมจะไม่สามารถทำ OCR ได้

## ✅ ลบได้ (ไม่เกี่ยวข้องกับโปรเจกต์):

### 4. **DE/**
- ไม่เกี่ยวข้องกับโปรเจกต์
- **ลบได้ถ้าไม่ต้องการ**

### 5. **drive-download-20250825.../**
- ไฟล์ที่ดาวน์โหลดมา
- **ลบได้ถ้าไม่ต้องการ**

### 6. **Installed_Programs/**
- โปรแกรมที่ติดตั้งไว้
- **ลบได้ถ้าไม่ต้องการ** (แต่ระวังว่าโปรแกรมที่ติดตั้งอาจหายไป)

### 7. **System Volume Information/**
- ไฟล์ระบบ Windows (ถ้ามี)
- **ลบได้ถ้าไม่ต้องการ**

## 📋 สรุป:

### ต้องเก็บไว้:
- ✅ **final/** - โปรเจกต์หลัก
- ✅ **CRAFT-pytorch/** - ยังใช้อยู่ (linked)
- ✅ **deep-text-recognition-benchmark/** - ยังใช้อยู่ (linked)

### ลบได้:
- ❌ **DE/** - ไม่เกี่ยวข้อง
- ❌ **drive-download-20250825.../** - ไฟล์ดาวน์โหลด
- ❌ **Installed_Programs/** - โปรแกรม (ระวัง)
- ❌ **System Volume Information/** - ไฟล์ระบบ

## ⚠️ คำเตือน:

ถ้าลบ **CRAFT-pytorch** หรือ **deep-text-recognition-benchmark**:
- โปรแกรมจะไม่สามารถใช้ text detection และ OCR ได้
- จะมี error เมื่อพยายามใช้ฟีเจอร์เหล่านี้

## 💡 คำแนะนำ:

ถ้าต้องการลบเพื่อประหยัดพื้นที่:
1. **อย่าลบ**: final, CRAFT-pytorch, deep-text-recognition-benchmark
2. **ลบได้**: DE, drive-download-*, Installed_Programs (ถ้าไม่ต้องการ), System Volume Information

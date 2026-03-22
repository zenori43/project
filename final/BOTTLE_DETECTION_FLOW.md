# 📋 สรุปการทำงานของระบบตรวจจับขวด

## 🔄 Flow การทำงานทั้งหมด

### 1️⃣ **ขั้นตอนการถ่ายภาพ**
- เมื่อได้รับสัญญาณ **M301 ON** → ถ่ายภาพจากกล้อง USB และ Sentech พร้อมกัน
- เก็บภาพไว้ใน `current_usb_image` และ `current_sentech_image`

---

### 2️⃣ **ขั้นตอนการตรวจจับขวด (Bottle Detection)**

#### 2.1 **YOLO Detection**
- ใช้โมเดล YOLO ตรวจจับวัตถุในภาพจากกล้อง USB
- Labels ที่ตรวจจับได้: `angle1`, `angle2`, `angle3`, `type`
- เก็บผลลัพธ์ใน `detected_labels` list

#### 2.2 **ตรวจสอบ Angle3 ก่อน (Priority Check)**
```
IF 'angle3' in detected_labels:
    → ตั้งค่า bottle_type = "M130" ทันที
    → ตั้งค่า combined_ocr_text = "angle3 detected by YOLO"
    → ข้ามการทำ OCR
    → ส่งผลลัพธ์ไปยัง handle_bottle_type_detection()
    → RETURN (ไม่ต้องทำ OCR)
```

#### 2.3 **Crop Type Regions (ถ้าไม่ใช่ angle3)**
- ครอปส่วน "type" region จากภาพ
- เก็บไว้ใน `type_crops` list

#### 2.4 **OCR Processing (ถ้าไม่ใช่ angle3)**
- ทำ OCR บน `type_crops` ทั้งหมด
- รวมข้อความที่อ่านได้เป็น `combined_ocr_text`
- ตรวจสอบคำสำคัญ:
  - `"เดิม"` → `bottle_type = "M100"` (ดั้งเดิม)
  - `"2%"` → `bottle_type = "M110"` (น้ำตาล 2%)
  - `"ลัก"` → `bottle_type = "M120"` (ผสมแมงลัก)

---

### 3️⃣ **ขั้นตอนการจัดการผลลัพธ์ (handle_bottle_type_detection)**

#### 3.1 **กรณี M130 (Angle3)**
```
IF bottle_type == "M130":
    → ตรวจสอบ OCR:
        - IF OCR อ่านได้ (ocr_text != "angle3 detected by YOLO" และไม่ว่าง):
            → D7009 = 40
        - ELSE (OCR อ่านไม่ได้):
            → D7009 = 50 (ฝาไม่ผ่าน)
    
    → ON M850 (angle3 signal)
    → ON M600
    → เริ่มอ่าน M402 (แทน D6006)
    → ข้ามการประมวลผลฝา (ไม่ต้องตรวจฝา)
    → RETURN
```

#### 3.2 **กรณี M100/M110/M120 (ไม่ใช่ angle3)**
```
IF bottle_type in ["M100", "M110", "M120"]:
    → เก็บ bottle_type ไว้ใน current_bottle_type
    → แสดงสถานะ "รอผลลัพธ์ฝาก่อน ON Modbus..."
    → เริ่มประมวลผลฝา (Cap Detection) จากภาพ Sentech
    → รอผลลัพธ์ฝาก่อนส่งสัญญาณ Modbus
```

---

### 4️⃣ **ขั้นตอนการตรวจจับฝา (Cap Detection)**

#### 4.1 **ประมวลผลฝา**
- ใช้ภาพจากกล้อง Sentech (`current_sentech_image`)
- ตรวจจับฝาและตรวจสอบความถูกต้อง

#### 4.2 **หลังจากตรวจฝาเสร็จ (on_cap_processing_complete)**
```
IF ฝาผ่าน (validate_cap_result() == True):
    → ตรวจสอบ bottle_type:
        - IF M100: D7009 = 10
        - IF M110: D7009 = 20
        - IF M120: D7009 = 30
    
    → ON M600
    → อัปเดต UI (lamp, status)

ELSE (ฝาไม่ผ่าน):
    → D7009 = 50 (ฝาไม่ผ่าน)
    → ON M140 (ฝาไม่ผ่าน signal)
    → อัปเดต UI
```

---

### 5️⃣ **โหมดถ่ายภาพซ้ำสำหรับ Angle3 (Angle3 Retry Mode)**

#### 5.1 **เมื่อ M402 ON (แทน D6006 = 300)**
```
→ RESET M130, M850, M600
→ เริ่มโหมดถ่ายภาพซ้ำ (angle3_retry_mode = True)
→ Reset การประมวลผล
→ ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)
```

#### 5.2 **ในโหมดถ่ายภาพซ้ำ**
```
IF ตรวจจับได้ bottle_type != "M130":
    → ส่งสัญญาณตาม bottle_type (D7009 = 10/20/30)
    → ON M600
    → หยุดโหมดถ่ายภาพซ้ำ (angle3_retry_mode = False)

ELSE (ยังเป็น M130):
    → ตรวจสอบ OCR:
        - IF OCR อ่านได้: D7009 = 40
        - ELSE: D7009 = 50
    
    → ON M850, M600
    → ถ่ายภาพ USB ใหม่ทันที (ถ่ายซ้ำ)
    → เพิ่ม angle3_retry_count
```

#### 5.3 **เมื่อ M402 OFF (แทน D6006 = 0)**
```
→ RESET M850
→ หยุดโหมดถ่ายภาพซ้ำ (angle3_retry_mode = False)
```

---

## 📊 สรุปค่า D7009

| ค่า | ความหมาย | เงื่อนไข |
|-----|----------|----------|
| `10` | M100 (ดั้งเดิม) | ตรวจจับได้ "เดิม" + ฝาผ่าน |
| `20` | M110 (น้ำตาล 2%) | ตรวจจับได้ "2%" + ฝาผ่าน |
| `30` | M120 (ผสมแมงลัก) | ตรวจจับได้ "ลัก" + ฝาผ่าน |
| `40` | M130 (angle3) | YOLO เจอ angle3 + OCR อ่านได้ |
| `50` | ฝาไม่ผ่าน | ฝาไม่ผ่าน หรือ angle3 แต่ OCR อ่านไม่ได้ |

---

## 🔄 Flow Diagram

```
M301 ON
  ↓
ถ่ายภาพ USB + Sentech
  ↓
YOLO Detection
  ↓
┌─────────────────┐
│ 'angle3' in     │
│ detected_labels?│
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
   YES       NO
    │         │
    ↓         ↓
M130 Path   OCR Path
    │         │
    │         ↓
    │    Crop Type Regions
    │         │
    │         ↓
    │    OCR Processing
    │         │
    │         ↓
    │    Check Keywords:
    │    "เดิม" → M100
    │    "2%" → M110
    │    "ลัก" → M120
    │         │
    └────┬────┘
         │
         ↓
handle_bottle_type_detection()
         │
    ┌────┴────┐
    │         │
  M130      M100/M110/M120
    │         │
    │         ↓
    │    Cap Detection
    │         │
    │    ┌────┴────┐
    │    │         │
    │  Pass      Fail
    │    │         │
    │    ↓         ↓
    │ D7009=10/20/30  D7009=50
    │    │         │
    │    └────┬────┘
    │         │
    └─────────┴─────────┐
                        │
                        ↓
                    ON M600
                        │
                        ↓
                    รอ M401 ON
                        │
                        ↓
                    RESET M600 + D7009=0
```

---

## 🎯 จุดสำคัญ

1. **Angle3 Detection**: ใช้ YOLO เป็นหลัก ไม่ต้องรอ OCR
2. **OCR Validation**: สำหรับ angle3 ใช้ OCR เพื่อยืนยัน (ส่ง 40 หรือ 50)
3. **Cap Detection**: ตรวจเฉพาะ M100/M110/M120 ไม่ตรวจ M130
4. **Retry Mode**: เมื่อ M402 ON จะถ่ายภาพซ้ำจนกว่าจะเจอ non-angle3
5. **D7009 Values**: 
   - 10/20/30 = ประเภทขวด + ฝาผ่าน
   - 40 = angle3 + OCR อ่านได้
   - 50 = ฝาไม่ผ่าน หรือ angle3 แต่ OCR อ่านไม่ได้

---

## 📝 ไฟล์ที่เกี่ยวข้อง

- `core/business_logic.py`: BottleDetectionThread (การประมวลผลภาพ)
- `core/image_processor.py`: check_bottle_type() (ตรวจสอบประเภทขวด)
- `libs/detection/bottledetect.py`: YOLO detection และ crop type regions
- `gui/handlers/bottle_handlers.py`: handle_bottle_type_detection() (จัดการผลลัพธ์)
- `gui/handlers/cap_handlers.py`: on_cap_processing_complete() (จัดการผลลัพธ์ฝา)
- `gui/handlers/modbus_handlers.py`: on_bottle_type_after_cap_validation() (ส่งสัญญาณ Modbus)


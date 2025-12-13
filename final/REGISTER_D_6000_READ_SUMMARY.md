# สรุปรายการ Register D ในช่วง 6000 ที่ระบบอ่านในโหมด Auto (ID 7)

## 1. ~~D6004~~ → **M401** - Reset M600 Trigger
- **ไฟล์**: `core/business_logic.py` (ฟังก์ชัน `ModbusThread.run()`)
- **Unit**: 1 (Coil)
- **เงื่อนไขการอ่าน**: 
  - อ่านเมื่อ `m600_reset_pending = True`
  - ใช้ทั้งโหมด auto และ capture only
- **การใช้งาน**:
  - เมื่อ `M401 = ON` → ทำการ reset M600 และ coils อื่นๆ (M100, M110, M120, M130, M140)
  - Reset D7009 = 0 เพื่อเตรียมรับค่าใหม่
  - ส่งสถานะไปยัง GUI ผ่าน signal `m401_status_updated`
- **สถานะ**: ✅ ใช้งานทั้งโหมด auto และ capture only (เปลี่ยนจาก D6004 = 200)

---

## 2. ~~D6006~~ → **M402** - M130 Monitoring & Angle3 Retry Mode
- **ไฟล์**: `core/business_logic.py` (ฟังก์ชัน `ModbusThread.run()`)
- **Unit**: 1 (Coil)
- **เงื่อนไขการอ่าน**: 
  - อ่านเมื่อ `d6006_monitoring = True` หรือ `angle3_retry_mode = True`
- **การใช้งาน**:
  - **เมื่อ `M402 ON` (แทน D6006 = 300)**:
    - Reset M130, M850, M600
    - หยุด monitoring M402
    - เริ่มโหมดถ่ายภาพซ้ำ (`angle3_retry_mode = True`)
    - Reset การประมวลผล
    - ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)
  - **เมื่อ `M402 OFF` (แทน D6006 = 0)**:
    - Reset M850
    - หยุดโหมดถ่ายภาพซ้ำ (`angle3_retry_mode = False`)
- **สถานะ**: ✅ ใช้งานเมื่อมีการ monitor M130 หรืออยู่ในโหมดถ่ายภาพซ้ำ (เปลี่ยนจาก D6006)

---

## 3. D6007 - Bottle Type Detection
- **ไฟล์**: `core/business_logic.py` (ฟังก์ชัน `ModbusThread.run()`)
- **Unit**: 2
- **เงื่อนไขการอ่าน**: 
  - อ่านตลอดเมื่อ `program_enabled = True`
  - และ `not self.capture_only_mode` (โหมด auto เท่านั้น)
- **การใช้งาน**:
  - ใช้เพื่อตรวจสอบประเภทขวด
  - ส่งค่าไปยัง GUI ผ่าน signal `d6007_status_updated` เฉพาะเมื่อค่าเปลี่ยน
  - เก็บค่าไว้ใน `last_d6007_value` เพื่อตรวจสอบการเปลี่ยนแปลง
- **สถานะ**: ✅ ใช้งานในโหมด auto

---

## สรุปตามเงื่อนไขการอ่าน

### อ่านตลอดในโหมด Auto:
- **D6007** - ตรวจสอบประเภทขวด

### อ่านเมื่อมีเงื่อนไขเฉพาะ:
- **M401** - อ่านเมื่อ `m600_reset_pending = True` (รอ reset M600) - ใช้แทน D6004 = 200
- **M402** - อ่านเมื่อ `d6006_monitoring = True` หรือ `angle3_retry_mode = True` (M130 monitoring หรือโหมดถ่ายภาพซ้ำ) - ใช้แทน D6006

---

## สรุปจำนวน Register D ที่อ่านในโหมด Auto
**ทั้งหมด 1 D-Register + 2 Coils:**
1. ~~D6004~~ → **M401** ✅ (เมื่อรอ reset M600) - เปลี่ยนจาก D6004 = 200 เป็น M401 ON
2. ~~D6006~~ → **M402** ✅ (เมื่อ monitor M130 หรือโหมดถ่ายภาพซ้ำ) - เปลี่ยนจาก D6006 เป็น M402
3. D6007 ✅ (อ่านตลอด)

---

## หมายเหตุ
- **เปลี่ยนจาก D6004 = 200 เป็น M401 ON** - ใช้ทั้งโหมด auto และ capture only
- **เปลี่ยนจาก D6006 เป็น M402** - M402 จะอ่านเฉพาะเมื่อมีการ monitor M130 หรืออยู่ในโหมดถ่ายภาพซ้ำเท่านั้น
- D6007 อ่านเฉพาะในโหมด auto (ไม่อ่านในโหมด capture only)


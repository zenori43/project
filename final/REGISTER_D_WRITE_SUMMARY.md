# สรุปรายการ Register D ที่ระบบเขียนไปทั้งหมด

## 1. D4005 - OCR Confidence Value
- **ไฟล์**: `core/business_logic.py` (ฟังก์ชัน `set_d4005`)
- **ค่า**: 100 (default)
- **วัตถุประสงค์**: ส่งค่า confidence ของ OCR
- **สถานะ**: มีฟังก์ชัน แต่ไม่เห็นการเรียกใช้ในโค้ดปัจจุบัน

---

## 2. D5012 - RESET Button (Momentary)
- **ไฟล์**: `gui/main_window.py` (ฟังก์ชัน `on_reset_pressed`, `on_reset_released`)
- **ค่า**:
  - `1` = เมื่อกดปุ่ม RESET (pressed)
  - `0` = เมื่อปล่อยปุ่ม RESET (released)
- **วัตถุประสงค์**: ปุ่ม RESET แบบ momentary (กดค้าง)
- **สถานะ**: ✅ ใช้งานอยู่

---

## 3. D5500 - Mode Selection
- **ไฟล์**: 
  - `gui/main_window.py` (ฟังก์ชัน `write_initial_mode`)
  - `gui/handlers/modbus_handlers.py` (ฟังก์ชัน `on_mode_changed`)
- **ค่า**:
  - `7` = ID 7 full auto mode
  - `5` = ID 5 capture only mode
- **วัตถุประสงค์**: เลือกโหมดการทำงานของระบบ
- **สถานะ**: ✅ ใช้งานอยู่

---

## 4. D9002 - Two Taste Mode Selection
- **ไฟล์**: `gui/main_window.py` (ฟังก์ชัน `send_two_taste_signal`)
- **ค่า**:
  - `1` = M100 + M110 (ขวดดั้งเดิม + ขวดน้ำตาล 2%)
  - `2` = M100 + M120 (ขวดดั้งเดิม + ขวดผสมแมงลัก)
  - `3` = M110 + M120 (ขวดน้ำตาล 2% + ขวดผสมแมงลัก)
  - `4` = M110 + M100 (ขวดน้ำตาล 2% + ขวดดั้งเดิม)
  - `5` = M120 + M110 (ขวดผสมแมงลัก + ขวดน้ำตาล 2%)
  - `6` = M120 + M100 (ขวดผสมแมงลัก + ขวดดั้งเดิม)
- **วัตถุประสงค์**: ส่งสัญญาณเลือกรสชาติ 2 รสชาติ
- **สถานะ**: ✅ ใช้งานอยู่

---

## 5. D9006 - Taste Mode (1/2/3 tastes)
- **ไฟล์**: 
  - `gui/handlers/modbus_handlers.py` (ฟังก์ชัน `on_taste_mode_changed`, `initialize_modbus_state`)
- **ค่า**:
  - `10` = โหมด 3 รสชาติ (M100, M110, M120 ทั้งหมด)
  - `20` = โหมด 2 รสชาติ
  - `30` = โหมด 1 รสชาติ
- **วัตถุประสงค์**: ตั้งค่าโหมดจำนวนรสชาติ
- **สถานะ**: ✅ ใช้งานอยู่

---

## 6. D7007 - One Taste Mode Selection
- **ไฟล์**: `gui/main_window.py` (ฟังก์ชัน `send_one_taste_signal`)
- **ค่า**:
  - `1` = M100 (ขวดดั้งเดิม)
  - `2` = M110 (ขวดน้ำตาล 2%)
  - `3` = M120 (ขวดผสมแมงลัก)
- **วัตถุประสงค์**: ส่งสัญญาณเลือกรสชาติ 1 รสชาติ
- **สถานะ**: ✅ ใช้งานอยู่

---

## 7. D7009 - Bottle Type Detection Result
- **ไฟล์**: 
  - `gui/handlers/modbus_handlers.py` (ฟังก์ชัน `on_bottle_type_after_cap_validation`)
  - `gui/main_window.py` (ฟังก์ชัน `validate_cap_text`)
  - `gui/handlers/bottle_handlers.py` (ฟังก์ชัน `process_bottle_detection`)
- **ค่า**:
  - `10` = M100 (พบคำว่า 'เดิม' - ฝาผ่าน)
  - `20` = M110 (พบคำว่า '2%' - ฝาผ่าน)
  - `30` = M120 (พบคำว่า 'ลัก' - ฝาผ่าน)
  - `40` = M130 (พบคำว่า 'angle3')
  - `50` = ฝาไม่ผ่าน (ไม่ตรงกับฟอร์ม)
- **วัตถุประสงค์**: ส่งผลการตรวจสอบประเภทขวดและสถานะฝา
- **สถานะ**: ✅ ใช้งานอยู่

---

## สรุปตามไฟล์

### `gui/main_window.py`
- D5012 (ค่า 1, 0) - RESET button
- D5500 (ค่า 7, 5, 8) - Mode selection (initial)
- D9002 (ค่า 1-6) - Two taste mode
- D7007 (ค่า 1-3) - One taste mode
- D7009 (ค่า 50) - Cap validation failed

### `gui/handlers/modbus_handlers.py`
- D5500 (ค่า 7, 5, 8) - Mode selection (on change)
- D9006 (ค่า 10, 20, 30) - Taste mode
- D7009 (ค่า 10, 20, 30) - Bottle type after cap validation

### `gui/handlers/bottle_handlers.py`
- D7009 (ค่า 40) - M130 (angle3)

### `core/business_logic.py`
- D4005 (ค่า 100) - OCR confidence (มีฟังก์ชันแต่ไม่เห็นการเรียกใช้)

---

## สรุปจำนวน Register D ที่เขียน
**ทั้งหมด 7 Registers:**
1. D4005 (ไม่ใช้งาน)
2. D5012 ✅
3. D5500 ✅
4. D9002 ✅
5. D9006 ✅
6. D7007 ✅
7. D7009 ✅


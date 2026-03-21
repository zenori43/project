# การเพิ่ม Performance และลดการค้าง (GUI ไม่ค้าง)

## สรุปการปรับที่ทำแล้วในโค้ด

1. **Sentech capture** – ลดจำนวน flush (ใช้ config `SENTECH_FLUSH_MAX`), ลดเวลารอเฟรมใหม่, ลด log ทุกเฟรม
2. **YOLO ขวด** – ใช้ `YOLO_IMGSZ` / `YOLO_MAX_EDGE` จาก config (เล็กลง = เร็วขึ้น, แม่นน้อยลง)
3. **UI หนัก (แสดงผล/ประวัติ)** – ใช้ `QTimer.singleShot(0, ...)` แยกงานหนักออกจาก callback ของ signal เพื่อไม่ให้ main thread ค้างยาว
4. **processEvents** – ลดการเรียกใน hot path (เช่น ตอนแสดงผลจากคิว) เพื่อลด re-entrancy และการค้าง

---

## Config ที่เกี่ยวกับความเร็ว (config/settings.py)

| ค่า | ความหมาย | ค่าแนะนำ |
|-----|----------|----------|
| `YOLO_IMGSZ` | ขนาดภาพที่ส่งเข้า YOLO ขวด (เล็ก = เร็ว) | 640 (ปกติ), 480 (เร็วขึ้น) |
| `YOLO_MAX_EDGE` | ความยาวด้านยาวสุดของภาพก่อนส่งเข้า YOLO (ย่อก่อนรัน) | 1280 หรือ 960 |
| `SENTECH_FLUSH_MAX` | จำนวนเฟรมที่ flush จาก buffer ก่อนดึงเฟรมล่าสุด | 20 (ลดจาก 50) |
| `SENTECH_WAIT_NEW_FRAME` | เวลารอ (วินาที) ก่อน fetch เฟรมใหม่ | 0.3–0.5 |

---

## แนวทางเพิ่ม Performance เพิ่มเติม (ทำเองได้)

### 1. กล้อง Sentech
- ลด `SENTECH_FLUSH_MAX` ถ้าครอบคลุม (เช่น 10–15) เพื่อลดเวลาถ่าย
- ลด `SENTECH_WAIT_NEW_FRAME` ถ้ากล้องตอบสนองเร็ว (เช่น 0.2)

### 2. YOLO / โมเดล
- ลด `YOLO_IMGSZ` เป็น 480 หรือ 384 ถ้ารับได้ว่าแม่นน้อยลง
- ลด `YOLO_MAX_EDGE` เป็น 960 เพื่อย่อภาพก่อน inference มากขึ้น
- ใช้ TensorRT / export โมเดล YOLO เป็น engine ถ้าใช้ Jetson (เร็วขึ้นมาก)

### 3. GUI
- ไม่เรียก `processEvents()` ใน loop หรือใน slot ที่ทำงานหนัก
- งานหนัก (resize ภาพใหญ่, add_to_history ที่มีหลายรายการ) ใส่ใน `QTimer.singleShot(0, ...)` หรือ worker thread
- ลดการอัปเดต label/status บ่อยเกิน (throttle อย่างน้อย 100–200 ms)

### 4. Log / print
- ลด `print()` ใน path ที่ถี่ (เช่น ทุกเฟรม, ทุก detection)
- ใช้ level log (DEBUG/INFO) และปิด DEBUG ตอนรันจริง

### 5. Memory
- หลังประมวลผลชุดใหญ่ (เช่น ทุก 20 รอบแสดงผล) เรียก `torch.cuda.empty_cache()` และ `gc.collect()` (มีอยู่แล้วใน display from queue)

---

## UI แสดงผล — อะไรกินทรัพยากรหลัก

| ส่วน | ที่กิน | หมายเหตุ |
|------|--------|----------|
| **ภาพ Sentech (ฝา)** | ภาพ 2048×2448 แปลง BGR→RGB ทั้งก้อน แล้วค่อย scale ใน Qt | ควรย่อขนาด (เช่น max 800px) ก่อน cvtColor + QImage |
| **แสดงผลฝา (display_cap_detection_results)** | หลายรูป (original, processed, crop, CRAFT, AI rotated, line crops…) + addWidget หลายสิบครั้ง | งานหนักที่สุดตอนกดแสดงผลฝา — ลดจำนวนรูปที่แสดงหรือย่อก่อนแสดง |
| **ประวัติ (add_to_history)** | เก็บ numpy ภาพใน list + สร้าง widget ใหม่ (resize, cvtColor, QImage, insertWidget) | จำกัดจำนวนรายการ (processing_history_max), อย่าเก็บภาพเต็มขนาดใน list |
| **QImage / QPixmap / setPixmap** | ทุกที่ที่แสดงรูป = copy buffer + วาด | ทำบน main thread → ย่อรูปก่อนแสดงเสมอ |
| **Debug panel** | ทุก print() → append QTextEdit + อัปเดต label | เมื่อปิด panel ไม่อัปเดต GUI (ทำแล้ว) |

สรุป: **ภาพใหญ่ไม่ย่อก่อนแสดง** และ **แสดงผลฝา (หลายรูป + หลาย widget)** เป็นจุดที่กิน CPU/ความจำมากที่สุด

---

## การตรวจสอบจุดค้าง

- เปิด Qt Creator / gdb แล้วดูว่า main thread ใช้เวลานานที่ฟังก์ชันไหน (เช่น `cv2.resize`, `setPixmap`, `add_to_history`)
- ใส่เวลา log ด้านหน้า/หลังงานหนัก (เช่น `time.time()`) เพื่อดูว่าขั้นตอนไหนช้า

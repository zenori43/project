# โหมด Full Auto (ID 7) — Flow การทำงานทั้งหมด

ไล่จากการเริ่มต้นจนถึงขวดถัดไปเข้ามา

---

## 1. เงื่อนไขเริ่มต้น

- **M511 = ON** (โปรแกรมเริ่ม) → `program_enabled = True`
- **โหมด = ID 7 full auto** → `capture_only_mode = False`
- **M600 reset แล้ว** → `m600_reset_pending = False` (ครั้งแรกหรือหลัง M401 มาครั้งล่าสุด)
- **ไม่ใช่ M130 กำลังทำงาน** → `d6006_monitoring = False`

---

## 2. ขวดแรกเข้ามา (M301 ครั้งที่ 1)

### 2.1 PLC ส่ง M301 OFF → ON

- **business_logic** (loop อ่าน Modbus):
  - อ่าน M301 เปลี่ยนเป็น ON
  - ตรวจ: `program_enabled` ✓, ไม่ `d6006_monitoring` ✓, ไม่ `capture_only_mode` ✓
  - ตรวจ: `m600_reset_pending == False` → **M600 พร้อม**
  - ตั้ง `m600_reset_pending = True`
  - **emit `trigger_detected`**

### 2.2 GUI รับ trigger_detected

- **modbus_handlers.on_modbus_trigger()** ถูกเรียก
- เรียก **bottle_handlers.start_full_auto_capture()**

### 2.3 ถ่ายภาพ (Full Auto แรก)

- **start_full_auto_capture()**:
  - สร้าง `CaptureBothThread(usb_camera, sentech_camera, delay_before_sec=2)`
  - เชื่อม `capture_done` → `_on_full_auto_capture_done`
  - `start()` → รอ 2 วินาที แล้วถ่าย USB + Sentech

### 2.4 ถ่ายเสร็จ → เริ่มประมวลผล

- **CaptureBothThread** emit `capture_done(usb_image, sentech_image)`
- **bottle_handlers._on_full_auto_capture_done(usb, sentech)**:
  - เรียก `_on_capture_both_done(usb, sentech)` → ตั้ง `current_image`, `current_sentech_image` แสดงใน GUI
  - ถ้ามี `usb_image`: เรียก **process_current_image()**

### 2.5 ประมวลผลขวดก่อน แล้วค่อยฝา (ไม่ parallel)

- **process_current_image()**:
  - สร้าง **BottleDetectionThread(current_image)** → `start()` (ขวดอย่างเดียว)
  - ไม่เริ่มฝาพร้อมกัน — รอขวดเสร็จก่อน

### 2.6 ขวดเสร็จ → เริ่มฝา

- **on_processing_complete(result)** (ไม่ silent):
  - แสดงผลขวด, เรียก **handle_bottle_type_detection** (ส่ง D7009, M110 ฯลฯ)
  - ถ้ามี `current_sentech_image` และ `cap_detector` และขวดผ่าน (ไม่ NG) → **เริ่ม process_cap_detection(started_from_parallel=True)** แล้ว return (ยังไม่ปิด loading)
- ฝาเสร็จ → **cap_handlers.on_cap_processing_complete** → แสดงผลฝา, validate, ส่ง M140 หรือ on_bottle_type_after_cap_validation, ปิด loading เปิดปุ่ม

---

## 3. M401 มาหลังขวดแรก (ขวดแรกประมวลเสร็จแล้ว)

- **business_logic** อ่าน M401 เปลี่ยนเป็น ON
  - `m600_reset_pending == True` → ทำบล็อก reset M600
  - เขียน M600=0, D7009=0, M100/M110/M120/M130/M140/M72x/M73x/M74x = 0
  - ตั้ง **`m600_reset_pending = False`**, **`waiting_for_late_result = False`**
  - ตรวจ `pending_results_queue`:
    - ถ้า **มีผลในคิว** → `pop(0)` → **emit `result_ready_to_display(bottle_result, cap_result)`**
    - ถ้า **คิวว่างแต่ `pending_m301_count > 0`** → ตั้ง **`waiting_for_late_result = True`**, `m600_reset_pending = True`, emit **`pending_display_requested`** (แสดงภาพล่าสุดระหว่างรอ)
    - ถ้า **คิวว่างและไม่มีคิว** → แค่ log “พร้อมรับ M301 ใหม่”

- **modbus_handlers.on_result_ready_to_display(bottle, cap)** (ถ้ามีผลในคิว):
  - กำหนดเวลา **singleShot** เรียก **_do_display_from_queue(bottle, cap)**
  - **_do_display_from_queue**:
    - แสดงผลขวด (display_result_from_queue)
    - เรียก handle_bottle_type_detection → ส่ง D7009, M100/M110/M120 ตามประเภทขวด
    - ถ้ามี cap_result: แสดงผลฝา, validate, ส่ง M140 ถ้าไม่ผ่าน หรือ on_bottle_type_after_cap_validation ถ้าผ่าน
    - add_to_history

หลัง M401 ครบแล้ว ระบบพร้อมรับ **M301 ขวดถัดไป**.

---

## 4. ขวดถัดไปเข้ามา (M301 ครั้งที่ 2 ขึ้นไป) — M600 ยังไม่ reset

สถานะ: ยังไม่ได้รับ M401 หลังขวดแรก → **`m600_reset_pending == True`**

### 4.1 PLC ส่ง M301 OFF → ON อีกครั้ง

- **business_logic**:
  - M301 เปลี่ยนเป็น ON
  - ตรวจ: `m600_reset_pending == True` → **M600 ยังไม่ reset**
  - **emit `silent_trigger_detected`**
  - **`pending_m301_count += 1`**

### 4.2 GUI รับ silent_trigger_detected

- **modbus_handlers.on_silent_trigger()**
- เรียก **bottle_handlers.capture_both_and_process_silently()**

### 4.3 ถ่ายภาพแบบเงียบ (Silent)

- **capture_both_and_process_silently()**:
  - สร้าง **CaptureBothThread(..., delay_before_sec=0)** ไม่รอ 2 วินาที
  - เชื่อม `capture_done` → **_on_silent_capture_both_done**
  - `start()` → ถ่าย USB + Sentech ทันที

### 4.4 ถ่ายเสร็จ → เริ่มขวดแบบ silent

- **_on_silent_capture_both_done(usb_image, sentech_image)**:
  - ตั้ง `current_image = usb_image`, `current_sentech_image = sentech_image` (ถ้ามี)
  - เรียก **_start_silent_bottle_then_cap(usb_image)**:
    - ตั้ง **`silent_mode = True`**
    - สร้าง **BottleDetectionThread(usb_image)** → `start()`
    - ไม่เริ่มฝาทันที — **ฝาจะเริ่มหลังขวดเสร็จ**

### 4.5 ขวดเสร็จ (โหมด silent)

- **bottle_handlers.on_processing_complete(result)** (silent_mode == True):
  - ตั้ง **`pending_bottle_result = result`**
  - ถ้า **ฝาพร้อมแล้ว** (`cap_handlers.pending_cap_result` มีค่า):
    - ถ้า `pending_m301_count > 0` → emit **result_ready_to_display(result, cap_result)** (แสดงทันที)
    - ไม่ก็ **append (result, cap_result)** เข้า **pending_results_queue**
  - ถ้า **ฝายังไม่พร้อม** และมีภาพ Sentech + cap_detector:
    - **ไม่ใส่ (result, None) เข้าคิว และไม่ส่งผลขวดอย่างเดียว**
    - เรียก **cap_handlers.process_cap_detection(started_from_parallel=True)** → เริ่มประมวลฝา
    - รอฝาเสร็จ แล้วค่อยรวมผลและส่งทีเดียว (ด้านล่าง)

### 4.6 ฝาเสร็จ (โหมด silent)

- **cap_handlers.on_cap_processing_complete(result)** (silent_mode == True):
  - ฝาจาง (faded_text_detected) ไม่ retry ถ่ายใหม่ ใช้ผลนี้รวมกับขวด
  - ตั้ง **pending_cap_result = result**
  - ถ้า **มี pending_bottle_result**:
    - ถ้า **waiting_for_late_result** (M401 มาแล้วแต่ผลมาสาย):
      - emit **result_ready_to_display(bottle_result, result)** → แสดง+ส่งขวด+ฝาทีเดียว
    - ถ้า **m600_reset_pending** (ยังไม่ M401):
      - หาในคิวรายการที่เป็น (bottle_result, None) → อัปเดตเป็น (bottle_result, result)
      - ถ้าไม่มีรายการแบบนั้น → **append_pending_result(bottle_result, result)** → ใส่ (ขวด, ฝา) เข้าคิว
    - ถ้า M401 ไปแล้ว (ไม่ pending ไม่ late) → emit **result_ready_to_display(bottle_result, result)**
  - ถ้า **ไม่มี pending_bottle_result** (ขวดส่งไปแล้ว ฝาเสร็จทีหลัง):
    - แสดงผลฝา + ส่ง Modbus เฉพาะฝา (M140/M600 หรือ validation ฝาผ่าน)

---

## 5. M401 มาอีกครั้ง (หลังขวดที่ 2)

- **business_logic** อ่าน M401 ON:
  - reset M600 และ coils ตามเดิม
  - ตั้ง **m600_reset_pending = False**
  - ถ้า **len(pending_results_queue) > 0**:
    - **pop(0)** → ได้ (bottle_result, cap_result)
    - **emit result_ready_to_display(bottle_result, cap_result)**
    - **pending_m301_count -= 1**
  - ถ้าคิวว่างแต่ **pending_m301_count > 0**:
    - ตั้ง **waiting_for_late_result = True**, **m600_reset_pending = True**
    - emit **pending_display_requested**
  - ถ้าคิวว่างและไม่มีคิว → พร้อมรับ M301 ใหม่

- **modbus_handlers._do_display_from_queue(bottle, cap)**:
  - แสดงผลขวด + ประเภทขวด (D7009, M110 ฯลฯ)
  - แสดงผลฝา + validate → M140 หรือ on_bottle_type_after_cap_validation
  - add_to_history

---

## 6. สรุปสั้น ๆ

| เหตุการณ์ | สถานะสำคัญ | การทำงาน |
|-----------|-------------|----------|
| M301 ครั้งแรก (M600 พร้อม) | m600_reset_pending = False | trigger_detected → start_full_auto_capture → รอ 2 วินาที ถ่ายทั้งคู่ → process_current_image (parallel ขวด+ฝา) |
| M301 ครั้งถัดไป (M600 ยังไม่ reset) | m600_reset_pending = True | silent_trigger_detected → capture_both_and_process_silently → ถ่ายทันที → ขวด silent → รอฝาเสร็จ → ใส่ (ขวด, ฝา) เข้าคิว หรือ emit ถ้า M401 มาหลักแล้ว |
| M401 ON | - | Reset M600 + coils, ถ้ามีของในคิว → pop → result_ready_to_display; ถ้าคิวว่างแต่มีคิวนับ → waiting_for_late_result = True |
| แสดงผลจากคิว | - | _do_display_from_queue(bottle, cap) → แสดงขวด+ฝา, ส่ง D7009/M110/M140 ฯลฯ, add_to_history |

---

## 7. ไฟล์ที่เกี่ยวข้อง

- **business_logic.py**: loop Modbus, M301/M401, trigger_detected / silent_trigger_detected, pending_results_queue, m600_reset_pending, waiting_for_late_result
- **modbus_handlers.py**: on_modbus_trigger, on_silent_trigger, on_result_ready_to_display, _do_display_from_queue
- **bottle_handlers.py**: start_full_auto_capture, _on_full_auto_capture_done, capture_both_and_process_silently, _on_silent_capture_both_done, _start_silent_bottle_then_cap, on_processing_complete (silent + parallel)
- **cap_handlers.py**: process_cap_detection, on_cap_processing_complete (silent: ใส่คิว/อัปเดตคิว/emit ตามสถานะ)

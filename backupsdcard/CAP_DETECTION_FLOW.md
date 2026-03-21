# 📋 ระบบตรวจจับฝาในโปรแกรมหลัก (Cap Detection Flow)

## 🎯 ภาพรวม
โปรแกรมหลักใช้ **YOLOv8** ตรวจจับฝา แล้วประมวลผลต่อด้วย **CRAFT** และ **Fade Detection**

---

## 🔄 Flow การทำงานหลัก

### **Step 1: ตรวจจับฝา (Cap Detection)**
**ไฟล์:** `final/libs/detection/capmodel.py`  
**ฟังก์ชัน:** `CapDetector.detect_caps()`

#### ขั้นตอน:
1. **โหลดภาพ** จาก `image_path` หรือ `image_array`
2. **Resize ภาพ** (ถ้าใหญ่กว่า 1280px) เพื่อเพิ่มความเร็ว
3. **YOLOv8 Detection:**
   - ใช้โมเดล YOLOv8 (`cap.pt`)
   - Threshold: `conf_threshold` (default: 0.5)
   - Device: CUDA (ถ้ามี) หรือ CPU
4. **Scale bbox กลับ** เป็นขนาดภาพต้นฉบับ
5. **Return ผลลัพธ์:**
   ```python
   {
       'detections': [
           {
               'bbox': [x1, y1, x2, y2],
               'confidence': float,
               'class_name': 'cap'
           },
           ...
       ],
       'total_detections': int
   }
   ```

---

### **Step 2: Crop ฝาที่ตรวจจับได้**
**ไฟล์:** `final/core/business_logic.py`  
**ฟังก์ชัน:** `CapDetector.crop_detections()`

#### ขั้นตอน:
1. **Crop ฝาแต่ละฝา** จากภาพต้นฉบับ
2. **เพิ่ม margin** (default: 20 pixels) รอบๆ bbox
3. **Return:** List ของ cropped images

---

### **Step 3: เลือกฝาล่างสุด (Bottommost Cap)**
**ไฟล์:** `final/core/business_logic.py` (line 209-251)

#### ขั้นตอน:
1. **คำนวณ y_center** ของแต่ละฝา: `(bbox[1] + bbox[3]) / 2`
2. **เลือกฝาที่ y_center สูงสุด** (อยู่ล่างสุดในภาพ)
3. **แสดงข้อมูล:**
   - จำนวนฝาที่พบ
   - ความเชื่อมั่นของแต่ละฝา
   - ฝาที่เลือก

---

### **Step 4: ตรวจสอบฝาจาง (Fade Detection)**
**ไฟล์:** `final/core/business_logic.py` (line 288-300)  
**ฟังก์ชัน:** `detect_faded_text_in_cap()` จาก `final/core/image_processor.py`

#### ⚠️ **สำคัญ:** ตรวจสอบ **ก่อน CRAFT rotation**

#### ขั้นตอน:
1. **Enhance Image:**
   - ใช้ `enhance_cap_image_for_fade_detection()`
   - ปรับ brightness/contrast/gamma
   - หรือใช้ Histogram Matching (ถ้าเปิดใช้งาน)

2. **Convert to Grayscale:**
   - แปลงภาพเป็น grayscale

3. **Gaussian Blur:**
   - ใช้ `cv2.GaussianBlur()` เพื่อลด noise

4. **Detect Circle (HoughCircles):**
   - หาวงกลมของฝา
   - ถ้าไม่เจอ → ใช้ fallback (วงกลมกลาง 95% ของขนาดภาพ)

5. **Create Circle Mask:**
   - สร้าง mask วงกลมเพื่อกรองเฉพาะส่วนกลางฝา

6. **Canny Edge Detection:**
   - ตรวจจับ edges ด้วย `cv2.Canny()`
   - ลบเส้นวงกลมออกจาก edges

7. **Text Edge Detection:**
   - **วิธีที่ 1:** ใช้ edges โดยตรง (ถ้า `USE_EDGES_DIRECTLY=True`)
   - **วิธีที่ 2:** ใช้ Adaptive Threshold (ถ้า `USE_ADAPTIVE_THRESHOLD=True`)
   - **วิธีที่ 3:** ใช้ Canny edges โดยตรง (default)

8. **Morphological Operations:**
   - Opening: ลบจุดเล็กๆ
   - Closing: ปิดช่องว่างในตัวอักษร

9. **Connected Component Analysis (CCA):**
   - ใช้ `cv2.connectedComponentsWithStats()`
   - กรองตาม:
     - **ขนาด (Area):** `MIN_AREA` (default: 20)
     - **ตำแหน่ง:** ระยะห่างจากจุดศูนย์กลาง < 40% ของรัศมี
     - **Aspect Ratio:** < 10
     - **Density:** > 0.15 (สำหรับ noise ที่เล็กและกระจาย)

10. **คำนวณพื้นที่:**
    - `total_area`: ผลรวมพื้นที่ของ components ที่ผ่านการกรอง
    - `num_chars`: จำนวน components

11. **ตัดสินใจ:**
    - เปรียบเทียบ `total_area` กับ `AREA_THRESH`
    - ถ้า `total_area < AREA_THRESH` → **"faded"**
    - ถ้า `total_area >= AREA_THRESH` → **"normal"**

#### Return:
```python
{
    'status': 'faded' | 'normal' | 'unknown',
    'num_chars': int,
    'total_area': int,
    'debug_images': {
        'roi_show': np.ndarray,
        'gray': np.ndarray,
        'mask_circle': np.ndarray,
        'roi_masked': np.ndarray,
        'edges': np.ndarray,
        'text_edges': np.ndarray,
        'enhanced_image': np.ndarray (optional)
    }
}
```

#### ⚠️ **หยุดการประมวลผลถ้าพบฝาจาง:**
- ถ้า `status == 'faded'`:
  - ส่งสัญญาณ **M140** (ฝาไม่ผ่าน)
  - ส่งสัญญาณ **M600** (ประมวลผลเสร็จสิ้น)
  - Return ผลลัพธ์ทันที (ไม่ทำ CRAFT/OCR ต่อ)

---

### **Step 5: CRAFT Text Detection & Rotation** (ถ้าฝาไม่จาง)
**ไฟล์:** `final/core/business_logic.py` (line 271-280)  
**ฟังก์ชัน:** `CRAFTTextDetector.detect_text_and_rotate()`

#### ขั้นตอน:
1. **Enhance Sharpness:**
   - ใช้ `enhance_cap_sharpness()` ก่อน CRAFT

2. **CRAFT Detection:**
   - ตรวจจับ text regions ด้วย CRAFT model
   - ได้ `text_boxes` (polygon coordinates)

3. **Calculate Rotation Angle:**
   - คำนวณมุมการหมุนจาก text boxes
   - ใช้ `cv2.minAreaRect()` หา bounding box

4. **Rotate Image:**
   - หมุนภาพให้ข้อความตรง (ถ้ามุม > 0.5°)

#### Return:
```python
{
    'text_boxes': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
    'rotated_image': np.ndarray,
    'rotation_angle': float,
    ...
}
```

---

### **Step 6: AI Rotation Model** (ถ้าจำเป็น)
**ไฟล์:** `final/core/business_logic.py` (line 343-346)  
**ฟังก์ชัน:** `RotationModel.process_craft_rotated_image_with_models()`

#### ขั้นตอน:
1. **รับภาพที่ CRAFT หมุนแล้ว**
2. **ใช้ AI Rotation Model** หมุนภาพให้ตรงมากขึ้น
3. **Retry Mechanism:** หมุนหลายครั้ง (สูงสุด 5 ครั้ง) จนได้ผลลัพธ์ที่ดี
4. **Line Detection:** ตรวจจับบรรทัดข้อความ
5. **OCR:** อ่านข้อความจากบรรทัด

---

## 📊 สรุป Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    เริ่มต้น (Input Image)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: YOLOv8 Cap Detection                                │
│  - ตรวจจับฝาในภาพ                                             │
│  - Return: detections[], total_detections                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Crop Detected Caps                                  │
│  - Crop แต่ละฝา + margin                                     │
│  - Return: cropped_images[]                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Select Bottommost Cap                               │
│  - เลือกฝาที่ y_center สูงสุด                                │
│  - Return: bottommost_cap, bottommost_index                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Fade Detection (⚠️ ก่อน CRAFT)                      │
│  ├─ Enhance Image                                            │
│  ├─ Grayscale + Blur                                         │
│  ├─ Detect Circle (HoughCircles)                             │
│  ├─ Canny Edge Detection                                     │
│  ├─ Text Edge Detection                                      │
│  ├─ Morphological Operations                                 │
│  ├─ Connected Component Analysis                             │
│  └─ Calculate Area & Status                                  │
│                                                              │
│  Return: {status, num_chars, total_area}                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
    ┌───────────────┐    ┌──────────────────────┐
    │ status ==     │    │ status == 'normal'    │
    │ 'faded'       │    │                       │
    └───────┬───────┘    └───────┬──────────────┘
            │                     │
            │                     ▼
            │         ┌───────────────────────────┐
            │         │ Step 5: CRAFT Detection   │
            │         │ - Detect text regions     │
            │         │ - Rotate image            │
            │         └───────┬───────────────────┘
            │                   │
            │                   ▼
            │         ┌───────────────────────────┐
            │         │ Step 6: AI Rotation        │
            │         │ - Fine-tune rotation       │
            │         │ - Line Detection          │
            │         │ - OCR                      │
            │         └───────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│  ถ้าพบฝาจาง:                                                  │
│  - ส่งสัญญาณ M140 (ฝาไม่ผ่าน)                                 │
│  - ส่งสัญญาณ M600 (เสร็จสิ้น)                                 │
│  - Return ผลลัพธ์ทันที (ไม่ทำ CRAFT/OCR)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration

### **Fade Detection Config:**
**ไฟล์:** `final/config/settings.py` → `FADED_TEXT_CONFIG`

```python
FADED_TEXT_CONFIG = {
    'AREA_THRESH': int,          # เกณฑ์พื้นที่ (ถ้า total_area < นี้ = faded)
    'MIN_AREA': int,              # พื้นที่ขั้นต่ำของ component (default: 20)
    'GAUSSIAN_BLUR_PARAMS': {
        'kernel_size': (int, int),
        'sigma': float
    },
    'HOUGH_CIRCLES_PARAMS': {
        'dp': float,
        'minDist': int,
        'param1': int,
        'param2': int,
        'minRadius': int,
        'maxRadius': int
    },
    'CANNY_PARAMS': {
        'low_threshold': int,
        'high_threshold': int
    },
    'USE_EDGES_DIRECTLY': bool,
    'USE_ADAPTIVE_THRESHOLD': bool,
    'USE_EDGES_FOR_ADAPTIVE': bool,
    'IMAGE_ENHANCEMENT': {
        'brightness': int,
        'contrast': float,
        'gamma': float
    },
    'USE_HISTOGRAM_MATCHING': bool,
    'REFERENCE_CAP_IMAGE_PATH': str
}
```

---

## 📝 หมายเหตุสำคัญ

1. **Fade Detection ทำก่อน CRAFT:**
   - เพื่อประหยัดเวลา (ถ้าฝาจาง ไม่ต้องทำ CRAFT/OCR)
   - เพื่อส่งสัญญาณ M140 ทันที

2. **เลือกฝาล่างสุด:**
   - เพราะฝาล่างสุดมักเป็นฝาที่ต้องการตรวจสอบ

3. **Enhance Image:**
   - ใช้ค่ากลางที่วิเคราะห์ได้ (brightness: 33)
   - ไม่ต้องรอ `bottle_type` แล้ว

4. **Threshold เดียวกัน:**
   - ใช้ `AREA_THRESH` เดียวกันสำหรับทุกรส (ไม่แยกรสแล้ว)

---

## 🐛 Debug & Troubleshooting

### **ตรวจสอบว่าโมเดลโหลดสำเร็จ:**
```python
if cap_detector.model is None:
    print("❌ Cap detector model not loaded")
    
if craft_detector.model is None:
    print("❌ CRAFT detector model not loaded")
```

### **ตรวจสอบผลลัพธ์ Fade Detection:**
```python
fade_result = detect_faded_text_in_cap(cap_image, show_debug=True)
print(f"Status: {fade_result['status']}")
print(f"Total Area: {fade_result['total_area']}")
print(f"Num Chars: {fade_result['num_chars']}")
```

### **ดู Debug Images:**
```python
debug_images = fade_result.get('debug_images', {})
# debug_images['roi_show']      - ภาพ ROI
# debug_images['gray']          - Grayscale
# debug_images['mask_circle']   - Circle mask
# debug_images['edges']         - Canny edges
# debug_images['text_edges']    - Text edges (cleaned)
```

---

## 📚 ไฟล์ที่เกี่ยวข้อง

1. **Cap Detection:**
   - `final/libs/detection/capmodel.py` - YOLOv8 cap detection

2. **Fade Detection:**
   - `final/core/image_processor.py` - `detect_faded_text_in_cap()`

3. **Business Logic:**
   - `final/core/business_logic.py` - `CapDetectionThread.run()`

4. **CRAFT Detection:**
   - `final/libs/processing/rotationCRAFT.py` - CRAFT text detection & rotation

5. **Configuration:**
   - `final/config/settings.py` - `FADED_TEXT_CONFIG`

---

**อัปเดตล่าสุด:** 2024

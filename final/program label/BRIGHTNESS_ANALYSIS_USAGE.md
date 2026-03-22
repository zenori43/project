# การใช้งานค่ากลางที่วิเคราะห์ได้ (Brightness Analysis Results)

## ค่าที่วิเคราะห์ได้

```
Recommended Brightness: 33
   (Average: 39)

Recommended Contrast Ratio: 0.436
   (Average: 0.419)
```

## ความหมายของค่า

### 1. Brightness Adjustment (33)
- **ความหมาย**: ต้องเพิ่ม brightness 33 เพื่อให้ Mean intensity ≈ 128
- **การใช้งาน**: ใช้เป็นค่า `brightness` ใน `IMAGE_ENHANCEMENT` หรือ `TASTE_ENHANCEMENT_MAP`
- **ตัวอย่าง**: `brightness: 33` (แทนที่จะเป็น 16 หรือ 20)

### 2. Contrast Ratio (0.436)
- **ความหมาย**: อัตราส่วนของ Std/Mean = 0.436 (ต่ำ = contrast ต่ำ)
- **การใช้งาน**: 
  - Contrast Ratio ต่ำ (< 0.5) = ภาพมีความ contrast ต่ำ → ควรเพิ่ม contrast
  - ค่า contrast ที่แนะนำ: 1.5 - 2.0 (ขึ้นอยู่กับการทดสอบ)
- **หมายเหตุ**: Contrast Ratio ไม่ใช่ contrast value โดยตรง แต่เป็นข้อมูลที่บอกว่าควรเพิ่ม contrast

## วิธีนำไปใช้ในโปรแกรมหลัก

### วิธีที่ 1: อัปเดตค่าใน `config/settings.py`

แก้ไขไฟล์ `final/config/settings.py`:

```python
FADED_TEXT_CONFIG = {
    # Image enhancement parameters
    'IMAGE_ENHANCEMENT': {
        'brightness': 33,      # เปลี่ยนจาก 20 เป็น 33
        'contrast': 1.8,       # เพิ่ม contrast (จาก 1.57 เป็น 1.8)
        'gamma': 1.0
    },
    'TASTE_ENHANCEMENT_MAP': {
        'M100': {
            'brightness': 33,  # เปลี่ยนจาก 16 เป็น 33
            'contrast': 1.8,   # เปลี่ยนจาก 1.7 เป็น 1.8
            'gamma': 1.0
        },
        'M110': {
            'brightness': 33,  # เปลี่ยนจาก 16 เป็น 33
            'contrast': 2.0,   # เปลี่ยนจาก 2.7 เป็น 2.0 (หรือทดสอบค่าใหม่)
            'gamma': 1.0
        },
        'M120': {
            'brightness': 33,  # เปลี่ยนจาก 16 เป็น 33
            'contrast': 2.0,   # เปลี่ยนจาก 2.7 เป็น 2.0 (หรือทดสอบค่าใหม่)
            'gamma': 1.0
        }
    }
}
```

### วิธีที่ 2: ใช้ผ่าน Settings Tab ใน GUI

1. เปิดโปรแกรมหลัก
2. ไปที่ Settings Tab
3. ปรับค่า Brightness และ Contrast ตามค่าที่วิเคราะห์ได้:
   - **Brightness**: 33 (หรือ 39 ถ้าใช้ค่าเฉลี่ย)
   - **Contrast**: 1.8 - 2.0 (ทดสอบดูว่าค่าไหนดีที่สุด)

### วิธีที่ 3: สร้างสคริปต์อัปเดตค่า

ใช้สคริปต์ `update_brightness_config.py` (สร้างด้านล่าง)

## ผลลัพธ์ที่คาดหวัง

หลังจากอัปเดตค่าแล้ว:
1. **ภาพฝาทุกภาพจะมีแสงสม่ำเสมอ** (Mean intensity ≈ 128)
2. **การตรวจจับ fade มีความแม่นยำมากขึ้น** เพราะภาพมี contrast ที่เหมาะสม
3. **ลด false positive/negative** ในการตรวจจับ fade

## หมายเหตุ

- **Brightness (33)**: ใช้ได้เลย - เป็นค่าที่คำนวณมาเพื่อให้ Mean ≈ 128
- **Contrast**: ต้องทดสอบ - Contrast Ratio 0.436 บอกว่า contrast ต่ำ แต่ต้องทดสอบว่าค่า contrast เท่าไหร่ที่เหมาะสม (แนะนำ 1.8 - 2.0)
- **การทดสอบ**: ควรทดสอบกับภาพฝาจริงหลายภาพเพื่อดูว่าค่าไหนให้ผลลัพธ์ดีที่สุด

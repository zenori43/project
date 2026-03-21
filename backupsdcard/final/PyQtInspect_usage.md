# วิธีใช้ PyQtInspect กับ UI โปรเจกต์นี้

PyQtInspect เป็นเครื่องมือแบบ Chrome DevTools สำหรับดีบัก PyQt: เลือก widget ด้วยการคลิก ดู properties / hierarchy และไล่ไปหา code ที่สร้าง element ได้

## ติดตั้ง

```bash
pip install PyQtInspect
```

ต้องการ: Python 3.7+ และ PyQt5 (โปรเจกต์นี้ใช้ PyQt5 อยู่แล้ว)

---

## วิธีรัน (แนะนำ: Direct mode)

รันทั้ง Inspector และแอปพร้อมกัน (ให้อยู่ที่โฟลเดอร์ `final`):

```bash
cd final
python -m PyQtInspect --direct --file main.py
```

ถ้าใช้ PyQt6 ต้องระบุ: `--qt-support=PyQt6` (โปรเจกต์นี้ใช้ PyQt5 ไม่ต้องใส่)

---

## โหมดแยก (Server + Client)

**เทอร์มินัลที่ 1 – เปิด Inspector (Server):**
```bash
pqi-server
```

**เทอร์มินัลที่ 2 – รันแอปผ่าน PyQtInspect (Client):**
```bash
cd final
python -m PyQtInspect --file main.py
```

แอปจะต่อกับ Inspector ที่รันอยู่แล้ว

---

## สิ่งที่ทำได้ใน Inspector

- คลิกเลือก widget บน UI เพื่อดู object, class, properties
- ดู hierarchy ของ widget (parent/child)
- ใช้ช่วยไล่หาว่าช่อง "ภาพฝา" หรือ widget อื่นอยู่ที่ไหนและถูกสร้างจาก code บรรทัดไหน

---

## หมายเหตุ

- รันจากโฟลเดอร์ `final` เสมอ (เพราะ `main.py` โหลด `config`, `core`, `gui` แบบ relative)
- ถ้า `pqi-server` หรือ `PyQtInspect` ไม่รู้จัก ให้ใช้ `python -m PyQtInspect` ตามตัวอย่างด้านบน

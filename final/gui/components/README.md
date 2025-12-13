# GUI Components Structure

ไฟล์นี้แยก components ของ main window ออกเป็นหลายไฟล์เพื่อให้ง่ายต่อการแก้ไข

## Structure

```
gui/
├── main_window.py          # Main window ที่รวมทุกอย่างเข้าด้วยกัน
├── login_dialog.py         # Login Dialog
├── debug_stream.py         # Debug Stream
└── components/
    ├── __init__.py
    ├── bottle_tab.py       # Bottle Detection Tab
    ├── cap_tab.py          # Cap Detection Tab
    ├── status_tab.py       # Status Tab (จะสร้างต่อไป)
    ├── ocr_test_tab.py     # OCR Test Tab (จะสร้างต่อไป)
    ├── settings_tab.py     # Settings Tab (จะสร้างต่อไป)
    └── README.md           # ไฟล์นี้
```

## Usage

แต่ละ component function จะ return:
- `(widget, widgets_dict)` tuple
  - `widget`: QWidget สำหรับ tab
  - `widgets_dict`: dict ที่เก็บ widgets ทั้งหมดที่ main window ต้องการอ้างอิง

## Example

```python
from gui.components.bottle_tab import create_bottle_tab

bottle_tab, bottle_widgets = create_bottle_tab()
# bottle_widgets['image_label'], bottle_widgets['results_text'], etc.
```

## Event Handlers

Event handlers จะถูกแยกออกไปยัง `event_handlers.py` หรือเก็บไว้ใน main_window.py ตามความเหมาะสม


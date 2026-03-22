# -*- coding: utf-8 -*-
"""
History Tab Component
Builds the processing history tab UI.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton,
    QFrame, QGridLayout, QListWidget, QListWidgetItem, QTabWidget
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np


def bottle_type_flavor_label(bottle_type):
    """
    Map bottle Modbus/type codes to product flavor names (English), aligned with D6007 product lines.
    """
    if bottle_type is None:
        return "Not specified"
    code = str(bottle_type).strip()
    if not code or code.lower() == "unknown":
        return "Not specified"
    mapping = {
        "M100": "Original soy milk",
        "M110": "Less sugar (2%)",
        "M120": "With lac seeds",
        "M130": "View 3 (angle3)",
    }
    return mapping.get(code, code)


def create_history_tab():
    """
    Create the History tab.
    Returns:
        tuple: (history_tab_widget, widgets_dict)
    """
    history_tab = QWidget()
    history_layout = QVBoxLayout(history_tab)
    
    # History Title
    history_title = QLabel("📋 Processing history")
    history_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; padding: 15px;")
    history_title.setAlignment(Qt.AlignCenter)
    history_layout.addWidget(history_title)
    
    # Control buttons
    control_layout = QHBoxLayout()
    btn_clear_history = QPushButton("🗑️ Clear History")
    btn_clear_history.setStyleSheet("""
        QPushButton {
            padding: 8px 15px;
            font-size: 12px;
            background-color: #e74c3c;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #c0392b;
        }
    """)
    control_layout.addWidget(btn_clear_history)
    control_layout.addStretch()
    
    history_count_label = QLabel("History count: 0")
    history_count_label.setStyleSheet("color: #7f8c8d; font-size: 12px; padding: 5px;")
    control_layout.addWidget(history_count_label)
    
    history_layout.addLayout(control_layout)
    
    # Sub-tabs: All, Good, NG
    history_tab_widget = QTabWidget()
    history_tab_widget.setStyleSheet("""
        QTabWidget::pane { border: 1px solid #bdc3c7; background-color: #f5f5f5; border-radius: 4px; }
        QTabBar::tab { padding: 8px 16px; font-weight: bold; }
    """)
    
    def _make_scroll_and_layout(empty_text):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(320, 360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("border: 2px solid #34495e; background-color: #ecf0f1;")
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        empty = QLabel(empty_text)
        empty.setStyleSheet("color: #95a5a6; font-size: 16px; padding: 50px;")
        empty.setAlignment(Qt.AlignCenter)
        layout.addWidget(empty)
        scroll.setWidget(content)
        return scroll, content, layout, empty
    
    # Tab: All
    history_scroll, history_content, history_content_layout, empty_label = _make_scroll_and_layout("📭 No processing history yet")
    history_tab_widget.addTab(history_scroll, "All")
    
    # Tab: Good
    history_scroll_good, history_content_good, history_content_layout_good, empty_label_good = _make_scroll_and_layout("📭 No Good results yet")
    history_tab_widget.addTab(history_scroll_good, "Good")
    
    # Tab: NG
    history_scroll_ng, history_content_ng, history_content_layout_ng, empty_label_ng = _make_scroll_and_layout("📭 No NG results yet")
    history_tab_widget.addTab(history_scroll_ng, "NG")
    
    history_layout.addWidget(history_tab_widget)
    
    # Create widgets dict for main window to reference
    widgets_dict = {
        'history_tab_widget': history_tab_widget,
        'history_scroll': history_scroll,
        'history_content': history_content,
        'history_content_layout': history_content_layout,
        'history_content_layout_good': history_content_layout_good,
        'history_content_layout_ng': history_content_layout_ng,
        'history_count_label': history_count_label,
        'btn_clear_history': btn_clear_history,
        'empty_label': empty_label,
        'empty_label_good': empty_label_good,
        'empty_label_ng': empty_label_ng,
    }
    
    return history_tab, widgets_dict


def _expiry_found(entry):
    """True if expiry date was detected (not missing/empty)."""
    expiry = entry.get('expiry_date')
    if expiry is None:
        return False
    if isinstance(expiry, str):
        s = expiry.strip()
        if s in ('', '-'):
            return False
    return True


def is_history_entry_good(entry):
    """True if entry is Good: bottle Good, cap not failed, and expiry date found."""
    if not _expiry_found(entry):
        return False
    bottle_defect = entry.get('bottle_defect')
    cap_status = entry.get('cap_status')
    if bottle_defect == 'NG':
        return False
    if cap_status == 'ไม่ผ่าน':
        return False
    return bottle_defect == 'Good' or (bottle_defect is None and cap_status != 'ไม่ผ่าน')


def is_history_entry_ng(entry):
    """True if entry is NG: bottle NG, cap failed, or expiry date not found."""
    bottle_defect = entry.get('bottle_defect')
    cap_status = entry.get('cap_status')
    if bottle_defect == 'NG':
        return True
    if cap_status == 'ไม่ผ่าน':
        return True
    if not _expiry_found(entry):
        return True
    return False


def create_history_item(bottle_image, cap_image, line_image=None, bottle_type=None, expiry_date=None, timestamp=None, ocr_text="", faded_status=None, total_area=None, num_chars=None, normalized_area=None, bottle_defect=None, bottle_defect_score=None, cap_status=None, gui_instance=None):
    """
    สร้าง history item widget สำหรับแสดงประวัติแต่ละรายการ

    Args:
        bottle_image: numpy array - ภาพขวด
        cap_image: numpy array - ภาพฝา (อาจเป็น None)
        line_image: numpy array - ภาพรวมบรรทัด (สูงสุด 3 บรรทัด, อาจเป็น None)
        bottle_type: str - ประเภทขวด (M100, M110, M120, M130)
        expiry_date: str - วันหมดอายุ
        timestamp: str - เวลาที่ประมวลผล
        ocr_text: str - ข้อความ OCR
        faded_status: str - สถานะจาง ('faded', 'normal', 'unknown', None)
        total_area: int - พื้นที่รวมของตัวอักษรที่อ่านได้
        num_chars: int - จำนวนตัวอักษรที่ตรวจจับได้
        normalized_area: float - พื้นที่ normalized (เปอร์เซ็นต์ของ unified region)
        bottle_defect: str - ขวด Good/NG จาก defect model (None ถ้าไม่มี)
        bottle_defect_score: float - คะแนน defect 0–1 (None ถ้าไม่มี)
        cap_status: str - ฝา ผ่าน/ไม่ผ่าน (None ถ้าไม่มี)
        gui_instance: BottleDetectionGUI instance - สำหรับเรียก show_image_zoom_popup
    
    Returns:
        QFrame: History item widget
    """
    # Create main frame
    item_frame = QFrame()
    item_frame.setStyleSheet("""
        QFrame {
            background-color: white;
            border: 2px solid #bdc3c7;
            border-radius: 10px;
            padding: 15px;
            margin: 5px;
        }
        QFrame:hover {
            border: 2px solid #3498db;
            background-color: #f8f9fa;
        }
    """)
    item_frame.setMinimumHeight(200)
    
    item_layout = QVBoxLayout(item_frame)
    item_layout.setSpacing(10)
    
    # Header with timestamp
    header_layout = QHBoxLayout()
    timestamp_label = QLabel(f"🕐 {timestamp}")
    timestamp_label.setStyleSheet("color: #7f8c8d; font-size: 11px; font-weight: bold;")
    header_layout.addWidget(timestamp_label)
    header_layout.addStretch()
    
    # Bottle type badge (show flavor name instead of M-code)
    flavor = bottle_type_flavor_label(bottle_type)
    bottle_type_label = QLabel(f"🏷️ {flavor}")
    if bottle_type == "M100":
        bottle_type_label.setStyleSheet("""
            background-color: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 12px;
        """)
    elif bottle_type == "M110":
        bottle_type_label.setStyleSheet("""
            background-color: #e67e22;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 12px;
        """)
    elif bottle_type == "M120":
        bottle_type_label.setStyleSheet("""
            background-color: #27ae60;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 12px;
        """)
    elif bottle_type == "M130":
        bottle_type_label.setStyleSheet("""
            background-color: #9b59b6;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 12px;
        """)
    else:
        bottle_type_label.setStyleSheet("""
            background-color: #95a5a6;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 12px;
        """)
    header_layout.addWidget(bottle_type_label)
    item_layout.addLayout(header_layout)
    
    # Content area with images and info
    content_layout = QHBoxLayout()
    
    # Left side - Images
    images_layout = QVBoxLayout()
    
    # Bottle image
    if bottle_image is not None:
        bottle_img_label = QLabel()
        bottle_img_label.setMinimumSize(200, 150)
        bottle_img_label.setMaximumSize(200, 150)
        bottle_img_label.setAlignment(Qt.AlignCenter)
        bottle_img_label.setStyleSheet("border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        
        # Resize image for display
        h, w = bottle_image.shape[:2]
        max_size = 200
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            bottle_image_resized = cv2.resize(bottle_image, (new_w, new_h))
        else:
            bottle_image_resized = bottle_image
        
        # Convert to QPixmap (ใช้ .copy() เพื่อให้ Qt เก็บข้อมูลเอง)
        if len(bottle_image_resized.shape) == 3:
            rgb_image = cv2.cvtColor(np.ascontiguousarray(bottle_image_resized), cv2.COLOR_BGR2RGB)
        else:
            rgb_image = np.ascontiguousarray(bottle_image_resized)
            if len(rgb_image.shape) == 2:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image.copy())
        bottle_img_label.setPixmap(pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Add double-click event for zoom popup
        if gui_instance is not None and bottle_image is not None:
            bottle_img_label.setCursor(Qt.PointingHandCursor)
            # Enable mouse tracking and events
            bottle_img_label.setMouseTracking(True)
            # Store original image for zoom popup
            orig_bottle_img = bottle_image.copy() if hasattr(bottle_image, 'copy') else bottle_image
            # Create function with proper closure
            def bottle_double_click(event):
                try:
                    if orig_bottle_img is not None and gui_instance is not None:
                        gui_instance.show_image_zoom_popup(orig_bottle_img, "ภาพขวด")
                except Exception as e:
                    print(f"❌ Error showing bottle image popup: {e}")
            bottle_img_label.mouseDoubleClickEvent = bottle_double_click
        
        images_layout.addWidget(bottle_img_label)
    
    # Cap image (ช่อง "ภาพฝา" — แสดงภาพฝา/ภาพรวมบรรทัด)
    if cap_image is not None:
        cap_img_label = QLabel()
        cap_img_label.setMinimumSize(200, 150)
        cap_img_label.setMaximumSize(200, 150)
        cap_img_label.setAlignment(Qt.AlignCenter)
        cap_img_label.setStyleSheet("border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        cap_img_label.setScaledContents(True)
        
        # ใช้วิธีเดียวกับ "ภาพบรรทัด" ใน main_window: resize → cvtColor → QImage จาก data แล้วเก็บ ref ที่ label
        img = np.asarray(cap_image, dtype=np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            h, w = 10, 10
        max_side = 200
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w, c = rgb.shape
        bytes_per_line = c * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # บังคับ copy ให้ Qt เก็บ pixel เอง (กัน pointer จาก numpy หลุด)
        if not qimg.isNull():
            qimg = qimg.copy()
        if not qimg.isNull():
            pixmap = QPixmap.fromImage(qimg)
            if not pixmap.isNull():
                cap_img_label.setPixmap(pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        images_layout.addWidget(cap_img_label)
        
        # Add double-click event for zoom popup
        if gui_instance is not None and cap_image is not None:
            cap_img_label.setCursor(Qt.PointingHandCursor)
            # Enable mouse tracking and events
            cap_img_label.setMouseTracking(True)
            # Store original image for zoom popup
            orig_cap_img = cap_image.copy() if hasattr(cap_image, 'copy') else cap_image
            # Create function with proper closure
            def cap_double_click(event):
                try:
                    if orig_cap_img is not None and gui_instance is not None:
                        gui_instance.show_image_zoom_popup(orig_cap_img, "ภาพฝา")
                except Exception as e:
                    print(f"❌ Error showing cap image popup: {e}")
            cap_img_label.mouseDoubleClickEvent = cap_double_click
    
    # Line image (ภาพรวมบรรทัด สูงสุด 3 บรรทัด) — ช่องเดียวกับ cap image
    if line_image is not None:
        line_img_label = QLabel()
        line_img_label.setMinimumSize(200, 150)
        line_img_label.setMaximumSize(200, 150)
        line_img_label.setAlignment(Qt.AlignCenter)
        line_img_label.setStyleSheet("border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        line_img_label.setScaledContents(True)
        
        img = np.asarray(line_image, dtype=np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            h, w = 10, 10
        max_side = 200
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w, c = rgb.shape
        bytes_per_line = c * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        if not qimg.isNull():
            qimg = qimg.copy()
        if not qimg.isNull():
            pixmap = QPixmap.fromImage(qimg)
            if not pixmap.isNull():
                line_img_label.setPixmap(pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Double-click เพื่อซูมภาพบรรทัด
        if gui_instance is not None and line_image is not None:
            line_img_label.setCursor(Qt.PointingHandCursor)
            line_img_label.setMouseTracking(True)
            orig_line_img = line_image.copy() if hasattr(line_image, 'copy') else np.asarray(line_image).copy()
            def line_double_click(event):
                try:
                    if orig_line_img is not None and gui_instance is not None:
                        gui_instance.show_image_zoom_popup(orig_line_img, "ภาพบรรทัด")
                except Exception as e:
                    print(f"❌ Error showing line image popup: {e}")
            line_img_label.mouseDoubleClickEvent = line_double_click
        
        images_layout.addWidget(line_img_label)
    
    content_layout.addLayout(images_layout)
    
    # Right side - Info
    info_layout = QVBoxLayout()
    info_layout.setSpacing(10)
    
    # Bottle type / flavor
    type_info = QLabel(f"<b>Flavor:</b> {flavor}")
    type_info.setStyleSheet("color: #2c3e50; font-size: 14px; padding: 5px;")
    info_layout.addWidget(type_info)
    
    # ขวด Good/NG (พร้อม score) และ ฝา ผ่าน/ไม่ผ่าน
    defect_lines = []
    if bottle_defect is not None:
        score_str = ""
        if bottle_defect_score is not None:
            try:
                score_str = f" <span style='color: #7f8c8d;'>({float(bottle_defect_score):.4f})</span>"
            except (TypeError, ValueError):
                pass
        if bottle_defect == "Good":
            defect_lines.append("🔬 <b>ขวด:</b> <span style='color: #27ae60;'>Good</span>" + score_str)
        else:
            defect_lines.append("🔬 <b>ขวด:</b> <span style='color: #e74c3c;'>NG</span>" + score_str)
    if cap_status is not None:
        if cap_status == "ผ่าน":
            defect_lines.append("📷 <b>ฝา:</b> <span style='color: #27ae60;'>ผ่าน</span>")
        else:
            defect_lines.append("📷 <b>ฝา:</b> <span style='color: #e74c3c;'>ไม่ผ่าน</span>")
    if defect_lines:
        defect_label = QLabel(" | ".join(defect_lines))
        defect_label.setStyleSheet("color: #2c3e50; font-size: 13px; padding: 3px 5px;")
        info_layout.addWidget(defect_label)
    
    # Faded status (สถานะจาง/ไม่จาง) และ Area
    if faded_status is not None:
        if faded_status == 'faded':
            status_text = "⚠️ <b>สถานะ:</b> <span style='color: #e74c3c;'>ข้อความจาง</span>"
            status_color = "#e74c3c"
        elif faded_status == 'normal':
            status_text = "✅ <b>สถานะ:</b> <span style='color: #27ae60;'>ข้อความไม่จาง</span>"
            status_color = "#27ae60"
        else:
            status_text = "❓ <b>สถานะ:</b> <span style='color: #95a5a6;'>ไม่ทราบ</span>"
            status_color = "#95a5a6"
        
        # เพิ่มข้อมูล area และ num_chars
        area_info_text = status_text
        if total_area is not None:
            area_info_text += f"<br>📊 <b>Area:</b> <span style='color: {status_color};'>{total_area}</span>"
        if num_chars is not None:
            area_info_text += f" | <b>ตัวอักษร:</b> <span style='color: {status_color};'>{num_chars}</span>"
        
        faded_info = QLabel(area_info_text)
        faded_info.setStyleSheet("color: #2c3e50; font-size: 14px; padding: 5px; font-weight: bold;")
        info_layout.addWidget(faded_info)
    
    # Expiry date (แสดงบรรทัดที่ 1-3 ที่อ่านได้)
    if expiry_date:
        # Display expiry date (lines 1-3) with line breaks
        expiry_text = expiry_date.replace('\n', '<br>')  # Replace newline with HTML line break
        expiry_info = QLabel(f"<b>วันหมดอายุ:</b><br>{expiry_text}")
        expiry_info.setStyleSheet("color: #27ae60; font-size: 12px; padding: 5px;")
        expiry_info.setWordWrap(True)
        expiry_info.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        info_layout.addWidget(expiry_info)
    else:
        expiry_info = QLabel("<b>วันหมดอายุ:</b> -")
        expiry_info.setStyleSheet("color: #95a5a6; font-size: 14px; padding: 5px;")
        info_layout.addWidget(expiry_info)
    
    # OCR text (if available)
    if ocr_text:
        ocr_info = QLabel(f"<b>ข้อความ OCR:</b><br>{ocr_text[:100]}...")
        ocr_info.setStyleSheet("color: #34495e; font-size: 12px; padding: 5px;")
        ocr_info.setWordWrap(True)
        info_layout.addWidget(ocr_info)
    
    info_layout.addStretch()
    content_layout.addLayout(info_layout)
    
    item_layout.addLayout(content_layout)
    
    return item_frame


def create_history_compact_item(history_entry, gui_instance=None):
    """
    สร้าง history item แบบย่อ ใช้สำหรับรายการเก่า ๆ เพื่อให้ UI เบา
    แสดงแค่เวลา, รสชาติ, สถานะขวด/ฝา และปุ่มดูรายละเอียด (เปิด dialog แยก)
    """
    item_frame = QFrame()
    item_frame.setStyleSheet("""
        QFrame {
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            padding: 8px 10px;
            margin: 4px;
        }
        QFrame:hover {
            border: 1px solid #3498db;
            background-color: #f8f9fa;
        }
    """)
    item_layout = QHBoxLayout(item_frame)
    item_layout.setSpacing(8)
    
    # Left: timestamp + bottle type
    left_layout = QVBoxLayout()
    ts = history_entry.get('timestamp', '')
    bottle_type = history_entry.get('bottle_type', 'Unknown')
    flavor = bottle_type_flavor_label(bottle_type)
    timestamp_label = QLabel(f"🕐 {ts}")
    timestamp_label.setStyleSheet("color: #7f8c8d; font-size: 11px; font-weight: bold;")
    left_layout.addWidget(timestamp_label)
    
    type_label = QLabel(f"🏷️ {flavor}")
    type_label.setStyleSheet("color: #2c3e50; font-size: 12px;")
    left_layout.addWidget(type_label)
    item_layout.addLayout(left_layout)
    
    # Middle: status summary
    middle_layout = QVBoxLayout()
    status_parts = []
    bottle_defect = history_entry.get('bottle_defect')
    cap_status = history_entry.get('cap_status')
    faded_status = history_entry.get('faded_status')
    
    if bottle_defect is not None:
        if bottle_defect == "Good":
            status_parts.append("ขวด: Good")
        else:
            status_parts.append("ขวด: NG")
    if cap_status is not None:
        status_parts.append(f"ฝา: {cap_status}")
    if faded_status is not None:
        if faded_status == 'faded':
            status_parts.append("ข้อความจาง")
        elif faded_status == 'normal':
            status_parts.append("ข้อความไม่จาง")
    
    status_text = " | ".join(status_parts) if status_parts else "รายละเอียดพร้อมดู"
    status_label = QLabel(status_text)
    status_label.setStyleSheet("color: #34495e; font-size: 11px;")
    middle_layout.addWidget(status_label)
    
    ocr_preview = (history_entry.get('ocr_text') or "").strip()
    if ocr_preview:
        if len(ocr_preview) > 50:
            ocr_preview = ocr_preview[:50] + "..."
        ocr_label = QLabel(ocr_preview)
        ocr_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        ocr_label.setWordWrap(True)
        middle_layout.addWidget(ocr_label)
    item_layout.addLayout(middle_layout, 1)
    
    # Right: "ดูรายละเอียด" label (clickable)
    if gui_instance is not None:
        detail_label = QLabel("ดูรายละเอียด ⤵")
        detail_label.setStyleSheet("""
            color: #2980b9;
            font-size: 11px;
            font-weight: bold;
            padding: 4px 8px;
            border: 1px solid #2980b9;
            border-radius: 4px;
        """)
        detail_label.setAlignment(Qt.AlignCenter)
        detail_label.setCursor(Qt.PointingHandCursor)
        
        def open_detail(event):
            try:
                if hasattr(gui_instance, 'show_history_detail'):
                    gui_instance.show_history_detail(history_entry)
            except Exception as e:
                print(f"❌ Error opening history detail: {e}")
        
        detail_label.mouseDoubleClickEvent = open_detail
        detail_label.mousePressEvent = open_detail
        item_layout.addWidget(detail_label)
    
    return item_frame


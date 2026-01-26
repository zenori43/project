# -*- coding: utf-8 -*-
"""
History Tab Component
สร้าง UI สำหรับ History Tab (ประวัติการประมวลผล)
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton,
    QFrame, QGridLayout, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np


def create_history_tab():
    """
    สร้าง History Tab (ประวัติการประมวลผล)
    Returns:
        tuple: (history_tab_widget, widgets_dict)
            - history_tab_widget: QWidget สำหรับ tab
            - widgets_dict: dict ที่เก็บ widgets ทั้งหมดที่ main window ต้องการอ้างอิง
    """
    history_tab = QWidget()
    history_layout = QVBoxLayout(history_tab)
    
    # History Title
    history_title = QLabel("📋 ประวัติการประมวลผล")
    history_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; padding: 15px;")
    history_title.setAlignment(Qt.AlignCenter)
    history_layout.addWidget(history_title)
    
    # Control buttons
    control_layout = QHBoxLayout()
    btn_clear_history = QPushButton("🗑️ ล้างประวัติ")
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
    
    history_count_label = QLabel("จำนวนประวัติ: 0")
    history_count_label.setStyleSheet("color: #7f8c8d; font-size: 12px; padding: 5px;")
    control_layout.addWidget(history_count_label)
    
    history_layout.addLayout(control_layout)
    
    # Create history display area with scroll
    history_scroll = QScrollArea()
    history_scroll.setWidgetResizable(True)
    history_scroll.setMinimumSize(800, 600)
    history_scroll.setStyleSheet("border: 2px solid #34495e; background-color: #ecf0f1;")
    
    # History content widget
    history_content = QWidget()
    history_content_layout = QVBoxLayout(history_content)
    history_content_layout.setAlignment(Qt.AlignTop)
    
    # Empty state label
    empty_label = QLabel("📭 ยังไม่มีประวัติการประมวลผล")
    empty_label.setStyleSheet("color: #95a5a6; font-size: 16px; padding: 50px;")
    empty_label.setAlignment(Qt.AlignCenter)
    history_content_layout.addWidget(empty_label)
    
    history_scroll.setWidget(history_content)
    history_layout.addWidget(history_scroll)
    
    # Create widgets dict for main window to reference
    widgets_dict = {
        'history_scroll': history_scroll,
        'history_content': history_content,
        'history_content_layout': history_content_layout,
        'history_count_label': history_count_label,
        'btn_clear_history': btn_clear_history,
        'empty_label': empty_label
    }
    
    return history_tab, widgets_dict


def create_history_item(bottle_image, cap_image, bottle_type, expiry_date, timestamp, ocr_text="", faded_status=None, total_area=None, num_chars=None, gui_instance=None):
    """
    สร้าง history item widget สำหรับแสดงประวัติแต่ละรายการ
    
    Args:
        bottle_image: numpy array - ภาพขวด
        cap_image: numpy array - ภาพฝา (อาจเป็น None)
        bottle_type: str - ประเภทขวด (M100, M110, M120, M130)
        expiry_date: str - วันหมดอายุ
        timestamp: str - เวลาที่ประมวลผล
        ocr_text: str - ข้อความ OCR
        faded_status: str - สถานะจาง ('faded', 'normal', 'unknown', None)
        total_area: int - พื้นที่รวมของตัวอักษรที่อ่านได้
        num_chars: int - จำนวนตัวอักษรที่ตรวจจับได้
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
    
    # Bottle type badge
    bottle_type_label = QLabel(f"🏷️ {bottle_type}")
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
        
        # Convert to QPixmap
        if len(bottle_image_resized.shape) == 3:
            rgb_image = cv2.cvtColor(bottle_image_resized, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = bottle_image_resized
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        bottle_img_label.setPixmap(pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        bottle_img_title = QLabel("📷 ภาพขวด")
        bottle_img_title.setStyleSheet("color: #34495e; font-size: 10px; font-weight: bold;")
        bottle_img_title.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(bottle_img_title)
        
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
    
    # Cap image
    if cap_image is not None:
        cap_img_label = QLabel()
        cap_img_label.setMinimumSize(200, 150)
        cap_img_label.setMaximumSize(200, 150)
        cap_img_label.setAlignment(Qt.AlignCenter)
        cap_img_label.setStyleSheet("border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        
        # Resize image for display
        h, w = cap_image.shape[:2]
        max_size = 200
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            cap_image_resized = cv2.resize(cap_image, (new_w, new_h))
        else:
            cap_image_resized = cap_image
        
        # Convert to QPixmap
        if len(cap_image_resized.shape) == 3:
            rgb_image = cv2.cvtColor(cap_image_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            # Grayscale image
            h, w = cap_image_resized.shape
            rgb_image = cv2.cvtColor(cap_image_resized, cv2.COLOR_GRAY2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        cap_img_label.setPixmap(pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        cap_img_title = QLabel("📷 ภาพฝา")
        cap_img_title.setStyleSheet("color: #34495e; font-size: 10px; font-weight: bold;")
        cap_img_title.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(cap_img_title)
        
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
        
        images_layout.addWidget(cap_img_label)
    
    content_layout.addLayout(images_layout)
    
    # Right side - Info
    info_layout = QVBoxLayout()
    info_layout.setSpacing(10)
    
    # Bottle type
    type_info = QLabel(f"<b>รสชาติ:</b> {bottle_type}")
    type_info.setStyleSheet("color: #2c3e50; font-size: 14px; padding: 5px;")
    info_layout.addWidget(type_info)
    
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


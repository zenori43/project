# -*- coding: utf-8 -*-
"""
Bottle Detection Tab Component
สร้าง UI สำหรับ Bottle Detection Tab
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, 
    QScrollArea, QTextEdit, QPushButton, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt


def create_bottle_tab():
    """
    สร้าง Bottle Detection Tab
    Returns:
        tuple: (bottle_tab_widget, widgets_dict)
            - bottle_tab_widget: QWidget สำหรับ tab
            - widgets_dict: dict ที่เก็บ widgets ทั้งหมดที่ main window ต้องการอ้างอิง
    """
    bottle_tab = QWidget()
    bottle_layout = QVBoxLayout(bottle_tab)
    
    # Create main content area with splitter for bottle detection
    bottle_splitter = QSplitter(Qt.Horizontal)
    
    # Left side - Image display
    image_widget = QWidget()
    image_layout = QVBoxLayout(image_widget)
    
    image_title = QLabel("ภาพจากกล้อง USB")
    image_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2980b9;")
    image_title.setAlignment(Qt.AlignCenter)
    image_layout.addWidget(image_title)
    
    # Image display area
    image_label = QLabel()
    image_label.setMinimumSize(200, 180)
    image_label.setAlignment(Qt.AlignCenter)
    image_label.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1;")
    image_label.setText("ยังไม่มีภาพจากกล้อง")
    image_layout.addWidget(image_label)
    
    # Current image info
    image_info_label = QLabel("ข้อมูลภาพ: -")
    image_info_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
    image_layout.addWidget(image_info_label)
    
    # ปุ่มก่อนหน้า/ถัดไป (แสดงเมื่อเลือกหลายภาพ)
    nav_row = QHBoxLayout()
    nav_row.addStretch()
    btn_prev_bottle_image = QPushButton('◀ ก่อนหน้า')
    btn_prev_bottle_image.setToolTip("แสดงภาพก่อนหน้าในรายการที่เลือก")
    btn_prev_bottle_image.setVisible(False)
    btn_prev_bottle_image.setEnabled(False)
    btn_prev_bottle_image.setStyleSheet("QPushButton { padding: 6px 12px; font-size: 11px; }")
    nav_row.addWidget(btn_prev_bottle_image)
    btn_next_bottle_image = QPushButton('ถัดไป ▶')
    btn_next_bottle_image.setToolTip("แสดงภาพถัดไปในรายการที่เลือก")
    btn_next_bottle_image.setVisible(False)
    btn_next_bottle_image.setEnabled(False)
    btn_next_bottle_image.setStyleSheet("QPushButton { padding: 6px 12px; font-size: 11px; }")
    nav_row.addWidget(btn_next_bottle_image)
    nav_row.addStretch()
    image_layout.addLayout(nav_row)
    
    bottle_splitter.addWidget(image_widget)
    
    # Middle - Cropped images display
    crops_widget = QWidget()
    crops_layout = QVBoxLayout(crops_widget)
    
    crops_title = QLabel("ภาพที่ครอปและผล OCR")
    crops_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e67e22;")
    crops_title.setAlignment(Qt.AlignCenter)
    crops_layout.addWidget(crops_title)
    
    # Scroll area for cropped images
    crops_scroll = QScrollArea()
    crops_scroll.setWidgetResizable(True)
    crops_scroll.setMinimumSize(160, 200)
    crops_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    crops_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    crops_scroll.setStyleSheet("border: 2px solid #e67e22; background-color: #fef9e7;")
    
    # Container for cropped images
    crops_container = QWidget()
    crops_container_layout = QVBoxLayout(crops_container)
    crops_container_layout.setAlignment(Qt.AlignTop)
    
    # Add placeholder text
    placeholder_label = QLabel("ยังไม่มีภาพที่ครอป")
    placeholder_label.setAlignment(Qt.AlignCenter)
    placeholder_label.setStyleSheet("color: #7f8c8d; padding: 20px;")
    crops_container_layout.addWidget(placeholder_label)
    
    crops_scroll.setWidget(crops_container)
    crops_layout.addWidget(crops_scroll)
    
    bottle_splitter.addWidget(crops_widget)
    
    # Right side - Results
    results_widget = QWidget()
    results_layout = QVBoxLayout(results_widget)
    
    results_title = QLabel("ผลการตรวจจับและ OCR")
    results_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c;")
    results_title.setAlignment(Qt.AlignCenter)
    results_layout.addWidget(results_title)
    
    # Results display
    results_text = QTextEdit()
    results_text.setReadOnly(True)
    results_text.setMinimumSize(160, 180)
    results_text.setLineWrapMode(QTextEdit.WidgetWidth)
    results_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    results_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    results_layout.addWidget(results_text)
    
    bottle_splitter.addWidget(results_widget)
    bottle_splitter.setStretchFactor(0, 1)
    bottle_splitter.setStretchFactor(1, 1)
    bottle_splitter.setStretchFactor(2, 1)

    # Add control buttons for bottle detection
    bottle_control_group = QGroupBox("Bottle operations")
    bottle_control_layout = QHBoxLayout()
    
    # Capture from USB camera (manual)
    btn_capture_usb = QPushButton('📸 Capture bottle')
    btn_capture_usb.setToolTip("Capture image from USB camera (bottle only)")
    btn_capture_usb.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #2196F3; color: white; border-radius: 5px; } QPushButton:hover { background-color: #1976D2; }")
    bottle_control_layout.addWidget(btn_capture_usb)
    
    # Process bottle button
    btn_process_bottle = QPushButton('🔍 Process bottle')
    btn_process_bottle.setEnabled(False)  # Disabled by default, enabled when image is available
    btn_process_bottle.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #27ae60; color: white; border-radius: 5px; }")
    bottle_control_layout.addWidget(btn_process_bottle)
    
    # Process all selected images button (shown when multiple images are selected)
    btn_process_all_bottle = QPushButton('🔍 Process all')
    btn_process_all_bottle.setEnabled(False)
    btn_process_all_bottle.setVisible(False)
    btn_process_all_bottle.setToolTip("Process all selected images at once (visible when multiple images are selected)")
    btn_process_all_bottle.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #16a085; color: white; border-radius: 5px; }")
    bottle_control_layout.addWidget(btn_process_all_bottle)
    
    # Save image button
    btn_save_bottle_image = QPushButton('💾 Save bottle image')
    btn_save_bottle_image.setEnabled(False)
    btn_save_bottle_image.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #9b59b6; color: white; border-radius: 5px; }")
    bottle_control_layout.addWidget(btn_save_bottle_image)
    
    # Select image file button for bottle detection
    btn_select_bottle_image = QPushButton('📁 Select bottle image file')
    btn_select_bottle_image.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #3498db; color: white; border-radius: 5px; }")
    bottle_control_layout.addWidget(btn_select_bottle_image)
    
    # Select multiple images button for batch processing
    btn_select_multiple_bottle_images = QPushButton('📁 Select multiple bottle images')
    btn_select_multiple_bottle_images.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #16a085; color: white; border-radius: 5px; }")
    btn_select_multiple_bottle_images.setToolTip("Select multiple files for batch processing and adding to history")
    bottle_control_layout.addWidget(btn_select_multiple_bottle_images)
    
    bottle_control_group.setLayout(bottle_control_layout)
    # Make Bottle operations panel hug its buttons (fixed height) and not stretch vertically
    bottle_control_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    bottle_layout.addWidget(bottle_control_group)
    
    bottle_layout.addWidget(bottle_splitter)
    
    # Create widgets dict for main window to reference
    widgets_dict = {
        'bottle_splitter': bottle_splitter,
        'image_label': image_label,
        'image_info_label': image_info_label,
        'btn_prev_bottle_image': btn_prev_bottle_image,
        'btn_next_bottle_image': btn_next_bottle_image,
        'crops_scroll': crops_scroll,
        'crops_container': crops_container,
        'crops_layout': crops_container_layout,
        'results_text': results_text,
        'btn_capture_usb': btn_capture_usb,
        'btn_process_bottle': btn_process_bottle,
        'btn_process_all_bottle': btn_process_all_bottle,
        'btn_save_bottle_image': btn_save_bottle_image,
        'btn_select_bottle_image': btn_select_bottle_image,
        'btn_select_multiple_bottle_images': btn_select_multiple_bottle_images
    }
    
    return bottle_tab, widgets_dict


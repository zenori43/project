# -*- coding: utf-8 -*-
"""
Home Tab Component
สร้าง UI สำหรับ Home Tab ที่แสดงผลการตรวจจับทั้งขวดและฝา
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, 
    QScrollArea, QTextEdit, QGroupBox, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets


def create_home_tab():
    """
    สร้าง Home Tab ที่แสดงผลการตรวจจับทั้งขวดและฝา
    Returns:
        tuple: (home_tab_widget, widgets_dict)
            - home_tab_widget: QWidget สำหรับ tab
            - widgets_dict: dict ที่เก็บ widgets ทั้งหมดที่ main window ต้องการอ้างอิง
    """
    home_tab = QWidget()
    home_layout = QVBoxLayout(home_tab)
    home_layout.setContentsMargins(10, 10, 10, 10)
    home_layout.setSpacing(10)
    
    # Title
    title_label = QLabel("🏠 หน้าหลัก - การตรวจจับขวดและฝา")
    title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50; padding: 10px;")
    title_label.setAlignment(Qt.AlignCenter)
    home_layout.addWidget(title_label)
    
    # Create main splitter (vertical) to separate bottle and cap sections
    main_splitter = QSplitter(Qt.Vertical)
    
    # ========== Top Section: Bottle Detection ==========
    bottle_group = QGroupBox("🔍 การตรวจจับขวด (Bottle Detection)")
    bottle_group.setStyleSheet("""
        QGroupBox {
            font-size: 16px; 
            font-weight: bold; 
            color: #2980b9; 
            border: 2px solid #2980b9;
            border-radius: 5px;
            margin-top: auto;
            padding-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    bottle_group_layout = QVBoxLayout(bottle_group)
    bottle_group_layout.setContentsMargins(10, 20, 10, 10)
    
    # Bottle splitter (horizontal)
    bottle_splitter = QSplitter(Qt.Horizontal)
    
    # Left: USB Camera Image
    bottle_image_widget = QWidget()
    bottle_image_layout = QVBoxLayout(bottle_image_widget)
    bottle_image_layout.setContentsMargins(5, 5, 5, 5)
    
    bottle_image_title = QLabel("ภาพจากกล้อง USB")
    bottle_image_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2980b9;")
    bottle_image_title.setAlignment(Qt.AlignCenter)
    bottle_image_layout.addWidget(bottle_image_title)
    
    bottle_image_label = QLabel()
    bottle_image_label.setMinimumSize(360, 240)
    bottle_image_label.setAlignment(Qt.AlignCenter)
    bottle_image_label.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1; border-radius: 5px;")
    bottle_image_label.setText("ยังไม่มีภาพจากกล้อง USB")
    bottle_image_label.setScaledContents(False)
    bottle_image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    bottle_image_layout.addWidget(bottle_image_label)
    
    bottle_image_info_label = QLabel("ข้อมูลภาพ: -")
    bottle_image_info_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 11px;")
    bottle_image_layout.addWidget(bottle_image_info_label)
    
    bottle_splitter.addWidget(bottle_image_widget)
    
    # Middle: Bottle Cropped Images
    bottle_crops_widget = QWidget()
    bottle_crops_layout = QVBoxLayout(bottle_crops_widget)
    bottle_crops_layout.setContentsMargins(5, 5, 5, 5)
    
    bottle_crops_title = QLabel("ภาพที่ครอป (Type Regions)")
    bottle_crops_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e67e22;")
    bottle_crops_title.setAlignment(Qt.AlignCenter)
    bottle_crops_layout.addWidget(bottle_crops_title)
    
    bottle_crops_scroll = QScrollArea()
    bottle_crops_scroll.setWidgetResizable(True)
    bottle_crops_scroll.setMinimumSize(240, 220)
    bottle_crops_scroll.setStyleSheet("border: 2px solid #e67e22; background-color: #fef9e7; border-radius: 5px;")
    
    bottle_crops_container = QWidget()
    bottle_crops_container_layout = QVBoxLayout(bottle_crops_container)
    bottle_crops_container_layout.setAlignment(Qt.AlignTop)
    
    bottle_crops_placeholder = QLabel("ยังไม่มีภาพที่ครอป")
    bottle_crops_placeholder.setAlignment(Qt.AlignCenter)
    bottle_crops_placeholder.setStyleSheet("color: #7f8c8d; padding: 20px;")
    bottle_crops_container_layout.addWidget(bottle_crops_placeholder)
    
    bottle_crops_scroll.setWidget(bottle_crops_container)
    bottle_crops_layout.addWidget(bottle_crops_scroll)
    
    bottle_splitter.addWidget(bottle_crops_widget)
    
    # Right: Bottle Results
    bottle_results_widget = QWidget()
    bottle_results_layout = QVBoxLayout(bottle_results_widget)
    bottle_results_layout.setContentsMargins(5, 5, 5, 5)
    
    bottle_results_title = QLabel("ผลการตรวจจับและ OCR")
    bottle_results_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c;")
    bottle_results_title.setAlignment(Qt.AlignCenter)
    bottle_results_layout.addWidget(bottle_results_title)
    
    # ป้ายผลขวด (NG/Good) ให้เห็นชัดเจน
    bottle_verdict_label = QLabel("—")
    bottle_verdict_label.setMinimumHeight(56)
    bottle_verdict_label.setAlignment(Qt.AlignCenter)
    bottle_verdict_label.setStyleSheet("""
        font-size: 22px; font-weight: bold; padding: 10px;
        border-radius: 8px; background-color: #ecf0f1; color: #7f8c8d;
    """)
    bottle_results_layout.addWidget(bottle_verdict_label)
    
    bottle_results_text = QTextEdit()
    bottle_results_text.setReadOnly(True)
    bottle_results_text.setMinimumSize(280, 220)
    bottle_results_text.setStyleSheet("border: 2px solid #e74c3c; background-color: #fff; border-radius: 5px; font-size: 11px;")
    bottle_results_layout.addWidget(bottle_results_text)
    
    bottle_splitter.addWidget(bottle_results_widget)
    bottle_splitter.setSizes([400, 250, 300])
    
    bottle_group_layout.addWidget(bottle_splitter)
    main_splitter.addWidget(bottle_group)
    
    # ========== Bottom Section: Cap Detection ==========
    cap_group = QGroupBox("🔍 การตรวจจับฝา (Cap Detection)")
    cap_group.setStyleSheet("""
        QGroupBox {
            font-size: 16px; 
            font-weight: bold; 
            color: #8e44ad; 
            border: 2px solid #8e44ad;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    cap_group_layout = QVBoxLayout(cap_group)
    cap_group_layout.setContentsMargins(10, 20, 10, 10)
    
    # Cap splitter (horizontal)
    cap_splitter = QSplitter(Qt.Horizontal)
    
    # Left: Sentech Camera Image
    cap_image_widget = QWidget()
    cap_image_layout = QVBoxLayout(cap_image_widget)
    cap_image_layout.setContentsMargins(5, 5, 5, 5)
    
    cap_image_title = QLabel("ภาพจากกล้อง Sentech")
    cap_image_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #8e44ad;")
    cap_image_title.setAlignment(Qt.AlignCenter)
    cap_image_layout.addWidget(cap_image_title)
    
    cap_image_label = QLabel()
    cap_image_label.setMinimumSize(360, 240)
    cap_image_label.setAlignment(Qt.AlignCenter)
    cap_image_label.setStyleSheet("border: 2px solid #8e44ad; background-color: #f4f3f4; border-radius: 5px;")
    cap_image_label.setText("ยังไม่มีภาพจากกล้อง Sentech")
    cap_image_label.setScaledContents(False)
    cap_image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    cap_image_layout.addWidget(cap_image_label)
    
    cap_image_info_label = QLabel("ข้อมูลภาพ: -")
    cap_image_info_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 11px;")
    cap_image_layout.addWidget(cap_image_info_label)
    
    cap_splitter.addWidget(cap_image_widget)
    
    # Middle: Cap Detection Results
    cap_results_widget = QWidget()
    cap_results_layout = QVBoxLayout(cap_results_widget)
    cap_results_layout.setContentsMargins(5, 5, 5, 5)
    
    cap_results_title = QLabel("ผลการตรวจจับฝาและข้อความ")
    cap_results_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e67e22;")
    cap_results_title.setAlignment(Qt.AlignCenter)
    cap_results_layout.addWidget(cap_results_title)
    
    cap_results_scroll = QScrollArea()
    cap_results_scroll.setWidgetResizable(True)
    cap_results_scroll.setMinimumSize(280, 220)
    cap_results_scroll.setStyleSheet("border: 2px solid #e67e22; background-color: #fef9e7; border-radius: 5px;")
    
    cap_results_container = QWidget()
    cap_results_container_layout = QVBoxLayout(cap_results_container)
    cap_results_container_layout.setAlignment(Qt.AlignTop)
    
    cap_results_placeholder = QLabel("ยังไม่มีผลการตรวจจับฝา")
    cap_results_placeholder.setAlignment(Qt.AlignCenter)
    cap_results_placeholder.setStyleSheet("color: #7f8c8d; padding: 20px;")
    cap_results_container_layout.addWidget(cap_results_placeholder)
    
    cap_results_scroll.setWidget(cap_results_container)
    cap_results_layout.addWidget(cap_results_scroll)
    
    cap_splitter.addWidget(cap_results_widget)
    
    # Right: Cap Detection Text Results
    cap_text_widget = QWidget()
    cap_text_layout = QVBoxLayout(cap_text_widget)
    cap_text_layout.setContentsMargins(5, 5, 5, 5)
    
    cap_text_title = QLabel("ผลการตรวจสอบฝา")
    cap_text_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c;")
    cap_text_title.setAlignment(Qt.AlignCenter)
    cap_text_layout.addWidget(cap_text_title)
    
    # ป้ายผลฝา (PASS/NG) ให้เห็นชัดเจน
    cap_verdict_label = QLabel("—")
    cap_verdict_label.setMinimumHeight(56)
    cap_verdict_label.setAlignment(Qt.AlignCenter)
    cap_verdict_label.setStyleSheet("""
        font-size: 22px; font-weight: bold; padding: 10px;
        border-radius: 8px; background-color: #ecf0f1; color: #7f8c8d;
    """)
    cap_text_layout.addWidget(cap_verdict_label)
    
    cap_detection_text = QTextEdit()
    cap_detection_text.setReadOnly(True)
    cap_detection_text.setMinimumSize(280, 220)
    cap_detection_text.setStyleSheet("border: 2px solid #e74c3c; background-color: #fff; border-radius: 5px; font-size: 11px;")
    cap_text_layout.addWidget(cap_detection_text)
    
    cap_splitter.addWidget(cap_text_widget)
    cap_splitter.setSizes([360, 260, 260])
    
    cap_group_layout.addWidget(cap_splitter)
    main_splitter.addWidget(cap_group)
    
    # Set main splitter sizes (ลดความสูงแต่ละส่วนให้พอดีหน้าจอ)
    main_splitter.setSizes([420, 360])
    
    home_layout.addWidget(main_splitter)
    
    # Create widgets dict for main window to reference
    widgets_dict = {
        'main_splitter': main_splitter,
        'bottle_splitter': bottle_splitter,
        'bottle_image_label': bottle_image_label,
        'bottle_image_info_label': bottle_image_info_label,
        'bottle_crops_scroll': bottle_crops_scroll,
        'bottle_crops_container': bottle_crops_container,
        'bottle_crops_layout': bottle_crops_container_layout,
        'bottle_verdict_label': bottle_verdict_label,
        'bottle_results_text': bottle_results_text,
        'cap_splitter': cap_splitter,
        'cap_image_label': cap_image_label,
        'cap_image_info_label': cap_image_info_label,
        'cap_results_scroll': cap_results_scroll,
        'cap_results_container': cap_results_container,
        'cap_results_layout': cap_results_container_layout,
        'cap_verdict_label': cap_verdict_label,
        'cap_detection_text': cap_detection_text
    }
    
    return home_tab, widgets_dict

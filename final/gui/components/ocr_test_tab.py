# -*- coding: utf-8 -*-
"""
OCR Test Tab Component
สร้าง UI สำหรับ OCR Test Tab
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSplitter, QPushButton, 
    QGroupBox, QTextEdit, QProgressBar
)
from PyQt5.QtCore import Qt


def create_ocr_test_tab():
    """
    สร้าง OCR Test Tab
    Returns:
        tuple: (ocr_test_tab_widget, widgets_dict)
            - ocr_test_tab_widget: QWidget สำหรับ tab
            - widgets_dict: dict ที่เก็บ widgets ทั้งหมดที่ main window ต้องการอ้างอิง
    """
    ocr_test_tab = QWidget()
    ocr_test_layout = QVBoxLayout(ocr_test_tab)
    
    # OCR Test Title
    ocr_test_title = QLabel("🔤 ทดสอบ OCR")
    ocr_test_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; padding: 15px;")
    ocr_test_title.setAlignment(Qt.AlignCenter)
    ocr_test_layout.addWidget(ocr_test_title)
    
    # Create splitter for OCR test
    ocr_test_splitter = QSplitter(Qt.Horizontal)
    
    # Left side - Image selection and display
    ocr_image_widget = QWidget()
    ocr_image_layout = QVBoxLayout(ocr_image_widget)
    
    # Image selection controls
    ocr_controls_group = QGroupBox("เลือกภาพ")
    ocr_controls_layout = QVBoxLayout(ocr_controls_group)
    
    # Select image button
    btn_select_ocr_image = QPushButton('📁 เลือกภาพสำหรับทดสอบ OCR')
    btn_select_ocr_image.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #3498db; color: white; }")
    ocr_controls_layout.addWidget(btn_select_ocr_image)
    
    # Process OCR button
    btn_process_ocr = QPushButton('🔤 ประมวลผล OCR')
    btn_process_ocr.setEnabled(False)
    btn_process_ocr.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #27ae60; color: white; }")
    ocr_controls_layout.addWidget(btn_process_ocr)
    
    # Enhanced OCR processing button
    btn_process_ocr_enhanced = QPushButton('🔤 ประมวลผล OCR (ปรับปรุง)')
    btn_process_ocr_enhanced.setEnabled(False)
    btn_process_ocr_enhanced.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #FF9800; color: white; }")
    ocr_controls_layout.addWidget(btn_process_ocr_enhanced)
    
    # OCR status label
    ocr_status_label = QLabel('⏸️ เลือกภาพเพื่อเริ่มทดสอบ OCR')
    ocr_status_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 11px;")
    ocr_controls_layout.addWidget(ocr_status_label)
    
    ocr_controls_group.setLayout(ocr_controls_layout)
    ocr_image_layout.addWidget(ocr_controls_group)
    
    # Image display
    ocr_image_label = QLabel("ไม่มีภาพ")
    ocr_image_label.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1; color: #7f8c8d;")
    ocr_image_label.setAlignment(Qt.AlignCenter)
    ocr_image_label.setMinimumHeight(300)
    ocr_image_layout.addWidget(ocr_image_label)
    
    # Right side - OCR results
    ocr_results_widget = QWidget()
    ocr_results_layout = QVBoxLayout(ocr_results_widget)
    
    # OCR results title
    ocr_results_title = QLabel("ผลลัพธ์ OCR")
    ocr_results_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c;")
    ocr_results_title.setAlignment(Qt.AlignCenter)
    ocr_results_layout.addWidget(ocr_results_title)
    
    # OCR results display
    ocr_results_text = QTextEdit()
    ocr_results_text.setReadOnly(True)
    ocr_results_text.setStyleSheet("""
        QTextEdit {
            background-color: #f8f9fa;
            border: 2px solid #e74c3c;
            border-radius: 5px;
            padding: 10px;
            font-size: 12px;
            color: #2c3e50;
        }
    """)
    ocr_results_text.setPlaceholderText("ผลลัพธ์ OCR จะแสดงที่นี่...")
    ocr_results_layout.addWidget(ocr_results_text)
    
    # OCR processing progress
    ocr_progress_bar = QProgressBar()
    ocr_progress_bar.setVisible(False)
    ocr_progress_bar.setStyleSheet("QProgressBar { border: 2px solid #e74c3c; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #e74c3c; }")
    ocr_results_layout.addWidget(ocr_progress_bar)
    
    # Add widgets to splitter
    ocr_test_splitter.addWidget(ocr_image_widget)
    ocr_test_splitter.addWidget(ocr_results_widget)
    
    # Set splitter sizes
    ocr_test_splitter.setSizes([400, 400])
    
    ocr_test_layout.addWidget(ocr_test_splitter)
    
    # Create widgets dict for main window to reference
    widgets_dict = {
        'ocr_test_splitter': ocr_test_splitter,
        'btn_select_ocr_image': btn_select_ocr_image,
        'btn_process_ocr': btn_process_ocr,
        'btn_process_ocr_enhanced': btn_process_ocr_enhanced,
        'ocr_status_label': ocr_status_label,
        'ocr_image_label': ocr_image_label,
        'ocr_results_text': ocr_results_text,
        'ocr_progress_bar': ocr_progress_bar
    }
    
    return ocr_test_tab, widgets_dict


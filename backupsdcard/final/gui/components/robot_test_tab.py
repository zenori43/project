# -*- coding: utf-8 -*-
"""
Modbus Register Test Tab Component
สร้าง UI สำหรับทดสอบการส่งค่าไปที่ Register D ในระบบ
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QGroupBox, QTextEdit, QSpinBox, QComboBox
)
from PyQt5.QtCore import Qt


def create_robot_test_tab():
    """
    สร้าง Modbus Register Test Tab
    Returns:
        tuple: (robot_test_tab_widget, widgets_dict)
            - robot_test_tab_widget: QWidget สำหรับ tab
            - widgets_dict: dict ที่เก็บ widgets ทั้งหมดที่ main window ต้องการอ้างอิง
    """
    robot_test_tab = QWidget()
    robot_test_layout = QVBoxLayout(robot_test_tab)
    
    # Title
    title = QLabel("🔧 ทดสอบ Register D (Modbus)")
    title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; padding: 15px;")
    title.setAlignment(Qt.AlignCenter)
    robot_test_layout.addWidget(title)
    
    # Info label
    info_label = QLabel("ใช้ Modbus connection ที่มีอยู่ในระบบ")
    info_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 11px;")
    info_label.setAlignment(Qt.AlignCenter)
    robot_test_layout.addWidget(info_label)
    
    # Register Settings Group
    register_group = QGroupBox("📝 เลือก Register D")
    register_layout = QVBoxLayout(register_group)
    
    # Register selection dropdown
    register_select_layout = QHBoxLayout()
    register_label = QLabel("Register:")
    register_label.setMinimumWidth(120)
    register_combo = QComboBox()
    register_combo.addItem("D5012 - RESET Button", 5012)
    register_combo.addItem("D5500 - Mode Selection", 5500)
    register_combo.addItem("D9002 - Two Taste Mode", 9002)
    register_combo.addItem("D9006 - Taste Mode (1/2/3)", 9006)
    register_combo.addItem("D7007 - One Taste Mode", 7007)
    register_combo.addItem("D7009 - Bottle Type Result", 7009)
    register_combo.addItem("Custom (ระบุเอง)", 0)
    register_select_layout.addWidget(register_label)
    register_select_layout.addWidget(register_combo)
    register_layout.addLayout(register_select_layout)
    
    # Address input (support both hex and decimal) - shown when Custom is selected
    address_layout = QHBoxLayout()
    address_label = QLabel("Address (Custom):")
    address_label.setMinimumWidth(120)
    address_input = QLineEdit("")
    address_input.setPlaceholderText("เช่น 5012 หรือ 0x1394")
    address_input.setEnabled(False)
    address_help = QLabel("(Decimal หรือ Hex)")
    address_help.setStyleSheet("color: #7f8c8d; font-size: 10px;")
    address_layout.addWidget(address_label)
    address_layout.addWidget(address_input)
    address_layout.addWidget(address_help)
    register_layout.addLayout(address_layout)
    
    # Value input with description
    value_layout = QVBoxLayout()
    value_header_layout = QHBoxLayout()
    value_label = QLabel("Value:")
    value_label.setMinimumWidth(120)
    value_input = QSpinBox()
    value_input.setRange(0, 65535)
    value_input.setValue(1)
    value_header_layout.addWidget(value_label)
    value_header_layout.addWidget(value_input)
    value_header_layout.addStretch()
    value_layout.addLayout(value_header_layout)
    
    # Value description
    value_desc_label = QLabel("D5012: 1=Press, 0=Release | D5500: 7=ID7, 5=ID5, 8=ID8, 9=ID9 refill")
    value_desc_label.setStyleSheet("color: #7f8c8d; font-size: 10px; padding-left: 120px;")
    value_desc_label.setWordWrap(True)
    value_layout.addWidget(value_desc_label)
    register_layout.addLayout(value_layout)
    
    robot_test_layout.addWidget(register_group)
    
    # Control Buttons Group
    control_group = QGroupBox("🎮 การควบคุม")
    control_layout = QVBoxLayout(control_group)
    
    # Write and Read buttons
    btn_layout = QHBoxLayout()
    btn_write = QPushButton("✍️ เขียนค่า (Write)")
    btn_write.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #3498db; color: white; }")
    btn_read = QPushButton("📖 อ่านค่า (Read)")
    btn_read.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #9b59b6; color: white; }")
    btn_write_read = QPushButton("🔄 เขียนและอ่าน (Write & Read)")
    btn_write_read.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #f39c12; color: white; }")
    btn_layout.addWidget(btn_write)
    btn_layout.addWidget(btn_read)
    btn_layout.addWidget(btn_write_read)
    control_layout.addLayout(btn_layout)
    
    robot_test_layout.addWidget(control_group)
    
    # Results Group
    results_group = QGroupBox("📊 ผลลัพธ์")
    results_layout = QVBoxLayout(results_group)
    
    results_text = QTextEdit()
    results_text.setReadOnly(True)
    results_text.setMinimumHeight(200)
    results_text.setStyleSheet("background-color: #2c3e50; color: #ecf0f1; font-family: monospace; font-size: 11px;")
    results_text.setPlaceholderText("ผลลัพธ์จะแสดงที่นี่...")
    results_layout.addWidget(results_text)
    
    # Clear button
    btn_clear = QPushButton("🗑️ ล้างผลลัพธ์")
    btn_clear.setStyleSheet("QPushButton { padding: 5px; font-size: 11px; }")
    results_layout.addWidget(btn_clear)
    
    robot_test_layout.addWidget(results_group)
    
    # Add stretch to push everything to top
    robot_test_layout.addStretch()
    
    # Create widgets dict
    widgets_dict = {
        'robot_register_combo': register_combo,
        'robot_address_input': address_input,
        'robot_value_input': value_input,
        'robot_value_desc_label': value_desc_label,
        'robot_btn_write': btn_write,
        'robot_btn_read': btn_read,
        'robot_btn_write_read': btn_write_read,
        'robot_results_text': results_text,
        'robot_btn_clear': btn_clear,
    }
    
    return robot_test_tab, widgets_dict


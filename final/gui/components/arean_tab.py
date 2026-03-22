# -*- coding: utf-8 -*-
"""
Area Navigation Tab Component
สร้าง UI สำหรับ Area Navigation Tab (สั่งหุ่นยนต์ไปตามช่อง)
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, 
    QPushButton, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


def create_arean_tab():
    """
    สร้าง Area Navigation Tab
    Returns:
        tuple: (arean_tab_widget, widgets_dict)
            - arean_tab_widget: QWidget สำหรับ tab
            - widgets_dict: dict ที่เก็บ widgets ทั้งหมดที่ main window ต้องการอ้างอิง
    """
    arean_tab = QWidget()
    arean_layout = QVBoxLayout(arean_tab)
    arean_layout.setContentsMargins(10, 10, 10, 10)
    arean_layout.setSpacing(15)
    
    # Title
    title_label = QLabel("📍 Area Navigation - สั่งหุ่นยนต์ไปตามช่อง")
    title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; padding: 10px;")
    title_label.setAlignment(Qt.AlignCenter)
    arean_layout.addWidget(title_label)
    
    # Create horizontal layout for two tables
    tables_layout = QHBoxLayout()
    tables_layout.setSpacing(20)
    
    # Left Side Table
    left_group = QGroupBox("Left Side")
    left_group.setStyleSheet("QGroupBox { font-weight: bold; color: #3498db; font-size: 14px; padding: 10px; margin: 5px; }")
    left_layout = QVBoxLayout(left_group)
    left_layout.setContentsMargins(10, 15, 10, 10)
    left_layout.setSpacing(10)
    
    left_info = QLabel("M900 - M908")
    left_info.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 5px;")
    left_info.setAlignment(Qt.AlignCenter)
    left_layout.addWidget(left_info)
    
    left_grid = QGridLayout()
    left_grid.setSpacing(5)
    left_buttons = {}
    
    # Create 3x3 grid for left side (M900-M908)
    for row in range(3):
        for col in range(3):
            index = row * 3 + col
            m_code = 900 + index  # M900 to M908
            
            btn = QPushButton(f"M{m_code}")
            btn.setMinimumSize(80, 80)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ecf0f1;
                    border: 2px solid #bdc3c7;
                    border-radius: 5px;
                    font-size: 12px;
                    font-weight: bold;
                    color: #2c3e50;
                }
                QPushButton:hover {
                    background-color: #3498db;
                    color: white;
                    border: 2px solid #2980b9;
                }
                QPushButton:pressed {
                    background-color: #2980b9;
                    color: white;
                }
            """)
            
            left_grid.addWidget(btn, row, col)
            left_buttons[f'btn_m{m_code}'] = btn
    
    left_layout.addLayout(left_grid)
    tables_layout.addWidget(left_group)
    
    # Right Side Table
    right_group = QGroupBox("Right Side")
    right_group.setStyleSheet("QGroupBox { font-weight: bold; color: #e74c3c; font-size: 14px; padding: 10px; margin: 5px; }")
    right_layout = QVBoxLayout(right_group)
    right_layout.setContentsMargins(10, 15, 10, 10)
    right_layout.setSpacing(10)
    
    right_info = QLabel("M909 - M917")
    right_info.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 5px;")
    right_info.setAlignment(Qt.AlignCenter)
    right_layout.addWidget(right_info)
    
    right_grid = QGridLayout()
    right_grid.setSpacing(5)
    right_buttons = {}
    
    # Create 3x3 grid for right side (M909-M917)
    for row in range(3):
        for col in range(3):
            index = row * 3 + col
            m_code = 909 + index  # M909 to M917
            
            btn = QPushButton(f"M{m_code}")
            btn.setMinimumSize(80, 80)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ecf0f1;
                    border: 2px solid #bdc3c7;
                    border-radius: 5px;
                    font-size: 12px;
                    font-weight: bold;
                    color: #2c3e50;
                }
                QPushButton:hover {
                    background-color: #e74c3c;
                    color: white;
                    border: 2px solid #c0392b;
                }
                QPushButton:pressed {
                    background-color: #c0392b;
                    color: white;
                }
            """)
            
            right_grid.addWidget(btn, row, col)
            right_buttons[f'btn_m{m_code}'] = btn
    
    right_layout.addLayout(right_grid)
    tables_layout.addWidget(right_group)
    
    arean_layout.addLayout(tables_layout)
    arean_layout.addStretch()
    
    # Create widgets dict for main window to reference
    widgets_dict = {
        **left_buttons,
        **right_buttons
    }
    
    return arean_tab, widgets_dict

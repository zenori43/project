# -*- coding: utf-8 -*-
"""
Settings Tab Component
สร้าง UI สำหรับ Settings Tab
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, 
    QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt


def create_settings_tab():
    """
    สร้าง Settings Tab
    Returns:
        tuple: (settings_tab_widget, widgets_dict)
            - settings_tab_widget: QWidget สำหรับ tab
            - widgets_dict: dict ที่เก็บ widgets ทั้งหมดที่ main window ต้องการอ้างอิง
    """
    settings_tab = QWidget()
    settings_layout = QVBoxLayout(settings_tab)
    settings_layout.setContentsMargins(10, 10, 10, 10)
    settings_layout.setSpacing(10)
    
    # Crop Settings Group
    crop_settings_group = QGroupBox("⚙️ การตั้งค่าการครอปภาพบรรทัด")
    crop_settings_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
    crop_settings_layout = QVBoxLayout(crop_settings_group)
    crop_settings_layout.setContentsMargins(10, 15, 10, 10)
    crop_settings_layout.setSpacing(15)
    
    # Padding setting
    padding_layout = QHBoxLayout()
    padding_label = QLabel("Padding (px):")
    padding_label.setStyleSheet("color: #2c3e50; font-size: 12px;")
    padding_label.setFixedWidth(150)
    padding_layout.addWidget(padding_label)
    
    padding_spinbox = QSpinBox()
    padding_spinbox.setMinimum(0)
    padding_spinbox.setMaximum(20)
    padding_spinbox.setValue(0)
    padding_spinbox.setSuffix(" px")
    padding_spinbox.setStyleSheet("padding: 5px; font-size: 12px;")
    padding_layout.addWidget(padding_spinbox)
    padding_layout.addStretch()
    crop_settings_layout.addLayout(padding_layout)
    
    # Shrink X Percent setting
    shrink_x_layout = QHBoxLayout()
    shrink_x_label = QLabel("Shrink X (%):")
    shrink_x_label.setStyleSheet("color: #2c3e50; font-size: 12px;")
    shrink_x_label.setFixedWidth(150)
    shrink_x_layout.addWidget(shrink_x_label)
    
    shrink_x_spinbox = QDoubleSpinBox()
    shrink_x_spinbox.setMinimum(0.0)
    shrink_x_spinbox.setMaximum(10.0)
    shrink_x_spinbox.setValue(2.0)
    shrink_x_spinbox.setSingleStep(0.1)
    shrink_x_spinbox.setDecimals(1)
    shrink_x_spinbox.setSuffix("%")
    shrink_x_spinbox.setStyleSheet("padding: 5px; font-size: 12px;")
    shrink_x_layout.addWidget(shrink_x_spinbox)
    shrink_x_layout.addStretch()
    crop_settings_layout.addLayout(shrink_x_layout)
    
    # Shrink Y Percent setting
    shrink_y_layout = QHBoxLayout()
    shrink_y_label = QLabel("Shrink Y (%):")
    shrink_y_label.setStyleSheet("color: #2c3e50; font-size: 12px;")
    shrink_y_label.setFixedWidth(150)
    shrink_y_layout.addWidget(shrink_y_label)
    
    shrink_y_spinbox = QDoubleSpinBox()
    shrink_y_spinbox.setMinimum(0.0)
    shrink_y_spinbox.setMaximum(10.0)
    shrink_y_spinbox.setValue(10.0)
    shrink_y_spinbox.setSingleStep(0.1)
    shrink_y_spinbox.setDecimals(1)
    shrink_y_spinbox.setSuffix("%")
    shrink_y_spinbox.setStyleSheet("padding: 5px; font-size: 12px;")
    shrink_y_layout.addWidget(shrink_y_spinbox)
    shrink_y_layout.addStretch()
    crop_settings_layout.addLayout(shrink_y_layout)
    
    # Max Height Ratio setting
    max_height_ratio_layout = QHBoxLayout()
    max_height_ratio_label = QLabel("Max Height Ratio (%):")
    max_height_ratio_label.setStyleSheet("color: #2c3e50; font-size: 12px;")
    max_height_ratio_label.setFixedWidth(150)
    max_height_ratio_layout.addWidget(max_height_ratio_label)
    
    max_height_ratio_spinbox = QDoubleSpinBox()
    max_height_ratio_spinbox.setMinimum(1.0)
    max_height_ratio_spinbox.setMaximum(30.0)
    max_height_ratio_spinbox.setValue(8.0)
    max_height_ratio_spinbox.setSingleStep(0.5)
    max_height_ratio_spinbox.setDecimals(1)
    max_height_ratio_spinbox.setSuffix("%")
    max_height_ratio_spinbox.setStyleSheet("padding: 5px; font-size: 12px;")
    max_height_ratio_layout.addWidget(max_height_ratio_spinbox)
    max_height_ratio_layout.addStretch()
    crop_settings_layout.addLayout(max_height_ratio_layout)
    
    # Info label
    info_label = QLabel("💡 หมายเหตุ: การเปลี่ยนแปลงจะมีผลทันทีเมื่อประมวลผลครั้งถัดไป")
    info_label.setStyleSheet("color: #7f8c8d; font-size: 11px; font-style: italic; padding: 5px;")
    crop_settings_layout.addWidget(info_label)
    
    settings_layout.addWidget(crop_settings_group)
    settings_layout.addStretch()
    
    # Create widgets dict for main window to reference
    widgets_dict = {
        'padding_spinbox': padding_spinbox,
        'shrink_x_spinbox': shrink_x_spinbox,
        'shrink_y_spinbox': shrink_y_spinbox,
        'max_height_ratio_spinbox': max_height_ratio_spinbox
    }
    
    return settings_tab, widgets_dict


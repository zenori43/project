# -*- coding: utf-8 -*-
"""
Settings Tab Component
สร้าง UI สำหรับ Settings Tab
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, 
    QSpinBox, QDoubleSpinBox, QSlider, QPushButton, QFileDialog,
    QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import FADED_TEXT_CONFIG


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
    
    # Fade Detection Settings Group
    fade_settings_group = QGroupBox("🔍 การตั้งค่าการตรวจจับ Fade ของฝา")
    fade_settings_group.setStyleSheet("QGroupBox { font-weight: bold; color: #e67e22; font-size: 14px; padding: 10px; margin: 5px; }")
    fade_settings_layout = QVBoxLayout(fade_settings_group)
    fade_settings_layout.setContentsMargins(10, 15, 10, 10)
    fade_settings_layout.setSpacing(15)
    
    # Image Enhancement Settings
    enhancement_group = QGroupBox("ปรับปรุงภาพ (Image Enhancement)")
    enhancement_group.setStyleSheet("QGroupBox { font-weight: normal; color: #2c3e50; font-size: 12px; padding: 5px; margin: 5px; }")
    enhancement_layout = QVBoxLayout(enhancement_group)
    enhancement_layout.setContentsMargins(10, 10, 10, 10)
    enhancement_layout.setSpacing(10)
    
    # Get default values from config
    default_enhancement = FADED_TEXT_CONFIG.get('IMAGE_ENHANCEMENT', {})
    default_brightness = default_enhancement.get('brightness', 20)
    default_contrast = default_enhancement.get('contrast', 1.57)
    default_gamma = default_enhancement.get('gamma', 1.0)
    
    # Brightness setting
    brightness_layout = QHBoxLayout()
    brightness_label = QLabel("Brightness (เพิ่มแสง):")
    brightness_label.setStyleSheet("color: #2c3e50; font-size: 12px;")
    brightness_label.setFixedWidth(150)
    brightness_layout.addWidget(brightness_label)
    
    brightness_spinbox = QSpinBox()
    brightness_spinbox.setMinimum(-100)
    brightness_spinbox.setMaximum(100)
    brightness_spinbox.setValue(default_brightness)
    brightness_spinbox.setSuffix("")
    brightness_spinbox.setStyleSheet("padding: 5px; font-size: 12px;")
    brightness_layout.addWidget(brightness_spinbox)
    
    brightness_slider = QSlider(Qt.Horizontal)
    brightness_slider.setMinimum(-100)
    brightness_slider.setMaximum(100)
    brightness_slider.setValue(default_brightness)
    brightness_slider.setStyleSheet("padding: 5px;")
    brightness_slider.valueChanged.connect(brightness_spinbox.setValue)
    brightness_spinbox.valueChanged.connect(brightness_slider.setValue)
    brightness_layout.addWidget(brightness_slider)
    brightness_layout.addStretch()
    enhancement_layout.addLayout(brightness_layout)
    
    # Contrast setting
    contrast_layout = QHBoxLayout()
    contrast_label = QLabel("Contrast (ความคมชัด):")
    contrast_label.setStyleSheet("color: #2c3e50; font-size: 12px;")
    contrast_label.setFixedWidth(150)
    contrast_layout.addWidget(contrast_label)
    
    contrast_spinbox = QDoubleSpinBox()
    contrast_spinbox.setMinimum(0.5)
    contrast_spinbox.setMaximum(3.0)
    contrast_spinbox.setValue(default_contrast)
    contrast_spinbox.setSingleStep(0.1)
    contrast_spinbox.setDecimals(2)
    contrast_spinbox.setStyleSheet("padding: 5px; font-size: 12px;")
    contrast_layout.addWidget(contrast_spinbox)
    
    contrast_slider = QSlider(Qt.Horizontal)
    contrast_slider.setMinimum(5)  # 0.5 * 10
    contrast_slider.setMaximum(30)  # 3.0 * 10
    contrast_slider.setValue(int(default_contrast * 10))
    contrast_slider.setStyleSheet("padding: 5px;")
    contrast_slider.valueChanged.connect(lambda v: contrast_spinbox.setValue(v / 10.0))
    contrast_spinbox.valueChanged.connect(lambda v: contrast_slider.setValue(int(v * 10)))
    contrast_layout.addWidget(contrast_slider)
    contrast_layout.addStretch()
    enhancement_layout.addLayout(contrast_layout)
    
    # Gamma setting
    gamma_layout = QHBoxLayout()
    gamma_label = QLabel("Gamma Correction:")
    gamma_label.setStyleSheet("color: #2c3e50; font-size: 12px;")
    gamma_label.setFixedWidth(150)
    gamma_layout.addWidget(gamma_label)
    
    gamma_spinbox = QDoubleSpinBox()
    gamma_spinbox.setMinimum(0.3)
    gamma_spinbox.setMaximum(3.0)
    gamma_spinbox.setValue(default_gamma)
    gamma_spinbox.setSingleStep(0.1)
    gamma_spinbox.setDecimals(2)
    gamma_spinbox.setStyleSheet("padding: 5px; font-size: 12px;")
    gamma_layout.addWidget(gamma_spinbox)
    
    gamma_slider = QSlider(Qt.Horizontal)
    gamma_slider.setMinimum(3)  # 0.3 * 10
    gamma_slider.setMaximum(30)  # 3.0 * 10
    gamma_slider.setValue(int(default_gamma * 10))
    gamma_slider.setStyleSheet("padding: 5px;")
    gamma_slider.valueChanged.connect(lambda v: gamma_spinbox.setValue(v / 10.0))
    gamma_spinbox.valueChanged.connect(lambda v: gamma_slider.setValue(int(v * 10)))
    gamma_layout.addWidget(gamma_slider)
    gamma_layout.addStretch()
    enhancement_layout.addLayout(gamma_layout)
    
    fade_settings_layout.addWidget(enhancement_group)
    
    # Histogram Matching Settings
    histogram_matching_group = QGroupBox("Histogram Matching (ปรับแสงอัตโนมัติ)")
    histogram_matching_group.setStyleSheet("QGroupBox { font-weight: normal; color: #2c3e50; font-size: 12px; padding: 5px; margin: 5px; }")
    histogram_matching_layout = QVBoxLayout(histogram_matching_group)
    histogram_matching_layout.setContentsMargins(10, 10, 10, 10)
    histogram_matching_layout.setSpacing(10)
    
    # Use Histogram Matching checkbox
    use_histogram_matching_check = QCheckBox("ใช้ Histogram Matching แทนการเพิ่มแสงสว่างแบบเดิม")
    use_histogram_matching_check.setChecked(FADED_TEXT_CONFIG.get('USE_HISTOGRAM_MATCHING', True))
    use_histogram_matching_check.setStyleSheet("color: #2c3e50; font-size: 12px; font-weight: bold;")
    histogram_matching_layout.addWidget(use_histogram_matching_check)
    
    # Reference Cap Image Selection
    reference_cap_layout = QHBoxLayout()
    reference_cap_label = QLabel("รูปฝาอ้างอิง (Reference):")
    reference_cap_label.setStyleSheet("color: #2c3e50; font-size: 12px;")
    reference_cap_label.setFixedWidth(150)
    reference_cap_layout.addWidget(reference_cap_label)
    
    reference_cap_path_label = QLabel("ยังไม่ได้เลือก")
    reference_cap_path_label.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 5px; border: 1px solid #bdc3c7; border-radius: 3px; background-color: #ecf0f1;")
    reference_cap_path_label.setMinimumHeight(30)
    reference_cap_path_label.setWordWrap(True)
    reference_cap_layout.addWidget(reference_cap_path_label, 1)
    
    select_reference_btn = QPushButton("เลือกรูป")
    select_reference_btn.setStyleSheet("padding: 5px 15px; font-size: 12px;")
    reference_cap_layout.addWidget(select_reference_btn)
    histogram_matching_layout.addLayout(reference_cap_layout)
    
    # Matching Method
    method_layout = QHBoxLayout()
    method_label = QLabel("วิธีปรับ Histogram:")
    method_label.setStyleSheet("color: #2c3e50; font-size: 12px;")
    method_label.setFixedWidth(150)
    method_layout.addWidget(method_label)
    
    matching_method_combo = QComboBox()
    matching_method_combo.addItems([
        "CLAHE (แนะนำ)",
        "Histogram Matching",
        "Brightness Adjustment",
        "Histogram Matching + CLAHE"
    ])
    current_method = FADED_TEXT_CONFIG.get('HISTOGRAM_MATCHING_METHOD', 'clahe')
    method_map = {'clahe': 0, 'histogram_matching': 1, 'brightness': 2, 'mixed': 3}
    matching_method_combo.setCurrentIndex(method_map.get(current_method, 0))
    matching_method_combo.setStyleSheet("padding: 5px; font-size: 12px;")
    method_layout.addWidget(matching_method_combo)
    method_layout.addStretch()
    histogram_matching_layout.addLayout(method_layout)
    
    # Prevent Saturation checkbox
    prevent_saturation_check = QCheckBox("ป้องกันการขาวเกิน (Saturation)")
    prevent_saturation_check.setChecked(FADED_TEXT_CONFIG.get('PREVENT_SATURATION', True))
    prevent_saturation_check.setStyleSheet("color: #2c3e50; font-size: 12px;")
    histogram_matching_layout.addWidget(prevent_saturation_check)
    
    fade_settings_layout.addWidget(histogram_matching_group)
    
    # Info label
    fade_info_label = QLabel("💡 หมายเหตุ: การเปลี่ยนแปลงจะมีผลทันทีเมื่อตรวจจับ Fade ครั้งถัดไป")
    fade_info_label.setStyleSheet("color: #7f8c8d; font-size: 11px; font-style: italic; padding: 5px;")
    fade_settings_layout.addWidget(fade_info_label)
    
    settings_layout.addWidget(fade_settings_group)
    settings_layout.addStretch()
    
    # Create widgets dict for main window to reference
    widgets_dict = {
        'padding_spinbox': padding_spinbox,
        'shrink_x_spinbox': shrink_x_spinbox,
        'shrink_y_spinbox': shrink_y_spinbox,
        'max_height_ratio_spinbox': max_height_ratio_spinbox,
        'brightness_spinbox': brightness_spinbox,
        'contrast_spinbox': contrast_spinbox,
        'gamma_spinbox': gamma_spinbox,
        # Histogram matching widgets
        'use_histogram_matching_check': use_histogram_matching_check,
        'reference_cap_path_label': reference_cap_path_label,
        'select_reference_btn': select_reference_btn,
        'matching_method_combo': matching_method_combo,
        'prevent_saturation_check': prevent_saturation_check
    }
    
    return settings_tab, widgets_dict


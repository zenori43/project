# -*- coding: utf-8 -*-
"""
Cap Detection Tab Component
Builds the Cap detection tab UI.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, 
    QScrollArea, QTextEdit, QGroupBox, QPushButton
)
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets


def create_cap_tab():
    """
    Create the Cap detection tab.
    Returns:
        tuple: (cap_tab_widget, widgets_dict)
    """
    cap_tab = QWidget()
    cap_layout = QVBoxLayout(cap_tab)
    
    # Create main content area with splitter for cap detection
    cap_splitter = QSplitter(Qt.Horizontal)
    
    # Left side - Sentech Image display
    sentech_image_widget = QWidget()
    sentech_image_layout = QVBoxLayout(sentech_image_widget)
    
    sentech_image_title = QLabel("Sentech camera image")
    sentech_image_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #8e44ad;")
    sentech_image_title.setAlignment(Qt.AlignCenter)
    sentech_image_layout.addWidget(sentech_image_title)
    
    # Sentech Image display area
    sentech_image_label = QLabel()
    sentech_image_label.setMinimumSize(200, 180)
    sentech_image_label.setAlignment(Qt.AlignCenter)
    sentech_image_label.setStyleSheet("border: 2px solid #8e44ad; background-color: #f4f3f4;")
    sentech_image_label.setText("No Sentech camera image yet")
    sentech_image_label.setScaledContents(False)
    sentech_image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    sentech_image_layout.addWidget(sentech_image_label)
    
    # Sentech image info
    sentech_image_info_label = QLabel("Image info: -")
    sentech_image_info_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
    sentech_image_layout.addWidget(sentech_image_info_label)
    
    cap_splitter.addWidget(sentech_image_widget)
    
    # Middle - Cap detection results display with enhanced scroll
    cap_results_widget = QWidget()
    cap_results_layout = QVBoxLayout(cap_results_widget)
    
    cap_results_title = QLabel("Cap detection and text results")
    cap_results_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e67e22;")
    cap_results_title.setAlignment(Qt.AlignCenter)
    cap_results_layout.addWidget(cap_results_title)
    
    # Enhanced Scroll area for cap detection results
    cap_results_scroll = QScrollArea()
    cap_results_scroll.setWidgetResizable(True)
    cap_results_scroll.setMinimumSize(300, 400)
    cap_results_scroll.setMaximumHeight(600)
    cap_results_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    cap_results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    cap_results_scroll.setStyleSheet("""
        QScrollArea {
            border: 2px solid #e67e22; 
            background-color: #fef9e7;
            border-radius: 5px;
        }
        QScrollBar:vertical {
            background: #f0f0f0;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #c0c0c0;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #a0a0a0;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            background: #f0f0f0;
            height: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal {
            background: #c0c0c0;
            border-radius: 6px;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #a0a0a0;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
    """)
    
    # Container for cap detection results
    cap_results_container = QWidget()
    cap_results_container_layout = QVBoxLayout(cap_results_container)
    cap_results_container_layout.setAlignment(Qt.AlignTop)
    cap_results_container_layout.setSpacing(10)
    cap_results_container_layout.setContentsMargins(10, 10, 10, 10)
    
    # Add placeholder text
    cap_placeholder_label = QLabel("No cap detection results yet")
    cap_placeholder_label.setAlignment(Qt.AlignCenter)
    cap_placeholder_label.setStyleSheet("color: #7f8c8d; padding: 20px; font-size: 14px;")
    cap_results_container_layout.addWidget(cap_placeholder_label)
    
    cap_results_scroll.setWidget(cap_results_container)
    cap_results_layout.addWidget(cap_results_scroll)
    
    cap_splitter.addWidget(cap_results_widget)
    
    # Right side - Cap detection results with scroll
    cap_detection_widget = QWidget()
    cap_detection_layout = QVBoxLayout(cap_detection_widget)
    
    cap_detection_title = QLabel("Cap processing results")
    cap_detection_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c;")
    cap_detection_title.setAlignment(Qt.AlignCenter)
    cap_detection_layout.addWidget(cap_detection_title)
    
    # Cap detection results display with scroll
    cap_detection_text = QTextEdit()
    cap_detection_text.setReadOnly(True)
    cap_detection_text.setMinimumSize(160, 180)
    cap_detection_text.setLineWrapMode(QTextEdit.WidgetWidth)
    cap_detection_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    cap_detection_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    cap_detection_text.setStyleSheet("""
        QTextEdit {
            border: 2px solid #e74c3c;
            border-radius: 5px;
            background-color: #fef9e7;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
            line-height: 1.4;
        }
        QScrollBar:vertical {
            background: #f0f0f0;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #c0c0c0;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #a0a0a0;
        }
    """)
    cap_detection_layout.addWidget(cap_detection_text)
    
    cap_splitter.addWidget(cap_detection_widget)
    cap_splitter.setStretchFactor(0, 1)
    cap_splitter.setStretchFactor(1, 1)
    cap_splitter.setStretchFactor(2, 1)

    # Set splitter sizes
    cap_splitter.setSizes([280, 240, 260])
    
    # Add control buttons for cap detection
    cap_control_group = QGroupBox("Cap operations")
    cap_control_layout = QHBoxLayout()
    
    # Capture from Sentech camera (manual)
    btn_capture_sentech = QPushButton('📸 Capture Cap')
    btn_capture_sentech.setToolTip("Capture image from Sentech camera (cap) only")
    btn_capture_sentech.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #9C27B0; color: white; border-radius: 5px; } QPushButton:hover { background-color: #7B1FA2; }")
    cap_control_layout.addWidget(btn_capture_sentech)
    
    # Process cap button (use default analysis parameters)
    btn_process_cap = QPushButton('🔍 Process cap')
    btn_process_cap.setEnabled(False)  # Disabled by default, enabled when image is available
    btn_process_cap.setVisible(True)
    btn_process_cap.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #95a5a6; color: white; border-radius: 5px; }")
    btn_process_cap.setToolTip("Run cap processing (default analysis parameters — brightness: 33)")
    cap_control_layout.addWidget(btn_process_cap)
    
    # Save Sentech image button
    btn_save_sentech_image = QPushButton('💾 Save cap image')
    btn_save_sentech_image.setEnabled(False)
    btn_save_sentech_image.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #9b59b6; color: white; border-radius: 5px; }")
    cap_control_layout.addWidget(btn_save_sentech_image)
    
    # Select image file button for cap detection
    btn_select_cap_image = QPushButton('📁 Select cap image')
    btn_select_cap_image.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #3498db; color: white; border-radius: 5px; }")
    cap_control_layout.addWidget(btn_select_cap_image)
    
    # Select multiple images button for batch processing
    btn_select_multiple_cap_images = QPushButton('📁 Select multiple cap images')
    btn_select_multiple_cap_images.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #16a085; color: white; border-radius: 5px; }")
    btn_select_multiple_cap_images.setToolTip("Select multiple files for batch processing and history")
    cap_control_layout.addWidget(btn_select_multiple_cap_images)
    
    cap_control_group.setLayout(cap_control_layout)
    cap_layout.addWidget(cap_control_group)
    
    # Add process steps info
    cap_steps_group = QGroupBox("Cap processing steps")
    cap_steps_layout = QVBoxLayout()
    
    steps_text = """
🔄 Cap processing steps:

Step 1: Detect cap with Cap Detector
Step 2: Crop detected regions
Step 2a: If no cap found → detect text with CRAFT
Step 2b: Process with AI Rotation
Step 3: Detect text lines
Step 4: Read text with OCR

⏱️ Typical time: about 5–15 seconds
    """
    
    steps_label = QLabel(steps_text)
    steps_label.setStyleSheet("color: #2c3e50; font-size: 11px; background-color: #ecf0f1; padding: 10px; border-radius: 5px;")
    steps_label.setWordWrap(True)
    cap_steps_layout.addWidget(steps_label)
    
    cap_steps_group.setLayout(cap_steps_layout)
    cap_layout.addWidget(cap_steps_group)
    
    cap_layout.addWidget(cap_splitter)
    
    # Create widgets dict for main window to reference
    widgets_dict = {
        'cap_splitter': cap_splitter,
        'sentech_image_label': sentech_image_label,
        'sentech_image_info_label': sentech_image_info_label,
        'cap_results_scroll': cap_results_scroll,
        'cap_results_container': cap_results_container,
        'cap_results_layout': cap_results_container_layout,
        'cap_detection_text': cap_detection_text,
        'btn_capture_sentech': btn_capture_sentech,
        'btn_save_sentech_image': btn_save_sentech_image,
        'btn_select_cap_image': btn_select_cap_image,
        'btn_select_multiple_cap_images': btn_select_multiple_cap_images,
        'btn_process_cap': btn_process_cap
    }
    
    return cap_tab, widgets_dict


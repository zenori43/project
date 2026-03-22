# -*- coding: utf-8 -*-
"""
Control Panel Component
ออกแบบหน้า Control Panel แบบ modern dark theme ด้วย card-based layout
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGridLayout, QFrame, QScrollArea, QGroupBox, QCheckBox,
    QDateTimeEdit, QProgressBar, QSpinBox, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtGui import QFont


class ControlCard(QFrame):
    """Card widget สำหรับแสดงข้อมูลและควบคุม"""
    def __init__(self, title, icon="", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #dddddd;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)
        
        # Title
        title_label = QLabel(f"{icon} {title}")
        title_label.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 15px;
                font-weight: bold;
                padding-bottom: 5px;
                border-bottom: 1px solid #dddddd;
            }
        """)
        layout.addWidget(title_label)
        
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(10)
        layout.addLayout(self.content_layout)
    
    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


class StatusIndicator(QFrame):
    """Status indicator widget"""
    def __init__(self, label, status="waiting", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-radius: 6px;
                border: 1px solid #dddddd;
                padding: 8px;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)
        
        # Status dot
        self.status_dot = QLabel('●')
        self.status_dot.setFixedWidth(20)
        self.status_dot.setAlignment(Qt.AlignCenter)
        self.set_status(status)
        layout.addWidget(self.status_dot)
        
        # Label
        self.label = QLabel(label)
        self.label.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.label)
        
        layout.addStretch()
    
    def set_status(self, status):
        """Set status: 'online', 'offline', 'waiting', 'error'"""
        colors = {
            'online': '#4CAF50',
            'offline': '#f44336',
            'waiting': '#ff9800',
            'error': '#f44336'
        }
        color = colors.get(status, '#ff9800')
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
    
    def set_text(self, text):
        self.label.setText(text)


def create_control_panel():
    """
    สร้าง Control Panel แบบ modern dark theme
    Returns:
        tuple: (control_panel_widget, widgets_dict)
    """
    # Main widget with scroll area
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setStyleSheet("""
        QScrollArea {
            border: none;
            background-color: #f5f5f5;
        }
        QScrollBar:vertical {
            background-color: #e8e8e8;
            width: 12px;
            border: none;
        }
        QScrollBar::handle:vertical {
            background-color: #b0b0b0;
            min-height: 30px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #95a5a6;
        }
    """)
    
    main_widget = QWidget()
    main_widget.setStyleSheet("background-color: #f5f5f5;")
    main_layout = QVBoxLayout(main_widget)
    main_layout.setContentsMargins(15, 15, 15, 15)
    main_layout.setSpacing(15)
    
    widgets_dict = {}
    
    # ========== System Status Card ==========
    status_card = ControlCard("System Status", "📊")
    
    # Status indicators grid
    status_grid = QGridLayout()
    status_grid.setSpacing(10)
    
    # Camera status
    camera_status = StatusIndicator("USB Camera", "waiting")
    status_grid.addWidget(camera_status, 0, 0)
    widgets_dict['camera_status_indicator'] = camera_status
    
    # Sentech camera status
    sentech_status = StatusIndicator("Sentech Camera", "waiting")
    status_grid.addWidget(sentech_status, 0, 1)
    widgets_dict['sentech_status_indicator'] = sentech_status
    
    # Modbus status
    modbus_status = StatusIndicator("Modbus", "waiting")
    status_grid.addWidget(modbus_status, 1, 0)
    widgets_dict['modbus_status_indicator'] = modbus_status
    
    # Queue status
    queue_status = StatusIndicator("Queue: 0", "waiting")
    status_grid.addWidget(queue_status, 1, 1)
    widgets_dict['queue_status_indicator'] = queue_status
    
    # D5002 status
    d5002_status = StatusIndicator("D5002: Waiting", "waiting")
    status_grid.addWidget(d5002_status, 2, 0)
    widgets_dict['d5002_status_indicator'] = d5002_status
    
    # D5001 status
    d5001_status = StatusIndicator("D5001: Waiting", "waiting")
    status_grid.addWidget(d5001_status, 2, 1)
    widgets_dict['d5001_status_indicator'] = d5001_status
    
    status_card.add_layout(status_grid)
    main_layout.addWidget(status_card)
    
    # ========== Control Actions Card ==========
    control_card = ControlCard("Controls", "🎮")
    
    control_grid = QGridLayout()
    control_grid.setSpacing(10)
    
    # Capture button (hidden from UI but kept for handlers)
    btn_capture = QPushButton('📸 Capture')
    btn_capture.setStyleSheet("""
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 12px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #1565C0;
        }
    """)
    # Not added to layout to hide from UI
    widgets_dict['btn_capture'] = btn_capture
    
    # Save image button
    btn_save = QPushButton('💾 Save')
    btn_save.setEnabled(False)
    btn_save.setStyleSheet("""
        QPushButton {
            background-color: #9C27B0;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 12px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover:enabled {
            background-color: #7B1FA2;
        }
        QPushButton:disabled {
            background-color: #555;
            color: #888;
        }
    """)
    # Not added to layout to hide from UI
    widgets_dict['btn_save_image'] = btn_save
    
    # Select image button
    btn_select = QPushButton('📁 Select file')
    btn_select.setStyleSheet("""
        QPushButton {
            background-color: #00BCD4;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 12px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #0097A7;
        }
    """)
    # Not added to layout to hide from UI
    widgets_dict['btn_select_image'] = btn_select
    
    # Process button (hidden from UI but kept for handlers)
    btn_process = QPushButton('🔍 Process')
    btn_process.setEnabled(False)
    btn_process.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 12px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover:enabled {
            background-color: #388E3C;
        }
        QPushButton:disabled {
            background-color: #555;
            color: #888;
        }
    """)
    # Not added to layout to hide from UI
    widgets_dict['btn_process'] = btn_process
    
    # Row 0: ถ่ายทั้งสอง (preview) + Stop + Capture & Save Both
    btn_capture_both = QPushButton('📸 Capture both cameras (USB + Sentech)')
    btn_capture_both.setToolTip(
        "Capture from USB and Sentech at once and show on screen (does not save files — use Capture & Save Both to save)"
    )
    btn_capture_both.setStyleSheet("""
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #1565C0;
        }
        QPushButton:disabled {
            background-color: #555;
            color: #888;
        }
    """)
    btn_capture_both.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    widgets_dict['btn_capture_both'] = btn_capture_both

    btn_stop = QPushButton('⏹️ Stop Processing')
    btn_stop.setEnabled(False)
    btn_stop.setStyleSheet("""
        QPushButton {
            background-color: #f44336;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover:enabled {
            background-color: #D32F2F;
        }
        QPushButton:disabled {
            background-color: #555;
            color: #888;
        }
    """)
    widgets_dict['btn_stop_processing'] = btn_stop
    
    btn_capture_and_save = QPushButton('📸💾 Capture & Save Both')
    btn_capture_and_save.setToolTip("Capture from USB + Sentech together and save both images to captured_images folder")
    btn_capture_and_save.setStyleSheet("""
        QPushButton {
            background-color: #FF9800;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #F57C00;
        }
        QPushButton:pressed {
            background-color: #E65100;
        }
    """)
    btn_stop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    btn_capture_and_save.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    # แถวเดียวกับปุ่มล่าง: 3 คอลัมน์ ความกว้างเท่ากัน
    control_grid.addWidget(btn_capture_both, 0, 0)
    control_grid.addWidget(btn_stop, 0, 1)
    control_grid.addWidget(btn_capture_and_save, 0, 2)
    widgets_dict['btn_capture_and_save'] = btn_capture_and_save
    
    # Row 1: process both (bottle + cap)
    btn_select_bottle_folder = QPushButton('📁 Select bottle folder')
    btn_select_bottle_folder.setToolTip("Choose a folder containing bottle images")
    btn_select_bottle_folder.setStyleSheet("""
        QPushButton {
            background-color: #27ae60;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #229954;
        }
    """)
    btn_select_bottle_folder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    control_grid.addWidget(btn_select_bottle_folder, 1, 0)
    widgets_dict['btn_select_bottle_folder'] = btn_select_bottle_folder
    
    btn_select_cap_folder = QPushButton('📁 Select cap folder')
    btn_select_cap_folder.setToolTip("Choose a folder containing cap images")
    btn_select_cap_folder.setStyleSheet("""
        QPushButton {
            background-color: #9C27B0;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #7B1FA2;
        }
    """)
    btn_select_cap_folder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    control_grid.addWidget(btn_select_cap_folder, 1, 1)
    widgets_dict['btn_select_cap_folder'] = btn_select_cap_folder
    
    btn_process_paired = QPushButton('🔄 Process both')
    btn_process_paired.setToolTip(
        "Process bottle and cap as a pair: from selected folders, or from the latest captures after Capture both cameras (USB + Sentech)"
    )
    btn_process_paired.setEnabled(False)
    btn_process_paired.setStyleSheet("""
        QPushButton {
            background-color: #FF9800;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover:enabled {
            background-color: #F57C00;
        }
        QPushButton:disabled {
            background-color: #555;
            color: #888;
        }
    """)
    btn_process_paired.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    control_grid.addWidget(btn_process_paired, 1, 2)
    widgets_dict['btn_process_paired'] = btn_process_paired

    for col in range(3):
        control_grid.setColumnStretch(col, 1)
    
    control_card.add_layout(control_grid)
    main_layout.addWidget(control_card)
    
    # ========== Taste Mode Card ==========
    taste_card = ControlCard("Taste Mode", "🍾")
    
    # Mode selection
    mode_layout = QHBoxLayout()
    mode_layout.setSpacing(15)
    
    mode_label = QLabel("Mode:")
    mode_label.setStyleSheet("color: #333333; font-size: 13px;")
    mode_layout.addWidget(mode_label)
    
    taste_mode_1 = QCheckBox("1 Taste")
    taste_mode_1.setStyleSheet("""
        QCheckBox {
            color: #333333;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """)
    mode_layout.addWidget(taste_mode_1)
    widgets_dict['taste_mode_1'] = taste_mode_1
    
    taste_mode_2 = QCheckBox("2 Tastes")
    taste_mode_2.setStyleSheet("""
        QCheckBox {
            color: #333333;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """)
    mode_layout.addWidget(taste_mode_2)
    widgets_dict['taste_mode_2'] = taste_mode_2
    
    taste_mode_3 = QCheckBox("3 Tastes")
    taste_mode_3.setChecked(True)
    taste_mode_3.setStyleSheet("""
        QCheckBox {
            color: #333333;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """)
    mode_layout.addWidget(taste_mode_3)
    widgets_dict['taste_mode_3'] = taste_mode_3
    
    mode_layout.addStretch()
    taste_card.add_layout(mode_layout)
    
    # Taste selection
    taste_selection_layout = QHBoxLayout()
    taste_selection_layout.setSpacing(15)
    
    taste_label = QLabel("Select taste:")
    taste_label.setStyleSheet("color: #333333; font-size: 13px;")
    taste_selection_layout.addWidget(taste_label)
    
    taste_m100 = QCheckBox("Original")
    taste_m100.setStyleSheet("""
        QCheckBox {
            color: #333333;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """)
    taste_selection_layout.addWidget(taste_m100)
    widgets_dict['taste_m100'] = taste_m100
    
    taste_m110 = QCheckBox("Less sugar 2%")
    taste_m110.setStyleSheet("""
        QCheckBox {
            color: #333333;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """)
    taste_selection_layout.addWidget(taste_m110)
    widgets_dict['taste_m110'] = taste_m110
    
    taste_m120 = QCheckBox("Basil seed mix")
    taste_m120.setStyleSheet("""
        QCheckBox {
            color: #333333;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """)
    taste_selection_layout.addWidget(taste_m120)
    widgets_dict['taste_m120'] = taste_m120
    
    taste_selection_layout.addStretch()
    widgets_dict['taste_selection_layout'] = taste_selection_layout
    taste_card.add_layout(taste_selection_layout)
    
    # Taste mode status
    taste_status = QLabel("Current mode: 3 tastes (Original, Less sugar 2%, Basil seed mix)")
    taste_status.setStyleSheet("""
        QLabel {
            color: #4CAF50;
            font-size: 12px;
            font-weight: bold;
            padding: 8px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }
    """)
    taste_card.add_widget(taste_status)
    widgets_dict['taste_mode_status'] = taste_status
    
    main_layout.addWidget(taste_card)
    
    # ========== Expiry Mode Card ==========
    expiry_card = ControlCard("Expiry Mode", "📅")
    
    # Expiry mode selection
    expiry_mode_layout = QHBoxLayout()
    expiry_mode_layout.setSpacing(15)
    
    expiry_mode_label = QLabel("Mode:")
    expiry_mode_label.setStyleSheet("color: #333333; font-size: 13px;")
    expiry_mode_layout.addWidget(expiry_mode_label)
    
    expiry_mode_normal = QCheckBox("Normal mode")
    expiry_mode_normal.setChecked(True)
    expiry_mode_normal.setStyleSheet("""
        QCheckBox {
            color: #333333;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """)
    expiry_mode_layout.addWidget(expiry_mode_normal)
    widgets_dict['expiry_mode_normal'] = expiry_mode_normal
    
    expiry_mode_filter = QCheckBox("Filter by date")
    expiry_mode_filter.setStyleSheet("""
        QCheckBox {
            color: #333333;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """)
    expiry_mode_layout.addWidget(expiry_mode_filter)
    widgets_dict['expiry_mode_filter'] = expiry_mode_filter
    
    expiry_mode_layout.addStretch()
    expiry_card.add_layout(expiry_mode_layout)
    
    # Filter type selection
    filter_type_layout = QHBoxLayout()
    filter_type_layout.setSpacing(15)
    
    filter_type_label = QLabel("Type:")
    filter_type_label.setStyleSheet("color: #333333; font-size: 13px;")
    filter_type_layout.addWidget(filter_type_label)
    
    expiry_filter_range = QCheckBox("Date range")
    expiry_filter_range.setChecked(True)
    expiry_filter_range.setStyleSheet("""
        QCheckBox {
            color: #333333;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """)
    filter_type_layout.addWidget(expiry_filter_range)
    widgets_dict['expiry_filter_range'] = expiry_filter_range
    
    expiry_filter_specific = QCheckBox("Specific date")
    expiry_filter_specific.setStyleSheet("""
        QCheckBox {
            color: #333333;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """)
    filter_type_layout.addWidget(expiry_filter_specific)
    widgets_dict['expiry_filter_specific'] = expiry_filter_specific
    
    filter_type_layout.addStretch()
    widgets_dict['expiry_filter_type_layout'] = filter_type_layout
    expiry_card.add_layout(filter_type_layout)
    
    # Use date from cap: MFG or BBF
    date_type_layout = QHBoxLayout()
    date_type_layout.setSpacing(15)
    date_type_label = QLabel("Use date:")
    date_type_label.setStyleSheet("color: #333333; font-size: 13px;")
    date_type_layout.addWidget(date_type_label)
    expiry_date_type_combo = QComboBox()
    expiry_date_type_combo.addItem("BBF", "BBF")
    expiry_date_type_combo.addItem("MFG", "MFG")
    expiry_date_type_combo.setCurrentIndex(0)
    expiry_date_type_combo.setStyleSheet("""
        QComboBox {
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 6px;
            font-size: 12px;
            min-width: 80px;
        }
    """)
    date_type_layout.addWidget(expiry_date_type_combo)
    date_type_layout.addStretch()
    widgets_dict['expiry_date_type_combo'] = expiry_date_type_combo
    widgets_dict['expiry_date_type_layout'] = date_type_layout
    expiry_card.add_layout(date_type_layout)
    
    # Date selection
    date_selection_layout = QVBoxLayout()
    date_selection_layout.setSpacing(10)
    
    # Range date selection
    range_date_layout = QHBoxLayout()
    range_date_layout.setSpacing(10)
    
    range_start_label = QLabel("จากวันที่:")
    range_start_label.setStyleSheet("color: #333333; font-size: 12px;")
    range_date_layout.addWidget(range_start_label)
    
    expiry_start_date = QDateTimeEdit()
    expiry_start_date.setDate(QDateTime.currentDateTime().date())
    expiry_start_date.setCalendarPopup(True)
    expiry_start_date.setStyleSheet("""
        QDateTimeEdit {
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 6px;
            font-size: 12px;
        }
        QDateTimeEdit:hover {
            border: 1px solid #bdc3c7;
        }
    """)
    range_date_layout.addWidget(expiry_start_date)
    widgets_dict['expiry_start_date_edit'] = expiry_start_date
    
    range_end_label = QLabel("ถึงวันที่:")
    range_end_label.setStyleSheet("color: #333333; font-size: 12px;")
    range_date_layout.addWidget(range_end_label)
    
    expiry_end_date = QDateTimeEdit()
    expiry_end_date.setDate(QDateTime.currentDateTime().date())
    expiry_end_date.setCalendarPopup(True)
    expiry_end_date.setStyleSheet("""
        QDateTimeEdit {
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 6px;
            font-size: 12px;
        }
        QDateTimeEdit:hover {
            border: 1px solid #bdc3c7;
        }
    """)
    range_date_layout.addWidget(expiry_end_date)
    widgets_dict['expiry_end_date_edit'] = expiry_end_date
    
    range_date_layout.addStretch()
    widgets_dict['range_date_layout'] = range_date_layout
    date_selection_layout.addLayout(range_date_layout)
    
    # Specific date selection
    specific_date_layout = QHBoxLayout()
    specific_date_layout.setSpacing(10)
    
    specific_date_label = QLabel("วันที่เฉพาะ:")
    specific_date_label.setStyleSheet("color: #333333; font-size: 12px;")
    specific_date_layout.addWidget(specific_date_label)
    
    expiry_specific_date = QDateTimeEdit()
    expiry_specific_date.setDate(QDateTime.currentDateTime().date())
    expiry_specific_date.setCalendarPopup(True)
    expiry_specific_date.setStyleSheet("""
        QDateTimeEdit {
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 6px;
            font-size: 12px;
        }
        QDateTimeEdit:hover {
            border: 1px solid #bdc3c7;
        }
    """)
    specific_date_layout.addWidget(expiry_specific_date)
    widgets_dict['expiry_specific_date_edit'] = expiry_specific_date
    
    specific_date_layout.addStretch()
    widgets_dict['specific_date_layout'] = specific_date_layout
    date_selection_layout.addLayout(specific_date_layout)
    
    widgets_dict['date_selection_layout'] = date_selection_layout
    expiry_card.add_layout(date_selection_layout)
    
    # Expiry mode status
    expiry_status = QLabel("Current mode: Normal (no expiry filter)")
    expiry_status.setStyleSheet("""
        QLabel {
            color: #4CAF50;
            font-size: 12px;
            font-weight: bold;
            padding: 8px;
            background-color: #1e1e1e;
            border-radius: 5px;
        }
    """)
    expiry_card.add_widget(expiry_status)
    widgets_dict['expiry_mode_status'] = expiry_status
    
    main_layout.addWidget(expiry_card)
    
    # ========== Gripper Control Card ==========
    gripper_card = ControlCard("Gripper Control", "🤖")
    
    gripper_layout = QHBoxLayout()
    gripper_layout.setSpacing(10)
    
    btn_gripper_open = QPushButton('🟢 Open')
    btn_gripper_open.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 12px 20px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #388E3C;
        }
    """)
    gripper_layout.addWidget(btn_gripper_open)
    widgets_dict['btn_gripper_open'] = btn_gripper_open
    
    btn_gripper_close = QPushButton('🔴 Close')
    btn_gripper_close.setStyleSheet("""
        QPushButton {
            background-color: #f44336;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 12px 20px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #D32F2F;
        }
    """)
    gripper_layout.addWidget(btn_gripper_close)
    widgets_dict['btn_gripper_close'] = btn_gripper_close
    
    btn_motor_reverse = QPushButton('🔄 Reverse Motor')
    btn_motor_reverse.setStyleSheet("""
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 12px 20px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #1565C0;
        }
    """)
    gripper_layout.addWidget(btn_motor_reverse)
    widgets_dict['btn_motor_reverse'] = btn_motor_reverse
    
    gripper_status = QLabel('Status: Waiting for command')
    gripper_status.setStyleSheet("""
        QLabel {
            color: #ff9800;
            font-size: 13px;
            font-weight: bold;
            padding: 8px 15px;
            background-color: #1e1e1e;
            border-radius: 5px;
        }
    """)
    gripper_layout.addWidget(gripper_status)
    widgets_dict['gripper_status_label'] = gripper_status
    
    gripper_card.add_layout(gripper_layout)
    main_layout.addWidget(gripper_card)
    
    # ========== Progress & Status Card ==========
    progress_card = ControlCard("Progress & Status", "📈")
    
    # Progress bars
    progress_bar = QProgressBar()
    progress_bar.setVisible(False)
    progress_bar.setStyleSheet("""
        QProgressBar {
            border: 2px solid #3a3a3a;
            border-radius: 5px;
            text-align: center;
            height: 25px;
            background-color: #1e1e1e;
        }
        QProgressBar::chunk {
            background-color: #2196F3;
            border-radius: 3px;
        }
    """)
    progress_card.add_widget(progress_bar)
    widgets_dict['progress_bar'] = progress_bar
    
    cap_progress_bar = QProgressBar()
    cap_progress_bar.setVisible(False)
    cap_progress_bar.setStyleSheet("""
        QProgressBar {
            border: 2px solid #3a3a3a;
            border-radius: 5px;
            text-align: center;
            height: 25px;
            background-color: #1e1e1e;
        }
        QProgressBar::chunk {
            background-color: #9C27B0;
            border-radius: 3px;
        }
    """)
    progress_card.add_widget(cap_progress_bar)
    widgets_dict['cap_progress_bar'] = cap_progress_bar
    
    # Status labels
    status_label = QLabel('⏸️ Ready to run')
    status_label.setStyleSheet("""
        QLabel {
            color: #ff9800;
            font-size: 12px;
            padding: 8px;
            background-color: #1e1e1e;
            border-radius: 5px;
        }
    """)
    progress_card.add_widget(status_label)
    widgets_dict['status_label'] = status_label
    
    combined_status = QLabel('📊 Overall status: System ready')
    combined_status.setStyleSheet("""
        QLabel {
            color: #4CAF50;
            font-size: 12px;
            font-weight: bold;
            padding: 8px;
            background-color: #1e1e1e;
            border-radius: 5px;
        }
    """)
    progress_card.add_widget(combined_status)
    widgets_dict['combined_status_label'] = combined_status
    
    queue_info = QLabel('📋 Queue: no pending items')
    queue_info.setStyleSheet("""
        QLabel {
            color: #ff9800;
            font-size: 12px;
            padding: 8px;
            background-color: #1e1e1e;
            border-radius: 5px;
        }
    """)
    progress_card.add_widget(queue_info)
    widgets_dict['queue_info_label'] = queue_info
    
    system_time = QLabel('🕐 Time: loading...')
    system_time.setStyleSheet("""
        QLabel {
            color: #9e9e9e;
            font-size: 11px;
            padding: 6px;
            background-color: #1e1e1e;
            border-radius: 5px;
        }
    """)
    progress_card.add_widget(system_time)
    widgets_dict['system_time_label'] = system_time
    
    processing_status = QLabel('⏸️ No processing')
    processing_status.setStyleSheet("""
        QLabel {
            color: #9e9e9e;
            font-size: 11px;
            padding: 6px;
            background-color: #1e1e1e;
            border-radius: 5px;
        }
    """)
    progress_card.add_widget(processing_status)
    widgets_dict['processing_status_label'] = processing_status
    
    # Add Progress & Status card to main layout so widgets stay alive (parent required).
    # Keep it in tree; can hide with setVisible(False) if not wanted in UI.
    main_layout.addWidget(progress_card)
    main_layout.addStretch()
    
    scroll_area.setWidget(main_widget)
    
    return scroll_area, widgets_dict

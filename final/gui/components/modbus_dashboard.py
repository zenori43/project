# -*- coding: utf-8 -*-
"""
Modbus Dashboard Component
ออกแบบหน้า dashboard สถานะ Modbus แบบ modern dark theme
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGridLayout, QFrame, QScrollArea, QGroupBox, QLineEdit
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor, QPalette
import os


class ToggleSwitch(QPushButton):
    """Custom Toggle Switch Widget"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(50, 26)
        self.toggled.connect(self.update_style)
        self.update_style()
    
    def update_style(self):
        if self.isChecked():
            self.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    border-radius: 13px;
                    border: none;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #555;
                    border-radius: 13px;
                    border: none;
                }
            """)
    
    def paintEvent(self, event):
        """Custom paint event to draw toggle switch"""
        from PyQt5.QtGui import QPainter, QBrush
        from PyQt5.QtCore import QRect
        
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background circle (indicator)
        if self.isChecked():
            # Position on right
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(26, 2, 22, 22)
        else:
            # Position on left
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(2, 2, 22, 22)


class StatusCard(QFrame):
    """Card widget สำหรับแสดงข้อมูล"""
    def __init__(self, title, icon="", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 8px;
                border: 1px solid #3a3a3a;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Title
        title_layout = QHBoxLayout()
        title_label = QLabel(f"{icon} {title}")
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(8)
        layout.addLayout(self.content_layout)
    
    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


def create_modbus_dashboard():
    """
    สร้าง Modbus Dashboard แบบ modern dark theme
    Returns:
        tuple: (dashboard_widget, widgets_dict)
    """
    # Main widget
    dashboard = QWidget()
    dashboard.setStyleSheet("""
        QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
        }
    """)
    
    main_layout = QVBoxLayout(dashboard)
    main_layout.setContentsMargins(20, 20, 20, 20)
    main_layout.setSpacing(15)
    
    # ========== MAIN CONTENT - CARDS GRID ==========
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet("""
        QScrollArea {
            border: none;
            background-color: transparent;
        }
    """)
    
    content_widget = QWidget()
    content_layout = QGridLayout(content_widget)
    content_layout.setSpacing(15)
    content_layout.setContentsMargins(0, 0, 0, 0)
    
    widgets_dict = {}
    
    # ========== COILS CONTROL - จัดกลุ่มตามประเภท ==========
    coil_toggles = {}
    
    # กำหนดค่าเริ่มต้นสำหรับ grid layout
    row = 0
    col = 0
    
    # กำหนด coils แยกตามกลุ่ม
    coil_groups = [
        {
            "title": "ประเภทขวด",
            "icon": "🍾",
            "coils": [
                ("M100", "ขวด1"), ("M110", "ขวด2"), ("M120", "ขวด3"), ("M130", "ขวด4")
            ]
        },
        {
            "title": "ควบคุมระบบ",
            "icon": "⚙",
            "coils": [
                ("M140", "ฝาNG"), ("M600", "เสร็จ"), ("M700", "เริ่ม"), ("M701", "หยุด")
            ]
        },
        {
            "title": "Gripper",
            "icon": "🤖",
            "coils": [
                ("M503", "จับ"), ("M505", "ปล่อย"), ("M507", "Grip3")
            ]
        },
        {
            "title": "โหมด & รสชาติ",
            "icon": "🎯",
            "coils": [
                ("M90", "โหมด"), ("M720", "รส1"), ("M721", "รส2"), ("M722", "รส3")
            ]
        },
        {
            "title": "รสผสม (2 รส)",
            "icon": "🍹",
            "coils": [
                ("M730", "รส1+2"), ("M731", "รส1+3"), ("M732", "รส2+3"),
                ("M733", "รส1+4"), ("M734", "รส2+4"), ("M735", "รส3+4")
            ]
        },
        {
            "title": "รสเดี่ยว",
            "icon": "🥤",
            "coils": [
                ("M740", "รส1"), ("M741", "รส2"), ("M742", "รส3")
            ]
        },
        {
            "title": "มุม",
            "icon": "📐",
            "coils": [
                ("M750", "มุม1"), ("M850", "มุม3")
            ]
        },
        {
            "title": "ควบคุมระบบ",
            "icon": "🔧",
            "coils": [
                ("M401", "รีเซ็ต"), ("M402", "ตรวจ"), ("M403", "เต้าหู้"), ("M404", "น้ำตาล"),
                ("M405", "แมงลัก"), ("M406", "NG")
            ]
        },
        {
            "title": "โปรแกรม",
            "icon": "💻",
            "coils": [
                ("M511", "โปรแกรม"), ("M512", "พร้อม"), ("M513", "หยุด")
            ]
        }
    ]
    
    # สร้าง card สำหรับแต่ละกลุ่ม
    row = 0
    col = 0
    for group in coil_groups:
        group_card = StatusCard(group["title"], group["icon"])
        
        # Grid สำหรับ coils ในกลุ่มนี้
        group_grid = QGridLayout()
        group_grid.setSpacing(10)
        group_grid.setContentsMargins(5, 5, 5, 5)
        
        for i, (coil, label) in enumerate(group["coils"]):
            grid_col = i % 4  # 4 คอลัมน์ต่อแถว
            grid_row = i // 4
            
            coil_widget = QWidget()
            coil_layout = QVBoxLayout(coil_widget)
            coil_layout.setContentsMargins(5, 5, 5, 5)
            coil_layout.setSpacing(4)
            
            # Coil name
            coil_name_label = QLabel(coil)
            coil_name_label.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: bold;")
            coil_name_label.setAlignment(Qt.AlignCenter)
            coil_layout.addWidget(coil_name_label)
            
            # Label สั้นๆ
            coil_desc_label = QLabel(label)
            coil_desc_label.setStyleSheet("color: #aaaaaa; font-size: 9px;")
            coil_desc_label.setAlignment(Qt.AlignCenter)
            coil_layout.addWidget(coil_desc_label)
            
            # Toggle switch
            toggle = ToggleSwitch()
            coil_toggles[coil] = toggle
            coil_layout.addWidget(toggle, alignment=Qt.AlignCenter)
            
            group_grid.addWidget(coil_widget, grid_row, grid_col)
        
        group_card.content_layout.addLayout(group_grid)
        
        # จัด layout เป็น 2 คอลัมน์
        content_layout.addWidget(group_card, row, col)
        col += 1
        if col >= 2:
            col = 0
            row += 1
    
    widgets_dict.update(coil_toggles)
    
    # ========== CARD: System Status ==========
    system_card = StatusCard("System Status", "⚡")
    
    # Modbus RTU Status
    modbus_rtu_layout = QHBoxLayout()
    modbus_rtu_dot = QLabel("●")
    modbus_rtu_dot.setStyleSheet("color: #4CAF50; font-size: 12px; font-weight: bold;")
    modbus_rtu_text = QLabel("Modbus RTU: Running")
    modbus_rtu_text.setStyleSheet("color: #ffffff; font-size: 12px;")
    modbus_rtu_layout.addWidget(modbus_rtu_dot)
    modbus_rtu_layout.addWidget(modbus_rtu_text)
    modbus_rtu_layout.addStretch()
    system_card.add_widget(create_status_row(modbus_rtu_layout))
    
    # Camera Feed Status
    camera_feed_layout = QHBoxLayout()
    camera_feed_dot = QLabel("●")
    camera_feed_dot.setStyleSheet("color: #4CAF50; font-size: 12px; font-weight: bold;")
    camera_feed_text = QLabel("Camera Feed: Disabled")
    camera_feed_text.setStyleSheet("color: #ffffff; font-size: 12px;")
    camera_feed_layout.addWidget(camera_feed_dot)
    camera_feed_layout.addWidget(camera_feed_text)
    camera_feed_layout.addStretch()
    system_card.add_widget(create_status_row(camera_feed_layout))
    
    # Data Flow
    data_flow_label = QLabel("Data Flow (last 1 min)")
    data_flow_label.setStyleSheet("color: #4CAF50; font-size: 11px; margin-top: 10px;")
    system_card.add_widget(data_flow_label)
    
    # Simple bar chart representation
    bar_chart = QFrame()
    bar_chart.setFixedHeight(20)
    bar_chart.setStyleSheet("""
        QFrame {
            background-color: #1a1a1a;
            border-radius: 4px;
        }
    """)
    bar_layout = QHBoxLayout(bar_chart)
    bar_layout.setContentsMargins(2, 2, 2, 2)
    bar_layout.setSpacing(2)
    for i in range(10):
        bar = QFrame()
        bar.setStyleSheet(f"background-color: #4CAF50; border-radius: 2px;")
        height = 15 + (i * 2)
        bar.setFixedHeight(height)
        bar.setFixedWidth(8)
        bar_layout.addWidget(bar)
    bar_layout.addStretch()
    system_card.add_widget(bar_chart)
    
    # วาง System Status card
    content_layout.addWidget(system_card, row, 0, 1, 2)
    row += 1
    
    # ========== CARD: Special Functions ==========
    special_card = StatusCard("Special Functions", "📊")
    
    # M402
    m402_layout = QHBoxLayout()
    m402_label = QLabel("M402: ON")
    m402_label.setStyleSheet("color: #888; font-size: 12px;")
    m402_off_btn = QPushButton("OFF")
    m402_off_btn.setStyleSheet("""
        QPushButton {
            background-color: #f44336;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 15px;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #d32f2f;
        }
    """)
    m402_on_btn = QPushButton("ON")
    m402_on_btn.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 15px;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
    """)
    m402_on_btn.setEnabled(True)
    m402_layout.addWidget(m402_label)
    m402_layout.addStretch()
    m402_layout.addWidget(m402_off_btn)
    m402_layout.addWidget(m402_on_btn)
    special_card.add_widget(create_status_row(m402_layout))
    
    # M403
    m403_layout = QHBoxLayout()
    m403_label = QLabel("M403: ON")
    m403_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
    m403_off_btn = QPushButton("OFF")
    m403_off_btn.setStyleSheet("""
        QPushButton {
            background-color: #f44336;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 15px;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #d32f2f;
        }
    """)
    m403_on_btn = QPushButton("ON")
    m403_on_btn.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 15px;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
    """)
    m403_on_btn.setEnabled(True)
    m403_layout.addWidget(m403_label)
    m403_layout.addStretch()
    m403_layout.addWidget(m403_off_btn)
    m403_layout.addWidget(m403_on_btn)
    special_card.add_widget(create_status_row(m403_layout))
    
    # วาง Special Functions card ไว้แถวสุดท้าย
    content_layout.addWidget(special_card, row, 0, 1, 2)
    row += 1
    
    widgets_dict['m402_off_btn'] = m402_off_btn
    widgets_dict['m402_on_btn'] = m402_on_btn
    widgets_dict['m403_off_btn'] = m403_off_btn
    widgets_dict['m403_on_btn'] = m403_on_btn
    
    # ========== CARD: Performance Overview ==========
    perf_card1 = StatusCard("Performance Overview", "🔄")
    
    total_cycles = QLabel("1,200")
    total_cycles.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
    total_cycles_label = QLabel("Total Cycles")
    total_cycles_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
    perf_card1.add_widget(total_cycles)
    perf_card1.add_widget(total_cycles_label)
    
    successful = QLabel("1,180")
    successful.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50;")
    successful_label = QLabel("Successful")
    successful_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
    perf_card1.add_widget(successful)
    perf_card1.add_widget(successful_label)
    
    errors_label = QLabel("Errors")
    errors_label.setStyleSheet("color: #aaaaaa; font-size: 11px; margin-top: 10px;")
    perf_card1.add_widget(errors_label)
    
    content_layout.addWidget(perf_card1, row, 0)
    
    widgets_dict['total_cycles'] = total_cycles
    widgets_dict['successful'] = successful
    
    # ========== CARD: Performance Overview (Detailed) ==========
    perf_card2 = StatusCard("Performance Overview", "📊")
    
    successful2 = QLabel("1,880")
    successful2.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50;")
    successful2_label = QLabel("Successful")
    successful2_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
    perf_card2.add_widget(successful2)
    perf_card2.add_widget(successful2_label)
    
    errors = QLabel("20")
    errors.setStyleSheet("font-size: 24px; font-weight: bold; color: #f44336;")
    errors_label2 = QLabel("Errors")
    errors_label2.setStyleSheet("color: #aaaaaa; font-size: 11px;")
    perf_card2.add_widget(errors)
    perf_card2.add_widget(errors_label2)
    
    success_rate = QLabel("98.3%")
    success_rate.setStyleSheet("font-size: 28px; font-weight: bold; color: #4CAF50; margin-top: 10px;")
    success_rate_label = QLabel("Success Rate")
    success_rate_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
    perf_card2.add_widget(success_rate)
    perf_card2.add_widget(success_rate_label)
    
    content_layout.addWidget(perf_card2, row, 1)
    
    widgets_dict['successful2'] = successful2
    widgets_dict['errors'] = errors
    widgets_dict['success_rate'] = success_rate
    
    
    scroll.setWidget(content_widget)
    main_layout.addWidget(scroll)
    
    # Add header widgets to dict (for backward compatibility)
    widgets_dict['modbus_rtu_dot'] = modbus_rtu_dot
    widgets_dict['modbus_rtu_text'] = modbus_rtu_text
    widgets_dict['camera_feed_dot'] = camera_feed_dot
    widgets_dict['camera_feed_text'] = camera_feed_text
    
    return dashboard, widgets_dict


def create_status_row(layout):
    """Helper function to create a status row widget"""
    widget = QWidget()
    widget.setLayout(layout)
    return widget

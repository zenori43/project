# -*- coding: utf-8 -*-
"""
Status Tab Component
สร้าง UI สำหรับ Status Tab (Modbus Status)
ใช้ Modbus Dashboard แบบใหม่
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QGroupBox
)
from PyQt5.QtCore import Qt
from gui.components.modbus_dashboard import create_modbus_dashboard


def create_status_tab():
    """
    สร้าง Status Tab (Modbus Status) - ใช้ Dashboard แบบใหม่
    Returns:
        tuple: (status_tab_widget, widgets_dict)
            - status_tab_widget: QWidget สำหรับ tab
            - widgets_dict: dict ที่เก็บ widgets ทั้งหมดที่ main window ต้องการอ้างอิง
    """
    # ใช้ dashboard ใหม่
    status_tab, dashboard_widgets = create_modbus_dashboard()
    
    # สร้าง widgets_dict ที่มี widgets ทั้งหมดที่ main_window.py ต้องการ
    from PyQt5.QtWidgets import QLabel
    
    widgets_dict = {}
    
    # Modbus Connection
    widgets_dict['modbus_connection_lamp'] = dashboard_widgets.get('modbus_rtu_dot', QLabel("●"))
    widgets_dict['modbus_connection_text'] = dashboard_widgets.get('modbus_rtu_text', QLabel("ไม่เชื่อมต่อ"))
    
    # Coils - สร้าง QLabel สำหรับแต่ละ coil (ใช้แทน toggle switches เพื่อความเข้ากันได้)
    coil_names = ['m100', 'm110', 'm120', 'm130', 'm140', 'm600', 'm700', 'm701', 
                  'm503', 'm505', 'm507', 'm90', 'm720', 'm721', 'm722', 'm750', 'm850']
    for coil in coil_names:
        coil_label = QLabel(coil.upper())
        coil_label.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        coil_label.setAlignment(Qt.AlignCenter)
        widgets_dict[coil + '_lamp'] = coil_label
    
    # Performance Metrics
    widgets_dict['total_images_processed'] = dashboard_widgets.get('total_cycles', QLabel("0"))
    widgets_dict['successful_detections'] = dashboard_widgets.get('successful', QLabel("0"))
    widgets_dict['error_count'] = dashboard_widgets.get('errors', QLabel("0"))
    widgets_dict['success_rate'] = dashboard_widgets.get('success_rate', QLabel("0.0%"))
    
    # Camera Status
    widgets_dict['usb_camera_status'] = dashboard_widgets.get('camera_status_text', QLabel("USB Camera: Disconnected"))
    widgets_dict['sentech_camera_status'] = QLabel("Sentech Camera: Disconnected")
    widgets_dict['sentech_camera_status'].setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
    
    # System Status
    widgets_dict['bottle_detection_status'] = QLabel("การตรวจจับขวด: พร้อม")
    widgets_dict['bottle_detection_status'].setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
    widgets_dict['cap_detection_status'] = QLabel("การตรวจจับฝา: พร้อม")
    widgets_dict['cap_detection_status'].setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
    widgets_dict['current_bottle_type_status'] = QLabel("ประเภทขวดปัจจุบัน: -")
    widgets_dict['current_bottle_type_status'].setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
    widgets_dict['processing_mode_status'] = QLabel("โหมดการประมวลผล: Manual")
    widgets_dict['processing_mode_status'].setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
    widgets_dict['queue_count_status'] = QLabel("จำนวนคิว: 0")
    widgets_dict['queue_count_status'].setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
    
    # Registers
    widgets_dict['d6004_lamp'] = QLabel("●")
    widgets_dict['d6004_lamp'].setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
    widgets_dict['d6004_text'] = QLabel("รอค่า")
    widgets_dict['d6004_text'].setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
    
    widgets_dict['d6007_lamp'] = QLabel("●")
    widgets_dict['d6007_lamp'].setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
    widgets_dict['d6007_text'] = QLabel("รอค่า")
    widgets_dict['d6007_text'].setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
    
    widgets_dict['d5002_lamp'] = QLabel("●")
    widgets_dict['d5002_lamp'].setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
    widgets_dict['d5002_text'] = QLabel("รอค่า")
    widgets_dict['d5002_text'].setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
    
    # M402-M406
    widgets_dict['m402_lamp'] = QLabel("●")
    widgets_dict['m402_lamp'].setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
    widgets_dict['m402_text'] = QLabel("OFF")
    widgets_dict['m402_text'].setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
    
    widgets_dict['m403_lamp'] = QLabel("●")
    widgets_dict['m403_lamp'].setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
    widgets_dict['m403_text'] = QLabel("OFF")
    widgets_dict['m403_text'].setStyleSheet("color: #e74c3c; font-size: 10px;")
    
    widgets_dict['m404_lamp'] = QLabel("●")
    widgets_dict['m404_lamp'].setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
    widgets_dict['m404_text'] = QLabel("OFF")
    widgets_dict['m404_text'].setStyleSheet("color: #e74c3c; font-size: 10px;")
    
    widgets_dict['m405_lamp'] = QLabel("●")
    widgets_dict['m405_lamp'].setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
    widgets_dict['m405_text'] = QLabel("OFF")
    widgets_dict['m405_text'].setStyleSheet("color: #e74c3c; font-size: 10px;")
    
    widgets_dict['m406_lamp'] = QLabel("●")
    widgets_dict['m406_lamp'].setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
    widgets_dict['m406_text'] = QLabel("OFF")
    widgets_dict['m406_text'].setStyleSheet("color: #e74c3c; font-size: 10px;")
    
    widgets_dict['m401_lamp'] = QLabel("●")
    widgets_dict['m401_lamp'].setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
    widgets_dict['m401_text'] = QLabel("รอค่า")
    widgets_dict['m401_text'].setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
    
    return status_tab, widgets_dict


def create_status_tab_old():
    """
    สร้าง Status Tab แบบเดิม (backup)
    """
    status_tab = QWidget()
    status_layout = QVBoxLayout(status_tab)
    
    # Modbus Status Title
    status_title = QLabel("🔌 สถานะ Modbus")
    status_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; padding: 15px;")
    status_title.setAlignment(Qt.AlignCenter)
    status_layout.addWidget(status_title)
    
    # Create status display area
    status_scroll = QScrollArea()
    status_scroll.setWidgetResizable(True)
    status_scroll.setMinimumSize(600, 500)
    status_scroll.setStyleSheet("border: 2px solid #34495e; background-color: #ecf0f1;")
    
    # Status content widget
    status_content = QWidget()
    status_content_layout = QVBoxLayout(status_content)
    
    # Modbus Connection Status
    connection_group = QGroupBox("🔗 การเชื่อมต่อ Modbus")
    connection_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
    connection_layout = QHBoxLayout(connection_group)
    connection_layout.setContentsMargins(10, 15, 10, 10)
    
    modbus_connection_lamp = QLabel("●")
    modbus_connection_lamp.setStyleSheet("color: #e74c3c; font-size: 24px; font-weight: bold;")
    modbus_connection_lamp.setAlignment(Qt.AlignCenter)
    connection_layout.addWidget(modbus_connection_lamp)
    
    modbus_connection_text = QLabel("ไม่เชื่อมต่อ")
    modbus_connection_text.setStyleSheet("color: #e74c3c; font-size: 14px; font-weight: bold;")
    connection_layout.addWidget(modbus_connection_text)
    
    status_content_layout.addWidget(connection_group)
    
    # Modbus Coils Status
    coils_group = QGroupBox("⚡ Coils Status")
    coils_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
    coils_layout = QVBoxLayout(coils_group)
    coils_layout.setContentsMargins(10, 15, 10, 10)
    coils_layout.setSpacing(10)
    
    # M100 - M130 (Bottle Types)
    bottle_types_layout = QHBoxLayout()
    bottle_types_layout.setSpacing(8)
    bottle_types_label = QLabel("ประเภทขวด:")
    bottle_types_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
    bottle_types_layout.addWidget(bottle_types_label)
    
    m100_lamp = QLabel("M100")
    m100_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m100_lamp.setAlignment(Qt.AlignCenter)
    bottle_types_layout.addWidget(m100_lamp)
    
    m110_lamp = QLabel("M110")
    m110_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m110_lamp.setAlignment(Qt.AlignCenter)
    bottle_types_layout.addWidget(m110_lamp)
    
    m120_lamp = QLabel("M120")
    m120_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m120_lamp.setAlignment(Qt.AlignCenter)
    bottle_types_layout.addWidget(m120_lamp)
    
    m130_lamp = QLabel("M130")
    m130_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m130_lamp.setAlignment(Qt.AlignCenter)
    bottle_types_layout.addWidget(m130_lamp)
    
    coils_layout.addLayout(bottle_types_layout)
    
    # M140 - M850 (Control Coils)
    control_coils_layout = QHBoxLayout()
    control_coils_layout.setSpacing(8)
    control_coils_label = QLabel("Control:")
    control_coils_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
    control_coils_layout.addWidget(control_coils_label)
    
    m140_lamp = QLabel("M140")
    m140_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m140_lamp.setAlignment(Qt.AlignCenter)
    control_coils_layout.addWidget(m140_lamp)
    
    m600_lamp = QLabel("M600")
    m600_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m600_lamp.setAlignment(Qt.AlignCenter)
    control_coils_layout.addWidget(m600_lamp)
    
    m700_lamp = QLabel("M700")
    m700_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m700_lamp.setAlignment(Qt.AlignCenter)
    control_coils_layout.addWidget(m700_lamp)
    
    m701_lamp = QLabel("M701")
    m701_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m701_lamp.setAlignment(Qt.AlignCenter)
    control_coils_layout.addWidget(m701_lamp)
    
    m750_lamp = QLabel("M750")
    m750_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m750_lamp.setAlignment(Qt.AlignCenter)
    control_coils_layout.addWidget(m750_lamp)
    
    m850_lamp = QLabel("M850")
    m850_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m850_lamp.setAlignment(Qt.AlignCenter)
    control_coils_layout.addWidget(m850_lamp)
    
    coils_layout.addLayout(control_coils_layout)
    
    # Gripper Control
    gripper_layout = QHBoxLayout()
    gripper_layout.setSpacing(8)
    gripper_label = QLabel("Gripper:")
    gripper_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
    gripper_layout.addWidget(gripper_label)
    
    m503_lamp = QLabel("M503")
    m503_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m503_lamp.setAlignment(Qt.AlignCenter)
    gripper_layout.addWidget(m503_lamp)
    
    m505_lamp = QLabel("M505")
    m505_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m505_lamp.setAlignment(Qt.AlignCenter)
    gripper_layout.addWidget(m505_lamp)
    
    m507_lamp = QLabel("M507")
    m507_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m507_lamp.setAlignment(Qt.AlignCenter)
    gripper_layout.addWidget(m507_lamp)
    
    coils_layout.addLayout(gripper_layout)
    
    # M90 - Mode Selection
    mode_coils_layout = QHBoxLayout()
    mode_coils_layout.setSpacing(8)
    mode_coils_label = QLabel("Mode:")
    mode_coils_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
    mode_coils_layout.addWidget(mode_coils_label)
    
    m90_lamp = QLabel("M90")
    m90_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m90_lamp.setAlignment(Qt.AlignCenter)
    mode_coils_layout.addWidget(m90_lamp)
    
    coils_layout.addLayout(mode_coils_layout)
    
    # M720-M722, M730-M735, M740-M742 (Additional Coils)
    additional_coils_layout = QHBoxLayout()
    additional_coils_layout.setSpacing(8)
    additional_coils_label = QLabel("Additional:")
    additional_coils_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
    additional_coils_layout.addWidget(additional_coils_label)
    
    # M720-M722
    m720_lamp = QLabel("M720")
    m720_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m720_lamp.setAlignment(Qt.AlignCenter)
    additional_coils_layout.addWidget(m720_lamp)
    
    m721_lamp = QLabel("M721")
    m721_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m721_lamp.setAlignment(Qt.AlignCenter)
    additional_coils_layout.addWidget(m721_lamp)
    
    m722_lamp = QLabel("M722")
    m722_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
    m722_lamp.setAlignment(Qt.AlignCenter)
    additional_coils_layout.addWidget(m722_lamp)
    
    coils_layout.addLayout(additional_coils_layout)
    
    status_content_layout.addWidget(coils_group)
    
    # Performance Statistics Group
    performance_group = QGroupBox("📈 สถิติประสิทธิภาพ")
    performance_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
    performance_layout = QVBoxLayout(performance_group)
    performance_layout.setContentsMargins(10, 15, 10, 10)
    performance_layout.setSpacing(10)
    
    total_images_processed = QLabel("จำนวนภาพที่ประมวลผล: 0")
    total_images_processed.setStyleSheet("color: #2c3e50; padding: 5px; font-size: 12px;")
    performance_layout.addWidget(total_images_processed)
    
    successful_detections = QLabel("การตรวจจับที่สำเร็จ: 0")
    successful_detections.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
    performance_layout.addWidget(successful_detections)
    
    error_count = QLabel("จำนวนข้อผิดพลาด: 0")
    error_count.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
    performance_layout.addWidget(error_count)
    
    success_rate = QLabel("อัตราความสำเร็จ: 0.0%")
    success_rate.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
    performance_layout.addWidget(success_rate)
    
    status_content_layout.addWidget(performance_group)
    
    # Camera Status Group
    camera_group = QGroupBox("📷 สถานะกล้อง")
    camera_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
    camera_layout = QVBoxLayout(camera_group)
    camera_layout.setContentsMargins(10, 15, 10, 10)
    camera_layout.setSpacing(10)
    
    usb_camera_status = QLabel("USB Camera: กำลังเริ่มต้น...")
    usb_camera_status.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
    camera_layout.addWidget(usb_camera_status)
    
    sentech_camera_status = QLabel("Sentech Camera: กำลังเริ่มต้น...")
    sentech_camera_status.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
    camera_layout.addWidget(sentech_camera_status)
    
    status_content_layout.addWidget(camera_group)
    
    # System Status Group
    system_group = QGroupBox("🖥️ สถานะระบบ")
    system_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
    system_layout = QVBoxLayout(system_group)
    system_layout.setContentsMargins(10, 15, 10, 10)
    system_layout.setSpacing(10)
    
    bottle_detection_status = QLabel("การตรวจจับขวด: พร้อม")
    bottle_detection_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
    system_layout.addWidget(bottle_detection_status)
    
    cap_detection_status = QLabel("การตรวจจับฝา: พร้อม")
    cap_detection_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
    system_layout.addWidget(cap_detection_status)
    
    current_bottle_type_status = QLabel("ประเภทขวดปัจจุบัน: -")
    current_bottle_type_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
    system_layout.addWidget(current_bottle_type_status)
    
    processing_mode_status = QLabel("โหมดการประมวลผล: Manual")
    processing_mode_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
    system_layout.addWidget(processing_mode_status)
    
    status_content_layout.addWidget(system_group)
    
    # Queue Status Group
    queue_group = QGroupBox("📋 สถานะคิว")
    queue_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
    queue_layout = QVBoxLayout(queue_group)
    queue_layout.setContentsMargins(10, 15, 10, 10)
    queue_layout.setSpacing(10)
    
    queue_count_status = QLabel("จำนวนคิว: 0")
    queue_count_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
    queue_layout.addWidget(queue_count_status)
    
    status_content_layout.addWidget(queue_group)
    
    # Modbus Registers Status Group
    registers_group = QGroupBox("📊 Registers Status")
    registers_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
    registers_layout = QVBoxLayout(registers_group)
    registers_layout.setContentsMargins(10, 15, 10, 10)
    registers_layout.setSpacing(10)
    
    # D6004 Status
    d6004_layout = QHBoxLayout()
    d6004_layout.setSpacing(10)
    d6004_label = QLabel("D6004:")
    d6004_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
    d6004_layout.addWidget(d6004_label)
    
    d6004_lamp = QLabel("●")
    d6004_lamp.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
    d6004_lamp.setAlignment(Qt.AlignCenter)
    d6004_layout.addWidget(d6004_lamp)
    
    d6004_text = QLabel("รอค่า")
    d6004_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
    d6004_layout.addWidget(d6004_text)
    
    registers_layout.addLayout(d6004_layout)
    
    # D6007 Status
    d6007_layout = QHBoxLayout()
    d6007_layout.setSpacing(10)
    d6007_label = QLabel("D6007:")
    d6007_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
    d6007_layout.addWidget(d6007_label)
    
    d6007_lamp = QLabel("●")
    d6007_lamp.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
    d6007_lamp.setAlignment(Qt.AlignCenter)
    d6007_layout.addWidget(d6007_lamp)
    
    d6007_text = QLabel("รอค่า")
    d6007_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
    d6007_layout.addWidget(d6007_text)
    
    registers_layout.addLayout(d6007_layout)
    
    # D5002 Status
    d5002_layout = QHBoxLayout()
    d5002_layout.setSpacing(10)
    d5002_label = QLabel("D5002:")
    d5002_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
    d5002_layout.addWidget(d5002_label)
    
    d5002_lamp = QLabel("●")
    d5002_lamp.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
    d5002_lamp.setAlignment(Qt.AlignCenter)
    d5002_layout.addWidget(d5002_lamp)
    
    d5002_text = QLabel("รอค่า")
    d5002_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
    d5002_layout.addWidget(d5002_text)
    
    registers_layout.addLayout(d5002_layout)
    
    # M402 Status (ใช้แทน D6006)
    m402_layout = QHBoxLayout()
    m402_layout.setSpacing(10)
    m402_label = QLabel("M402:")
    m402_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
    m402_layout.addWidget(m402_label)
    
    m402_lamp = QLabel("●")
    m402_lamp.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
    m402_lamp.setAlignment(Qt.AlignCenter)
    m402_layout.addWidget(m402_lamp)
    
    m402_text = QLabel("OFF")
    m402_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
    m402_layout.addWidget(m402_text)
    
    coils_layout.addLayout(m402_layout)
    
    # M403-M406 Status (ใช้แทน D6007)
    bottle_type_coils_label = QLabel("ประเภทขวด (Coils):")
    bottle_type_coils_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-top: 10px;")
    coils_layout.addWidget(bottle_type_coils_label)
    
    bottle_type_coils_layout = QHBoxLayout()
    bottle_type_coils_layout.setSpacing(8)
    
    # M403 - น้ำเต้าหู้รสดั้งเดิม
    m403_layout = QVBoxLayout()
    m403_label = QLabel("M403:")
    m403_label.setStyleSheet("font-size: 11px; font-weight: bold;")
    m403_lamp = QLabel("●")
    m403_lamp.setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
    m403_lamp.setAlignment(Qt.AlignCenter)
    m403_text = QLabel("OFF")
    m403_text.setStyleSheet("color: #e74c3c; font-size: 10px;")
    m403_layout.addWidget(m403_label)
    m403_layout.addWidget(m403_lamp)
    m403_layout.addWidget(m403_text)
    bottle_type_coils_layout.addLayout(m403_layout)
    
    # M404 - น้ำตาลน้อย 2%
    m404_layout = QVBoxLayout()
    m404_label = QLabel("M404:")
    m404_label.setStyleSheet("font-size: 11px; font-weight: bold;")
    m404_lamp = QLabel("●")
    m404_lamp.setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
    m404_lamp.setAlignment(Qt.AlignCenter)
    m404_text = QLabel("OFF")
    m404_text.setStyleSheet("color: #e74c3c; font-size: 10px;")
    m404_layout.addWidget(m404_label)
    m404_layout.addWidget(m404_lamp)
    m404_layout.addWidget(m404_text)
    bottle_type_coils_layout.addLayout(m404_layout)
    
    # M405 - ผสมเม็ดแมงลัก
    m405_layout = QVBoxLayout()
    m405_label = QLabel("M405:")
    m405_label.setStyleSheet("font-size: 11px; font-weight: bold;")
    m405_lamp = QLabel("●")
    m405_lamp.setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
    m405_lamp.setAlignment(Qt.AlignCenter)
    m405_text = QLabel("OFF")
    m405_text.setStyleSheet("color: #e74c3c; font-size: 10px;")
    m405_layout.addWidget(m405_label)
    m405_layout.addWidget(m405_lamp)
    m405_layout.addWidget(m405_text)
    bottle_type_coils_layout.addLayout(m405_layout)
    
    # M406 - NG เต็ม
    m406_layout = QVBoxLayout()
    m406_label = QLabel("M406:")
    m406_label.setStyleSheet("font-size: 11px; font-weight: bold;")
    m406_lamp = QLabel("●")
    m406_lamp.setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
    m406_lamp.setAlignment(Qt.AlignCenter)
    m406_text = QLabel("OFF")
    m406_text.setStyleSheet("color: #e74c3c; font-size: 10px;")
    m406_layout.addWidget(m406_label)
    m406_layout.addWidget(m406_lamp)
    m406_layout.addWidget(m406_text)
    bottle_type_coils_layout.addLayout(m406_layout)
    
    coils_layout.addLayout(bottle_type_coils_layout)
    
    # M401 Status (Coil)
    m401_layout = QHBoxLayout()
    m401_layout.setSpacing(10)
    m401_label = QLabel("M401:")
    m401_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
    m401_layout.addWidget(m401_label)
    
    m401_lamp = QLabel("●")
    m401_lamp.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
    m401_lamp.setAlignment(Qt.AlignCenter)
    m401_layout.addWidget(m401_lamp)
    
    m401_text = QLabel("รอค่า")
    m401_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
    m401_layout.addWidget(m401_text)
    
    registers_layout.addLayout(m401_layout)
    
    status_content_layout.addWidget(registers_group)
    
    # Add spacing and stretch
    status_content_layout.addSpacing(15)
    status_content_layout.addStretch()
    
    status_scroll.setWidget(status_content)
    status_layout.addWidget(status_scroll)
    
    # Create widgets dict for main window to reference
    widgets_dict = {
        'modbus_connection_lamp': modbus_connection_lamp,
        'modbus_connection_text': modbus_connection_text,
        'm100_lamp': m100_lamp,
        'm110_lamp': m110_lamp,
        'm120_lamp': m120_lamp,
        'm130_lamp': m130_lamp,
        'm140_lamp': m140_lamp,
        'm600_lamp': m600_lamp,
        'm700_lamp': m700_lamp,
        'm701_lamp': m701_lamp,
        'm503_lamp': m503_lamp,
        'm505_lamp': m505_lamp,
        'm507_lamp': m507_lamp,
        'm90_lamp': m90_lamp,
        'm720_lamp': m720_lamp,
        'm721_lamp': m721_lamp,
        'm722_lamp': m722_lamp,
        'm750_lamp': m750_lamp,
        'm850_lamp': m850_lamp,
        'total_images_processed': total_images_processed,
        'successful_detections': successful_detections,
        'error_count': error_count,
        'success_rate': success_rate,
        'usb_camera_status': usb_camera_status,
        'sentech_camera_status': sentech_camera_status,
        'bottle_detection_status': bottle_detection_status,
        'cap_detection_status': cap_detection_status,
        'current_bottle_type_status': current_bottle_type_status,
        'processing_mode_status': processing_mode_status,
        'queue_count_status': queue_count_status,
        'd6004_lamp': d6004_lamp,
        'd6004_text': d6004_text,
        'd6007_lamp': d6007_lamp,
        'd6007_text': d6007_text,
        'd5002_lamp': d5002_lamp,
        'd5002_text': d5002_text,
        'm402_lamp': m402_lamp,
        'm402_text': m402_text,
        'm401_lamp': m401_lamp,
        'm401_text': m401_text,
        'm403_lamp': m403_lamp,
        'm403_text': m403_text,
        'm404_lamp': m404_lamp,
        'm404_text': m404_text,
        'm405_lamp': m405_lamp,
        'm405_text': m405_text,
        'm406_lamp': m406_lamp,
        'm406_text': m406_text,
        'm403_lamp': m403_lamp,
        'm403_text': m403_text,
        'm404_lamp': m404_lamp,
        'm404_text': m404_text,
        'm405_lamp': m405_lamp,
        'm405_text': m405_text,
        'm406_lamp': m406_lamp,
        'm406_text': m406_text
    }
    
    return status_tab, widgets_dict


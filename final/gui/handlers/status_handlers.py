# -*- coding: utf-8 -*-
"""
Status Update Event Handlers
Handles all events related to status updates and display
"""

from PyQt5.QtWidgets import QMessageBox, QLabel
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5 import QtWidgets


class StatusHandlers:
    """Event handlers for status updates and display"""
    
    def __init__(self, gui_instance):
        """
        Initialize handlers with reference to GUI instance
        
        Args:
            gui_instance: Reference to BottleDetectionGUI instance
        """
        self.gui = gui_instance
        # รอแค่ตัวที่ ON (ที่ทำให้โผล่ dialog) เปลี่ยนเป็น OFF → reset M450 + ถ่ายรูปใหม่ (ไม่ต้องรอทั้ง 4 ตัว OFF)
        self._waiting_m40x_off_after_ok = False
        self._which_m40x_triggered = None  # 403, 404, 405 หรือ 406 ตาม D6007 value 100,200,300,400
    
    def update_all_status(self):
        """Update all status displays continuously"""
        try:
            # อัพเดท camera status
            self.update_camera_status()
            
            # อัพเดท modbus status
            self.update_modbus_connection_status()
            
            # อัพเดท performance stats
            self.update_performance_stats()
            
            # อัพเดท system status
            self.update_system_status()
            
        except Exception as e:
            print(f"❌ Error in update_all_status: {e}")
    
    def update_camera_status(self):
        """Update camera status continuously"""
        try:
            # USB Camera status
            if self.gui.usb_camera and self.gui.usb_camera.is_running:
                self.update_usb_camera_status("พร้อมใช้งาน", True)
            else:
                self.update_usb_camera_status("ไม่พร้อมใช้งาน", False)
            
            # Sentech Camera status
            if self.gui.sentech_camera and self.gui.sentech_camera._is_initialized:
                self.update_sentech_camera_status("พร้อมใช้งาน", True)
            else:
                self.update_sentech_camera_status("ไม่พร้อมใช้งาน", False)
                
        except Exception as e:
            print(f"❌ Error updating camera status: {e}")
    
    def update_modbus_connection_status(self):
        """Update Modbus connection status continuously"""
        try:
            if self.gui.modbus_thread and self.gui.modbus_thread.is_running:
                # ตรวจสอบการเชื่อมต่อ Modbus
                if self.gui.modbus_thread.modbus_client and self.gui.modbus_thread.modbus_client.is_socket_open():
                    self.update_modbus_connection_status_ui(True)
                else:
                    self.update_modbus_connection_status_ui(False)
            else:
                self.update_modbus_connection_status_ui(False)
        except Exception as e:
            print(f"❌ Error updating Modbus connection status: {e}")
    
    def update_system_status(self):
        """Update system status continuously"""
        try:
            # อัพเดทเวลาปัจจุบัน
            current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
            if hasattr(self.gui, 'system_time_label'):
                self.gui.system_time_label.setText(f"🕐 Current time: {current_time}")
            
            # อัพเดทสถานะการประมวลผล
            if hasattr(self.gui, 'processing_status_label'):
                if self.gui.current_image is not None:
                    self.gui.processing_status_label.setText("🔄 Processing image")
                    self.gui.processing_status_label.setStyleSheet("color: #f39c12; padding: 5px;")
                else:
                    self.gui.processing_status_label.setText("⏸️ No processing")
                    self.gui.processing_status_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
                    
        except Exception as e:
            print(f"❌ Error updating system status: {e}")
    
    def update_modbus_status(self, status):
        """Update Modbus status display"""
        self.gui.modbus_status_label.setText(f'📡 Modbus: {status}')
        
        # อัปเดตสถานะคิว
        if "เพิ่มคิวถ่ายภาพ" in status:
            # แยกตัวเลขคิวออกจากข้อความ
            try:
                queue_text = status.split("คิวปัจจุบัน: ")[1].split(")")[0]
                queue_count = int(queue_text)
                self.update_queue_status(queue_count)
                # อัปเดต status label หลัก
                self.gui.status_label.setText(f'📋 มีคิวถ่ายภาพ {queue_count} รายการ รอ M600 reset')
                self.gui.status_label.setStyleSheet("color: #e67e22; padding: 5px;")
            except:
                pass
        
        if "ยังมีคิวรออยู่" in status:
            try:
                queue_text = status.split("คิวรออยู่: ")[1]
                queue_count = int(queue_text)
                self.update_queue_status(queue_count)
            except:
                pass
        
        if "ไม่มีคิวถ่ายภาพ" in status:
            self.update_queue_status(0)
            # อัปเดต status label หลัก
            self.gui.status_label.setText('✅ ไม่มีคิวถ่ายภาพ พร้อมรับ M301 ใหม่')
            self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        
        # อัปเดตสีตามสถานะ
        if "สำเร็จ" in status or "เชื่อมต่อ" in status:
            self.gui.modbus_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            # Update Top Panel indicator
            if hasattr(self.gui, 'modbus_status_indicator'):
                self.gui.modbus_status_indicator.setStyleSheet("color: #27ae60; font-size: 16px; font-weight: bold;")
        elif "ข้อผิดพลาด" in status or "ไม่สามารถ" in status:
            self.gui.modbus_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            # Update Top Panel indicator
            if hasattr(self.gui, 'modbus_status_indicator'):
                self.gui.modbus_status_indicator.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
        elif "M511 ON" in status:
            self.gui.modbus_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            # Update Top Panel indicator
            if hasattr(self.gui, 'modbus_status_indicator'):
                self.gui.modbus_status_indicator.setStyleSheet("color: #27ae60; font-size: 16px; font-weight: bold;")
            self.gui.status_label.setText('✅ โปรแกรมเริ่มทำงาน - รอสัญญาณ M301')
            self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        elif "M511 OFF" in status:
            self.gui.modbus_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            # Update Top Panel indicator
            if hasattr(self.gui, 'modbus_status_indicator'):
                self.gui.modbus_status_indicator.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
            self.gui.status_label.setText('⏸️ โปรแกรมหยุดทำงาน (รอ M511 เพื่อเริ่มการทำงาน)')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        elif "M513 ON" in status:
            self.gui.modbus_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            # Update Top Panel indicator
            if hasattr(self.gui, 'modbus_status_indicator'):
                self.gui.modbus_status_indicator.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
            self.gui.status_label.setText('🛑 โปรแกรมหยุดการทำงาน')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        else:
            self.gui.modbus_status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            # Update Top Panel indicator (yellow/orange for waiting)
            if hasattr(self.gui, 'modbus_status_indicator'):
                self.gui.modbus_status_indicator.setStyleSheet("color: #f39c12; font-size: 16px; font-weight: bold;")
    
    def update_queue_status(self, queue_count):
        """Update queue status display"""
        if queue_count > 0:
            self.gui.queue_status_label.setText(f'📋 คิว: {queue_count}')
            self.gui.queue_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
            self.gui.queue_info_label.setText(f'📋 สถานะคิว: มีคิวรอ {queue_count} รายการ')
            self.gui.queue_info_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
        else:
            self.gui.queue_status_label.setText('📋 คิว: 0')
            self.gui.queue_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-weight: bold;")
            self.gui.queue_info_label.setText('📋 สถานะคิว: ไม่มีคิวรอ')
            self.gui.queue_info_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
        
        # Update status tab
        self.gui.queue_count_status.setText(f"จำนวนคิว: {queue_count}")
        if queue_count > 0:
            self.gui.queue_count_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 12px;")
        else:
            self.gui.queue_count_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
    
    def update_d6004_status(self, d6004_value):
        """Update D6004 status display"""
        if d6004_value is None:
            self.gui.d6004_status_label.setText('🔍 D6004: ไม่สามารถอ่านได้')
            self.gui.d6004_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6004", "ไม่สามารถอ่านได้", False)
        elif d6004_value == 200:
            self.gui.d6004_status_label.setText('✅ D6004: 200 (พร้อม Reset)')
            self.gui.d6004_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6004", "200", True)
        else:
            self.gui.d6004_status_label.setText(f'⏳ D6004: {d6004_value} (รอ 200)')
            self.gui.d6004_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6004", str(d6004_value), False)
    
    def update_d6007_status(self, d6007_value):
        """Update D6007 status display"""
        if d6007_value is None:
            self.gui.d6007_status_label.setText('🔍 D6007: ไม่สามารถอ่านได้')
            self.gui.d6007_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", "ไม่สามารถอ่านได้", False)
        elif d6007_value == 100:
            self.gui.d6007_status_label.setText('🥛 D6007: 100 (น้ำเต้าหู้รสดั้งเดิม)')
            self.gui.d6007_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", "100", True)
            # แสดง dialog แจ้งเตือนผู้ใช้
            self.show_d6007_dialog(100, "น้ำเต้าหู้รสดั้งเดิม", "🥛")
        elif d6007_value == 200:
            self.gui.d6007_status_label.setText('🍯 D6007: 200 (น้ำตาลน้อย 2%)')
            self.gui.d6007_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", "200", True)
            # แสดง dialog แจ้งเตือนผู้ใช้
            self.show_d6007_dialog(200, "น้ำตาลน้อย 2%", "🍯")
        elif d6007_value == 300:
            self.gui.d6007_status_label.setText('🌿 D6007: 300 (ผสมเม็ดแมงลัก)')
            self.gui.d6007_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", "300", True)
            # แสดง dialog แจ้งเตือนผู้ใช้
            self.show_d6007_dialog(300, "ผสมเม็ดแมงลัก", "🌿")
        elif d6007_value == 400:
            self.gui.d6007_status_label.setText('❌ D6007: 400 (NG เต็ม)')
            self.gui.d6007_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", "400", True)
            # แสดง dialog แจ้งเตือนผู้ใช้
            self.show_d6007_dialog(400, "NG เต็ม", "❌")
        else:
            self.gui.d6007_status_label.setText(f'⏳ D6007: {d6007_value} (รอ 100/200/300/400)')
            self.gui.d6007_status_label.setStyleSheet("color: #9b59b6; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", str(d6007_value), False)
    
    def update_d5002_status(self, d5002_value):
        """Update D5002 status display (0: Close, 1: Running, 2: Break Point, 3: Pause, 4: Pre-run)"""
        # อัปเดต label ใน main status bar (ข้างๆโหมด) และ top panel
        header = getattr(self.gui, 'd5002_status_header', None)
        top_panel = getattr(self.gui, 'top_panel_robot_status', None)
        targets = [t for t in (header, top_panel) if t is not None]
        if not targets:
            return
        if d5002_value is None:
            text = '🔍 Waiting for D5002'
            style = """
                QLabel {
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 5px;
                    background-color: #ecf0f1;
                    border: 2px solid #bdc3c7;
                    color: #7f8c8d;
                }
            """
        elif d5002_value == 0:
            text = '🔴 Close'
            style = """
                QLabel {
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 5px;
                    background-color: #fee;
                    border: 2px solid #e74c3c;
                    color: #e74c3c;
                }
            """
        elif d5002_value == 1:
            text = '🟢 Running'
            style = """
                QLabel {
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 5px;
                    background-color: #d5f4e6;
                    border: 2px solid #27ae60;
                    color: #27ae60;
                }
            """
        elif d5002_value == 2:
            text = '🟡 Break Point'
            style = """
                QLabel {
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 5px;
                    background-color: #fef5e7;
                    border: 2px solid #f39c12;
                    color: #f39c12;
                }
            """
        elif d5002_value == 3:
            text = '🟠 Pause'
            style = """
                QLabel {
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 5px;
                    background-color: #fdebd0;
                    border: 2px solid #e67e22;
                    color: #e67e22;
                }
            """
        elif d5002_value == 4:
            text = '🔵 Pre-run'
            style = """
                QLabel {
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 5px;
                    background-color: #ebf5fb;
                    border: 2px solid #3498db;
                    color: #3498db;
                }
            """
        else:
            text = f'⏳ D5002: {d5002_value} (Unknown)'
            style = """
                QLabel {
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 5px;
                    background-color: #ecf0f1;
                    border: 2px solid #9b59b6;
                    color: #9b59b6;
                }
            """
        for w in targets:
            w.setText(text)
            w.setStyleSheet(style)
        
        # อัปเดต label ใน status tab (เดิม) — เช็คว่ามี widget ก่อนใช้
        lbl = getattr(self.gui, 'd5002_status_label', None)
        if lbl is not None:
            if d5002_value is None:
                lbl.setText('🔍 D5002: ไม่สามารถอ่านได้')
                lbl.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            elif d5002_value == 0:
                lbl.setText('🔴 D5002: 0 (Close)')
                lbl.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            elif d5002_value == 1:
                lbl.setText('🟢 D5002: 1 (Running)')
                lbl.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            elif d5002_value == 2:
                lbl.setText('🟡 D5002: 2 (Break Point)')
                lbl.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            elif d5002_value == 3:
                lbl.setText('🟠 D5002: 3 (Pause)')
                lbl.setStyleSheet("color: #e67e22; padding: 5px; font-size: 11px;")
            elif d5002_value == 4:
                lbl.setText('🔵 D5002: 4 (Pre-run)')
                lbl.setStyleSheet("color: #3498db; padding: 5px; font-size: 11px;")
            else:
                lbl.setText(f'⏳ D5002: {d5002_value} (Unknown)')
                lbl.setStyleSheet("color: #9b59b6; padding: 5px; font-size: 11px;")
        if d5002_value is None:
            self.update_register_lamp("d5002", "ไม่สามารถอ่านได้", False)
        elif d5002_value == 0:
            self.update_register_lamp("d5002", "Close", False)
        elif d5002_value == 1:
            self.update_register_lamp("d5002", "Running", True)
        elif d5002_value == 2:
            self.update_register_lamp("d5002", "Break Point", False)
        elif d5002_value == 3:
            self.update_register_lamp("d5002", "Pause", False)
        elif d5002_value == 4:
            self.update_register_lamp("d5002", "Pre-run", False)
        else:
            self.update_register_lamp("d5002", str(d5002_value), False)
    
    def update_d5001_status(self, d5001_value):
        """Update D5001 error code display"""
        header = getattr(self.gui, 'd5001_status_header', None)
        if header is None:
            return
        if d5001_value is None:
            header.setText('🔍 D5001: ไม่สามารถอ่านได้')
            header.setStyleSheet("""
                QLabel {
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 5px;
                    background-color: #ecf0f1;
                    border: 2px solid #bdc3c7;
                    color: #7f8c8d;
                }
            """)
            # ไม่มีข้อมูลชัดเจนเรื่อง error จากหุ่นยนต์ → ซ่อน Robot Alarm
            if hasattr(self.gui, 'set_robot_alarm_active'):
                self.gui.set_robot_alarm_active(False)
        elif d5001_value == 0:
            header.setText('✅ D5001: 0 (No Error)')
            header.setStyleSheet("""
                QLabel {
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 5px;
                    background-color: #d5f4e6;
                    border: 2px solid #27ae60;
                    color: #27ae60;
                }
            """)
            # D5001 = 0 → ไม่มี alarm จากหุ่นยนต์
            if hasattr(self.gui, 'set_robot_alarm_active'):
                self.gui.set_robot_alarm_active(False)
        else:
            header.setText(f'❌ D5001: {d5001_value} (Error)')
            header.setStyleSheet("""
                QLabel {
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 5px;
                    background-color: #fee;
                    border: 2px solid #e74c3c;
                    color: #e74c3c;
                }
            """)
            # มี error code จาก D5001 → แสดง Robot Alarm แถบแดงตัวอักษรวิ่งด้านบน
            if hasattr(self.gui, 'set_robot_alarm_active'):
                self.gui.set_robot_alarm_active(True)
    
    def show_d6007_dialog(self, value, bottle_type, emoji):
        """แสดง dialog แจ้งเตือนเมื่อ D6007 มีค่า 100, 200, 300, หรือ 400"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("📢 แจ้งเตือน - ตรวจพบประเภทขวด")
        msg.setText(f"{emoji} ตรวจพบขวด{bottle_type} (D6007 = {value})")
        
        # ข้อความแจ้งเตือนที่แตกต่างกันตามประเภทขวด
        if value == 100:
            msg.setInformativeText("🥛 ขวดน้ำเต้าหู้รสดั้งเดิม เต็ม\n\nกรุณาเดินไปเอาขวดออกจากสายการผลิต\n\nหลังจากเอาขวดออกแล้ว กด OK เพื่อถ่ายรูปใหม่")
        elif value == 200:
            msg.setInformativeText("🍯 ขวดน้ำตาลน้อย 2%\n\nกรุณาเดินไปเอาขวดออกจากสายการผลิต\n\nหลังจากเอาขวดออกแล้ว กด OK เพื่อถ่ายรูปใหม่")
        elif value == 300:
            msg.setInformativeText("🌿 ขวดผสมเม็ดแมงลัก\n\nกรุณาเดินไปเอาขวดออกจากสายการผลิต\n\nหลังจากเอาขวดออกแล้ว กด OK เพื่อถ่ายรูปใหม่")
        elif value == 400:
            msg.setInformativeText("❌ ขวด NG เต็ม\n\nกรุณาเดินไปเอาขวดออกจากสายการผลิต\n\nหลังจากเอาขวดออกแล้ว กด OK เพื่อถ่ายรูปใหม่")
        
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)
        
        # ตั้งค่าสไตล์ของ dialog
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #f8f9fa;
                color: #2c3e50;
            }
            QMessageBox QLabel {
                color: #2c3e50;
                font-size: 14px;
                padding: 10px;
            }
            QMessageBox QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        # แสดง dialog และรอการตอบกลับ
        result = msg.exec_()
        
        if result == QMessageBox.Ok:
            print(f"✅ ผู้ใช้กด OK - เข้า process ขวดเต็ม: ON M450, รอ M403/404/405/406 OFF แล้วค่อยถ่ายรูปใหม่ + reset M450")
            print(f"🔄 กำลัง ON M450 (รอ M403/404/405/406 OFF...)")
            # ตั้ง flag ว่าแถวเต็มทำแล้ว (ทำได้แค่ครั้ง 1 ต่อแถว)
            coil_num = {100: 403, 200: 404, 300: 405, 400: 406}.get(value)
            if coil_num:
                setattr(self.gui, f'_row_full_processed_{coil_num}', True)
                print(f"🏷️ ตั้ง flag: แถว M{coil_num} ทำแล้ว (ทำได้แค่ครั้ง 1 ต่อแถว)")
            if self.gui.modbus_thread:
                success = self.gui.modbus_thread.on_m450()
                if success:
                    self._waiting_m40x_off_after_ok = True
                    self._which_m40x_triggered = coil_num
                    print(f"✅ ON M450 สำเร็จ - รอ M{self._which_m40x_triggered} เป็น OFF แล้วจะ reset M450 และถ่ายรูปใหม่")
                else:
                    print(f"❌ ไม่สามารถ ON M450 ได้")
            else:
                print("❌ Modbus thread ไม่พร้อมใช้งาน")
        else:
            # ผู้ใช้กดปิด (X) หรือปิด dialog โดยไม่กด OK — ไม่ ON M450
            print(f"ℹ️ ผู้ใช้ปิด dialog (ไม่กด OK) - ไม่ ON M450")
    
    def update_usb_camera_status(self, status, is_connected=False):
        """Update USB camera status in status tab"""
        try:
            if not hasattr(self.gui, 'usb_camera_status'):
                return  # UI not ready yet
            if is_connected:
                self.gui.usb_camera_status.setText("USB Camera: พร้อมใช้งาน")
                self.gui.usb_camera_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            else:
                self.gui.usb_camera_status.setText(f"USB Camera: {status}")
                self.gui.usb_camera_status.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating USB camera status: {e}")
    
    def update_sentech_camera_status(self, status, is_connected=False):
        """Update Sentech camera status in status tab"""
        try:
            if not hasattr(self.gui, 'sentech_camera_status'):
                return  # UI not ready yet
            if is_connected:
                self.gui.sentech_camera_status.setText("Sentech Camera: พร้อมใช้งาน")
                self.gui.sentech_camera_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            else:
                self.gui.sentech_camera_status.setText(f"Sentech Camera: {status}")
                self.gui.sentech_camera_status.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating Sentech camera status: {e}")
    
    def update_modbus_connection_status_ui(self, is_connected=False):
        """Update Modbus connection status in status tab"""
        try:
            if not hasattr(self.gui, 'modbus_connection_lamp'):
                return  # UI not ready yet
            if is_connected:
                self.gui.modbus_connection_lamp.setText("●")
                self.gui.modbus_connection_lamp.setStyleSheet("color: #27ae60; font-size: 24px; font-weight: bold;")
                if hasattr(self.gui, 'modbus_connection_text'):
                    self.gui.modbus_connection_text.setText("เชื่อมต่อแล้ว")
                    self.gui.modbus_connection_text.setStyleSheet("color: #27ae60; font-size: 14px; font-weight: bold;")
            else:
                self.gui.modbus_connection_lamp.setText("●")
                self.gui.modbus_connection_lamp.setStyleSheet("color: #e74c3c; font-size: 24px; font-weight: bold;")
                if hasattr(self.gui, 'modbus_connection_text'):
                    self.gui.modbus_connection_text.setText("ไม่เชื่อมต่อ")
                    self.gui.modbus_connection_text.setStyleSheet("color: #e74c3c; font-size: 14px; font-weight: bold;")
        except Exception as e:
            print(f"❌ Error updating Modbus connection status: {e}")
    
    def update_coil_lamp(self, coil_name, is_on=False):
        """Update individual coil lamp status"""
        try:
            lamp_widget = getattr(self.gui, f"{coil_name.lower()}_lamp", None)
            if lamp_widget:
                if is_on:
                    lamp_widget.setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold; padding: 5px; border: 2px solid #27ae60; background-color: #d5f4e6;")
                else:
                    lamp_widget.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        except Exception as e:
            print(f"❌ Error updating coil lamp {coil_name}: {e}")
    
    def show_row_full_warning_dialog(self, coil_num, bottle_type, emoji):
        """แสดง dialog แจ้งเตือนเมื่อแถวเต็มแล้ว (ทำได้แค่ครั้ง 1 ต่อแถว)"""
        # กำหนดข้อความตาม coil_num
        if coil_num == 403:
            location = "หน้า"
        elif coil_num == 404:
            location = "หน้า"
        elif coil_num == 405:
            location = "ขวา"
        elif coil_num == 406:
            location = "หน้า"
        else:
            location = "แถว"
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("⚠️ แจ้งเตือน - แถวเต็ม")
        msg.setText(f"{emoji} ขวด{bottle_type} {location}เต็ม")
        msg.setInformativeText(f"⚠️ แถว{bottle_type}เต็มแล้ว (ทำได้แค่ครั้ง 1 ต่อแถว)\n\nกรุณาเอาขวดออกจากสายการผลิตก่อน\n\nหลังจากเอาขวดออกแล้ว กด OK เพื่อ ON M450 ตามปกติ")
        
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)
        
        # ตั้งค่าสไตล์ของ dialog
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #f8f9fa;
                color: #2c3e50;
            }
            QMessageBox QLabel {
                color: #2c3e50;
                font-size: 14px;
                padding: 10px;
            }
            QMessageBox QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background-color: #e67e22;
            }
            QMessageBox QPushButton[text="Cancel"] {
                background-color: #95a5a6;
            }
            QMessageBox QPushButton[text="Cancel"]:hover {
                background-color: #7f8c8d;
            }
        """)
        
        # แสดง dialog และรอการตอบกลับ
        result = msg.exec_()
        
        if result == QMessageBox.Ok:
            print(f"✅ ผู้ใช้กด OK - ON M450 ตามปกติ (แถว{bottle_type}เต็ม)")
            # Reset flag เพื่อให้ทำได้อีกครั้ง
            setattr(self.gui, f'_row_full_processed_{coil_num}', False)
            print(f"🔄 Reset flag: แถว M{coil_num} พร้อมทำอีกครั้ง")
            # ON M450 ตามปกติ
            if self.gui.modbus_thread:
                success = self.gui.modbus_thread.on_m450()
                if success:
                    self._waiting_m40x_off_after_ok = True
                    self._which_m40x_triggered = coil_num
                    print(f"✅ ON M450 สำเร็จ - รอ M{coil_num} เป็น OFF แล้วจะ reset M450 และถ่ายรูปใหม่")
                else:
                    print(f"❌ ไม่สามารถ ON M450 ได้")
            else:
                print("❌ Modbus thread ไม่พร้อมใช้งาน")
        else:
            # ผู้ใช้กด Cancel — ไม่ ON M450
            print(f"ℹ️ ผู้ใช้กด Cancel - ไม่ ON M450")
    
    def _check_m40x_off_then_trigger(self, coil_num):
        """เมื่อตัวที่ trigger (M403/404/405/406) เปลี่ยนเป็น OFF หลังกด OK → ล้างผลตรวจก่อน แล้ว reset M450 และถ่ายรูปใหม่"""
        if not self._waiting_m40x_off_after_ok or self._which_m40x_triggered is None:
            return
        if self._which_m40x_triggered != coil_num:
            return
        self._waiting_m40x_off_after_ok = False
        self._which_m40x_triggered = None
        print(f"✅ M{coil_num} OFF — ล้าง M600/M100-M140 ก่อน แล้ว reset M450 และใช้ผลเดิมหรือถ่ายรูปใหม่")
        # ล้าง coils ผลตรวจก่อน (กันค่าค้างจากขวดที่ส่งไปแล้ว)
        if self.gui.modbus_thread and hasattr(self.gui.modbus_thread, 'reset_result_coils_for_next_capture'):
            self.gui.modbus_thread.reset_result_coils_for_next_capture()
        # ปิดหลอดผลตรวจบน GUI ให้ตรงกับค่า
        for lamp in ('m600', 'm100', 'm110', 'm120', 'm130', 'm140'):
            self.update_coil_lamp(lamp, False)
        if self.gui.modbus_thread:
            self.gui.modbus_thread.reset_m450()
        if hasattr(self.gui, 'modbus_handlers') and self.gui.modbus_handlers:
            # ถ้ามีผลขวด+ฝาที่ส่งไปแล้ว (แถวเต็ม) → ใช้ผลนั้นเลย ไม่ถ่ายใหม่
            stored = getattr(self.gui, 'last_bottle_cap_result_for_reuse', None)
            if stored and len(stored) == 2:
                self.gui.modbus_handlers.reuse_slot_full_result()
            else:
                self.gui.modbus_handlers.on_queue_trigger()
    
    def update_m401_status(self, m401_value):
        """Update M401 status display"""
        try:
            if m401_value:
                # M401 = ON
                self.gui.m401_lamp.setText("●")
                self.gui.m401_lamp.setStyleSheet("color: #27ae60; font-size: 20px; font-weight: bold;")
                self.gui.m401_text.setText("ON")
                self.gui.m401_text.setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold;")
            else:
                # M401 = OFF
                self.gui.m401_lamp.setText("●")
                self.gui.m401_lamp.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
                self.gui.m401_text.setText("OFF")
                self.gui.m401_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
        except Exception as e:
            print(f"❌ Error updating M401 status: {e}")
    
    def update_m402_status(self, m402_value):
        """Update M402 status display"""
        try:
            if m402_value:
                self.gui.m402_lamp.setText("●")
                self.gui.m402_lamp.setStyleSheet("color: #27ae60; font-size: 20px; font-weight: bold;")
                self.gui.m402_text.setText("ON")
                self.gui.m402_text.setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold;")
            else:
                self.gui.m402_lamp.setText("●")
                self.gui.m402_lamp.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
                self.gui.m402_text.setText("OFF")
                self.gui.m402_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
        except Exception as e:
            print(f"❌ Error updating M402 status: {e}")
    
    def update_m403_status(self, m403_value):
        """Update M403 status display (น้ำเต้าหู้รสดั้งเดิม - แทน D6007=100)"""
        try:
            if m403_value:
                self.gui.m403_lamp.setStyleSheet("color: #27ae60; font-size: 18px; font-weight: bold;")
                self.gui.m403_text.setText("ON")
                self.gui.m403_text.setStyleSheet("color: #27ae60; font-size: 10px;")
                
                # ตรวจสอบว่าแถวเต็มเคยทำแล้วหรือยัง (ทำได้แค่ครั้ง 1 ต่อแถว)
                row_full_processed = getattr(self.gui, '_row_full_processed_403', False)
                if row_full_processed:
                    # แถวเต็มแล้ว → แสดง dialog แจ้งเตือนให้เอาขวดออกก่อน
                    self.show_row_full_warning_dialog(403, "น้ำเต้าหู้รสดั้งเดิม", "🥛")
                    return
                
                # แถวเต็ม → เก็บผลขวด+ฝาที่ส่งไปรอบนี้ไว้ใช้เมื่อกลับมา (ไม่ถ่ายใหม่)
                sent = getattr(self.gui, '_last_sent_bottle_cap_result', None)
                if sent and len(sent) == 2:
                    self.gui.last_bottle_cap_result_for_reuse = sent
                    # เก็บสถานะ Modbus ที่ส่งไปแล้วด้วย
                    last_signal = getattr(self.gui, 'last_sent_modbus_signal', None)
                    if last_signal:
                        self.gui.last_sent_modbus_signal_for_reuse = last_signal
                # แสดง dialog แจ้งเตือน (เหมือน D6007=100)
                self.show_d6007_dialog(100, "น้ำเต้าหู้รสดั้งเดิม", "🥛")
            else:
                self.gui.m403_lamp.setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
                self.gui.m403_text.setText("OFF")
                self.gui.m403_text.setStyleSheet("color: #e74c3c; font-size: 10px;")
                # Reset flag เมื่อ M403 OFF
                self.gui._row_full_processed_403 = False
                self._check_m40x_off_then_trigger(403)
        except Exception as e:
            pass  # ไม่ print log เพื่อไม่ให้รกตา
    
    def update_m404_status(self, m404_value):
        """Update M404 status display (น้ำตาลน้อย 2% - แทน D6007=200)"""
        try:
            if m404_value:
                self.gui.m404_lamp.setStyleSheet("color: #27ae60; font-size: 18px; font-weight: bold;")
                self.gui.m404_text.setText("ON")
                self.gui.m404_text.setStyleSheet("color: #27ae60; font-size: 10px;")
                
                # ตรวจสอบว่าแถวเต็มเคยทำแล้วหรือยัง (ทำได้แค่ครั้ง 1 ต่อแถว)
                row_full_processed = getattr(self.gui, '_row_full_processed_404', False)
                if row_full_processed:
                    # แถวเต็มแล้ว → แสดง dialog แจ้งเตือนให้เอาขวดออกก่อน
                    self.show_row_full_warning_dialog(404, "น้ำตาลน้อย 2%", "🍯")
                    return
                
                sent = getattr(self.gui, '_last_sent_bottle_cap_result', None)
                if sent and len(sent) == 2:
                    self.gui.last_bottle_cap_result_for_reuse = sent
                    # เก็บสถานะ Modbus ที่ส่งไปแล้วด้วย
                    last_signal = getattr(self.gui, 'last_sent_modbus_signal', None)
                    if last_signal:
                        self.gui.last_sent_modbus_signal_for_reuse = last_signal
                self.show_d6007_dialog(200, "น้ำตาลน้อย 2%", "🍯")
            else:
                self.gui.m404_lamp.setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
                self.gui.m404_text.setText("OFF")
                self.gui.m404_text.setStyleSheet("color: #e74c3c; font-size: 10px;")
                # Reset flag เมื่อ M404 OFF
                self.gui._row_full_processed_404 = False
                self._check_m40x_off_then_trigger(404)
        except Exception as e:
            pass  # ไม่ print log เพื่อไม่ให้รกตา
    
    def update_m405_status(self, m405_value):
        """Update M405 status display (ผสมเม็ดแมงลัก - แทน D6007=300)"""
        try:
            if m405_value:
                self.gui.m405_lamp.setStyleSheet("color: #27ae60; font-size: 18px; font-weight: bold;")
                self.gui.m405_text.setText("ON")
                self.gui.m405_text.setStyleSheet("color: #27ae60; font-size: 10px;")
                
                # ตรวจสอบว่าแถวเต็มเคยทำแล้วหรือยัง (ทำได้แค่ครั้ง 1 ต่อแถว)
                row_full_processed = getattr(self.gui, '_row_full_processed_405', False)
                if row_full_processed:
                    # แถวเต็มแล้ว → แสดง dialog แจ้งเตือนให้เอาขวดออกก่อน
                    self.show_row_full_warning_dialog(405, "ผสมเม็ดแมงลัก", "🌿")
                    return
                
                sent = getattr(self.gui, '_last_sent_bottle_cap_result', None)
                if sent and len(sent) == 2:
                    self.gui.last_bottle_cap_result_for_reuse = sent
                    # เก็บสถานะ Modbus ที่ส่งไปแล้วด้วย
                    last_signal = getattr(self.gui, 'last_sent_modbus_signal', None)
                    if last_signal:
                        self.gui.last_sent_modbus_signal_for_reuse = last_signal
                self.show_d6007_dialog(300, "ผสมเม็ดแมงลัก", "🌿")
            else:
                self.gui.m405_lamp.setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
                self.gui.m405_text.setText("OFF")
                self.gui.m405_text.setStyleSheet("color: #e74c3c; font-size: 10px;")
                # Reset flag เมื่อ M405 OFF
                self.gui._row_full_processed_405 = False
                self._check_m40x_off_then_trigger(405)
        except Exception as e:
            pass  # ไม่ print log เพื่อไม่ให้รกตา
    
    def update_m406_status(self, m406_value):
        """Update M406 status display (NG เต็ม - แทน D6007=400)"""
        try:
            if m406_value:
                self.gui.m406_lamp.setStyleSheet("color: #27ae60; font-size: 18px; font-weight: bold;")
                self.gui.m406_text.setText("ON")
                self.gui.m406_text.setStyleSheet("color: #27ae60; font-size: 10px;")
                
                # ตรวจสอบว่าแถวเต็มเคยทำแล้วหรือยัง (ทำได้แค่ครั้ง 1 ต่อแถว)
                row_full_processed = getattr(self.gui, '_row_full_processed_406', False)
                if row_full_processed:
                    # แถวเต็มแล้ว → แสดง dialog แจ้งเตือนให้เอาขวดออกก่อน
                    self.show_row_full_warning_dialog(406, "NG เต็ม", "❌")
                    return
                
                sent = getattr(self.gui, '_last_sent_bottle_cap_result', None)
                if sent and len(sent) == 2:
                    self.gui.last_bottle_cap_result_for_reuse = sent
                    # เก็บสถานะ Modbus ที่ส่งไปแล้วด้วย
                    last_signal = getattr(self.gui, 'last_sent_modbus_signal', None)
                    if last_signal:
                        self.gui.last_sent_modbus_signal_for_reuse = last_signal
                self.show_d6007_dialog(400, "NG เต็ม", "❌")
            else:
                self.gui.m406_lamp.setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
                self.gui.m406_text.setText("OFF")
                self.gui.m406_text.setStyleSheet("color: #e74c3c; font-size: 10px;")
                # Reset flag เมื่อ M406 OFF
                self.gui._row_full_processed_406 = False
                self._check_m40x_off_then_trigger(406)
        except Exception as e:
            pass  # ไม่ print log เพื่อไม่ให้รกตา
    
    def update_register_lamp(self, register_name, value, is_active=False):
        """Update register lamp and text status"""
        try:
            lamp_widget = getattr(self.gui, f"{register_name.lower()}_lamp", None)
            text_widget = getattr(self.gui, f"{register_name.lower()}_text", None)
            
            if lamp_widget and text_widget:
                if is_active:
                    lamp_widget.setText("●")
                    lamp_widget.setStyleSheet("color: #27ae60; font-size: 20px; font-weight: bold;")
                    text_widget.setText(f"{value}")
                    text_widget.setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold;")
                else:
                    lamp_widget.setText("●")
                    lamp_widget.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
                    text_widget.setText(f"{value}" if value is not None else "รอค่า")
                    text_widget.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
        except Exception as e:
            print(f"❌ Error updating register lamp {register_name}: {e}")
    
    def update_bottle_detection_status(self, status, is_processing=False):
        """Update bottle detection status in status tab"""
        try:
            if is_processing:
                self.gui.bottle_detection_status.setText(f"การตรวจจับขวด: {status}")
                self.gui.bottle_detection_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 12px;")
            else:
                self.gui.bottle_detection_status.setText(f"การตรวจจับขวด: {status}")
                self.gui.bottle_detection_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating bottle detection status: {e}")
    
    def update_cap_detection_status(self, status, is_processing=False):
        """Update cap detection status in status tab"""
        try:
            if is_processing:
                self.gui.cap_detection_status.setText(f"การตรวจจับฝา: {status}")
                self.gui.cap_detection_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 12px;")
            else:
                self.gui.cap_detection_status.setText(f"การตรวจจับฝา: {status}")
                self.gui.cap_detection_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating cap detection status: {e}")
    
    def update_bottle_type_status(self, bottle_type, type_conf=None, angle1_conf=None, type_easyocr_mean=None):
        """Update current bottle type status in status tab (YOLO angle1/type + EasyOCR โซน type ถ้ามี)"""
        try:
            if bottle_type:
                line = f"ประเภทขวดปัจจุบัน: {bottle_type}"
                bits = []
                if angle1_conf is not None:
                    bits.append(f"angle1: {angle1_conf:.4f}")
                if type_conf is not None:
                    bits.append(f"type(YOLO): {type_conf:.4f}")
                if type_easyocr_mean is not None:
                    bits.append(f"type(EasyOCR): {type_easyocr_mean:.4f}")
                if bits:
                    line += "  ·  " + " · ".join(bits)
                self.gui.current_bottle_type_status.setText(line)
                self.gui.current_bottle_type_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            else:
                self.gui.current_bottle_type_status.setText("ประเภทขวดปัจจุบัน: -")
                self.gui.current_bottle_type_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating bottle type status: {e}")
    
    def update_processing_mode_status(self, mode):
        """Update processing mode status in status tab"""
        try:
            self.gui.current_processing_mode = mode
            self.gui.processing_mode_status.setText(f"โหมดการประมวลผล: {mode}")
            if mode == "Auto":
                self.gui.processing_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            else:
                self.gui.processing_mode_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating processing mode status: {e}")
    
    def update_performance_stats(self, images_processed=None, successful_detections=None, errors=None):
        """Update performance statistics in status tab"""
        try:
            if images_processed is not None:
                self.gui.total_images_processed_count = images_processed
            if successful_detections is not None:
                self.gui.successful_detections_count = successful_detections
            if errors is not None:
                self.gui.error_count_value = errors
            
            # อัพเดท display แบบ real-time
            if hasattr(self.gui, 'total_images_processed'):
                self.gui.total_images_processed.setText(f"จำนวนภาพที่ประมวลผล: {self.gui.total_images_processed_count}")
                self.gui.total_images_processed.setStyleSheet("color: #2c3e50; padding: 5px; font-size: 12px;")
            
            if hasattr(self.gui, 'successful_detections'):
                self.gui.successful_detections.setText(f"การตรวจจับที่สำเร็จ: {self.gui.successful_detections_count}")
                self.gui.successful_detections.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            
            if hasattr(self.gui, 'error_count'):
                self.gui.error_count.setText(f"จำนวนข้อผิดพลาด: {self.gui.error_count_value}")
                if self.gui.error_count_value > 0:
                    self.gui.error_count.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
                else:
                    self.gui.error_count.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            
            # คำนวณ success rate
            if hasattr(self.gui, 'success_rate') and self.gui.total_images_processed_count > 0:
                success_rate = (self.gui.successful_detections_count / self.gui.total_images_processed_count) * 100
                self.gui.success_rate.setText(f"อัตราความสำเร็จ: {success_rate:.1f}%")
                if success_rate >= 90:
                    self.gui.success_rate.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
                elif success_rate >= 70:
                    self.gui.success_rate.setStyleSheet("color: #f39c12; padding: 5px; font-size: 12px;")
                else:
                    self.gui.success_rate.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
                
        except Exception as e:
            print(f"❌ Error updating performance stats: {e}")


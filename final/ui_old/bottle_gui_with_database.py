#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ตัวอย่างการใช้งาน Database Manager ใน bottle_gui.py
"""

# เพิ่ม import สำหรับ database
from database_manager import (
    save_detection_to_database,
    save_gripper_action_to_database,
    save_system_status_to_database
)
import datetime

# ตัวอย่างการเพิ่มโค้ดใน bottle_gui.py

class BottleDetectionGUI(QWidget):
    def __init__(self):
        super().__init__()
        # ... existing code ...
        
        # เพิ่มตัวแปรสำหรับ database
        self.database_enabled = True  # สามารถเปิด/ปิดการใช้งาน database ได้
        
    def on_processing_complete(self, result):
        """Handle processing completion - เพิ่มการบันทึกข้อมูลลง database"""
        print("✅ PROCESS COMPLETE: Processing finished")
        self.progress_bar.setVisible(False)
        
        if "error" not in result:
            print("✅ PROCESS COMPLETE: No error in result")
            print(f"✅ PROCESS COMPLETE: Result keys: {list(result.keys())}")
            
            # บันทึกข้อมูลลง database (ถ้าเปิดใช้งาน)
            if self.database_enabled:
                try:
                    success = save_detection_to_database(result)
                    if success:
                        print("✅ DATABASE: บันทึกข้อมูลการตรวจจับสำเร็จ")
                    else:
                        print("❌ DATABASE: บันทึกข้อมูลการตรวจจับไม่สำเร็จ")
                except Exception as e:
                    print(f"❌ DATABASE ERROR: {e}")
            
            # ... existing code for displaying results ...
            
            # Handle bottle type detection and Modbus control
            if result.get('bottle_type') and result.get('combined_ocr_text'):
                print(f"🎯 PROCESS COMPLETE: Bottle type detected: {result['bottle_type']}")
                print(f"📝 PROCESS COMPLETE: OCR text: '{result['combined_ocr_text']}'")
                self.handle_bottle_type_detection(result['bottle_type'], result['combined_ocr_text'])
            else:
                print("❌ PROCESS COMPLETE: No bottle type or OCR text found")

            self.status_label.setText('✅ ประมวลผลเสร็จสิ้น')
            self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        else:
            print(f"❌ PROCESS COMPLETE: Error in result: {result['error']}")
            self.status_label.setText(f'❌ ข้อผิดพลาด: {result["error"]}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def open_gripper(self):
        """เปิด Gripper โดยส่งสัญญาณ M503 - เพิ่มการบันทึกข้อมูลลง database"""
        if self.modbus_thread:
            success = self.modbus_thread.on_m503()
            if success:
                self.gripper_status_label.setText('🟢 Gripper: เปิด (M503 ON)')
                self.gripper_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-weight: bold;")
                print("✅ GRIPPER: M503 ON - เปิด Gripper")
                
                # บันทึกการทำงานลง database
                if self.database_enabled:
                    try:
                        db_success = save_gripper_action_to_database(
                            action='open',
                            success=True,
                            modbus_signal='M503'
                        )
                        if db_success:
                            print("✅ DATABASE: บันทึกการเปิด Gripper สำเร็จ")
                        else:
                            print("❌ DATABASE: บันทึกการเปิด Gripper ไม่สำเร็จ")
                    except Exception as e:
                        print(f"❌ DATABASE ERROR: {e}")
                
                # Reset M503 หลังจาก 0.5 วินาที (Push button behavior)
                QTimer.singleShot(500, self.reset_gripper_open)
            else:
                self.gripper_status_label.setText('❌ Gripper: ไม่สามารถเปิดได้')
                self.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
                print("❌ GRIPPER: ไม่สามารถ ON M503 ได้")
                
                # บันทึกการทำงานที่ไม่สำเร็จลง database
                if self.database_enabled:
                    try:
                        save_gripper_action_to_database(
                            action='open',
                            success=False,
                            modbus_signal='M503'
                        )
                    except Exception as e:
                        print(f"❌ DATABASE ERROR: {e}")
        else:
            self.gripper_status_label.setText('❌ Modbus ไม่พร้อมใช้งาน')
            self.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
            print("❌ GRIPPER: Modbus thread ไม่พร้อมใช้งาน")
    
    def close_gripper(self):
        """ปิด Gripper โดยส่งสัญญาณ M505 - เพิ่มการบันทึกข้อมูลลง database"""
        if self.modbus_thread:
            success = self.modbus_thread.on_m505()
            if success:
                self.gripper_status_label.setText('🔴 Gripper: ปิด (M505 ON)')
                self.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
                print("✅ GRIPPER: M505 ON - ปิด Gripper")
                
                # บันทึกการทำงานลง database
                if self.database_enabled:
                    try:
                        db_success = save_gripper_action_to_database(
                            action='close',
                            success=True,
                            modbus_signal='M505'
                        )
                        if db_success:
                            print("✅ DATABASE: บันทึกการปิด Gripper สำเร็จ")
                        else:
                            print("❌ DATABASE: บันทึกการปิด Gripper ไม่สำเร็จ")
                    except Exception as e:
                        print(f"❌ DATABASE ERROR: {e}")
                
                # Reset M505 หลังจาก 0.5 วินาที (Push button behavior)
                QTimer.singleShot(500, self.reset_gripper_close)
            else:
                self.gripper_status_label.setText('❌ Gripper: ไม่สามารถปิดได้')
                self.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
                print("❌ GRIPPER: ไม่สามารถ ON M505 ได้")
                
                # บันทึกการทำงานที่ไม่สำเร็จลง database
                if self.database_enabled:
                    try:
                        save_gripper_action_to_database(
                            action='close',
                            success=False,
                            modbus_signal='M505'
                        )
                    except Exception as e:
                        print(f"❌ DATABASE ERROR: {e}")
        else:
            self.gripper_status_label.setText('❌ Modbus ไม่พร้อมใช้งาน')
            self.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
            print("❌ GRIPPER: Modbus thread ไม่พร้อมใช้งาน")
    
    def update_modbus_status(self, status):
        """Update Modbus status display - เพิ่มการบันทึกสถานะลง database"""
        self.modbus_status_label.setText(f'📡 Modbus: {status}')
        
        # บันทึกสถานะลง database (เฉพาะสถานะสำคัญ)
        if self.database_enabled and any(keyword in status for keyword in ['เชื่อมต่อ', 'M511 ON', 'M511 OFF', 'M513 ON']):
            try:
                status_data = {
                    'modbus_status': status,
                    'camera_status': self.camera_status_label.text(),
                    'queue_count': self.get_queue_count(),
                    'auto_mode': self.checkbox_auto_mode.isChecked(),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                save_system_status_to_database(status_data)
                print("✅ DATABASE: บันทึกสถานะระบบสำเร็จ")
            except Exception as e:
                print(f"❌ DATABASE ERROR: {e}")
        
        # ... existing code for updating status ...
    
    def get_queue_count(self):
        """ดึงจำนวนคิวปัจจุบัน"""
        try:
            # ดึงจำนวนคิวจาก queue_status_label
            queue_text = self.queue_status_label.text()
            if 'คิว:' in queue_text:
                count_text = queue_text.split('คิว:')[1].strip()
                return int(count_text)
            return 0
        except:
            return 0
    
    def toggle_database(self):
        """เปิด/ปิดการใช้งาน database"""
        self.database_enabled = not self.database_enabled
        status = "เปิดใช้งาน" if self.database_enabled else "ปิดใช้งาน"
        print(f"🔧 DATABASE: {status}")
        
        # อัปเดตสถานะใน GUI
        if hasattr(self, 'database_status_label'):
            self.database_status_label.setText(f'🗄️ Database: {status}')
            color = "#27ae60" if self.database_enabled else "#e74c3c"
            self.database_status_label.setStyleSheet(f"color: {color}; padding: 5px;")
    
    def closeEvent(self, event):
        """Handle window close event - เพิ่มการบันทึกสถานะสุดท้าย"""
        # บันทึกสถานะสุดท้ายลง database
        if self.database_enabled:
            try:
                final_status_data = {
                    'event': 'application_closed',
                    'camera_status': self.camera_status_label.text(),
                    'modbus_status': self.modbus_status_label.text(),
                    'queue_count': self.get_queue_count(),
                    'auto_mode': self.checkbox_auto_mode.isChecked(),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                save_system_status_to_database(final_status_data)
                print("✅ DATABASE: บันทึกสถานะสุดท้ายสำเร็จ")
            except Exception as e:
                print(f"❌ DATABASE ERROR: {e}")
        
        # Stop Modbus thread
        if self.modbus_thread:
            self.modbus_thread.stop()
            self.modbus_thread.wait(1000)
        
        # Close camera
        if self.usb_camera:
            self.usb_camera.close_camera()
        
        event.accept()

# ตัวอย่างการเพิ่มปุ่มควบคุม database ใน GUI
def add_database_controls_to_gui(self):
    """เพิ่มปุ่มควบคุม database ใน GUI"""
    
    # สร้างปุ่มควบคุม database
    database_control_group = QGroupBox("ควบคุม Database")
    database_layout = QHBoxLayout()
    
    # ปุ่มเปิด/ปิด database
    self.btn_toggle_database = QPushButton('🗄️ เปิด/ปิด Database')
    self.btn_toggle_database.clicked.connect(self.toggle_database)
    self.btn_toggle_database.setStyleSheet("QPushButton { padding: 8px; font-size: 11px; background-color: #3498db; color: white; border-radius: 3px; }")
    database_layout.addWidget(self.btn_toggle_database)
    
    # สถานะ database
    self.database_status_label = QLabel('🗄️ Database: เปิดใช้งาน')
    self.database_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-weight: bold;")
    database_layout.addWidget(self.database_status_label)
    
    database_control_group.setLayout(database_layout)
    
    # เพิ่มใน layout หลัก (หลังจาก gripper controls)
    # self.layout().insertWidget(3, database_control_group)  # แทรกหลัง gripper controls

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    print("📝 ตัวอย่างการใช้งาน Database Manager ใน bottle_gui.py")
    print("=" * 60)
    print()
    print("🔧 การเพิ่มโค้ดใน bottle_gui.py:")
    print("1. เพิ่ม import database_manager")
    print("2. เพิ่มการบันทึกข้อมูลใน on_processing_complete()")
    print("3. เพิ่มการบันทึกการทำงาน gripper")
    print("4. เพิ่มการบันทึกสถานะระบบ")
    print("5. เพิ่มปุ่มควบคุม database (ถ้าต้องการ)")
    print()
    print("✅ ตัวอย่างโค้ดพร้อมใช้งานแล้ว!")

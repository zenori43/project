# -*- coding: utf-8 -*-
"""
Modbus Register Test Handlers
Event handlers สำหรับ Modbus Register Test Tab
"""

import time


class RobotTestHandlers:
    """Event handlers for Modbus Register Test operations"""
    
    def __init__(self, gui):
        self.gui = gui
        
    def log_result(self, message, is_error=False):
        """เพิ่มข้อความลงใน results text"""
        if self.gui.robot_results_text:
            color = "#e74c3c" if is_error else "#27ae60"
            timestamp = time.strftime("%H:%M:%S")
            self.gui.robot_results_text.append(
                f'<span style="color: #95a5a6;">[{timestamp}]</span> '
                f'<span style="color: {color};">{message}</span>'
            )
            # Auto scroll to bottom
            scrollbar = self.gui.robot_results_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def parse_address(self, address_str):
        """
        แปลง address string เป็น integer
        รองรับทั้ง hex (0x3101) และ decimal (12545)
        """
        try:
            address_str = address_str.strip()
            if address_str.startswith('0x') or address_str.startswith('0X'):
                # Hex format
                return int(address_str, 16)
            else:
                # Decimal format
                return int(address_str)
        except ValueError:
            return None
    
    def get_register_address(self):
        """ได้ register address จาก combo box หรือ custom input"""
        register_data = self.gui.robot_register_combo.currentData()
        if register_data and register_data != 0:
            # ใช้ address จาก combo box
            return register_data
        else:
            # ใช้ custom address
            address_str = self.gui.robot_address_input.text().strip()
            if not address_str:
                return None
            return self.parse_address(address_str)
    
    def update_value_description(self):
        """อัปเดตคำอธิบายค่าตาม register ที่เลือก"""
        register_data = self.gui.robot_register_combo.currentData()
        desc_text = ""
        
        if register_data == 5012:
            desc_text = "D5012: 1=Press (กด), 0=Release (ปล่อย)"
        elif register_data == 5500:
            desc_text = "D5500: 7=ID7 full auto, 5=ID5 capture only, 8=ID8 reset modbus"
        elif register_data == 9002:
            desc_text = "D9002: 1=M100+M110, 2=M100+M120, 3=M110+M120, 4=M110+M100, 5=M120+M110, 6=M120+M100"
        elif register_data == 9006:
            desc_text = "D9006: 10=3 tastes, 20=2 tastes, 30=1 taste"
        elif register_data == 7007:
            desc_text = "D7007: 1=M100, 2=M110, 3=M120"
        elif register_data == 7009:
            desc_text = "D7009: 10=M100, 20=M110, 30=M120, 40=M130, 50=ฝาไม่ผ่าน"
        else:
            desc_text = "ระบุค่าเอง (0-65535)"
        
        if self.gui.robot_value_desc_label:
            self.gui.robot_value_desc_label.setText(desc_text)
    
    def on_register_changed(self):
        """เมื่อเปลี่ยน register ใน combo box"""
        register_data = self.gui.robot_register_combo.currentData()
        
        # Enable/disable custom address input
        if register_data == 0:
            # Custom selected
            self.gui.robot_address_input.setEnabled(True)
        else:
            # Predefined register selected
            self.gui.robot_address_input.setEnabled(False)
            self.gui.robot_address_input.clear()
        
        # Update value description
        self.update_value_description()
        
        # Set default value based on register
        if register_data == 5012:
            self.gui.robot_value_input.setValue(1)  # Press
        elif register_data == 5500:
            self.gui.robot_value_input.setValue(7)  # ID7
        elif register_data == 9002:
            self.gui.robot_value_input.setValue(1)  # M100 + M110
        elif register_data == 9006:
            self.gui.robot_value_input.setValue(10)  # 3 tastes
        elif register_data == 7007:
            self.gui.robot_value_input.setValue(1)  # M100
        elif register_data == 7009:
            self.gui.robot_value_input.setValue(10)  # M100
    
    def write_register(self):
        """เขียนค่าไปที่ register"""
        try:
            if not self.gui.modbus_thread or not self.gui.modbus_thread.modbus_client:
                self.log_result("❌ Modbus ยังไม่ได้เชื่อมต่อ", is_error=True)
                return
            
            if not self.gui.modbus_thread.modbus_client.is_socket_open():
                self.log_result("❌ Modbus connection ไม่พร้อมใช้งาน", is_error=True)
                return
            
            # ได้ address
            address = self.get_register_address()
            if address is None:
                self.log_result("❌ กรุณาเลือก Register หรือระบุ Address", is_error=True)
                return
            
            # อ่านค่า
            value = self.gui.robot_value_input.value()
            
            self.log_result(f"🔄 กำลังเขียนค่า {value} ไปที่ D{address}...")
            
            # เขียนค่า
            result = self.gui.modbus_thread.write_register(address, value)
            
            if result:
                self.log_result(f"✅ เขียนค่า {value} สำเร็จที่ D{address}")
            else:
                self.log_result(f"❌ ไม่สามารถเขียนค่าได้ที่ D{address}", is_error=True)
                
        except Exception as e:
            self.log_result(f"❌ เกิดข้อผิดพลาดในการเขียนค่า: {str(e)}", is_error=True)
            import traceback
            self.log_result(traceback.format_exc(), is_error=True)
    
    def read_register(self):
        """อ่านค่าจาก register"""
        try:
            if not self.gui.modbus_thread or not self.gui.modbus_thread.modbus_client:
                self.log_result("❌ Modbus ยังไม่ได้เชื่อมต่อ", is_error=True)
                return
            
            if not self.gui.modbus_thread.modbus_client.is_socket_open():
                self.log_result("❌ Modbus connection ไม่พร้อมใช้งาน", is_error=True)
                return
            
            # ได้ address
            address = self.get_register_address()
            if address is None:
                self.log_result("❌ กรุณาเลือก Register หรือระบุ Address", is_error=True)
                return
            
            self.log_result(f"🔄 กำลังอ่านค่าจาก D{address}...")
            
            # อ่านค่า
            result = self.gui.modbus_thread.modbus_client.read_holding_registers(address, 1, unit=1)
            
            if result.isError():
                error_msg = str(result)
                self.log_result(f"❌ ไม่สามารถอ่านค่าได้: {error_msg}", is_error=True)
            else:
                read_value = result.registers[0]
                self.log_result(f"✅ อ่านค่าได้: {read_value} จาก D{address}")
                
        except Exception as e:
            self.log_result(f"❌ เกิดข้อผิดพลาดในการอ่านค่า: {str(e)}", is_error=True)
            import traceback
            self.log_result(traceback.format_exc(), is_error=True)
    
    def write_and_read(self):
        """เขียนค่าและอ่านค่ากลับมา"""
        try:
            if not self.gui.modbus_thread or not self.gui.modbus_thread.modbus_client:
                self.log_result("❌ Modbus ยังไม่ได้เชื่อมต่อ", is_error=True)
                return
            
            if not self.gui.modbus_thread.modbus_client.is_socket_open():
                self.log_result("❌ Modbus connection ไม่พร้อมใช้งาน", is_error=True)
                return
            
            # ได้ address
            address = self.get_register_address()
            if address is None:
                self.log_result("❌ กรุณาเลือก Register หรือระบุ Address", is_error=True)
                return
            
            # อ่านค่า
            value = self.gui.robot_value_input.value()
            
            # เขียนค่า
            self.log_result(f"🔄 กำลังเขียนค่า {value} ไปที่ D{address}...")
            write_result = self.gui.modbus_thread.write_register(address, value)
            
            if not write_result:
                self.log_result(f"❌ ไม่สามารถเขียนค่าได้ที่ D{address}", is_error=True)
                return
            
            self.log_result(f"✅ เขียนค่า {value} สำเร็จ!")
            
            # รอสักครู่แล้วอ่านค่ากลับมา
            time.sleep(0.5)
            self.log_result(f"🔄 กำลังอ่านค่ากลับมาเพื่อยืนยัน...")
            
            read_result = self.gui.modbus_thread.modbus_client.read_holding_registers(address, 1, unit=1)
            
            if read_result.isError():
                error_msg = str(read_result)
                self.log_result(f"⚠️ ไม่สามารถอ่านค่าได้: {error_msg}", is_error=True)
                self.log_result("   (แต่การเขียนอาจสำเร็จแล้ว)")
            else:
                read_value = read_result.registers[0]
                self.log_result(f"✅ อ่านค่าได้: {read_value}")
                
                if read_value == value:
                    self.log_result(f"✅ ยืนยันสำเร็จ: ค่าที่เขียนและอ่านตรงกัน ({value})")
                else:
                    self.log_result(f"⚠️ ค่าที่อ่านได้ไม่ตรง: เขียน {value} แต่อ่านได้ {read_value}")
            
        except Exception as e:
            self.log_result(f"❌ เกิดข้อผิดพลาด: {str(e)}", is_error=True)
            import traceback
            self.log_result(traceback.format_exc(), is_error=True)
    
    def clear_results(self):
        """ล้างผลลัพธ์"""
        if self.gui.robot_results_text:
            self.gui.robot_results_text.clear()

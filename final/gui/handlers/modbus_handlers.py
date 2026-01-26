# -*- coding: utf-8 -*-
"""
Modbus Event Handlers
Handles all events related to Modbus operations and signals
"""

from PyQt5.QtCore import QTimer, Qt
import cv2
import os
import datetime


class ModbusHandlers:
    """Event handlers for Modbus operations"""
    
    def __init__(self, gui_instance):
        """
        Initialize handlers with reference to GUI instance
        
        Args:
            gui_instance: Reference to BottleDetectionGUI instance
        """
        self.gui = gui_instance
    
    def on_modbus_trigger(self):
        """Handle Modbus trigger (M301)"""
        print("🎯 MODBUS TRIGGER RECEIVED - Starting capture sequence")
        # Auto mode is always enabled
        # Capture from both cameras simultaneously
        self.gui.bottle_handlers.capture_image_auto()
        self.gui.cap_handlers.capture_sentech_image_auto()
    
    def on_queue_trigger(self):
        """Handle queue trigger (from M600 reset)"""
        print("🎯 QUEUE TRIGGER RECEIVED - Starting capture from queue")
        # Auto mode is always enabled
        # Capture from both cameras simultaneously
        self.gui.bottle_handlers.capture_image_from_queue()
        self.gui.cap_handlers.capture_sentech_image_from_queue()
    
    def on_silent_trigger(self):
        """Handle silent trigger - capture and process silently (no UI display, no Modbus signals)"""
        print("🔇 SILENT TRIGGER RECEIVED - Starting silent capture and processing")
        # Capture and process silently (store results in queue, don't display UI, don't send signals)
        self.gui.bottle_handlers.capture_and_process_silently()
        self.gui.cap_handlers.capture_and_process_silently()
    
    def on_result_ready_to_display(self, bottle_result, cap_result):
        """Handle result ready to display - show results from queue and send Modbus signals"""
        print("📋 RESULT READY TO DISPLAY - Showing results from queue and sending Modbus signals")
        try:
            # Display bottle results
            if bottle_result and "error" not in bottle_result:
                print("📋 DISPLAY: Displaying bottle results from queue")
                self.gui.bottle_handlers.display_result_from_queue(bottle_result)
                
                # Handle bottle type detection and send Modbus signals
                if bottle_result.get('bottle_type') and bottle_result.get('combined_ocr_text'):
                    print(f"🎯 DISPLAY: Bottle type detected: {bottle_result['bottle_type']}")
                    self.gui.bottle_handlers.handle_bottle_type_detection(
                        bottle_result['bottle_type'], 
                        bottle_result['combined_ocr_text']
                    )
            
            # Display cap results
            if cap_result and "error" not in cap_result:
                print("📋 DISPLAY: Displaying cap results from queue")
                self.gui.cap_handlers.display_result_from_queue(cap_result)
                
                # Validate and send Modbus signals for cap
                if self.gui.current_bottle_type in ["M100", "M110", "M120"]:
                    is_valid = self.gui.cap_handlers.validate_cap_result(cap_result)
                    if is_valid:
                        print(f"✅ DISPLAY: Cap validation passed - Sending Modbus signals")
                        self.on_bottle_type_after_cap_validation()
                    else:
                        print(f"❌ DISPLAY: Cap validation failed - Writing D7009 = 50")
                        if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                            # M140 → D7009 = 50
                            m140_success = self.gui.modbus_thread.write_register(7009, 50)
                            if m140_success:
                                print("🏷️ D7009 = 50 (ฝาไม่ผ่าน)")
                                m600_success = self.gui.modbus_thread.on_m600()
                                if m600_success:
                                    self.gui.status_handlers.update_coil_lamp("m600", True)
            
            # Add to history
            if hasattr(self.gui, 'add_to_history'):
                self.gui.add_to_history(bottle_result, cap_result)
            
            print("✅ DISPLAY: Results displayed and Modbus signals sent")
            
        except Exception as e:
            print(f"❌ ERROR in on_result_ready_to_display: {e}")
            import traceback
            traceback.print_exc()
    
    def create_capture_folders(self):
        """สร้างโฟลเดอร์สำหรับเก็บภาพใน capture only mode"""
        try:
            # สร้างโฟลเดอร์หลักสำหรับเก็บภาพ
            main_folder = "captured_images"
            if not os.path.exists(main_folder):
                os.makedirs(main_folder)
                print(f"📁 สร้างโฟลเดอร์หลัก: {main_folder}")
            
            # สร้างโฟลเดอร์สำหรับกล้อง USB
            usb_folder = os.path.join(main_folder, "usb_camera")
            if not os.path.exists(usb_folder):
                os.makedirs(usb_folder)
                print(f"📁 สร้างโฟลเดอร์กล้อง USB: {usb_folder}")
            
            # สร้างโฟลเดอร์สำหรับกล้อง Sentech
            sentech_folder = os.path.join(main_folder, "sentech_camera")
            if not os.path.exists(sentech_folder):
                os.makedirs(sentech_folder)
                print(f"📁 สร้างโฟลเดอร์กล้อง Sentech: {sentech_folder}")
            
            self.gui.capture_only_usb_folder = usb_folder
            self.gui.capture_only_sentech_folder = sentech_folder
            
            return usb_folder, sentech_folder
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการสร้างโฟลเดอร์: {e}")
            return None, None
    
    def save_capture_image(self, image, prefix, folder_path):
        """บันทึกภาพลงในโฟลเดอร์ที่กำหนด"""
        try:
            if image is None:
                return None
            
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{now}.png"
            filepath = os.path.join(folder_path, filename)
            cv2.imwrite(filepath, image)
            print(f"💾 บันทึกภาพ: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการบันทึกภาพ: {e}")
            return None
    
    def on_capture_only_trigger(self):
        """Handle capture only trigger (ID 5) - capture images without processing"""
        print("📸 CAPTURE ONLY TRIGGER: ID 5 mode - ถ่ายภาพทันที (ไม่ประมวลผล)")
        try:
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            if self.gui.capture_only_usb_folder is None or self.gui.capture_only_sentech_folder is None:
                self.create_capture_folders()
            
            # ON M600 (M91 ON แล้วตอนเลือก ID 5 ไม่ต้อง ON อีก)
            if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                try:
                    # ON M600 (ใช้ modbus_client.write_coil() เพราะต้องส่ง unit parameter)
                    if self.gui.modbus_thread.modbus_client:
                        self.gui.modbus_thread.modbus_client.write_coil(600, True, unit=1)
                        print("✅ CAPTURE ONLY: ON M600 เรียบร้อย!")
                    else:
                        print("❌ CAPTURE ONLY: Modbus client ไม่พร้อมใช้งาน")
                except Exception as e:
                    print(f"❌ CAPTURE ONLY: ไม่สามารถ ON M600 ได้: {e}")
            
            # รอ 2 วินาทีแล้วถ่ายภาพ
            QTimer.singleShot(2000, self._capture_images_after_delay)
            
        except Exception as e:
            print(f"❌ CAPTURE ONLY ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def _capture_images_after_delay(self):
        """ถ่ายภาพหลังจากรอ 2 วินาที"""
        try:
            print("📸 CAPTURE ONLY: เริ่มถ่ายภาพ...")
            
            # ถ่ายภาพจากกล้อง USB
            if self.gui.usb_camera is not None:
                usb_image = self.gui.usb_camera.capture_image()
                if usb_image is not None:
                    print(f"📸 CAPTURE ONLY: Captured USB image: {usb_image.shape}")
                    if self.gui.capture_only_usb_folder:
                        self.save_capture_image(usb_image, "usb", self.gui.capture_only_usb_folder)
                    if hasattr(self.gui.modbus_thread, 'capture_image_count'):
                        self.gui.modbus_thread.capture_image_count += 1
                else:
                    print("❌ CAPTURE ONLY: ไม่สามารถจับภาพจาก USB ได้")
            
            # ถ่ายภาพจากกล้อง Sentech
            if self.gui.sentech_camera is not None:
                try:
                    sentech_image = self.gui.sentech_camera.capture_image()
                    if sentech_image is not None:
                        print(f"📸 CAPTURE ONLY: Captured Sentech image: {sentech_image.shape}")
                        if self.gui.capture_only_sentech_folder:
                            self.save_capture_image(sentech_image, "sentech", self.gui.capture_only_sentech_folder)
                        if hasattr(self.gui.modbus_thread, 'capture_image_count'):
                            self.gui.modbus_thread.capture_image_count += 1
                    else:
                        print("⚠️ CAPTURE ONLY: ไม่สามารถดึงภาพจาก Sentech ได้")
                except Exception as e:
                    print(f"❌ CAPTURE ONLY: เกิดข้อผิดพลาดในการถ่ายภาพ Sentech: {e}")
            
            # Set M600 reset pending และ ON M600 ต่อทันทีเพื่อรอ M401=ON ถัดไป
            if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                self.gui.modbus_thread.m600_reset_pending = True
                # ไม่ต้อง print log นี้ทุกครั้ง (ลด spam)
                
                # ON M600 ต่อทันทีเพื่อรอ M401 ON ถัดไป (สำหรับคิวถัดไป)
                try:
                    if self.gui.modbus_thread.modbus_client:
                        self.gui.modbus_thread.modbus_client.write_coil(600, True, unit=1)
                        print("✅ CAPTURE ONLY: ON M600 ต่อทันที (รอ M401 ON ถัดไป)")
                    else:
                        print("❌ CAPTURE ONLY: Modbus client ไม่พร้อมใช้งาน")
                except Exception as e:
                    print(f"❌ CAPTURE ONLY: ไม่สามารถ ON M600 ต่อได้: {e}")
            
            count = getattr(self.gui.modbus_thread, 'capture_image_count', 0) if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread else 0
            print(f"✅ CAPTURE ONLY: ถ่ายภาพเสร็จแล้ว (จำนวนภาพทั้งหมด: {count})")
            
            # ตรวจสอบโหมด capture with limit (ID 5)
            if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                if getattr(self.gui.modbus_thread, 'capture_with_limit_mode', False):
                    self.gui.modbus_thread.capture_current_count += 1
                    current = self.gui.modbus_thread.capture_current_count
                    limit = self.gui.modbus_thread.capture_limit_count
                    print(f"📊 CAPTURE WITH LIMIT: ถ่ายแล้ว {current}/{limit} ครั้ง")
                    
                    # ถ้าถ่ายครบจำนวนแล้ว ตั้ง flag หยุดการถ่ายภาพต่อ
                    if current >= limit and not self.gui.modbus_thread.capture_limit_reached:
                        self.gui.modbus_thread.capture_limit_reached = True
                        print(f"✅ CAPTURE WITH LIMIT: ถ่ายครบ {limit} ครั้งแล้ว - หยุดการถ่ายภาพต่อ (รอ M401 ON แล้วหยุดระบบหลัง 3 วินาที)")
                        self.gui.status_label.setText(f'✅ ถ่ายครบ {limit} ครั้งแล้ว - หยุดการถ่ายภาพต่อ (รอ M401 ON แล้วหยุดระบบ)')
                        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
        except Exception as e:
            print(f"❌ CAPTURE ONLY ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def on_capture_limit_reached(self):
        """Handle capture limit reached signal - หยุดระบบหลัง M401 ON 3 วินาที"""
        print("⏳ CAPTURE LIMIT REACHED: รอ 3 วินาทีหลัง M401 ON แล้วหยุดระบบ")
        try:
            # รอ 3 วินาทีแล้วหยุดระบบ
            QTimer.singleShot(3000, self.gui.stop_system)
        except Exception as e:
            print(f"❌ CAPTURE LIMIT REACHED ERROR: {e}")
    
    def on_m513_stop(self):
        """Handle M513 stop signal - reset everything like pressing Stop button"""
        print("🛑 M513 STOP: รับสัญญาณหยุดจาก M513 - เริ่ม reset ทั้งหมด")
        try:
            # อัปเดตสถานะ UI
            self.gui.status_label.setText('🛑 M513: หยุดการทำงาน - กำลัง reset ระบบ')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            
            # เรียกใช้ฟังก์ชัน reset ทั้งหมดเหมือนกดปุ่ม Stop
            self.reset_to_initial_state()
            
            print("✅ M513 STOP: Reset ระบบเสร็จสิ้น")
            
        except Exception as e:
            print(f"❌ M513 STOP: ข้อผิดพลาดในการ reset: {e}")
            self.gui.status_label.setText('❌ M513: ข้อผิดพลาดในการ reset ระบบ')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def on_mode_changed(self, index):
        """Handle mode selection change"""
        try:
            if self.gui.modbus_thread is None:
                print("⚠️ MODE CHANGE: Modbus thread not ready yet")
                return
            
            mode_text = self.gui.mode_combo.currentText()
            print(f"🔄 MODE CHANGE: Selected mode = {mode_text} (index: {index})")
            
            if mode_text == "ID 7 full auto":
                # Write 7 to D5500 when selecting "ID 7 full auto"
                success = self.gui.modbus_thread.write_register(5500, 7)
                if success:
                    print("✅ MODE CHANGE: D5500 = 7 (ID 7 full auto mode)")
                    # Disable capture only mode
                    self.gui.modbus_thread.capture_only_mode = False
                else:
                    print("❌ MODE CHANGE: Failed to write D5500 = 7")
            elif mode_text == "ID 5 capture only":
                # Enable capture only mode (no processing)
                self.gui.modbus_thread.capture_only_mode = True
                self.gui.modbus_thread.capture_with_limit_mode = True  # เปิดใช้งาน limit mode
                self.gui.modbus_thread.capture_limit_count = self.gui.capture_limit_spin.value()
                self.gui.modbus_thread.capture_current_count = 0
                self.gui.capture_limit_label.setVisible(True)
                self.gui.capture_limit_spin.setVisible(True)
                print(f"✅ MODE CHANGE: ID 5 capture only mode enabled (limit: {self.gui.modbus_thread.capture_limit_count} times)")
                # Write 5 to D5500 for ID 5 mode
                success = self.gui.modbus_thread.write_register(5500, 5)
                if success:
                    print(f"✅ MODE CHANGE: D5500 = 5 (ID 5 capture only mode - limit: {self.gui.modbus_thread.capture_limit_count} times)")
                else:
                    print("❌ MODE CHANGE: Failed to write D5500 = 5")
                # สร้างโฟลเดอร์สำหรับเก็บภาพ
                self.create_capture_folders()
            elif mode_text == "ID 8 reset modbus":
                # Write 8 to D5500 for ID 8 reset modbus mode
                success = self.gui.modbus_thread.write_register(5500, 8)
                if success:
                    print("✅ MODE CHANGE: D5500 = 8 (ID 8 reset modbus mode)")
                    # Reset modbus to initial state
                    print("🔄 MODE CHANGE: Resetting Modbus to initial state...")
                    self.reset_to_initial_state()
                    print("✅ MODE CHANGE: Modbus reset completed")
                else:
                    print("❌ MODE CHANGE: Failed to write D5500 = 8")
                # Disable capture only mode
                self.gui.modbus_thread.capture_only_mode = False
                self.gui.modbus_thread.capture_with_limit_mode = False
                self.gui.capture_limit_label.setVisible(False)
                self.gui.capture_limit_spin.setVisible(False)
            else:
                # For other modes, disable capture only mode
                self.gui.modbus_thread.capture_only_mode = False
                self.gui.modbus_thread.capture_with_limit_mode = False
                self.gui.capture_limit_label.setVisible(False)
                self.gui.capture_limit_spin.setVisible(False)
                print(f"⚠️ MODE CHANGE: Unknown mode: {mode_text}")
                    
        except Exception as e:
            print(f"❌ MODE CHANGE ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def on_bottle_type_after_cap_validation(self):
        """Write bottle type value to D7009 หลังจากตรวจสอบฝาผ่านแล้ว"""
        try:
            print(f"🏷️ WRITE BOTTLE TYPE TO D7009 AFTER CAP VALIDATION: {self.gui.current_bottle_type}")
            
            success = False
            if self.gui.current_bottle_type == "M100":
                # M100 → D7009 = 10
                success = self.gui.modbus_thread.write_register(7009, 10)
                print("🏷️ D7009 = 10 (พบคำว่า 'เดิม' - ฝาผ่าน)")
            elif self.gui.current_bottle_type == "M110":
                # M110 → D7009 = 20
                success = self.gui.modbus_thread.write_register(7009, 20)
                print("🏷️ D7009 = 20 (พบคำว่า '2%' - ฝาผ่าน)")
            elif self.gui.current_bottle_type == "M120":
                # M120 → D7009 = 30
                success = self.gui.modbus_thread.write_register(7009, 30)
                print("🏷️ D7009 = 30 (พบคำว่า 'ลัก' - ฝาผ่าน)")
            
            if success:
                # ON M600 หลังจาก ON M100/M110/M120
                print("⏳ ON M600 และรอ M401 ON เพื่อ reset ทั้งหมด...")
                m600_success = self.gui.modbus_thread.on_m600()
                if m600_success:
                    self.gui.status_handlers.update_coil_lamp("m600", True)
                
                # อัปเดตสถานะ
                self.gui.status_label.setText(f'✅ ฝาผ่าน → ON {self.gui.current_bottle_type} + M600 (รอ reset)')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                
                print(f"✅ BOTTLE TYPE ON SUCCESS: {self.gui.current_bottle_type} + M600")
            else:
                print(f"❌ ไม่สามารถ ON {self.gui.current_bottle_type} ได้")
                self.gui.status_label.setText(f'❌ ไม่สามารถ ON {self.gui.current_bottle_type} ได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            print(f"❌ ERROR in on_bottle_type_after_cap_validation: {e}")
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def initialize_modbus_state(self):
        """Initialize Modbus state when program starts"""
        try:
            print("🔧 Initializing Modbus state...")
            
            # Reset M100, M110, M120, M130, M750, M600
            if self.gui.modbus_thread:
                success = True
                success &= self.gui.modbus_thread.reset_m100()
                success &= self.gui.modbus_thread.reset_m110()
                success &= self.gui.modbus_thread.reset_m120()
                success &= self.gui.modbus_thread.reset_m130()
                success &= self.gui.modbus_thread.reset_m750()
                success &= self.gui.modbus_thread.reset_m600()
                
                # Stop D6006 monitoring
                self.gui.modbus_thread.d6006_monitoring = False
                
                if success:
                    print("✅ RESET M100, M110, M120, M130, M750, M600 = 0")
                    self.gui.modbus_status_label.setText('📡 Modbus: Reset bottle types and M600 completed')
                else:
                    print("❌ Failed to reset bottle types and M600")
                
                # Set initial taste mode based on current_taste_mode
                print(f"🔧 Setting initial taste mode: {self.gui.current_taste_mode}")
                if self.gui.current_taste_mode == 1:
                    # 1-taste mode: Write 30 to D9006
                    success = self.gui.modbus_thread.write_register(9006, 30)
                    if success:
                        print("✅ D9006 = 30 (โหมด 1 รสชาติ) - Initial setup")
                    else:
                        print("❌ Failed to write D9006 = 30")
                elif self.gui.current_taste_mode == 2:
                    # 2-taste mode: Write 20 to D9006
                    success = self.gui.modbus_thread.write_register(9006, 20)
                    if success:
                        print("✅ D9006 = 20 (โหมด 2 รสชาติ) - Initial setup")
                    else:
                        print("❌ Failed to write D9006 = 20")
                elif self.gui.current_taste_mode == 3:
                    # 3-taste mode: Write 10 to D9006
                    success = self.gui.modbus_thread.write_register(9006, 10)
                    if success:
                        print("✅ D9006 = 10 (โหมด 3 รสชาติ) - Initial setup")
                    else:
                        print("❌ Failed to write D9006 = 10")
                
                # ON M503 แล้ว OFF (pulse)
                try:
                    # ON M503
                    result_on = self.gui.modbus_thread.modbus_client.write_coil(503, True, unit=1)
                    if not result_on.isError():
                        print("✅ ON M503 = 1")
                        
                        # รอสักครู่แล้ว OFF M503
                        QTimer.singleShot(100, lambda: self.off_m503())
                    else:
                        print("❌ Failed to ON M503")
                except Exception as e:
                    print(f"❌ Error with M503: {e}")
                    
        except Exception as e:
            print(f"❌ Error initializing Modbus state: {e}")
    
    def off_m503(self):
        """Turn OFF M503"""
        try:
            if self.gui.modbus_thread and self.gui.modbus_thread.modbus_client:
                result = self.gui.modbus_thread.modbus_client.write_coil(503, False, unit=1)
                if not result.isError():
                    print("✅ OFF M503 = 0")
                    self.gui.modbus_status_label.setText('📡 Modbus: M503 pulse completed')
                else:
                    print("❌ Failed to OFF M503")
        except Exception as e:
            print(f"❌ Error turning OFF M503: {e}")
    
    def on_taste_mode_changed(self):
        """Handle taste mode selection change - ทำงานแบบ radio button (ถ้ากดช่องใหม่ ช่องเก่าจะถูกยกเลิกอัตโนมัติ)"""
        try:
            # ตรวจสอบว่า checkbox ไหนถูก check
            sender = self.gui.sender()
            
            # ถ้า checkbox ถูก uncheck (ไม่ใช่ check) ให้ข้าม
            if sender and not sender.isChecked():
                return
            
            # ตรวจสอบว่า checkbox ไหนถูก check และยกเลิกช่องอื่นๆ
            if sender == self.gui.taste_mode_1:
                # ยกเลิกช่องอื่นๆ (block signals เพื่อป้องกันการ trigger ซ้ำ)
                self.gui.taste_mode_2.blockSignals(True)
                self.gui.taste_mode_3.blockSignals(True)
                self.gui.taste_mode_2.setChecked(False)
                self.gui.taste_mode_3.setChecked(False)
                self.gui.taste_mode_2.blockSignals(False)
                self.gui.taste_mode_3.blockSignals(False)
                
                # ตรวจสอบว่า checkbox ถูก check หรือไม่
                if not self.gui.taste_mode_1.isChecked():
                    self.gui.taste_mode_1.blockSignals(True)
                    self.gui.taste_mode_1.setChecked(True)
                    self.gui.taste_mode_1.blockSignals(False)
                
                selected_mode = 1
                
            elif sender == self.gui.taste_mode_2:
                # ยกเลิกช่องอื่นๆ
                self.gui.taste_mode_1.blockSignals(True)
                self.gui.taste_mode_3.blockSignals(True)
                self.gui.taste_mode_1.setChecked(False)
                self.gui.taste_mode_3.setChecked(False)
                self.gui.taste_mode_1.blockSignals(False)
                self.gui.taste_mode_3.blockSignals(False)
                
                # ตรวจสอบว่า checkbox ถูก check หรือไม่
                if not self.gui.taste_mode_2.isChecked():
                    self.gui.taste_mode_2.blockSignals(True)
                    self.gui.taste_mode_2.setChecked(True)
                    self.gui.taste_mode_2.blockSignals(False)
                
                selected_mode = 2
                
            elif sender == self.gui.taste_mode_3:
                # ยกเลิกช่องอื่นๆ
                self.gui.taste_mode_1.blockSignals(True)
                self.gui.taste_mode_2.blockSignals(True)
                self.gui.taste_mode_1.setChecked(False)
                self.gui.taste_mode_2.setChecked(False)
                self.gui.taste_mode_1.blockSignals(False)
                self.gui.taste_mode_2.blockSignals(False)
                
                # ตรวจสอบว่า checkbox ถูก check หรือไม่
                if not self.gui.taste_mode_3.isChecked():
                    self.gui.taste_mode_3.blockSignals(True)
                    self.gui.taste_mode_3.setChecked(True)
                    self.gui.taste_mode_3.blockSignals(False)
                
                selected_mode = 3
            else:
                return  # ไม่ใช่ checkbox ที่เราต้องการ
            
            # Reset all taste mode signals first
            if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                self.gui.modbus_thread.reset_m720()
                self.gui.modbus_thread.reset_m721()
                self.gui.modbus_thread.reset_m722()
                
                # Reset all taste combination signals
                self.gui.modbus_thread.reset_m730()
                self.gui.modbus_thread.reset_m731()
                self.gui.modbus_thread.reset_m732()
                self.gui.modbus_thread.reset_m733()
                self.gui.modbus_thread.reset_m734()
                self.gui.modbus_thread.reset_m735()
                
                # Reset all 1-taste signals
                self.gui.modbus_thread.reset_m740()
                self.gui.modbus_thread.reset_m741()
                self.gui.modbus_thread.reset_m742()
                
                # Reset lamp status
                self.gui.status_handlers.update_coil_lamp("m720", False)
                self.gui.status_handlers.update_coil_lamp("m721", False)
                self.gui.status_handlers.update_coil_lamp("m722", False)
                self.gui.status_handlers.update_coil_lamp("m730", False)
                self.gui.status_handlers.update_coil_lamp("m731", False)
                self.gui.status_handlers.update_coil_lamp("m732", False)
                self.gui.status_handlers.update_coil_lamp("m733", False)
                self.gui.status_handlers.update_coil_lamp("m734", False)
                self.gui.status_handlers.update_coil_lamp("m735", False)
                self.gui.status_handlers.update_coil_lamp("m740", False)
                self.gui.status_handlers.update_coil_lamp("m741", False)
                self.gui.status_handlers.update_coil_lamp("m742", False)
                
                print("🔄 TASTE MODE: Reset all taste mode and combination signals")
            
            # ตั้งค่าโหมดที่เลือก
            self.gui.current_taste_mode = selected_mode
            
            if selected_mode == 1:
                self.show_taste_selection()
                self.gui.taste_mode_status.setText("โหมดปัจจุบัน: 1 รสชาติ - กรุณาเลือกรสชาติ")
                self.gui.taste_mode_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px; font-weight: bold;")
                
                # Write 30 to D9006 (1-taste mode)
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    success = self.gui.modbus_thread.write_register(9006, 30)
                    if success:
                        print("🏷️ D9006 = 30 (โหมด 1 รสชาติ)")
                    else:
                        print("❌ ไม่สามารถเขียน D9006 = 30 ได้")
                
            elif selected_mode == 2:
                self.show_taste_selection()
                self.gui.taste_mode_status.setText("โหมดปัจจุบัน: 2 รสชาติ - กรุณาเลือกรสชาติ")
                self.gui.taste_mode_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px; font-weight: bold;")
                
                # Write 20 to D9006 (2-taste mode)
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    success = self.gui.modbus_thread.write_register(9006, 20)
                    if success:
                        print("🏷️ D9006 = 20 (โหมด 2 รสชาติ)")
                    else:
                        print("❌ ไม่สามารถเขียน D9006 = 20 ได้")
                
            elif selected_mode == 3:
                self.hide_taste_selection()
                self.gui.selected_tastes = ["M100", "M110", "M120"]  # All tastes
                self.gui.taste_mode_status.setText("โหมดปัจจุบัน: 3 รสชาติ (M100, M110, M120)")
                self.gui.taste_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px; font-weight: bold;")
                
                # Write 10 to D9006 (3-taste mode)
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    success = self.gui.modbus_thread.write_register(9006, 10)
                    if success:
                        print("🏷️ D9006 = 10 (โหมด 3 รสชาติ)")
                    else:
                        print("❌ ไม่สามารถเขียน D9006 = 10 ได้")
            
            print(f"🔄 TASTE MODE: Changed to {self.gui.current_taste_mode}-taste mode")
            print(f"🔄 TASTE MODE: Selected tastes: {self.gui.selected_tastes}")
            
        except Exception as e:
            print(f"❌ Error in on_taste_mode_changed: {e}")
    
    def show_taste_selection(self):
        """Show taste selection checkboxes"""
        try:
            # Show all taste checkboxes
            for i in range(self.gui.taste_selection_layout.count()):
                widget = self.gui.taste_selection_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
            
            # Reset all taste selections
            self.gui.taste_m100.setChecked(False)
            self.gui.taste_m110.setChecked(False)
            self.gui.taste_m120.setChecked(False)
            self.gui.selected_tastes = []
            
            print("🔄 TASTE SELECTION: Showing taste selection checkboxes")
            
        except Exception as e:
            print(f"❌ Error in show_taste_selection: {e}")
    
    def hide_taste_selection(self):
        """Hide taste selection checkboxes"""
        try:
            # Hide taste selection labels and checkboxes
            for i in range(self.gui.taste_selection_layout.count()):
                widget = self.gui.taste_selection_layout.itemAt(i).widget()
                if widget and isinstance(widget, type(self.gui.taste_m100)):
                    widget.setVisible(False)
                elif widget and hasattr(widget, 'text') and "เลือกรสชาติ" in widget.text():
                    widget.setVisible(False)
            
            print("🔄 TASTE SELECTION: Hiding taste selection checkboxes")
            
        except Exception as e:
            print(f"❌ Error in hide_taste_selection: {e}")
    
    def reset_to_initial_state(self):
        """Reset all UI elements and states to initial program state"""
        try:
            print("🔄 Resetting system to initial state...")
            
            # 1. Reset Modbus state
            self.reset_modbus_to_initial_state()
            
            # 2. Reset UI elements to initial state
            self.reset_ui_to_initial_state()
            
            # 3. Clear current images and results
            self.clear_current_data()
            
            # 4. Reset progress bars
            self.reset_progress_bars()
            
            # 5. Reset camera states
            self.reset_camera_states()
            
            # 6. Reset lamp status
            self.reset_lamp_status()
            
            print("✅ System reset to initial state completed")
            
        except Exception as e:
            print(f"❌ Error during system reset: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_modbus_to_initial_state(self):
        """Reset Modbus state to initial values"""
        try:
            if self.gui.modbus_thread:
                print("🔄 Resetting Modbus state...")
                
                # Reset all bottle type coils
                self.gui.modbus_thread.reset_m100()
                self.gui.modbus_thread.reset_m110() 
                self.gui.modbus_thread.reset_m120()
                self.gui.modbus_thread.reset_m130()
                self.gui.modbus_thread.reset_m750()
                self.gui.modbus_thread.reset_m850()
                self.gui.modbus_thread.reset_m140()
                self.gui.modbus_thread.reset_m600()
                
                # Reset M700, M701
                self.gui.modbus_thread.reset_m700()
                self.gui.modbus_thread.reset_m701()
                
                # Reset gripper coils
                self.gui.modbus_thread.reset_m503()
                self.gui.modbus_thread.reset_m505()
                
                # Reset taste mode signals
                self.gui.modbus_thread.reset_m720()
                self.gui.modbus_thread.reset_m721()
                self.gui.modbus_thread.reset_m722()
                
                # Reset 2-taste combination signals
                self.gui.modbus_thread.reset_m730()
                self.gui.modbus_thread.reset_m731()
                self.gui.modbus_thread.reset_m732()
                self.gui.modbus_thread.reset_m733()
                self.gui.modbus_thread.reset_m734()
                self.gui.modbus_thread.reset_m735()
                
                # Reset 1-taste signals
                self.gui.modbus_thread.reset_m740()
                self.gui.modbus_thread.reset_m741()
                self.gui.modbus_thread.reset_m742()
                
                # Reset queue count
                self.reset_queue_count()
                
                # Stop D6006 monitoring for M130
                self.gui.modbus_thread.d6006_monitoring = False
                print("✅ Stopped D6006 monitoring")
                
                # Reset D6007 tracking
                self.gui.modbus_thread.last_d6007_value = None
                print("✅ Reset D6007 tracking")
                
                print("✅ Modbus state reset completed")
                
        except Exception as e:
            print(f"❌ Error resetting Modbus state: {e}")
    
    def reset_queue_count(self):
        """Reset queue count to zero"""
        try:
            if self.gui.modbus_thread:
                print("🔄 Resetting queue count...")
                self.gui.modbus_thread.pending_m301_count = 0
                self.gui.modbus_thread.m600_reset_pending = False
                self.gui.modbus_thread.d6006_monitoring = False
                print("✅ Queue count and monitoring reset to 0")
        except Exception as e:
            print(f"❌ Error resetting queue count: {e}")
    
    def reset_ui_to_initial_state(self):
        """Reset all UI elements to initial state"""
        try:
            print("🔄 Resetting UI elements...")
            
            # Reset main status labels
            self.gui.status_label.setText('⏸️ โปรแกรมพร้อมทำงาน (รอ M511 เพื่อเริ่มการทำงาน)')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            
            self.gui.combined_status_label.setText('📊 สถานะรวม: กล้อง USB + Sentech พร้อมทำงาน')
            self.gui.combined_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-weight: bold;")
            
            # Reset Modbus status
            self.gui.modbus_status_label.setText('📡 Modbus: พร้อมใช้งาน')
            self.gui.modbus_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            # Reset queue status
            self.gui.queue_status_label.setText('📋 คิว: 0')
            self.gui.queue_status_label.setStyleSheet("color: #e67e22; padding: 5px; font-weight: bold;")
            
            # Reset queue info label
            self.gui.queue_info_label.setText('📋 สถานะคิว: ไม่มีคิวรอ')
            self.gui.queue_info_label.setStyleSheet("color: #e67e22; padding: 5px; font-size: 11px;")
            
            # Reset D6004 status
            self.gui.d6004_status_label.setText('🔍 D6004: รอค่า')
            self.gui.d6004_status_label.setStyleSheet("color: #9b59b6; padding: 5px; font-size: 11px;")
            
            # Reset D6007 status
            self.gui.d6007_status_label.setText('🔍 D6007: รอค่า')
            self.gui.d6007_status_label.setStyleSheet("color: #9b59b6; padding: 5px; font-size: 11px;")
            
            # Reset gripper status
            self.gui.gripper_status_label.setText('🤖 สถานะ Gripper: รอคำสั่ง')
            self.gui.gripper_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-weight: bold;")
            
            # Reset camera status labels
            self.gui.camera_status_label.setText('📷 กล้อง: พร้อมใช้งาน')
            self.gui.camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            self.gui.sentech_camera_status_label.setText('📷 Sentech: พร้อมใช้งาน')
            self.gui.sentech_camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            # Clear cap detection UI displays
            if hasattr(self.gui, 'cap_detection_text') and self.gui.cap_detection_text:
                self.gui.cap_detection_text.clear()
            
            # Clear cap results layout (if it exists)
            if hasattr(self.gui, 'cap_results_layout') and self.gui.cap_results_layout:
                for i in reversed(range(self.gui.cap_results_layout.count())):
                    widget = self.gui.cap_results_layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)
            
            print("✅ UI elements reset completed")
            
        except Exception as e:
            print(f"❌ Error resetting UI elements: {e}")
    
    def clear_current_data(self):
        """Clear current images and results"""
        try:
            print("🔄 Clearing current data...")
            
            # Clear current images
            self.gui.current_image = None
            self.gui.current_result = None
            self.gui.current_sentech_image = None
            self.gui.current_cap_result = None
            self.gui.original_cap_image = None
            self.gui.ai_rotated_image = None
            self.gui.current_rotation_angle = 0
            
            # Clear batch results
            self.gui.batch_results = []
            
            # Clear image displays (if they exist)
            if hasattr(self.gui, 'image_label') and self.gui.image_label:
                self.gui.image_label.clear()
                self.gui.image_label.setText("📷 ไม่มีภาพ")
            
            # Clear cap detection UI displays
            if hasattr(self.gui, 'cap_detection_text') and self.gui.cap_detection_text:
                self.gui.cap_detection_text.clear()
            
            # Clear cap results layout (if it exists)
            if hasattr(self.gui, 'cap_results_layout') and self.gui.cap_results_layout:
                for i in reversed(range(self.gui.cap_results_layout.count())):
                    widget = self.gui.cap_results_layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)
                self.gui.image_label.setStyleSheet("color: #7f8c8d; padding: 20px; border: 2px dashed #7f8c8d;")
            
            if hasattr(self.gui, 'sentech_image_label') and self.gui.sentech_image_label:
                self.gui.sentech_image_label.clear()
                self.gui.sentech_image_label.setText("📷 ไม่มีภาพ Sentech")
                self.gui.sentech_image_label.setStyleSheet("color: #7f8c8d; padding: 20px; border: 2px dashed #7f8c8d;")
            
            print("✅ Current data cleared")
            
        except Exception as e:
            print(f"❌ Error clearing current data: {e}")
    
    def reset_progress_bars(self):
        """Reset and hide all progress bars"""
        try:
            print("🔄 Resetting progress bars...")
            
            # Reset and hide main progress bar
            if hasattr(self.gui, 'progress_bar'):
                self.gui.progress_bar.setValue(0)
                self.gui.progress_bar.setVisible(False)
            
            # Reset and hide cap progress bar
            if hasattr(self.gui, 'cap_progress_bar'):
                self.gui.cap_progress_bar.setValue(0)
                self.gui.cap_progress_bar.setVisible(False)
            
            print("✅ Progress bars reset")
            
        except Exception as e:
            print(f"❌ Error resetting progress bars: {e}")
    
    def reset_camera_states(self):
        """Reset camera states"""
        try:
            print("🔄 Resetting camera states...")
            
            # Reset camera status indicators
            if self.gui.usb_camera:
                self.gui.camera_status_label.setText('📷 กล้อง: พร้อมใช้งาน')
                self.gui.camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            if self.gui.sentech_camera:
                self.gui.sentech_camera_status_label.setText('📷 Sentech: พร้อมใช้งาน')
                self.gui.sentech_camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            print("✅ Camera states reset")
            
        except Exception as e:
            print(f"❌ Error resetting camera states: {e}")
    
    def reset_lamp_status(self):
        """Reset all lamp status to OFF state"""
        try:
            print("🔄 Resetting lamp status...")
            
            # Reset connection status
            self.gui.status_handlers.update_modbus_connection_status_ui(False)
            
            # Reset all coil lamps to OFF
            coil_names = ["m100", "m110", "m120", "m130", "m140", "m600", "m700", "m701", "m750", "m850", "m503", "m505", "m720", "m721", "m722", "m730", "m731", "m732", "m733", "m734", "m735", "m740", "m741", "m742"]
            for coil_name in coil_names:
                self.gui.status_handlers.update_coil_lamp(coil_name, False)
            
            # Reset register lamps
            self.gui.status_handlers.update_register_lamp("d6004", "รอค่า", False)
            # M402 ไม่ต้อง reset เพราะเป็น coil
            self.gui.status_handlers.update_register_lamp("d6007", "รอค่า", False)
            
            # Reset bottle type status
            self.gui.status_handlers.update_bottle_type_status(None)
            
            print("✅ Lamp status reset completed")
            
        except Exception as e:
            print(f"❌ Error resetting lamp status: {e}")
    
    def on_motor_reverse_press(self, event):
        """Handle mouse press on motor reverse button - ON M507"""
        from PyQt5.QtWidgets import QPushButton
        if event.button() == Qt.LeftButton:
            if self.gui.modbus_thread:
                success = self.gui.modbus_thread.on_m507()
                if success:
                    self.gui.btn_motor_reverse.setText('🔄 กลับทาง Motor (กดค้าง)')
                    self.gui.btn_motor_reverse.setStyleSheet("""
                        QPushButton {
                            padding: 10px;
                            font-size: 12px;
                            background-color: #21618c;
                            color: white;
                            border-radius: 5px;
                            font-weight: bold;
                        }
                    """)
                    self.gui.status_handlers.update_coil_lamp("m507", True)
                    print("✅ MOTOR REVERSE: M507 ON - กลับทาง Motor")
                else:
                    print("❌ MOTOR REVERSE: ไม่สามารถ ON M507 ได้")
            else:
                print("❌ MOTOR REVERSE: Modbus thread ไม่พร้อมใช้งาน")
        
        # Call original mousePressEvent
        QPushButton.mousePressEvent(self.gui.btn_motor_reverse, event)
    
    def on_motor_reverse_release(self, event):
        """Handle mouse release on motor reverse button - RESET M507"""
        from PyQt5.QtWidgets import QPushButton
        if event.button() == Qt.LeftButton:
            if self.gui.modbus_thread:
                success = self.gui.modbus_thread.reset_m507()
                if success:
                    self.gui.btn_motor_reverse.setText('🔄 กลับทาง Motor (M507)')
                    self.gui.btn_motor_reverse.setStyleSheet("""
                        QPushButton {
                            padding: 10px;
                            font-size: 12px;
                            background-color: #3498db;
                            color: white;
                            border-radius: 5px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: #2980b9;
                        }
                        QPushButton:pressed {
                            background-color: #21618c;
                        }
                    """)
                    self.gui.status_handlers.update_coil_lamp("m507", False)
                    print("✅ MOTOR REVERSE: M507 RESET - ปล่อยปุ่ม")
                else:
                    print("❌ MOTOR REVERSE: ไม่สามารถ RESET M507 ได้")
            else:
                print("❌ MOTOR REVERSE: Modbus thread ไม่พร้อมใช้งาน")
        
        # Call original mouseReleaseEvent
        QPushButton.mouseReleaseEvent(self.gui.btn_motor_reverse, event)
    
    def open_gripper(self):
        """เปิด Gripper โดยส่งสัญญาณ M503"""
        if self.gui.modbus_thread:
            success = self.gui.modbus_thread.on_m503()
            if success:
                self.gui.gripper_status_label.setText('🟢 Gripper: เปิด (M503 ON)')
                self.gui.gripper_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-weight: bold;")
                self.gui.status_handlers.update_coil_lamp("m503", True)
                print("✅ GRIPPER: M503 ON - เปิด Gripper")
                
                # Reset M503 หลังจาก 0.5 วินาที (Push button behavior)
                QTimer.singleShot(500, self.reset_gripper_open)
            else:
                self.gui.gripper_status_label.setText('❌ Gripper: ไม่สามารถเปิดได้')
                self.gui.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
                print("❌ GRIPPER: ไม่สามารถ ON M503 ได้")
        else:
            self.gui.gripper_status_label.setText('❌ Modbus ไม่พร้อมใช้งาน')
            self.gui.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
            print("❌ GRIPPER: Modbus thread ไม่พร้อมใช้งาน")
    
    def close_gripper(self):
        """ปิด Gripper โดยส่งสัญญาณ M505"""
        if self.gui.modbus_thread:
            success = self.gui.modbus_thread.on_m505()
            if success:
                self.gui.gripper_status_label.setText('🔴 Gripper: ปิด (M505 ON)')
                self.gui.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
                self.gui.status_handlers.update_coil_lamp("m505", True)
                print("✅ GRIPPER: M505 ON - ปิด Gripper")
                
                # Reset M505 หลังจาก 0.5 วินาที (Push button behavior)
                QTimer.singleShot(500, self.reset_gripper_close)
            else:
                self.gui.gripper_status_label.setText('❌ Gripper: ไม่สามารถปิดได้')
                self.gui.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
                print("❌ GRIPPER: ไม่สามารถ ON M505 ได้")
        else:
            self.gui.gripper_status_label.setText('❌ Modbus ไม่พร้อมใช้งาน')
            self.gui.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
            print("❌ GRIPPER: Modbus thread ไม่พร้อมใช้งาน")
    
    def reset_gripper_open(self):
        """Reset M503 หลังจากเปิด Gripper"""
        if self.gui.modbus_thread:
            success = self.gui.modbus_thread.reset_m503()
            if success:
                self.gui.gripper_status_label.setText('🤖 Gripper: รอคำสั่ง (M503 Reset)')
                self.gui.gripper_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-weight: bold;")
                self.gui.status_handlers.update_coil_lamp("m503", False)
                print("✅ GRIPPER: M503 Reset - Push button completed")
            else:
                print("❌ GRIPPER: ไม่สามารถ Reset M503 ได้")
    
    def reset_gripper_close(self):
        """Reset M505 หลังจากปิด Gripper"""
        if self.gui.modbus_thread:
            success = self.gui.modbus_thread.reset_m505()
            if success:
                self.gui.gripper_status_label.setText('🤖 Gripper: รอคำสั่ง (M505 Reset)')
                self.gui.gripper_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-weight: bold;")
                self.gui.status_handlers.update_coil_lamp("m505", False)
                print("✅ GRIPPER: M505 Reset - Push button completed")
            else:
                print("❌ GRIPPER: ไม่สามารถ Reset M505 ได้")


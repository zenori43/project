# -*- coding: utf-8 -*-
"""
Bottle Detection Event Handlers
Handles all events related to bottle detection processing
"""

# CRITICAL: Setup CUDA paths BEFORE importing any libs modules
try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    pass

from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLabel, QWidget, QVBoxLayout
from PyQt5 import QtWidgets, QtGui
from core.business_logic import BottleDetectionThread
from libs.detection.bottledetect import draw_detections_on_image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread
import cv2
import numpy as np
import datetime
import os

# Import CUDA image utilities
from core.cuda_image_utils import cuda_resize, cuda_cvtColor


class BottleDetectionHandlers:
    """Event handlers for bottle detection functionality"""
    
    def __init__(self, gui_instance):
        """
        Initialize handlers with reference to GUI instance
        
        Args:
            gui_instance: Reference to BottleDetectionGUI instance
        """
        self.gui = gui_instance
        self.silent_mode = False  # Flag for silent processing (no UI display, no Modbus signals)
        self.pending_bottle_result = None  # Store result for silent processing
    
    def capture_both_cameras(self):
        """Capture images from both USB and Sentech cameras simultaneously"""
        try:
            self.gui.status_label.setText('📸 กำลังถ่ายภาพจาก USB และ Sentech...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            usb_image = None
            sentech_image = None
            
            # Capture from USB camera
            if self.gui.usb_camera is None:
                QMessageBox.warning(self.gui, "ข้อผิดพลาด", "กล้อง USB ยังไม่ได้เริ่มต้น")
                return
            else:
                print("📸 CAPTURE BOTH: กำลังถ่ายภาพจาก USB...")
                for attempt in range(3):
                    print(f"📸 CAPTURE BOTH USB: Attempt {attempt + 1}/3")
                    usb_image = self.gui.usb_camera.capture_image()
                    if usb_image is not None:
                        print(f"✅ CAPTURE BOTH USB: Success on attempt {attempt + 1}")
                        break
                    QThread.msleep(100)
            
            # Capture from Sentech camera
            if self.gui.sentech_camera is None:
                print("⚠️ CAPTURE BOTH: กล้อง Sentech ยังไม่ได้เริ่มต้น - ข้าม")
            else:
                print("📸 CAPTURE BOTH: กำลังถ่ายภาพจาก Sentech...")
                try:
                    sentech_image = self.gui.sentech_camera.capture_image()
                    print(f"📸 CAPTURE BOTH Sentech: Result = {sentech_image is not None}")
                except Exception as e:
                    print(f"❌ CAPTURE BOTH Sentech Error: {e}")
                    sentech_image = None
            
            # Update USB image display
            if usb_image is not None:
                self.gui.current_image = usb_image
                self.display_image(usb_image)
                self.gui.image_info_label.setText(f"ขนาด: {usb_image.shape[1]}x{usb_image.shape[0]}")
                if hasattr(self.gui, 'home_bottle_image_info_label'):
                    self.gui.home_bottle_image_info_label.setText(f"ขนาด: {usb_image.shape[1]}x{usb_image.shape[0]}")
                self.gui.btn_process.setEnabled(True)
                self.gui.btn_save_image.setEnabled(True)
                # Also update tab buttons if they exist
                if hasattr(self.gui, 'btn_process_bottle_tab'):
                    self.gui.btn_process_bottle_tab.setEnabled(True)
                if hasattr(self.gui, 'btn_save_bottle_image_tab'):
                    self.gui.btn_save_bottle_image_tab.setEnabled(True)
                print("✅ CAPTURE BOTH: USB image captured and displayed")
            else:
                print("❌ CAPTURE BOTH: ไม่สามารถถ่ายภาพจาก USB ได้")
            
            # Update Sentech image display
            if sentech_image is not None:
                self.gui.current_sentech_image = sentech_image
                # ใช้ cap_handlers จาก GUI เพื่อแสดงภาพ Sentech
                if hasattr(self.gui, 'cap_handlers') and self.gui.cap_handlers:
                    self.gui.cap_handlers.display_sentech_image(sentech_image)
                else:
                    # Fallback: แสดงภาพโดยตรงถ้า cap_handlers ยังไม่มี
                    print("⚠️ CAPTURE BOTH: cap_handlers ยังไม่มี - ข้ามการแสดงภาพ Sentech")
                self.gui.sentech_image_info_label.setText(f"ขนาด: {sentech_image.shape[1]}x{sentech_image.shape[0]}")
                self.gui.btn_save_sentech_image.setEnabled(True)
                # Enable process cap button when Sentech image is available (ไม่ต้องรอ bottle_type)
                if hasattr(self.gui, 'btn_process_cap'):
                    self.gui.btn_process_cap.setEnabled(True)
                    print("✅ CAPTURE BOTH: เปิดใช้งานปุ่มประมวลผลฝา (มีภาพจาก Sentech)")
                print("✅ CAPTURE BOTH: Sentech image captured and displayed")
            else:
                print("⚠️ CAPTURE BOTH: ไม่สามารถถ่ายภาพจาก Sentech ได้")
            
            # Update status
            if usb_image is not None and sentech_image is not None:
                self.gui.status_label.setText('✅ ถ่ายภาพจาก USB และ Sentech สำเร็จ')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            elif usb_image is not None:
                self.gui.status_label.setText('✅ ถ่ายภาพจาก USB สำเร็จ (Sentech ไม่พร้อม)')
                self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            elif sentech_image is not None:
                self.gui.status_label.setText('✅ ถ่ายภาพจาก Sentech สำเร็จ (USB ไม่พร้อม)')
                self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            else:
                self.gui.status_label.setText('❌ ไม่สามารถถ่ายภาพได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            print(f"❌ CAPTURE BOTH ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_image_manual(self):
        """Capture image manually"""
        if self.gui.usb_camera is None:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "กล้องยังไม่ได้เริ่มต้น")
            return
        
        try:
            self.gui.status_label.setText('📸 กำลังถ่ายภาพ...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # Capture image - ดึงเฟรมล่าสุด
            captured_image = None
            for attempt in range(3):
                print(f"📸 MANUAL CAPTURE: Attempt {attempt + 1}/3")
                captured_image = self.gui.usb_camera.capture_image()
                if captured_image is not None:
                    print(f"✅ MANUAL CAPTURE: Success on attempt {attempt + 1}")
                    break
                QThread.msleep(100)  # Non-blocking sleep
            
            if captured_image is not None:
                self.gui.current_image = captured_image
                self.display_image(captured_image)
                self.gui.image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]}")
                self.gui.btn_process.setEnabled(True)
                self.gui.btn_save_image.setEnabled(True)
                # Also update tab buttons if they exist
                if hasattr(self.gui, 'btn_process_bottle_tab'):
                    self.gui.btn_process_bottle_tab.setEnabled(True)
                if hasattr(self.gui, 'btn_save_bottle_image_tab'):
                    self.gui.btn_save_bottle_image_tab.setEnabled(True)  # Enable save button when image is captured
                self.gui.status_label.setText('✅ ถ่ายภาพสำเร็จ - พร้อมประมวลผล')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            else:
                self.gui.status_label.setText('❌ ไม่สามารถถ่ายภาพได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def select_image_file(self):
        """Select image file for bottle detection processing"""
        try:
            # Open file dialog to select image
            file_path, _ = QFileDialog.getOpenFileName(
                self.gui,
                "เลือกไฟล์ภาพสำหรับการตรวจจับขวด",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            
            if file_path:
                print(f"📁 IMAGE SELECTION: Selected file: {file_path}")
                self.gui.status_label.setText('📁 กำลังโหลดภาพ...')
                self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
                QtWidgets.QApplication.processEvents()
                
                # Load image using OpenCV
                image = cv2.imread(file_path)
                if image is not None:
                    self.gui.current_image = image
                    self.display_image(image)
                    self.gui.image_info_label.setText(f"ขนาด: {image.shape[1]}x{image.shape[0]} | ไฟล์: {file_path.split('/')[-1]}")
                    self.gui.btn_process.setEnabled(True)
                    self.gui.btn_save_image.setEnabled(True)  # Enable save button when image is loaded
                    # Also update tab buttons if they exist
                    if hasattr(self.gui, 'btn_process_bottle_tab'):
                        self.gui.btn_process_bottle_tab.setEnabled(True)
                    if hasattr(self.gui, 'btn_save_bottle_image_tab'):
                        self.gui.btn_save_bottle_image_tab.setEnabled(True)
                    self.gui.status_label.setText('✅ โหลดภาพสำเร็จ - พร้อมประมวลผล')
                    self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                    print(f"✅ IMAGE SELECTION: Successfully loaded image with shape: {image.shape}")
                else:
                    self.gui.status_label.setText('❌ ไม่สามารถโหลดภาพได้')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    QMessageBox.warning(self.gui, "ข้อผิดพลาด", f"ไม่สามารถโหลดภาพจากไฟล์:\n{file_path}")
                    print(f"❌ IMAGE SELECTION: Failed to load image from: {file_path}")
                    
        except Exception as e:
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเลือกไฟล์:\n{str(e)}")
            print(f"❌ IMAGE SELECTION ERROR: {str(e)}")
    
    def save_captured_image(self):
        """Save the currently captured/loaded image to disk"""
        if self.gui.current_image is None:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "ไม่มีภาพให้บันทึก\nกรุณาถ่ายภาพหรือเลือกไฟล์ภาพก่อน")
            return
        
        try:
            # Create default folder for manual captures
            default_folder = "captured_images"
            manual_folder = os.path.join(default_folder, "manual_capture")
            os.makedirs(manual_folder, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"manual_capture_{timestamp}.png"
            default_path = os.path.join(manual_folder, default_filename)
            
            # Ask user where to save (with default path)
            file_path, _ = QFileDialog.getSaveFileName(
                self.gui,
                "บันทึกรูปภาพ",
                default_path,
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
            )
            
            if file_path:
                # Save image
                success = cv2.imwrite(file_path, self.gui.current_image)
                if success:
                    self.gui.status_label.setText(f'✅ บันทึกรูปภาพสำเร็จ: {os.path.basename(file_path)}')
                    self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                    print(f"💾 บันทึกรูปภาพสำเร็จ: {file_path}")
                    QMessageBox.information(self.gui, "สำเร็จ", f"บันทึกรูปภาพสำเร็จ\n{file_path}")
                else:
                    self.gui.status_label.setText('❌ ไม่สามารถบันทึกรูปภาพได้')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    QMessageBox.warning(self.gui, "ข้อผิดพลาด", "ไม่สามารถบันทึกรูปภาพได้")
                    print(f"❌ ไม่สามารถบันทึกรูปภาพได้: {file_path}")
        except Exception as e:
            error_msg = f"เกิดข้อผิดพลาดในการบันทึกรูปภาพ: {str(e)}"
            self.gui.status_label.setText(f'❌ {error_msg}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", error_msg)
            print(f"❌ SAVE IMAGE ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _clear_bottle_result_panels(self):
        """ล้างผลและภาพครอปขวดบนหน้าหลัก/แท็บ เพื่อไม่ให้ค้างผลเก่า"""
        try:
            if hasattr(self.gui, 'home_bottle_crops_layout') and self.gui.home_bottle_crops_layout:
                for i in reversed(range(self.gui.home_bottle_crops_layout.count())):
                    w = self.gui.home_bottle_crops_layout.itemAt(i).widget()
                    if w:
                        w.setParent(None)
                ph = QLabel("ยังไม่มีภาพที่ครอป")
                ph.setAlignment(Qt.AlignCenter)
                ph.setStyleSheet("color: #7f8c8d; padding: 20px;")
                self.gui.home_bottle_crops_layout.addWidget(ph)
            if hasattr(self.gui, 'home_bottle_results_text') and self.gui.home_bottle_results_text:
                self.gui.home_bottle_results_text.clear()
            if hasattr(self.gui, 'crops_layout') and self.gui.crops_layout:
                for i in reversed(range(self.gui.crops_layout.count())):
                    w = self.gui.crops_layout.itemAt(i).widget()
                    if w:
                        w.setParent(None)
            if hasattr(self.gui, 'results_text') and self.gui.results_text:
                self.gui.results_text.clear()
        except Exception as e:
            print(f"⚠️ _clear_bottle_result_panels: {e}")
    
    def capture_image_auto(self):
        """Capture image automatically from Modbus trigger"""
        if self.gui.usb_camera is None:
            print("❌ Camera not initialized - cannot capture")
            return
        
        try:
            # รอบใหม่ — เพิ่ม cycle เพื่อไม่ให้ผลประมวลผลเก่า overwrite ภาพใหม่
            self.gui.capture_cycle_id = getattr(self.gui, 'capture_cycle_id', 0) + 1
            print(f"📸 AUTO CAPTURE: New cycle id = {self.gui.capture_cycle_id}")
            print("📸 AUTO CAPTURE: Starting image capture after M301 ON...")
            self.gui.status_label.setText('📸 กำลังถ่ายภาพอัตโนมัติ...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # รอ 2 วินาทีหลังได้รับสัญญาณ M301 เพื่อให้ได้เฟรมล่าสุด
            print("⏰ AUTO CAPTURE: Waiting 2 seconds for latest frame...")
            self.gui.status_label.setText('⏰ รอ 2 วินาทีเพื่อเฟรมล่าสุด...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # รอ 2 วินาทีเพื่อให้ได้เฟรมล่าสุดหลังจาก M301 ON (non-blocking)
            QThread.msleep(2000)  # Non-blocking sleep
            
            # ล้าง buffer และดึงเฟรมล่าสุด
            print("📸 AUTO CAPTURE: Capturing latest frame after M301 ON...")
            self.gui.status_label.setText('📸 กำลังถ่ายภาพเฟรมล่าสุด...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ลองถ่ายภาพหลายครั้งเพื่อให้ได้เฟรมล่าสุด
            captured_image = None
            for attempt in range(5):
                print(f"📸 AUTO CAPTURE: Attempt {attempt + 1}/5")
                
                # ตรวจสอบว่า M301 ยัง ON อยู่หรือไม่
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    try:
                        result = self.gui.modbus_thread.modbus_client.read_coils(301, 1, unit=1)
                        m301_status = not result.isError() and result.bits[0]
                    except:
                        m301_status = True  # ถ้าตรวจสอบไม่ได้ให้ถือว่า ON
                else:
                    m301_status = True
                
                captured_image = self.gui.usb_camera.capture_image()
                if captured_image is not None:
                    print(f"✅ AUTO CAPTURE: Success on attempt {attempt + 1}")
                    break
                QtCore.QThread.msleep(200)  # Non-blocking sleep
            
            if captured_image is not None:
                # แสดงเวลาที่ถ่ายภาพ
                capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"🕐 Capture time: {capture_time}")
                
                self.gui.current_image = captured_image
                self._clear_bottle_result_panels()  # ล้างผลเก่าเพื่อไม่ให้ภาพค้าง
                self.display_image(captured_image)
                self.gui.image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]} | เวลา: {capture_time}")
                self.gui.btn_process.setEnabled(True)
                # Also update tab buttons if they exist
                if hasattr(self.gui, 'btn_process_bottle_tab'):
                    self.gui.btn_process_bottle_tab.setEnabled(True)
                self.gui.status_label.setText('✅ ถ่ายภาพอัตโนมัติสำเร็จ - พร้อมประมวลผล')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                
                # Auto process is always enabled
                if True:  # Auto mode is always enabled
                    # ตรวจสอบว่าเป็นโหมดถ่ายภาพซ้ำหรือไม่
                    if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread and self.gui.modbus_thread.angle3_retry_mode:
                        print("🔄 AUTO PROCESS: Angle3 retry mode - processing bottle only (no cap detection)")
                        self.process_current_image()  # ประมวลผลขวดเท่านั้น
                    else:
                        print("🔄 AUTO PROCESS: Normal mode - starting automatic processing...")
                    self.process_current_image()
                else:
                    print("⚠️ Auto processing disabled - image captured but not processed")
            else:
                self.gui.status_label.setText('❌ ไม่สามารถถ่ายภาพอัตโนมัติได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_image_from_queue(self):
        """Capture image from queue (without waiting for M301)"""
        if self.gui.usb_camera is None:
            print("❌ Camera not initialized - cannot capture from queue")
            return
        
        try:
            self.gui.capture_cycle_id = getattr(self.gui, 'capture_cycle_id', 0) + 1
            print(f"📸 QUEUE CAPTURE: New cycle id = {self.gui.capture_cycle_id}")
            print("📸 QUEUE CAPTURE: Starting image capture from queue...")
            self.gui.status_label.setText('📸 กำลังถ่ายภาพจากคิว...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ถ่ายภาพทันทีโดยไม่รอ (เพราะเป็นงานจากคิว)
            print("📸 QUEUE CAPTURE: Capturing image immediately...")
            self.gui.status_label.setText('📸 กำลังถ่ายภาพ...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ลองถ่ายภาพหลายครั้ง
            captured_image = None
            for attempt in range(3):
                print(f"📸 QUEUE CAPTURE: Attempt {attempt + 1}/3")
                captured_image = self.gui.usb_camera.capture_image()
                if captured_image is not None:
                    print(f"✅ QUEUE CAPTURE: Success on attempt {attempt + 1}")
                    break
                time.sleep(0.1)
            
            if captured_image is not None:
                # แสดงเวลาที่ถ่ายภาพ
                capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"🕐 Queue Capture time: {capture_time}")
                print(f"📸 QUEUE CAPTURE: Image shape: {captured_image.shape}")
                
                self.gui.current_image = captured_image
                self._clear_bottle_result_panels()
                self.display_image(captured_image)
                self.gui.image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]} | เวลา: {capture_time} | จากคิว")
                self.gui.btn_process.setEnabled(True)
                # Also update tab buttons if they exist
                if hasattr(self.gui, 'btn_process_bottle_tab'):
                    self.gui.btn_process_bottle_tab.setEnabled(True)
                self.gui.status_label.setText('✅ ถ่ายภาพจากคิวสำเร็จ - พร้อมประมวลผล')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                
                # Auto process is always enabled
                if True:  # Auto mode is always enabled
                    print("🔄 QUEUE PROCESS: Starting automatic processing...")
                    print("🔄 QUEUE PROCESS: Auto mode is enabled, calling process_current_image()")
                    self.process_current_image()
                else:
                    print("⚠️ Auto processing disabled - image captured but not processed")
            else:
                self.gui.status_label.setText('❌ ไม่สามารถถ่ายภาพจากคิวได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_and_process_silently(self):
        """Capture image and process silently (no UI display, no Modbus signals) - store result in queue"""
        if self.gui.usb_camera is None:
            print("❌ Camera not initialized - cannot capture silently")
            return
        
        try:
            print("🔇 SILENT CAPTURE: Starting silent image capture and processing...")
            self.silent_mode = True  # Enable silent mode
            
            # Capture image immediately (no UI updates)
            captured_image = None
            for attempt in range(3):
                print(f"🔇 SILENT CAPTURE: Attempt {attempt + 1}/3")
                captured_image = self.gui.usb_camera.capture_image()
                if captured_image is not None:
                    print(f"✅ SILENT CAPTURE: Success on attempt {attempt + 1}")
                    break
                time.sleep(0.1)
            
            if captured_image is not None:
                print(f"🔇 SILENT CAPTURE: Image shape: {captured_image.shape}")
                self.gui.current_image = captured_image
                
                # Process silently (no UI display, no Modbus signals)
                print("🔇 SILENT PROCESS: Starting silent processing...")
                self.gui.progress_bar.setVisible(False)  # Don't show progress bar
                
                # Start processing thread
                self.gui.processing_thread = BottleDetectionThread(captured_image, selected_tastes=self.gui.selected_tastes)
                self.gui.processing_thread.result_ready.connect(self.on_processing_complete)
                self.gui.processing_thread.start()
                print("🔇 SILENT PROCESS: Processing thread started")
            else:
                print("❌ SILENT CAPTURE: Failed to capture image")
                self.silent_mode = False
                
        except Exception as e:
            print(f"❌ SILENT CAPTURE ERROR: {str(e)}")
            self.silent_mode = False
    
    def display_result_from_queue(self, result):
        """Display result from queue (after D6004=200)"""
        try:
            print("📋 DISPLAY FROM QUEUE: Displaying bottle result from queue")
            
            # Store result
            self.gui.current_result = result
            
            # Display results in text area
            self.display_single_results(result)
            
            # Draw detections on image and display
            if result.get('detections'):
                result_image = draw_detections_on_image(
                    result['image'], 
                    result['detections']
                )
                self.display_result_image(result_image)
            else:
                if result.get('image') is not None:
                    self.display_result_image(result['image'])
            
            # Display cropped images with OCR
            if result.get('type_crops'):
                self.display_cropped_images(result['type_crops'])
            
            # Update status tab
            self.gui.status_handlers.update_bottle_detection_status("เสร็จสิ้น", False)
            
            # Update performance stats
            if result.get('bottle_type') and result.get('combined_ocr_text'):
                self.gui.successful_detections_count += 1
                self.gui.status_handlers.update_performance_stats(successful_detections=self.gui.successful_detections_count)
            
            self.gui.status_label.setText('✅ แสดงผลลัพธ์จากคิว')
            self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            print("✅ DISPLAY FROM QUEUE: Bottle result displayed")
            
        except Exception as e:
            print(f"❌ DISPLAY FROM QUEUE ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_image(self, image):
        """Display image in label"""
        try:
            if image is not None:
                # Resize image for display
                h, w = image.shape[:2]
                max_size = 400
                if h > max_size or w > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cuda_resize(image, (new_w, new_h))
                
                # Convert to RGB for Qt
                rgb_image = cuda_cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, c = rgb_image.shape
                bytes_per_line = c * w
                
                qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                self.gui.image_label.setPixmap(pixmap)
                
                # Update Home tab bottle image
                if hasattr(self.gui, 'home_bottle_image_label'):
                    # Scale pixmap to fit label while maintaining aspect ratio
                    label_size = self.gui.home_bottle_image_label.size()
                    if label_size.width() > 0 and label_size.height() > 0:
                        scale_w = label_size.width() / pixmap.width()
                        scale_h = label_size.height() / pixmap.height()
                        scale = min(scale_w, scale_h)
                        scaled_pixmap = pixmap.scaled(
                            int(pixmap.width() * scale), 
                            int(pixmap.height() * scale), 
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        )
                        self.gui.home_bottle_image_label.setPixmap(scaled_pixmap)
            else:
                self.gui.image_label.setText("ไม่สามารถโหลดภาพได้")

        except Exception as e:
            self.gui.image_label.setText(f"ข้อผิดพลาด: {str(e)}")
            
    def display_cropped_images(self, type_crops):
        """Display cropped type images with OCR results"""
        print(f"🔄 DISPLAY CROPPED IMAGES: Starting to display {len(type_crops) if type_crops else 0} crops")
        
        # Clear existing crops
        for i in reversed(range(self.gui.crops_layout.count())):
            widget = self.gui.crops_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # Clear Home tab crops too
        if hasattr(self.gui, 'home_bottle_crops_layout'):
            for i in reversed(range(self.gui.home_bottle_crops_layout.count())):
                widget = self.gui.home_bottle_crops_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
        
        print("✅ DISPLAY CROPPED IMAGES: Cleared existing crops")
        
        if not type_crops:
            # Show placeholder
            placeholder_label = QLabel("ไม่มีภาพที่ครอป")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_label.setStyleSheet("color: #7f8c8d; padding: 20px;")
            self.gui.crops_layout.addWidget(placeholder_label)
            
            # Also show placeholder in Home tab
            if hasattr(self.gui, 'home_bottle_crops_layout'):
                home_placeholder = QLabel("ไม่มีภาพที่ครอป")
                home_placeholder.setAlignment(Qt.AlignCenter)
                home_placeholder.setStyleSheet("color: #7f8c8d; padding: 20px;")
                self.gui.home_bottle_crops_layout.addWidget(home_placeholder)
            return
        
        # Display each cropped image
        for i, crop in enumerate(type_crops):
            # Create container for this crop
            crop_container = QWidget()
            crop_container.setStyleSheet("border: 1px solid #e67e22; margin: 5px; padding: 5px; background-color: white;")
            crop_layout = QVBoxLayout(crop_container)
            
            # Title for this crop
            crop_title = QLabel(f"Type Region {i+1}")
            crop_title.setStyleSheet("font-weight: bold; color: #e67e22;")
            crop_title.setAlignment(Qt.AlignCenter)
            crop_layout.addWidget(crop_title)
            
            # Display the cropped image
            crop_image = crop['original_crop']
            if crop_image is not None:
                # Resize for display
                h, w = crop_image.shape[:2]
                max_size = 200
                if h > max_size or w > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    crop_image = cuda_resize(crop_image, (new_w, new_h))
                
                # Convert to RGB for Qt
                rgb_crop = cuda_cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                h, w, c = rgb_crop.shape
                bytes_per_line = c * w
                
                qimg = QtGui.QImage(rgb_crop.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                
                crop_label = QLabel()
                crop_label.setPixmap(pixmap)
                crop_label.setAlignment(Qt.AlignCenter)
                crop_layout.addWidget(crop_label)
                
                # Also add to Home tab crops
                if hasattr(self.gui, 'home_bottle_crops_layout'):
                    home_crop_label = QLabel()
                    home_crop_label.setPixmap(pixmap)
                    home_crop_label.setAlignment(Qt.AlignCenter)
                    home_crop_container = QWidget()
                    home_crop_container.setStyleSheet("border: 1px solid #e67e22; margin: 5px; padding: 5px; background-color: white;")
                    home_crop_container_layout = QVBoxLayout(home_crop_container)
                    home_crop_title = QLabel(f"Type Region {i+1}")
                    home_crop_title.setStyleSheet("font-weight: bold; color: #e67e22; font-size: 11px;")
                    home_crop_title.setAlignment(Qt.AlignCenter)
                    home_crop_container_layout.addWidget(home_crop_title)
                    home_crop_container_layout.addWidget(home_crop_label)
                    if 'ocr_text' in crop:
                        home_ocr_label = QLabel(f"OCR: {crop['ocr_text']}")
                        home_ocr_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
                        home_crop_container_layout.addWidget(home_ocr_label)
                    self.gui.home_bottle_crops_layout.addWidget(home_crop_container)
                
                # Add confidence info
                confidence_label = QLabel(f"Detection Confidence: {crop['confidence']:.3f}")
                confidence_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
                confidence_label.setAlignment(Qt.AlignCenter)
                crop_layout.addWidget(confidence_label)
                
                # Add OCR results
                ocr_results = crop.get('ocr_results')
                if ocr_results:
                    # ตรวจสอบว่า ocr_results เป็น list หรือไม่
                    if not isinstance(ocr_results, list):
                        print(f"⚠️ DISPLAY CROPPED IMAGES: Crop {i+1} ocr_results is not a list: {type(ocr_results)}")
                        ocr_results = []
                    
                    if len(ocr_results) > 0:
                        ocr_title = QLabel("📝 OCR Results:")
                        ocr_title.setStyleSheet("font-weight: bold; color: #27ae60; font-size: 11px;")
                        ocr_title.setAlignment(Qt.AlignCenter)
                        crop_layout.addWidget(ocr_title)
                    
                        for j, ocr_result in enumerate(ocr_results):
                            try:
                                # ตรวจสอบว่า ocr_result เป็น dictionary หรือไม่
                                if not isinstance(ocr_result, dict):
                                    print(f"⚠️ DISPLAY CROPPED IMAGES: Crop {i+1}, OCR result {j+1} is not a dict: {type(ocr_result)}")
                                    # ถ้าไม่ใช่ dict ให้แปลงเป็น string
                                    ocr_text = f"  {j+1}. '{str(ocr_result)}'"
                                else:
                                    # ตรวจสอบว่ามี 'text' และ 'confidence' หรือไม่
                                    text = ocr_result.get('text', str(ocr_result.get('recognized_text', '')))
                                    confidence = ocr_result.get('confidence', ocr_result.get('confidence_score', 0.0))
                                    
                                    if not text:
                                        # ถ้าไม่มี text ให้ลองใช้ values ทั้งหมด
                                        text = str(ocr_result)
                                    
                                    ocr_text = f"  {j+1}. '{text}'"
                                    if isinstance(confidence, (int, float)):
                                        ocr_text += f" ({confidence:.3f})"
                                
                                ocr_label = QLabel(ocr_text)
                                ocr_label.setStyleSheet("color: #2c3e50; font-size: 10px; background-color: #ecf0f1; padding: 2px;")
                                ocr_label.setWordWrap(True)
                                crop_layout.addWidget(ocr_label)
                            except Exception as ocr_error:
                                print(f"⚠️ DISPLAY CROPPED IMAGES: Error displaying OCR result {j+1} for crop {i+1}: {ocr_error}")
                                # แสดง error message
                                error_label = QLabel(f"  {j+1}. Error: {str(ocr_error)}")
                                error_label.setStyleSheet("color: #e74c3c; font-size: 10px; font-style: italic;")
                                error_label.setWordWrap(True)
                                crop_layout.addWidget(error_label)
                else:
                    no_ocr_label = QLabel("ไม่พบข้อความ")
                    no_ocr_label.setStyleSheet("color: #e74c3c; font-size: 10px; font-style: italic;")
                    no_ocr_label.setAlignment(Qt.AlignCenter)
                    crop_layout.addWidget(no_ocr_label)
            
            self.gui.crops_layout.addWidget(crop_container)
        
        print(f"✅ DISPLAY CROPPED IMAGES: Displayed {len(type_crops)} cropped images")
        
        # Force GUI update
        QtWidgets.QApplication.processEvents()
        
    def process_current_image(self):
        """Process current image"""
        if self.gui.current_image is None:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "ไม่มีภาพให้ประมวลผล")
            return
        
        print(f"🔄 PROCESS: Starting processing for image shape: {self.gui.current_image.shape}")
        print("🔄 PROCESS: Creating BottleDetectionThread...")
        
        # Update status tab
        self.gui.status_handlers.update_bottle_detection_status("กำลังประมวลผล...", True)
            
        # Start processing thread
        self.gui.progress_bar.setVisible(True)
        self.gui.progress_bar.setValue(0)
        self.gui.status_label.setText('🔄 กำลังประมวลผล...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        
        # Disable process button and enable stop button
        self.gui.btn_process.setEnabled(False)
        # Also update tab buttons if they exist
        if hasattr(self.gui, 'btn_process_bottle_tab'):
            self.gui.btn_process_bottle_tab.setEnabled(False)
        self.gui.btn_stop_processing.setEnabled(True)
        
        self.gui.bottle_processing_cycle_id = getattr(self.gui, 'capture_cycle_id', 0)
        self.gui.processing_thread = BottleDetectionThread(self.gui.current_image, selected_tastes=self.gui.selected_tastes)
        self.gui.processing_thread.result_ready.connect(self.on_processing_complete)
        self.gui.processing_thread.status_updated.connect(self.gui.status_label.setText)
        self.gui.processing_thread.progress_updated.connect(self.update_bottle_progress_bar)
        print("🔄 PROCESS: Starting processing thread...")
        print(f"🔄 PROCESS: Using selected tastes: {self.gui.selected_tastes}")
        self.gui.processing_thread.start()
    
    def on_processing_complete(self, result):
        """Handle processing completion"""
        print("✅ PROCESS COMPLETE: Processing finished")
        self.gui.progress_bar.setVisible(False)
        
        # Update performance stats
        self.gui.total_images_processed_count += 1
        self.gui.status_handlers.update_performance_stats(images_processed=self.gui.total_images_processed_count)
        
        # Reset button states
        self.gui.btn_process.setEnabled(True)
        # Also update tab buttons if they exist
        if hasattr(self.gui, 'btn_process_bottle_tab'):
            self.gui.btn_process_bottle_tab.setEnabled(True)
        self.gui.btn_stop_processing.setEnabled(False)
        
        # ข้ามการแสดงผลถ้าเป็นผลเก่า (ขวดใหม่ถ่ายไปแล้ว)
        if getattr(self.gui, 'bottle_processing_cycle_id', -1) != getattr(self.gui, 'capture_cycle_id', 0):
            print("⚠️ PROCESS COMPLETE: Stale bottle result (new capture already started), skipping display")
            return
        
        # Check if silent mode - store result in queue instead of displaying
        if self.silent_mode:
            print("🔇 SILENT MODE: Storing result in queue (no UI display, no Modbus signals)")
            self.pending_bottle_result = result
            
            # Check if cap result is also ready
            if hasattr(self.gui, 'cap_handlers') and self.gui.cap_handlers.pending_cap_result is not None:
                # Both results are ready - add to queue
                print("🔇 SILENT MODE: Both bottle and cap results ready - adding to queue")
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    self.gui.modbus_thread.pending_results_queue.append((result, self.gui.cap_handlers.pending_cap_result))
                    print(f"🔇 SILENT MODE: Results added to queue (queue size: {len(self.gui.modbus_thread.pending_results_queue)})")
                    # Clear pending results
                    self.pending_bottle_result = None
                    self.gui.cap_handlers.pending_cap_result = None
                else:
                    print("⚠️ SILENT MODE: Modbus thread not available")
            else:
                # Wait for cap result
                print("🔇 SILENT MODE: Bottle result stored, waiting for cap result...")
            
            # Reset silent mode
            self.silent_mode = False
            return
        
        if "error" not in result:
            print("✅ PROCESS COMPLETE: No error in result")
            print(f"✅ PROCESS COMPLETE: Result keys: {list(result.keys())}")
            
            # Store result first
            self.gui.current_result = result
            
            # M100/M110/M120 + มีภาพ Sentech ของ cycle นี้ → ประมวลผลฝาทันที แต่ค่อยสั่งผลเมื่อถึงคิว (ไม่แสดงผลขวดก่อน)
            cycle_id = getattr(self.gui, 'bottle_processing_cycle_id', -1)
            sentech_for_cycle = getattr(self.gui, 'sentech_image_for_cycle', {}).get(cycle_id)
            if (result.get('bottle_type') in ["M100", "M110", "M120"] and result.get('combined_ocr_text')
                    and sentech_for_cycle is not None
                    and hasattr(self.gui, 'cap_handlers') and self.gui.cap_handlers):
                print(f"📋 PROCESS COMPLETE: {result['bottle_type']} - ใส่คิวประมวลผลฝา (จะสั่งผลเมื่อถึงคิว)")
                if not hasattr(self.gui, 'last_bottle_result'):
                    self.gui.last_bottle_result = None
                self.gui.last_bottle_result = result
                self.gui.successful_detections_count += 1
                self.gui.status_handlers.update_performance_stats(successful_detections=self.gui.successful_detections_count)
                self.gui.cap_handlers.process_cap_detection_with_image(sentech_for_cycle, result)
                return
            
            # Display results in text area
            print("🔄 PROCESS COMPLETE: Displaying results in text area...")
            self.display_single_results(result)
            print("✅ PROCESS COMPLETE: Results displayed in text area")
            
            # Draw detections on image and display
            if result.get('detections'):
                print(f"🔄 PROCESS COMPLETE: Drawing {len(result['detections'])} detections on image...")
                result_image = draw_detections_on_image(
                    result['image'], 
                    result['detections']
                )
                self.display_result_image(result_image)
                print("✅ PROCESS COMPLETE: Result image displayed")
            else:
                print("⚠️ PROCESS COMPLETE: No detections to draw")
                # Display original image if no detections
                if result.get('image') is not None:
                    self.display_result_image(result['image'])
                    print("✅ PROCESS COMPLETE: Original image displayed")
            
            # Display cropped images with OCR
            if result.get('type_crops'):
                print(f"🔄 PROCESS COMPLETE: Displaying {len(result['type_crops'])} cropped images...")
                self.display_cropped_images(result['type_crops'])
                print("✅ PROCESS COMPLETE: Cropped images displayed")
            else:
                print("⚠️ PROCESS COMPLETE: No cropped images to display")
            
            # Update status tab
            self.gui.status_handlers.update_bottle_detection_status("เสร็จสิ้น", False)
            
            # Handle bottle type detection and Modbus control
            if result.get('bottle_type') and result.get('combined_ocr_text'):
                print(f"🎯 PROCESS COMPLETE: Bottle type detected: {result['bottle_type']}")
                print(f"📝 PROCESS COMPLETE: OCR text: '{result['combined_ocr_text']}'")
                # เก็บ result ไว้เพื่อใช้ใน handle_bottle_type_detection
                if not hasattr(self.gui, 'last_bottle_result'):
                    self.gui.last_bottle_result = None
                self.gui.last_bottle_result = result
                self.gui.successful_detections_count += 1
                self.gui.status_handlers.update_performance_stats(successful_detections=self.gui.successful_detections_count)
                self.handle_bottle_type_detection(result['bottle_type'], result['combined_ocr_text'])
                
                # ไม่ต้องเปิดใช้งานปุ่มประมวลผลฝาแล้ว - จะเปิดเมื่อมีภาพจาก Sentech
                # (ใช้ค่ากลางที่วิเคราะห์ได้โดยตรง ไม่ต้องรอ bottle_type)
            else:
                print("❌ PROCESS COMPLETE: No bottle type or OCR text found")
                print(f"❌ PROCESS COMPLETE: bottle_type: {result.get('bottle_type')}")
                print(f"❌ PROCESS COMPLETE: combined_ocr_text: {result.get('combined_ocr_text')}")
            
            # Add to history (always add, even if no bottle_type detected)
            # Wait a bit for cap processing to complete if it's still running
            if hasattr(self.gui, 'add_to_history'):
                # Check if cap processing is still running
                cap_result = getattr(self.gui, 'current_cap_result', None)
                # If cap result is not ready yet, add bottle result only
                # Cap result will be added later when cap processing completes
                self.gui.add_to_history(result, cap_result)

            self.gui.status_label.setText('✅ ประมวลผลเสร็จสิ้น')
            self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        else:
            print(f"❌ PROCESS COMPLETE: Error in result: {result['error']}")
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {result["error"]}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            self.gui.error_count_value += 1
            self.gui.status_handlers.update_performance_stats(errors=self.gui.error_count_value)
    
    def handle_bottle_type_detection(self, bottle_type, ocr_text):
        """Handle bottle type detection and control Modbus"""
        print(f"🎯 BOTTLE TYPE DETECTED: {bottle_type} (OCR: '{ocr_text}')")
        
        # เก็บประเภทขวดที่ตรวจพบ
        self.gui.current_bottle_type = bottle_type
        
        # Update status tab
        self.gui.status_handlers.update_bottle_type_status(bottle_type)
        self.gui.status_handlers.update_bottle_detection_status("ตรวจพบแล้ว", False)
        
        # ตรวจสอบว่าเป็นโหมดถ่ายภาพซ้ำหรือไม่
        if self.gui.modbus_thread.angle3_retry_mode:
            print(f"🔄 ANGLE3 RETRY MODE: Detected {bottle_type} - checking if non-angle3...")
            
            if bottle_type != "M130":
                # ตรวจจับได้อย่างอื่นนอกจาก angle3 - ส่งสัญญาณตามประเภทขวดและหยุดโหมดถ่ายภาพซ้ำ
                print(f"✅ NON-ANGLE3 DETECTED: {bottle_type} - ON bottle type signal and M600")
                
                # ส่งสัญญาณตามประเภทขวด (ON M100/M110/M120)
                bottle_signal_success = False
                if bottle_type == "M100":
                    bottle_signal_success = self.gui.modbus_thread.on_m100()
                    if bottle_signal_success:
                        print(f"🏷️ M100 = ON ({bottle_type} - non-angle3 detection)")
                elif bottle_type == "M110":
                    bottle_signal_success = self.gui.modbus_thread.on_m110()
                    if bottle_signal_success:
                        print(f"🏷️ M110 = ON ({bottle_type} - non-angle3 detection)")
                elif bottle_type == "M120":
                    bottle_signal_success = self.gui.modbus_thread.on_m120()
                    if bottle_signal_success:
                        print(f"🏷️ M120 = ON ({bottle_type} - non-angle3 detection)")
                
                if bottle_signal_success:
                    # อัปเดต lamp ตามประเภทขวดเพื่อให้ UI เห็นสถานะ
                    if bottle_type == "M100":
                        self.gui.status_handlers.update_coil_lamp("m100", True)
                        self.gui.status_handlers.update_bottle_type_status("M100")
                    elif bottle_type == "M110":
                        self.gui.status_handlers.update_coil_lamp("m110", True)
                        self.gui.status_handlers.update_bottle_type_status("M110")
                    elif bottle_type == "M120":
                        self.gui.status_handlers.update_coil_lamp("m120", True)
                        self.gui.status_handlers.update_bottle_type_status("M120")
                
                # ส่ง M600
                m600_success = self.gui.modbus_thread.on_m600()
                if m600_success:
                    print("🚀 ON M600 (non-angle3 detection)")
                    self.gui.status_handlers.update_coil_lamp("m600", True)
                    
                    if bottle_signal_success:
                        self.gui.status_label.setText(f'🚀 ตรวจพบ: {bottle_type} (ไม่ใช่ angle3) → ON {bottle_type}+M600')
                        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                    else:
                        self.gui.status_label.setText(f'🚀 ตรวจพบ: {bottle_type} (ไม่ใช่ angle3) → ON M600 (ไม่สามารถ ON {bottle_type} ได้)')
                        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
                    
                    # หยุดโหมดถ่ายภาพซ้ำ
                    self.gui.modbus_thread.angle3_retry_mode = False
                    self.gui.modbus_thread.angle3_retry_count = 0
                    self.gui.status_handlers.update_angle3_retry_status(False)
                    print("🛑 หยุดโหมดถ่ายภาพซ้ำ (พบ non-angle3)")
                else:
                    print("❌ ไม่สามารถ ON M600 ได้")
                    self.gui.status_label.setText('❌ ไม่สามารถ ON M600 ได้')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            else:
                # ยังเป็น angle3 - ส่งสัญญาณและถ่ายภาพซ้ำ
                self.gui.modbus_thread.angle3_retry_count += 1
                print(f"🔄 ANGLE3 RETRY: Still angle3 (attempt {self.gui.modbus_thread.angle3_retry_count}/{self.gui.modbus_thread.max_angle3_retries})")
                
                # ตรวจสอบว่า angle3 ถูกตรวจจับเพียงอย่างเดียวหรือไม่ (สำหรับ retry mode)
                angle3_alone = False
                if hasattr(self.gui, 'last_bottle_result') and self.gui.last_bottle_result:
                    angle3_alone = self.gui.last_bottle_result.get('angle3_detected_alone', False)
                elif ocr_text and "angle3 detected by YOLO (alone)" in ocr_text:
                    angle3_alone = True
                
                # ส่งสัญญาณ M130 หรือ M140, M850, M600 สำหรับ angle3
                if angle3_alone:
                    success = self.gui.modbus_thread.on_m130()
                    if success:
                        print(f"🏷️ M130 = ON (angle3 retry - {'OCR อ่านได้' if ocr_readable else 'OCR อ่านไม่ได้'})")
                else:
                    success = self.gui.modbus_thread.on_m140()
                    if success:
                        print(f"🏷️ M140 = ON (angle3 retry - ไม่ใช่ angle3 จริง)")
                if success:
                    
                    # ON M850 สำหรับ M130
                    m850_success = self.gui.modbus_thread.on_m850()
                    if m850_success:
                        print("🏷️ ON M850 (angle3 retry)")
                        self.gui.status_handlers.update_coil_lamp("m850", True)
                    
                    # ON M600 สำหรับ angle3 ด้วย
                    m600_success = self.gui.modbus_thread.on_m600()
                    if m600_success:
                        print("⏳ Angle3 retry: ส่ง M850 และ M600 สำเร็จ")
                        self.gui.status_handlers.update_coil_lamp("m600", True)
                        self.gui.modbus_thread.start_d6006_monitoring()
                        self.gui.status_handlers.update_register_lamp("d6006", "กำลังอ่าน...", True)
                        self.gui.status_label.setText(f'🔄 ยังเป็น angle3 - ส่งสัญญาณและถ่ายภาพใหม่ (ครั้งที่ {self.gui.modbus_thread.angle3_retry_count})')
                        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
                        
                        # Update status tab
                        self.gui.status_handlers.update_angle3_retry_status(True, self.gui.modbus_thread.angle3_retry_count, self.gui.modbus_thread.max_angle3_retries)
                        
                        # ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)
                        print("📸 ANGLE3 RETRY: ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)")
                        self.capture_image_auto()  # ถ่ายภาพ USB ใหม่ทันที
                    else:
                        print("❌ ไม่สามารถ ON M600 ได้ (angle3 retry)")
                        self.gui.status_label.setText('❌ ไม่สามารถ ON M600 ได้ (angle3 retry)')
                        self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                else:
                    print(f"❌ ไม่สามารถ ON M130/M140 ได้ (angle3 retry)")
                    self.gui.status_label.setText(f'❌ ไม่สามารถ ON M130/M140 ได้ (angle3 retry)')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            return
        
        # ตรวจสอบประเภทขวดและดำเนินการตามลำดับ
        if bottle_type == "M130":
            # M130 (angle3) - ตรวจสอบว่า angle3 ถูกตรวจจับเพียงอย่างเดียวหรือไม่
            # ถ้า angle3 ถูกตรวจจับพร้อมกับ labels อื่นๆ (angle1, type) → ส่ง 50
            # ถ้า angle3 ถูกตรวจจับเพียงอย่างเดียว → ส่ง 40
            
            # ตรวจสอบจาก result ว่า angle3_detected_alone เป็น True หรือไม่
            # ถ้าไม่มีข้อมูลนี้ ให้ตรวจสอบจาก ocr_text
            angle3_alone = False
            if hasattr(self.gui, 'last_bottle_result') and self.gui.last_bottle_result:
                angle3_alone = self.gui.last_bottle_result.get('angle3_detected_alone', False)
            elif ocr_text and "angle3 detected by YOLO (alone)" in ocr_text:
                angle3_alone = True
            
            if angle3_alone:
                # angle3 ถูกตรวจจับเพียงอย่างเดียว → ON M130
                print("🏷️ M130 = ON (angle3 ถูกตรวจจับเพียงอย่างเดียว - เป็น angle3 จริงๆ)")
                success = self.gui.modbus_thread.on_m130()
            else:
                # angle3 ถูกตรวจจับพร้อมกับ labels อื่นๆ → ON M140
                print("🏷️ M140 = ON (angle3 ถูกตรวจจับพร้อมกับ labels อื่นๆ - ไม่ใช่ angle3 จริงๆ)")
                success = self.gui.modbus_thread.on_m140()
            if success:
                
                # ON M850 สำหรับ M130
                m850_success = self.gui.modbus_thread.on_m850()
                if m850_success:
                    print("🏷️ ON M850 (angle3)")
                    self.gui.update_coil_lamp("m850", True)
                else:
                    print("❌ ไม่สามารถ ON M850 ได้")
                
                # ON M600 สำหรับ M130 ด้วย
                m600_success = self.gui.modbus_thread.on_m600()
                if m600_success:
                    print("⏳ ON M130, M850 และ M600 สำเร็จ - เริ่มอ่าน D6006...")
                    self.gui.status_handlers.update_coil_lamp("m600", True)
                    self.gui.modbus_thread.start_d6006_monitoring()
                    self.gui.update_register_lamp("d6006", "กำลังอ่าน...", True)
                    self.gui.status_label.setText('🏷️ ตรวจพบ: M130 → ON M850+M600 → กำลังอ่าน D6006... (ไม่ตรวจจับฝา)')
                    self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                    
                    # ไม่ต้องประมวลผลฝาสำหรับ angle3
                    print("⚠️ ANGLE3: ข้ามการประมวลผลฝา")
                else:
                    print("❌ ไม่สามารถ ON M600 ได้")
                    self.gui.status_label.setText('❌ ไม่สามารถ ON M600 ได้')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            else:
                print("❌ ไม่สามารถ ON M130 ได้")
                self.gui.status_label.setText('❌ ไม่สามารถ ON M130 ได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            return  # M130 ไม่ต้องไปต่อที่ M600 logic
        
        # M100/M110/M120 - เก็บประเภทขวดไว้รอผลลัพธ์ฝา
        print(f"⏳ ตรวจพบ: {bottle_type} - รอผลลัพธ์ฝาก่อน ON Modbus...")
        self.gui.status_label.setText(f'⏳ ตรวจพบ: {bottle_type} - รอผลลัพธ์ฝาก่อน ON Modbus...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        
        # เริ่มประมวลผลฝา (ถ้ามีภาพจาก Sentech)
        if self.gui.current_sentech_image is not None:
            print("🔄 AUTO CAP PROCESS: เริ่มประมวลผลฝา...")
            
            # แสดงผล GUI สำหรับการประมวลผลฝา
            self.gui.cap_handlers.display_cap_processing_ui()
            
            # เรียกใช้ process_cap_detection() เหมือนการกดปุ่ม (ทำงานใน thread แยก ไม่บล็อก GUI)
            print("🔍 AUTO CAP PROCESS: Calling process_cap_detection()...")
            if hasattr(self.gui, 'cap_handlers') and self.gui.cap_handlers:
                self.gui.cap_handlers.process_cap_detection()
                print("🔍 AUTO CAP PROCESS: Cap processing started (ผลจะแสดงเมื่อเสร็จ via signal)")
            else:
                print("❌ AUTO CAP PROCESS: cap_handlers ไม่พร้อม")
            # ไม่รอ thread ที่นี่ — ผลจะมาที่ on_cap_processing_complete ผ่าน signal เพื่อไม่ให้ GUI ค้าง
        else:
            print("⚠️ AUTO CAP PROCESS: ไม่มีภาพจาก Sentech - ไม่สามารถประมวลผลฝาได้")
            self.gui.status_label.setText('⚠️ ไม่มีภาพจาก Sentech - ไม่สามารถประมวลผลฝาได้')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def display_single_results(self, result):
        """Display single image results"""
        print("🔄 DISPLAY SINGLE RESULTS: Starting to display results")
        
        summary = result.get('summary', {})
        detections = result.get('detections', [])
        
        print(f"🔄 DISPLAY SINGLE RESULTS: Summary: {summary}")
        print(f"🔄 DISPLAY SINGLE RESULTS: Detections: {len(detections)}")
        
        detail_text = f"""
=== ผลการตรวจจับขวดและ OCR ===
📊 วัตถุที่ตรวจจับได้: {summary.get('total_detections', 0)}
🔍 Type regions: {summary.get('type_detections', 0)}
🏷️ Labels: {', '.join(summary.get('detected_labels', []))}
        """
            
        if result.get('type_crops'):
            detail_text += f"\n🖼️ ภาพที่ครอป: {len(result['type_crops'])} รูป\n"
            
            # Add OCR results summary
            total_ocr_texts = 0
            for crop in result['type_crops']:
                if crop.get('ocr_results'):
                    total_ocr_texts += len(crop['ocr_results'])
            
            detail_text += f"📝 ข้อความที่อ่านได้: {total_ocr_texts} รายการ\n"
            
            # Add combined OCR text and bottle type
            if result.get('combined_ocr_text'):
                detail_text += f"\n📝 ข้อความรวม: '{result['combined_ocr_text']}'\n"
                
                if result.get('bottle_type'):
                    detail_text += f"🏷️ ประเภทขวด: {result['bottle_type']}\n"
                    if result['bottle_type'] == "M100":
                        detail_text += "   → พบคำว่า 'เดิม' (ดั้งเดิม)\n"
                    elif result['bottle_type'] == "M110":
                        detail_text += "   → พบคำว่า '2%' (น้ำตาล 2%)\n"
                    elif result['bottle_type'] == "M120":
                        detail_text += "   → พบคำว่า 'ลัก' (ผสมแมงลัก)\n"
                else:
                    detail_text += "❌ ไม่พบคำว่า 'เดิม', '2%', หรือ 'ลัก'\n"
            
        self.gui.results_text.setText(detail_text)
        
        # Update Home tab results too
        if hasattr(self.gui, 'home_bottle_results_text'):
            self.gui.home_bottle_results_text.setText(detail_text)
        print("✅ DISPLAY SINGLE RESULTS: Results text set")
        
        # Force GUI update
        QtWidgets.QApplication.processEvents()
        
    def display_result_image(self, image):
        """Display result image with detections"""
        try:
            print(f"🔄 DISPLAY RESULT IMAGE: Image shape: {image.shape}")
            
            # Resize image for display (use CPU to avoid blocking)
            h, w = image.shape[:2]
            max_size = 400
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                # Use CPU resize for display to avoid CUDA blocking
                image = cv2.resize(image, (new_w, new_h))
                print(f"🔄 DISPLAY RESULT IMAGE: Resized to {new_w}x{new_h}")
            
            # Convert to RGB for Qt (use CPU to avoid blocking)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_image.shape
            bytes_per_line = c * w
            
            # Ensure image is contiguous
            if not rgb_image.flags['C_CONTIGUOUS']:
                rgb_image = np.ascontiguousarray(rgb_image)
            
            qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.gui.image_label.setPixmap(pixmap)
            print("✅ DISPLAY RESULT IMAGE: Image displayed in label")
            
            # Force GUI update (non-blocking)
            QtWidgets.QApplication.processEvents()
        except Exception as e:
            print(f"❌ Error displaying result image: {e}")
            import traceback
            traceback.print_exc()
    
    def update_bottle_progress_bar(self, value):
        """Update bottle detection progress bar with specific value"""
        self.gui.progress_bar.setValue(value)
        self.gui.progress_bar.setFormat(f"กำลังประมวลผล... ({value}%)")
    
    def reset_processing_for_new_image(self):
        """Reset processing state for new image after D6006=300"""
        try:
            print("🔄 RESET PROCESSING: Clearing current data for new image...")
            
            # Clear current images
            self.gui.current_image = None
            self.gui.current_sentech_image = None
            self.gui.current_cap_image = None
            self.gui.original_cap_image = None
            self.gui.ai_rotated_image = None
            
            # Clear current results
            self.gui.current_cap_result = None
            self.gui.current_rotation_angle = 0
            
            # Reset progress bars
            self.gui.progress_bar.setVisible(False)
            self.gui.cap_progress_bar.setVisible(False)
            
            # Reset button states
            self.gui.btn_process.setEnabled(False)
            self.gui.btn_save_image.setEnabled(False)  # Disable save button when image is cleared
            # Also update tab buttons if they exist
            if hasattr(self.gui, 'btn_process_bottle_tab'):
                self.gui.btn_process_bottle_tab.setEnabled(False)
            if hasattr(self.gui, 'btn_save_bottle_image_tab'):
                self.gui.btn_save_bottle_image_tab.setEnabled(False)
            self.gui.btn_process_cap.setEnabled(False)
            self.gui.btn_stop_processing.setEnabled(False)
            
            # Clear image displays
            self.gui.image_label.clear()
            self.gui.sentech_image_label.clear()
            
            # Clear text displays
            self.gui.results_text.clear()
            self.gui.cap_detection_text.clear()
            
            # Update status
            self.gui.status_label.setText('🔄 RESET การประมวลผล - พร้อมรับภาพใหม่')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            
            print("✅ RESET PROCESSING: Processing state cleared for new image")
            
        except Exception as e:
            print(f"❌ Error resetting processing state: {e}")
    
    def update_bottle_progress_bar(self, value):
        """Update bottle detection progress bar with specific value"""
        self.gui.progress_bar.setValue(value)
        self.gui.progress_bar.setFormat(f"กำลังประมวลผล... ({value}%)")


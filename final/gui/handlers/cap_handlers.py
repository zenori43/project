# -*- coding: utf-8 -*-
"""
Cap Detection Event Handlers
Handles all events related to cap detection processing
"""

from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QThread
from core.business_logic import CapDetectionThread
import cv2
import numpy as np
import datetime
import os

# Import CUDA image utilities
from core.cuda_image_utils import cuda_cvtColor


class CapDetectionHandlers:
    """Event handlers for cap detection functionality"""
    
    def __init__(self, gui_instance):
        """
        Initialize handlers with reference to GUI instance
        
        Args:
            gui_instance: Reference to BottleDetectionGUI instance
        """
        self.gui = gui_instance
        self.silent_mode = False  # Flag for silent processing (no UI display, no Modbus signals)
        self.pending_cap_result = None  # Store result for silent processing
    
    def select_cap_image_file(self):
        """Select image file for cap detection processing"""
        try:
            # Open file dialog to select image
            file_path, _ = QFileDialog.getOpenFileName(
                self.gui,
                "เลือกไฟล์ภาพสำหรับการตรวจจับฝา",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            
            if file_path:
                print(f"📁 CAP IMAGE SELECTION: Selected file: {file_path}")
                self.gui.status_label.setText('📁 กำลังโหลดภาพฝา...')
                self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
                QtWidgets.QApplication.processEvents()
                
                # Load image using OpenCV
                image = cv2.imread(file_path)
                if image is not None:
                    # Set both current_cap_image and current_sentech_image for compatibility
                    self.gui.current_cap_image = image
                    self.gui.current_sentech_image = image
                    self.gui.original_cap_image = image.copy()  # Store original for rotation
                    self.gui.current_rotation_angle = 0  # Reset rotation angle
                    self.display_sentech_image(image)
                    self.gui.sentech_image_info_label.setText(f"ขนาด: {image.shape[1]}x{image.shape[0]} | ไฟล์: {file_path.split('/')[-1]}")
                    if hasattr(self.gui, 'home_cap_image_info_label'):
                        self.gui.home_cap_image_info_label.setText(f"ขนาด: {image.shape[1]}x{image.shape[0]} | ไฟล์: {file_path.split('/')[-1]}")
                    self.gui.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
                    self.gui.btn_save_sentech_image.setEnabled(True)  # Enable save button when image is loaded
                    # Manual rotation buttons removed
                    self.gui.status_label.setText('✅ โหลดภาพฝาสำเร็จ - พร้อมประมวลผล')
                    self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                    print(f"✅ CAP IMAGE SELECTION: Successfully loaded image with shape: {image.shape}")
                else:
                    self.gui.status_label.setText('❌ ไม่สามารถโหลดภาพฝาได้')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    QMessageBox.warning(self.gui, "ข้อผิดพลาด", f"ไม่สามารถโหลดภาพจากไฟล์:\n{file_path}")
                    print(f"❌ CAP IMAGE SELECTION: Failed to load image from: {file_path}")
                    
        except Exception as e:
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเลือกไฟล์ฝา:\n{str(e)}")
            print(f"❌ CAP IMAGE SELECTION ERROR: {str(e)}")
    
    def select_multiple_cap_images(self):
        """Select multiple image files for batch cap detection processing"""
        try:
            # Open file dialog to select multiple images
            file_paths, _ = QFileDialog.getOpenFileNames(
                self.gui,
                "เลือกหลายไฟล์ภาพสำหรับการตรวจจับฝา",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            
            if not file_paths:
                return
            
            print(f"📁 BATCH CAP PROCESSING: Selected {len(file_paths)} files")
            
            # Confirm batch processing
            reply = QMessageBox.question(
                self.gui,
                "ยืนยันการประมวลผลหลายภาพ",
                f"คุณต้องการประมวลผล {len(file_paths)} ภาพและบันทึกลงประวัติหรือไม่?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # Setup progress bar for batch processing
            if hasattr(self.gui, 'cap_progress_bar'):
                self.gui.cap_progress_bar.setVisible(True)
                self.gui.cap_progress_bar.setValue(0)
                self.gui.cap_progress_bar.setMaximum(len(file_paths))
                self.gui.cap_progress_bar.setFormat("กำลังเริ่มต้น... (0%)")
            
            # Process each image
            self.gui.status_label.setText(f'🔄 กำลังประมวลผล {len(file_paths)} ภาพ...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            if not all([self.gui.cap_detector, self.gui.craft_detector, self.gui.rotation_model, self.gui.line_detector, self.gui.ocr_model]):
                QMessageBox.warning(self.gui, "ข้อผิดพลาด", "โมเดลตรวจจับฝายังไม่ได้โหลดเสร็จ")
                if hasattr(self.gui, 'cap_progress_bar'):
                    self.gui.cap_progress_bar.setVisible(False)
                return
            
            success_count = 0
            error_count = 0
            faded_files = []  # เก็บรายชื่อไฟล์ที่จาง
            
            for idx, file_path in enumerate(file_paths, 1):
                try:
                    # Update progress bar
                    progress_percent = int((idx - 1) / len(file_paths) * 100)
                    if hasattr(self.gui, 'cap_progress_bar'):
                        self.gui.cap_progress_bar.setValue(idx - 1)
                        self.gui.cap_progress_bar.setFormat(f"ภาพ {idx-1}/{len(file_paths)} เสร็จแล้ว ({progress_percent}%)")
                    
                    print(f"📁 BATCH [{idx}/{len(file_paths)}]: Processing {file_path}")
                    
                    # Update status with current image name
                    file_name = os.path.basename(file_path)
                    if len(file_name) > 40:
                        file_name = file_name[:37] + "..."
                    self.gui.status_label.setText(f'🔄 กำลังประมวลผลภาพ {idx}/{len(file_paths)}: {file_name}')
                    QtWidgets.QApplication.processEvents()
                    
                    # Load image
                    image = cv2.imread(file_path)
                    if image is None:
                        print(f"❌ BATCH [{idx}/{len(file_paths)}]: Failed to load {file_path}")
                        error_count += 1
                        # Update progress even on error
                        if hasattr(self.gui, 'cap_progress_bar'):
                            self.gui.cap_progress_bar.setValue(idx)
                            self.gui.cap_progress_bar.setFormat(f"ภาพ {idx}/{len(file_paths)} เสร็จแล้ว (มีข้อผิดพลาด)")
                        continue
                    
                    # Process cap detection
                    result = self.process_cap_detection_sync(image, file_path)
                    
                    # Check if result is valid (success or faded text detected)
                    is_faded_text = result and "error" in result and result.get("error") == "faded_text_detected"
                    is_success = result and "error" not in result
                    
                    if is_success or is_faded_text:
                        # Create a dummy bottle result for history (since we're only processing caps)
                        bottle_result = {
                            'image': None,
                            'bottle_type': 'Batch Processing' if is_success else 'Faded Text Detected',
                            'combined_ocr_text': f'Batch: {os.path.basename(file_path)}'
                        }
                        
                        # Add to history (บันทึกทั้ง success และ faded text)
                        if hasattr(self.gui, 'add_to_history'):
                            self.gui.add_to_history(bottle_result, result)
                            if is_faded_text:
                                # เก็บชื่อไฟล์ที่จาง
                                file_name = os.path.basename(file_path)
                                faded_files.append(file_name)
                                print(f"⚠️ BATCH [{idx}/{len(file_paths)}]: Added faded text result to history - {file_name}")
                                # นับ faded text เป็น success (เพราะเป็นผลลัพธ์ที่ถูกต้อง แค่เป็นข้อความจาง)
                                success_count += 1
                            else:
                                print(f"✅ BATCH [{idx}/{len(file_paths)}]: Added to history")
                                success_count += 1
                    else:
                        print(f"❌ BATCH [{idx}/{len(file_paths)}]: Processing failed - {result.get('error', 'Unknown error') if result else 'No result'}")
                        error_count += 1
                    
                    # Update progress bar after processing
                    final_progress = int(idx / len(file_paths) * 100)
                    if hasattr(self.gui, 'cap_progress_bar'):
                        self.gui.cap_progress_bar.setValue(idx)
                        if idx < len(file_paths):
                            self.gui.cap_progress_bar.setFormat(f"ภาพ {idx}/{len(file_paths)} เสร็จแล้ว ({final_progress}%)")
                        else:
                            self.gui.cap_progress_bar.setFormat(f"เสร็จสิ้นทั้งหมด {len(file_paths)} ภาพ (100%)")
                    
                    # Small delay to allow GUI update
                    QtWidgets.QApplication.processEvents()
                    
                except Exception as e:
                    print(f"❌ BATCH [{idx}/{len(file_paths)}]: Error processing {file_path}: {e}")
                    error_count += 1
                    # Update progress even on error
                    if hasattr(self.gui, 'cap_progress_bar'):
                        self.gui.cap_progress_bar.setValue(idx)
                        self.gui.cap_progress_bar.setFormat(f"ภาพ {idx}/{len(file_paths)} เสร็จแล้ว (มีข้อผิดพลาด)")
                    import traceback
                    traceback.print_exc()
            
            # Hide progress bar
            if hasattr(self.gui, 'cap_progress_bar'):
                self.gui.cap_progress_bar.setVisible(False)
            
            # Show completion message
            self.gui.status_label.setText(f'✅ ประมวลผลเสร็จสิ้น: สำเร็จ {success_count}/{len(file_paths)}, ผิดพลาด {error_count}')
            self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            # สร้างข้อความสรุปผล
            summary_message = f"ประมวลผลเสร็จสิ้น:\n✅ สำเร็จ: {success_count} ภาพ\n❌ ผิดพลาด: {error_count} ภาพ"
            
            # เพิ่มรายชื่อภาพที่จาง
            if faded_files:
                summary_message += f"\n\n⚠️ ภาพที่จาง ({len(faded_files)} ภาพ):"
                # แสดงสูงสุด 10 ภาพแรก (เพื่อไม่ให้ข้อความยาวเกินไป)
                display_files = faded_files[:10]
                for faded_file in display_files:
                    summary_message += f"\n  • {faded_file}"
                if len(faded_files) > 10:
                    summary_message += f"\n  ... และอีก {len(faded_files) - 10} ภาพ"
            
            summary_message += "\n\nผลลัพธ์ถูกบันทึกลง tab ประวัติแล้ว"
            
            QMessageBox.information(
                self.gui,
                "ประมวลผลเสร็จสิ้น",
                summary_message
            )
            print(f"✅ BATCH PROCESSING COMPLETE: Success: {success_count}, Errors: {error_count}, Faded: {len(faded_files)}")
            if faded_files:
                print(f"⚠️ FADED FILES: {faded_files}")
            
        except Exception as e:
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการประมวลผลหลายภาพ:\n{str(e)}")
            print(f"❌ BATCH CAP PROCESSING ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def process_cap_detection_sync(self, image, file_path=None):
        """Process cap detection synchronously (for batch processing)"""
        try:
            if not all([self.gui.cap_detector, self.gui.craft_detector, self.gui.rotation_model, self.gui.line_detector, self.gui.ocr_model]):
                return {'error': 'Models not loaded'}
            
            # Create processing thread
            cap_thread = CapDetectionThread(
                image,
                self.gui.cap_detector,
                self.gui.craft_detector,
                self.gui.rotation_model,
                self.gui.line_detector,
                self.gui.ocr_model,
                getattr(self.gui, 'faded_text_yolo_model', None),
                bottle_type=None
            )
            
            # Store result
            result_container = {'result': None, 'completed': False}
            
            def on_result_ready(result):
                result_container['result'] = result
                result_container['completed'] = True
            
            def on_status_updated(status):
                # Update status during processing (for batch mode, we can show sub-progress)
                if hasattr(self.gui, 'status_label') and file_path:
                    # Keep the batch progress message but add sub-status
                    current_text = self.gui.status_label.text()
                    if "กำลังประมวลผลภาพ" in current_text:
                        # Extract image number from current text
                        parts = current_text.split(": ")
                        if len(parts) > 1:
                            base_msg = parts[0]  # "🔄 กำลังประมวลผลภาพ X/Y"
                            self.gui.status_label.setText(f"{base_msg}: {status}")
                            QtWidgets.QApplication.processEvents()
            
            cap_thread.result_ready.connect(on_result_ready)
            cap_thread.status_updated.connect(on_status_updated)
            cap_thread.start()
            
            # Wait for completion (with timeout)
            import time
            timeout = 60  # 60 seconds timeout
            start_time = time.time()
            while not result_container['completed']:
                QtWidgets.QApplication.processEvents()
                if time.time() - start_time > timeout:
                    print(f"⏱️ BATCH: Timeout waiting for result")
                    cap_thread.terminate()
                    return {'error': 'Timeout'}
                time.sleep(0.1)
            
            cap_thread.wait(1000)  # Wait for thread to finish
            
            return result_container['result']
            
        except Exception as e:
            print(f"❌ BATCH SYNC PROCESSING ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def capture_sentech_image(self):
        """Capture image from Sentech camera using Harvesters"""
        if self.gui.sentech_camera is None:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "กล้อง Sentech ยังไม่ได้เริ่มต้น")
            return
        
        try:
            self.gui.status_label.setText('📸 กำลังถ่ายภาพจาก Sentech...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # Capture image from Sentech
            try:
                print("📸 Calling sentech_camera.capture_image()...")
                captured_image = self.gui.sentech_camera.capture_image()
                print(f"📸 Capture result: {captured_image is not None}")
            except Exception as e:
                print(f"❌ Error calling sentech_camera.capture_image(): {e}")
                import traceback
                traceback.print_exc()
                captured_image = None
            
            if captured_image is not None:
                # ล้างข้อมูลการประมวลผลฝาเก่าก่อนที่จะใช้ภาพใหม่
                self.gui.current_cap_result = None
                if hasattr(self.gui, 'cap_detection_text') and self.gui.cap_detection_text:
                    self.gui.cap_detection_text.clear()
                
                # ล้าง cap results layout
                if hasattr(self.gui, 'cap_results_layout') and self.gui.cap_results_layout:
                    for i in reversed(range(self.gui.cap_results_layout.count())):
                        widget = self.gui.cap_results_layout.itemAt(i).widget()
                        if widget:
                            widget.setParent(None)
                
                self.gui.current_sentech_image = captured_image
                self.display_sentech_image(captured_image)
                self.gui.sentech_image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]}")
                if hasattr(self.gui, 'home_cap_image_info_label'):
                    self.gui.home_cap_image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]}")
                self.gui.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
                self.gui.btn_save_sentech_image.setEnabled(True)  # Enable save button when image is captured
                self.gui.status_label.setText('✅ ถ่ายภาพจาก Sentech สำเร็จ - พร้อมประมวลผลฝา')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            else:
                print("❌ Failed to capture from Sentech - trying USB camera fallback...")
                # ลองใช้ USB camera เป็น fallback
                if self.gui.usb_camera and hasattr(self.gui.usb_camera, 'is_opened') and self.gui.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback...")
                    fallback_image = self.gui.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful")
                        # ใช้ภาพจาก USB camera แทน
                        self.gui.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.gui.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        if hasattr(self.gui, 'home_cap_image_info_label'):
                            self.gui.home_cap_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        self.gui.btn_process_cap.setEnabled(True)
                        self.gui.btn_save_sentech_image.setEnabled(True)  # Enable save button when fallback image is used
                        self.gui.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        return
                
                self.gui.status_label.setText('❌ ไม่สามารถถ่ายภาพจาก Sentech ได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            print(f"❌ Error capturing Sentech image: {e} - trying USB camera fallback...")
            # ลองใช้ USB camera เป็น fallback
            try:
                if self.gui.usb_camera and hasattr(self.gui.usb_camera, 'is_opened') and self.gui.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback after error...")
                    fallback_image = self.gui.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful after error")
                        # ใช้ภาพจาก USB camera แทน
                        self.gui.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.gui.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        if hasattr(self.gui, 'home_cap_image_info_label'):
                            self.gui.home_cap_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        self.gui.btn_process_cap.setEnabled(True)
                        self.gui.btn_save_sentech_image.setEnabled(True)  # Enable save button when fallback image is used
                        self.gui.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        return
            except:
                pass
            
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_sentech_image_auto(self):
        """Capture image automatically from Sentech camera after M301 trigger using Harvesters"""
        if self.gui.sentech_camera is None:
            print("❌ Sentech camera not initialized - cannot capture")
            return
        
        # รอให้กล้องเริ่มต้นเสร็จ (ถ้ายังไม่เสร็จ)
        max_wait_time = 10.0  # รอสูงสุด 10 วินาที
        wait_time = 0.0
        while hasattr(self.gui.sentech_camera, '_is_initialized') and not self.gui.sentech_camera._is_initialized and wait_time < max_wait_time:
            QThread.msleep(100)  # Non-blocking sleep
            QtWidgets.QApplication.processEvents()  # Keep GUI responsive
            wait_time += 0.1
        
        if hasattr(self.gui.sentech_camera, '_is_initialized') and not self.gui.sentech_camera._is_initialized:
            print("❌ Sentech camera initialization timeout - cannot capture")
            return
        
        try:
            print("📸 SENTECH AUTO CAPTURE: Starting image capture after M301 ON...")
            self.gui.status_label.setText('📸 กำลังถ่ายภาพอัตโนมัติจาก Sentech...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ตรวจสอบ Harvesters acquisition status
            print("🔍 SENTECH AUTO CAPTURE: Checking Harvesters acquisition status...")
            try:
                if hasattr(self.gui.sentech_camera, '_ia') and self.gui.sentech_camera._ia is not None:
                    print("✅ SENTECH AUTO CAPTURE: Image acquirer is available")
                else:
                    print("❌ SENTECH AUTO CAPTURE: Image acquirer is not available")
                    return
            except Exception as e:
                print(f"❌ SENTECH AUTO CAPTURE: Error checking image acquirer: {e}")
                return
            
            # ถ่ายภาพทันทีโดยไม่รอ callback
            print("📸 SENTECH AUTO CAPTURE: Capturing image immediately after M301 ON...")
            self.gui.status_label.setText('📸 กำลังถ่ายภาพจาก Sentech หลัง M301 ON...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ใช้ capture_image() ที่ใช้ Harvesters
            try:
                print("📸 Calling sentech_camera.capture_image()...")
                captured_image = self.gui.sentech_camera.capture_image()
                print(f"📸 Capture result: {captured_image is not None}")
            except Exception as e:
                print(f"❌ Error calling sentech_camera.capture_image(): {e}")
                import traceback
                traceback.print_exc()
                captured_image = None
            
            if captured_image is not None:
                # ล้างข้อมูลการประมวลผลฝาเก่าก่อนที่จะใช้ภาพใหม่
                self.gui.current_cap_result = None
                if hasattr(self.gui, 'cap_detection_text') and self.gui.cap_detection_text:
                    self.gui.cap_detection_text.clear()
                
                # ล้าง cap results layout
                if hasattr(self.gui, 'cap_results_layout') and self.gui.cap_results_layout:
                    for i in reversed(range(self.gui.cap_results_layout.count())):
                        widget = self.gui.cap_results_layout.itemAt(i).widget()
                        if widget:
                            widget.setParent(None)
                
                # แสดงเวลาที่ถ่ายภาพ
                capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"🕐 Sentech Capture time: {capture_time}")
                
                self.gui.current_sentech_image = captured_image
                self.display_sentech_image(captured_image)
                self.gui.sentech_image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]} | เวลา: {capture_time}")
                if hasattr(self.gui, 'home_cap_image_info_label'):
                    self.gui.home_cap_image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]} | เวลา: {capture_time}")
                self.gui.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
                self.gui.btn_save_sentech_image.setEnabled(True)  # Enable save button when image is captured
                self.gui.status_label.setText('✅ ถ่ายภาพอัตโนมัติจาก Sentech สำเร็จ - พร้อมประมวลผลฝา')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                
                # ฝาไม่เริ่มประมวลผลเอง — รอผลขวดก่อน ถ้าขวด Good ค่อยเริ่มจาก handle_bottle_type_detection
                print("🔄 SENTECH AUTO: เก็บภาพฝาไว้ - รอผลขวด (ถ้าขวด Good จะเริ่มประมวลผลฝาอัตโนมัติ)")
            else:
                # Fallback logic similar to capture_sentech_image
                print("❌ Failed to capture from Sentech - trying USB camera fallback...")
                if self.gui.usb_camera and hasattr(self.gui.usb_camera, 'is_opened') and self.gui.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback...")
                    fallback_image = self.gui.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful")
                        capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        self.gui.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.gui.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} | เวลา: {capture_time} (USB Fallback)")
                        if hasattr(self.gui, 'home_cap_image_info_label'):
                            self.gui.home_cap_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} | เวลา: {capture_time} (USB Fallback)")
                        self.gui.btn_process_cap.setEnabled(True)
                        self.gui.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        
                        # รอผลขวดก่อน — ฝาไม่เริ่มประมวลผลเอง
                        print("🔄 USB FALLBACK: เก็บภาพฝาไว้ - รอผลขวด (ถ้าขวด Good จะเริ่มประมวลผลฝาอัตโนมัติ)")
                        return
                
                self.gui.status_label.setText('❌ ไม่สามารถถ่ายภาพอัตโนมัติจาก Sentech ได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            # Similar fallback logic as in capture_sentech_image
            print(f"❌ Error capturing Sentech image (auto): {e} - trying USB camera fallback...")
            try:
                if self.gui.usb_camera and hasattr(self.gui.usb_camera, 'is_opened') and self.gui.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback after error (auto)...")
                    fallback_image = self.gui.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful after error (auto)")
                        self.gui.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.gui.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        if hasattr(self.gui, 'home_cap_image_info_label'):
                            self.gui.home_cap_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        self.gui.btn_process_cap.setEnabled(True)
                        self.gui.btn_save_sentech_image.setEnabled(True)  # Enable save button when fallback image is used
                        self.gui.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        
                        # รอผลขวดก่อน — ฝาไม่เริ่มประมวลผลเอง
                        print("🔄 USB FALLBACK AUTO: เก็บภาพฝาไว้ - รอผลขวด (ถ้าขวด Good จะเริ่มประมวลผลฝาอัตโนมัติ)")
                        return
            except:
                pass
            
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_sentech_image_from_queue(self):
        """Capture image from Sentech camera from queue (without waiting for M301)"""
        if self.gui.sentech_camera is None:
            print("❌ Sentech camera not initialized - cannot capture from queue")
            return
        
        # รอให้กล้องเริ่มต้นเสร็จ (ถ้ายังไม่เสร็จ)
        max_wait_time = 10.0  # รอสูงสุด 10 วินาที
        wait_time = 0.0
        while hasattr(self.gui.sentech_camera, '_is_initialized') and not self.gui.sentech_camera._is_initialized and wait_time < max_wait_time:
            QThread.msleep(100)  # Non-blocking sleep
            QtWidgets.QApplication.processEvents()  # Keep GUI responsive
            wait_time += 0.1
        
        if hasattr(self.gui.sentech_camera, '_is_initialized') and not self.gui.sentech_camera._is_initialized:
            print("❌ Sentech camera initialization timeout - cannot capture from queue")
            return
        
        try:
            print("📸 SENTECH QUEUE CAPTURE: Starting image capture from queue...")
            self.gui.status_label.setText('📸 กำลังถ่ายภาพจาก Sentech จากคิว...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            
            # ถ่ายภาพทันทีโดยไม่รอ (เพราะเป็นงานจากคิว)
            print("📸 SENTECH QUEUE CAPTURE: Capturing image immediately...")
            self.gui.status_label.setText('📸 กำลังถ่ายภาพจาก Sentech...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ใช้ capture_image() ที่เริ่ม acquisition ใหม่
            try:
                print("📸 Calling sentech_camera.capture_image()...")
                captured_image = self.gui.sentech_camera.capture_image()
                print(f"📸 Capture result: {captured_image is not None}")
            except Exception as e:
                print(f"❌ Error calling sentech_camera.capture_image(): {e}")
                import traceback
                traceback.print_exc()
                captured_image = None
            
            if captured_image is not None:
                # ล้างข้อมูลการประมวลผลฝาเก่าก่อนที่จะใช้ภาพใหม่
                self.gui.current_cap_result = None
                if hasattr(self.gui, 'cap_detection_text') and self.gui.cap_detection_text:
                    self.gui.cap_detection_text.clear()
                
                # ล้าง cap results layout
                if hasattr(self.gui, 'cap_results_layout') and self.gui.cap_results_layout:
                    for i in reversed(range(self.gui.cap_results_layout.count())):
                        widget = self.gui.cap_results_layout.itemAt(i).widget()
                        if widget:
                            widget.setParent(None)
                
                # แสดงเวลาที่ถ่ายภาพ
                capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"🕐 Sentech Queue Capture time: {capture_time}")
                print(f"📸 SENTECH QUEUE CAPTURE: Image shape: {captured_image.shape}")
                
                self.gui.current_sentech_image = captured_image
                self.display_sentech_image(captured_image)
                self.gui.sentech_image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]} | เวลา: {capture_time} | จากคิว")
                if hasattr(self.gui, 'home_cap_image_info_label'):
                    self.gui.home_cap_image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]} | เวลา: {capture_time} | จากคิว")
                self.gui.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
                self.gui.btn_save_sentech_image.setEnabled(True)  # Enable save button when image is captured
                self.gui.status_label.setText('✅ ถ่ายภาพจาก Sentech จากคิวสำเร็จ - พร้อมประมวลผลฝา')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                
                # ฝาไม่เริ่มประมวลผลเอง — รอผลขวดก่อน (ถ้าขวด NG ไม่ประมวลผลฝา)
                print("🔄 SENTECH QUEUE: เก็บภาพฝาไว้ - รอผลขวด (ถ้าขวด Good จะเริ่มประมวลผลฝาอัตโนมัติ)")
            else:
                # Similar fallback logic
                print("❌ Failed to capture from Sentech queue - trying USB camera fallback...")
                if self.gui.usb_camera and hasattr(self.gui.usb_camera, 'is_opened') and self.gui.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback for queue...")
                    fallback_image = self.gui.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful for queue")
                        capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        self.gui.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.gui.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} | เวลา: {capture_time} | USB Fallback")
                        self.gui.btn_process_cap.setEnabled(True)
                        self.gui.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        
                        # รอผลขวดก่อน — ฝาไม่เริ่มประมวลผลเอง
                        print("🔄 USB FALLBACK QUEUE: เก็บภาพฝาไว้ - รอผลขวด (ถ้าขวด Good จะเริ่มประมวลผลฝาอัตโนมัติ)")
                        return
                
                self.gui.status_label.setText('❌ ไม่สามารถถ่ายภาพจาก Sentech จากคิวได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            # Similar fallback logic
            print(f"❌ Error capturing Sentech image (queue): {e} - trying USB camera fallback...")
            try:
                if self.gui.usb_camera and hasattr(self.gui.usb_camera, 'is_opened') and self.gui.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback after error (queue)...")
                    fallback_image = self.gui.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful after error (queue)")
                        self.gui.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.gui.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        if hasattr(self.gui, 'home_cap_image_info_label'):
                            self.gui.home_cap_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        self.gui.btn_process_cap.setEnabled(True)
                        self.gui.btn_save_sentech_image.setEnabled(True)  # Enable save button when fallback image is used
                        self.gui.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        
                        # รอผลขวดก่อน — ฝาไม่เริ่มประมวลผลเอง
                        print("🔄 USB FALLBACK QUEUE (error): เก็บภาพฝาไว้ - รอผลขวด")
                        return
            except:
                pass
            
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_and_process_silently(self):
        """Capture image and process silently (no UI display, no Modbus signals) - store result in queue"""
        if self.gui.sentech_camera is None:
            print("❌ Sentech camera not initialized - cannot capture silently")
            return
        
        # รอให้กล้องเริ่มต้นเสร็จ (ถ้ายังไม่เสร็จ)
        max_wait_time = 10.0
        wait_time = 0.0
        while hasattr(self.gui.sentech_camera, '_is_initialized') and not self.gui.sentech_camera._is_initialized and wait_time < max_wait_time:
            QThread.msleep(100)  # Non-blocking sleep
            QtWidgets.QApplication.processEvents()  # Keep GUI responsive
            wait_time += 0.1
        
        if hasattr(self.gui.sentech_camera, '_is_initialized') and not self.gui.sentech_camera._is_initialized:
            print("❌ Sentech camera initialization timeout - cannot capture silently")
            return
        
        try:
            print("🔇 SILENT CAPTURE: Starting silent Sentech image capture and processing...")
            self.silent_mode = True  # Enable silent mode
            
            # Capture image immediately (no UI updates)
            captured_image = None
            try:
                captured_image = self.gui.sentech_camera.capture_image()
            except Exception as e:
                print(f"❌ Error calling sentech_camera.capture_image() (silent): {e}")
                captured_image = None
            
            if captured_image is not None:
                print(f"🔇 SILENT CAPTURE: Image shape: {captured_image.shape}")
                self.gui.current_sentech_image = captured_image
                
                # Process silently (no UI display, no Modbus signals)
                print("🔇 SILENT PROCESS: Starting silent cap processing...")
                self.gui.cap_progress_bar.setVisible(False)  # Don't show progress bar
                
                # Start processing thread
                from core.business_logic import CapDetectionThread
                # ส่ง bottle_type ไปยัง CapDetectionThread เพื่อใช้ค่าที่เหมาะสมในการตรวจจับ fade
                bottle_type = getattr(self.gui, 'current_bottle_type', None)
                print(f"🏷️ CAP AUTO CAPTURE: ส่ง bottle_type = {bottle_type} ไปยัง CapDetectionThread")
                
                self.gui.cap_processing_thread = CapDetectionThread(
                    captured_image,
                    self.gui.cap_detector,
                    self.gui.craft_detector,
                    self.gui.rotation_model,
                    self.gui.line_detector,
                    self.gui.ocr_model,
                    getattr(self.gui, 'faded_text_yolo_model', None),
                    bottle_type=bottle_type  # ส่ง bottle_type เพื่อใช้ค่าที่เหมาะสม
                )
                self.gui.cap_processing_thread.result_ready.connect(self.on_cap_processing_complete)
                self.gui.cap_processing_thread.start()
                print("🔇 SILENT PROCESS: Cap processing thread started")
            else:
                print("❌ SILENT CAPTURE: Failed to capture Sentech image")
                self.silent_mode = False
                
        except Exception as e:
            print(f"❌ SILENT CAPTURE ERROR: {str(e)}")
            self.silent_mode = False
    
    def display_result_from_queue(self, result):
        """Display result from queue (after D6004=200)"""
        try:
            print("📋 DISPLAY FROM QUEUE: Displaying cap result from queue")
            
            # Store result
            self.gui.current_cap_result = result
            
            # อัปเดตภาพฝาบนแท็บฝาและหน้าหลักให้ตรงกับผลจากคิว
            cap_img = result.get('image') or result.get('processed_image')
            if cap_img is not None:
                self.gui.current_sentech_image = cap_img
                self.display_sentech_image(cap_img)
            
            # Display cap detection results
            self.display_cap_detection_results(result)
            
            # Update status tab
            self.gui.status_handlers.update_cap_detection_status("เสร็จสิ้น", False)
            
            # Update status
            if self.gui.current_bottle_type not in ["M100", "M110", "M120"]:
                self.gui.status_label.setText('✅ แสดงผลลัพธ์ฝาจากคิว')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            print("✅ DISPLAY FROM QUEUE: Cap result displayed")
            
        except Exception as e:
            print(f"❌ DISPLAY FROM QUEUE ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_sentech_image(self):
        """Save the currently captured/loaded Sentech image to disk"""
        if self.gui.current_sentech_image is None:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "ไม่มีภาพฝาให้บันทึก\nกรุณาถ่ายภาพจาก Sentech หรือเลือกไฟล์ภาพก่อน")
            return
        
        try:
            # Create default folder for manual captures
            default_folder = "captured_images"
            manual_folder = os.path.join(default_folder, "manual_capture")
            sentech_folder = os.path.join(manual_folder, "sentech")
            os.makedirs(sentech_folder, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"sentech_cap_{timestamp}.png"
            default_path = os.path.join(sentech_folder, default_filename)
            
            # Ask user where to save (with default path)
            file_path, _ = QFileDialog.getSaveFileName(
                self.gui,
                "บันทึกรูปภาพฝา",
                default_path,
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
            )
            
            if file_path:
                # Handle grayscale images (Mono8 from Sentech camera)
                image_to_save = self.gui.current_sentech_image
                if len(image_to_save.shape) == 2:
                    # Grayscale image - save as is or convert to BGR for JPEG
                    if file_path.lower().endswith(('.jpg', '.jpeg')):
                        # Convert grayscale to BGR for JPEG
                        image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_GRAY2BGR)
                elif len(image_to_save.shape) == 3:
                    # Color image - ensure it's BGR for OpenCV
                    if image_to_save.shape[2] == 3:
                        # Already BGR or RGB, assume BGR
                        pass
                
                # Save image
                success = cv2.imwrite(file_path, image_to_save)
                if success:
                    self.gui.status_label.setText(f'✅ บันทึกรูปภาพฝาสำเร็จ: {os.path.basename(file_path)}')
                    self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                    print(f"💾 บันทึกรูปภาพฝาสำเร็จ: {file_path}")
                    QMessageBox.information(self.gui, "สำเร็จ", f"บันทึกรูปภาพฝาสำเร็จ\n{file_path}")
                else:
                    self.gui.status_label.setText('❌ ไม่สามารถบันทึกรูปภาพฝาได้')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    QMessageBox.warning(self.gui, "ข้อผิดพลาด", "ไม่สามารถบันทึกรูปภาพฝาได้")
                    print(f"❌ ไม่สามารถบันทึกรูปภาพฝาได้: {file_path}")
        except Exception as e:
            error_msg = f"เกิดข้อผิดพลาดในการบันทึกรูปภาพฝา: {str(e)}"
            self.gui.status_label.setText(f'❌ {error_msg}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", error_msg)
            print(f"❌ SAVE SENTECH IMAGE ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_sentech_image(self, image):
        """Display Sentech image in label"""
        try:
            if image is not None:
                print(f"🔍 Display image shape: {image.shape}, dtype: {image.dtype}")
                print(f"🔍 Display image min: {image.min()}, max: {image.max()}")
                
                # Handle grayscale image (Mono8 from SENTECH camera)
                if len(image.shape) == 2:
                    # Grayscale image - convert to RGB
                    rgb_image = cuda_cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif len(image.shape) == 3:
                    # Color image - convert BGR to RGB
                    rgb_image = cuda_cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # Unknown format - use as is
                    rgb_image = image
                
                h, w, c = rgb_image.shape
                bytes_per_line = c * w
                
                qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                
                # Scale pixmap to fit label while maintaining aspect ratio
                label_size = self.gui.sentech_image_label.size()
                # Calculate scale factor to fit the image in the label
                scale_w = label_size.width() / pixmap.width()
                scale_h = label_size.height() / pixmap.height()
                scale = min(scale_w, scale_h)  # Use smaller scale to fit completely
                
                scaled_pixmap = pixmap.scaled(
                    int(pixmap.width() * scale), 
                    int(pixmap.height() * scale), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.gui.sentech_image_label.setPixmap(scaled_pixmap)
                
                # Update Home tab cap image too
                if hasattr(self.gui, 'home_cap_image_label'):
                    label_size = self.gui.home_cap_image_label.size()
                    if label_size.width() > 0 and label_size.height() > 0:
                        scale_w = label_size.width() / pixmap.width()
                        scale_h = label_size.height() / pixmap.height()
                        scale = min(scale_w, scale_h)
                        home_scaled_pixmap = pixmap.scaled(
                            int(pixmap.width() * scale), 
                            int(pixmap.height() * scale), 
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        )
                        self.gui.home_cap_image_label.setPixmap(home_scaled_pixmap)
                
                print(f"✅ Image displayed successfully: {rgb_image.shape} -> scaled to {scaled_pixmap.size().width()}x{scaled_pixmap.size().height()} (full image visible)")
            else:
                self.gui.sentech_image_label.setText("ไม่สามารถโหลดภาพได้")

        except Exception as e:
            print(f"❌ Error displaying image: {e}")
            self.gui.sentech_image_label.setText(f"ข้อผิดพลาด: {str(e)}")
    
    def process_cap_detection(self):
        """Process cap detection on Sentech image or selected image file
        ใช้ค่ากลางที่วิเคราะห์ได้ (brightness: 33) โดยตรง ไม่ต้องรอ bottle_type"""
        # Check for image from either Sentech camera or file selection
        image_to_process = None
        image_source = ""
        
        if self.gui.current_sentech_image is not None:
            image_to_process = self.gui.current_sentech_image
            image_source = "Sentech camera"
        elif hasattr(self.gui, 'current_cap_image') and self.gui.current_cap_image is not None:
            image_to_process = self.gui.current_cap_image
            image_source = "selected file"
        else:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "ไม่มีภาพให้ประมวลผล\nกรุณาถ่ายภาพจากกล้อง Sentech หรือเลือกไฟล์ภาพ")
            return
        
        # Update status tab
        self.gui.status_handlers.update_cap_detection_status("กำลังประมวลผล...", True)
        
        if not all([self.gui.cap_detector, self.gui.craft_detector, self.gui.rotation_model, self.gui.line_detector, self.gui.ocr_model]):
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "โมเดลตรวจจับฝายังไม่ได้โหลดเสร็จ")
            return
        
        print(f"🔄 CAP PROCESS: Starting cap detection for image shape: {image_to_process.shape} (from {image_source})")
        print(f"🔍 CAP PROCESS: ใช้ค่ากลางที่วิเคราะห์ได้ (brightness: 33) สำหรับการปรับภาพ")
        
        # แสดงผล GUI สำหรับการประมวลผลฝา
        self.display_cap_processing_ui()
        
        # แสดง loading ระดับทั้งแอป (มองเห็นทุกแท็บ)
        self.gui.show_global_loading("กำลังประมวลผลฝา...")
        
        # Start processing thread
        self.gui.cap_progress_bar.setVisible(True)
        self.gui.cap_progress_bar.setValue(0)
        self.gui.cap_progress_bar.setFormat("กำลังเริ่มต้น...")
        self.gui.status_label.setText('🔄 กำลังประมวลผลฝา...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        
        # Disable process button and enable stop button
        self.gui.btn_process_cap.setEnabled(False)
        self.gui.btn_save_sentech_image.setEnabled(False)  # Disable save button when processing starts
        self.gui.btn_stop_processing.setEnabled(True)
        
        self.gui.cap_processing_thread = CapDetectionThread(
            image_to_process,
            self.gui.cap_detector,
            self.gui.craft_detector,
            self.gui.rotation_model,
            self.gui.line_detector,
            self.gui.ocr_model,
            getattr(self.gui, 'faded_text_yolo_model', None),
            bottle_type=None  # ไม่ใช้แล้ว - ใช้ค่ากลางที่วิเคราะห์ได้โดยตรง
        )
        self.gui.cap_processing_thread.result_ready.connect(self.on_cap_processing_complete)
        self.gui.cap_processing_thread.status_updated.connect(self.update_cap_progress)
        self.gui.cap_processing_thread.progress_updated.connect(self.update_cap_progress_bar)
        print("🔄 CAP PROCESS: Starting cap detection thread...")
        self.gui.cap_processing_thread.start()
    
    def update_cap_progress(self, status):
        """Update cap detection progress bar and status"""
        if "ตรวจจับฝา" in status:
            self.gui.cap_progress_bar.setValue(20)
            self.gui.cap_progress_bar.setFormat("Step 1: ตรวจจับฝา (20%)")
        elif "ตัดภาพ" in status:
            self.gui.cap_progress_bar.setValue(40)
            self.gui.cap_progress_bar.setFormat("Step 2: ตัดภาพ (40%)")
        elif "CRAFT" in status:
            self.gui.cap_progress_bar.setValue(50)
            self.gui.cap_progress_bar.setFormat("Step 2a: CRAFT (50%)")
        elif "AI rotation" in status:
            self.gui.cap_progress_bar.setValue(60)
            self.gui.cap_progress_bar.setFormat("Step 2b: AI Rotation (60%)")
        elif "บรรทัดข้อความ" in status:
            self.gui.cap_progress_bar.setValue(80)
            self.gui.cap_progress_bar.setFormat("Step 3: ตรวจจับบรรทัด (80%)")
        elif "OCR" in status:
            self.gui.cap_progress_bar.setValue(90)
            self.gui.cap_progress_bar.setFormat("Step 4: OCR (90%)")
        
        self.gui.status_label.setText(status)
    
    def update_cap_progress_bar(self, value):
        """Update cap detection progress bar with specific value"""
        self.gui.cap_progress_bar.setValue(value)
        self.gui.cap_progress_bar.setFormat(f"กำลังประมวลผล... ({value}%)")
    
    def display_cap_processing_ui(self):
        """Display cap processing UI when starting cap detection"""
        try:
            # Clear previous results
            if hasattr(self.gui, 'cap_results_layout') and self.gui.cap_results_layout:
                for i in reversed(range(self.gui.cap_results_layout.count())):
                    widget = self.gui.cap_results_layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)
            
            # Create processing container
            processing_container = QWidget()
            processing_container.setStyleSheet("border: 2px solid #f39c12; margin: 5px; padding: 10px; background-color: #fef9e7; border-radius: 5px;")
            processing_layout = QVBoxLayout(processing_container)
            
            # Processing title
            processing_title = QLabel("🔄 กำลังประมวลผลฝา...")
            processing_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f39c12; margin: 10px;")
            processing_title.setAlignment(Qt.AlignCenter)
            processing_layout.addWidget(processing_title)
            
            # Processing steps
            steps_info = QLabel("📋 ขั้นตอนการประมวลผล:\n1. ตรวจจับฝาด้วย YOLO\n2. ใช้ CRAFT หมุนภาพ\n3. ใช้ AI Rotation Model\n4. ตรวจจับบรรทัดข้อความ\n5. อ่านข้อความด้วย OCR\n6. ตรวจสอบรูปแบบ MFG")
            steps_info.setStyleSheet("font-size: 12px; color: #2c3e50; margin: 5px; padding: 10px; background-color: #ecf0f1; border-radius: 3px;")
            steps_info.setAlignment(Qt.AlignLeft)
            processing_layout.addWidget(steps_info)
            
            # Progress info
            progress_info = QLabel("⏳ กรุณารอสักครู่... ระบบกำลังประมวลผล")
            progress_info.setStyleSheet("font-size: 11px; color: #7f8c8d; margin: 5px; font-style: italic;")
            progress_info.setAlignment(Qt.AlignCenter)
            processing_layout.addWidget(progress_info)
            
            if hasattr(self.gui, 'cap_results_layout') and self.gui.cap_results_layout:
                self.gui.cap_results_layout.addWidget(processing_container)
            
            # Update cap detection text
            if hasattr(self.gui, 'cap_detection_text') and self.gui.cap_detection_text:
                self.gui.cap_detection_text.setText("🔄 กำลังประมวลผลฝา... กรุณารอสักครู่")
                if hasattr(self.gui, 'home_cap_detection_text'):
                    self.gui.home_cap_detection_text.setText("🔄 กำลังประมวลผลฝา... กรุณารอสักครู่")
                self.gui.cap_detection_text.setStyleSheet("""
                    QTextEdit {
                        background-color: #fef9e7;
                        border: 2px solid #f39c12;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 12px;
                        color: #2c3e50;
                    }
                """)
            
            print("✅ CAP PROCESSING UI: Displayed processing state for cap detection")
            
        except Exception as e:
            print(f"❌ Error displaying cap processing UI: {e}")
    
    def display_cap_processing_complete_ui(self):
        """Display cap processing complete UI when cap detection is finished"""
        try:
            # Create complete container
            complete_container = QWidget()
            complete_container.setStyleSheet("border: 2px solid #27ae60; margin: 5px; padding: 10px; background-color: #d5f4e6; border-radius: 5px;")
            complete_layout = QVBoxLayout(complete_container)
            
            # Complete title
            complete_title = QLabel("✅ การประมวลผลฝาเสร็จสิ้น")
            complete_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60; margin: 10px;")
            complete_title.setAlignment(Qt.AlignCenter)
            complete_layout.addWidget(complete_title)
            
            # Status info
            if self.gui.current_bottle_type in ["M100", "M110", "M120"]:
                status_info = QLabel(f"📋 ฝาผ่านการตรวจสอบ - กำลังส่งสัญญาณ {self.gui.current_bottle_type}")
                status_info.setStyleSheet("font-size: 14px; color: #27ae60; margin: 5px; font-weight: bold;")
            else:
                status_info = QLabel("📋 การประมวลผลฝาเสร็จสิ้น")
                status_info.setStyleSheet("font-size: 14px; color: #27ae60; margin: 5px; font-weight: bold;")
            status_info.setAlignment(Qt.AlignCenter)
            complete_layout.addWidget(status_info)
            
            # System status
            system_status = QLabel("✅ ระบบพร้อมส่งสัญญาณ Modbus\n✅ การประมวลผลเสร็จสิ้นแล้ว\n✅ GUI แสดงผลครบถ้วน")
            system_status.setStyleSheet("font-size: 12px; color: #27ae60; margin: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 3px;")
            system_status.setAlignment(Qt.AlignCenter)
            complete_layout.addWidget(system_status)
            
            if hasattr(self.gui, 'cap_results_layout') and self.gui.cap_results_layout:
                self.gui.cap_results_layout.addWidget(complete_container)
            
            # Update cap detection text
            if hasattr(self.gui, 'cap_detection_text') and self.gui.cap_detection_text:
                if self.gui.current_bottle_type in ["M100", "M110", "M120"]:
                    self.gui.cap_detection_text.setText(f"✅ การประมวลผลฝาเสร็จสิ้น - กำลังส่งสัญญาณ {self.gui.current_bottle_type}")
                    if hasattr(self.gui, 'home_cap_detection_text'):
                        self.gui.home_cap_detection_text.setText(f"✅ การประมวลผลฝาเสร็จสิ้น - กำลังส่งสัญญาณ {self.gui.current_bottle_type}")
                else:
                    self.gui.cap_detection_text.setText("✅ การประมวลผลฝาเสร็จสิ้น")
                    if hasattr(self.gui, 'home_cap_detection_text'):
                        self.gui.home_cap_detection_text.setText("✅ การประมวลผลฝาเสร็จสิ้น")
                self.gui.cap_detection_text.setStyleSheet("""
                    QTextEdit {
                        background-color: #d5f4e6;
                        border: 2px solid #27ae60;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 12px;
                        color: #2c3e50;
                    }
                """)
            
            print("✅ CAP PROCESSING COMPLETE UI: Displayed complete state for cap detection")
            
        except Exception as e:
            print(f"❌ Error displaying cap processing complete UI: {e}")
    
    def on_cap_processing_complete(self, result):
        """Handle cap detection processing completion"""
        print("✅ CAP PROCESS COMPLETE: Cap detection finished")
        self.gui.cap_progress_bar.setVisible(False)
        self.gui.hide_global_loading()
        
        # Update performance stats
        self.gui.total_images_processed_count += 1
        self.gui.status_handlers.update_performance_stats(images_processed=self.gui.total_images_processed_count)
        
        # Reset button states
        self.gui.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
        self.gui.btn_save_sentech_image.setEnabled(True)  # Enable save button when processing completes
        self.gui.btn_stop_processing.setEnabled(False)
        
        # Check if silent mode - store result in queue instead of displaying
        if self.silent_mode:
            print("🔇 SILENT MODE: Storing cap result in queue (no UI display, no Modbus signals)")
            self.pending_cap_result = result
            
            # Check if bottle result is also ready
            if hasattr(self.gui, 'bottle_handlers') and self.gui.bottle_handlers.pending_bottle_result is not None:
                # Both results are ready - add to queue หรือแสดงทันทีถ้ามีคิวนับอยู่
                bottle_result = self.gui.bottle_handlers.pending_bottle_result
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    mt = self.gui.modbus_thread
                    if mt.pending_m301_count > 0:
                        mt.pending_m301_count -= 1
                        print(f"🔇 SILENT MODE: ผลพร้อม — แสดงจากคิวทันที (ลดคิวเหลือ: {mt.pending_m301_count})")
                        mt.result_ready_to_display.emit(bottle_result, result)
                    else:
                        mt.pending_results_queue.append((bottle_result, result))
                        print(f"🔇 SILENT MODE: Results added to queue (queue size: {len(mt.pending_results_queue)})")
                    self.pending_cap_result = None
                    self.gui.bottle_handlers.pending_bottle_result = None
                else:
                    print("⚠️ SILENT MODE: Modbus thread not available")
            else:
                # Wait for bottle result
                print("🔇 SILENT MODE: Cap result stored, waiting for bottle result...")
            
            # Reset silent mode
            self.silent_mode = False
            return
        
        # ตรวจสอบว่าเป็นข้อความจางหรือไม่ - แสดงผลลัพธ์ก่อน (ภาพและค่า area) และส่ง Modbus เหมือนขวด NG
        if "error" in result and result["error"] == "faded_text_detected":
            print("❌ CAP PROCESS COMPLETE: ตรวจพบข้อความจาง - แสดงผลลัพธ์ (ภาพและค่า area)")
            
            # เก็บผลลัพธ์ไว้
            self.gui.current_cap_result = result
            
            # ส่ง Modbus เหมือนขวด NG (D7009=50, M140, M600)
            if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                m140_ok = self.gui.modbus_thread.on_m140()
                if m140_ok:
                    self.gui.status_handlers.update_coil_lamp("m140", True)
                    m600_ok = self.gui.modbus_thread.on_m600()
                    if m600_ok:
                        self.gui.status_handlers.update_coil_lamp("m600", True)
                        self.gui.status_label.setText('❌ ฝาจาง → D7009=50, M140, M600 ส่งแล้ว')
                    else:
                        self.gui.status_label.setText('❌ ฝาจาง แต่ไม่สามารถ ON M600 ได้')
                else:
                    self.gui.status_label.setText('❌ ฝาจาง แต่ไม่สามารถส่ง D7009=50 ได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            else:
                self.gui.status_label.setText('❌ ฝาจาง แต่ Modbus ไม่พร้อม')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            
            # Update status tab
            self.gui.status_handlers.update_cap_detection_status("ข้อความจาง - แสดงผลลัพธ์", False)
            
            # Display cap detection results (แสดงผลลัพธ์ฝาที่เจอข้อความจาง - รวมภาพและค่า area)
            self.display_cap_detection_results(result)
            
            # แสดงค่า area และข้อมูลอื่นๆ
            faded_text_result = result.get('faded_text_result', {})
            total_area = faded_text_result.get('total_area', 0)
            num_chars = faded_text_result.get('num_chars', 0)
            status = faded_text_result.get('status', 'unknown')
            
            if not (hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread):
                self.gui.status_label.setText(f'❌ ตรวจพบข้อความจาง - total_area: {total_area}, num_chars: {num_chars}')
            
            print(f"📊 CAP PROCESS COMPLETE: แสดงผลลัพธ์ข้อความจาง - total_area: {total_area}, num_chars: {num_chars}, status: {status}")
            
            # บันทึกลงประวัติแม้ว่าจะเป็น faded text (เพราะเป็นผลลัพธ์ที่ถูกต้อง)
            if hasattr(self.gui, 'add_to_history'):
                # ตรวจสอบว่ามี bottle result หรือไม่
                if hasattr(self.gui, 'current_result') and self.gui.current_result:
                    bottle_result = self.gui.current_result
                else:
                    # สร้าง dummy bottle result สำหรับกรณี batch processing หรือไม่มี bottle result
                    bottle_result = {
                        'image': None,
                        'bottle_type': 'Faded Text Detected',
                        'combined_ocr_text': f'Faded Text - total_area: {total_area}, num_chars: {num_chars}'
                    }
                
                # บันทึกลงประวัติ
                print("📋 CAP PROCESS COMPLETE: Adding faded text result to history")
                self.gui.add_to_history(bottle_result, result)
            
            return
        
        if "error" not in result:
            print("✅ CAP PROCESS COMPLETE: No error in result")
            print(f"✅ CAP PROCESS COMPLETE: Result keys: {list(result.keys())}")
            
            # Store result
            self.gui.current_cap_result = result
            
            # ไม่ validate ที่นี่ - รอให้การประมวลผลเสร็จสิ้นก่อน
            print("ℹ️ CAP PROCESS COMPLETE: Skipping early validation - will validate after processing is complete")
            
            # Update status tab
            self.gui.status_handlers.update_cap_detection_status("เสร็จสิ้น", False)
            
            # Display cap detection results (แสดงผลลัพธ์ฝาทุกกรณี)
            self.display_cap_detection_results(result)
            
            # Update status
            if self.gui.current_bottle_type not in ["M100", "M110", "M120"]:
                self.gui.status_label.setText('✅ ประมวลผลฝาเสร็จสิ้น')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            # แสดงผล GUI ที่ชัดเจนว่าการประมวลผลเสร็จสิ้นแล้ว
            self.display_cap_processing_complete_ui()
            
            # ON M100/M110/M120 ตาม bottle type ทันทีหลังจากตรวจฝาเสร็จ (ไม่ต้องกด OK)
            # แต่ต้อง validate ก่อนว่า cap result ถูกต้องหรือไม่
            if self.gui.current_bottle_type in ["M100", "M110", "M120"]:
                print(f"🔍 CAP PROCESS COMPLETE: Bottle type is {self.gui.current_bottle_type} - Validating cap result...")
                print(f"🔍 CAP PROCESS COMPLETE: Result keys: {list(result.keys())}")
                
                # Validate cap result ก่อนส่ง Modbus signal
                is_valid = self.validate_cap_result(result)
                print(f"🔍 CAP PROCESS COMPLETE: Validation result: {is_valid}")
                
                if is_valid:
                    # Good = ฝาผ่าน + ขวดผ่าน เท่านั้น ถึงส่ง Modbus ปกติ (D7009=10/20/30)
                    bottle_result = getattr(self.gui, 'current_result', None) or getattr(self.gui, 'last_bottle_result', None)
                    bottle_defect_ng = False
                    if bottle_result and bottle_result.get('defect_inspection'):
                        if bottle_result['defect_inspection'].get('result') == 'NG':
                            bottle_defect_ng = True
                    if bottle_defect_ng:
                        print(f"❌ ขวดไม่ผ่าน (Defect NG) - ส่ง Modbus NG แทน Good")
                        if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                            m140_ok = self.gui.modbus_thread.on_m140()
                            if m140_ok:
                                self.gui.status_handlers.update_coil_lamp("m140", True)
                                m600_ok = self.gui.modbus_thread.on_m600()
                                if m600_ok:
                                    self.gui.status_handlers.update_coil_lamp("m600", True)
                                self.gui.status_label.setText('❌ ขวดไม่ผ่าน (Defect NG) → D7009=50, M140, M600')
                                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    else:
                        print(f"✅ CAP VALIDATION PASSED: ฝาผ่าน + ขวดผ่าน - Sending Modbus signals")
                        print(f"✅ CAP PROCESS COMPLETE: Calling on_bottle_type_after_cap_validation()")
                        if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                            print(f"✅ CAP PROCESS COMPLETE: Modbus thread is available - Sending signals")
                            self.on_bottle_type_after_cap_validation()
                        else:
                            print(f"❌ CAP PROCESS COMPLETE: Modbus thread is not available!")
                            self.gui.status_label.setText(f'❌ Modbus thread ไม่พร้อม - ไม่สามารถส่งสัญญาณ {self.gui.current_bottle_type} ได้')
                            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                else:
                    print(f"❌ CAP VALIDATION FAILED: Cap result is invalid for {self.gui.current_bottle_type} - Sending M140 instead")
                    # ON M140 (ฝาไม่ผ่าน - ไม่ตรงกับฟอร์ม)
                    if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                        m140_success = self.gui.modbus_thread.on_m140()
                        if m140_success:
                            self.gui.status_handlers.update_coil_lamp("m140", True)
                            print("❌ ON M140 (ฝาไม่ผ่าน - ไม่ตรงกับฟอร์ม)")
                            
                            # ON M600 หลังจาก ON M140 (เหมือน M100/M110/M120)
                            print("⏳ ON M600 และรอ D6004=200 เพื่อ reset ทั้งหมด...")
                            m600_success = self.gui.modbus_thread.on_m600()
                            if m600_success:
                                self.gui.status_handlers.update_coil_lamp("m600", True)
                            
                            # อัปเดตสถานะ
                            self.gui.status_label.setText('❌ ฝาไม่ผ่าน → ON M140 + M600 (รอ reset)')
                            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                            
                            print(f"✅ M140 ON SUCCESS: M140 + M600")
                        else:
                            print("❌ ไม่สามารถ ON M140 ได้")
                            self.gui.status_label.setText('❌ ไม่สามารถ ON M140 ได้')
                            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    else:
                        print(f"❌ CAP PROCESS COMPLETE: Modbus thread is not available for M140!")
            else:
                print(f"ℹ️ CAP PROCESS COMPLETE: Bottle type is {self.gui.current_bottle_type} - Not M100/M110/M120, skipping Modbus signals")
            
            # Update history with cap result if bottle result already exists
            # Check if there's a recent bottle result that needs cap result
            if hasattr(self.gui, 'add_to_history') and hasattr(self.gui, 'current_result') and self.gui.current_result:
                # Update the last history entry with cap result if it matches
                # For now, just add a new entry with both results
                bottle_result = self.gui.current_result
                if hasattr(self.gui, 'add_to_history'):
                    # Add to history with both bottle and cap results
                    print("📋 CAP PROCESS COMPLETE: Adding to history with cap result")
                    self.gui.add_to_history(bottle_result, result)
        else:
            print(f"❌ CAP PROCESS COMPLETE: Error in result: {result['error']}")
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {result["error"]}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            
            # Show error message
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการประมวลผลฝา: {result['error']}")
    
    # Note: display_cap_detection_results, validate_cap_result, validate_date_time_format,
    # extract_expiry_date_from_text, check_expiry_date_filter, and on_bottle_type_after_cap_validation
    # are very large methods. They will be added in a separate commit or can be extracted
    # to this file later. For now, these methods will remain in main_window.py and be
    # called through the gui instance.
    
    # Placeholder methods that delegate to main_window for now:
    def display_cap_detection_results(self, result):
        """Display cap detection results - delegates to main_window for now"""
        # This method is very large (~900 lines), so we'll keep it in main_window for now
        # and call it through the gui instance
        if hasattr(self.gui, 'display_cap_detection_results'):
            self.gui.display_cap_detection_results(result)
        else:
            print("⚠️ display_cap_detection_results not found in gui")
    
    def validate_cap_result(self, cap_result):
        """Validate cap result - delegates to main_window for now"""
        if hasattr(self.gui, 'validate_cap_result'):
            return self.gui.validate_cap_result(cap_result)
        else:
            print("⚠️ validate_cap_result not found in gui")
            return False
    
    def validate_date_time_format(self, ocr_text):
        """Validate date/time format - delegates to main_window for now"""
        if hasattr(self.gui, 'validate_date_time_format'):
            return self.gui.validate_date_time_format(ocr_text)
        else:
            print("⚠️ validate_date_time_format not found in gui")
            return False
    
    def extract_expiry_date_from_text(self, ocr_text):
        """Extract expiry date from text - delegates to main_window for now"""
        if hasattr(self.gui, 'extract_expiry_date_from_text'):
            return self.gui.extract_expiry_date_from_text(ocr_text)
        else:
            print("⚠️ extract_expiry_date_from_text not found in gui")
            return None
    
    def check_expiry_date_filter(self, expiry_date):
        """Check expiry date filter - delegates to main_window for now"""
        if hasattr(self.gui, 'check_expiry_date_filter'):
            return self.gui.check_expiry_date_filter(expiry_date)
        else:
            print("⚠️ check_expiry_date_filter not found in gui")
            return False
    
    def on_bottle_type_after_cap_validation(self):
        """ON M100/M110/M120 after cap validation - delegates to modbus_handlers"""
        if hasattr(self.gui, 'modbus_handlers') and self.gui.modbus_handlers:
            self.gui.modbus_handlers.on_bottle_type_after_cap_validation()
        else:
            print("⚠️ modbus_handlers not found in gui - cannot call on_bottle_type_after_cap_validation")


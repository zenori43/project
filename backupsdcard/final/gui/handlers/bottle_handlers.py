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
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np
import datetime
import os
import tempfile

# Import CUDA image utilities
from core.cuda_image_utils import cuda_resize, cuda_cvtColor


class CaptureBothThread(QThread):
    """Worker thread สำหรับถ่ายภาพ USB + Sentech ไม่บล็อก GUI"""
    capture_done = pyqtSignal(object, object)  # usb_image, sentech_image (อาจเป็น None)

    def __init__(self, usb_camera, sentech_camera, delay_before_sec=0, parent=None):
        super().__init__(parent)
        self.usb_camera = usb_camera
        self.sentech_camera = sentech_camera
        self.delay_before_sec = delay_before_sec

    def run(self):
        if self.delay_before_sec and self.delay_before_sec > 0:
            QThread.msleep(int(self.delay_before_sec * 1000))
        usb_image = None
        sentech_image = None
        if self.usb_camera is not None:
            print("📸 CAPTURE BOTH: กำลังถ่ายภาพจาก USB...")
            for attempt in range(3):
                print(f"📸 CAPTURE BOTH USB: Attempt {attempt + 1}/3")
                usb_image = self.usb_camera.capture_image()
                if usb_image is not None:
                    print(f"✅ CAPTURE BOTH USB: Success on attempt {attempt + 1}")
                    break
                QThread.msleep(100)
        if self.sentech_camera is not None:
            print("📸 CAPTURE BOTH: กำลังถ่ายภาพจาก Sentech...")
            try:
                sentech_image = self.sentech_camera.capture_image()
                print(f"📸 CAPTURE BOTH Sentech: Result = {sentech_image is not None}")
            except Exception as e:
                print(f"❌ CAPTURE BOTH Sentech Error: {e}")
        self.capture_done.emit(usb_image, sentech_image)


class BottleBatchWorker(QThread):
    """รัน batch ขวดใน thread แยก ไม่บล็อก GUI และส่ง progress ให้ progress bar อัปเดต"""
    progress_updated = pyqtSignal(int, int)  # current (1-based), total
    image_done = pyqtSignal(object, str, int, bool)  # result, file_path, idx, success
    finished_with_stats = pyqtSignal(int, int, int, int)  # success_count, error_count, good_count, ng_count
    error_occurred = pyqtSignal(str)

    def __init__(self, file_paths, handler):
        super().__init__()
        self.file_paths = file_paths
        self.handler = handler
        # ใช้ selected_tastes จาก GUI
        self.selected_tastes = getattr(handler.gui, 'selected_tastes', ["M100", "M110", "M120"])

    def run(self):
        from libs.detection.bottledetect import process_bottle_image_simple
        from core.image_processor import (
            perform_ocr_on_image,
            check_bottle_type,
            aggregate_easyocr_confidences_from_type_crops,
        )
        import cv2
        
        success_count = 0
        error_count = 0
        good_count = 0
        ng_count = 0
        total = len(self.file_paths)
        for idx, file_path in enumerate(self.file_paths, 1):
            self.progress_updated.emit(idx - 1, total)
            try:
                # โหลดภาพ
                image = cv2.imread(file_path)
                if image is None:
                    print(f"❌ BATCH BOTTLE [{idx}]: Cannot load image: {file_path}")
                    error_count += 1
                    self.progress_updated.emit(idx, total)
                    continue
                
                # ใช้ process_bottle_image_simple (รับ file_path)
                result = process_bottle_image_simple(file_path)
                
                if result and "error" not in result:
                    # เพิ่ม image info
                    result['image'] = image
                    result['image_shape'] = image.shape
                    
                    # ตรวจสอบ defect inspection (Good/NG)
                    defect_inspection = result.get('defect_inspection')
                    if defect_inspection:
                        defect_result = defect_inspection.get('result', '')
                        if defect_result == 'Good':
                            good_count += 1
                        elif defect_result == 'NG':
                            ng_count += 1
                    
                    # ทำ OCR บน type crops (เหมือน BottleDetectionThread)
                    combined_ocr_text = ""
                    type_crops = result.get('type_crops')
                    if type_crops and isinstance(type_crops, list) and len(type_crops) > 0:
                        print(f"🔄 BATCH BOTTLE [{idx}]: Found {len(type_crops)} type crops, performing OCR...")
                        for i, crop in enumerate(type_crops):
                            if not isinstance(crop, dict) or 'original_crop' not in crop:
                                continue
                            
                            # ทำ OCR
                            ocr_results = perform_ocr_on_image(crop['original_crop'])
                            crop['ocr_results'] = ocr_results
                            
                            # รวมข้อความที่อ่านได้
                            if ocr_results and isinstance(ocr_results, list):
                                for ocr_result in ocr_results:
                                    if isinstance(ocr_result, dict) and 'text' in ocr_result:
                                        combined_ocr_text += ocr_result['text'] + " "
                        
                        # ลบช่องว่างที่เกิน
                        combined_ocr_text = combined_ocr_text.strip()
                        result.update(aggregate_easyocr_confidences_from_type_crops(type_crops))
                        
                        # ตรวจสอบ bottle type
                        bottle_type = check_bottle_type(combined_ocr_text, self.selected_tastes)
                        result['combined_ocr_text'] = combined_ocr_text
                        result['bottle_type'] = bottle_type
                        
                        print(f"📝 BATCH BOTTLE [{idx}]: OCR Text: '{combined_ocr_text}', Type: {bottle_type}")
                    else:
                        print(f"⚠️ BATCH BOTTLE [{idx}]: No type crops found")
                        result['combined_ocr_text'] = ""
                        result['bottle_type'] = None
                    
                    success_count += 1
                    self.image_done.emit(result, file_path, idx, True)
                else:
                    error_count += 1
                    print(f"❌ BATCH BOTTLE [{idx}]: Error in result: {result.get('error', 'Unknown error')}")
            except Exception as e:
                error_count += 1
                print(f"❌ BATCH BOTTLE [{idx}]: Error processing {os.path.basename(file_path)}: {e}")
                import traceback
                traceback.print_exc()
            self.progress_updated.emit(idx, total)
        self.finished_with_stats.emit(success_count, error_count, good_count, ng_count)


class CaptureSentechOnlyThread(QThread):
    """Worker thread สำหรับถ่ายภาพ Sentech อย่างเดียว (ใช้ตอน retry ภาพฝา)"""
    capture_done = pyqtSignal(object)  # sentech_image or None

    def __init__(self, sentech_camera, parent=None):
        super().__init__(parent)
        self.sentech_camera = sentech_camera

    def run(self):
        img = None
        if self.sentech_camera is not None:
            try:
                img = self.sentech_camera.capture_image()
            except Exception as e:
                print(f"❌ CAPTURE SENTECH ONLY Error: {e}")
        self.capture_done.emit(img)


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
        self.pending_bottle_result = None  # Store result for silent processing (ขวดเสร็จก่อน รอฝา)
        self._capture_both_thread = None  # เก็บ reference เพื่อไม่ให้ถูก gc
        self._bottle_batch_worker = None  # เก็บ reference เพื่อไม่ให้ถูก gc
        self._silent_cap_retry_count = 0  # นับรอบ retry ภาพฝา (สูงสุด 2 ครั้ง)
        self._capture_sentech_only_thread = None  # สำหรับ retry ถ่ายฝาอย่างเดียว

    def _on_capture_both_done(self, usb_image, sentech_image):
        """อัปเดต GUI หลังถ่ายภาพทั้งสองกล้องเสร็จ (รันบน main thread)"""
        try:
            if usb_image is not None:
                # รูปใหม่มา → เคลียร์รูปครอปในหน้าหลักและแท็บขวด
                self.clear_cropped_images_display()
                self.gui.current_image = usb_image
                self.display_image(usb_image)
                self.gui.image_info_label.setText(f"ขนาด: {usb_image.shape[1]}x{usb_image.shape[0]}")
                if hasattr(self.gui, 'home_bottle_image_info_label'):
                    self.gui.home_bottle_image_info_label.setText(f"ขนาด: {usb_image.shape[1]}x{usb_image.shape[0]}")
                self.gui.btn_process.setEnabled(True)
                self.gui.btn_save_image.setEnabled(True)
                if hasattr(self.gui, 'btn_process_bottle_tab'):
                    self.gui.btn_process_bottle_tab.setEnabled(True)
                if hasattr(self.gui, 'btn_save_bottle_image_tab'):
                    self.gui.btn_save_bottle_image_tab.setEnabled(True)
                print("✅ CAPTURE BOTH: USB image captured and displayed")
            else:
                print("❌ CAPTURE BOTH: ไม่สามารถถ่ายภาพจาก USB ได้")
            if sentech_image is not None:
                self.gui.current_sentech_image = sentech_image
                if hasattr(self.gui, 'cap_handlers') and self.gui.cap_handlers:
                    self.gui.cap_handlers.display_sentech_image(sentech_image)
                self.gui.sentech_image_info_label.setText(f"ขนาด: {sentech_image.shape[1]}x{sentech_image.shape[0]}")
                self.gui.btn_save_sentech_image.setEnabled(True)
                if hasattr(self.gui, 'btn_process_cap'):
                    self.gui.btn_process_cap.setEnabled(True)
                    print("✅ CAPTURE BOTH: เปิดใช้งานปุ่มประมวลผลฝา (มีภาพจาก Sentech)")
                print("✅ CAPTURE BOTH: Sentech image captured and displayed")
            else:
                print("⚠️ CAPTURE BOTH: ไม่สามารถถ่ายภาพจาก Sentech ได้")
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
        finally:
            if hasattr(self.gui, 'btn_capture'):
                self.gui.btn_capture.setEnabled(True)
            if getattr(self.gui, 'btn_capture_both', None):
                self.gui.btn_capture_both.setEnabled(True)
            self._update_process_paired_button_state()
            self._capture_both_thread = None

    def _on_full_auto_capture_done(self, usb_image, sentech_image):
        """อัปเดต GUI หลังถ่ายภาพ full auto (รันบน main thread) แล้วเริ่มประมวลผลขวด"""
        try:
            self._on_capture_both_done(usb_image, sentech_image)
            if usb_image is not None:
                print("🚀 FULL AUTO: ถ่ายภาพเสร็จ - เริ่มประมวลผลขวด (process_current_image)")
                self.process_current_image()
            else:
                print("❌ FULL AUTO: ไม่ได้ภาพจากกล้อง USB - ไม่ประมวลผลขวด")
        finally:
            self._capture_both_thread = None

    def capture_both_cameras(self):
        """ถ่ายภาพจาก USB + Sentech ใน worker thread ไม่บล็อก GUI"""
        if self.gui.usb_camera is None:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "กล้อง USB ยังไม่ได้เริ่มต้น")
            return
        if self._capture_both_thread is not None and self._capture_both_thread.isRunning():
            return
        self.gui.status_label.setText('📸 กำลังถ่ายภาพจาก USB และ Sentech...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        if hasattr(self.gui, 'btn_capture'):
            self.gui.btn_capture.setEnabled(False)
        if getattr(self.gui, 'btn_capture_both', None):
            self.gui.btn_capture_both.setEnabled(False)
        self._capture_both_thread = CaptureBothThread(
            self.gui.usb_camera,
            self.gui.sentech_camera,
            parent=self.gui
        )
        self._capture_both_thread.capture_done.connect(self._on_capture_both_done)
        self._capture_both_thread.start()

    def _on_capture_both_and_save_done(self, usb_image, sentech_image):
        """หลังถ่ายทั้งคู่เสร็จ แสดงผลแล้วบันทึกทั้งสองไฟล์ลงโฟลเดอร์เริ่มต้น (ไม่เปิด dialog)"""
        try:
            self._on_capture_both_done(usb_image, sentech_image)
            if usb_image is not None or sentech_image is not None:
                self.save_both_images_to_default()
        finally:
            if hasattr(self.gui, 'btn_capture_and_save'):
                self.gui.btn_capture_and_save.setEnabled(True)
            self._capture_both_thread = None

    def capture_both_and_save(self):
        """ถ่ายภาพจาก USB + Sentech พร้อมกัน แล้วบันทึกทั้งคู่ลงโฟลเดอร์เริ่มต้น (ปุ่มเดียว)"""
        if self.gui.usb_camera is None:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "กล้อง USB ยังไม่ได้เริ่มต้น")
            return
        if self._capture_both_thread is not None and self._capture_both_thread.isRunning():
            return
        self.gui.status_label.setText('📸 กำลังถ่ายภาพและจะบันทึกทั้งคู่...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        if hasattr(self.gui, 'btn_capture_and_save'):
            self.gui.btn_capture_and_save.setEnabled(False)
        self._capture_both_thread = CaptureBothThread(
            self.gui.usb_camera,
            self.gui.sentech_camera,
            parent=self.gui
        )
        self._capture_both_thread.capture_done.connect(self._on_capture_both_and_save_done)
        self._capture_both_thread.start()

    def save_both_images_to_default(self):
        """บันทึกภาพขวด (USB) และภาพฝา (Sentech) ลงโฟลเดอร์เริ่มต้นด้วย timestamp เดียวกัน (ไม่เปิด dialog)"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_folder = "captured_images"
        manual_folder = os.path.join(default_folder, "manual_capture")
        usb_folder = manual_folder
        sentech_folder = os.path.join(manual_folder, "sentech")
        os.makedirs(sentech_folder, exist_ok=True)
        saved = []
        try:
            if getattr(self.gui, 'current_image', None) is not None:
                usb_path = os.path.join(usb_folder, f"usb_{timestamp}.png")
                if cv2.imwrite(usb_path, self.gui.current_image):
                    saved.append(("USB", usb_path))
            if getattr(self.gui, 'current_sentech_image', None) is not None:
                img = self.gui.current_sentech_image
                if len(img.shape) == 2 and (img.dtype == np.uint8 or img.dtype == np.uint16):
                    sentech_path = os.path.join(sentech_folder, f"sentech_{timestamp}.png")
                else:
                    sentech_path = os.path.join(sentech_folder, f"sentech_{timestamp}.png")
                if len(img.shape) == 2 and sentech_path.lower().endswith(('.jpg', '.jpeg')):
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if cv2.imwrite(sentech_path, img):
                    saved.append(("Sentech", sentech_path))
            if saved:
                names = ", ".join(f"{k}: {os.path.basename(p)}" for k, p in saved)
                self.gui.status_label.setText(f'✅ ถ่ายและบันทึกสำเร็จ: {names}')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                print(f"💾 บันทึกทั้งคู่: {[p for _, p in saved]}")
                QMessageBox.information(self.gui, "สำเร็จ", f"บันทึกสำเร็จ\n{chr(10).join(p for _, p in saved)}")
            else:
                self.gui.status_label.setText('⚠️ บันทึกได้บางส่วน (ไม่มีภาพที่บันทึกได้)')
                self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        except Exception as e:
            self.gui.status_label.setText(f'❌ บันทึกผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            print(f"❌ save_both_images_to_default: {e}")
            import traceback
            traceback.print_exc()

    def start_full_auto_capture(self):
        """เริ่มถ่ายภาพอัตโนมัติใน worker thread (รอ 2 วินาที แล้วถ่าย USB + Sentech) ไม่บล็อก GUI"""
        if self.gui.usb_camera is None:
            print("❌ FULL AUTO: กล้อง USB ยังไม่ได้เริ่มต้น - ไม่สามารถถ่าย/ตรวจจับขวดได้")
            self.gui.status_label.setText('❌ โหมด Full Auto: กล้อง USB ยังไม่ได้เริ่มต้น')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            return
        if self._capture_both_thread is not None and self._capture_both_thread.isRunning():
            print("⚠️ FULL AUTO: กำลังถ่ายภาพอยู่แล้ว - ข้าม M301 นี้")
            return
        self.gui.status_label.setText('📸 กำลังถ่ายภาพอัตโนมัติ (รอ 2 วินาที)...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        self._capture_both_thread = CaptureBothThread(
            self.gui.usb_camera,
            self.gui.sentech_camera,
            delay_before_sec=2,
            parent=self.gui
        )
        self._capture_both_thread.capture_done.connect(self._on_full_auto_capture_done)
        self._capture_both_thread.start()
    
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
    
    def select_multiple_bottle_images(self):
        """Select multiple image files for batch bottle detection (รันใน thread แยก ไม่ให้ GUI ค้าง)"""
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self.gui,
                "เลือกหลายไฟล์ภาพสำหรับการตรวจจับขวด",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            if not file_paths:
                return
            print(f"📁 BATCH BOTTLE PROCESSING: Selected {len(file_paths)} files")
            reply = QMessageBox.question(
                self.gui,
                "ยืนยันการประมวลผลหลายภาพ",
                f"คุณต้องการประมวลผล {len(file_paths)} ภาพและบันทึกลงประวัติหรือไม่?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                return
            total = len(file_paths)
            if hasattr(self.gui, 'progress_bar'):
                self.gui.progress_bar.setVisible(True)
                self.gui.progress_bar.setMaximum(total)
                self.gui.progress_bar.setValue(0)
                self.gui.progress_bar.setFormat("Starting... (0/%d)" % total)
            self.gui.show_global_loading(f"Processing {total} bottle images...")
            self.gui.status_label.setText(f'🔄 Processing 0/{total} images...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()

            self._bottle_batch_worker = BottleBatchWorker(file_paths, self)
            self._bottle_batch_worker.progress_updated.connect(self._on_bottle_batch_progress)
            self._bottle_batch_worker.image_done.connect(self._on_bottle_batch_image_done)
            self._bottle_batch_worker.finished_with_stats.connect(self._on_bottle_batch_finished)
            self._bottle_batch_worker.error_occurred.connect(self._on_bottle_batch_error)
            self._bottle_batch_worker.start()
        except Exception as e:
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการประมวลผลหลายภาพ:\n{str(e)}")
            print(f"❌ BATCH BOTTLE PROCESSING ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _on_bottle_batch_progress(self, current, total):
        """อัปเดต progress bar และ status จาก thread (GUI ไม่ค้าง)"""
        if hasattr(self.gui, 'progress_bar'):
            self.gui.progress_bar.setValue(current)
            pct = int(current / total * 100) if total else 0
            self.gui.progress_bar.setFormat("ภาพ %d/%d (%d%%)" % (current, total, pct))
        self.gui.status_label.setText('🔄 Processing %d/%d images...' % (current, total))

    def _on_bottle_batch_image_done(self, result, file_path, idx, success):
        """เพิ่มผลแต่ละภาพเข้าประวัติ (เรียกจาก main thread)"""
        if hasattr(self.gui, 'add_to_history'):
            # สร้าง bottle_result จาก result
            bottle_result = result if isinstance(result, dict) else {
                'image': None,
                'bottle_type': result.get('bottle_type', 'Unknown') if isinstance(result, dict) else 'Unknown',
                'combined_ocr_text': result.get('combined_ocr_text', '') if isinstance(result, dict) else ''
            }
            self.gui.add_to_history(bottle_result, None)
        print("✅ BATCH BOTTLE [%d]: Added to history - %s" % (idx, os.path.basename(file_path)))

    def _on_bottle_batch_finished(self, success_count, error_count, good_count, ng_count):
        """ซ่อน loading, แสดงสรุป (เรียกจาก main thread)"""
        total = success_count + error_count
        if hasattr(self.gui, 'progress_bar'):
            self.gui.progress_bar.setVisible(False)
            self.gui.progress_bar.setValue(0)
        self.gui.hide_global_loading()
        self.gui.status_label.setText('✅ ประมวลผลเสร็จสิ้น: สำเร็จ %d/%d, ผิดพลาด %d' % (success_count, total, error_count))
        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        
        # สร้างข้อความสรุปผล
        summary_message = "ประมวลผลเสร็จสิ้น:\n\n"
        summary_message += "📊 สรุปผลการตรวจสอบ:\n"
        summary_message += "  ✅ Good: %d ภาพ\n" % good_count
        summary_message += "  ❌ NG: %d ภาพ\n" % ng_count
        summary_message += "  ⚠️ ไม่มีผลตรวจสอบ: %d ภาพ\n\n" % (success_count - good_count - ng_count)
        summary_message += "📈 สรุปการประมวลผล:\n"
        summary_message += "  ✅ สำเร็จ: %d ภาพ\n" % success_count
        summary_message += "  ❌ ผิดพลาด: %d ภาพ\n\n" % error_count
        summary_message += "ผลลัพธ์ถูกบันทึกลง tab ประวัติแล้ว"
        
        QMessageBox.information(self.gui, "ประมวลผลเสร็จสิ้น", summary_message)
        print("✅ BATCH BOTTLE COMPLETE: Success: %d, Errors: %d, Good: %d, NG: %d" % (success_count, error_count, good_count, ng_count))

    def _on_bottle_batch_error(self, msg):
        if hasattr(self.gui, 'progress_bar'):
            self.gui.progress_bar.setVisible(False)
        self.gui.hide_global_loading()
        self.gui.status_label.setText('❌ ข้อผิดพลาด: %s' % msg)
        self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        QMessageBox.critical(self.gui, "ข้อผิดพลาด", msg)
        if not paths or idx <= 0:
            return
        self.gui._current_bottle_index = idx - 1
        path = paths[self.gui._current_bottle_index]
        image = cv2.imread(path)
        if image is not None:
            self.gui.current_image = image
            self.display_image(image)
            name = os.path.basename(path)
            n, total = self.gui._current_bottle_index + 1, len(paths)
            self.gui.image_info_label.setText(f"ขนาด: {image.shape[1]}x{image.shape[0]} | ไฟล์: {name} ({n}/{total})")
            if hasattr(self.gui, 'btn_next_bottle_image'):
                self.gui.btn_next_bottle_image.setEnabled(True)
            if hasattr(self.gui, 'btn_prev_bottle_image'):
                self.gui.btn_prev_bottle_image.setEnabled(self.gui._current_bottle_index > 0)
    
    def go_next_bottle_image(self):
        """แสดงภาพถัดไปในรายการที่เลือกหลายภาพ"""
        paths = getattr(self.gui, 'selected_image_paths', None)
        idx = getattr(self.gui, '_current_bottle_index', 0)
        if not paths or idx >= len(paths) - 1:
            return
        self.gui._current_bottle_index = idx + 1
        path = paths[self.gui._current_bottle_index]
        image = cv2.imread(path)
        if image is not None:
            self.gui.current_image = image
            self.display_image(image)
            name = os.path.basename(path)
            n, total = self.gui._current_bottle_index + 1, len(paths)
            self.gui.image_info_label.setText(f"ขนาด: {image.shape[1]}x{image.shape[0]} | ไฟล์: {name} ({n}/{total})")
            if hasattr(self.gui, 'btn_prev_bottle_image'):
                self.gui.btn_prev_bottle_image.setEnabled(True)
            if hasattr(self.gui, 'btn_next_bottle_image'):
                self.gui.btn_next_bottle_image.setEnabled(self.gui._current_bottle_index < len(paths) - 1)

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
    
    def capture_image_auto(self):
        """Capture image automatically from Modbus trigger"""
        if self.gui.usb_camera is None:
            print("❌ Camera not initialized - cannot capture")
            return
        
        try:
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
                    print("🔄 AUTO PROCESS: starting automatic processing...")
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
                # รูปใหม่จากคิว → เคลียร์รูปครอปในหน้าหลักและแท็บขวด
                self.clear_cropped_images_display()
                # แสดงเวลาที่ถ่ายภาพ
                capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"🕐 Queue Capture time: {capture_time}")
                print(f"📸 QUEUE CAPTURE: Image shape: {captured_image.shape}")
                
                self.gui.current_image = captured_image
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
    
    def capture_both_and_process_silently(self):
        """ถ่ายทั้งสองกล้อง (USB + Sentech) แล้วประมวลผลขวดแบบเงียบ — ขวด OK จึงจะเริ่มฝา พอ M401 มาแสดงผล+ส่ง"""
        if self.gui.usb_camera is None:
            print("❌ Camera not initialized - cannot capture silently")
            return
        sentech = getattr(self.gui, 'sentech_camera', None)
        if sentech is None or not getattr(sentech, '_is_initialized', False):
            print("🔇 SILENT: ไม่มี Sentech — ใช้ถ่าย USB อย่างเดียว (ผลขวดไม่มีฝา)")
            self.capture_and_process_silently()
            return
        if self._capture_both_thread is not None and self._capture_both_thread.isRunning():
            return
        self._silent_cap_retry_count = 0  # รีเซ็ตนับ retry ทุกครั้งที่เริ่มรอบใหม่
        try:
            print("🔇 SILENT CAPTURE: ถ่ายทั้งสองกล้อง แล้วประมวลผลขวดก่อน...")
            self._capture_both_thread = CaptureBothThread(
                self.gui.usb_camera,
                sentech,
                delay_before_sec=0,
                parent=self.gui
            )
            self._capture_both_thread.capture_done.connect(self._on_silent_capture_both_done)
            self._capture_both_thread.start()
        except Exception as e:
            print(f"❌ SILENT CAPTURE BOTH ERROR: {str(e)}")
            self.silent_mode = False

    def _on_silent_capture_both_done(self, usb_image, sentech_image):
        """หลังถ่ายทั้งสองกล้อง (silent): ตั้งภาพแล้วเริ่มประมวลผลขวด ถ้าภาพฝาไม่มา ถ่ายใหม่สูงสุด 2 ครั้ง"""
        try:
            self._capture_both_thread = None
            if usb_image is None:
                print("❌ SILENT CAPTURE: ไม่ได้ภาพ USB")
                self.silent_mode = False
                return
            self.gui.current_image = usb_image
            if sentech_image is not None:
                self.gui.current_sentech_image = sentech_image
                self._silent_cap_retry_count = 0
                self._start_silent_bottle_then_cap(usb_image)
                return
            # ภาพฝาไม่มา — ถ่ายใหม่ (retry สูงสุด 2 ครั้ง)
            self._silent_cap_retry_count += 1
            if self._silent_cap_retry_count <= 2:
                print(f"🔇 SILENT: ภาพฝาไม่มา — ถ่ายใหม่ครั้งที่ {self._silent_cap_retry_count}/2")
                self._retry_silent_sentech_then_start(usb_image)
                return
            # ครบ 2 ครั้งแล้วยังไม่ได้ภาพฝา — แสดง error แล้วเริ่มขวดอย่างเดียว (ผลขวดไม่มีฝา)
            reason = "กล้อง Sentech ไม่ได้ภาพฝาหลังถ่ายใหม่ 2 ครั้ง"
            print(f"❌ SILENT CAP ERROR: {reason}")
            QMessageBox.warning(
                self.gui,
                "ภาพฝาไม่มา",
                f"{reason}\n\nจะประมวลผลเฉพาะขวด (ไม่มีผลฝา)"
            )
            self.gui.current_sentech_image = None
            self._silent_cap_retry_count = 0
            self._start_silent_bottle_then_cap(usb_image)
        except Exception as e:
            print(f"❌ SILENT CAPTURE BOTH DONE ERROR: {e}")
            self.silent_mode = False
            self._current_bottle_started_silent = False

    def _start_silent_bottle_then_cap(self, usb_image):
        """เริ่มประมวลผลขวดแบบ silent (ฝาจะเริ่มเมื่อขวด OK)"""
        self.silent_mode = True
        self._current_bottle_started_silent = True
        self.gui.progress_bar.setVisible(False)
        self.gui.processing_thread = BottleDetectionThread(usb_image, selected_tastes=self.gui.selected_tastes)
        self.gui.processing_thread.result_ready.connect(self.on_processing_complete)
        self.gui.processing_thread.start()
        print("🔇 SILENT PROCESS: ขวดเริ่มประมวลผล (ฝาจะเริ่มเมื่อขวด OK)")

    def _retry_silent_sentech_then_start(self, usb_image):
        """ถ่ายภาพ Sentech อีกครั้ง (ใน thread) แล้วถ้าได้ภาพ ตั้ง current_sentech_image และเริ่มขวด"""
        sentech = getattr(self.gui, 'sentech_camera', None)
        if sentech is None:
            self._silent_cap_retry_count -= 1
            self._start_silent_bottle_then_cap(usb_image)
            return
        if self._capture_sentech_only_thread is not None and self._capture_sentech_only_thread.isRunning():
            return
        self._capture_sentech_only_thread = CaptureSentechOnlyThread(sentech, parent=self.gui)
        self._capture_sentech_only_thread.capture_done.connect(
            lambda img: self._on_silent_sentech_retry_done(usb_image, img)
        )
        self._capture_sentech_only_thread.start()

    def _on_silent_sentech_retry_done(self, usb_image, sentech_image):
        """หลัง retry ถ่าย Sentech: ถ้าได้ภาพ ตั้งแล้วเริ่มขวด ถ้าไม่ได้และยัง retry ไม่ครบ 2 ครั้ง ถ่ายอีก"""
        try:
            self._capture_sentech_only_thread = None
            if sentech_image is not None:
                self.gui.current_sentech_image = sentech_image
                self._silent_cap_retry_count = 0
                print("✅ SILENT RETRY: ได้ภาพฝาแล้ว")
                self._start_silent_bottle_then_cap(usb_image)
                return
            self._silent_cap_retry_count += 1
            if self._silent_cap_retry_count <= 2:
                print(f"🔇 SILENT: ภาพฝายังไม่มา — ถ่ายใหม่ครั้งที่ {self._silent_cap_retry_count}/2")
                self._retry_silent_sentech_then_start(usb_image)
                return
            reason = "กล้อง Sentech ไม่ได้ภาพฝาหลังถ่ายใหม่ 2 ครั้ง"
            print(f"❌ SILENT CAP ERROR: {reason}")
            QMessageBox.warning(
                self.gui,
                "ภาพฝาไม่มา",
                f"{reason}\n\nจะประมวลผลเฉพาะขวด (ไม่มีผลฝา)"
            )
            self.gui.current_sentech_image = None
            self._silent_cap_retry_count = 0
            self._start_silent_bottle_then_cap(usb_image)
        except Exception as e:
            print(f"❌ SILENT SENTECH RETRY ERROR: {e}")
            self.silent_mode = False
            self._current_bottle_started_silent = False

    def capture_sentech_then_callback(self, callback):
        """ถ่ายภาพ Sentech อย่างเดียว (ใน thread) แล้วเรียก callback(image) บน main thread — ใช้ตอน retry ประมวลผลฝา"""
        sentech = getattr(self.gui, 'sentech_camera', None)
        if sentech is None:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: callback(None))
            return
        if self._capture_sentech_only_thread is not None and self._capture_sentech_only_thread.isRunning():
            return
        self._capture_sentech_only_thread = CaptureSentechOnlyThread(sentech, parent=self.gui)
        self._capture_sentech_only_thread.capture_done.connect(
            lambda img: self._on_sentech_for_callback_done(img, callback)
        )
        self._capture_sentech_only_thread.start()

    def _on_sentech_for_callback_done(self, sentech_image, callback):
        self._capture_sentech_only_thread = None
        if sentech_image is not None:
            self.gui.current_sentech_image = sentech_image
        callback(sentech_image)

    def capture_and_process_silently(self):
        """Capture image (USB เท่านั้น) and process silently - ใช้เมื่อไม่มี Sentech หรือ fallback"""
        if self.gui.usb_camera is None:
            print("❌ Camera not initialized - cannot capture silently")
            return
        try:
            print("🔇 SILENT CAPTURE: Starting silent image capture and processing...")
            self.silent_mode = True
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
                print("🔇 SILENT PROCESS: Starting silent processing...")
                self.gui.progress_bar.setVisible(False)
                self._current_bottle_started_silent = True
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
            self._current_bottle_started_silent = False
    
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
            
    def clear_cropped_images_display(self):
        """เคลียร์พื้นที่แสดงรูปครอปในแท็บขวดและหน้าหลัก (ใช้เมื่อมีรูปใหม่มาในโหมด full auto)"""
        try:
            for i in reversed(range(self.gui.crops_layout.count())):
                widget = self.gui.crops_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            if hasattr(self.gui, 'home_bottle_crops_layout'):
                for i in reversed(range(self.gui.home_bottle_crops_layout.count())):
                    widget = self.gui.home_bottle_crops_layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)
            placeholder_label = QLabel("ยังไม่มีภาพที่ครอป")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_label.setStyleSheet("color: #7f8c8d; padding: 20px;")
            self.gui.crops_layout.addWidget(placeholder_label)
            if hasattr(self.gui, 'home_bottle_crops_layout'):
                home_placeholder = QLabel("ยังไม่มีภาพที่ครอป")
                home_placeholder.setAlignment(Qt.AlignCenter)
                home_placeholder.setStyleSheet("color: #7f8c8d; padding: 20px;")
                self.gui.home_bottle_crops_layout.addWidget(home_placeholder)
            print("✅ CLEAR CROPPED: เคลียร์รูปครอปในหน้าหลักและแท็บขวดแล้ว")
        except Exception as e:
            print(f"⚠️ clear_cropped_images_display: {e}")

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
        
    def process_current_image(self):
        """Process current image. รันขวดก่อน แล้วค่อยรันฝา (ไม่ parallel)."""
        if self.gui.current_image is None:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "ไม่มีภาพให้ประมวลผล")
            return
        
        print(f"🔄 PROCESS: Starting processing for image shape: {self.gui.current_image.shape}")
        print("🔄 PROCESS: Creating BottleDetectionThread...")
        
        # Update status tab
        self.gui.status_handlers.update_bottle_detection_status("Processing...", True)
        
        # แสดง loading — ขวดก่อน แล้วค่อยฝา
        self.gui.show_global_loading("Processing bottle...")
            
        # Start processing thread (ขวดอย่างเดียว — ฝาจะเริ่มเมื่อขวดเสร็จ)
        self.gui.progress_bar.setVisible(True)
        self.gui.progress_bar.setValue(0)
        self.gui.status_label.setText('🔄 Processing bottle...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        
        # Disable process button and enable stop button
        self.gui.btn_process.setEnabled(False)
        if hasattr(self.gui, 'btn_process_bottle_tab'):
            self.gui.btn_process_bottle_tab.setEnabled(False)
        self.gui.btn_process_cap.setEnabled(False)
        self.gui.btn_stop_processing.setEnabled(True)
        
        self.gui.processing_thread = BottleDetectionThread(self.gui.current_image, selected_tastes=self.gui.selected_tastes)
        self.gui.processing_thread.result_ready.connect(self.on_processing_complete)
        self.gui.processing_thread.status_updated.connect(self.gui.status_label.setText)
        self.gui.processing_thread.progress_updated.connect(self.update_bottle_progress_bar)
        print("🔄 PROCESS: Starting processing thread...")
        print(f"🔄 PROCESS: Using selected tastes: {self.gui.selected_tastes}")
        self.gui.processing_thread.start()
    
    def process_all_bottle_images(self):
        """ประมวลผลทุกภาพที่เลือก (เมื่อเลือกหลายภาพ) — ตอนนี้ประมวลผลภาพปัจจุบันเท่านั้น"""
        paths = getattr(self.gui, 'selected_image_paths', None)
        if paths and isinstance(paths, (list, tuple)) and len(paths) > 1:
            self.gui.status_label.setText('🔄 ประมวลผลภาพแรกจากรายการ...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            if self.gui.current_image is not None:
                self.process_current_image()
            else:
                QMessageBox.warning(self.gui, "ข้อผิดพลาด", "ไม่มีภาพให้ประมวลผล")
        else:
            self.process_current_image()
    
    def on_processing_complete(self, result):
        """Handle processing completion"""
        if getattr(self.gui, '_stopping_processing', False):
            return
        print("✅ PROCESS COMPLETE: Processing finished")
        self.gui.progress_bar.setVisible(False)
        # ถ้าขวดเสร็จแล้วจะเริ่มฝาต่อ (มีภาพ Sentech + ขวดผ่าน) — ยังไม่ปิด loading / ยังไม่เปิดปุ่ม จนกว่าฝาจะเสร็จ
        bottle_ng = result.get('defect_inspection') and result.get('defect_inspection', {}).get('result') == 'NG'
        will_start_cap = (
            not self.silent_mode
            and result.get('bottle_type')
            and result.get('combined_ocr_text')
            and not bottle_ng
            and getattr(self.gui, 'current_sentech_image', None) is not None
            and getattr(self.gui, 'cap_detector', None) is not None
            and hasattr(self.gui, 'cap_handlers')
            and self.gui.cap_handlers
        )
        if not will_start_cap:
            self.gui.hide_global_loading()
            self.gui.btn_process.setEnabled(True)
            if hasattr(self.gui, 'btn_process_bottle_tab'):
                self.gui.btn_process_bottle_tab.setEnabled(True)
            self.gui.btn_stop_processing.setEnabled(False)
        
        # Update performance stats
        self.gui.total_images_processed_count += 1
        self.gui.status_handlers.update_performance_stats(images_processed=self.gui.total_images_processed_count)
        
        bottle_defect_ng = result.get('defect_inspection') and result['defect_inspection'].get('result') == 'NG'
        # Check if silent mode - store result in queue instead of displaying
        if self.silent_mode:
            print("🔇 SILENT MODE: Storing result in queue (no UI display, no Modbus signals)")
            self.pending_bottle_result = result
            
            # ขวด NG (Defect) — หยุดประมวลผล ไม่ไปต่อที่ฝา ส่ง NG เลย (ใส่คิว/emit ตาม M401)
            if bottle_defect_ng:
                print("🔇 SILENT MODE: ขวด NG — ไม่ประมวลผลฝา ส่งผล NG ทันที")
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    mt = self.gui.modbus_thread
                    if getattr(mt, 'waiting_for_late_result', False):
                        mt.waiting_for_late_result = False
                        mt.result_ready_to_display.emit(result, None)
                    else:
                        mt.append_pending_result(result, None)
                        print(f"🔇 SILENT MODE: ผลขวด NG เข้าคิว (queue size: {len(mt.pending_results_queue)})")
                    self.pending_bottle_result = None
                else:
                    print("🔇 SILENT MODE: ขวด NG แต่ Modbus ไม่พร้อม")
                self.silent_mode = False
                return
            
            # ขวดผ่าน — รอผลฝาให้เสร็จ แล้วค่อยส่ง (ขวด+ฝา) ทีเดียว
            if getattr(self.gui, 'current_sentech_image', None) is not None and getattr(self.gui, 'cap_detector', None) is not None and hasattr(self.gui, 'cap_handlers') and self.gui.cap_handlers:
                print("🔇 SILENT MODE: รอผลฝาให้เสร็จ แล้วค่อยส่งขวด+ฝาทีเดียว")
                self.gui.cap_handlers.silent_mode = True
                self.gui.cap_handlers.process_cap_detection(started_from_parallel=True)
            else:
                # ไม่มีภาพฝา/โมเดลฝา — ใส่ (ขวด, None) เข้าคิวเพื่อให้ M401 ส่งได้
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    mt = self.gui.modbus_thread
                    if getattr(mt, 'waiting_for_late_result', False):
                        mt.waiting_for_late_result = False
                        mt.result_ready_to_display.emit(result, None)
                        self.pending_bottle_result = None
                    else:
                        mt.append_pending_result(result, None)
                        print(f"🔇 SILENT MODE: ไม่มีฝา — ใส่ผลขวดเข้าคิว (queue size: {len(mt.pending_results_queue)})")
                else:
                    print("🔇 SILENT MODE: Bottle result stored, waiting for cap result...")
            
            # Reset silent mode
            self.silent_mode = False
            return
        
        if "error" not in result:
            print("✅ PROCESS COMPLETE: No error in result")
            print(f"✅ PROCESS COMPLETE: Result keys: {list(result.keys())}")
            
            # Store result first
            self.gui.current_result = result
            
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
            status_ng_shown = False
            bottle_ng = result.get('defect_inspection') and result['defect_inspection'].get('result') == 'NG'
            if result.get('bottle_type') and result.get('combined_ocr_text'):
                print(f"🎯 PROCESS COMPLETE: Bottle type detected: {result['bottle_type']}")
                print(f"📝 PROCESS COMPLETE: OCR text: '{result['combined_ocr_text']}'")
                if bottle_ng:
                    # ขวด NG — ไม่ประมวลผลฝา, ส่ง Modbus NG เท่านั้น
                    print("🏷️ PROCESS COMPLETE: ขวด NG — ไม่เริ่มประมวลผลฝา")
                    if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                        m140_ok = self.gui.modbus_thread.on_m140()
                        if m140_ok:
                            self.gui.status_handlers.update_coil_lamp("m140", True)
                            m600_ok = self.gui.modbus_thread.on_m600()
                            if m600_ok:
                                self.gui.status_handlers.update_coil_lamp("m600", True)
                                self.gui.status_label.setText('🏷️ ขวด NG (Defect) → D7009=50, M140, M600 ส่งแล้ว (ไม่ประมวลฝา)')
                            else:
                                self.gui.status_label.setText('❌ ขวด NG แต่ไม่สามารถ ON M600 ได้')
                            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                        else:
                            self.gui.status_label.setText('❌ ขวด NG แต่ไม่สามารถส่ง D7009=50 ได้')
                            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    else:
                        self.gui.status_label.setText('❌ ขวด NG แต่ Modbus ไม่พร้อม')
                        self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    if hasattr(self.gui, 'add_to_history'):
                        self.gui.add_to_history(result, None)
                    status_ng_shown = True
                else:
                    # ขวดผ่าน — เก็บ result แล้วส่ง D7009/M110 ฯลฯ จากนั้นเริ่มประมวลผลฝา (ขวดแล้วค่อยฝา)
                    if not hasattr(self.gui, 'last_bottle_result'):
                        self.gui.last_bottle_result = None
                    self.gui.last_bottle_result = result
                    self.gui.successful_detections_count += 1
                    self.gui.status_handlers.update_performance_stats(successful_detections=self.gui.successful_detections_count)
                    self.handle_bottle_type_detection(
                        result['bottle_type'], result['combined_ocr_text'], bottle_result=result
                    )
                    # handle_bottle_type_detection เริ่มประมวลผลฝาแล้ว (M100/M110/M120) — ไม่เรียก process_cap_detection ซ้ำ (กัน add history สองครั้ง)
                    if will_start_cap:
                        return
            elif bottle_ng:
                # ขวด defect NG — ส่ง Modbus เหมือน NG อื่นๆ (D7009=50, M140, M600)
                print("🏷️ PROCESS COMPLETE: Defect NG — ส่ง Modbus (D7009=50, M140, M600)")
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    m140_ok = self.gui.modbus_thread.on_m140()
                    if m140_ok:
                        self.gui.status_handlers.update_coil_lamp("m140", True)
                        m600_ok = self.gui.modbus_thread.on_m600()
                        if m600_ok:
                            self.gui.status_handlers.update_coil_lamp("m600", True)
                            self.gui.status_label.setText('🏷️ ขวด NG (Defect) → D7009=50, M140, M600 ส่งแล้ว')
                            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                        else:
                            self.gui.status_label.setText('❌ ขวด NG แต่ไม่สามารถ ON M600 ได้')
                            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    else:
                        self.gui.status_label.setText('❌ ขวด NG แต่ไม่สามารถส่ง D7009=50 ได้')
                        self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                else:
                    self.gui.status_label.setText('❌ ขวด NG แต่ Modbus ไม่พร้อม')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                status_ng_shown = True
            else:
                print("❌ PROCESS COMPLETE: No bottle type or OCR text found")
                print(f"❌ PROCESS COMPLETE: bottle_type: {result.get('bottle_type')}")
                print(f"❌ PROCESS COMPLETE: combined_ocr_text: {result.get('combined_ocr_text')}")
            
            # Add to history: ถ้าเป็น M100/M110/M120 และมีภาพ Sentech จะรอผลฝาแล้ว add ครั้งเดียวใน cap_handlers
            # จึงไม่ add ที่นี่เพื่อไม่ให้ซ้ำ 2 รายการ
            if hasattr(self.gui, 'add_to_history'):
                bottle_type = result.get('bottle_type')
                will_wait_cap = (
                    bottle_type in ["M100", "M110", "M120"] and
                    getattr(self.gui, 'current_sentech_image', None) is not None
                )
                if will_wait_cap:
                    # จะมี add_to_history ตอนฝาเสร็จใน cap_handlers เท่านั้น
                    pass
                else:
                    cap_result = getattr(self.gui, 'current_cap_result', None)
                    self.gui.add_to_history(result, cap_result)

            if not status_ng_shown:
                self.gui.status_label.setText('✅ ประมวลผลเสร็จสิ้น')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        else:
            print(f"❌ PROCESS COMPLETE: Error in result: {result['error']}")
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {result["error"]}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            self.gui.error_count_value += 1
            self.gui.status_handlers.update_performance_stats(errors=self.gui.error_count_value)
    
    def handle_bottle_type_detection(self, bottle_type, ocr_text, from_queue=False, bottle_result=None):
        """Handle bottle type detection and control Modbus.
        from_queue=True: แสดงผลจากคิว — มีผลฝาอยู่แล้ว ไม่เริ่มประมวลผลฝาใหม่ (กันส่ง M140 ซ้ำ)
        bottle_result: dict จากการตรวจจับขวด (ใช้ดึงความมั่นใจ YOLO angle1/type)
        """
        print(f"🎯 BOTTLE TYPE DETECTED: {bottle_type} (OCR: '{ocr_text}')" + (" [from queue]" if from_queue else ""))
        
        # เก็บประเภทขวดที่ตรวจพบ
        self.gui.current_bottle_type = bottle_type
        
        br = bottle_result if bottle_result is not None else getattr(self.gui, 'last_bottle_result', None)
        summary = (br or {}).get('summary') or {}
        type_conf = summary.get('type_confidence')
        angle1_conf = summary.get('angle1_confidence')
        type_easyocr_mean = (br or {}).get('type_easyocr_mean_confidence')
        # Update status tab
        self.gui.status_handlers.update_bottle_type_status(
            bottle_type,
            type_conf=type_conf,
            angle1_conf=angle1_conf,
            type_easyocr_mean=type_easyocr_mean,
        )
        self.gui.status_handlers.update_bottle_detection_status("ตรวจพบแล้ว", False)
        
        # แสดงผลจากคิว: มีผลฝาอยู่แล้ว — ไม่เริ่ม process_cap_detection (จะ validate ฝาที่มีแล้วส่ง Modbus ใน on_result_ready_to_display)
        if from_queue:
            return
        
        # ตรวจสอบประเภทขวดและดำเนินการตามลำดับ
        if bottle_type == "M130":
            # angle3 → ตีเป็น NG เลย (D7009=50, M140, M600) ไม่ใช้ M130 หรือ D7009=40
            m140_ok = self.gui.modbus_thread.on_m140()
            if m140_ok:
                self.gui.status_handlers.update_coil_lamp("m140", True)
                m600_ok = self.gui.modbus_thread.on_m600()
                if m600_ok:
                    self.gui.status_handlers.update_coil_lamp("m600", True)
                    self.gui.status_label.setText('🏷️ angle3 → NG (D7009=50, M140, M600) ส่งแล้ว')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    print("🏷️ ตรวจพบ angle3 → ตีเป็น NG (D7009=50, M140, M600) - ไม่ใช้ M130/D7009=40")
                else:
                    self.gui.status_label.setText('❌ ไม่สามารถ ON M600 ได้')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            else:
                self.gui.status_label.setText('❌ ไม่สามารถส่ง M140 (angle3 → NG) ได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            return  # angle3 ไม่ต้องไปต่อ ไม่ตรวจจับฝา
        
        # M100/M110/M120 - เก็บประเภทขวดไว้รอผลลัพธ์ฝา
        print(f"⏳ ตรวจพบ: {bottle_type} - รอผลลัพธ์ฝาก่อน ON Modbus...")
        self.gui.status_label.setText(f'⏳ ตรวจพบ: {bottle_type} - รอผลลัพธ์ฝาก่อน ON Modbus...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        
        # เริ่มประมวลผลฝา (ถ้ามีภาพจาก Sentech)
        if self.gui.current_sentech_image is not None:
            print("🔄 AUTO CAP PROCESS: เริ่มประมวลผลฝา...")
            
            # แสดงผล GUI สำหรับการประมวลผลฝา
            self.gui.cap_handlers.display_cap_processing_ui()
            
            # เรียกใช้ process_cap_detection() เหมือนการกดปุ่ม
            print("🔍 AUTO CAP PROCESS: Calling process_cap_detection()...")
            if hasattr(self.gui, 'cap_handlers') and self.gui.cap_handlers:
                self.gui.cap_handlers.process_cap_detection()
                print("🔍 AUTO CAP PROCESS: process_cap_detection() completed")
            else:
                print("❌ AUTO CAP PROCESS: cap_handlers ไม่พร้อม")
            
            # ไม่รอบน main thread — ผลฝาจะมาที่ on_cap_processing_complete เมื่อ thread เสร็จ (GUI ไม่ค้าง)
            print("🔍 AUTO CAP PROCESS: Cap processing started (ผลจะแสดงเมื่อประมวลผลฝาเสร็จ)")
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
        # ความมั่นใจ YOLO ของกล่อง angle1 / type (ถ้าไม่มีใน summary ให้คำนวณจาก detections)
        def _conf_for(label_key, lab):
            c = summary.get(label_key)
            if c is None and detections:
                confs = [d['confidence'] for d in detections if d.get('label') == lab]
                c = max(confs) if confs else None
            return c
        a1c = _conf_for('angle1_confidence', 'angle1')
        tyc = _conf_for('type_confidence', 'type')
        if a1c is not None or tyc is not None:
            detail_text += "\n🎯 ความมั่นใจ YOLO:"
            if a1c is not None:
                detail_text += f"\n   angle1: {a1c:.4f}"
            if tyc is not None:
                detail_text += f"\n   type: {tyc:.4f}"
        # แสดงผล defect inspection (angle1 Good/NG) ถ้ามี
        if result.get('defect_inspection'):
            di = result['defect_inspection']
            detail_text += f"\n🔬 Defect (angle1): Score={di.get('score', 0):.4f} → {di.get('result', 'N/A')}\n"
            
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
        
        # Update Home tab bottle verdict label (Good / NG / Waiting)
        if hasattr(self.gui, 'home_bottle_verdict_label') and self.gui.home_bottle_verdict_label:
            v = self.gui.home_bottle_verdict_label
            defect_info = result.get('defect_inspection') or {}
            defect_result = defect_info.get('result')
            bottle_type = result.get('bottle_type')

            if defect_result == 'NG':
                v.setText("⚠️  RESULT: NG (Defect)  ⚠️")
                v.setStyleSheet(
                    "font-size: 22px; font-weight: bold; padding: 10px; "
                    "border-radius: 8px; background-color: #fadbd8; color: #c0392b; border: 2px solid #c0392b;"
                )
            elif bottle_type:
                bits = []
                if a1c is not None:
                    bits.append(f"angle1 {a1c:.4f}")
                if tyc is not None:
                    bits.append(f"type {tyc:.4f}")
                eo = result.get('type_easyocr_mean_confidence')
                if eo is not None:
                    bits.append(f"EasyOCR {eo:.4f}")
                extra = f"\n({ ' · '.join(bits) })" if bits else ""
                v.setText(f"✅  RESULT: {bottle_type}  ✅{extra}")
                v.setStyleSheet(
                    "font-size: 22px; font-weight: bold; padding: 10px; "
                    "border-radius: 8px; background-color: #d5f4e6; color: #27ae60; border: 2px solid #27ae60;"
                )
            else:
                v.setText("—")
                v.setStyleSheet(
                    "font-size: 22px; font-weight: bold; padding: 10px; "
                    "border-radius: 8px; background-color: #ecf0f1; color: #7f8c8d;"
                )

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
            # อัปเดตภาพขวดบนหน้าหลักให้ตรงกับแท็บตรวจจับขวด
            if hasattr(self.gui, 'home_bottle_image_label'):
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
        self.gui.progress_bar.setFormat(f"Processing... ({value}%)")
    
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
        self.gui.progress_bar.setFormat(f"Processing... ({value}%)")
    
    def select_bottle_folder(self):
        """เลือกโฟลเดอร์ที่มีภาพขวด"""
        try:
            folder = QFileDialog.getExistingDirectory(
                self.gui,
                "เลือกโฟลเดอร์ภาพขวด",
                ""
            )
            if folder:
                self.gui.bottle_folder_path = folder
                print(f"📁 BOTTLE FOLDER: Selected: {folder}")
                self.gui.status_label.setText(f'📁 โฟลเดอร์ขวด: {os.path.basename(folder)}')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                self._check_paired_ready()
        except Exception as e:
            print(f"❌ Error selecting bottle folder: {e}")
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเลือกโฟลเดอร์:\n{str(e)}")
    
    def select_cap_folder(self):
        """เลือกโฟลเดอร์ที่มีภาพฝา"""
        try:
            folder = QFileDialog.getExistingDirectory(
                self.gui,
                "เลือกโฟลเดอร์ภาพฝา",
                ""
            )
            if folder:
                self.gui.cap_folder_path = folder
                print(f"📁 CAP FOLDER: Selected: {folder}")
                self.gui.status_label.setText(f'📁 โฟลเดอร์ฝา: {os.path.basename(folder)}')
                self.gui.status_label.setStyleSheet("color: #9C27B0; padding: 5px;")
                self._check_paired_ready()
        except Exception as e:
            print(f"❌ Error selecting cap folder: {e}")
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเลือกโฟลเดอร์:\n{str(e)}")
    
    def _update_process_paired_button_state(self):
        """เปิดปุ่ม Process both เมื่อเลือกโฟลเดอร์ขวด+ฝา หรือมีภาพ USB+Sentech ใน memory จากการถ่ายทั้งสองกล้อง"""
        if not hasattr(self.gui, 'btn_process_paired') or self.gui.btn_process_paired is None:
            return
        folders_ok = bool(
            getattr(self.gui, 'bottle_folder_path', None)
            and getattr(self.gui, 'cap_folder_path', None)
        )
        live_ok = (
            getattr(self.gui, 'current_image', None) is not None
            and getattr(self.gui, 'current_sentech_image', None) is not None
        )
        self.gui.btn_process_paired.setEnabled(bool(folders_ok or live_ok))

    def _check_paired_ready(self):
        """ตรวจสอบว่าพร้อมประมวลผลคู่กันหรือไม่ (โฟลเดอร์ หรือ ภาพถ่ายสองกล้องล่าสุด)"""
        self._update_process_paired_button_state()
        if getattr(self.gui, 'bottle_folder_path', None) and getattr(self.gui, 'cap_folder_path', None):
            self.gui.status_label.setText('✅ พร้อมประมวลผลคู่กัน')
            self.gui.status_label.setStyleSheet("color: #4CAF50; padding: 5px;")
    
    def _match_files(self, bottle_files, cap_files):
        """จับคู่ไฟล์ขวดกับฝาแบบง่าย: จับคู่ตามลำดับเรียงชื่อไฟล์

        สมมติว่าจำนวนไฟล์ขวดและฝาเท่ากันหรือใกล้เคียง และถูกถ่ายมาเป็นคู่ ๆ
        วิธีนี้ไม่ต้องพึ่งชื่อไฟล์ให้ตรงกัน (usb_* กับ sentech_*) แต่จะจับคู่ตามลำดับที่เรียงแล้วแทน
        """
        # เรียงลำดับชื่อไฟล์เพื่อให้จับคู่ได้คงที่
        bottle_files_sorted = sorted(bottle_files)
        cap_files_sorted = sorted(cap_files)

        matched_pairs = []

        # จับคู่ตามลำดับจนกว่าจะหมดด้านใดด้านหนึ่ง
        pair_count = min(len(bottle_files_sorted), len(cap_files_sorted))
        for i in range(pair_count):
            matched_pairs.append((bottle_files_sorted[i], cap_files_sorted[i]))

        # ถ้ามีไฟล์ขวดเกิน ให้ถือว่าไม่มีฝาคู่ (None)
        for i in range(pair_count, len(bottle_files_sorted)):
            matched_pairs.append((bottle_files_sorted[i], None))

        return matched_pairs
    
    def process_paired_images(self):
        """ประมวลผลขวดและฝาคู่กัน — จากโฟลเดอร์ หรือจากภาพถ่ายทั้งสองกล้อง (USB + Sentech) ใน memory"""
        try:
            folders_ok = bool(
                getattr(self.gui, 'bottle_folder_path', None)
                and getattr(self.gui, 'cap_folder_path', None)
            )
            live_ok = (
                getattr(self.gui, 'current_image', None) is not None
                and getattr(self.gui, 'current_sentech_image', None) is not None
            )
            if not folders_ok and not live_ok:
                QMessageBox.warning(
                    self.gui,
                    "คำเตือน",
                    "เลือกโฟลเดอร์ขวดและโฟลเดอร์ฝา หรือกดถ่ายทั้งสองกล้องให้ได้ภาพ USB และ Sentech ครบก่อน",
                )
                return

            if not folders_ok and live_ok:
                self._process_live_bottle_cap_pair()
                return

            # อ่านไฟล์จากโฟลเดอร์
            bottle_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
            bottle_files = []
            for ext in bottle_extensions:
                bottle_files.extend([os.path.join(self.gui.bottle_folder_path, f) 
                                   for f in os.listdir(self.gui.bottle_folder_path) 
                                   if f.lower().endswith(ext)])
            
            cap_files = []
            for ext in bottle_extensions:
                cap_files.extend([os.path.join(self.gui.cap_folder_path, f) 
                                for f in os.listdir(self.gui.cap_folder_path) 
                                if f.lower().endswith(ext)])
            
            if not bottle_files:
                QMessageBox.warning(self.gui, "คำเตือน", "ไม่พบไฟล์ภาพในโฟลเดอร์ขวด")
                return
            
            # จับคู่ไฟล์
            matched_pairs = self._match_files(bottle_files, cap_files)
            
            if not matched_pairs:
                QMessageBox.warning(self.gui, "คำเตือน", "ไม่พบไฟล์ที่จับคู่ได้")
                return
            
            # แสดง dialog ยืนยัน
            reply = QMessageBox.question(
                self.gui,
                "ยืนยันการประมวลผล",
                f"พบ {len(matched_pairs)} คู่ไฟล์\nต้องการประมวลผลทั้งหมดหรือไม่?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # เริ่มประมวลผลใน thread
                self._process_paired_thread(matched_pairs)
            else:
                return
                
        except Exception as e:
            print(f"❌ Error in process_paired_images: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", f"เกิดข้อผิดพลาด:\n{str(e)}")

    def _process_live_bottle_cap_pair(self):
        """ประมวลผล 1 คู่จาก current_image (USB) + current_sentech_image (Sentech) โดยไม่ใช้โฟลเดอร์"""
        try:
            bottle_img = getattr(self.gui, 'current_image', None)
            cap_img = getattr(self.gui, 'current_sentech_image', None)
            if bottle_img is None or cap_img is None:
                QMessageBox.warning(self.gui, "คำเตือน", "ไม่มีภาพขวด (USB) หรือภาพฝา (Sentech)")
                return
            reply = QMessageBox.question(
                self.gui,
                "ยืนยันการประมวลผล",
                "ประมวลผลภาพขวด (USB) และภาพฝา (Sentech) ที่ถ่ายไว้ตอนนี้ 1 คู่?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

            self.gui.status_label.setText("🔄 กำลังประมวลผลคู่ภาพจากกล้อง...")
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()

            fd_b, path_b = tempfile.mkstemp(suffix=".png")
            fd_c, path_c = tempfile.mkstemp(suffix=".png")
            os.close(fd_b)
            os.close(fd_c)
            try:
                cv2.imwrite(path_b, bottle_img)
                cv2.imwrite(path_c, cap_img)
                from libs.detection.bottledetect import process_bottle_image_simple
                from core.image_processor import (
                    perform_ocr_on_image,
                    check_bottle_type,
                    aggregate_easyocr_confidences_from_type_crops,
                )

                bottle_result = process_bottle_image_simple(path_b)
                if isinstance(bottle_result, dict) and bottle_result.get("error"):
                    QMessageBox.warning(self.gui, "ข้อผิดพลาด", str(bottle_result.get("error")))
                    return
                # EasyOCR บนโซน type (เหมือน thread หลัก)
                selected_tastes = getattr(self.gui, "selected_tastes", ["M100", "M110", "M120"])
                type_crops = bottle_result.get("type_crops")
                combined_ocr_text = ""
                if type_crops and isinstance(type_crops, list):
                    for crop in type_crops:
                        if not isinstance(crop, dict) or "original_crop" not in crop:
                            continue
                        ocr_results = perform_ocr_on_image(crop["original_crop"])
                        crop["ocr_results"] = ocr_results
                        if ocr_results and isinstance(ocr_results, list):
                            for ocr_result in ocr_results:
                                if isinstance(ocr_result, dict) and "text" in ocr_result:
                                    combined_ocr_text += ocr_result["text"] + " "
                    combined_ocr_text = combined_ocr_text.strip()
                    bottle_result["combined_ocr_text"] = combined_ocr_text
                    bottle_result["bottle_type"] = check_bottle_type(combined_ocr_text, selected_tastes)
                    bottle_result.update(aggregate_easyocr_confidences_from_type_crops(type_crops))
                cap_result = self.gui.cap_handlers.process_cap_detection_sync(cap_img, path_c)
                self._on_paired_done(path_b, path_c, bottle_result, cap_result)
            finally:
                for p in (path_b, path_c):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

            self.gui.status_label.setText("✅ ประมวลผลคู่ภาพจากกล้องเสร็จแล้ว")
            self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            QMessageBox.information(self.gui, "เสร็จสิ้น", "ประมวลผลภาพคู่จากกล้องเสร็จแล้ว (บันทึกในประวัติแล้ว)")
        except Exception as e:
            print(f"❌ Error in _process_live_bottle_cap_pair: {e}")
            import traceback

            traceback.print_exc()
            self.gui.status_label.setText(f"❌ ประมวลผลคู่ภาพผิดพลาด: {e}")
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", str(e))
    
    def _process_paired_thread(self, matched_pairs):
        """ประมวลผลคู่กันใน thread แยก"""
        from PyQt5.QtCore import QThread, pyqtSignal
        
        class PairedProcessingThread(QThread):
            progress = pyqtSignal(int, int)  # current, total
            pair_done = pyqtSignal(str, str, object, object)  # bottle_path, cap_path, bottle_result, cap_result
            finished = pyqtSignal(int, int)  # success_count, error_count
            
            def __init__(self, matched_pairs, handlers):
                super().__init__()
                self.matched_pairs = matched_pairs
                self.handlers = handlers
                self.should_stop = False
            
            def run(self):
                success_count = 0
                error_count = 0
                total = len(self.matched_pairs)
                
                for idx, (bottle_path, cap_path) in enumerate(self.matched_pairs):
                    if self.should_stop:
                        break
                    
                    try:
                        self.progress.emit(idx + 1, total)
                        
                        # โหลดภาพขวด
                        bottle_image = cv2.imread(bottle_path)
                        if bottle_image is None:
                            print(f"❌ Cannot load bottle image: {bottle_path}")
                            error_count += 1
                            continue
                        
                        # ประมวลผลขวด
                        from libs.detection.bottledetect import process_bottle_image_simple
                        from core.image_processor import (
                            perform_ocr_on_image,
                            check_bottle_type,
                            aggregate_easyocr_confidences_from_type_crops,
                        )

                        bottle_result = process_bottle_image_simple(bottle_path)
                        selected_tastes = getattr(
                            self.handlers.gui, "selected_tastes", ["M100", "M110", "M120"]
                        )
                        type_crops = bottle_result.get("type_crops")
                        combined_ocr_text = ""
                        if type_crops and isinstance(type_crops, list):
                            for crop in type_crops:
                                if not isinstance(crop, dict) or "original_crop" not in crop:
                                    continue
                                ocr_results = perform_ocr_on_image(crop["original_crop"])
                                crop["ocr_results"] = ocr_results
                                if ocr_results and isinstance(ocr_results, list):
                                    for ocr_result in ocr_results:
                                        if isinstance(ocr_result, dict) and "text" in ocr_result:
                                            combined_ocr_text += ocr_result["text"] + " "
                            combined_ocr_text = combined_ocr_text.strip()
                            bottle_result["combined_ocr_text"] = combined_ocr_text
                            bottle_result["bottle_type"] = check_bottle_type(
                                combined_ocr_text, selected_tastes
                            )
                            bottle_result.update(
                                aggregate_easyocr_confidences_from_type_crops(type_crops)
                            )
                        
                        # ประมวลผลฝา (ถ้ามี)
                        cap_result = None
                        if cap_path:
                            cap_image = cv2.imread(cap_path)
                            if cap_image is not None:
                                # ใช้ cap_handlers ประมวลผลฝา
                                cap_result = self.handlers.gui.cap_handlers.process_cap_detection_sync(
                                    cap_image, cap_path
                                )
                        
                        # ส่งผลลัพธ์
                        self.pair_done.emit(bottle_path, cap_path or "", bottle_result, cap_result)
                        success_count += 1
                        
                    except Exception as e:
                        print(f"❌ Error processing pair {idx+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        error_count += 1
                
                self.finished.emit(success_count, error_count)
        
        # สร้างและเริ่ม thread
        self.paired_thread = PairedProcessingThread(matched_pairs, self)
        self.paired_thread.progress.connect(self._on_paired_progress)
        self.paired_thread.pair_done.connect(self._on_paired_done)
        self.paired_thread.finished.connect(self._on_paired_finished)
        
        # แสดง progress bar
        if hasattr(self.gui, 'progress_bar'):
            self.gui.progress_bar.setVisible(True)
            self.gui.progress_bar.setMaximum(len(matched_pairs))
            self.gui.progress_bar.setValue(0)
        
        self.gui.status_label.setText(f'🔄 Processing {len(matched_pairs)} pairs...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        
        # Disable process-both button and enable Stop so user can cancel batch
        if hasattr(self.gui, 'btn_process_paired'):
            self.gui.btn_process_paired.setEnabled(False)
        if hasattr(self.gui, 'btn_stop_processing'):
            self.gui.btn_stop_processing.setEnabled(True)
        
        self.paired_thread.start()
    
    def _on_paired_progress(self, current, total):
        """อัปเดต progress"""
        if hasattr(self.gui, 'progress_bar'):
            self.gui.progress_bar.setValue(current)
            self.gui.progress_bar.setFormat(f"Processing... ({current}/{total})")
        self.gui.status_label.setText(f'🔄 Processing {current}/{total}...')
        QtWidgets.QApplication.processEvents()
    
    def _on_paired_done(self, bottle_path, cap_path, bottle_result, cap_result):
        """เมื่อประมวลผลคู่หนึ่งเสร็จ"""
        try:
            # เพิ่มเข้า history
            if hasattr(self.gui, 'add_to_history'):
                self.gui.add_to_history(bottle_result, cap_result)
            print(f"✅ Processed pair: {os.path.basename(bottle_path)} + {os.path.basename(cap_path) if cap_path else 'None'}")
        except Exception as e:
            print(f"❌ Error in _on_paired_done: {e}")
    
    def _on_paired_finished(self, success_count, error_count):
        """เมื่อประมวลผลทั้งหมดเสร็จ"""
        if hasattr(self.gui, 'progress_bar'):
            self.gui.progress_bar.setVisible(False)
        
        self._update_process_paired_button_state()
        
        self.gui.status_label.setText(
            f'✅ เสร็จสิ้น! สำเร็จ: {success_count}, ผิดพลาด: {error_count}'
        )
        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        
        QMessageBox.information(
            self.gui,
            "เสร็จสิ้น",
            f"ประมวลผลเสร็จสิ้น\nสำเร็จ: {success_count} คู่\nผิดพลาด: {error_count} คู่"
        )


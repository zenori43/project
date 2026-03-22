# -*- coding: utf-8 -*-
"""
Cap Detection Event Handlers
Handles all events related to cap detection processing
"""

from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEventLoop, QTimer
from core.business_logic import CapDetectionThread
import cv2
import numpy as np
import datetime
import os
import time

from config.settings import USE_SIMULATED_IMAGES, SIMULATED_SENTECH_IMAGE_FOLDER
from core.simulated_capture import get_next_sentech_image

# Import CUDA image utilities
from core.cuda_image_utils import cuda_resize, cuda_cvtColor


class CaptureSentechWorker(QThread):
    """ถ่ายภาพ Sentech ใน thread แยก ไม่บล็อก GUI"""
    image_captured = pyqtSignal(object)  # numpy array or None

    def __init__(self, sentech_camera, usb_camera=None):
        super().__init__()
        self.sentech_camera = sentech_camera
        self.usb_camera = usb_camera

    def run(self):
        img = None
        try:
            if self.sentech_camera:
                img = self.sentech_camera.capture_image()
            if img is None and self.usb_camera and getattr(self.usb_camera, 'is_opened', lambda: False)():
                img = self.usb_camera.capture_image()
        except Exception as e:
            print("❌ CaptureSentechWorker: %s" % e)
        self.image_captured.emit(img)


class QueueCaptureSentechWorker(QThread):
    """ถ่ายภาพ Sentech จากคิวใน thread (รอ init ได้ใน thread) ไม่บล็อก GUI"""
    image_captured = pyqtSignal(object)  # numpy array or None

    def __init__(self, sentech_camera, usb_camera=None):
        super().__init__()
        self.sentech_camera = sentech_camera
        self.usb_camera = usb_camera

    def run(self):
        import time
        cam = self.sentech_camera
        if cam is None:
            self.image_captured.emit(None)
            return
        wait_time = 0.0
        while getattr(cam, '_is_initialized', True) is False and wait_time < 10.0:
            time.sleep(0.1)
            wait_time += 0.1
        img = None
        try:
            img = cam.capture_image()
        except Exception as e:
            print("❌ QueueCaptureSentechWorker: %s" % e)
        if img is None and self.usb_camera and getattr(self.usb_camera, 'is_opened', lambda: False)():
            try:
                img = self.usb_camera.capture_image()
            except Exception:
                pass
        self.image_captured.emit(img)


class CapBatchWorker(QThread):
    """รัน batch ฝาใน thread แยก ไม่บล็อก GUI และส่ง progress ให้ progress bar อัปเดต"""
    progress_updated = pyqtSignal(int, int)  # current (1-based), total
    image_done = pyqtSignal(object, str, int, bool, bool)  # result, file_path, idx, success, is_faded
    finished_with_stats = pyqtSignal(int, int, list)  # success_count, error_count, faded_files
    error_occurred = pyqtSignal(str)

    def __init__(self, file_paths, handler):
        super().__init__()
        self.file_paths = file_paths
        self.handler = handler

    def run(self):
        success_count = 0
        error_count = 0
        faded_files = []
        total = len(self.file_paths)
        for idx, file_path in enumerate(self.file_paths, 1):
            self.progress_updated.emit(idx - 1, total)
            try:
                image = cv2.imread(file_path)
                if image is None:
                    error_count += 1
                    self.progress_updated.emit(idx, total)
                    continue
                result = self.handler.process_cap_detection_sync(image, file_path)
                is_faded = result and result.get("error") == "faded_text_detected"
                is_success = result and "error" not in result
                if is_success or is_faded:
                    success_count += 1
                    if is_faded:
                        faded_files.append(os.path.basename(file_path))
                    self.image_done.emit(result, file_path, idx, True, is_faded)
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                import traceback
                traceback.print_exc()
            self.progress_updated.emit(idx, total)
        self.finished_with_stats.emit(success_count, error_count, faded_files)


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
        self._silent_cap_process_retry = 0  # นับรอบ retry ประมวลผลฝา (สูงสุด 2 ครั้ง)
    
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
                    self.gui.sentech_image_info_label.setText(f"Size: {image.shape[1]}x{image.shape[0]} | File: {file_path.split('/')[-1]}")
                    if hasattr(self.gui, 'home_cap_image_info_label'):
                        self.gui.home_cap_image_info_label.setText(f"Size: {image.shape[1]}x{image.shape[0]} | File: {file_path.split('/')[-1]}")
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
        """Select multiple image files for batch cap detection (รันใน thread แยก ไม่ให้ GUI ค้าง)"""
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self.gui,
                "เลือกหลายไฟล์ภาพสำหรับการตรวจจับฝา",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            if not file_paths:
                return
            print(f"📁 BATCH CAP PROCESSING: Selected {len(file_paths)} files")
            reply = QMessageBox.question(
                self.gui,
                "ยืนยันการประมวลผลหลายภาพ",
                f"คุณต้องการประมวลผล {len(file_paths)} ภาพและบันทึกลงประวัติหรือไม่?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                return
            if not self.gui.cap_detector:
                QMessageBox.warning(self.gui, "ข้อผิดพลาด", "โมเดลตรวจจับฝายังไม่ได้โหลดเสร็จ")
                return
            total = len(file_paths)
            if hasattr(self.gui, 'cap_progress_bar'):
                self.gui.cap_progress_bar.setVisible(True)
                self.gui.cap_progress_bar.setMaximum(total)
                self.gui.cap_progress_bar.setValue(0)
                self.gui.cap_progress_bar.setFormat("Starting... (0/%d)" % total)
            self.gui.show_global_loading(f"Processing {total} cap images...")
            self.gui.status_label.setText(f'🔄 Processing 0/{total} images...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()

            self._cap_batch_worker = CapBatchWorker(file_paths, self)
            self._cap_batch_worker.progress_updated.connect(self._on_cap_batch_progress)
            self._cap_batch_worker.image_done.connect(self._on_cap_batch_image_done)
            self._cap_batch_worker.finished_with_stats.connect(self._on_cap_batch_finished)
            self._cap_batch_worker.error_occurred.connect(self._on_cap_batch_error)
            self._cap_batch_worker.start()
        except Exception as e:
            self.gui.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self.gui, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการประมวลผลหลายภาพ:\n{str(e)}")
            print(f"❌ BATCH CAP PROCESSING ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    def _on_cap_batch_progress(self, current, total):
        """อัปเดต progress bar และ status จาก thread (GUI ไม่ค้าง)"""
        if hasattr(self.gui, 'cap_progress_bar'):
            self.gui.cap_progress_bar.setValue(current)
            pct = int(current / total * 100) if total else 0
            self.gui.cap_progress_bar.setFormat("Image %d/%d (%d%%)" % (current, total, pct))
        self.gui.status_label.setText('🔄 Processing %d/%d images...' % (current, total))

    def _on_cap_batch_image_done(self, result, file_path, idx, success, is_faded):
        """เพิ่มผลแต่ละภาพเข้าประวัติ (เรียกจาก main thread)"""
        bottle_result = {
            'image': None,
            'bottle_type': 'Batch Processing' if success else 'Faded Text Detected',
            'combined_ocr_text': 'Batch: %s' % os.path.basename(file_path)
        }
        if hasattr(self.gui, 'add_to_history'):
            self.gui.add_to_history(bottle_result, result)
        if is_faded:
            print("⚠️ BATCH [%d]: Added faded text to history - %s" % (idx, os.path.basename(file_path)))
        else:
            print("✅ BATCH [%d]: Added to history" % idx)

    def _on_cap_batch_finished(self, success_count, error_count, faded_files):
        """ซ่อน loading, แสดงสรุป (เรียกจาก main thread)"""
        total = success_count + error_count
        if hasattr(self.gui, 'cap_progress_bar'):
            self.gui.cap_progress_bar.setVisible(False)
            self.gui.cap_progress_bar.setValue(0)
        self.gui.hide_global_loading()
        self.gui.status_label.setText('✅ ประมวลผลเสร็จสิ้น: สำเร็จ %d/%d, ผิดพลาด %d' % (success_count, total, error_count))
        self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        summary_message = "ประมวลผลเสร็จสิ้น:\n✅ สำเร็จ: %d ภาพ\n❌ ผิดพลาด: %d ภาพ" % (success_count, error_count)
        if faded_files:
            summary_message += "\n\n⚠️ ภาพที่จาง (%d ภาพ):" % len(faded_files)
            for f in faded_files[:10]:
                summary_message += "\n  • " + f
            if len(faded_files) > 10:
                summary_message += "\n  ... และอีก %d ภาพ" % (len(faded_files) - 10)
        summary_message += "\n\nผลลัพธ์ถูกบันทึกลง tab ประวัติแล้ว"
        QMessageBox.information(self.gui, "ประมวลผลเสร็จสิ้น", summary_message)
        print("✅ BATCH CAP COMPLETE: Success: %d, Errors: %d, Faded: %d" % (success_count, error_count, len(faded_files)))

    def _on_cap_batch_error(self, msg):
        if hasattr(self.gui, 'cap_progress_bar'):
            self.gui.cap_progress_bar.setVisible(False)
        self.gui.hide_global_loading()
        self.gui.status_label.setText('❌ ข้อผิดพลาด: %s' % msg)
        self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        QMessageBox.critical(self.gui, "ข้อผิดพลาด", msg)
    
    def process_cap_detection_sync(self, image, file_path=None):
        """Process cap detection synchronously (for batch) — ใช้ QEventLoop รอผล ไม่หมุน loop ให้ GUI ไม่ค้าง"""
        try:
            if not self.gui.cap_detector:
                return {'error': 'Cap detector model not loaded'}
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
            result_container = {'result': None, 'completed': False}
            loop = QEventLoop()
            timeout_timer = QTimer()
            timeout_timer.setSingleShot(True)

            def on_result_ready(result):
                result_container['result'] = result
                result_container['completed'] = True
                timeout_timer.stop()
                loop.quit()

            def on_timeout():
                if not result_container['completed']:
                    print("⏱️ BATCH: Timeout waiting for result")
                    cap_thread.terminate()
                    cap_thread.wait(1000)
                loop.quit()

            def on_status_updated(status):
                if hasattr(self.gui, 'status_label') and file_path and "กำลังประมวลผล" in self.gui.status_label.text():
                    parts = self.gui.status_label.text().split(": ", 1)
                    base = parts[0] if len(parts) > 1 else self.gui.status_label.text()
                    self.gui.status_label.setText("%s: %s" % (base, status))

            cap_thread.result_ready.connect(on_result_ready)
            cap_thread.status_updated.connect(on_status_updated)
            timeout_timer.timeout.connect(on_timeout)
            timeout_timer.start(60000)
            cap_thread.start()
            loop.exec_()
            if not result_container['completed']:
                return {'error': 'Timeout'}
            return result_container['result']
        except Exception as e:
            print("❌ BATCH SYNC PROCESSING ERROR: %s" % e)
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def capture_sentech_image(self):
        """Capture image from Sentech ใน thread แยก ไม่ให้ GUI ค้าง"""
        if self.gui.sentech_camera is None:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "กล้อง Sentech ยังไม่ได้เริ่มต้น")
            return
        self.gui.status_label.setText('📸 กำลังถ่ายภาพจาก Sentech...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        QtWidgets.QApplication.processEvents()
        worker = CaptureSentechWorker(
            self.gui.sentech_camera,
            getattr(self.gui, 'usb_camera', None)
        )
        worker.image_captured.connect(self._on_sentech_capture_done)
        worker.finished.connect(lambda: setattr(self, '_capture_sentech_worker', None))
        self._capture_sentech_worker = worker
        worker.start()

    def _on_sentech_capture_done(self, captured_image):
        """อัปเดต GUI หลังถ่ายภาพ Sentech เสร็จ (เรียกบน main thread)"""
        if captured_image is not None:
            self.gui.current_cap_result = None
            if hasattr(self.gui, 'cap_detection_text') and self.gui.cap_detection_text:
                self.gui.cap_detection_text.clear()
            if hasattr(self.gui, 'cap_results_layout') and self.gui.cap_results_layout:
                for i in reversed(range(self.gui.cap_results_layout.count())):
                    w = self.gui.cap_results_layout.itemAt(i).widget()
                    if w:
                        w.setParent(None)
            self.gui.current_sentech_image = captured_image
            self.display_sentech_image(captured_image)
            self.gui.sentech_image_info_label.setText("Size: %dx%d" % (captured_image.shape[1], captured_image.shape[0]))
            if hasattr(self.gui, 'home_cap_image_info_label'):
                self.gui.home_cap_image_info_label.setText("Size: %dx%d" % (captured_image.shape[1], captured_image.shape[0]))
            self.gui.btn_process_cap.setEnabled(True)
            self.gui.btn_save_sentech_image.setEnabled(True)
            self.gui.status_label.setText('✅ ถ่ายภาพจาก Sentech สำเร็จ - พร้อมประมวลผลฝา')
            self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        else:
            self.gui.status_label.setText('❌ ไม่สามารถถ่ายภาพจาก Sentech ได้')
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_sentech_image_auto(self):
        """Capture image automatically from Sentech camera after M301 trigger using Harvesters"""
        try:
            # โหมดจำลอง: ดึงภาพถัดไปจากโฟลเดอร์ฝา
            if USE_SIMULATED_IMAGES:
                print("🧪 SIMULATION: SENTECH AUTO CAPTURE ใช้ภาพฝาจำลองจากโฟลเดอร์ (ไล่ลงเรื่อยๆ)")
                captured_image, _ = get_next_sentech_image(SIMULATED_SENTECH_IMAGE_FOLDER)
                if captured_image is None:
                    self.gui.status_label.setText('❌ ไม่สามารถโหลดภาพฝาจำลองได้ (ตรวจสอบโฟลเดอร์)')
                    self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    return
            else:
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
                self.gui.sentech_image_info_label.setText(f"Size: {captured_image.shape[1]}x{captured_image.shape[0]} | Time: {capture_time}")
                if hasattr(self.gui, 'home_cap_image_info_label'):
                    self.gui.home_cap_image_info_label.setText(f"Size: {captured_image.shape[1]}x{captured_image.shape[0]} | Time: {capture_time}")
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
                        self.gui.sentech_image_info_label.setText(f"Size: {fallback_image.shape[1]}x{fallback_image.shape[0]} | Time: {capture_time} (USB fallback)")
                        if hasattr(self.gui, 'home_cap_image_info_label'):
                            self.gui.home_cap_image_info_label.setText(f"Size: {fallback_image.shape[1]}x{fallback_image.shape[0]} | Time: {capture_time} (USB fallback)")
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
                        self.gui.sentech_image_info_label.setText(f"Size: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB fallback)")
                        if hasattr(self.gui, 'home_cap_image_info_label'):
                            self.gui.home_cap_image_info_label.setText(f"Size: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB fallback)")
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
        """Capture image จากคิวใน worker thread ไม่บล็อก GUI"""
        if self.gui.sentech_camera is None:
            print("❌ Sentech camera not initialized - cannot capture from queue")
            return
        print("📸 SENTECH QUEUE CAPTURE: Starting (worker thread)...")
        self.gui.status_label.setText('📸 กำลังถ่ายภาพจาก Sentech จากคิว...')
        self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        QtWidgets.QApplication.processEvents()
        worker = QueueCaptureSentechWorker(
            self.gui.sentech_camera,
            getattr(self.gui, 'usb_camera', None)
        )
        worker.image_captured.connect(self._on_queue_sentech_capture_done)
        worker.finished.connect(lambda: setattr(self, '_queue_sentech_worker', None))
        self._queue_sentech_worker = worker
        worker.start()

    def _on_queue_sentech_capture_done(self, captured_image):
        """อัปเดต GUI หลังถ่ายภาพ Sentech จากคิวเสร็จ (รันบน main thread)"""
        try:
            if captured_image is not None:
                self.gui.current_cap_result = None
                if hasattr(self.gui, 'cap_detection_text') and self.gui.cap_detection_text:
                    self.gui.cap_detection_text.clear()
                if hasattr(self.gui, 'cap_results_layout') and self.gui.cap_results_layout:
                    for i in reversed(range(self.gui.cap_results_layout.count())):
                        w = self.gui.cap_results_layout.itemAt(i).widget()
                        if w:
                            w.setParent(None)
                capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print("🕐 Sentech Queue Capture time: %s" % capture_time)
                self.gui.current_sentech_image = captured_image
                self.display_sentech_image(captured_image)
                self.gui.sentech_image_info_label.setText("Size: %dx%d | Time: %s | From queue" % (captured_image.shape[1], captured_image.shape[0], capture_time))
                if hasattr(self.gui, 'home_cap_image_info_label'):
                    self.gui.home_cap_image_info_label.setText("Size: %dx%d | Time: %s | From queue" % (captured_image.shape[1], captured_image.shape[0], capture_time))
                self.gui.btn_process_cap.setEnabled(True)
                self.gui.btn_save_sentech_image.setEnabled(True)
                self.gui.status_label.setText('✅ ถ่ายภาพจาก Sentech จากคิวสำเร็จ - พร้อมประมวลผลฝา')
                self.gui.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                print("🔄 SENTECH QUEUE: เก็บภาพฝาไว้ - รอผลขวด")
            else:
                self.gui.status_label.setText('❌ ไม่สามารถถ่ายภาพจาก Sentech จากคิวได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        except Exception as e:
            self.gui.status_label.setText('❌ ข้อผิดพลาด: %s' % str(e))
            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_and_process_silently(self):
        """Capture image and process silently (no UI display, no Modbus signals) - store result in queue"""
        if self.gui.sentech_camera is None:
            print("❌ Sentech camera not initialized - cannot capture silently")
            # ถ้ากล้องฝาใช้ไม่ได้ แต่มีผลขวดรออยู่ ให้ดันผลขวดอย่างเดียวเข้า queue/แสดงผล
            self._flush_silent_bottle_only_if_pending(reason="sentech_not_initialized")
            return
        print("🔇 SILENT CAPTURE: Starting (worker thread)...")
        self.silent_mode = True
        worker = QueueCaptureSentechWorker(
            self.gui.sentech_camera,
            getattr(self.gui, 'usb_camera', None)
        )
        worker.image_captured.connect(self._on_silent_capture_cap_done)
        worker.finished.connect(lambda: setattr(self, '_silent_capture_cap_worker', None))
        self._silent_capture_cap_worker = worker
        worker.start()

    def _on_silent_capture_cap_done(self, captured_image):
        """หลังถ่ายภาพฝาแบบเงียบเสร็จ เริ่มประมวลผลฝาแบบเงียบ (รันบน main thread)"""
        try:
            if captured_image is not None:
                print("🔇 SILENT CAPTURE: Image shape: %s" % str(captured_image.shape))
                self.gui.current_sentech_image = captured_image
                self.gui.cap_progress_bar.setVisible(False)
                bottle_type = getattr(self.gui, 'current_bottle_type', None)
                print("🏷️ CAP SILENT: bottle_type = %s" % bottle_type)
                self.gui.cap_processing_thread = CapDetectionThread(
                    captured_image,
                    self.gui.cap_detector,
                    self.gui.craft_detector,
                    self.gui.rotation_model,
                    self.gui.line_detector,
                    self.gui.ocr_model,
                    getattr(self.gui, 'faded_text_yolo_model', None),
                    bottle_type=bottle_type
                )
                self.gui.cap_processing_thread.result_ready.connect(self.on_cap_processing_complete)
                self.gui.cap_processing_thread.start()
                print("🔇 SILENT PROCESS: Cap processing thread started")
            else:
                print("❌ SILENT CAPTURE: Failed to capture Sentech image")
                # ถ้าถ่ายภาพฝาแบบเงียบไม่สำเร็จ ให้แสดงผลเฉพาะขวดจากคิว
                self._flush_silent_bottle_only_if_pending(reason="sentech_capture_failed")
                self.silent_mode = False
        except Exception as e:
            print("❌ SILENT CAPTURE ERROR: %s" % e)
            self._flush_silent_bottle_only_if_pending(reason="sentech_silent_exception")
            self.silent_mode = False
    
    def _flush_silent_bottle_only_if_pending(self, reason: str = ""):
        """
        ใช้ในโหมด SILENT เมื่อฝาไม่สามารถประมวลผลได้
        ถ้ามีผลขวดรออยู่ (pending_bottle_result)
        - ถ้ายังรอ M401 อยู่ (m600_reset_pending=True) → เก็บเข้า queue เพื่อรอแสดงตอน M401 ON
        - ถ้า M401 ผ่านไปแล้ว (m600_reset_pending=False) → แสดงผลขวดทันทีจาก queue ปัจจุบัน
        """
        try:
            bottle_handlers = getattr(self.gui, "bottle_handlers", None)
            if bottle_handlers is None:
                return
            bottle_result = getattr(bottle_handlers, "pending_bottle_result", None)
            if bottle_result is None:
                return
            
            modbus_thread = getattr(self.gui, "modbus_thread", None)
            if modbus_thread:
                mt = modbus_thread
                if getattr(mt, "waiting_for_late_result", False):
                    mt.waiting_for_late_result = False
                    print(f"🔇 SILENT MODE: ผลขวดมาสายหลัง M401 (bottle-only, {reason}) — แสดงและส่งผลทันที")
                    mt.result_ready_to_display.emit(bottle_result, {})
                elif getattr(mt, "m600_reset_pending", False):
                    # ยังรอ M401 อยู่ — เก็บผลไว้ใน queue แล้วให้ M401 รอบถัดไปดึงมาแสดง
                    mt.append_pending_result(bottle_result, {})
                    print(f"🔇 SILENT MODE: Bottle-only result queued ({reason}) (queue size: {len(mt.pending_results_queue)})")
                else:
                    # M401 ผ่านไปแล้ว แต่ผลขวดเพิ่งเสร็จ → แสดงทันทีจาก queue ปัจจุบัน
                    if mt.pending_m301_count > 0:
                        mt.pending_m301_count -= 1
                    print(f"🔇 SILENT MODE: Bottle-only result ready ({reason}) — display immediately (pending_m301_count={mt.pending_m301_count})")
                    mt.result_ready_to_display.emit(bottle_result, {})
            else:
                print(f"⚠️ SILENT MODE: Modbus thread not available while flushing bottle-only result ({reason})")
            
            bottle_handlers.pending_bottle_result = None
            if hasattr(self, "pending_cap_result"):
                self.pending_cap_result = None
        except Exception as e:
            print(f"❌ SILENT MODE: Error flushing bottle-only result: {e}")
    
    def display_result_from_queue(self, result):
        """Display result from queue (after D6004=200)"""
        try:
            print("📋 DISPLAY FROM QUEUE: Displaying cap result from queue")
            
            # Store result
            self.gui.current_cap_result = result
            
            # อัปเดตภาพฝาบนแท็บฝาและหน้าหลักให้ตรงกับผลจากคิว (อย่าใช้ or กับ numpy array — ใช้ is None แทน)
            cap_img = result.get('image')
            if cap_img is None:
                cap_img = result.get('processed_image')
            if cap_img is not None:
                self.gui.current_sentech_image = cap_img
                self.display_sentech_image(cap_img)
            QtWidgets.QApplication.processEvents()

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
        """Display Sentech image in label. กล้อง Sentech ตั้งเป็น Mono8 → ภาพเป็น grayscale 2D (H,W). ย่อก่อนแสดง (max 800px) เพื่อลด CPU/memory."""
        try:
            if image is not None:
                # Sentech = Mono8 → shape (H, W). ย่อก่อน cvtColor/QImage เพื่อลดทรัพยากร
                h, w = image.shape[:2]
                max_display_edge = 800
                if max(h, w) > max_display_edge:
                    scale = max_display_edge / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cuda_resize(image, (new_w, new_h))
                    h, w = image.shape[:2]
                # Handle grayscale image (Mono8 from SENTECH camera)
                if len(image.shape) == 2:
                    rgb_image = cuda_cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif len(image.shape) == 3:
                    rgb_image = cuda_cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = image
                h, w, c = rgb_image.shape
                bytes_per_line = c * w
                if not rgb_image.flags['C_CONTIGUOUS']:
                    rgb_image = np.ascontiguousarray(rgb_image)
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
            else:
                self.gui.sentech_image_label.setText("ไม่สามารถโหลดภาพได้")

        except Exception as e:
            print(f"❌ Error displaying image: {e}")
            self.gui.sentech_image_label.setText(f"Error: {str(e)}")
    
    def process_cap_detection(self, started_from_parallel=False):
        """Process cap detection on Sentech image or selected image file
        ใช้ค่ากลางที่วิเคราะห์ได้ (brightness: 33) โดยตรง ไม่ต้องรอ bottle_type
        started_from_parallel=True: ถูกเรียกจาก process_current_image โหมด parallel (ข้าม UI เบื้องต้น)
        """
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
        self.gui.status_handlers.update_cap_detection_status("Processing...", True)
        
        # ตรวจสอบเฉพาะ cap_detector เท่านั้น (โมเดลหลักสำหรับการตรวจจับฝา)
        if not self.gui.cap_detector:
            QMessageBox.warning(self.gui, "ข้อผิดพลาด", "โมเดลตรวจจับฝายังไม่ได้โหลดเสร็จ")
            return
        
        # กันไม่ให้เริ่มประมวลผลฝาซ้อน (จะทำให้ add history สองครั้ง)
        cap_thread = getattr(self.gui, 'cap_processing_thread', None)
        if cap_thread is not None and cap_thread.isRunning():
            print("ℹ️ CAP PROCESS: ข้าม — กำลังประมวลผลฝาอยู่แล้ว")
            return
        
        print(f"🔄 CAP PROCESS: Starting cap detection for image shape: {image_to_process.shape} (from {image_source})")
        print(f"🔍 CAP PROCESS: ใช้ค่ากลางที่วิเคราะห์ได้ (brightness: 33) สำหรับการปรับภาพ")
        
        if not started_from_parallel:
            # แสดงผล GUI สำหรับการประมวลผลฝา (ถ้า loading ยังไม่แสดงอยู่แล้ว เช่น มาจากขวด → ฝา ไม่ต้อง show ซ้ำ)
            self.display_cap_processing_ui()
            if getattr(self.gui, '_global_loading_count', 0) == 0:
                self.gui.show_global_loading("Processing cap...")
            else:
                self.gui.global_loading_label.setText("🔄 Processing cap...")
            self.gui.cap_progress_bar.setVisible(True)
            self.gui.cap_progress_bar.setValue(0)
            self.gui.cap_progress_bar.setFormat("Starting...")
            self.gui.status_label.setText('🔄 Processing cap...')
            self.gui.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            self.gui.btn_process_cap.setEnabled(False)
            self.gui.btn_save_sentech_image.setEnabled(False)
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
    
    def _on_cap_retry_sentech_done(self, sentech_image):
        """หลัง retry ถ่ายภาพฝา (เมื่อประมวลผลฝา error): ถ้าได้ภาพ เริ่มประมวลผลฝาใหม่ ถ้าไม่ได้ แสดง error และส่งผลขวดอย่างเดียว"""
        if sentech_image is not None:
            self.gui.current_sentech_image = sentech_image
            self.silent_mode = True
            self.process_cap_detection(started_from_parallel=True)
            return
        # ถ่ายภาพฝาไม่ได้
        print("❌ SILENT CAP RETRY: ถ่ายภาพฝาไม่ได้")
        QMessageBox.warning(
            self.gui,
            "ภาพฝาไม่มา",
            "ถ่ายภาพฝาใหม่ไม่ได้\n\nจะส่งเฉพาะผลขวด (ไม่มีผลฝา)"
        )
        if hasattr(self.gui, 'bottle_handlers') and self.gui.bottle_handlers.pending_bottle_result is not None:
            bottle_result = self.gui.bottle_handlers.pending_bottle_result
            self.gui.bottle_handlers.pending_bottle_result = None
            if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                mt = self.gui.modbus_thread
                if getattr(mt, 'waiting_for_late_result', False):
                    mt.waiting_for_late_result = False
                    mt.result_ready_to_display.emit(bottle_result, None)
                elif mt.m600_reset_pending:
                    mt.append_pending_result(bottle_result, None)
                else:
                    mt.result_ready_to_display.emit(bottle_result, None)
        self._silent_cap_process_retry = 0
        self.silent_mode = False
    
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
        self.gui.cap_progress_bar.setFormat(f"Processing… ({value}%)")
    
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
                self.gui.cap_detection_text.setText("🔄 Processing cap… please wait")
                if hasattr(self.gui, 'home_cap_detection_text'):
                    self.gui.home_cap_detection_text.setText("🔄 Processing cap… please wait")
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
                    self.gui.cap_detection_text.setText(f"✅ Cap processing finished — sending signal {self.gui.current_bottle_type}")
                    if hasattr(self.gui, 'home_cap_detection_text'):
                        self.gui.home_cap_detection_text.setText(f"✅ Cap processing finished — sending signal {self.gui.current_bottle_type}")
                else:
                    self.gui.cap_detection_text.setText("✅ Cap processing finished")
                    if hasattr(self.gui, 'home_cap_detection_text'):
                        self.gui.home_cap_detection_text.setText("✅ Cap processing finished")
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
        """Handle cap detection processing completion — defer heavy UI work เพื่อไม่ให้ main thread ค้าง"""
        print("✅ CAP PROCESS COMPLETE: Cap detection finished")
        QTimer.singleShot(0, lambda r=result: self._on_cap_processing_complete_work(r))

    def _on_cap_processing_complete_work(self, result):
        """ทำงานจริงหลัง defer (แสดงผล/อัปเดต UI) — ไม่เรียก processEvents เพื่อลดการค้าง"""
        if getattr(self.gui, '_stopping_processing', False):
            return
        self.gui.cap_progress_bar.setVisible(False)
        self.gui.hide_global_loading()
        
        # Update performance stats
        self.gui.total_images_processed_count += 1
        self.gui.status_handlers.update_performance_stats(images_processed=self.gui.total_images_processed_count)
        
        # Reset button states
        self.gui.btn_process_cap.setEnabled(True)
        self.gui.btn_save_sentech_image.setEnabled(True)
        self.gui.btn_stop_processing.setEnabled(False)
        
        # Check if silent mode - store result in queue instead of displaying
        if self.silent_mode:
            # ฝาจาง (faded_text_detected) = มีผลฝา (NG) ใช้ผลนี้รวมกับขวด ส่ง M140 ได้ — ไม่ retry
            # error อื่นๆ เท่านั้นที่ retry ถ่ายใหม่
            if "error" in result and result.get("error") != "faded_text_detected":
                self._silent_cap_process_retry = getattr(self, '_silent_cap_process_retry', 0) + 1
                if self._silent_cap_process_retry <= 2:
                    print(f"🔇 SILENT: ประมวลผลฝา error — ถ่ายใหม่และประมวลใหม่ ครั้งที่ {self._silent_cap_process_retry}/2 (เหตุผล: {result.get('error', '')})")
                    self.gui.bottle_handlers.capture_sentech_then_callback(self._on_cap_retry_sentech_done)
                    return
                err_msg = result.get("error", "ไม่ทราบสาเหตุ")
                print(f"❌ SILENT CAP PROCESS ERROR: {err_msg} (หลังลอง 2 ครั้ง)")
                QMessageBox.warning(
                    self.gui,
                    "ประมวลผลฝาไม่สำเร็จ",
                    f"หลังลองถ่ายใหม่และประมวลใหม่ 2 ครั้ง ยังไม่สำเร็จ\n\nเหตุผล: {err_msg}\n\nจะส่งเฉพาะผลขวด (ไม่มีผลฝา)"
                )
                if hasattr(self.gui, 'bottle_handlers') and self.gui.bottle_handlers.pending_bottle_result is not None:
                    bottle_result = self.gui.bottle_handlers.pending_bottle_result
                    self.gui.bottle_handlers.pending_bottle_result = None
                    if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                        mt = self.gui.modbus_thread
                        if getattr(mt, 'waiting_for_late_result', False):
                            mt.waiting_for_late_result = False
                            mt.result_ready_to_display.emit(bottle_result, None)
                        elif mt.m600_reset_pending:
                            mt.append_pending_result(bottle_result, None)
                        else:
                            mt.result_ready_to_display.emit(bottle_result, None)
                self._silent_cap_process_retry = 0
                self.silent_mode = False
                return
            if "error" in result and result.get("error") == "faded_text_detected":
                print("🔇 SILENT MODE: ผลฝาจาง (NG) — ใช้ผลนี้รวมกับขวด ส่ง M110 + M140 (ไม่ retry)")
            self._silent_cap_process_retry = 0  # สำเร็จหรือฝาจาง (ใช้ผลได้)
            print("🔇 SILENT MODE: Storing cap result in queue (no UI display, no Modbus signals)")
            self.pending_cap_result = result
            # Check if bottle result is also ready
            if hasattr(self.gui, 'bottle_handlers') and self.gui.bottle_handlers.pending_bottle_result is not None:
                bottle_result = self.gui.bottle_handlers.pending_bottle_result
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    mt = self.gui.modbus_thread
                    if getattr(mt, 'waiting_for_late_result', False):
                        # M401 มาขณะคิวว่างแล้ว — ผลมาสาย แสดง+ส่งทันที
                        mt.waiting_for_late_result = False
                        print("🔇 SILENT MODE: ผลขวด+ฝามาสายหลัง M401 — แสดงและส่งผลทันที")
                        mt.result_ready_to_display.emit(bottle_result, result)
                    elif mt.m600_reset_pending:
                        # ถ้าขวดเคยใส่ (bottle, None) ไว้ในคิวแล้ว ให้อัปเดตเป็น (bottle, cap) แทนการ append ซ้ำ
                        q = mt.pending_results_queue
                        updated = False
                        for i, (b, c) in enumerate(q):
                            if c is None and b is bottle_result:
                                q[i] = (bottle_result, result)
                                print(f"🔇 SILENT MODE: อัปเดตผลฝาในคิว (แทนที่รายการขวดอย่างเดียว) (queue size: {len(q)})")
                                updated = True
                                break
                        if not updated:
                            print("🔇 SILENT MODE: ผลขวด+ฝาพร้อม — เก็บเข้าคิว รอ M401 reset ก่อน")
                            mt.append_pending_result(bottle_result, result)
                    else:
                        print("🔇 SILENT MODE: ผลขวด+ฝาพร้อม — M401 ไปแล้ว ส่งผล/แสดงทันที")
                        mt.result_ready_to_display.emit(bottle_result, result)
                    self.pending_cap_result = None
                    self.gui.bottle_handlers.pending_bottle_result = None
                else:
                    print("⚠️ SILENT MODE: Modbus thread not available")
            else:
                # ขวดถูกส่งไปแล้ว (ผลมาสาย) — ฝาเสร็จทีหลัง ส่งผลฝาเดียว (แสดงผลฝา + M140/M600 หรือ validate ฝาผ่าน)
                print("🔇 SILENT MODE: ฝาเสร็จทีหลัง (ขวดส่งไปแล้ว) — แสดงผลฝาและส่งสัญญาณ Modbus")
                self.display_cap_detection_results(result)
                self.gui.current_cap_result = result
                if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                    if "error" in result and result.get("error") == "faded_text_detected":
                        if self.gui.modbus_thread.on_m140():
                            self.gui.status_handlers.update_coil_lamp("m140", True)
                        self.gui.modbus_thread.on_m600()
                        self.gui.status_handlers.update_coil_lamp("m600", True)
                        print("🔇 SILENT: ส่ง M140, M600 (ฝาจาง)")
                    else:
                        is_valid = self.validate_cap_result(result)
                        if is_valid and self.gui.current_bottle_type in ["M100", "M110", "M120"]:
                            self.gui.modbus_handlers.on_bottle_type_after_cap_validation()
                            print("🔇 SILENT: ฝาผ่าน — ส่งสัญญาณตาม validation")
                        else:
                            if self.gui.modbus_thread.on_m140():
                                self.gui.status_handlers.update_coil_lamp("m140", True)
                            self.gui.modbus_thread.on_m600()
                            self.gui.status_handlers.update_coil_lamp("m600", True)
                            print("🔇 SILENT: ฝาไม่ผ่าน — ส่ง M140, M600")
                self.pending_cap_result = None
            
            # Reset silent mode
            self.silent_mode = False
            return
        
        # ตรวจสอบว่าเป็นข้อความจางหรือไม่ - แสดงผลลัพธ์ก่อน (ภาพและค่า area) และส่ง Modbus เหมือนขวด NG
        if "error" in result and result["error"] == "faded_text_detected":
            print("❌ CAP PROCESS COMPLETE: ตรวจพบข้อความจาง - แสดงผลลัพธ์ (ภาพและค่า area)")
            
            # เก็บผลลัพธ์ไว้
            self.gui.current_cap_result = result
            
            # ส่ง Modbus เหมือนขวด NG (M140, M600)
            if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                m140_ok = self.gui.modbus_thread.on_m140()
                if m140_ok:
                    self.gui.status_handlers.update_coil_lamp("m140", True)
                    m600_ok = self.gui.modbus_thread.on_m600()
                    if m600_ok:
                        self.gui.status_handlers.update_coil_lamp("m600", True)
                        self.gui.status_label.setText('❌ ฝาจาง → M140, M600 ส่งแล้ว')
                        # เก็บสถานะ Modbus ที่ส่งไปแล้ว
                        self.gui.last_sent_modbus_signal = "M140"
                        # เก็บผลที่ส่งล่าสุด
                        bottle_result = getattr(self.gui, 'current_result', None)
                        if bottle_result is not None:
                            self.gui._last_sent_bottle_cap_result = (bottle_result, result)
                    else:
                        self.gui.status_label.setText('❌ ฝาจาง แต่ไม่สามารถ ON M600 ได้')
                else:
                    self.gui.status_label.setText('❌ ฝาจาง แต่ไม่สามารถON M140 ได้')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            else:
                self.gui.status_label.setText('❌ ฝาจาง แต่ Modbus ไม่พร้อม')
                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            
            # Update status tab
            self.gui.status_handlers.update_cap_detection_status("ข้อความจาง - แสดงผลลัพธ์", False)
            
            # Display cap detection results (แสดงผลลัพธ์ฝาที่เจอข้อความจาง - รวมภาพและค่า area)
            self.display_cap_detection_results(result)
            
            faded_text_result = result.get('faded_text_result', {})
            status = faded_text_result.get('status', 'unknown')
            result_label = faded_text_result.get('result', '')
            score = faded_text_result.get('score')
            total_area = faded_text_result.get('total_area', 0)
            num_chars = faded_text_result.get('num_chars', 0)
            
            detail = f"result: {result_label}"
            if score is not None:
                detail += f", score: {score}"
            if not (hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread):
                self.gui.status_label.setText(f'❌ ตรวจพบฝาจาง - {detail}')
            print(f"📊 CAP PROCESS COMPLETE: ฝาจาง - {detail}, status: {status}")
            
            if hasattr(self.gui, 'add_to_history'):
                if hasattr(self.gui, 'current_result') and self.gui.current_result:
                    bottle_result = self.gui.current_result
                else:
                    bottle_result = {
                        'image': None,
                        'bottle_type': 'ฝาจาง (NG Fade)',
                        'combined_ocr_text': f'ฝาจาง - {result_label}' + (f', score: {score}' if score is not None else '')
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
                    # Good = ฝาผ่าน + ขวดผ่าน เท่านั้น ถึงส่ง Modbus ปกติ (ON M100/M110/M120)
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
                                self.gui.status_label.setText('❌ ขวดไม่ผ่าน (Defect NG) → M140, M600')
                                self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                                # เก็บสถานะ Modbus ที่ส่งไปแล้ว
                                self.gui.last_sent_modbus_signal = "M140"
                                # เก็บผลที่ส่งล่าสุด
                                if bottle_result is not None:
                                    self.gui._last_sent_bottle_cap_result = (bottle_result, result)
                    else:
                        print(f"✅ CAP VALIDATION PASSED: ฝาผ่าน + ขวดผ่าน - Sending Modbus signals")
                        print(f"✅ CAP PROCESS COMPLETE: Calling on_bottle_type_after_cap_validation()")
                        # เก็บผลที่ส่งล่าสุด — จะถูก copy ไป last_bottle_cap_result_for_reuse ตอน M403/M404/M405/M406 ON (แถวเต็ม)
                        bottle_result = getattr(self.gui, 'current_result', None) or getattr(self.gui.bottle_handlers, 'pending_bottle_result', None)
                        if bottle_result is not None:
                            self.gui._last_sent_bottle_cap_result = (bottle_result, result)
                            # เก็บสถานะ Modbus ที่ส่งไปแล้ว
                            self.gui.last_sent_modbus_signal = self.gui.current_bottle_type
                        if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                            print(f"✅ CAP PROCESS COMPLETE: Modbus thread is available - Sending signals")
                            self.on_bottle_type_after_cap_validation()
                        else:
                            print(f"❌ CAP PROCESS COMPLETE: Modbus thread is not available!")
                            self.gui.status_label.setText(f'❌ Modbus thread ไม่พร้อม - ไม่สามารถส่งสัญญาณ {self.gui.current_bottle_type} ได้')
                            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                else:
                    # ฝาไม่ผ่าน: ถ้าเป็นเพราะไม่ตรงกับวันที่ที่เลือก ให้แสดงสถานะนั้น ไม่ใช่แค่ "NG"
                    fail_reason = getattr(self.gui, '_last_cap_fail_reason', None)
                    status_text = '❌ ไม่ตรงกับวันที่เลือก → M140, M600 (รอ reset)' if fail_reason == "ไม่ตรงกับวันที่เลือก" else '❌ ฝาไม่ผ่าน → ON M140 + M600 (รอ reset)'
                    print(f"❌ CAP VALIDATION FAILED: Cap result is invalid for {self.gui.current_bottle_type} - Sending M140 instead")
                    # ON M140 (ฝาไม่ผ่าน - ไม่ตรงกับฟอร์ม หรือไม่ตรงกับวันที่เลือก)
                    if hasattr(self.gui, 'modbus_thread') and self.gui.modbus_thread:
                        m140_success = self.gui.modbus_thread.on_m140()
                        if m140_success:
                            self.gui.status_handlers.update_coil_lamp("m140", True)
                            print("❌ ON M140 (ฝาไม่ผ่าน)" + (" - ไม่ตรงกับวันที่เลือก" if fail_reason == "ไม่ตรงกับวันที่เลือก" else " - ไม่ตรงกับฟอร์ม"))
                            
                            # ON M600 หลังจาก ON M140 (เหมือน M100/M110/M120)
                            print("⏳ ON M600 และรอ D6004=200 เพื่อ reset ทั้งหมด...")
                            m600_success = self.gui.modbus_thread.on_m600()
                            if m600_success:
                                self.gui.status_handlers.update_coil_lamp("m600", True)
                            
                            # อัปเดตสถานะ (แสดง "ไม่ตรงกับวันที่เลือก" ไม่ใช่ "NG" เมื่อเป็นกรณีคัดกรองวัน)
                            self.gui.status_label.setText(status_text)
                            self.gui.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                            
                            # เก็บสถานะ Modbus ที่ส่งไปแล้ว
                            self.gui.last_sent_modbus_signal = "M140"
                            # เก็บผลที่ส่งล่าสุด
                            bottle_result = getattr(self.gui, 'current_result', None) or getattr(self.gui.bottle_handlers, 'pending_bottle_result', None)
                            if bottle_result is not None:
                                self.gui._last_sent_bottle_cap_result = (bottle_result, result)
                            
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


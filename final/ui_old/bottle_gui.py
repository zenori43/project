
# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QWidget, QGroupBox, QTextEdit, QGridLayout, QListWidget, QProgressBar,
    QSplitter, QMessageBox, QCheckBox, QScrollArea, QTabWidget,
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QFrame, QApplication,
    QDateTimeEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os
import time
from typing import List, Dict
import sys
import easyocr
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import Harvesters for SENTECH camera
try:
    from harvesters.core import Harvester
    HARVESTERS_AVAILABLE = True
    print("✅ Harvesters imported successfully")
except ImportError:
    HARVESTERS_AVAILABLE = False
    print("⚠️ Harvesters not available")
import matplotlib.pyplot as plt
import numpy as np
import threading
from pymodbus.client import ModbusTcpClient
from difflib import SequenceMatcher

# Import ultralytics for YOLO model
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ Ultralytics YOLO imported successfully")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"⚠️ Ultralytics YOLO not available: {e}")

# =============================================================================
# MODEL PATHS - แก้ไขที่นี่เพื่อเปลี่ยน path ของโมเดล
# =============================================================================
CAP_MODEL_PATH = r"/home/nvidia/Desktop/final_boss/backupsdcard/final/cap.pt"
ROTATION_MODEL_PATH = r"/home/nvidia/Desktop/final_boss/backupsdcard/final/text_rotation_model.pth"
CRAFT_MODEL_PATH = r"/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch/craft_mlt_25k.pth"
CRAFT_REFINER_PATH = r"/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch/craft_refiner_CTW1500.pth"
OCR_MODEL_PATH = r"/home/nvidia/Desktop/final_boss/backupsdcard/fix10.pth"

# =============================================================================
# FADED TEXT DETECTION CONFIGURATION
# =============================================================================
FADED_TEXT_CONFIG = {
    'MIN_AREA': -1,           # เกณฑ์ตัด noise ของ component
    'AREA_THRESH': -1,      # เกณฑ์ตัดสิน faded/normal จาก total_area
    'CIRCLE_FALLBACK_MARGIN': 6,
    'HOUGH_CIRCLES_PARAMS': {
        'dp': 1.2,
        'minDist': 50,
        'param1': 100,
        'param2': 30,
        'minRadius': 90,
        'maxRadius': 100
    },
    'CANNY_PARAMS': {
        'low_threshold': 100,
        'high_threshold': 200
    },
    'GAUSSIAN_BLUR_PARAMS': {
        'kernel_size': (9, 9),
        'sigma': 2
    }
}

# Fix for PIL ANTIALIAS compatibility with newer Pillow versions
try:
    import PIL.Image
    if not hasattr(PIL.Image, 'ANTIALIAS'):
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
        print("✅ Applied PIL ANTIALIAS compatibility fix")
except Exception as e:
    print(f"Warning: Could not apply PIL compatibility fix: {e}")

# Import our simplified bottle detection module
from bottledetect import (
    detect_bottle_and_crop_type, 
    process_bottle_image_simple,
    get_type_crops_only,
    get_detection_summary,
    draw_detections_on_image,
    save_cropped_type
)

# Import USBCamera from external file
import sys
import os

# Add the path to usbcamra.py
import platform
if platform.system() == "Windows":
    usbcamra_path = r"C:\Users\Win 10 Home\Desktop\Myproject\straure_code\usbcamra.py"
else:
    usbcamra_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/final/usbcamra.py"

if usbcamra_path not in sys.path:
    sys.path.append(usbcamra_path)

try:
    from usbcamra import USBCamera
    print("✅ Successfully imported USBCamera from usbcamra.py")
except ImportError as e:
    print(f"❌ Failed to import USBCamera: {e}")
    print("Using fallback USBCamera class...")
    
    # Fallback USBCamera class if import fails
    class USBCamera:
        def __init__(self):
            self.cap = None
            self.is_running = False
            self._thread = None
            
        def open_camera(self):
            try:
                # Try camera index 1 first (as detected by diagnostic)
                self.cap = cv2.VideoCapture(1)
                if not self.cap.isOpened():
                    # Fallback to camera index 0
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        return False
                return True
            except:
                return False
                
        def close_camera(self):
            if self.cap:
                self.cap.release()
                self.cap = None
                
        def capture_image(self):
            """
            ถ่ายภาพหนึ่งเฟรม - ดึงเฟรมล่าสุดมาใช้
            """
            if not self.cap or not self.cap.isOpened():
                return None
            
            # ล้าง buffer เพื่อดึงเฟรมล่าสุด
            for _ in range(10):  # ล้าง buffer 10 เฟรม
                self.cap.grab()
            
            # รอสักครู่เพื่อให้ได้เฟรมใหม่
            time.sleep(0.1)
            
            # ดึงเฟรมล่าสุด
            ret, frame = self.cap.read()
            if ret and frame is not None:
                print(f"📸 Captured latest frame: {frame.shape}")
                return frame
            
            return None

# Add CRAFT-pytorch to Python path
import sys
sys.path.append('/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch')
# Add PyQt5 system path for Python 3.9 compatibility
sys.path.insert(0, '/usr/lib/python3/dist-packages')

# Import cap detection modules
try:
    from capmodel import initialize_detector, detect_caps_from_path
    from rotationCRAFT import initialize_detector as initialize_craft_detector, detect_text_and_rotate_from_array
    from rotationmodel import (
        initialize_model, 
        get_rotated_image_from_array,
        process_craft_rotated_from_array,
        get_cropped_from_craft_rotated_array,
        process_craft_rotated_with_verification_from_array,
        get_cropped_with_verification_from_craft_rotated_array
    )
    from craft_line_detection import (
        initialize_detector as initialize_line_detector,
        detect_lines_from_rotation_result,
        get_cropped_lines_from_rotation_result
    )
    from deep_ocr import DeepOCRModel, initialize_ocr_model, recognize_text_from_craft_lines
    CAP_DETECTION_AVAILABLE = True
    print("✅ Cap detection modules imported successfully")
except ImportError as e:
    CAP_DETECTION_AVAILABLE = False
    print(f"⚠️ Cap detection modules not available: {e}")


class SentechCamera:
    """Sentech camera class for cap detection - using Harvesters only"""
    
    def __init__(self, gentl_path='/opt/sentech/lib/libstgentl.cti'):
        self._is_initialized = False
        self._h = None  # Harvester instance
        self._ia = None  # Image acquirer
        self._gentl_path = gentl_path
        self._connected = False
        
    @property
    def image(self):
        """Get latest captured image"""
        if self._ia is not None and self._connected:
            try:
                # Wait a bit for camera to be ready
                time.sleep(0.1)
                
                with self._ia.fetch() as buffer:
                    # Get image data
                    component = buffer.payload.components[0]
                    image_data = component.data
                    
                    # Get dimensions
                    height = component.height
                    width = component.width
                    
                    # Reshape image data
                    if len(image_data.shape) == 1:
                        # 1D array - reshape to 2D
                        # For Mono8, reshape to 2D grayscale
                        image = image_data.reshape((height, width))
                    else:
                        image = image_data
                    
                    return image
            except Exception as e:
                print(f"❌ Error getting image: {e}")
        return None
    
    @property 
    def callback_count(self):
        """Return callback count for compatibility"""
        return 1 if self._connected else 0

    def initialize(self):
        """Initialize Sentech camera using Harvesters only"""
        if self._is_initialized:
            return False
        
        if not HARVESTERS_AVAILABLE:
            print("❌ Harvesters not available")
            return False
            
        try:
            print("🔧 Initializing SENTECH camera with Harvesters...")
            
            # Create Harvester instance
            self._h = Harvester()
            
            # Add GenTL Producer
            self._h.add_file(self._gentl_path)
            self._h.update()
            
            if len(self._h.device_info_list) == 0:
                print("❌ No SENTECH devices found")
                return False
            
            print(f"✅ Found {len(self._h.device_info_list)} SENTECH device(s)")
            
            # Create image acquirer
            self._ia = self._h.create(0)
            
            # Configure camera (with error handling)
            try:
                # Try to configure camera parameters
                if hasattr(self._ia.device.node_map, 'Width'):
                    self._ia.device.node_map.Width.value = 2448
                if hasattr(self._ia.device.node_map, 'Height'):
                    self._ia.device.node_map.Height.value = 2048
                if hasattr(self._ia.device.node_map, 'PixelFormat'):
                    self._ia.device.node_map.PixelFormat.value = 'Mono8'
                if hasattr(self._ia.device.node_map, 'ExposureTime'):
                    self._ia.device.node_map.ExposureTime.value = 10000
                if hasattr(self._ia.device.node_map, 'Gain'):
                    self._ia.device.node_map.Gain.value = 1.0
                
                print("✅ Camera configured successfully")
            except Exception as config_error:
                print(f"⚠️ Camera configuration warning: {config_error}")
                print("📷 Using default camera settings")
            
            # Start acquisition
            self._ia.start()
            self._connected = True
            self._is_initialized = True
            
            print("✅ SENTECH camera initialized with Harvesters")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize SENTECH camera with Harvesters: {e}")
            return False
    
    def capture_image(self):
        """Capture image from Sentech camera using Harvesters only"""
        if not self._is_initialized:
            print("❌ Sentech camera not initialized")
            return None
        
        if self._ia is not None and self._connected:
            try:
                print("📸 Capturing image with Harvesters...")
                
                # Wait a bit for camera to be ready
                time.sleep(0.1)
                
                with self._ia.fetch() as buffer:
                    # Get image data
                    component = buffer.payload.components[0]
                    image_data = component.data
                    
                    # Get dimensions
                    height = component.height
                    width = component.width
                    
                    print(f"  Image data shape: {image_data.shape}")
                    print(f"  Image dimensions: {width}x{height}")
                    print(f"  Data type: {image_data.dtype}")
                    print(f"  Min value: {image_data.min()}, Max value: {image_data.max()}")
                    
                    # Reshape image data
                    if len(image_data.shape) == 1:
                        # 1D array - reshape to 2D
                        # For Mono8, reshape to 2D grayscale
                        image = image_data.reshape((height, width))
                    else:
                        image = image_data
                    
                    print(f"✅ Captured image: {image.shape}")
                    return image
                    
            except Exception as e:
                print(f"❌ Error capturing image with Harvesters: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("❌ Camera not connected")
            return None
    
    def cleanup(self):
        """Cleanup Sentech camera using Harvesters only"""
        try:
            if self._is_initialized:
                try:
                    if self._ia is not None:
                        # Stop acquisition
                        self._ia.stop()
                        print("🛑 Stopped Harvesters acquisition")
                        
                        # Destroy image acquirer
                        self._ia.destroy()
                        self._ia = None
                        print("🧹 Destroyed image acquirer")
                    
                    if self._h is not None:
                        # Reset Harvester
                        self._h.reset()
                        self._h = None
                        print("🧹 Reset Harvester")
                    
                    self._connected = False
                    self._is_initialized = False
                    print("✅ SENTECH camera cleaned up with Harvesters")
                    
                except Exception as e:
                    print(f"❌ Error cleaning up Harvesters: {e}")
                    
        except Exception as e:
            print(f"❌ Error in cleanup: {e}")

class CapDetectionThread(QThread):
    """Thread for processing cap detection to avoid GUI freezing"""
    result_ready = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    rotation_attempt_updated = pyqtSignal(int, int, object, object, object)  # attempt_num, total_attempts, rotated_image, ocr_results, format_valid
    
    def __init__(self, image, cap_detector, craft_detector, rotation_model, line_detector, ocr_model, faded_text_yolo_model=None):
        super().__init__()
        self.image = image
        self.cap_detector = cap_detector
        self.craft_detector = craft_detector
        self.rotation_model = rotation_model
        self.line_detector = line_detector
        self.ocr_model = ocr_model
        self.faded_text_yolo_model = faded_text_yolo_model
        
    def run(self):
        try:
            print("🔄 CAP DETECTION THREAD: Starting processing...")
            print(f"🔄 CAP DETECTION THREAD: Image shape: {self.image.shape}")
            self.status_updated.emit("กำลังประมวลผลฝา...")
            self.progress_updated.emit(5)
            
            # Save temporary image for processing
            temp_path = "temp_sentech_image.jpg"
            cv2.imwrite(temp_path, self.image)
            print("🔄 CAP DETECTION THREAD: Saved temp image")
            self.progress_updated.emit(10)
            
            # Step 1: Detect caps
            self.status_updated.emit("กำลังตรวจจับฝา...")
            self.progress_updated.emit(20)
            print("🔄 CAP DETECTION THREAD: Step 1 - ตรวจจับฝา")
            cap_result = self.cap_detector.detect_caps(temp_path)
            
            # Get the processed image from cap detector
            processed_image = self.cap_detector.last_processed_image
            if processed_image is None:
                raise ValueError("ไม่สามารถประมวลผลรูปภาพได้")
            
            self.progress_updated.emit(30)
            
            # Step 2: Crop detected regions and process each crop
            self.status_updated.emit("กำลังตัดภาพ...")
            self.progress_updated.emit(40)
            print("🔄 CAP DETECTION THREAD: Step 2 - ตัดภาพ")
            cropped_images = self.cap_detector.crop_detections(margin=20)
            
            result = {
                'cap_result': cap_result,
                'processed_image': processed_image,
                'cropped_images': cropped_images,
                'image': self.image,
                'image_shape': self.image.shape
            }
            
            if not cropped_images:
                self.status_updated.emit("ไม่พบฝา กำลังประมวลผลข้อความ...")
                self.progress_updated.emit(50)
                print("🔄 CAP DETECTION THREAD: ไม่พบฝา เริ่มประมวลผลข้อความ...")
                
                # Step 2a: Use CRAFT to detect text and rotate original image
                self.status_updated.emit("กำลังตรวจจับข้อความด้วย CRAFT...")
                self.progress_updated.emit(60)
                print("🔄 CAP DETECTION THREAD: Step 2a - ตรวจจับข้อความด้วย CRAFT")
                craft_result = self.craft_detector.detect_text_and_rotate(image_array=processed_image)
                
                if craft_result and 'rotated_image' in craft_result and craft_result['rotated_image'] is not None:
                    # CRAFT already rotated the image, now use rotation model on CRAFT's rotated image
                    craft_rotated_image = craft_result['rotated_image']
                    result['craft_rotated_image'] = craft_rotated_image.copy()
                    print("🔄 CAP DETECTION THREAD: CRAFT หมุนภาพสำเร็จ")
                    self.progress_updated.emit(65)
                    
                    # Step 2b: Use rotation model to process CRAFT rotated image (simple processing)
                    self.status_updated.emit("กำลังประมวลผลด้วย AI rotation...")
                    self.progress_updated.emit(70)
                    print("🔄 CAP DETECTION THREAD: Step 2b - ประมวลผลด้วย AI rotation")
                    print("🔄 CAP DETECTION THREAD: Calling process_craft_rotated_image_with_models...")
                    combined_result = self.rotation_model.process_craft_rotated_image_with_models(
                        craft_rotated_image, self.line_detector, self.ocr_model)
                    print(f"🔄 CAP DETECTION THREAD: Combined result type: {type(combined_result)}")
                    if combined_result:
                        print(f"🔄 CAP DETECTION THREAD: Combined result keys: {list(combined_result.keys()) if isinstance(combined_result, dict) else 'Not a dict'}")
                    
                    if combined_result and 'ai_rotated_image' in combined_result and combined_result['ai_rotated_image'] is not None:
                        rotated_image = combined_result['ai_rotated_image']
                        result['rotated_image'] = rotated_image
                        print("🔄 CAP DETECTION THREAD: AI rotation สำเร็จ")
                        self.progress_updated.emit(75)
                        
                        # Step 2c: Get cropped image from AI rotated image
                        if combined_result.get('cropped_image') is not None:
                            result['cropped_image'] = combined_result['cropped_image']
                            print("🔄 CAP DETECTION THREAD: ได้ภาพที่ครอปแล้ว")
                        
                        result['combined_result'] = combined_result
                        
                        # Step 3: Use results from combined_result (includes retry mechanism)
                        self.status_updated.emit("กำลังประมวลผลผลลัพธ์...")
                        self.progress_updated.emit(80)
                        print("🔄 CAP DETECTION THREAD: Step 3 - ใช้ผลลัพธ์จาก retry mechanism")
                        
                        # Use line detection and OCR results from combined_result (includes retry)
                        if 'line_detection_result' in combined_result:
                            result['line_detection_result'] = combined_result['line_detection_result']
                            print("🔄 CAP DETECTION THREAD: ใช้ผลลัพธ์ line detection จาก retry")
                        
                        if 'ocr_results' in combined_result:
                            result['ocr_results'] = combined_result['ocr_results']
                            print("🔄 CAP DETECTION THREAD: ใช้ผลลัพธ์ OCR จาก retry")
                            self.progress_updated.emit(85)
                            
                            # Display retry information
                            if 'ai_rotation_angle' in combined_result:
                                rotation_angle = combined_result['ai_rotation_angle']
                                print(f"🔄 CAP DETECTION THREAD: มุมการหมุนที่ใช้: {rotation_angle}°")
                            
                            if 'score' in combined_result:
                                score = combined_result['score']
                                print(f"🔄 CAP DETECTION THREAD: คะแนนการหมุน: {score:.1f}")
                            
                            if 'rotation_attempt' in combined_result:
                                attempt = combined_result['rotation_attempt']
                                print(f"🔄 CAP DETECTION THREAD: ครั้งที่หมุน: {attempt}/5")
                            
                            self.progress_updated.emit(95)
                        else:
                            print("⚠️ CAP DETECTION THREAD: ไม่สามารถตรวจจับบรรทัดข้อความได้")
                    else:
                        print("⚠️ CAP DETECTION THREAD: AI rotation ไม่สำเร็จ")
                else:
                    print("⚠️ CAP DETECTION THREAD: CRAFT ไม่สามารถหมุนภาพได้")
            else:
                print(f"🔄 CAP DETECTION THREAD: พบฝา {len(cropped_images)} ฝา")
                self.progress_updated.emit(50)
                
                # Select only the bottommost cap (highest y-coordinate)
                bottommost_cap = None
                bottommost_index = 0
                max_y = 0
                
                # Find the cap with the highest y-coordinate (bottom of image)
                if cropped_images and 'detections' in cap_result:
                    detections = cap_result['detections']
                    print(f"🔄 CAP DETECTION THREAD: ตรวจสอบตำแหน่งฝา {len(detections)} ฝา")
                    
                    # Find the cap with the highest y-coordinate (bottom of image)
                    for i, detection in enumerate(detections):
                        if 'bbox' in detection:
                            # bbox format: [x1, y1, x2, y2]
                            bbox = detection['bbox']
                            y_center = (bbox[1] + bbox[3]) / 2  # Use center y-coordinate
                            print(f"🔄 CAP DETECTION THREAD: ฝาที่ {i+1} - y_center: {y_center:.1f}")
                            
                            if y_center > max_y:
                                max_y = y_center
                                bottommost_cap = cropped_images[i]
                                bottommost_index = i
                    
                    print(f"🔄 CAP DETECTION THREAD: เลือกฝาที่ {bottommost_index + 1} (y_center: {max_y:.1f}) เป็นฝาล่างสุด")
                elif cropped_images:
                    # Fallback: use the last cap if no bounding box info available
                    bottommost_cap = cropped_images[-1]
                    bottommost_index = len(cropped_images) - 1
                    print(f"🔄 CAP DETECTION THREAD: ไม่มีข้อมูลตำแหน่ง ใช้ฝาลำดับสุดท้าย จาก {len(cropped_images)} ฝา")
                
                print(f"🔄 CAP DETECTION THREAD: เลือกฝาที่ {bottommost_index + 1} (ฝาล่างสุด) จาก {len(cropped_images)} ฝา")
                
                # Process only the bottommost cap through the full pipeline
                all_cap_results = []
                if bottommost_cap is not None:
                    self.status_updated.emit(f"กำลังประมวลผลฝาที่ {bottommost_index + 1} (ฝาล่างสุด)...")
                    self.progress_updated.emit(50)
                    print(f"🔄 CAP DETECTION THREAD: ประมวลผลฝาที่ {bottommost_index + 1} (ฝาล่างสุด)")
                    
                    # Step 2a: Use CRAFT to detect text and rotate cropped cap
                    self.status_updated.emit(f"กำลังตรวจจับข้อความในฝาที่ {bottommost_index + 1} ด้วย CRAFT...")
                    craft_result = self.craft_detector.detect_text_and_rotate(image_array=bottommost_cap)
                    
                    cap_result = {
                        'cap_index': bottommost_index,
                        'original_crop': bottommost_cap,
                        'craft_result': craft_result
                    }
                    
                    # Step 2a.5: Detect faded text in the cropped cap BEFORE CRAFT rotation
                    self.status_updated.emit(f"กำลังตรวจสอบรอยจางในฝาที่ {bottommost_index + 1}...")
                    print(f"🔄 CAP DETECTION THREAD: Step 2a.5 - ตรวจสอบรอยจางในฝาที่ {bottommost_index + 1} (ก่อน CRAFT)")
                    faded_text_result = detect_faded_text_in_cap(
                        bottommost_cap, 
                        yolo_model=None,  # ไม่ใช้ YOLO model ใหม่ ใช้รูปที่ crop แล้วโดยตรง
                        show_debug=True  # เปิด debug mode เพื่อแสดงภาพใน UI
                    )
                    cap_result['faded_text_result'] = faded_text_result
                    print(f"🔄 CAP DETECTION THREAD: ตรวจสอบรอยจางฝาที่ {bottommost_index + 1} สำเร็จ - สถานะ: {faded_text_result['status']}")
                    
                    # หยุดประมวลผลเมื่อเจอข้อความจาง
                    if faded_text_result.get('status') == 'faded':
                        print("❌ CAP DETECTION THREAD: ตรวจพบข้อความจาง - หยุดประมวลผล")
                        print("🚨 CAP DETECTION THREAD: ส่งสัญญาณ M140 (ฝาไม่ผ่าน)")
                        print("🚀 CAP DETECTION THREAD: ส่งสัญญาณ M600 (ประมวลผลเสร็จสิ้น)")
                        
                        # ส่งสัญญาณ M140 (ฝาไม่ผ่าน)
                        if hasattr(self, 'modbus_thread') and self.modbus_thread:
                            self.modbus_thread.on_m140()
                            print("✅ M140 ส่งสัญญาณเรียบร้อย")
                            
                            # ส่งสัญญาณ M600 (ประมวลผลเสร็จสิ้น)
                            self.modbus_thread.on_m600()
                            print("✅ M600 ส่งสัญญาณเรียบร้อย")
                        
                        # สร้างผลลัพธ์ที่บ่งบอกว่าเป็นข้อความจาง
                        result = {
                            'error': 'faded_text_detected',
                            'message': 'ตรวจพบข้อความจาง - หยุดประมวลผล',
                            'faded_text_result': faded_text_result,
                            'cap_processing_results': [cap_result]
                        }
                        
                        self.result_ready.emit(result)
                        return
                    
                    if craft_result and 'rotated_image' in craft_result and craft_result['rotated_image'] is not None:
                        craft_rotated_image = craft_result['rotated_image']
                        cap_result['craft_rotated_image'] = craft_rotated_image
                        print(f"🔄 CAP DETECTION THREAD: CRAFT หมุนฝาที่ {bottommost_index + 1} สำเร็จ")
                        
                        # Step 2b: Use rotation model to process CRAFT rotated cap (simple processing)
                        self.status_updated.emit(f"กำลังประมวลผลฝาที่ {bottommost_index + 1} ด้วย AI rotation...")
                        combined_result = self.rotation_model.process_craft_rotated_image_with_models(
                            craft_rotated_image, self.line_detector, self.ocr_model)
                        
                        if combined_result and 'ai_rotated_image' in combined_result and combined_result['ai_rotated_image'] is not None:
                            cap_result['ai_rotated_image'] = combined_result['ai_rotated_image']
                            cap_result['combined_result'] = combined_result
                            print(f"🔄 CAP DETECTION THREAD: AI rotation ฝาที่ {bottommost_index + 1} สำเร็จ")
                            
                            # Step 3: Detect text lines from rotation result
                            self.status_updated.emit(f"กำลังตรวจจับบรรทัดข้อความในฝาที่ {bottommost_index + 1}...")
                            line_detection_result = detect_lines_from_rotation_result(combined_result)
                            cap_result['line_detection_result'] = line_detection_result
                            
                            if line_detection_result and 'error' not in line_detection_result:
                                print(f"🔄 CAP DETECTION THREAD: ตรวจจับบรรทัดข้อความฝาที่ {bottommost_index + 1} สำเร็จ")
                                
                                # Step 4: Perform OCR on detected lines
                                self.status_updated.emit(f"กำลังอ่านข้อความในฝาที่ {bottommost_index + 1} ด้วย OCR...")
                                ocr_results = recognize_text_from_craft_lines(line_detection_result)
                                cap_result['ocr_results'] = ocr_results
                                print(f"🔄 CAP DETECTION THREAD: OCR ฝาที่ {bottommost_index + 1} สำเร็จ - อ่านได้ {len(ocr_results)} รายการ")
                            else:
                                print(f"⚠️ CAP DETECTION THREAD: ไม่สามารถตรวจจับบรรทัดข้อความฝาที่ {bottommost_index + 1} ได้")
                        else:
                            print(f"⚠️ CAP DETECTION THREAD: AI rotation ฝาที่ {bottommost_index + 1} ไม่สำเร็จ")
                    else:
                        print(f"⚠️ CAP DETECTION THREAD: CRAFT ไม่สามารถหมุนฝาที่ {bottommost_index + 1} ได้")
                    
                    all_cap_results.append(cap_result)
                
                result['cap_processing_results'] = all_cap_results
                self.progress_updated.emit(90)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
                print("🔄 CAP DETECTION THREAD: ลบไฟล์ชั่วคราวแล้ว")
            except:
                pass
            
            # Final progress update
            self.progress_updated.emit(100)
            self.status_updated.emit("ประมวลผลเสร็จสิ้น")
            
            print("🔄 CAP DETECTION THREAD: Emitting result...")
            self.result_ready.emit(result)
                
        except Exception as e:
            print(f"❌ CAP DETECTION THREAD: Error: {e}")
            error_result = {
                'error': str(e),
                'image': self.image,
                'image_shape': self.image.shape if self.image is not None else None
            }
            self.result_ready.emit(error_result)

# Initialize EasyOCR reader for Thai and English
CUDA_AVAILABLE = torch.cuda.is_available()
try:
    ocr_reader = easyocr.Reader(["th", "en"], gpu=CUDA_AVAILABLE)
    print(f"✅ EasyOCR initialized successfully (GPU: {CUDA_AVAILABLE})")
except Exception as e:
    print(f"❌ Error initializing EasyOCR: {e}")
    ocr_reader = None

def fuzzy_match(text, target, threshold=0.7):
    """
    Fuzzy matching function using SequenceMatcher
    
    Args:
        text: text to compare
        target: target text to match against
        threshold: similarity threshold (0.0 to 1.0)
        
    Returns:
        bool: True if similarity >= threshold
    """
    try:
        similarity = SequenceMatcher(None, text.lower(), target.lower()).ratio()
        return similarity >= threshold
    except:
        return False

def check_bottle_type(ocr_text, selected_tastes=None):
    """
    Check bottle type using keyword matching
    
    Args:
        ocr_text: combined OCR text
        selected_tastes: list of selected taste types to check against
        
    Returns:
        str: M100, M110, M120, M130, or None
    """
    if not ocr_text:
        return None
    
    # If no selected tastes provided, check all (default behavior)
    if selected_tastes is None:
        selected_tastes = ["M100", "M110", "M120"]
    
    # ตรวจสอบคำว่า "เดิม" (สำหรับ M100 - ดั้งเดิม)
    if "เดิม" in ocr_text and "M100" in selected_tastes:
        return "M100"
    
    # ตรวจสอบคำว่า "2%" (สำหรับ M110 - น้ำตาล 2%)
    elif "2%" in ocr_text and "M110" in selected_tastes:
        return "M110"
    
    # ตรวจสอบคำว่า "ลัก" (สำหรับ M120 - ผสมแมงลัก)
    elif "ลัก" in ocr_text and "M120" in selected_tastes:
        return "M120"

# =============================================================================
# FADED TEXT DETECTION FUNCTIONS
# =============================================================================

def safe_show_img(img, to_bgr=False):
    """รับภาพ 1 หรือ 3 แชนเนล แล้วทำให้ปลอดภัยต่อการแสดงผล"""
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if len(img.shape) == 2:
        disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        disp = img.copy()
        if to_bgr:
            disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
    return disp

def fit_to_same_size(images, target_size=(280, 280)):
    """รีไซส์ภาพทุกใบให้เท่ากันเพื่อง่ายต่อการ hstack/vstack"""
    out = []
    th, tw = target_size
    for im in images:
        h, w = im.shape[:2]
        # keep aspect by padding (letterbox)
        scale = min(tw / w, th / h)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        y0 = (th - nh) // 2
        x0 = (tw - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        out.append(canvas)
    return out

def make_montage(title_imgs_rows, cell_size=(280, 280)):
    """title_imgs_rows: list ของ rows; แต่ละ row คือ list ของ (title, image_bgr)"""
    rows_visual = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for row in title_imgs_rows:
        row_imgs = []
        for title, img in row:
            disp = safe_show_img(img)
            # วาง title บนแถบหัว
            bar_h = 28
            th, tw = disp.shape[0], disp.shape[1]
            bar = np.full((bar_h, tw, 3), 32, dtype=np.uint8)
            cv2.putText(bar, title, (8, bar_h-8), font, 0.6, (255,255,255), 1, cv2.LINE_AA)
            disp = np.vstack([bar, disp])
            row_imgs.append(disp)
        # รีไซส์ให้เท่ากันทั้งแถว (ขนาดเท่ากับ cell_size + bar)
        target_w, target_h = cell_size[1], cell_size[0] + 28
        row_imgs = fit_to_same_size(row_imgs, target_size=(target_h, target_w))
        rows_visual.append(cv2.hconcat(row_imgs))

    montage = cv2.vconcat(rows_visual)
    return montage

def detect_faded_text_in_cap(cap_image, yolo_model=None, show_debug=False):
    """
    Detect faded text in cap image using the provided algorithm
    
    Args:
        cap_image: numpy array of the cropped cap image (from Step 2: Crop detected regions)
        yolo_model: YOLO model for cap detection (ไม่ใช้แล้ว - ใช้รูปที่ crop แล้วโดยตรง)
        show_debug: whether to show debug visualization
        
    Returns:
        dict: {
            'status': 'faded' or 'normal' or 'unknown',
            'num_chars': int,
            'total_area': int,
            'debug_images': dict (if show_debug=True)
        }
    """
    if cap_image is None:
        return {'status': 'unknown', 'num_chars': 0, 'total_area': 0}
    
    try:
        # Convert to BGR if needed
        if len(cap_image.shape) == 3 and cap_image.shape[2] == 3:
            img = cap_image.copy()
        else:
            img = cv2.cvtColor(cap_image, cv2.COLOR_GRAY2BGR)
        
        # Initialize result
        result = {
            'status': 'unknown',
            'num_chars': 0,
            'total_area': 0
        }
        
        if show_debug:
            result['debug_images'] = {}
        
        # ใช้รูปที่ crop แล้วโดยตรง ไม่ต้องใช้ YOLO model ใหม่
        # เพราะรูปที่ส่งมาเป็น cap image ที่ crop แล้วจาก Step 2
        roi = img.copy()
        roi_show = roi.copy()
        
        # Prepare debug images
        gray = None
        mask_circle = None
        roi_masked = None
        edges = None
        text_edges = None
        
        # Gray + Blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, 
                               FADED_TEXT_CONFIG['GAUSSIAN_BLUR_PARAMS']['kernel_size'], 
                               FADED_TEXT_CONFIG['GAUSSIAN_BLUR_PARAMS']['sigma'])
        
        # Detect circles
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, 
            **FADED_TEXT_CONFIG['HOUGH_CIRCLES_PARAMS']
        )
        
        # Prepare mask/circle
        mask_circle = np.zeros_like(gray)
        circle_edges = np.zeros_like(gray)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            cx, cy, r = circles[0, 0]  # ใช้วงแรกพอ
        else:
            # ROI วงกลมกลาง 
            h, w = gray.shape
            cx, cy = w // 2, h // 2
            r = min(w, h)//2 - FADED_TEXT_CONFIG['CIRCLE_FALLBACK_MARGIN']
        
        cv2.circle(mask_circle, (int(cx), int(cy)), int(r), 255, -1)
        cv2.circle(circle_edges, (int(cx), int(cy)), int(r), 255, 2)
        cv2.circle(roi_show, (int(cx), int(cy)), int(r), (0,255,0), 2)
        
        # Mask ROI + Edges
        roi_masked = cv2.bitwise_and(gray, gray, mask=mask_circle)
        edges = cv2.Canny(roi_masked, 
                         FADED_TEXT_CONFIG['CANNY_PARAMS']['low_threshold'], 
                         FADED_TEXT_CONFIG['CANNY_PARAMS']['high_threshold'])
        
        # ลบเส้นวงกลม
        text_edges = cv2.bitwise_and(edges, cv2.bitwise_not(circle_edges))
        
        # CCA (Connected Component Analysis)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_edges, connectivity=8)
        stat_fg = stats[1:, :]  # ลบแบ็คกราวด์
        if stat_fg.size > 0:
            areas_all = stat_fg[:, cv2.CC_STAT_AREA]
            keep_mask = areas_all > FADED_TEXT_CONFIG['MIN_AREA']
            areas_keep = areas_all[keep_mask]
            num_chars = int(keep_mask.sum())
            total_area = int(areas_keep.sum())
        else:
            num_chars, total_area = 0, 0
        
        # Determine status
        status = "faded" if total_area < FADED_TEXT_CONFIG['AREA_THRESH'] else "normal"
        
        result.update({
            'status': status,
            'num_chars': num_chars,
            'total_area': total_area
        })
        
        if show_debug:
            result['debug_images'] = {
                'roi_show': roi_show,
                'gray': gray,
                'mask_circle': mask_circle,
                'roi_masked': roi_masked,
                'edges': edges,
                'text_edges': text_edges
            }
        
        print(f"Faded text detection - Components: {num_chars}, Area: {total_area}, Status: {status}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in faded text detection: {e}")
        return {'status': 'unknown', 'num_chars': 0, 'total_area': 0, 'error': str(e)}
    
    return None

def perform_ocr_on_image(image):
    """
    Perform OCR on an image using EasyOCR
    
    Args:
        image: numpy array image
        
    Returns:
        List of OCR results with text and confidence
    """
    try:
        # Check if OCR reader is available
        if ocr_reader is None:
            print("OCR Error: EasyOCR reader not initialized")
            return []
        
        # Ensure image is in the correct format
        if image is None:
            print("OCR Error: Image is None")
            return []
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already BGR, convert to RGB for better OCR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            # Grayscale, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        results = ocr_reader.readtext(image_rgb, detail=1)
        ocr_results = []
        
        for (bbox, text, confidence) in results:
            ocr_results.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        return ocr_results
    except Exception as e:
        print(f"OCR Error: {e}")
        # Try with original image if RGB conversion failed
        try:
            if ocr_reader is not None:
                results = ocr_reader.readtext(image, detail=1)
                ocr_results = []
                
                for (bbox, text, confidence) in results:
                    ocr_results.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                
                return ocr_results
        except Exception as e2:
            print(f"OCR Error (fallback): {e2}")
        return []

class ModbusThread(QThread):
    """Thread for Modbus communication"""
    modbus_status = pyqtSignal(str)
    trigger_detected = pyqtSignal()
    queue_trigger_detected = pyqtSignal()  # สำหรับการถ่ายภาพจากคิว
    bottle_type_detected = pyqtSignal(str, str)  # (bottle_type, ocr_text)
    d6004_status_updated = pyqtSignal(int)  # D6004 value
    d6007_status_updated = pyqtSignal(int)  # D6007 value
    reset_processing_signal = pyqtSignal()  # สำหรับ reset การประมวลผล
    m513_stop_signal = pyqtSignal()  # สำหรับ M513 stop program
    
    def __init__(self, modbus_ip="192.168.1.5", modbus_port=502):
        super().__init__()
        self.modbus_ip = modbus_ip
        self.modbus_port = modbus_port
        self.modbus_client = None
        self.is_running = False
        self.last_m301 = False
        self.last_m512 = False
        self.last_m513 = False
        self.last_m511 = False
        self.m512_ready = False
        self.stop_requested = False
        self.m600_reset_pending = False
        self.pending_m301_count = 0
        self.program_enabled = False  # เริ่มต้นปิดการทำงานของโปรแกรม (รอ M511)
        self.d6006_monitoring = False  # สำหรับ M130 monitoring
        self.angle3_retry_mode = False  # สำหรับการถ่ายภาพซ้ำหลัง D6006=300
        self.angle3_retry_count = 0  # นับจำนวนครั้งที่ถ่ายภาพซ้ำ
        self.max_angle3_retries = 10  # จำนวนครั้งสูงสุดที่ถ่ายภาพซ้ำ
        self.last_d6007_value = None  # เก็บค่า D6007 ครั้งล่าสุดเพื่อตรวจสอบการเปลี่ยนแปลง
        
    def run(self):
        """Main Modbus loop"""
        try:
            # Create Modbus client
            self.modbus_client = ModbusTcpClient(self.modbus_ip, port=self.modbus_port)
            if not self.modbus_client.connect():
                self.modbus_status.emit("❌ ไม่สามารถเชื่อมต่อ Modbus ได้")
                return
            
            self.modbus_status.emit("✅ เชื่อมต่อ Modbus สำเร็จ")
            self.is_running = True
            
            while self.is_running and not self.stop_requested:
                # Check M511 (start program)
                try:
                    result = self.modbus_client.read_coils(511, 1, unit=1)
                    if result.isError():
                        m511 = False
                    else:
                        m511 = result.bits[0]
                        # ตรวจสอบการเปลี่ยนสถานะ M511
                        if m511 and not self.last_m511:
                            print("✅ M511 ON: เริ่มการทำงาน")
                            self.modbus_status.emit("✅ M511 ON: เริ่มการทำงาน")
                            self.program_enabled = True
                        elif not m511 and self.last_m511:
                            print("❌ M511 OFF: หยุดการทำงาน")
                            self.modbus_status.emit("❌ M511 OFF: หยุดการทำงาน")
                            self.program_enabled = False
                except Exception as e:
                    print(f"❌ Modbus M511 read error: {e}")
                    m511 = False
                
                self.last_m511 = m511
                
                # Check M301 (trigger) - เพิ่มการตรวจสอบที่แม่นยำขึ้น
                try:
                    result = self.modbus_client.read_coils(301, 1, unit=1)
                    if result.isError():
                        m301 = False
                    else:
                        m301 = result.bits[0]
                        # Debug: แสดงสถานะ M301 (เฉพาะเมื่อเปลี่ยน)
                        # if m301 != self.last_m301:
                        #     print(f"🔍 M301 Status: {m301} (Last: {self.last_m301})")  # Reduced spam
                except Exception as e:
                    print(f"❌ Modbus read error: {e}")
                    m301 = False
                    
                # ตรวจสอบการเปลี่ยนสถานะจาก OFF เป็น ON เท่านั้น
                if m301 and not self.last_m301:
                    # print("✅ M301 TRIGGER DETECTED: OFF → ON")  # Reduced spam
                    self.modbus_status.emit("🔔 M301 ON: ตรวจพบสัญญาณถ่ายภาพ")
                    
                    # ตรวจสอบว่าโปรแกรมเปิดใช้งานอยู่หรือไม่
                    if not self.program_enabled:
                        # print("⚠️ M301 detected but program is stopped (M511 OFF)")  # Reduced spam
                        self.modbus_status.emit("⚠️ M301 ตรวจพบแต่โปรแกรมหยุดทำงาน (M511 OFF)")
                        continue
                    
                    # ตรวจสอบว่า M130 กำลังทำงานอยู่หรือไม่
                    if self.d6006_monitoring:
                        # print("⚠️ M301 detected but M130 is active - ignoring trigger")  # Reduced spam
                        self.modbus_status.emit("⚠️ M301 ตรวจพบแต่ M130 กำลังทำงาน - ไม่สนใจสัญญาณ")
                        continue
                    
                    if not self.m600_reset_pending:
                        # M600 พร้อม - เริ่มกระบวนการทันที
                        print("🚀 M600 พร้อม - เริ่มถ่ายภาพทันที")
                        self.trigger_detected.emit()
                    else:
                        # M600 ยังไม่ reset - เพิ่มคิว
                        self.pending_m301_count += 1
                        print(f"📋 เพิ่มคิวถ่ายภาพ (คิวปัจจุบัน: {self.pending_m301_count})")
                        self.modbus_status.emit(f"📋 เพิ่มคิวถ่ายภาพ (คิวปัจจุบัน: {self.pending_m301_count})")
                        
                elif not m301 and self.last_m301:
                    print("📉 M301 RESET: ON → OFF")
                    self.modbus_status.emit("📉 M301 OFF: รอสัญญาณถัดไป")
                
                self.last_m301 = m301
                
                # ตรวจสอบ D6006 สำหรับ M130 และ M850
                if self.d6006_monitoring or self.angle3_retry_mode:
                    try:
                        result = self.modbus_client.read_holding_registers(6006, count=1, unit=2)
                        d6006 = result.registers[0] if not result.isError() else None
                        # print(f"🔍 D6006 value: {d6006}")  # Reduced spam
                    except:
                        d6006 = None
                        print("❌ D6006 Read Error")
                        
                    if d6006 == 300:
                        try:
                            # RESET M130, M850 และ M600
                            self.modbus_client.write_coil(130, False, unit=1)
                            # print("✅ RESET M130 = 0 (D6006 = 300)")  # Reduced spam
                            self.modbus_status.emit("✅ RESET M130 = 0 (D6006 = 300)")
                            
                            self.modbus_client.write_coil(850, False, unit=1)
                            # print("✅ RESET M850 = 0 (D6006 = 300)")  # Reduced spam
                            self.modbus_status.emit("✅ RESET M850 = 0 (D6006 = 300)")
                            
                            self.modbus_client.write_coil(600, False, unit=1)
                            # print("✅ RESET M600 = 0 (D6006 = 300)")  # Reduced spam
                            self.modbus_status.emit("✅ RESET M600 = 0 (D6006 = 300)")
                            
                            # หยุด monitoring D6006 และ reset pending
                            self.d6006_monitoring = False
                            self.m600_reset_pending = False
                            
                            # เริ่มโหมดถ่ายภาพซ้ำสำหรับ angle3
                            self.angle3_retry_mode = True
                            self.angle3_retry_count = 0
                            
                            # Reset การประมวลผลเพื่อให้ใช้ภาพใหม่
                            print("🔄 RESET การประมวลผลเพื่อใช้ภาพใหม่...")
                            self.modbus_status.emit("🔄 RESET การประมวลผลเพื่อใช้ภาพใหม่...")
                            
                            # ส่งสัญญาณให้ GUI reset การประมวลผล
                            self.reset_processing_signal.emit()
                            
                            # ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)
                            print("📸 D6006=300: ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)")
                            self.modbus_status.emit("📸 D6006=300: ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)")
                            
                            # ส่งสัญญาณให้ GUI ถ่ายภาพ USB ใหม่ทันที
                            self.trigger_detected.emit()  # ส่งสัญญาณเหมือน M301
                            
                        except Exception as e:
                            print(f"❌ ไม่สามารถ RESET M130, M750 และ M600 ได้: {e}")
                    
                    if d6006 == 0:
                        # D6006 = 0 - Reset M850 และหยุดโหมดถ่ายภาพซ้ำ
                        if self.angle3_retry_mode:
                            try:
                                self.modbus_client.write_coil(850, False, unit=1)
                                # print("✅ RESET M850 = 0 (D6006 = 0)")  # Reduced spam
                                self.modbus_status.emit("✅ RESET M850 = 0 (D6006 = 0)")
                                
                                # หยุดโหมดถ่ายภาพซ้ำ
                                self.angle3_retry_mode = False
                                self.angle3_retry_count = 0
                                # print("🛑 หยุดโหมดถ่ายภาพซ้ำ (D6006 = 0)")  # Reduced spam
                                self.modbus_status.emit("🛑 หยุดโหมดถ่ายภาพซ้ำ (D6006 = 0)")
                                
                            except Exception as e:
                                print(f"❌ ไม่สามารถ RESET M850 ได้: {e}")
                
                # ตรวจสอบ D6004 เพื่อ reset M600
                if self.m600_reset_pending:
                    try:
                        result = self.modbus_client.read_holding_registers(6004, count=1, unit=2)
                        d6004 = result.registers[0] if not result.isError() else None
                        # ส่งค่า D6004 ไปยัง GUI
                        if d6004 is not None:
                            self.d6004_status_updated.emit(d6004)
                    except:
                        d6004 = None
                        print("❌ D6004 Read Error")
                        
                    if d6004 == 200:
                        try:
                            # RESET M600
                            self.modbus_client.write_coil(600, False, unit=1)
                            print("✅ RESET M600 = 0")
                            self.modbus_status.emit("✅ RESET M600 = 0")
                            
                            # RESET M100, M110, M120, M130
                            self.modbus_client.write_coil(100, False, unit=1)
                            print("✅ RESET M100 = 0")
                            self.modbus_status.emit("✅ RESET M100 = 0")
                            
                            self.modbus_client.write_coil(110, False, unit=1)
                            print("✅ RESET M110 = 0")
                            self.modbus_status.emit("✅ RESET M110 = 0")
                            
                            self.modbus_client.write_coil(120, False, unit=1)
                            print("✅ RESET M120 = 0")
                            self.modbus_status.emit("✅ RESET M120 = 0")
                            
                            self.modbus_client.write_coil(130, False, unit=1)
                            print("✅ RESET M130 = 0")
                            self.modbus_status.emit("✅ RESET M130 = 0")
                            
                            # RESET M140 (ฝาไม่ผ่าน)
                            self.modbus_client.write_coil(140, False, unit=1)
                            print("✅ RESET M140 = 0")
                            self.modbus_status.emit("✅ RESET M140 = 0")
                            
                            # RESET M720, M721, M722 (โหมดรสชาติ)
                            self.modbus_client.write_coil(720, False, unit=1)
                            print("✅ RESET M720 = 0")
                            self.modbus_status.emit("✅ RESET M720 = 0")
                            
                            self.modbus_client.write_coil(721, False, unit=1)
                            print("✅ RESET M721 = 0")
                            self.modbus_status.emit("✅ RESET M721 = 0")
                            
                            self.modbus_client.write_coil(722, False, unit=1)
                            print("✅ RESET M722 = 0")
                            self.modbus_status.emit("✅ RESET M722 = 0")
                            
                            # RESET M730-M735 (2-taste combinations)
                            self.modbus_client.write_coil(730, False, unit=1)
                            print("✅ RESET M730 = 0")
                            self.modbus_status.emit("✅ RESET M730 = 0")
                            
                            self.modbus_client.write_coil(731, False, unit=1)
                            print("✅ RESET M731 = 0")
                            self.modbus_status.emit("✅ RESET M731 = 0")
                            
                            self.modbus_client.write_coil(732, False, unit=1)
                            print("✅ RESET M732 = 0")
                            self.modbus_status.emit("✅ RESET M732 = 0")
                            
                            self.modbus_client.write_coil(733, False, unit=1)
                            print("✅ RESET M733 = 0")
                            self.modbus_status.emit("✅ RESET M733 = 0")
                            
                            self.modbus_client.write_coil(734, False, unit=1)
                            print("✅ RESET M734 = 0")
                            self.modbus_status.emit("✅ RESET M734 = 0")
                            
                            self.modbus_client.write_coil(735, False, unit=1)
                            print("✅ RESET M735 = 0")
                            self.modbus_status.emit("✅ RESET M735 = 0")
                            
                            # RESET M740-M742 (1-taste signals)
                            self.modbus_client.write_coil(740, False, unit=1)
                            print("✅ RESET M740 = 0")
                            self.modbus_status.emit("✅ RESET M740 = 0")
                            
                            self.modbus_client.write_coil(741, False, unit=1)
                            print("✅ RESET M741 = 0")
                            self.modbus_status.emit("✅ RESET M741 = 0")
                            
                            self.modbus_client.write_coil(742, False, unit=1)
                            print("✅ RESET M742 = 0")
                            self.modbus_status.emit("✅ RESET M742 = 0")
                            
                        except Exception as e:
                            print(f"❌ ไม่สามารถ RESET Modbus ได้: {e}")
                            
                        self.m600_reset_pending = False
                        
                        # ถ่ายภาพทันทีหลัง M600, M100, M110, M120, M130, M140 reset (ถ้ามีคิว)
                        if self.pending_m301_count > 0:
                            print(f"📸 ถ่ายภาพทันทีหลัง reset ทั้งหมด (คิว: {self.pending_m301_count})")
                            self.modbus_status.emit(f"📸 ถ่ายภาพทันทีหลัง reset ทั้งหมด (คิว: {self.pending_m301_count})")
                            # ลดคิวลง 1 และถ่ายภาพจากคิว
                            self.pending_m301_count -= 1
                            print(f"📋 ลดคิวลงเหลือ: {self.pending_m301_count}")
                            self.queue_trigger_detected.emit()  # ใช้ signal สำหรับคิว
                            
                            # ถ้ายังมีคิวอยู่ ให้แสดงสถานะ
                            if self.pending_m301_count > 0:
                                print(f"📋 ยังมีคิวรออยู่: {self.pending_m301_count}")
                                self.modbus_status.emit(f"📋 ยังมีคิวรออยู่: {self.pending_m301_count}")
                        else:
                            print("✅ ไม่มีคิวถ่ายภาพ พร้อมรับ M301 ใหม่")
                            self.modbus_status.emit("✅ ไม่มีคิวถ่ายภาพ พร้อมรับ M301 ใหม่")
                
                # Check M512 (condition for M513)
                try:
                    result = self.modbus_client.read_coils(512, 1, unit=1)
                    m512 = not result.isError() and result.bits[0]
                except:
                    m512 = False
                    
                if m512 and not self.last_m512:
                    self.modbus_status.emit("🔔 M512 ON: พร้อมรับ M513 เพื่อออกจากโปรแกรม")
                    self.m512_ready = True
                
                self.last_m512 = m512
                
                # Check M513 (stop program) - ไม่ต้องรอ M512
                try:
                    result = self.modbus_client.read_coils(513, 1, unit=1)
                    m513 = not result.isError() and result.bits[0]
                except:
                    m513 = False
                    
                if m513 and not self.last_m513:
                    self.modbus_status.emit("🔔 M513 ON: หยุดการทำงาน")
                    self.program_enabled = False
                    
                    # Reset M700 และ M701 เมื่อ M513 ON
                    try:
                        self.reset_m700()
                        print("✅ RESET M700 = 0 (จาก M513)")
                        self.modbus_status.emit("✅ RESET M700 = 0 (จาก M513)")
                        
                        self.reset_m701()
                        print("✅ RESET M701 = 0 (จาก M513)")
                        self.modbus_status.emit("✅ RESET M701 = 0 (จาก M513)")
                    except Exception as e:
                        print(f"❌ ไม่สามารถ RESET M700/M701 ได้: {e}")
                    
                    # ส่งสัญญาณให้ GUI ทำการ reset ทั้งหมดเหมือนกดปุ่ม Stop
                    self.m513_stop_signal.emit()
                    print("🔄 M513: ส่งสัญญาณให้ GUI reset ทั้งหมด")
                    
                    # ไม่ break ออกจาก loop เพื่อให้ Modbus ยังทำงานต่อ
                
                self.last_m513 = m513
                
                # ตรวจสอบ D6007 เพื่อตรวจสอบประเภทขวด
                try:
                    result = self.modbus_client.read_holding_registers(6007, count=1, unit=2)
                    d6007 = result.registers[0] if not result.isError() else None
                    # ส่งค่า D6007 ไปยัง GUI เฉพาะเมื่อค่าเปลี่ยน
                    if d6007 is not None and d6007 != self.last_d6007_value:
                        self.d6007_status_updated.emit(d6007)
                        self.last_d6007_value = d6007
                except:
                    d6007 = None
                    print("❌ D6007 Read Error")
                
                # Sleep to reduce CPU usage
                time.sleep(0.1)
                
        except Exception as e:
            self.modbus_status.emit(f"❌ Modbus Error: {e}")
        finally:
            if self.modbus_client:
                self.modbus_client.close()
                self.modbus_status.emit("🔌 ปิด Modbus connection")
    
    def write_coil(self, coil_address, value):
        """Write coil to Modbus"""
        # ไม่ต้องตรวจสอบเงื่อนไขใดๆ - สามารถเขียนได้ตลอด
        try:
            if self.modbus_client and self.modbus_client.is_socket_open():
                result = self.modbus_client.write_coil(coil_address, value, unit=1)
                if not result.isError():
                    print(f"✅ ON M{coil_address} = {value}")
                    return True
                else:
                    print(f"❌ ไม่สามารถ ON M{coil_address} ได้")
                    return False
            else:
                print("❌ Modbus client ไม่พร้อมใช้งาน")
                return False
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการ ON M{coil_address}: {e}")
            return False
    
    def write_register(self, register_address, value):
        """Write register to Modbus"""
        try:
            print(f"🔄 write_register: กำลังเขียน D{register_address} = {value}")
            if self.modbus_client and self.modbus_client.is_socket_open():
                print(f"✅ Modbus client พร้อมใช้งาน - กำลังส่งคำสั่ง")
                result = self.modbus_client.write_register(register_address, value, unit=1)
                if not result.isError():
                    print(f"✅ Write D{register_address} = {value} สำเร็จ")
                    return True
                else:
                    print(f"❌ ไม่สามารถเขียน D{register_address} ได้ - Error: {result}")
                    return False
            else:
                print("❌ Modbus client ไม่พร้อมใช้งาน")
                print(f"   - modbus_client exists: {self.modbus_client is not None}")
                if self.modbus_client:
                    print(f"   - socket open: {self.modbus_client.is_socket_open()}")
                return False
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการเขียน D{register_address}: {e}")
            return False
    
    def set_d4005(self, value=100):
        """Set D4005 to value without automatic reset"""
        try:
            print(f"🔄 set_d4005: เริ่มส่งค่า D4005 = {value}")
            # ส่งค่า D4005 = value
            success = self.write_register(4005, value)
            if success:
                print(f"✅ ส่งค่า D4005 = {value} สำเร็จ")
                return True
            else:
                print(f"❌ ไม่สามารถส่งค่า D4005 = {value} ได้")
                return False
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการส่งค่า D4005: {e}")
            return False
    
    # def reset_d4005(self):
    #     """Reset D4005 to 0 - Currently not used as auto-reset is disabled"""
    #     try:
    #         success = self.write_register(4005, 0)
    #         if success:
    #             print("✅ รีเซ็ท D4005 = 0 สำเร็จ")
    #         else:
    #             print("❌ ไม่สามารถรีเซ็ท D4005 = 0 ได้")
    #         return success
    #     except Exception as e:
    #         print(f"❌ ข้อผิดพลาดในการรีเซ็ท D4005: {e}")
    #         return False
    
    def on_m100(self):
        """ON M100 (ดั้งเดิม)"""
        return self.write_coil(100, True)
    
    def on_m110(self):
        """ON M110 (น้ำตาล 2%)"""
        return self.write_coil(110, True)
    
    def on_m120(self):
        """ON M120 (ผสมแมงลัก)"""
        return self.write_coil(120, True)
    
    def on_m130(self):
        """ON M130 (angle3)"""
        return self.write_coil(130, True)
    
    def on_m750(self):
        """ON M750 (angle3)"""
        return self.write_coil(750, True)
    
    def reset_m750(self):
        """RESET M750"""
        return self.write_coil(750, False)
    
    def on_m850(self):
        """ON M850 (non-angle3 detection)"""
        return self.write_coil(850, True)
    
    def on_m450(self):
        """ON M450"""
        return self.write_coil(450, True)
    
    def reset_m450(self):
        """RESET M450"""
        return self.write_coil(450, False)
    
    def reset_m850(self):
        """RESET M850"""
        return self.write_coil(850, False)
    
    def on_m140(self):
        """ON M140 (ฝาไม่ผ่าน - ไม่ตรงกับฟอร์ม)"""
        return self.write_coil(140, True)
    
    def reset_m140(self):
        """RESET M140"""
        return self.write_coil(140, False)
    
    def on_m600(self):
        """ON M600"""
        success = self.write_coil(600, True)
        if success:
            self.m600_reset_pending = True
            print("⏳ รอ D6004=200 เพื่อ RESET M600, M100, M110, M120, M130, M140...")
        return success
    
    def reset_m100(self):
        """RESET M100"""
        return self.write_coil(100, False)
    
    def reset_m110(self):
        """RESET M110"""
        return self.write_coil(110, False)
    
    def reset_m120(self):
        """RESET M120"""
        return self.write_coil(120, False)
    
    def reset_m130(self):
        """RESET M130"""
        return self.write_coil(130, False)
    
    def reset_all_bottle_types(self):
        """RESET M100, M110, M120, M130 ทั้งหมด"""
        success = True
        success &= self.reset_m100()
        success &= self.reset_m110()
        success &= self.reset_m120()
        success &= self.reset_m130()
        return success
    
    def on_m700(self):
        """ON M700 (START) and reset M701"""
        success = True
        # Reset M701 ก่อน
        success &= self.reset_m701()
        # จากนั้น ON M700
        success &= self.write_coil(700, True)
        return success
    
    def on_m701(self):
        """ON M701 (STOP) and reset M700"""
        success = True
        # Reset M700 ก่อน
        success &= self.reset_m700()
        # จากนั้น ON M701
        success &= self.write_coil(701, True)
        # Reset M701 ตัวเองหลัง 1.5 วินาที (pulse)
        QTimer.singleShot(1500, lambda: self.reset_m701_delayed())
        return success
    
    def reset_m701_delayed(self):
        """Reset M701 after delay"""
        success = self.reset_m701()
        if success:
            print("✅ RESET M701 = 0 (หลัง 1.5 วินาที)")
        else:
            print("❌ ไม่สามารถ RESET M701 ได้")
    
    def reset_m700(self):
        """RESET M700"""
        return self.write_coil(700, False)
    
    def reset_m701(self):
        """RESET M701"""
        return self.write_coil(701, False)
    
    def on_m503(self):
        """ON M503 (เปิด Gripper)"""
        return self.write_coil(503, True)
    
    def on_m505(self):
        """ON M505 (ปิด Gripper)"""
        return self.write_coil(505, True)
    
    def reset_m503(self):
        """RESET M503"""
        return self.write_coil(503, False)
    
    def reset_m505(self):
        """RESET M505"""
        return self.write_coil(505, False)
    
    def reset_m600(self):
        """RESET M600"""
        success = self.write_coil(600, False)
        if success:
            self.m600_reset_pending = False
            print("✅ RESET M600 = 0 (พร้อมรับงานใหม่)")
        return success
    
    def on_m720(self):
        """ON M720 (โหมด 3 รสชาติ)"""
        return self.write_coil(720, True)
    
    def on_m721(self):
        """ON M721 (โหมด 2 รสชาติ)"""
        return self.write_coil(721, True)
    
    def on_m722(self):
        """ON M722 (โหมด 1 รสชาติ)"""
        return self.write_coil(722, True)
    
    def reset_m720(self):
        """RESET M720"""
        return self.write_coil(720, False)
    
    def reset_m721(self):
        """RESET M721"""
        return self.write_coil(721, False)
    
    def reset_m722(self):
        """RESET M722"""
        return self.write_coil(722, False)
    
    def on_m730(self):
        """ON M730 (ขวดดั้งเดิม + ขวดน้ำตาล 2%)"""
        return self.write_coil(730, True)
    
    def on_m731(self):
        """ON M731 (ขวดดั้งเดิม + ขวดผสมแมงลัก)"""
        return self.write_coil(731, True)
    
    def on_m732(self):
        """ON M732 (ขวดน้ำตาล 2% + ขวดผสมแมงลัก)"""
        return self.write_coil(732, True)
    
    def on_m733(self):
        """ON M733 (ขวดน้ำตาล 2% + ขวดดั้งเดิม)"""
        return self.write_coil(733, True)
    
    def on_m734(self):
        """ON M734 (ขวดผสมแมงลัก + ขวดน้ำตาล 2%)"""
        return self.write_coil(734, True)
    
    def on_m735(self):
        """ON M735 (ขวดผสมแมงลัก + ขวดดั้งเดิม)"""
        return self.write_coil(735, True)
    
    def reset_m730(self):
        """RESET M730"""
        return self.write_coil(730, False)
    
    def reset_m731(self):
        """RESET M731"""
        return self.write_coil(731, False)
    
    def reset_m732(self):
        """RESET M732"""
        return self.write_coil(732, False)
    
    def reset_m733(self):
        """RESET M733"""
        return self.write_coil(733, False)
    
    def reset_m734(self):
        """RESET M734"""
        return self.write_coil(734, False)
    
    def reset_m735(self):
        """RESET M735"""
        return self.write_coil(735, False)
    
    def on_m740(self):
        """ON M740 (ขวดดั้งเดิม)"""
        return self.write_coil(740, True)
    
    def on_m741(self):
        """ON M741 (ขวดน้ำตาล 2%)"""
        return self.write_coil(741, True)
    
    def on_m742(self):
        """ON M742 (ขวดผสมแมงลัก)"""
        return self.write_coil(742, True)
    
    def reset_m740(self):
        """RESET M740"""
        return self.write_coil(740, False)
    
    def reset_m741(self):
        """RESET M741"""
        return self.write_coil(741, False)
    
    def reset_m742(self):
        """RESET M742"""
        return self.write_coil(742, False)
    
    def start_d6006_monitoring(self):
        """Start monitoring D6006 for M130"""
        self.d6006_monitoring = True
        print("🔍 เริ่มต้นการอ่าน D6006 สำหรับ M130...")
        self.modbus_status.emit("🔍 เริ่มต้นการอ่าน D6006 สำหรับ M130...")
    
    def stop(self):
        """Stop Modbus thread"""
        self.is_running = False
        self.stop_requested = True

class BottleDetectionThread(QThread):
    """Thread for processing bottle detection to avoid GUI freezing"""
    result_ready = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self, image, save_crops: bool = False, output_folder: str = "", selected_tastes: list = None):
        super().__init__()
        self.image = image
        self.save_crops = save_crops
        self.selected_tastes = selected_tastes or ["M100", "M110", "M120"]  # Default to all tastes
        self.output_folder = output_folder
        
    def run(self):
        try:
            print("🔄 BOTTLE DETECTION THREAD: Starting processing...")
            print(f"🔄 BOTTLE DETECTION THREAD: Image shape: {self.image.shape}")
            self.status_updated.emit("กำลังประมวลผลภาพ...")
            self.progress_updated.emit(10)
        
            # Save temporary image for processing
            temp_path = "temp_camera_image.jpg"
            cv2.imwrite(temp_path, self.image)
            print("🔄 BOTTLE DETECTION THREAD: Saved temp image")
            self.progress_updated.emit(20)
            
            # Process the image
            print("🔄 BOTTLE DETECTION THREAD: Calling process_bottle_image_simple...")
            self.status_updated.emit("กำลังตรวจจับขวด...")
            self.progress_updated.emit(30)
            result = process_bottle_image_simple(temp_path)
            print(f"🔄 BOTTLE DETECTION THREAD: Processing result keys: {list(result.keys())}")
            self.progress_updated.emit(50)
            
            # Add image info to result
            result['image'] = self.image
            result['image_shape'] = self.image.shape
            
            # Check for "angle3" in YOLO detections first (before cropping type regions)
            self.status_updated.emit("กำลังตรวจสอบ angle3...")
            self.progress_updated.emit(55)
            
            print(f"🔍 BOTTLE DETECTION THREAD: YOLO detections: {result.get('detections', [])}")
            print(f"🔍 BOTTLE DETECTION THREAD: Detected labels: {result.get('summary', {}).get('detected_labels', [])}")
            
            # Check if "angle3" is found in YOLO detections
            detected_labels = result.get('summary', {}).get('detected_labels', [])
            if 'angle3' in detected_labels:
                print(f"🎯 BOTTLE DETECTION THREAD: Found 'angle3' in YOLO detections")
                result['combined_ocr_text'] = "angle3 detected by YOLO"
                result['bottle_type'] = "M130"
                result['angle3_detected'] = True
                self.progress_updated.emit(100)
                self.status_updated.emit("พบ angle3 - ประมวลผลเสร็จสิ้น")
                self.result_ready.emit(result)
                return
            else:
                print(f"❌ BOTTLE DETECTION THREAD: 'angle3' not found in YOLO detections")
            
            # Perform OCR on type crops and combine text
            combined_ocr_text = ""
            if result.get('type_crops'):
                print(f"🔄 BOTTLE DETECTION THREAD: Found {len(result['type_crops'])} type crops")
                self.status_updated.emit("กำลังประมวลผล OCR...")
                self.progress_updated.emit(60)
                
                total_crops = len(result['type_crops'])
                for i, crop in enumerate(result['type_crops']):
                    print(f"🔄 BOTTLE DETECTION THREAD: Processing crop {i+1}")
                    self.status_updated.emit(f"กำลังประมวลผล OCR... ({i+1}/{total_crops})")
                    ocr_results = perform_ocr_on_image(crop['original_crop'])
                    crop['ocr_results'] = ocr_results
                    print(f"🔄 BOTTLE DETECTION THREAD: Crop {i+1} OCR results: {len(ocr_results)}")
                    
                    # อัปเดต progress สำหรับแต่ละ crop
                    progress = 60 + (i + 1) * 20 // total_crops
                    self.progress_updated.emit(progress)
                
                    # รวมข้อความที่อ่านได้
                    for ocr_result in ocr_results:
                        combined_ocr_text += ocr_result['text'] + " "
                
                # ลบช่องว่างที่เกิน
                combined_ocr_text = combined_ocr_text.strip()
                
                # ตรวจสอบ bottle type
                self.status_updated.emit("กำลังตรวจสอบประเภทขวด...")
                self.progress_updated.emit(85)
                bottle_type = check_bottle_type(combined_ocr_text, self.selected_tastes)
                result['combined_ocr_text'] = combined_ocr_text
                result['bottle_type'] = bottle_type
                
                print(f"📝 BOTTLE DETECTION THREAD: Combined OCR Text: '{combined_ocr_text}'")
                if bottle_type:
                    print(f"🏷️ BOTTLE DETECTION THREAD: Detected Bottle Type: {bottle_type}")
                else:
                    print("❌ BOTTLE DETECTION THREAD: ไม่พบคำว่า 'เดิม', '2%', หรือ 'ลัก'")
            else:
                print("❌ BOTTLE DETECTION THREAD: No type crops found")
                self.progress_updated.emit(85)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            # Final progress update
            self.progress_updated.emit(100)
            self.status_updated.emit("ประมวลผลเสร็จสิ้น")
            
            print("🔄 BOTTLE DETECTION THREAD: Emitting result...")
            self.result_ready.emit(result)
                
        except Exception as e:
            print(f"❌ BOTTLE DETECTION THREAD: Error: {e}")
            error_result = {
                'error': str(e),
                'image': self.image,
                'image_shape': self.image.shape if self.image is not None else None
            }
            self.result_ready.emit(error_result)

class BottleDetectionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ระบบตรวจจับขวดและ OCR - USB Camera + Modbus + Sentech Camera')
        
        # ปรับขนาดหน้าต่างให้เหมาะสมกับ Jetson
        self.resize(1800, 1200)
        
        # ตรวจสอบว่าเป็น Jetson หรือไม่
        self.is_jetson = self.check_if_jetson()
        if self.is_jetson:
            print("🚀 Detected Jetson - Enabling fullscreen support")
            self.setWindowState(Qt.WindowFullScreen)
        
        self.current_image = None
        self.current_result = None
        self.batch_results = []
        self.usb_camera = None
        self.sentech_camera = None
        self.modbus_thread = None
        
        # Cap detection models
        self.cap_detector = None
        self.craft_detector = None
        self.rotation_model = None
        self.line_detector = None
        self.ocr_model = None
        
        # Cap detection results
        self.current_sentech_image = None
        self.current_cap_result = None
        self.current_bottle_type = None  # เก็บประเภทขวดที่ตรวจพบ
        
        # Initialize taste mode variables
        self.current_taste_mode = 3  # Default to 3-taste mode
        self.selected_tastes = ["M100", "M110", "M120"]  # Default to all tastes
        
        # Initialize expiry filtering mode variables
        self.current_expiry_mode = "normal"  # "normal" or "filter"
        self.expiry_filter_type = "range"  # "range" or "specific"
        self.expiry_start_date = None
        self.expiry_end_date = None
        self.expiry_specific_date = None
        
        # Initialize OCR test variables
        self.ocr_test_image = None
        self.ocr_test_image_path = None
        self.current_rotation_angle = 0  # Track current rotation angle
        self.original_cap_image = None  # Store original image for rotation reference
        # Manual rotation removed
        
        # Statistics for status tab
        self.total_images_processed_count = 0
        self.successful_detections_count = 0
        self.error_count_value = 0
        self.current_processing_mode = "Manual"
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_all_status)
        self.status_timer.start(1000)  # อัพเดททุก 1 วินาที
        
        self.setup_ui()
        self.setup_camera()
        self.setup_sentech_camera()
        self.setup_cap_detection_models()
        self.setup_modbus()
        
        # เพิ่ม keyboard shortcuts สำหรับ Jetson
        if self.is_jetson:
            self.setup_jetson_shortcuts()
        
    def setup_ui(self):
        # สร้าง main scroll area สำหรับหน้าต่างหลัก
        main_scroll_area = QScrollArea()
        main_scroll_area.setWidgetResizable(True)
        main_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2c3e50;
            }
            QScrollBar:vertical {
                background: #34495e;
                width: 16px;
                border-radius: 8px;
            }
            QScrollBar::handle:vertical {
                background: #7f8c8d;
                border-radius: 8px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #95a5a6;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #34495e;
                height: 16px;
                border-radius: 8px;
            }
            QScrollBar::handle:horizontal {
                background: #7f8c8d;
                border-radius: 8px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #95a5a6;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        # สร้าง main widget ที่จะใส่ใน scroll area
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #2c3e50;")
        layout = QVBoxLayout(main_widget)
        
        # Title
        title_label = QLabel("ระบบตรวจจับขวดและ OCR - USB Camera + Modbus + Sentech Camera")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px; color: #ecf0f1;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Status group
        status_group = QGroupBox("สถานะระบบ")
        status_layout = QHBoxLayout()
        
        # Camera status
        self.camera_status_label = QLabel('📷 กล้อง: กำลังเริ่มต้น...')
        self.camera_status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        status_layout.addWidget(self.camera_status_label)
        
        # Sentech camera status
        self.sentech_camera_status_label = QLabel('📷 Sentech: กำลังเริ่มต้น...')
        self.sentech_camera_status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        status_layout.addWidget(self.sentech_camera_status_label)
        
        # Modbus status
        self.modbus_status_label = QLabel('📡 Modbus: กำลังเชื่อมต่อ...')
        self.modbus_status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        status_layout.addWidget(self.modbus_status_label)
        
        # Queue status
        self.queue_status_label = QLabel('📋 คิว: 0')
        self.queue_status_label.setStyleSheet("color: #e67e22; padding: 5px; font-weight: bold;")
        status_layout.addWidget(self.queue_status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Control buttons
        control_group = QGroupBox("ตัวเลือกการทำงาน")
        control_layout = QHBoxLayout()
        
        # Start/Stop buttons
        self.btn_start = QPushButton('▶️ START (M700)')
        self.btn_start.clicked.connect(self.start_system)
        self.btn_start.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #27ae60; color: white; }")
        control_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton('⏹️ STOP (M701)')
        self.btn_stop.clicked.connect(self.stop_system)
        self.btn_stop.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #e74c3c; color: white; }")
        control_layout.addWidget(self.btn_stop)
        
        # Manual capture button
        self.btn_capture = QPushButton('📸 ถ่ายภาพด้วยตนเอง')
        self.btn_capture.clicked.connect(self.capture_image_manual)
        self.btn_capture.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; }")
        control_layout.addWidget(self.btn_capture)
        
        # Select image file button
        self.btn_select_image = QPushButton('📁 เลือกไฟล์ภาพ')
        self.btn_select_image.clicked.connect(self.select_image_file)
        self.btn_select_image.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #3498db; color: white; }")
        control_layout.addWidget(self.btn_select_image)
        
        # Auto mode is always enabled (checkbox removed)
        
        # Process button
        self.btn_process = QPushButton('🔍 ประมวลผล')
        self.btn_process.clicked.connect(self.process_current_image)
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #27ae60; color: white; }")
        control_layout.addWidget(self.btn_process)
        
        # Stop processing button
        self.btn_stop_processing = QPushButton('⏹️ หยุดประมวลผล')
        self.btn_stop_processing.clicked.connect(self.stop_processing)
        self.btn_stop_processing.setEnabled(False)
        self.btn_stop_processing.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #e74c3c; color: white; }")
        control_layout.addWidget(self.btn_stop_processing)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Taste Mode Selection Group
        taste_mode_group = QGroupBox("โหมดการตรวจสอบรสชาติ")
        taste_mode_layout = QVBoxLayout()
        
        # Taste mode selection
        mode_selection_layout = QHBoxLayout()
        mode_selection_layout.addWidget(QLabel("เลือกโหมด:"))
        
        # Radio buttons for taste modes
        self.taste_mode_1 = QCheckBox("1 รสชาติ")
        self.taste_mode_1.setChecked(False)
        self.taste_mode_1.toggled.connect(self.on_taste_mode_changed)
        self.taste_mode_1.setStyleSheet("QCheckBox { color: white; font-size: 12px; }")
        mode_selection_layout.addWidget(self.taste_mode_1)
        
        self.taste_mode_2 = QCheckBox("2 รสชาติ")
        self.taste_mode_2.setChecked(False)
        self.taste_mode_2.toggled.connect(self.on_taste_mode_changed)
        self.taste_mode_2.setStyleSheet("QCheckBox { color: white; font-size: 12px; }")
        mode_selection_layout.addWidget(self.taste_mode_2)
        
        self.taste_mode_3 = QCheckBox("3 รสชาติ (ปัจจุบัน)")
        self.taste_mode_3.setChecked(True)  # Default to current mode
        self.taste_mode_3.toggled.connect(self.on_taste_mode_changed)
        self.taste_mode_3.setStyleSheet("QCheckBox { color: white; font-size: 12px; }")
        mode_selection_layout.addWidget(self.taste_mode_3)
        
        mode_selection_layout.addStretch()
        taste_mode_layout.addLayout(mode_selection_layout)
        
        # Taste selection (initially hidden)
        self.taste_selection_layout = QHBoxLayout()
        self.taste_selection_layout.addWidget(QLabel("เลือกรสชาติ:"))
        
        # Taste checkboxes
        self.taste_m100 = QCheckBox("M100 (เดิม)")
        self.taste_m100.setStyleSheet("QCheckBox { color: white; font-size: 12px; }")
        self.taste_m100.toggled.connect(self.update_selected_tastes)
        self.taste_selection_layout.addWidget(self.taste_m100)
        
        self.taste_m110 = QCheckBox("M110 (2%)")
        self.taste_m110.setStyleSheet("QCheckBox { color: white; font-size: 12px; }")
        self.taste_m110.toggled.connect(self.update_selected_tastes)
        self.taste_selection_layout.addWidget(self.taste_m110)
        
        self.taste_m120 = QCheckBox("M120 (ลัก)")
        self.taste_m120.setStyleSheet("QCheckBox { color: white; font-size: 12px; }")
        self.taste_m120.toggled.connect(self.update_selected_tastes)
        self.taste_selection_layout.addWidget(self.taste_m120)
        
        self.taste_selection_layout.addStretch()
        taste_mode_layout.addLayout(self.taste_selection_layout)
        
        # Initially hide taste selection (since 3-taste mode is default)
        self.hide_taste_selection()
        
        # Taste mode status label
        self.taste_mode_status = QLabel("โหมดปัจจุบัน: 3 รสชาติ (M100, M110, M120)")
        self.taste_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px; font-weight: bold;")
        taste_mode_layout.addWidget(self.taste_mode_status)
        
        taste_mode_group.setLayout(taste_mode_layout)
        layout.addWidget(taste_mode_group)
        
        # Expiry Filtering Mode Group
        expiry_mode_group = QGroupBox("โหมดการคัดกรองวันหมดอายุ (BBF)")
        expiry_mode_layout = QVBoxLayout()
        
        # Expiry mode selection
        expiry_mode_selection_layout = QHBoxLayout()
        expiry_mode_selection_layout.addWidget(QLabel("เลือกโหมด:"))
        
        # Radio buttons for expiry modes
        self.expiry_mode_normal = QCheckBox("โหมดปกติ (ไม่คัดกรองวัน)")
        self.expiry_mode_normal.setChecked(True)  # Default to normal mode
        self.expiry_mode_normal.toggled.connect(self.on_expiry_mode_changed)
        self.expiry_mode_normal.setStyleSheet("QCheckBox { color: white; font-size: 12px; }")
        expiry_mode_selection_layout.addWidget(self.expiry_mode_normal)
        
        self.expiry_mode_filter = QCheckBox("โหมดคัดกรองวันหมดอายุ")
        self.expiry_mode_filter.setChecked(False)
        self.expiry_mode_filter.toggled.connect(self.on_expiry_mode_changed)
        self.expiry_mode_filter.setStyleSheet("QCheckBox { color: white; font-size: 12px; }")
        expiry_mode_selection_layout.addWidget(self.expiry_mode_filter)
        
        expiry_mode_selection_layout.addStretch()
        expiry_mode_layout.addLayout(expiry_mode_selection_layout)
        
        # Expiry filter type selection (initially hidden)
        self.expiry_filter_type_layout = QHBoxLayout()
        self.expiry_filter_type_layout.addWidget(QLabel("ประเภทการกรอง:"))
        
        # Filter type radio buttons
        self.expiry_filter_range = QCheckBox("ช่วงวันที่")
        self.expiry_filter_range.setChecked(True)  # Default to range
        self.expiry_filter_range.toggled.connect(self.on_expiry_filter_type_changed)
        self.expiry_filter_range.setStyleSheet("QCheckBox { color: white; font-size: 12px; }")
        self.expiry_filter_type_layout.addWidget(self.expiry_filter_range)
        
        self.expiry_filter_specific = QCheckBox("วันที่เฉพาะ")
        self.expiry_filter_specific.setChecked(False)
        self.expiry_filter_specific.toggled.connect(self.on_expiry_filter_type_changed)
        self.expiry_filter_specific.setStyleSheet("QCheckBox { color: white; font-size: 12px; }")
        self.expiry_filter_type_layout.addWidget(self.expiry_filter_specific)
        
        self.expiry_filter_type_layout.addStretch()
        expiry_mode_layout.addLayout(self.expiry_filter_type_layout)
        
        # Date selection controls (initially hidden)
        self.date_selection_layout = QVBoxLayout()
        
        # Range date selection
        self.range_date_layout = QHBoxLayout()
        self.range_date_layout.addWidget(QLabel("จากวันที่:"))
        self.expiry_start_date_edit = QDateTimeEdit()
        self.expiry_start_date_edit.setDate(QDateTime.currentDateTime().date())
        self.expiry_start_date_edit.setCalendarPopup(True)
        self.expiry_start_date_edit.setStyleSheet("QDateTimeEdit { color: white; background-color: #34495e; border: 1px solid #7f8c8d; }")
        self.expiry_start_date_edit.dateChanged.connect(self.update_expiry_mode_status)
        self.range_date_layout.addWidget(self.expiry_start_date_edit)
        
        self.range_date_layout.addWidget(QLabel("ถึงวันที่:"))
        self.expiry_end_date_edit = QDateTimeEdit()
        self.expiry_end_date_edit.setDate(QDateTime.currentDateTime().date())
        self.expiry_end_date_edit.setCalendarPopup(True)
        self.expiry_end_date_edit.setStyleSheet("QDateTimeEdit { color: white; background-color: #34495e; border: 1px solid #7f8c8d; }")
        self.expiry_end_date_edit.dateChanged.connect(self.update_expiry_mode_status)
        self.range_date_layout.addWidget(self.expiry_end_date_edit)
        
        self.range_date_layout.addStretch()
        self.date_selection_layout.addLayout(self.range_date_layout)
        
        # Specific date selection
        self.specific_date_layout = QHBoxLayout()
        self.specific_date_layout.addWidget(QLabel("วันที่เฉพาะ:"))
        self.expiry_specific_date_edit = QDateTimeEdit()
        self.expiry_specific_date_edit.setDate(QDateTime.currentDateTime().date())
        self.expiry_specific_date_edit.setCalendarPopup(True)
        self.expiry_specific_date_edit.setStyleSheet("QDateTimeEdit { color: white; background-color: #34495e; border: 1px solid #7f8c8d; }")
        self.expiry_specific_date_edit.dateChanged.connect(self.update_expiry_mode_status)
        self.specific_date_layout.addWidget(self.expiry_specific_date_edit)
        
        self.specific_date_layout.addStretch()
        self.date_selection_layout.addLayout(self.specific_date_layout)
        
        expiry_mode_layout.addLayout(self.date_selection_layout)
        
        # Initially hide expiry filter controls (since normal mode is default)
        self.hide_expiry_filter_controls()
        
        # Expiry mode status label
        self.expiry_mode_status = QLabel("โหมดปัจจุบัน: ปกติ (ไม่คัดกรองวันหมดอายุ)")
        self.expiry_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px; font-weight: bold;")
        expiry_mode_layout.addWidget(self.expiry_mode_status)
        
        expiry_mode_group.setLayout(expiry_mode_layout)
        layout.addWidget(expiry_mode_group)
        
        # Gripper Control Group
        gripper_group = QGroupBox("ควบคุม Gripper")
        gripper_layout = QHBoxLayout()
        
        # Gripper Open Button (M503)
        self.btn_gripper_open = QPushButton('🟢 เปิด Gripper (M503)')
        self.btn_gripper_open.clicked.connect(self.open_gripper)
        self.btn_gripper_open.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #27ae60; color: white; border-radius: 5px; } QPushButton:hover { background-color: #2ecc71; }")
        gripper_layout.addWidget(self.btn_gripper_open)
        
        # Gripper Close Button (M505)
        self.btn_gripper_close = QPushButton('🔴 ปิด Gripper (M505)')
        self.btn_gripper_close.clicked.connect(self.close_gripper)
        self.btn_gripper_close.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #e74c3c; color: white; border-radius: 5px; } QPushButton:hover { background-color: #c0392b; }")
        gripper_layout.addWidget(self.btn_gripper_close)
        
        # Gripper Status Label
        self.gripper_status_label = QLabel('🤖 สถานะ Gripper: รอคำสั่ง')
        self.gripper_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-weight: bold;")
        gripper_layout.addWidget(self.gripper_status_label)
        
        gripper_group.setLayout(gripper_layout)
        layout.addWidget(gripper_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Cap detection progress bar
        self.cap_progress_bar = QProgressBar()
        self.cap_progress_bar.setVisible(False)
        self.cap_progress_bar.setStyleSheet("QProgressBar { border: 2px solid #8e44ad; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #8e44ad; }")
        layout.addWidget(self.cap_progress_bar)
        
        # Status label
        self.status_label = QLabel('⏸️ โปรแกรมพร้อมทำงาน (รอ M511 เพื่อเริ่มการทำงาน)')
        self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Combined status label
        self.combined_status_label = QLabel('📊 สถานะรวม: กล้อง USB + Sentech พร้อมทำงาน')
        self.combined_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-weight: bold;")
        layout.addWidget(self.combined_status_label)
        
        # Queue info label
        self.queue_info_label = QLabel('📋 สถานะคิว: ไม่มีคิวรอ')
        self.queue_info_label.setStyleSheet("color: #e67e22; padding: 5px; font-size: 11px;")
        layout.addWidget(self.queue_info_label)
        
        # D6004 status label
        self.d6004_status_label = QLabel('🔍 D6004: รอค่า')
        self.d6004_status_label.setStyleSheet("color: #9b59b6; padding: 5px; font-size: 11px;")
        layout.addWidget(self.d6004_status_label)
        
        # D6007 status label
        self.d6007_status_label = QLabel('🔍 D6007: รอค่า')
        self.d6007_status_label.setStyleSheet("color: #9b59b6; padding: 5px; font-size: 11px;")
        layout.addWidget(self.d6007_status_label)
        
        # System time label (for real-time updates)
        self.system_time_label = QLabel('🕐 เวลาปัจจุบัน: กำลังโหลด...')
        self.system_time_label.setStyleSheet("color: #34495e; padding: 5px; font-size: 11px;")
        layout.addWidget(self.system_time_label)
        
        # Processing status label (for real-time updates)
        self.processing_status_label = QLabel('⏸️ ไม่มีการประมวลผล')
        self.processing_status_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 11px;")
        layout.addWidget(self.processing_status_label)
        
        # Create tab widget for bottle and cap detection
        self.tab_widget = QTabWidget()
        
        # Tab 1: Bottle Detection (USB Camera)
        bottle_tab = QWidget()
        bottle_layout = QVBoxLayout(bottle_tab)
        
        # Create main content area with splitter for bottle detection
        self.bottle_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Image display
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        
        image_title = QLabel("ภาพจากกล้อง USB")
        image_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2980b9;")
        image_title.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(image_title)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1;")
        self.image_label.setText("ยังไม่มีภาพจากกล้อง")
        image_layout.addWidget(self.image_label)
        
        # Current image info
        self.image_info_label = QLabel("ข้อมูลภาพ: -")
        self.image_info_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
        image_layout.addWidget(self.image_info_label)
        
        self.bottle_splitter.addWidget(image_widget)
        
        # Middle - Cropped images display
        crops_widget = QWidget()
        crops_layout = QVBoxLayout(crops_widget)
        
        crops_title = QLabel("ภาพที่ครอปและผล OCR")
        crops_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e67e22;")
        crops_title.setAlignment(Qt.AlignCenter)
        crops_layout.addWidget(crops_title)
        
        # Scroll area for cropped images
        self.crops_scroll = QScrollArea()
        self.crops_scroll.setWidgetResizable(True)
        self.crops_scroll.setMinimumSize(300, 400)
        self.crops_scroll.setStyleSheet("border: 2px solid #e67e22; background-color: #fef9e7;")
        
        # Container for cropped images
        self.crops_container = QWidget()
        self.crops_layout = QVBoxLayout(self.crops_container)
        self.crops_layout.setAlignment(Qt.AlignTop)
        
        # Add placeholder text
        placeholder_label = QLabel("ยังไม่มีภาพที่ครอป")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: #7f8c8d; padding: 20px;")
        self.crops_layout.addWidget(placeholder_label)
        
        self.crops_scroll.setWidget(self.crops_container)
        crops_layout.addWidget(self.crops_scroll)
        
        self.bottle_splitter.addWidget(crops_widget)
        
        # Right side - Results
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        results_title = QLabel("ผลการตรวจจับและ OCR")
        results_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c;")
        results_title.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(results_title)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        self.bottle_splitter.addWidget(results_widget)
        
        # Set splitter sizes
        self.bottle_splitter.setSizes([500, 300, 400])
        
        bottle_layout.addWidget(self.bottle_splitter)
        self.tab_widget.addTab(bottle_tab, "🔍 ตรวจจับขวด (USB Camera)")
        
        # Tab 2: Cap Detection (Sentech Camera)
        cap_tab = QWidget()
        cap_layout = QVBoxLayout(cap_tab)
        
        # Create main content area with splitter for cap detection
        self.cap_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Sentech Image display
        sentech_image_widget = QWidget()
        sentech_image_layout = QVBoxLayout(sentech_image_widget)
        
        sentech_image_title = QLabel("ภาพจากกล้อง Sentech")
        sentech_image_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #8e44ad;")
        sentech_image_title.setAlignment(Qt.AlignCenter)
        sentech_image_layout.addWidget(sentech_image_title)
        
        # Sentech Image display area
        self.sentech_image_label = QLabel()
        self.sentech_image_label.setMinimumSize(600, 500)  # Smaller size to show full image
        self.sentech_image_label.setAlignment(Qt.AlignCenter)
        self.sentech_image_label.setStyleSheet("border: 2px solid #8e44ad; background-color: #f4f3f4;")
        self.sentech_image_label.setText("ยังไม่มีภาพจากกล้อง Sentech")
        self.sentech_image_label.setScaledContents(False)  # Disable automatic scaling
        self.sentech_image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)  # Allow expansion
        sentech_image_layout.addWidget(self.sentech_image_label)
        
        # Sentech image info
        self.sentech_image_info_label = QLabel("ข้อมูลภาพ: -")
        self.sentech_image_info_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
        sentech_image_layout.addWidget(self.sentech_image_info_label)
        
        self.cap_splitter.addWidget(sentech_image_widget)
        
        # Middle - Cap detection results display with enhanced scroll
        cap_results_widget = QWidget()
        cap_results_layout = QVBoxLayout(cap_results_widget)
        
        cap_results_title = QLabel("ผลการตรวจจับฝาและข้อความ")
        cap_results_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e67e22;")
        cap_results_title.setAlignment(Qt.AlignCenter)
        cap_results_layout.addWidget(cap_results_title)
        
        # Enhanced Scroll area for cap detection results
        self.cap_results_scroll = QScrollArea()
        self.cap_results_scroll.setWidgetResizable(True)
        self.cap_results_scroll.setMinimumSize(300, 400)
        self.cap_results_scroll.setMaximumHeight(600)  # จำกัดความสูงสูงสุด
        self.cap_results_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # แสดง scroll bar เสมอ
        self.cap_results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # แสดง horizontal scroll bar ถ้าจำเป็น
        self.cap_results_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #e67e22; 
                background-color: #fef9e7;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #f0f0f0;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #c0c0c0;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        # Container for cap detection results
        self.cap_results_container = QWidget()
        self.cap_results_layout = QVBoxLayout(self.cap_results_container)
        self.cap_results_layout.setAlignment(Qt.AlignTop)
        self.cap_results_layout.setSpacing(10)  # เพิ่มระยะห่างระหว่าง widgets
        self.cap_results_layout.setContentsMargins(10, 10, 10, 10)  # เพิ่ม margin
        
        # Add placeholder text
        cap_placeholder_label = QLabel("ยังไม่มีผลการตรวจจับฝา")
        cap_placeholder_label.setAlignment(Qt.AlignCenter)
        cap_placeholder_label.setStyleSheet("color: #7f8c8d; padding: 20px; font-size: 14px;")
        self.cap_results_layout.addWidget(cap_placeholder_label)
        
        self.cap_results_scroll.setWidget(self.cap_results_container)
        cap_results_layout.addWidget(self.cap_results_scroll)
        
        self.cap_splitter.addWidget(cap_results_widget)
        
        # Right side - Cap detection results with scroll
        cap_detection_widget = QWidget()
        cap_detection_layout = QVBoxLayout(cap_detection_widget)
        
        cap_detection_title = QLabel("ผลการประมวลผลฝา")
        cap_detection_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c;")
        cap_detection_title.setAlignment(Qt.AlignCenter)
        cap_detection_layout.addWidget(cap_detection_title)
        
        # Cap detection results display with scroll
        self.cap_detection_text = QTextEdit()
        self.cap_detection_text.setReadOnly(True)
        self.cap_detection_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.cap_detection_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.cap_detection_text.setStyleSheet("""
            QTextEdit {
                border: 2px solid #e74c3c;
                border-radius: 5px;
                background-color: #fef9e7;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
            QScrollBar:vertical {
                background: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
        """)
        cap_detection_layout.addWidget(self.cap_detection_text)
        
        self.cap_splitter.addWidget(cap_detection_widget)
        
        # Set splitter sizes - ปรับให้เหมาะสมกับ Jetson
        self.cap_splitter.setSizes([600, 350, 450])
        
        # Add control buttons for cap detection
        cap_control_group = QGroupBox("ตัวเลือกการทำงานฝา")
        cap_control_layout = QHBoxLayout()
        
        # Sentech capture button
        self.btn_sentech_capture = QPushButton('📸 ถ่ายภาพจาก Sentech')
        self.btn_sentech_capture.clicked.connect(self.capture_sentech_image)
        self.btn_sentech_capture.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #8e44ad; color: white; border-radius: 5px; }")
        cap_control_layout.addWidget(self.btn_sentech_capture)
        
        # Select image file button for cap detection
        self.btn_select_cap_image = QPushButton('📁 เลือกไฟล์ภาพฝา')
        self.btn_select_cap_image.clicked.connect(self.select_cap_image_file)
        self.btn_select_cap_image.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #3498db; color: white; border-radius: 5px; }")
        cap_control_layout.addWidget(self.btn_select_cap_image)
        
        # Manual rotation controls removed
        
        # Process cap button
        self.btn_process_cap = QPushButton('🔍 ประมวลผลฝา')
        self.btn_process_cap.clicked.connect(self.process_cap_detection)
        self.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
        self.btn_process_cap.setVisible(True)  # VISIBLE - cap processing turned on
        self.btn_process_cap.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #95a5a6; color: white; border-radius: 5px; }")
        cap_control_layout.addWidget(self.btn_process_cap)
        
        cap_control_group.setLayout(cap_control_layout)
        cap_layout.addWidget(cap_control_group)
        
        # Add process steps info
        cap_steps_group = QGroupBox("ขั้นตอนการประมวลผลฝา")
        cap_steps_layout = QVBoxLayout()
        
        steps_text = """
🔄 ขั้นตอนการประมวลผลฝา:

Step 1: ตรวจจับฝาด้วย Cap Detector
Step 2: ตัดภาพที่ตรวจจับได้
Step 2a: หากไม่พบฝา → ตรวจจับข้อความด้วย CRAFT
Step 2b: ประมวลผลด้วย AI Rotation
Step 3: ตรวจจับบรรทัดข้อความ
Step 4: อ่านข้อความด้วย OCR

⏱️ เวลาที่ใช้: ประมาณ 5-15 วินาที
        """
        
        steps_label = QLabel(steps_text)
        steps_label.setStyleSheet("color: #2c3e50; font-size: 11px; background-color: #ecf0f1; padding: 10px; border-radius: 5px;")
        steps_label.setWordWrap(True)
        cap_steps_layout.addWidget(steps_label)
        
        cap_steps_group.setLayout(cap_steps_layout)
        cap_layout.addWidget(cap_steps_group)
        
        cap_layout.addWidget(self.cap_splitter)
        self.tab_widget.addTab(cap_tab, "🔍 ตรวจจับฝา")
        
        # Tab 3: Modbus Status
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
        connection_layout.setContentsMargins(10, 15, 10, 10)  # left, top, right, bottom
        
        self.modbus_connection_lamp = QLabel("●")
        self.modbus_connection_lamp.setStyleSheet("color: #e74c3c; font-size: 24px; font-weight: bold;")
        self.modbus_connection_lamp.setAlignment(Qt.AlignCenter)
        connection_layout.addWidget(self.modbus_connection_lamp)
        
        self.modbus_connection_text = QLabel("ไม่เชื่อมต่อ")
        self.modbus_connection_text.setStyleSheet("color: #e74c3c; font-size: 14px; font-weight: bold;")
        connection_layout.addWidget(self.modbus_connection_text)
        
        status_content_layout.addWidget(connection_group)
        
        # Modbus Coils Status
        coils_group = QGroupBox("⚡ Coils Status")
        coils_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
        coils_layout = QVBoxLayout(coils_group)
        coils_layout.setContentsMargins(10, 15, 10, 10)  # left, top, right, bottom
        coils_layout.setSpacing(10)  # ระยะห่างระหว่างแถว
        
        # M100 - M130 (Bottle Types)
        bottle_types_layout = QHBoxLayout()
        bottle_types_layout.setSpacing(8)  # ระยะห่างระหว่างปุ่ม
        bottle_types_label = QLabel("ประเภทขวด:")
        bottle_types_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
        bottle_types_layout.addWidget(bottle_types_label)
        
        self.m100_lamp = QLabel("M100")
        self.m100_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m100_lamp.setAlignment(Qt.AlignCenter)
        bottle_types_layout.addWidget(self.m100_lamp)
        
        self.m110_lamp = QLabel("M110")
        self.m110_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m110_lamp.setAlignment(Qt.AlignCenter)
        bottle_types_layout.addWidget(self.m110_lamp)
        
        self.m120_lamp = QLabel("M120")
        self.m120_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m120_lamp.setAlignment(Qt.AlignCenter)
        bottle_types_layout.addWidget(self.m120_lamp)
        
        self.m130_lamp = QLabel("M130")
        self.m130_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m130_lamp.setAlignment(Qt.AlignCenter)
        bottle_types_layout.addWidget(self.m130_lamp)
        
        coils_layout.addLayout(bottle_types_layout)
        
        # M140 - M850 (Control Coils)
        control_coils_layout = QHBoxLayout()
        control_coils_layout.setSpacing(8)  # ระยะห่างระหว่างปุ่ม
        control_coils_label = QLabel("Control:")
        control_coils_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
        control_coils_layout.addWidget(control_coils_label)
        
        self.m140_lamp = QLabel("M140")
        self.m140_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m140_lamp.setAlignment(Qt.AlignCenter)
        control_coils_layout.addWidget(self.m140_lamp)
        
        self.m600_lamp = QLabel("M600")
        self.m600_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m600_lamp.setAlignment(Qt.AlignCenter)
        control_coils_layout.addWidget(self.m600_lamp)
        
        self.m700_lamp = QLabel("M700")
        self.m700_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m700_lamp.setAlignment(Qt.AlignCenter)
        control_coils_layout.addWidget(self.m700_lamp)
        
        self.m701_lamp = QLabel("M701")
        self.m701_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m701_lamp.setAlignment(Qt.AlignCenter)
        control_coils_layout.addWidget(self.m701_lamp)
        
        self.m750_lamp = QLabel("M750")
        self.m750_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m750_lamp.setAlignment(Qt.AlignCenter)
        control_coils_layout.addWidget(self.m750_lamp)
        
        self.m850_lamp = QLabel("M850")
        self.m850_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m850_lamp.setAlignment(Qt.AlignCenter)
        control_coils_layout.addWidget(self.m850_lamp)
        
        coils_layout.addLayout(control_coils_layout)
        
        # Gripper Control
        gripper_layout = QHBoxLayout()
        gripper_layout.setSpacing(8)  # ระยะห่างระหว่างปุ่ม
        gripper_label = QLabel("Gripper:")
        gripper_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
        gripper_layout.addWidget(gripper_label)
        
        self.m503_lamp = QLabel("M503")
        self.m503_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m503_lamp.setAlignment(Qt.AlignCenter)
        gripper_layout.addWidget(self.m503_lamp)
        
        self.m505_lamp = QLabel("M505")
        self.m505_lamp.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.m505_lamp.setAlignment(Qt.AlignCenter)
        gripper_layout.addWidget(self.m505_lamp)
        
        coils_layout.addLayout(gripper_layout)
        
        status_content_layout.addWidget(coils_group)
        
        # Performance Statistics
        performance_group = QGroupBox("📈 สถิติประสิทธิภาพ")
        performance_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
        performance_layout = QVBoxLayout(performance_group)
        performance_layout.setContentsMargins(10, 15, 10, 10)
        performance_layout.setSpacing(10)
        
        # Images processed
        self.total_images_processed = QLabel("จำนวนภาพที่ประมวลผล: 0")
        self.total_images_processed.setStyleSheet("color: #2c3e50; padding: 5px; font-size: 12px;")
        performance_layout.addWidget(self.total_images_processed)
        
        # Successful detections
        self.successful_detections = QLabel("การตรวจจับที่สำเร็จ: 0")
        self.successful_detections.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
        performance_layout.addWidget(self.successful_detections)
        
        # Error count
        self.error_count = QLabel("จำนวนข้อผิดพลาด: 0")
        self.error_count.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
        performance_layout.addWidget(self.error_count)
        
        # Success rate
        self.success_rate = QLabel("อัตราความสำเร็จ: 0.0%")
        self.success_rate.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
        performance_layout.addWidget(self.success_rate)
        
        status_content_layout.addWidget(performance_group)
        
        # Camera Status
        camera_group = QGroupBox("📷 สถานะกล้อง")
        camera_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setContentsMargins(10, 15, 10, 10)
        camera_layout.setSpacing(10)
        
        # USB Camera status
        self.usb_camera_status = QLabel("USB Camera: กำลังเริ่มต้น...")
        self.usb_camera_status.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
        camera_layout.addWidget(self.usb_camera_status)
        
        # Sentech Camera status
        self.sentech_camera_status = QLabel("Sentech Camera: กำลังเริ่มต้น...")
        self.sentech_camera_status.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
        camera_layout.addWidget(self.sentech_camera_status)
        
        status_content_layout.addWidget(camera_group)
        
        # System Status
        system_group = QGroupBox("🖥️ สถานะระบบ")
        system_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
        system_layout = QVBoxLayout(system_group)
        system_layout.setContentsMargins(10, 15, 10, 10)
        system_layout.setSpacing(10)
        
        # Processing status
        self.bottle_detection_status = QLabel("การตรวจจับขวด: พร้อม")
        self.bottle_detection_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
        system_layout.addWidget(self.bottle_detection_status)
        
        # Cap detection status
        self.cap_detection_status = QLabel("การตรวจจับฝา: พร้อม")
        self.cap_detection_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
        system_layout.addWidget(self.cap_detection_status)
        
        # Current bottle type
        self.current_bottle_type_status = QLabel("ประเภทขวดปัจจุบัน: -")
        self.current_bottle_type_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
        system_layout.addWidget(self.current_bottle_type_status)
        
        # Processing mode
        self.processing_mode_status = QLabel("โหมดการประมวลผล: Manual")
        self.processing_mode_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
        system_layout.addWidget(self.processing_mode_status)
        
        # Angle3 retry status
        self.angle3_retry_status = QLabel("โหมดถ่ายภาพซ้ำ: ปิด")
        self.angle3_retry_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
        system_layout.addWidget(self.angle3_retry_status)
        
        status_content_layout.addWidget(system_group)
        
        # Queue Status
        queue_group = QGroupBox("📋 สถานะคิว")
        queue_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
        queue_layout = QVBoxLayout(queue_group)
        queue_layout.setContentsMargins(10, 15, 10, 10)
        queue_layout.setSpacing(10)
        
        # Queue count
        self.queue_count_status = QLabel("จำนวนคิว: 0")
        self.queue_count_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
        queue_layout.addWidget(self.queue_count_status)
        
        status_content_layout.addWidget(queue_group)
        
        # Modbus Registers Status
        registers_group = QGroupBox("📊 Registers Status")
        registers_group.setStyleSheet("QGroupBox { font-weight: bold; color: #8e44ad; font-size: 14px; padding: 10px; margin: 5px; }")
        registers_layout = QVBoxLayout(registers_group)
        registers_layout.setContentsMargins(10, 15, 10, 10)  # left, top, right, bottom
        registers_layout.setSpacing(10)  # ระยะห่างระหว่างแถว
        
        # D6004 Status
        d6004_layout = QHBoxLayout()
        d6004_layout.setSpacing(10)  # ระยะห่างระหว่าง elements
        d6004_label = QLabel("D6004:")
        d6004_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
        d6004_layout.addWidget(d6004_label)
        
        self.d6004_lamp = QLabel("●")
        self.d6004_lamp.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
        self.d6004_lamp.setAlignment(Qt.AlignCenter)
        d6004_layout.addWidget(self.d6004_lamp)
        
        self.d6004_text = QLabel("รอค่า")
        self.d6004_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
        d6004_layout.addWidget(self.d6004_text)
        
        registers_layout.addLayout(d6004_layout)
        
        # D6007 Status
        d6007_layout = QHBoxLayout()
        d6007_layout.setSpacing(10)  # ระยะห่างระหว่าง elements
        d6007_label = QLabel("D6007:")
        d6007_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
        d6007_layout.addWidget(d6007_label)
        
        self.d6007_lamp = QLabel("●")
        self.d6007_lamp.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
        self.d6007_lamp.setAlignment(Qt.AlignCenter)
        d6007_layout.addWidget(self.d6007_lamp)
        
        self.d6007_text = QLabel("รอค่า")
        self.d6007_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
        d6007_layout.addWidget(self.d6007_text)
        
        registers_layout.addLayout(d6007_layout)
        
        # D6006 Status
        d6006_layout = QHBoxLayout()
        d6006_layout.setSpacing(10)  # ระยะห่างระหว่าง elements
        d6006_label = QLabel("D6006:")
        d6006_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-right: 10px;")
        d6006_layout.addWidget(d6006_label)
        
        self.d6006_lamp = QLabel("●")
        self.d6006_lamp.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
        self.d6006_lamp.setAlignment(Qt.AlignCenter)
        d6006_layout.addWidget(self.d6006_lamp)
        
        self.d6006_text = QLabel("รอค่า")
        self.d6006_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;")
        d6006_layout.addWidget(self.d6006_text)
        
        registers_layout.addLayout(d6006_layout)
        
        status_content_layout.addWidget(registers_group)
        
        # เพิ่ม spacing ระหว่าง groups
        status_content_layout.addSpacing(15)
        
        # Add stretch to push everything to top
        status_content_layout.addStretch()
        
        # Set content to scroll area
        status_scroll.setWidget(status_content)
        status_layout.addWidget(status_scroll)
        
        self.tab_widget.addTab(status_tab, "📊 สถานะระบบ")
        
        # Tab 4: OCR Test
        ocr_test_tab = QWidget()
        ocr_test_layout = QVBoxLayout(ocr_test_tab)
        
        # OCR Test Title
        ocr_test_title = QLabel("🔤 ทดสอบ OCR")
        ocr_test_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; padding: 15px;")
        ocr_test_title.setAlignment(Qt.AlignCenter)
        ocr_test_layout.addWidget(ocr_test_title)
        
        # Create splitter for OCR test
        self.ocr_test_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Image selection and display
        ocr_image_widget = QWidget()
        ocr_image_layout = QVBoxLayout(ocr_image_widget)
        
        # Image selection controls
        ocr_controls_group = QGroupBox("เลือกภาพ")
        ocr_controls_layout = QVBoxLayout(ocr_controls_group)
        
        # Select image button
        self.btn_select_ocr_image = QPushButton('📁 เลือกภาพสำหรับทดสอบ OCR')
        self.btn_select_ocr_image.clicked.connect(self.select_ocr_test_image)
        self.btn_select_ocr_image.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #3498db; color: white; }")
        ocr_controls_layout.addWidget(self.btn_select_ocr_image)
        
        # Process OCR button
        self.btn_process_ocr = QPushButton('🔤 ประมวลผล OCR')
        self.btn_process_ocr.clicked.connect(self.process_ocr_test)
        self.btn_process_ocr.setEnabled(False)
        self.btn_process_ocr.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #27ae60; color: white; }")
        ocr_controls_layout.addWidget(self.btn_process_ocr)
        
        # Enhanced OCR processing button
        self.btn_process_ocr_enhanced = QPushButton('🔤 ประมวลผล OCR (ปรับปรุง)')
        self.btn_process_ocr_enhanced.clicked.connect(self.process_ocr_test_enhanced)
        self.btn_process_ocr_enhanced.setEnabled(False)
        self.btn_process_ocr_enhanced.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; background-color: #FF9800; color: white; }")
        ocr_controls_layout.addWidget(self.btn_process_ocr_enhanced)
        
        # OCR status label
        self.ocr_status_label = QLabel('⏸️ เลือกภาพเพื่อเริ่มทดสอบ OCR')
        self.ocr_status_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 11px;")
        ocr_controls_layout.addWidget(self.ocr_status_label)
        
        ocr_controls_group.setLayout(ocr_controls_layout)
        ocr_image_layout.addWidget(ocr_controls_group)
        
        # Image display
        self.ocr_image_label = QLabel("ไม่มีภาพ")
        self.ocr_image_label.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1; color: #7f8c8d;")
        self.ocr_image_label.setAlignment(Qt.AlignCenter)
        self.ocr_image_label.setMinimumHeight(300)
        ocr_image_layout.addWidget(self.ocr_image_label)
        
        # Right side - OCR results
        ocr_results_widget = QWidget()
        ocr_results_layout = QVBoxLayout(ocr_results_widget)
        
        # OCR results title
        ocr_results_title = QLabel("ผลลัพธ์ OCR")
        ocr_results_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c;")
        ocr_results_title.setAlignment(Qt.AlignCenter)
        ocr_results_layout.addWidget(ocr_results_title)
        
        # OCR results display
        self.ocr_results_text = QTextEdit()
        self.ocr_results_text.setReadOnly(True)
        self.ocr_results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 2px solid #e74c3c;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
                color: #2c3e50;
            }
        """)
        self.ocr_results_text.setPlaceholderText("ผลลัพธ์ OCR จะแสดงที่นี่...")
        ocr_results_layout.addWidget(self.ocr_results_text)
        
        # OCR processing progress
        self.ocr_progress_bar = QProgressBar()
        self.ocr_progress_bar.setVisible(False)
        self.ocr_progress_bar.setStyleSheet("QProgressBar { border: 2px solid #e74c3c; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #e74c3c; }")
        ocr_results_layout.addWidget(self.ocr_progress_bar)
        
        # Add widgets to splitter
        self.ocr_test_splitter.addWidget(ocr_image_widget)
        self.ocr_test_splitter.addWidget(ocr_results_widget)
        
        # Set splitter sizes
        self.ocr_test_splitter.setSizes([400, 400])
        
        ocr_test_layout.addWidget(self.ocr_test_splitter)
        
        self.tab_widget.addTab(ocr_test_tab, "🔤 ทดสอบ OCR")
        
        layout.addWidget(self.tab_widget)
        
        # เพิ่ม main scroll area เข้าไปใน layout หลัก
        main_scroll_area.setWidget(main_widget)
        
        # สร้าง main layout สำหรับหน้าต่างหลัก
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(main_scroll_area)
        
    def setup_camera(self):
        """Setup USB camera"""
        try:
            self.usb_camera = USBCamera()
            if self.usb_camera.open_camera():
                self.camera_status_label.setText('📷 กล้อง: พร้อมใช้งาน')
                self.camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                self.update_usb_camera_status("พร้อมใช้งาน", True)
                print("✅ USB Camera initialized successfully")
            else:
                self.camera_status_label.setText('📷 กล้อง: ไม่สามารถเปิดได้')
                self.camera_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                self.update_usb_camera_status("ไม่สามารถเปิดได้", False)
                print("❌ Failed to initialize USB Camera")
        except Exception as e:
            self.camera_status_label.setText(f'📷 กล้อง: ข้อผิดพลาด - {str(e)}')
            self.camera_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            self.update_usb_camera_status(f"ข้อผิดพลาด - {str(e)}", False)
            print(f"❌ Camera setup error: {e}")
    

    def setup_sentech_camera(self):
        """Setup Sentech camera for cap detection using Harvesters only"""
        try:
            print("🔧 Setting up SENTECH Camera with Harvesters...")
            
            # Create SentechCamera instance with Harvesters
            self.sentech_camera = SentechCamera()
            
            # Initialize camera
            print("🔧 Initializing SENTECH camera...")
            success = self.sentech_camera.initialize()
            
            if success:
                self.sentech_camera_status_label.setText('📷 Sentech: พร้อมใช้งาน (Harvesters)')
                self.sentech_camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                self.update_sentech_camera_status("พร้อมใช้งาน", True)
                print("✅ SENTECH camera initialized successfully with Harvesters")
            else:
                self.sentech_camera_status_label.setText('📷 Sentech: ไม่สามารถเชื่อมต่อได้')
                self.sentech_camera_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                self.update_sentech_camera_status("ไม่สามารถเชื่อมต่อได้", False)
                print("❌ Failed to initialize SENTECH camera")
            
        except Exception as e:
            self.sentech_camera_status_label.setText(f'📷 Sentech: ข้อผิดพลาด - {str(e)}')
            self.sentech_camera_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            self.update_sentech_camera_status(f"ข้อผิดพลาด - {str(e)}", False)
            print(f"❌ SENTECH Camera setup error: {e}")
    
    def setup_cap_detection_models(self):
        """Setup cap detection models - ENABLED (cap processing turned on)"""
        print("🔧 Setting up cap detection models...")
        
        # ENABLED CODE - cap processing turned on
        if not CAP_DETECTION_AVAILABLE:
            print("⚠️ Cap detection modules not available")
            return
            
        try:
            print("🔧 Setting up cap detection models...")
            
            # Initialize cap detector
            self.cap_detector = initialize_detector(CAP_MODEL_PATH)
            print("✅ Cap detector initialized")
            
            # Initialize CRAFT detector
            self.craft_detector = initialize_craft_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
            print("✅ CRAFT detector initialized")
            
            # Initialize rotation model
            self.rotation_model = initialize_model(ROTATION_MODEL_PATH)
            print("✅ Rotation model initialized")
            
            # Initialize CRAFT detector in rotation model
            self.rotation_model.init_craft_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
            print("✅ CRAFT detector initialized in rotation model")
            
            # Initialize line detector
            self.line_detector = initialize_line_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
            print("✅ Line detector initialized")
            
            # Initialize OCR model
            self.ocr_model = initialize_ocr_model(OCR_MODEL_PATH)
            print("✅ OCR model initialized")
            
            # Optimize OCR settings for better number recognition
            self.optimize_ocr_for_numbers()
            
            # Initialize YOLO model for faded text detection
            if YOLO_AVAILABLE:
                try:
                    self.faded_text_yolo_model = YOLO(CAP_MODEL_PATH)
                    print("✅ YOLO model for faded text detection initialized")
                except Exception as e:
                    print(f"⚠️ Failed to initialize YOLO model for faded text detection: {e}")
                    self.faded_text_yolo_model = None
            else:
                print("⚠️ YOLO not available for faded text detection")
                self.faded_text_yolo_model = None
            
            print("✅ All cap detection models initialized successfully")
            
        except Exception as e:
            print(f"❌ Error setting up cap detection models: {e}")
    
    def optimize_ocr_for_numbers(self):
        """
        ปรับปรุงการตั้งค่า DeepOCR เพื่อให้อ่านเลขได้ดีขึ้น
        """
        try:
            if self.ocr_model is None:
                print("⚠️ OCR model not available for optimization")
                return
            
            print("🔧 Optimizing OCR settings for better number recognition...")
            
            # ปรับปรุงการตั้งค่าสำหรับการอ่านเลข
            optimized_settings = {
                'imgH': 64,  # เพิ่มความสูงจาก 32 เป็น 64 เพื่อให้อ่านเลขได้ชัดขึ้น
                'imgW': 200,  # เพิ่มความกว้างจาก 100 เป็น 200 เพื่อให้อ่านเลขหลายตัวได้ดีขึ้น
                'batch_max_length': 10,  # ลดจาก 25 เป็น 10 เพราะเราต้องการอ่านเลขสั้นๆ
                'character': '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:/',  # เน้นตัวเลข
                'sensitive': True,
                'PAD': True,  # เปิดใช้ padding เพื่อให้ภาพมีขนาดสม่ำเสมอ
            }
            
            # อัปเดตการตั้งค่า
            self.ocr_model.update_settings(**optimized_settings)
            
            print("✅ OCR settings optimized for number recognition:")
            print(f"   - Image height: {optimized_settings['imgH']}")
            print(f"   - Image width: {optimized_settings['imgW']}")
            print(f"   - Max length: {optimized_settings['batch_max_length']}")
            print(f"   - Padding enabled: {optimized_settings['PAD']}")
            
        except Exception as e:
            print(f"❌ Error optimizing OCR settings: {e}")
    
    def preprocess_image_for_ocr(self, image):
        """
        ปรับปรุงภาพก่อนส่งให้ OCR เพื่อให้อ่านเลขได้ดีขึ้น
        """
        try:
            if image is None:
                return image
            
            print("🔧 Preprocessing image for better OCR...")
            
            # แปลงเป็น grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # เพิ่มขนาดภาพเพื่อให้อ่านได้ชัดขึ้น
            height, width = gray.shape
            scale_factor = 2.0  # เพิ่มขนาด 2 เท่า
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize ด้วย INTER_CUBIC เพื่อให้ภาพชัดขึ้น
            resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # ปรับปรุงความคมชัดด้วย Gaussian blur + unsharp mask
            blurred = cv2.GaussianBlur(resized, (0, 0), 2.0)
            sharpened = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
            
            # ปรับปรุงความคมชัดด้วย CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened)
            
            # ปรับปรุงความคมชัดด้วย morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # แปลงกลับเป็น BGR format สำหรับ DeepOCR
            if len(image.shape) == 3:
                result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                result = enhanced
            
            print(f"✅ Image preprocessed: {width}x{height} -> {new_width}x{new_height}")
            return result
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return image
    
    def setup_modbus(self):
        """Setup Modbus connection"""
        try:
            print("🔧 Setting up Modbus connection...")
            self.modbus_thread = ModbusThread("192.168.1.5", 502)
            self.modbus_thread.modbus_status.connect(self.update_modbus_status)
            self.modbus_thread.trigger_detected.connect(self.on_modbus_trigger)
            self.modbus_thread.queue_trigger_detected.connect(self.on_queue_trigger)
            self.modbus_thread.bottle_type_detected.connect(self.on_bottle_type_detected)
            self.modbus_thread.d6004_status_updated.connect(self.update_d6004_status)
            self.modbus_thread.d6007_status_updated.connect(self.update_d6007_status)
            self.modbus_thread.reset_processing_signal.connect(self.reset_processing_for_new_image)
            self.modbus_thread.m513_stop_signal.connect(self.on_m513_stop)
            print("🔧 Starting Modbus thread...")
            self.modbus_thread.start()
            print("✅ Modbus thread started successfully")
            
            # Update status tab
            self.update_modbus_connection_status(True)
            
            # รอให้ Modbus เชื่อมต่อสำเร็จก่อน
            QTimer.singleShot(2000, self.initialize_modbus_state)
            
            print("⏸️ โปรแกรมพร้อมทำงาน (รอ M511 เพื่อเริ่มการทำงาน)")
        except Exception as e:
            self.modbus_status_label.setText(f'📡 Modbus: ข้อผิดพลาด - {str(e)}')
            self.modbus_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            self.update_modbus_connection_status(False)
            print(f"❌ Modbus setup error: {e}")
    
    def initialize_modbus_state(self):
        """Initialize Modbus state when program starts"""
        try:
            print("🔧 Initializing Modbus state...")
            
            # Reset M100, M110, M120, M130, M750, M600
            if self.modbus_thread:
                success = True
                success &= self.modbus_thread.reset_m100()
                success &= self.modbus_thread.reset_m110()
                success &= self.modbus_thread.reset_m120()
                success &= self.modbus_thread.reset_m130()
                success &= self.modbus_thread.reset_m750()
                success &= self.modbus_thread.reset_m600()
                
                # Stop D6006 monitoring
                self.modbus_thread.d6006_monitoring = False
                
                if success:
                    print("✅ RESET M100, M110, M120, M130, M750, M600 = 0")
                    self.modbus_status_label.setText('📡 Modbus: Reset bottle types and M600 completed')
                else:
                    print("❌ Failed to reset bottle types and M600")
                
                # Set initial taste mode based on current_taste_mode
                print(f"🔧 Setting initial taste mode: {self.current_taste_mode}")
                if self.current_taste_mode == 1:
                    # 1-taste mode: ON M722
                    success = self.modbus_thread.on_m722()
                    if success:
                        print("✅ ON M722 (โหมด 1 รสชาติ) - Initial setup")
                        self.update_coil_lamp("m722", True)
                    else:
                        print("❌ Failed to ON M722")
                elif self.current_taste_mode == 2:
                    # 2-taste mode: ON M721
                    success = self.modbus_thread.on_m721()
                    if success:
                        print("✅ ON M721 (โหมด 2 รสชาติ) - Initial setup")
                        self.update_coil_lamp("m721", True)
                    else:
                        print("❌ Failed to ON M721")
                elif self.current_taste_mode == 3:
                    # 3-taste mode: ON M720
                    success = self.modbus_thread.on_m720()
                    if success:
                        print("✅ ON M720 (โหมด 3 รสชาติ) - Initial setup")
                        self.update_coil_lamp("m720", True)
                    else:
                        print("❌ Failed to ON M720")
                
                # ON M503 แล้ว OFF (pulse)
                try:
                    # ON M503
                    result_on = self.modbus_thread.modbus_client.write_coil(503, True, unit=1)
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
            if self.modbus_thread and self.modbus_thread.modbus_client:
                result = self.modbus_thread.modbus_client.write_coil(503, False, unit=1)
                if not result.isError():
                    print("✅ OFF M503 = 0")
                    self.modbus_status_label.setText('📡 Modbus: M503 pulse completed')
                else:
                    print("❌ Failed to OFF M503")
        except Exception as e:
            print(f"❌ Error turning OFF M503: {e}")
    
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
            if self.usb_camera and self.usb_camera.is_running:
                self.update_usb_camera_status("พร้อมใช้งาน", True)
            else:
                self.update_usb_camera_status("ไม่พร้อมใช้งาน", False)
            
            # Sentech Camera status
            if self.sentech_camera and self.sentech_camera._is_initialized:
                self.update_sentech_camera_status("พร้อมใช้งาน", True)
            else:
                self.update_sentech_camera_status("ไม่พร้อมใช้งาน", False)
                
        except Exception as e:
            print(f"❌ Error updating camera status: {e}")
    
    def update_modbus_connection_status(self):
        """Update Modbus connection status continuously"""
        try:
            if self.modbus_thread and self.modbus_thread.is_running:
                # ตรวจสอบการเชื่อมต่อ Modbus
                if self.modbus_thread.modbus_client and self.modbus_thread.modbus_client.is_socket_open():
                    self.update_modbus_connection_status(True)
                else:
                    self.update_modbus_connection_status(False)
            else:
                self.update_modbus_connection_status(False)
        except Exception as e:
            print(f"❌ Error updating Modbus connection status: {e}")
    
    def update_system_status(self):
        """Update system status continuously"""
        try:
            # อัพเดทเวลาปัจจุบัน
            current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
            if hasattr(self, 'system_time_label'):
                self.system_time_label.setText(f"🕐 เวลาปัจจุบัน: {current_time}")
            
            # อัพเดทสถานะการประมวลผล
            if hasattr(self, 'processing_status_label'):
                if self.current_image is not None:
                    self.processing_status_label.setText("🔄 กำลังประมวลผลภาพ")
                    self.processing_status_label.setStyleSheet("color: #f39c12; padding: 5px;")
                else:
                    self.processing_status_label.setText("⏸️ ไม่มีการประมวลผล")
                    self.processing_status_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
                    
        except Exception as e:
            print(f"❌ Error updating system status: {e}")

    def update_modbus_status(self, status):
        """Update Modbus status display"""
        self.modbus_status_label.setText(f'📡 Modbus: {status}')
        
        # อัปเดตสถานะคิว
        if "เพิ่มคิวถ่ายภาพ" in status:
            # แยกตัวเลขคิวออกจากข้อความ
            try:
                queue_text = status.split("คิวปัจจุบัน: ")[1].split(")")[0]
                queue_count = int(queue_text)
                self.update_queue_status(queue_count)
                # อัปเดต status label หลัก
                self.status_label.setText(f'📋 มีคิวถ่ายภาพ {queue_count} รายการ รอ M600 reset')
                self.status_label.setStyleSheet("color: #e67e22; padding: 5px;")
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
            self.status_label.setText('✅ ไม่มีคิวถ่ายภาพ พร้อมรับ M301 ใหม่')
            self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        
        # อัปเดตสีตามสถานะ
        if "สำเร็จ" in status or "เชื่อมต่อ" in status:
            self.modbus_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        elif "ข้อผิดพลาด" in status or "ไม่สามารถ" in status:
            self.modbus_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        elif "M511 ON" in status:
            self.modbus_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            self.status_label.setText('✅ โปรแกรมเริ่มทำงาน - รอสัญญาณ M301')
            self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        elif "M511 OFF" in status:
            self.modbus_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            self.status_label.setText('⏸️ โปรแกรมหยุดทำงาน (รอ M511 เพื่อเริ่มการทำงาน)')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        elif "M513 ON" in status:
            self.modbus_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            self.status_label.setText('🛑 โปรแกรมหยุดการทำงาน')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        else:
            self.modbus_status_label.setStyleSheet("color: #f39c12; padding: 5px;")
    
    def update_queue_status(self, queue_count):
        """Update queue status display"""
        if queue_count > 0:
            self.queue_status_label.setText(f'📋 คิว: {queue_count}')
            self.queue_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
            self.queue_info_label.setText(f'📋 สถานะคิว: มีคิวรอ {queue_count} รายการ')
            self.queue_info_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
        else:
            self.queue_status_label.setText('📋 คิว: 0')
            self.queue_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-weight: bold;")
            self.queue_info_label.setText('📋 สถานะคิว: ไม่มีคิวรอ')
            self.queue_info_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
        
        # Update status tab
        self.queue_count_status.setText(f"จำนวนคิว: {queue_count}")
        if queue_count > 0:
            self.queue_count_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 12px;")
        else:
            self.queue_count_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
    
    def update_d6004_status(self, d6004_value):
        """Update D6004 status display"""
        if d6004_value is None:
            self.d6004_status_label.setText('🔍 D6004: ไม่สามารถอ่านได้')
            self.d6004_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6004", "ไม่สามารถอ่านได้", False)
        elif d6004_value == 200:
            self.d6004_status_label.setText('✅ D6004: 200 (พร้อม Reset)')
            self.d6004_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6004", "200", True)
        else:
            self.d6004_status_label.setText(f'⏳ D6004: {d6004_value} (รอ 200)')
            self.d6004_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6004", str(d6004_value), False)
    
    def update_d6007_status(self, d6007_value):
        """Update D6007 status display"""
        if d6007_value is None:
            self.d6007_status_label.setText('🔍 D6007: ไม่สามารถอ่านได้')
            self.d6007_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", "ไม่สามารถอ่านได้", False)
        elif d6007_value == 100:
            self.d6007_status_label.setText('🥛 D6007: 100 (น้ำเต้าหู้รสดั้งเดิม)')
            self.d6007_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", "100", True)
            # แสดง dialog แจ้งเตือนผู้ใช้
            self.show_d6007_dialog(100, "น้ำเต้าหู้รสดั้งเดิม", "🥛")
        elif d6007_value == 200:
            self.d6007_status_label.setText('🍯 D6007: 200 (น้ำตาลน้อย 2%)')
            self.d6007_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", "200", True)
            # แสดง dialog แจ้งเตือนผู้ใช้
            self.show_d6007_dialog(200, "น้ำตาลน้อย 2%", "🍯")
        elif d6007_value == 300:
            self.d6007_status_label.setText('🌿 D6007: 300 (ผสมเม็ดแมงลัก)')
            self.d6007_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", "300", True)
            # แสดง dialog แจ้งเตือนผู้ใช้
            self.show_d6007_dialog(300, "ผสมเม็ดแมงลัก", "🌿")
        elif d6007_value == 400:
            self.d6007_status_label.setText('❌ D6007: 400 (NG เต็ม)')
            self.d6007_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", "400", True)
            # แสดง dialog แจ้งเตือนผู้ใช้
            self.show_d6007_dialog(400, "NG เต็ม", "❌")
        else:
            self.d6007_status_label.setText(f'⏳ D6007: {d6007_value} (รอ 100/200/300/400)')
            self.d6007_status_label.setStyleSheet("color: #9b59b6; padding: 5px; font-size: 11px;")
            self.update_register_lamp("d6007", str(d6007_value), False)
    
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
            print(f"✅ ผู้ใช้กด OK - เริ่มทำงานตามคิวที่ค้างไว้ (D6007 = {value})")
            # ON M450 หลังจากกด OK
            print(f"🔄 กำลัง ON M450")
            if self.modbus_thread:
                print(f"✅ Modbus thread พร้อมใช้งาน - เรียกใช้ on_m450()")
                success = self.modbus_thread.on_m450()
                if success:
                    print(f"✅ ON M450 สำเร็จ - จะ reset ตัวเองหลังจาก 5 วินาที")
                    # Auto reset M450 หลังจาก 5 วินาที
                    QTimer.singleShot(5000, lambda: self.modbus_thread.reset_m450())
                else:
                    print(f"❌ ไม่สามารถ ON M450 ได้")
            else:
                print("❌ Modbus thread ไม่พร้อมใช้งาน")
            # ทำงานตามคิวที่ค้างไว้แทนการถ่ายรูปใหม่
            self.on_queue_trigger()
    
    def update_usb_camera_status(self, status, is_connected=False):
        """Update USB camera status in status tab"""
        try:
            if is_connected:
                self.usb_camera_status.setText("USB Camera: พร้อมใช้งาน")
                self.usb_camera_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            else:
                self.usb_camera_status.setText(f"USB Camera: {status}")
                self.usb_camera_status.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating USB camera status: {e}")
    
    def update_sentech_camera_status(self, status, is_connected=False):
        """Update Sentech camera status in status tab"""
        try:
            if is_connected:
                self.sentech_camera_status.setText("Sentech Camera: พร้อมใช้งาน")
                self.sentech_camera_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            else:
                self.sentech_camera_status.setText(f"Sentech Camera: {status}")
                self.sentech_camera_status.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating Sentech camera status: {e}")
    
    def update_modbus_connection_status(self, is_connected=False):
        """Update Modbus connection status in status tab"""
        try:
            if is_connected:
                self.modbus_connection_lamp.setText("●")
                self.modbus_connection_lamp.setStyleSheet("color: #27ae60; font-size: 24px; font-weight: bold;")
                self.modbus_connection_text.setText("เชื่อมต่อแล้ว")
                self.modbus_connection_text.setStyleSheet("color: #27ae60; font-size: 14px; font-weight: bold;")
            else:
                self.modbus_connection_lamp.setText("●")
                self.modbus_connection_lamp.setStyleSheet("color: #e74c3c; font-size: 24px; font-weight: bold;")
                self.modbus_connection_text.setText("ไม่เชื่อมต่อ")
                self.modbus_connection_text.setStyleSheet("color: #e74c3c; font-size: 14px; font-weight: bold;")
        except Exception as e:
            print(f"❌ Error updating Modbus connection status: {e}")
    
    def update_coil_lamp(self, coil_name, is_on=False):
        """Update individual coil lamp status"""
        try:
            lamp_widget = getattr(self, f"{coil_name.lower()}_lamp", None)
            if lamp_widget:
                if is_on:
                    lamp_widget.setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold; padding: 5px; border: 2px solid #27ae60; background-color: #d5f4e6;")
                else:
                    lamp_widget.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        except Exception as e:
            print(f"❌ Error updating coil lamp {coil_name}: {e}")
    
    def update_register_lamp(self, register_name, value, is_active=False):
        """Update register lamp and text status"""
        try:
            lamp_widget = getattr(self, f"{register_name.lower()}_lamp", None)
            text_widget = getattr(self, f"{register_name.lower()}_text", None)
            
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
    
    def update_modbus_coils_status(self, coils_info):
        """Update Modbus coils status in status tab"""
        try:
            # This function is kept for compatibility but not used in lamp mode
            pass
        except Exception as e:
            print(f"❌ Error updating Modbus coils status: {e}")
    
    def update_modbus_registers_status(self, registers_info):
        """Update Modbus registers status in status tab"""
        try:
            # This function is kept for compatibility but not used in lamp mode
            pass
        except Exception as e:
            print(f"❌ Error updating Modbus registers status: {e}")
    
    def update_bottle_detection_status(self, status, is_processing=False):
        """Update bottle detection status in status tab"""
        try:
            if is_processing:
                self.bottle_detection_status.setText(f"การตรวจจับขวด: {status}")
                self.bottle_detection_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 12px;")
            else:
                self.bottle_detection_status.setText(f"การตรวจจับขวด: {status}")
                self.bottle_detection_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating bottle detection status: {e}")
    
    def update_cap_detection_status(self, status, is_processing=False):
        """Update cap detection status in status tab"""
        try:
            if is_processing:
                self.cap_detection_status.setText(f"การตรวจจับฝา: {status}")
                self.cap_detection_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 12px;")
            else:
                self.cap_detection_status.setText(f"การตรวจจับฝา: {status}")
                self.cap_detection_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating cap detection status: {e}")
    
    def update_bottle_type_status(self, bottle_type):
        """Update current bottle type status in status tab"""
        try:
            if bottle_type:
                self.current_bottle_type_status.setText(f"ประเภทขวดปัจจุบัน: {bottle_type}")
                self.current_bottle_type_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            else:
                self.current_bottle_type_status.setText("ประเภทขวดปัจจุบัน: -")
                self.current_bottle_type_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating bottle type status: {e}")
    
    def update_processing_mode_status(self, mode):
        """Update processing mode status in status tab"""
        try:
            self.current_processing_mode = mode
            self.processing_mode_status.setText(f"โหมดการประมวลผล: {mode}")
            if mode == "Auto":
                self.processing_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            else:
                self.processing_mode_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating processing mode status: {e}")
    
    def update_angle3_retry_status(self, is_active=False, retry_count=0, max_retries=0):
        """Update angle3 retry status in status tab"""
        try:
            if is_active:
                self.angle3_retry_status.setText(f"โหมดถ่ายภาพซ้ำ: เปิด (ครั้งที่ {retry_count}/{max_retries})")
                self.angle3_retry_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 12px;")
            else:
                self.angle3_retry_status.setText("โหมดถ่ายภาพซ้ำ: ปิด")
                self.angle3_retry_status.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 12px;")
        except Exception as e:
            print(f"❌ Error updating angle3 retry status: {e}")
    
    def update_performance_stats(self, images_processed=None, successful_detections=None, errors=None):
        """Update performance statistics in status tab"""
        try:
            if images_processed is not None:
                self.total_images_processed_count = images_processed
            if successful_detections is not None:
                self.successful_detections_count = successful_detections
            if errors is not None:
                self.error_count_value = errors
            
            # อัพเดท display แบบ real-time
            if hasattr(self, 'total_images_processed'):
                self.total_images_processed.setText(f"จำนวนภาพที่ประมวลผล: {self.total_images_processed_count}")
                self.total_images_processed.setStyleSheet("color: #2c3e50; padding: 5px; font-size: 12px;")
            
            if hasattr(self, 'successful_detections'):
                self.successful_detections.setText(f"การตรวจจับที่สำเร็จ: {self.successful_detections_count}")
                self.successful_detections.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            
            if hasattr(self, 'error_count'):
                self.error_count.setText(f"จำนวนข้อผิดพลาด: {self.error_count_value}")
                if self.error_count_value > 0:
                    self.error_count.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
                else:
                    self.error_count.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
            
            # คำนวณ success rate
            if hasattr(self, 'success_rate') and self.total_images_processed_count > 0:
                success_rate = (self.successful_detections_count / self.total_images_processed_count) * 100
                self.success_rate.setText(f"อัตราความสำเร็จ: {success_rate:.1f}%")
                if success_rate >= 90:
                    self.success_rate.setStyleSheet("color: #27ae60; padding: 5px; font-size: 12px;")
                elif success_rate >= 70:
                    self.success_rate.setStyleSheet("color: #f39c12; padding: 5px; font-size: 12px;")
                else:
                    self.success_rate.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 12px;")
                
        except Exception as e:
            print(f"❌ Error updating performance stats: {e}")
    
    def on_modbus_trigger(self):
        """Handle Modbus trigger (M301)"""
        print("🎯 MODBUS TRIGGER RECEIVED - Starting capture sequence")
        # Auto mode is always enabled
        # Capture from both cameras simultaneously
        self.capture_image_auto()
        self.capture_sentech_image_auto()
    
    def on_queue_trigger(self):
        """Handle queue trigger (from M600 reset)"""
        print("🎯 QUEUE TRIGGER RECEIVED - Starting capture from queue")
        # Auto mode is always enabled
        # Capture from both cameras simultaneously
        self.capture_image_from_queue()
        self.capture_sentech_image_from_queue()
    
    def toggle_auto_mode(self, state):
        """Toggle auto mode - Function kept for compatibility but auto mode is always enabled"""
        # Auto mode is always enabled, this function is kept for compatibility
        self.update_processing_mode_status("Auto")
    
    def capture_image_manual(self):
        """Capture image manually"""
        if self.usb_camera is None:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กล้องยังไม่ได้เริ่มต้น")
            return
        
        try:
            self.status_label.setText('📸 กำลังถ่ายภาพ...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # Capture image - ดึงเฟรมล่าสุด
            captured_image = None
            for attempt in range(3):
                print(f"📸 MANUAL CAPTURE: Attempt {attempt + 1}/3")
                captured_image = self.usb_camera.capture_image()
                if captured_image is not None:
                    print(f"✅ MANUAL CAPTURE: Success on attempt {attempt + 1}")
                    break
                time.sleep(0.1)  # รอสักครู่แล้วลองใหม่
            
            if captured_image is not None:
                self.current_image = captured_image
                self.display_image(captured_image)
                self.image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]}")
                self.btn_process.setEnabled(True)
                self.status_label.setText('✅ ถ่ายภาพสำเร็จ - พร้อมประมวลผล')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            else:
                self.status_label.setText('❌ ไม่สามารถถ่ายภาพได้')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            self.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def select_image_file(self):
        """Select image file for bottle detection processing"""
        try:
            # Open file dialog to select image
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "เลือกไฟล์ภาพสำหรับการตรวจจับขวด",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            
            if file_path:
                print(f"📁 IMAGE SELECTION: Selected file: {file_path}")
                self.status_label.setText('📁 กำลังโหลดภาพ...')
                self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
                QtWidgets.QApplication.processEvents()
                
                # Load image using OpenCV
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_image = image
                    self.display_image(image)
                    self.image_info_label.setText(f"ขนาด: {image.shape[1]}x{image.shape[0]} | ไฟล์: {file_path.split('/')[-1]}")
                    self.btn_process.setEnabled(True)
                    self.status_label.setText('✅ โหลดภาพสำเร็จ - พร้อมประมวลผล')
                    self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                    print(f"✅ IMAGE SELECTION: Successfully loaded image with shape: {image.shape}")
                else:
                    self.status_label.setText('❌ ไม่สามารถโหลดภาพได้')
                    self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    QMessageBox.warning(self, "ข้อผิดพลาด", f"ไม่สามารถโหลดภาพจากไฟล์:\n{file_path}")
                    print(f"❌ IMAGE SELECTION: Failed to load image from: {file_path}")
                    
        except Exception as e:
            self.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเลือกไฟล์:\n{str(e)}")
            print(f"❌ IMAGE SELECTION ERROR: {str(e)}")
    
    def select_cap_image_file(self):
        """Select image file for cap detection processing"""
        try:
            # Open file dialog to select image
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "เลือกไฟล์ภาพสำหรับการตรวจจับฝา",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            
            if file_path:
                print(f"📁 CAP IMAGE SELECTION: Selected file: {file_path}")
                self.status_label.setText('📁 กำลังโหลดภาพฝา...')
                self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
                QtWidgets.QApplication.processEvents()
                
                # Load image using OpenCV
                image = cv2.imread(file_path)
                if image is not None:
                    # Set both current_cap_image and current_sentech_image for compatibility
                    self.current_cap_image = image
                    self.current_sentech_image = image
                    self.original_cap_image = image.copy()  # Store original for rotation
                    self.current_rotation_angle = 0  # Reset rotation angle
                    self.display_sentech_image(image)
                    self.sentech_image_info_label.setText(f"ขนาด: {image.shape[1]}x{image.shape[0]} | ไฟล์: {file_path.split('/')[-1]}")
                    self.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
                    # Manual rotation buttons removed
                    self.status_label.setText('✅ โหลดภาพฝาสำเร็จ - พร้อมประมวลผล')
                    self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                    print(f"✅ CAP IMAGE SELECTION: Successfully loaded image with shape: {image.shape}")
                else:
                    self.status_label.setText('❌ ไม่สามารถโหลดภาพฝาได้')
                    self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                    QMessageBox.warning(self, "ข้อผิดพลาด", f"ไม่สามารถโหลดภาพจากไฟล์:\n{file_path}")
                    print(f"❌ CAP IMAGE SELECTION: Failed to load image from: {file_path}")
                    
        except Exception as e:
            self.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเลือกไฟล์ฝา:\n{str(e)}")
            print(f"❌ CAP IMAGE SELECTION ERROR: {str(e)}")
    
    def capture_image_auto(self):
        """Capture image automatically from Modbus trigger"""
        if self.usb_camera is None:
            print("❌ Camera not initialized - cannot capture")
            return
        
        try:
            print("📸 AUTO CAPTURE: Starting image capture after M301 ON...")
            self.status_label.setText('📸 กำลังถ่ายภาพอัตโนมัติ...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # รอ 2 วินาทีหลังได้รับสัญญาณ M301 เพื่อให้ได้เฟรมล่าสุด
            print("⏰ AUTO CAPTURE: Waiting 2 seconds for latest frame...")
            self.status_label.setText('⏰ รอ 2 วินาทีเพื่อเฟรมล่าสุด...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # รอ 2 วินาทีเพื่อให้ได้เฟรมล่าสุดหลังจาก M301 ON
            time.sleep(2)
            
            # ล้าง buffer และดึงเฟรมล่าสุด
            print("📸 AUTO CAPTURE: Capturing latest frame after M301 ON...")
            self.status_label.setText('📸 กำลังถ่ายภาพเฟรมล่าสุด...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ลองถ่ายภาพหลายครั้งเพื่อให้ได้เฟรมล่าสุด
            captured_image = None
            for attempt in range(5):  # เพิ่มจำนวนครั้ง
                print(f"📸 AUTO CAPTURE: Attempt {attempt + 1}/5")
                
                # ตรวจสอบว่า M301 ยัง ON อยู่หรือไม่
                if hasattr(self, 'modbus_thread') and self.modbus_thread:
                    try:
                        result = self.modbus_thread.modbus_client.read_coils(301, 1, unit=1)
                        m301_status = not result.isError() and result.bits[0]
                        # print(f"🔍 M301 Status during capture: {m301_status}")  # Reduced spam
                    except:
                        m301_status = True  # ถ้าตรวจสอบไม่ได้ให้ถือว่า ON
                else:
                    m301_status = True
                
                captured_image = self.usb_camera.capture_image()
                if captured_image is not None:
                    print(f"✅ AUTO CAPTURE: Success on attempt {attempt + 1}")
                    break
                time.sleep(0.2)  # รอมากขึ้นระหว่างการลอง
            
            if captured_image is not None:
                # แสดงเวลาที่ถ่ายภาพ
                import datetime
                capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"🕐 Capture time: {capture_time}")
                
                self.current_image = captured_image
                self.display_image(captured_image)
                self.image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]} | เวลา: {capture_time}")
                self.btn_process.setEnabled(True)
                self.status_label.setText('✅ ถ่ายภาพอัตโนมัติสำเร็จ - พร้อมประมวลผล')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                
                # Auto process is always enabled
                if True:  # Auto mode is always enabled
                    # ตรวจสอบว่าเป็นโหมดถ่ายภาพซ้ำหรือไม่
                    if hasattr(self, 'modbus_thread') and self.modbus_thread and self.modbus_thread.angle3_retry_mode:
                        print("🔄 AUTO PROCESS: Angle3 retry mode - processing bottle only (no cap detection)")
                        self.process_current_image()  # ประมวลผลขวดเท่านั้น
                    else:
                        print("🔄 AUTO PROCESS: Normal mode - starting automatic processing...")
                    self.process_current_image()
                else:
                    print("⚠️ Auto processing disabled - image captured but not processed")
            else:
                self.status_label.setText('❌ ไม่สามารถถ่ายภาพอัตโนมัติได้')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            self.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_image_from_queue(self):
        """Capture image from queue (without waiting for M301)"""
        if self.usb_camera is None:
            print("❌ Camera not initialized - cannot capture from queue")
            return
        
        try:
            print("📸 QUEUE CAPTURE: Starting image capture from queue...")
            self.status_label.setText('📸 กำลังถ่ายภาพจากคิว...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ถ่ายภาพทันทีโดยไม่รอ (เพราะเป็นงานจากคิว)
            print("📸 QUEUE CAPTURE: Capturing image immediately...")
            self.status_label.setText('📸 กำลังถ่ายภาพ...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ลองถ่ายภาพหลายครั้ง
            captured_image = None
            for attempt in range(3):
                print(f"📸 QUEUE CAPTURE: Attempt {attempt + 1}/3")
                captured_image = self.usb_camera.capture_image()
                if captured_image is not None:
                    print(f"✅ QUEUE CAPTURE: Success on attempt {attempt + 1}")
                    break
                time.sleep(0.1)
            
            if captured_image is not None:
                # แสดงเวลาที่ถ่ายภาพ
                import datetime
                capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"🕐 Queue Capture time: {capture_time}")
                print(f"📸 QUEUE CAPTURE: Image shape: {captured_image.shape}")
                
                self.current_image = captured_image
                self.display_image(captured_image)
                self.image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]} | เวลา: {capture_time} | จากคิว")
                self.btn_process.setEnabled(True)
                self.status_label.setText('✅ ถ่ายภาพจากคิวสำเร็จ - พร้อมประมวลผล')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                
                # Auto process is always enabled
                if True:  # Auto mode is always enabled
                    print("🔄 QUEUE PROCESS: Starting automatic processing...")
                    print("🔄 QUEUE PROCESS: Auto mode is enabled, calling process_current_image()")
                    self.process_current_image()
                else:
                    print("⚠️ Auto processing disabled - image captured but not processed")
            else:
                self.status_label.setText('❌ ไม่สามารถถ่ายภาพจากคิวได้')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            self.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
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
                    image = cv2.resize(image, (new_w, new_h))
                
                # Convert to RGB for Qt
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, c = rgb_image.shape
                bytes_per_line = c * w
                
                qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                self.image_label.setPixmap(pixmap)
            else:
                self.image_label.setText("ไม่สามารถโหลดภาพได้")

        except Exception as e:
            self.image_label.setText(f"ข้อผิดพลาด: {str(e)}")
            
    def display_cropped_images(self, type_crops):
        """Display cropped type images with OCR results"""
        # Clear existing crops
        for i in reversed(range(self.crops_layout.count())):
            self.crops_layout.itemAt(i).widget().setParent(None)
        
        if not type_crops:
            # Show placeholder
            placeholder_label = QLabel("ไม่มีภาพที่ครอป")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_label.setStyleSheet("color: #7f8c8d; padding: 20px;")
            self.crops_layout.addWidget(placeholder_label)
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
                    crop_image = cv2.resize(crop_image, (new_w, new_h))
                
                # Convert to RGB for Qt
                rgb_crop = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                h, w, c = rgb_crop.shape
                bytes_per_line = c * w
                
                qimg = QtGui.QImage(rgb_crop.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                
                crop_label = QLabel()
                crop_label.setPixmap(pixmap)
                crop_label.setAlignment(Qt.AlignCenter)
                crop_layout.addWidget(crop_label)
                
                # Add confidence info
                confidence_label = QLabel(f"Detection Confidence: {crop['confidence']:.3f}")
                confidence_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
                confidence_label.setAlignment(Qt.AlignCenter)
                crop_layout.addWidget(confidence_label)
                
                # Add OCR results
                if crop.get('ocr_results'):
                    ocr_title = QLabel("📝 OCR Results:")
                    ocr_title.setStyleSheet("font-weight: bold; color: #27ae60; font-size: 11px;")
                    ocr_title.setAlignment(Qt.AlignCenter)
                    crop_layout.addWidget(ocr_title)
                    
                    for j, ocr_result in enumerate(crop['ocr_results']):
                        ocr_text = f"  {j+1}. '{ocr_result['text']}' ({ocr_result['confidence']:.3f})"
                        ocr_label = QLabel(ocr_text)
                        ocr_label.setStyleSheet("color: #2c3e50; font-size: 10px; background-color: #ecf0f1; padding: 2px;")
                        ocr_label.setWordWrap(True)
                        crop_layout.addWidget(ocr_label)
                else:
                    no_ocr_label = QLabel("ไม่พบข้อความ")
                    no_ocr_label.setStyleSheet("color: #e74c3c; font-size: 10px; font-style: italic;")
                    no_ocr_label.setAlignment(Qt.AlignCenter)
                    crop_layout.addWidget(no_ocr_label)
            
            self.crops_layout.addWidget(crop_container)
        
    def process_current_image(self):
        """Process current image"""
        if self.current_image is None:
            QMessageBox.warning(self, "ข้อผิดพลาด", "ไม่มีภาพให้ประมวลผล")
            return
        
        print(f"🔄 PROCESS: Starting processing for image shape: {self.current_image.shape}")
        print("🔄 PROCESS: Creating BottleDetectionThread...")
        
        # Update status tab
        self.update_bottle_detection_status("กำลังประมวลผล...", True)
            
        # Start processing thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText('🔄 กำลังประมวลผล...')
        self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        
        # Disable process button and enable stop button
        self.btn_process.setEnabled(False)
        self.btn_stop_processing.setEnabled(True)
        
        self.processing_thread = BottleDetectionThread(self.current_image, selected_tastes=self.selected_tastes)
        self.processing_thread.result_ready.connect(self.on_processing_complete)
        self.processing_thread.status_updated.connect(self.status_label.setText)
        self.processing_thread.progress_updated.connect(self.update_bottle_progress_bar)
        print("🔄 PROCESS: Starting processing thread...")
        print(f"🔄 PROCESS: Using selected tastes: {self.selected_tastes}")
        self.processing_thread.start()
    
    def on_processing_complete(self, result):
        """Handle processing completion"""
        print("✅ PROCESS COMPLETE: Processing finished")
        self.progress_bar.setVisible(False)
        
        # Update performance stats
        self.total_images_processed_count += 1
        self.update_performance_stats(images_processed=self.total_images_processed_count)
        
        # Reset button states
        self.btn_process.setEnabled(True)
        self.btn_stop_processing.setEnabled(False)
        
        if "error" not in result:
            print("✅ PROCESS COMPLETE: No error in result")
            print(f"✅ PROCESS COMPLETE: Result keys: {list(result.keys())}")
            
            # Display results
            self.display_single_results(result)
            
            # Draw detections on image
            if result.get('detections'):
                result_image = draw_detections_on_image(
                    result['image'], 
                    result['detections']
                )
                self.display_result_image(result_image)
            
            # Store result
            self.current_result = result
            
            # Display cropped images with OCR
            if result.get('type_crops'):
                self.display_cropped_images(result['type_crops'])
            
            # Update status tab
            self.update_bottle_detection_status("เสร็จสิ้น", False)
            
            # Handle bottle type detection and Modbus control
            if result.get('bottle_type') and result.get('combined_ocr_text'):
                print(f"🎯 PROCESS COMPLETE: Bottle type detected: {result['bottle_type']}")
                print(f"📝 PROCESS COMPLETE: OCR text: '{result['combined_ocr_text']}'")
                self.successful_detections_count += 1
                self.update_performance_stats(successful_detections=self.successful_detections_count)
                self.handle_bottle_type_detection(result['bottle_type'], result['combined_ocr_text'])
            else:
                print("❌ PROCESS COMPLETE: No bottle type or OCR text found")
                print(f"❌ PROCESS COMPLETE: bottle_type: {result.get('bottle_type')}")
                print(f"❌ PROCESS COMPLETE: combined_ocr_text: {result.get('combined_ocr_text')}")

            self.status_label.setText('✅ ประมวลผลเสร็จสิ้น')
            self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        else:
            print(f"❌ PROCESS COMPLETE: Error in result: {result['error']}")
            self.status_label.setText(f'❌ ข้อผิดพลาด: {result["error"]}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            self.error_count_value += 1
            self.update_performance_stats(errors=self.error_count_value)
                
    def handle_bottle_type_detection(self, bottle_type, ocr_text):
        """Handle bottle type detection and control Modbus"""
        print(f"🎯 BOTTLE TYPE DETECTED: {bottle_type} (OCR: '{ocr_text}')")
        
        # เก็บประเภทขวดที่ตรวจพบ
        self.current_bottle_type = bottle_type
        
        # Update status tab
        self.update_bottle_type_status(bottle_type)
        self.update_bottle_detection_status("ตรวจพบแล้ว", False)
        
        # ตรวจสอบว่าเป็นโหมดถ่ายภาพซ้ำหรือไม่
        if self.modbus_thread.angle3_retry_mode:
            print(f"🔄 ANGLE3 RETRY MODE: Detected {bottle_type} - checking if non-angle3...")
            
            if bottle_type != "M130":
                # ตรวจจับได้อย่างอื่นนอกจาก angle3 - ส่งสัญญาณตามประเภทขวดและหยุดโหมดถ่ายภาพซ้ำ
                print(f"✅ NON-ANGLE3 DETECTED: {bottle_type} - ON bottle type signal and M600")
                
                # ส่งสัญญาณตามประเภทขวด
                bottle_signal_success = False
                if bottle_type == "M100":
                    bottle_signal_success = self.modbus_thread.on_m100()
                    if bottle_signal_success:
                        print("🏷️ ON M100 (non-angle3 detection)")
                        self.update_coil_lamp("m100", True)
                elif bottle_type == "M110":
                    bottle_signal_success = self.modbus_thread.on_m110()
                    if bottle_signal_success:
                        print("🏷️ ON M110 (non-angle3 detection)")
                        self.update_coil_lamp("m110", True)
                elif bottle_type == "M120":
                    bottle_signal_success = self.modbus_thread.on_m120()
                    if bottle_signal_success:
                        print("🏷️ ON M120 (non-angle3 detection)")
                        self.update_coil_lamp("m120", True)
                
                # ส่ง M600
                m600_success = self.modbus_thread.on_m600()
                if m600_success:
                    print("🚀 ON M600 (non-angle3 detection)")
                    self.update_coil_lamp("m600", True)
                    
                    if bottle_signal_success:
                        self.status_label.setText(f'🚀 ตรวจพบ: {bottle_type} (ไม่ใช่ angle3) → ON {bottle_type}+M600')
                        self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                    else:
                        self.status_label.setText(f'🚀 ตรวจพบ: {bottle_type} (ไม่ใช่ angle3) → ON M600 (ไม่สามารถ ON {bottle_type} ได้)')
                        self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
                    
                    # หยุดโหมดถ่ายภาพซ้ำ
                    self.modbus_thread.angle3_retry_mode = False
                    self.modbus_thread.angle3_retry_count = 0
                    self.update_angle3_retry_status(False)
                    print("🛑 หยุดโหมดถ่ายภาพซ้ำ (พบ non-angle3)")
                else:
                    print("❌ ไม่สามารถ ON M600 ได้")
                    self.status_label.setText('❌ ไม่สามารถ ON M600 ได้')
                    self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            else:
                # ยังเป็น angle3 - ส่งสัญญาณและถ่ายภาพซ้ำ
                self.modbus_thread.angle3_retry_count += 1
                print(f"🔄 ANGLE3 RETRY: Still angle3 (attempt {self.modbus_thread.angle3_retry_count}/{self.modbus_thread.max_angle3_retries})")
                
                # ส่งสัญญาณ M130, M850, M600 สำหรับ angle3
                success = self.modbus_thread.on_m130()
                if success:
                    self.update_coil_lamp("m130", True)
                    
                    # ON M850 สำหรับ M130
                    m850_success = self.modbus_thread.on_m850()
                    if m850_success:
                        print("🏷️ ON M850 (angle3 retry)")
                        self.update_coil_lamp("m850", True)
                    
                    # ON M600 สำหรับ M130 ด้วย
                    m600_success = self.modbus_thread.on_m600()
                    if m600_success:
                        print("⏳ ON M130, M850 และ M600 สำเร็จ (angle3 retry)")
                        self.update_coil_lamp("m600", True)
                        self.modbus_thread.start_d6006_monitoring()
                        self.update_register_lamp("d6006", "กำลังอ่าน...", True)
                        self.status_label.setText(f'🔄 ยังเป็น angle3 - ส่งสัญญาณและถ่ายภาพใหม่ (ครั้งที่ {self.modbus_thread.angle3_retry_count})')
                        self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
                        
                        # Update status tab
                        self.update_angle3_retry_status(True, self.modbus_thread.angle3_retry_count, self.modbus_thread.max_angle3_retries)
                        
                        # ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)
                        print("📸 ANGLE3 RETRY: ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)")
                        self.capture_image_auto()  # ถ่ายภาพ USB ใหม่ทันที
                    else:
                        print("❌ ไม่สามารถ ON M600 ได้ (angle3 retry)")
                        self.status_label.setText('❌ ไม่สามารถ ON M600 ได้ (angle3 retry)')
                        self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                else:
                    print("❌ ไม่สามารถ ON M130 ได้ (angle3 retry)")
                    self.status_label.setText('❌ ไม่สามารถ ON M130 ได้ (angle3 retry)')
                    self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            return
        
        # ตรวจสอบประเภทขวดและดำเนินการตามลำดับ
        if bottle_type == "M130":
            # M130 (angle3) - ON ทันทีและข้ามการประมวลผลฝา
            success = self.modbus_thread.on_m130()
            print("🏷️ ON M130 (พบคำว่า 'angle3')")
            if success:
                # Update lamp status
                self.update_coil_lamp("m130", True)
                
                # ON M850 สำหรับ M130
                m850_success = self.modbus_thread.on_m850()
                if m850_success:
                    print("🏷️ ON M850 (angle3)")
                    self.update_coil_lamp("m850", True)
                else:
                    print("❌ ไม่สามารถ ON M850 ได้")
                
                # ON M600 สำหรับ M130 ด้วย
                m600_success = self.modbus_thread.on_m600()
                if m600_success:
                    print("⏳ ON M130, M850 และ M600 สำเร็จ - เริ่มอ่าน D6006...")
                    self.update_coil_lamp("m600", True)
                    self.modbus_thread.start_d6006_monitoring()
                    self.update_register_lamp("d6006", "กำลังอ่าน...", True)
                    self.status_label.setText('🏷️ ตรวจพบ: M130 → ON M850+M600 → กำลังอ่าน D6006... (ไม่ตรวจจับฝา)')
                    self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                    
                    # ไม่ต้องประมวลผลฝาสำหรับ angle3
                    print("⚠️ ANGLE3: ข้ามการประมวลผลฝา")
                else:
                    print("❌ ไม่สามารถ ON M600 ได้")
                    self.status_label.setText('❌ ไม่สามารถ ON M600 ได้')
                    self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            else:
                print("❌ ไม่สามารถ ON M130 ได้")
                self.status_label.setText('❌ ไม่สามารถ ON M130 ได้')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            return  # M130 ไม่ต้องไปต่อที่ M600 logic
        
        # M100/M110/M120 - เก็บประเภทขวดไว้รอผลลัพธ์ฝา
        print(f"⏳ ตรวจพบ: {bottle_type} - รอผลลัพธ์ฝาก่อน ON Modbus...")
        self.status_label.setText(f'⏳ ตรวจพบ: {bottle_type} - รอผลลัพธ์ฝาก่อน ON Modbus...')
        self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        
        # เริ่มประมวลผลฝา (ถ้ามีภาพจาก Sentech)
        if self.current_sentech_image is not None:
            print("🔄 AUTO CAP PROCESS: เริ่มประมวลผลฝา...")
            # print(f"🔍 AUTO CAP PROCESS: current_sentech_image shape: {self.current_sentech_image.shape}")  # Reduced spam
            # print(f"🔍 AUTO CAP PROCESS: current_bottle_type: {self.current_bottle_type}")  # Reduced spam
            
            # แสดงผล GUI สำหรับการประมวลผลฝา
            self.display_cap_processing_ui()
            # print("🔍 AUTO CAP PROCESS: Called display_cap_processing_ui()")  # Reduced spam
            
            # เรียกใช้ process_cap_detection() เหมือนการกดปุ่ม
            print("🔍 AUTO CAP PROCESS: Calling process_cap_detection()...")
            self.process_cap_detection()
            print("🔍 AUTO CAP PROCESS: process_cap_detection() completed")
            
            # รอให้การประมวลผลเสร็จสิ้น
            print("🔍 AUTO CAP PROCESS: Waiting for cap processing to complete...")
            if hasattr(self, 'cap_processing_thread') and self.cap_processing_thread:
                # รอให้ thread เสร็จสิ้น (ไม่เกิน 30 วินาที)
                if self.cap_processing_thread.wait(30000):  # 30 วินาที
                    print("🔍 AUTO CAP PROCESS: Cap processing thread completed")
                else:
                    print("⚠️ AUTO CAP PROCESS: Cap processing thread timeout")
            else:
                print("⚠️ AUTO CAP PROCESS: No cap processing thread found")
        else:
            print("⚠️ AUTO CAP PROCESS: ไม่มีภาพจาก Sentech - ไม่สามารถประมวลผลฝาได้")
            self.status_label.setText('⚠️ ไม่มีภาพจาก Sentech - ไม่สามารถประมวลผลฝาได้')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def on_bottle_type_detected(self, bottle_type, ocr_text):
        """Handle bottle type detection signal"""
        print(f"🎯 BOTTLE TYPE SIGNAL: {bottle_type} (OCR: '{ocr_text}')")
        self.handle_bottle_type_detection(bottle_type, ocr_text)
    
    def on_m513_stop(self):
        """Handle M513 stop signal - reset everything like pressing Stop button"""
        print("🛑 M513 STOP: รับสัญญาณหยุดจาก M513 - เริ่ม reset ทั้งหมด")
        try:
            # อัปเดตสถานะ UI
            self.status_label.setText('🛑 M513: หยุดการทำงาน - กำลัง reset ระบบ')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            
            # เรียกใช้ฟังก์ชัน reset ทั้งหมดเหมือนกดปุ่ม Stop
            self.reset_to_initial_state()
            
            print("✅ M513 STOP: Reset ระบบเสร็จสิ้น")
            
        except Exception as e:
            print(f"❌ M513 STOP: ข้อผิดพลาดในการ reset: {e}")
            self.status_label.setText('❌ M513: ข้อผิดพลาดในการ reset ระบบ')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def reset_processing_for_new_image(self):
        """Reset processing state for new image after D6006=300"""
        try:
            print("🔄 RESET PROCESSING: Clearing current data for new image...")
            
            # Clear current images
            self.current_image = None
            self.current_sentech_image = None
            self.current_cap_image = None
            self.original_cap_image = None
            self.ai_rotated_image = None
            
            # Clear current results
            self.current_cap_result = None
            self.current_rotation_angle = 0
            
            # Reset progress bars
            self.progress_bar.setVisible(False)
            self.cap_progress_bar.setVisible(False)
            
            # Reset button states
            self.btn_process.setEnabled(False)
            self.btn_process_cap.setEnabled(False)
            self.btn_stop_processing.setEnabled(False)
            
            # Manual rotation buttons removed
            
            # Clear image displays
            self.image_label.clear()
            self.sentech_image_label.clear()
            
            # Clear text displays
            self.results_text.clear()
            self.cap_detection_text.clear()
            
            # Update status
            self.status_label.setText('🔄 RESET การประมวลผล - พร้อมรับภาพใหม่')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            
            print("✅ RESET PROCESSING: Processing state cleared for new image")
            
        except Exception as e:
            print(f"❌ Error resetting processing state: {e}")
    
    def capture_sentech_image(self):
        """Capture image from Sentech camera using Harvesters"""
        if self.sentech_camera is None:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กล้อง Sentech ยังไม่ได้เริ่มต้น")
            return
        
        try:
            self.status_label.setText('📸 กำลังถ่ายภาพจาก Sentech...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # Capture image from Sentech
            try:
                print("📸 Calling sentech_camera.capture_image()...")
                captured_image = self.sentech_camera.capture_image()
                print(f"📸 Capture result: {captured_image is not None}")
            except Exception as e:
                print(f"❌ Error calling sentech_camera.capture_image(): {e}")
                import traceback
                traceback.print_exc()
                captured_image = None
            
            if captured_image is not None:
                # ล้างข้อมูลการประมวลผลฝาเก่าก่อนที่จะใช้ภาพใหม่
                self.current_cap_result = None
                if hasattr(self, 'cap_detection_text') and self.cap_detection_text:
                    self.cap_detection_text.clear()
                
                # ล้าง cap results layout
                if hasattr(self, 'cap_results_layout') and self.cap_results_layout:
                    for i in reversed(range(self.cap_results_layout.count())):
                        widget = self.cap_results_layout.itemAt(i).widget()
                        if widget:
                            widget.setParent(None)
                
                self.current_sentech_image = captured_image
                self.display_sentech_image(captured_image)
                self.sentech_image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]}")
                self.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
                self.status_label.setText('✅ ถ่ายภาพจาก Sentech สำเร็จ - พร้อมประมวลผลฝา')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            else:
                print("❌ Failed to capture from Sentech - trying USB camera fallback...")
                # ลองใช้ USB camera เป็น fallback
                if self.usb_camera and hasattr(self.usb_camera, 'is_opened') and self.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback...")
                    fallback_image = self.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful")
                        # ใช้ภาพจาก USB camera แทน
                        self.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        self.btn_process_cap.setEnabled(True)
                        self.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        return
                
                self.status_label.setText('❌ ไม่สามารถถ่ายภาพจาก Sentech ได้')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            print(f"❌ Error capturing Sentech image: {e} - trying USB camera fallback...")
            # ลองใช้ USB camera เป็น fallback
            try:
                if self.usb_camera and hasattr(self.usb_camera, 'is_opened') and self.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback after error...")
                    fallback_image = self.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful after error")
                        # ใช้ภาพจาก USB camera แทน
                        self.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        self.btn_process_cap.setEnabled(True)
                        self.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        return
            except:
                pass
            
            self.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_sentech_image_auto(self):
        """Capture image automatically from Sentech camera after M301 trigger using Harvesters"""
        if self.sentech_camera is None:
            print("❌ Sentech camera not initialized - cannot capture")
            return
        
        # รอให้กล้องเริ่มต้นเสร็จ (ถ้ายังไม่เสร็จ)
        max_wait_time = 10.0  # รอสูงสุด 10 วินาที
        wait_time = 0.0
        while hasattr(self.sentech_camera, '_is_initialized') and not self.sentech_camera._is_initialized and wait_time < max_wait_time:
            time.sleep(0.1)
            wait_time += 0.1
        
        if hasattr(self.sentech_camera, '_is_initialized') and not self.sentech_camera._is_initialized:
            print("❌ Sentech camera initialization timeout - cannot capture")
            return
        
        try:
            print("📸 SENTECH AUTO CAPTURE: Starting image capture after M301 ON...")
            self.status_label.setText('📸 กำลังถ่ายภาพอัตโนมัติจาก Sentech...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ตรวจสอบ Harvesters acquisition status
            print("🔍 SENTECH AUTO CAPTURE: Checking Harvesters acquisition status...")
            try:
                if hasattr(self.sentech_camera, '_ia') and self.sentech_camera._ia is not None:
                    print("✅ SENTECH AUTO CAPTURE: Image acquirer is available")
                else:
                    print("❌ SENTECH AUTO CAPTURE: Image acquirer is not available")
                    return
            except Exception as e:
                print(f"❌ SENTECH AUTO CAPTURE: Error checking image acquirer: {e}")
                return
            
            # ถ่ายภาพทันทีโดยไม่รอ callback
            print("📸 SENTECH AUTO CAPTURE: Capturing image immediately after M301 ON...")
            self.status_label.setText('📸 กำลังถ่ายภาพจาก Sentech หลัง M301 ON...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ใช้ capture_image() ที่ใช้ Harvesters
            try:
                print("📸 Calling sentech_camera.capture_image()...")
                captured_image = self.sentech_camera.capture_image()
                print(f"📸 Capture result: {captured_image is not None}")
            except Exception as e:
                print(f"❌ Error calling sentech_camera.capture_image(): {e}")
                import traceback
                traceback.print_exc()
                captured_image = None
            
            if captured_image is not None:
                # ล้างข้อมูลการประมวลผลฝาเก่าก่อนที่จะใช้ภาพใหม่
                self.current_cap_result = None
                if hasattr(self, 'cap_detection_text') and self.cap_detection_text:
                    self.cap_detection_text.clear()
                
                # ล้าง cap results layout
                if hasattr(self, 'cap_results_layout') and self.cap_results_layout:
                    for i in reversed(range(self.cap_results_layout.count())):
                        widget = self.cap_results_layout.itemAt(i).widget()
                        if widget:
                            widget.setParent(None)
                
                # แสดงเวลาที่ถ่ายภาพ
                import datetime
                capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"🕐 Sentech Capture time: {capture_time}")
                
                self.current_sentech_image = captured_image
                self.display_sentech_image(captured_image)
                self.sentech_image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]} | เวลา: {capture_time}")
                self.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
                self.status_label.setText('✅ ถ่ายภาพอัตโนมัติจาก Sentech สำเร็จ - พร้อมประมวลผลฝา')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                
                # Auto process is always enabled (BOTTLE AND CAP processing enabled)
                if True:  # Auto mode is always enabled
                    # ตรวจสอบว่าเป็นโหมดถ่ายภาพซ้ำหรือไม่
                    if hasattr(self, 'modbus_thread') and self.modbus_thread and self.modbus_thread.angle3_retry_mode:
                        print("🔄 SENTECH AUTO PROCESS: Angle3 retry mode - skipping cap processing")
                        # ไม่ประมวลผลฝาในโหมดถ่ายภาพซ้ำ
                    else:
                        print("🔄 SENTECH AUTO PROCESS: Normal mode - processing both bottle and cap")
                        # แสดงผล GUI สำหรับการประมวลผลฝา
                        self.display_cap_processing_ui()
                        self.process_cap_detection()  # ENABLED - cap processing turned on
                else:
                    print("⚠️ Auto processing disabled - Sentech image captured but not processed")
            else:
                print("❌ Failed to capture from Sentech - trying USB camera fallback...")
                # ลองใช้ USB camera เป็น fallback
                if self.usb_camera and hasattr(self.usb_camera, 'is_opened') and self.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback...")
                    fallback_image = self.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful")
                        # ใช้ภาพจาก USB camera แทน
                        self.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} | เวลา: {capture_time} (USB Fallback)")
                        self.btn_process_cap.setEnabled(True)
                        self.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        
                        # Auto process
                        if True:
                            if hasattr(self, 'modbus_thread') and self.modbus_thread and self.modbus_thread.angle3_retry_mode:
                                print("🔄 USB FALLBACK: Angle3 retry mode - skipping cap processing")
                            else:
                                print("🔄 USB FALLBACK: Normal mode - processing both bottle and cap")
                                self.display_cap_processing_ui()
                                self.process_cap_detection()
                        return
                
                self.status_label.setText('❌ ไม่สามารถถ่ายภาพอัตโนมัติจาก Sentech ได้')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            print(f"❌ Error capturing Sentech image (auto): {e} - trying USB camera fallback...")
            # ลองใช้ USB camera เป็น fallback
            try:
                if self.usb_camera and hasattr(self.usb_camera, 'is_opened') and self.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback after error (auto)...")
                    fallback_image = self.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful after error (auto)")
                        # ใช้ภาพจาก USB camera แทน
                        self.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        self.btn_process_cap.setEnabled(True)
                        self.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        
                        # Auto process
                        if True:
                            if hasattr(self, 'modbus_thread') and self.modbus_thread and self.modbus_thread.angle3_retry_mode:
                                print("🔄 USB FALLBACK AUTO: Angle3 retry mode - skipping cap processing")
                            else:
                                print("🔄 USB FALLBACK AUTO: Normal mode - processing both bottle and cap")
                                self.display_cap_processing_ui()
                                self.process_cap_detection()
                        return
            except:
                pass
            
            self.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def capture_sentech_image_from_queue(self):
        """Capture image from Sentech camera from queue (without waiting for M301)"""
        if self.sentech_camera is None:
            print("❌ Sentech camera not initialized - cannot capture from queue")
            return
        
        # รอให้กล้องเริ่มต้นเสร็จ (ถ้ายังไม่เสร็จ)
        max_wait_time = 10.0  # รอสูงสุด 10 วินาที
        wait_time = 0.0
        while hasattr(self.sentech_camera, '_is_initialized') and not self.sentech_camera._is_initialized and wait_time < max_wait_time:
            time.sleep(0.1)
            wait_time += 0.1
        
        if hasattr(self.sentech_camera, '_is_initialized') and not self.sentech_camera._is_initialized:
            print("❌ Sentech camera initialization timeout - cannot capture from queue")
            return
        
        try:
            print("📸 SENTECH QUEUE CAPTURE: Starting image capture from queue...")
            self.status_label.setText('📸 กำลังถ่ายภาพจาก Sentech จากคิว...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            
            # ถ่ายภาพทันทีโดยไม่รอ (เพราะเป็นงานจากคิว) (เหมือน sentech_modbus_trigger.py)
            print("📸 SENTECH QUEUE CAPTURE: Capturing image immediately...")
            self.status_label.setText('📸 กำลังถ่ายภาพจาก Sentech...')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            QtWidgets.QApplication.processEvents()
            
            # ใช้ capture_image() ที่เริ่ม acquisition ใหม่ (เหมือน sentech_modbus_trigger.py)
            try:
                print("📸 Calling sentech_camera.capture_image()...")
                captured_image = self.sentech_camera.capture_image()
                print(f"📸 Capture result: {captured_image is not None}")
            except Exception as e:
                print(f"❌ Error calling sentech_camera.capture_image(): {e}")
                import traceback
                traceback.print_exc()
                captured_image = None
            
            if captured_image is not None:
                # ล้างข้อมูลการประมวลผลฝาเก่าก่อนที่จะใช้ภาพใหม่
                self.current_cap_result = None
                if hasattr(self, 'cap_detection_text') and self.cap_detection_text:
                    self.cap_detection_text.clear()
                
                # ล้าง cap results layout
                if hasattr(self, 'cap_results_layout') and self.cap_results_layout:
                    for i in reversed(range(self.cap_results_layout.count())):
                        widget = self.cap_results_layout.itemAt(i).widget()
                        if widget:
                            widget.setParent(None)
                
                # แสดงเวลาที่ถ่ายภาพ
                import datetime
                capture_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"🕐 Sentech Queue Capture time: {capture_time}")
                print(f"📸 SENTECH QUEUE CAPTURE: Image shape: {captured_image.shape}")
                
                self.current_sentech_image = captured_image
                self.display_sentech_image(captured_image)
                self.sentech_image_info_label.setText(f"ขนาด: {captured_image.shape[1]}x{captured_image.shape[0]} | เวลา: {capture_time} | จากคิว")
                self.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
                self.status_label.setText('✅ ถ่ายภาพจาก Sentech จากคิวสำเร็จ - พร้อมประมวลผลฝา')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                
                # Auto process is always enabled (BOTTLE AND CAP processing enabled)
                if True:  # Auto mode is always enabled
                    # ตรวจสอบว่าเป็นโหมดถ่ายภาพซ้ำหรือไม่
                    if hasattr(self, 'modbus_thread') and self.modbus_thread and self.modbus_thread.angle3_retry_mode:
                        print("🔄 SENTECH QUEUE PROCESS: Angle3 retry mode - skipping cap processing")
                        # ไม่ประมวลผลฝาในโหมดถ่ายภาพซ้ำ
                    else:
                        print("🔄 SENTECH QUEUE PROCESS: Normal mode - processing both bottle and cap")
                        # แสดงผล GUI สำหรับการประมวลผลฝา
                        self.display_cap_processing_ui()
                        self.process_cap_detection()  # ENABLED - cap processing turned on
                else:
                    print("⚠️ Auto processing disabled - Sentech image captured but not processed")
            else:
                print("❌ Failed to capture from Sentech queue - trying USB camera fallback...")
                # ลองใช้ USB camera เป็น fallback
                if self.usb_camera and hasattr(self.usb_camera, 'is_opened') and self.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback for queue...")
                    fallback_image = self.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful for queue")
                        # ใช้ภาพจาก USB camera แทน
                        self.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} | เวลา: {capture_time} | USB Fallback")
                        self.btn_process_cap.setEnabled(True)
                        self.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        
                        # Auto process
                        if True:
                            if hasattr(self, 'modbus_thread') and self.modbus_thread and self.modbus_thread.angle3_retry_mode:
                                print("🔄 USB FALLBACK QUEUE: Angle3 retry mode - skipping cap processing")
                            else:
                                print("🔄 USB FALLBACK QUEUE: Normal mode - processing both bottle and cap")
                                self.display_cap_processing_ui()
                                self.process_cap_detection()
                        return
                
                self.status_label.setText('❌ ไม่สามารถถ่ายภาพจาก Sentech จากคิวได้')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            print(f"❌ Error capturing Sentech image (queue): {e} - trying USB camera fallback...")
            # ลองใช้ USB camera เป็น fallback
            try:
                if self.usb_camera and hasattr(self.usb_camera, 'is_opened') and self.usb_camera.is_opened():
                    print("🔄 Using USB camera as fallback after error (queue)...")
                    fallback_image = self.usb_camera.capture_image()
                    if fallback_image is not None:
                        print("✅ Fallback to USB camera successful after error (queue)")
                        # ใช้ภาพจาก USB camera แทน
                        self.current_sentech_image = fallback_image
                        self.display_sentech_image(fallback_image)
                        self.sentech_image_info_label.setText(f"ขนาด: {fallback_image.shape[1]}x{fallback_image.shape[0]} (USB Fallback)")
                        self.btn_process_cap.setEnabled(True)
                        self.status_label.setText('✅ ใช้ USB camera แทน Sentech - พร้อมประมวลผลฝา')
                        self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                        
                        # Auto process
                        if True:
                            if hasattr(self, 'modbus_thread') and self.modbus_thread and self.modbus_thread.angle3_retry_mode:
                                print("🔄 USB FALLBACK QUEUE: Angle3 retry mode - skipping cap processing")
                            else:
                                print("🔄 USB FALLBACK QUEUE: Normal mode - processing both bottle and cap")
                                self.display_cap_processing_ui()
                                self.process_cap_detection()
                        return
            except:
                pass
            
            self.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def display_sentech_image(self, image):
        """Display Sentech image in label"""
        try:
            if image is not None:
                print(f"🔍 Display image shape: {image.shape}, dtype: {image.dtype}")
                print(f"🔍 Display image min: {image.min()}, max: {image.max()}")
                
                # Handle grayscale image (Mono8 from SENTECH camera)
                if len(image.shape) == 2:
                    # Grayscale image - convert to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif len(image.shape) == 3:
                    # Color image - convert BGR to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # Unknown format - use as is
                    rgb_image = image
                
                h, w, c = rgb_image.shape
                bytes_per_line = c * w
                
                qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                
                # Scale pixmap to fit label while maintaining aspect ratio
                label_size = self.sentech_image_label.size()
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
                self.sentech_image_label.setPixmap(scaled_pixmap)
                
                print(f"✅ Image displayed successfully: {rgb_image.shape} -> scaled to {scaled_pixmap.size().width()}x{scaled_pixmap.size().height()} (full image visible)")
            else:
                self.sentech_image_label.setText("ไม่สามารถโหลดภาพได้")

        except Exception as e:
            print(f"❌ Error displaying image: {e}")
            self.sentech_image_label.setText(f"ข้อผิดพลาด: {str(e)}")
    
    def process_cap_detection(self):
        """Process cap detection on Sentech image or selected image file"""
        # Check for image from either Sentech camera or file selection
        image_to_process = None
        image_source = ""
        
        if self.current_sentech_image is not None:
            image_to_process = self.current_sentech_image
            image_source = "Sentech camera"
        elif self.current_cap_image is not None:
            image_to_process = self.current_cap_image
            image_source = "selected file"
        else:
            QMessageBox.warning(self, "ข้อผิดพลาด", "ไม่มีภาพให้ประมวลผล\nกรุณาถ่ายภาพจากกล้อง Sentech หรือเลือกไฟล์ภาพ")
            return
        
        # Update status tab
        self.update_cap_detection_status("กำลังประมวลผล...", True)
        
        if not all([self.cap_detector, self.craft_detector, self.rotation_model, self.line_detector, self.ocr_model]):
            QMessageBox.warning(self, "ข้อผิดพลาด", "โมเดลตรวจจับฝายังไม่ได้โหลดเสร็จ")
            return
        
        print(f"🔄 CAP PROCESS: Starting cap detection for image shape: {image_to_process.shape} (from {image_source})")
        
        # แสดงผล GUI สำหรับการประมวลผลฝา
        # print("🔍 CAP PROCESS: Calling display_cap_processing_ui()")  # Reduced spam
        self.display_cap_processing_ui()
        # print("🔍 CAP PROCESS: display_cap_processing_ui() completed")  # Reduced spam
        
        # Start processing thread
        # print("🔍 CAP PROCESS: Setting up progress bar and status")  # Reduced spam
        self.cap_progress_bar.setVisible(True)
        self.cap_progress_bar.setValue(0)
        self.cap_progress_bar.setFormat("กำลังเริ่มต้น...")
        self.status_label.setText('🔄 กำลังประมวลผลฝา...')
        self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
        
        # Disable process button and enable stop button
        # print("🔍 CAP PROCESS: Updating button states")  # Reduced spam
        self.btn_process_cap.setEnabled(False)
        self.btn_stop_processing.setEnabled(True)
        
        # print("🔍 CAP PROCESS: Creating CapDetectionThread")  # Reduced spam
        self.cap_processing_thread = CapDetectionThread(
            image_to_process,
            self.cap_detector,
            self.craft_detector,
            self.rotation_model,
            self.line_detector,
            self.ocr_model,
            getattr(self, 'faded_text_yolo_model', None)
        )
        # print("🔍 CAP PROCESS: Connecting signals")  # Reduced spam
        self.cap_processing_thread.result_ready.connect(self.on_cap_processing_complete)
        self.cap_processing_thread.status_updated.connect(self.update_cap_progress)
        self.cap_processing_thread.progress_updated.connect(self.update_cap_progress_bar)
        print("🔄 CAP PROCESS: Starting cap detection thread...")
        self.cap_processing_thread.start()
        # print("🔍 CAP PROCESS: Thread started successfully")  # Reduced spam
    
    def update_cap_progress(self, status):
        """Update cap detection progress bar and status"""
        if "ตรวจจับฝา" in status:
            self.cap_progress_bar.setValue(20)
            self.cap_progress_bar.setFormat("Step 1: ตรวจจับฝา (20%)")
        elif "ตัดภาพ" in status:
            self.cap_progress_bar.setValue(40)
            self.cap_progress_bar.setFormat("Step 2: ตัดภาพ (40%)")
        elif "CRAFT" in status:
            self.cap_progress_bar.setValue(50)
            self.cap_progress_bar.setFormat("Step 2a: CRAFT (50%)")
        elif "AI rotation" in status:
            self.cap_progress_bar.setValue(60)
            self.cap_progress_bar.setFormat("Step 2b: AI Rotation (60%)")
        elif "บรรทัดข้อความ" in status:
            self.cap_progress_bar.setValue(80)
            self.cap_progress_bar.setFormat("Step 3: ตรวจจับบรรทัด (80%)")
        elif "OCR" in status:
            self.cap_progress_bar.setValue(90)
            self.cap_progress_bar.setFormat("Step 4: OCR (90%)")
        
        self.status_label.setText(status)
    
    def update_cap_progress_bar(self, value):
        """Update cap detection progress bar with specific value"""
        self.cap_progress_bar.setValue(value)
        self.cap_progress_bar.setFormat(f"กำลังประมวลผล... ({value}%)")
    
    def show_rotation_attempt(self, attempt_num, total_attempts, rotated_image, ocr_results=None, format_valid=None):
        """Display current rotation attempt image and OCR results"""
        try:
            # Convert numpy array to QPixmap
            height, width, channel = rotated_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rotated_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale the image to fit the display area
            scaled_pixmap = pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Update the sentech image display
            self.sentech_image_label.setPixmap(scaled_pixmap)
            self.sentech_image_label.setAlignment(Qt.AlignCenter)
            
            # Update status to show current rotation attempt
            angle = [0, 90, 180, 270, -90][attempt_num - 1]
            
            # Build status message with OCR results if available
            if ocr_results is not None and format_valid is not None:
                # Extract text from OCR results
                ocr_text = self.extract_ocr_text(ocr_results)
                status_text = f"📋 Step {attempt_num}: Rotation {angle}° | OCR: {ocr_text[:25]}{'...' if len(ocr_text) > 25 else ''} | {'✅ Valid' if format_valid else '❌ Invalid'}"
            else:
                status_text = f"📋 Step {attempt_num}: Rotating to {angle}°..."
            
            self.status_label.setText(status_text)
            
            # Force GUI update
            QApplication.processEvents()
            
            if ocr_results is not None:
                print(f"📋 STEP {attempt_num}: Rotation {angle}° | OCR: {self.extract_ocr_text(ocr_results)[:50]}... | {'✅ Valid' if format_valid else '❌ Invalid'}")
            else:
                print(f"📋 STEP {attempt_num}: Rotating to {angle}°...")
            
        except Exception as e:
            print(f"❌ Error displaying rotation attempt: {e}")
    
    def extract_ocr_text(self, ocr_results):
        """Extract text from OCR results for display"""
        try:
            if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                # Extract text from line results
                text_parts = []
                for line_result in ocr_results['line_results']:
                    if isinstance(line_result, dict) and 'text' in line_result:
                        text_parts.append(line_result['text'])
                return ' | '.join(text_parts) if text_parts else 'No text found'
            elif isinstance(ocr_results, str):
                return ocr_results
            else:
                return 'Unknown OCR format'
        except Exception as e:
            return f'Error extracting text: {e}'
    
    # Manual rotation function removed
    
    # Manual re-OCR function removed
    
    # Manual OCR display functions removed
    
    def update_bottle_progress_bar(self, value):
        """Update bottle detection progress bar with specific value"""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"กำลังประมวลผล... ({value}%)")
    
    def stop_processing(self):
        """Stop current processing"""
        try:
            # Stop bottle detection thread
            if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
                print("🛑 Stopping bottle detection thread...")
                self.processing_thread.terminate()
                self.processing_thread.wait(1000)  # Wait up to 1 second
                self.processing_thread = None
            
            # Stop cap detection thread
            if hasattr(self, 'cap_processing_thread') and self.cap_processing_thread and self.cap_processing_thread.isRunning():
                print("🛑 Stopping cap detection thread...")
                self.cap_processing_thread.terminate()
                self.cap_processing_thread.wait(1000)  # Wait up to 1 second
                self.cap_processing_thread = None
            
            # Reset UI
            self.progress_bar.setVisible(False)
            self.cap_progress_bar.setVisible(False)
            self.btn_process.setEnabled(True)
            self.btn_stop_processing.setEnabled(False)
            self.status_label.setText('⏹️ หยุดการประมวลผลแล้ว')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            
            print("✅ Processing stopped successfully")
            
        except Exception as e:
            print(f"❌ Error stopping processing: {e}")
            self.status_label.setText(f'❌ ข้อผิดพลาดในการหยุด: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def on_cap_processing_complete(self, result):
        """Handle cap detection processing completion"""
        print("✅ CAP PROCESS COMPLETE: Cap detection finished")
        self.cap_progress_bar.setVisible(False)
        
        # Update performance stats
        self.total_images_processed_count += 1
        self.update_performance_stats(images_processed=self.total_images_processed_count)
        
        # Reset button states
        self.btn_process_cap.setEnabled(True)  # ENABLED - cap processing turned on
        self.btn_stop_processing.setEnabled(False)
        
        # ตรวจสอบว่าเป็นข้อความจางหรือไม่
        if "error" in result and result["error"] == "faded_text_detected":
            print("❌ CAP PROCESS COMPLETE: ตรวจพบข้อความจาง - หยุดประมวลผล")
            print(f"❌ CAP PROCESS COMPLETE: {result.get('message', 'Unknown error')}")
            
            # Update status tab
            self.update_cap_detection_status("ข้อความจาง - หยุดประมวลผล", False)
            
            # Display cap detection results (แสดงผลลัพธ์ฝาที่เจอข้อความจาง)
            self.display_cap_detection_results(result)
            
            # Update status label
            self.status_label.setText('❌ ตรวจพบข้อความจาง - หยุดประมวลผล')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            
            return
        
        if "error" not in result:
            print("✅ CAP PROCESS COMPLETE: No error in result")
            print(f"✅ CAP PROCESS COMPLETE: Result keys: {list(result.keys())}")
            
            # Store result
            self.current_cap_result = result
            
            # ไม่ validate ที่นี่ - รอให้การประมวลผลเสร็จสิ้นก่อน
            print("ℹ️ CAP PROCESS COMPLETE: Skipping early validation - will validate after processing is complete")
            
            # Update status tab
            self.update_cap_detection_status("เสร็จสิ้น", False)
            
            # Display cap detection results (แสดงผลลัพธ์ฝาทุกกรณี)
            # print("🔄 CAP PROCESS COMPLETE: Calling display_cap_detection_results")  # Reduced spam
            self.display_cap_detection_results(result)
            # print("✅ CAP PROCESS COMPLETE: display_cap_detection_results completed")  # Reduced spam
            
            # Update status
            if self.current_bottle_type not in ["M100", "M110", "M120"]:
                self.status_label.setText('✅ ประมวลผลฝาเสร็จสิ้น')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            # แสดงผล GUI ที่ชัดเจนว่าการประมวลผลเสร็จสิ้นแล้ว
            self.display_cap_processing_complete_ui()
            
            # ON M100/M110/M120 ตาม bottle type ทันทีหลังจากตรวจฝาเสร็จ (ไม่ต้องกด OK)
            if self.current_bottle_type in ["M100", "M110", "M120"]:
                print("🔄 CAP PROCESS COMPLETE: Calling on_bottle_type_after_cap_validation immediately after cap validation")
                self.on_bottle_type_after_cap_validation()
        else:
            print(f"❌ CAP PROCESS COMPLETE: Error in result: {result['error']}")
            self.status_label.setText(f'❌ ข้อผิดพลาด: {result["error"]}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            
            # Show error message
            QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการประมวลผลฝา: {result['error']}")
    
    def validate_cap_result(self, cap_result):
        """
        ตรวจสอบความถูกต้องของผลลัพธ์ฝาสำหรับ M100/M110/M120
        
        Args:
            cap_result: ผลลัพธ์จากการประมวลผลฝา
            
        Returns:
            bool: True ถ้าผลลัพธ์ถูกต้อง, False ถ้าไม่ถูกต้อง
        """
        try:
            print(f"🔍 CAP VALIDATION: Validating cap result for {self.current_bottle_type}")
            
            # ตรวจสอบว่ามีผลลัพธ์ OCR หรือไม่
            # ตรวจสอบใน cap_processing_results ก่อน
            ocr_results = None
            if cap_result.get('cap_processing_results'):
                # ใช้ผลลัพธ์จาก cap_processing_results
                cap_processing_results = cap_result['cap_processing_results']
                if cap_processing_results and len(cap_processing_results) > 0:
                    # ใช้ผลลัพธ์จากฝาแรก
                    first_cap = cap_processing_results[0]
                    if first_cap.get('ocr_results'):
                        ocr_results = first_cap['ocr_results']
                        print(f"🔍 CAP VALIDATION: Using cap_processing_results")
            elif cap_result.get('final_ocr_results'):
                # ใช้ผลลัพธ์จาก final_ocr_results (fallback)
                ocr_results = cap_result['final_ocr_results']
                print(f"🔍 CAP VALIDATION: Using final_ocr_results")
            elif cap_result.get('ocr_results'):
                # ใช้ผลลัพธ์จาก ocr_results (สำหรับกรณีไม่มีฝา)
                ocr_results = cap_result['ocr_results']
                print(f"🔍 CAP VALIDATION: Using ocr_results (no cap detected)")
            
            if not ocr_results:
                print("❌ CAP VALIDATION: No OCR results found")
                return False
            
            # ตรวจสอบ faded text detection status
            faded_text_status = None
            if cap_result.get('cap_processing_results'):
                cap_processing_results = cap_result['cap_processing_results']
                if cap_processing_results and len(cap_processing_results) > 0:
                    first_cap = cap_processing_results[0]
                    if first_cap.get('faded_text_result'):
                        faded_text_status = first_cap['faded_text_result'].get('status', 'unknown')
                        print(f"🔍 CAP VALIDATION: Faded text status: {faded_text_status}")
                        
                        # หยุดประมวลผลและส่ง M140 เมื่อเจอข้อความจาง
                        if faded_text_status == 'faded':
                            print("❌ CAP VALIDATION: ตรวจพบข้อความจาง - หยุดประมวลผล")
                            print("🚨 CAP VALIDATION: ส่งสัญญาณ M140 (ฝาไม่ผ่าน)")
                            print("🚀 CAP VALIDATION: ส่งสัญญาณ M600 (ประมวลผลเสร็จสิ้น)")
                            
                            # ส่งสัญญาณ M140 (ฝาไม่ผ่าน)
                            if hasattr(self, 'modbus_thread') and self.modbus_thread:
                                self.modbus_thread.on_m140()
                                print("✅ M140 ส่งสัญญาณเรียบร้อย")
                                
                                # ส่งสัญญาณ M600 (ประมวลผลเสร็จสิ้น)
                                self.modbus_thread.on_m600()
                                print("✅ M600 ส่งสัญญาณเรียบร้อย")
                            
                            return False
            
            # ตรวจสอบว่ามีข้อความที่อ่านได้หรือไม่
            ocr_text = ""
            
            # ตรวจสอบว่า ocr_results เป็น dictionary หรือ list
            if isinstance(ocr_results, dict):
                # ถ้าเป็น dictionary ให้หา total_recognized_text หรือ text
                if 'total_recognized_text' in ocr_results and ocr_results['total_recognized_text']:
                    ocr_text = ocr_results['total_recognized_text']
                    print(f"🔍 CAP VALIDATION: Using total_recognized_text")
                elif 'text' in ocr_results and ocr_results['text']:
                    ocr_text = ocr_results['text']
                    print(f"🔍 CAP VALIDATION: Using text field")
                elif 'recognized_text' in ocr_results and ocr_results['recognized_text']:
                    ocr_text = ocr_results['recognized_text']
                    print(f"🔍 CAP VALIDATION: Using recognized_text field")
                else:
                    # ถ้าไม่มี text fields หรือเป็น empty ให้ใช้ values ทั้งหมด
                    ocr_text = " ".join(str(v) for v in ocr_results.values() if v and str(v).strip())
                    print(f"🔍 CAP VALIDATION: Using all values")
            elif isinstance(ocr_results, list):
                # ถ้าเป็น list ให้วนลูปผ่าน items
                for ocr_result in ocr_results:
                    if isinstance(ocr_result, dict):
                        # ถ้าเป็น dictionary ให้ใช้ .get()
                        ocr_text += ocr_result.get('text', '') + " "
                    elif isinstance(ocr_result, str):
                        # ถ้าเป็น string ให้ใช้ตรงๆ
                        ocr_text += ocr_result + " "
                    else:
                        # ถ้าเป็นประเภทอื่น ให้แปลงเป็น string
                        ocr_text += str(ocr_result) + " "
                ocr_text = ocr_text.strip()
            else:
                # ถ้าเป็นประเภทอื่น ให้แปลงเป็น string
                ocr_text = str(ocr_results)
            
            ocr_text = ocr_text.strip()
            
            if not ocr_text:
                print("❌ CAP VALIDATION: No text found in OCR results")
                return False
            
            print(f"🔍 CAP VALIDATION: OCR text: '{ocr_text}'")
            print(f"🔍 CAP VALIDATION: OCR results type: {type(ocr_results)}")
            if isinstance(ocr_results, dict):
                print(f"🔍 CAP VALIDATION: OCR results keys: {list(ocr_results.keys())}")
                # ลดการ print ข้อมูลที่เยอะเกินไป
                if 'line_results' in ocr_results:
                    print(f"🔍 CAP VALIDATION: Found {len(ocr_results['line_results'])} lines")
                if 'total_lines' in ocr_results:
                    print(f"🔍 CAP VALIDATION: Total lines: {ocr_results['total_lines']}")
            elif isinstance(ocr_results, list):
                print(f"🔍 CAP VALIDATION: OCR results length: {len(ocr_results)}")
                # ลดการ print ข้อมูลที่เยอะเกินไป
                if len(ocr_results) > 0:
                    print(f"🔍 CAP VALIDATION: First OCR result type: {type(ocr_results[0])}")
                    # ไม่ print ข้อมูลทั้งหมดของ OCR result
            
            # ตรวจสอบรูปแบบวันที่/เวลาตาม bottle type
            if self.current_bottle_type == "M100":
                # M100 (ดั้งเดิม) - ตรวจสอบรูปแบบวันที่/เวลา
                return self.validate_date_time_format(ocr_text)
            elif self.current_bottle_type == "M110":
                # M110 (น้ำตาล 2%) - ตรวจสอบรูปแบบวันที่/เวลา
                return self.validate_date_time_format(ocr_text)
            elif self.current_bottle_type == "M120":
                # M120 (ผสมแมงลัก) - ตรวจสอบรูปแบบวันที่/เวลา
                return self.validate_date_time_format(ocr_text)
            
            return True
            
        except Exception as e:
            print(f"❌ CAP VALIDATION ERROR: {e}")
            return False
    
    def validate_date_time_format(self, ocr_text):
        """
        ตรวจสอบรูปแบบวันที่/เวลาจากข้อความ OCR และคัดกรองวันหมดอายุตามโหมดที่เลือก
        
        Args:
            ocr_text: ข้อความที่อ่านได้จาก OCR
            
        Returns:
            bool: True ถ้ารูปแบบถูกต้องและผ่านการคัดกรอง, False ถ้าไม่ถูกต้องหรือไม่ผ่านการคัดกรอง
        """
        try:
            print(f"🔍 DATE TIME VALIDATION: Checking format for text: '{ocr_text[:50]}...'")  # แสดงแค่ 50 ตัวอักษรแรก
            
            # รูปแบบที่ยอมรับ: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',  # DD/MM/YYYY หรือ DD-MM-YYYY
                r'\d{1,2}\.\d{1,2}\.\d{4}',      # DD.MM.YYYY
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD หรือ YYYY-MM-DD
                r'\d{4}\.\d{1,2}\.\d{1,2}',      # YYYY.MM.DD
            ]
            
            # รูปแบบเวลาที่ยอมรับ: HH:MM, HH:MM:SS
            time_patterns = [
                r'\d{1,2}:\d{2}(:\d{2})?',       # HH:MM หรือ HH:MM:SS
            ]
            
            import re
            from datetime import datetime
            
            # ตรวจสอบว่ามีรูปแบบวันที่หรือไม่
            has_date = any(re.search(pattern, ocr_text) for pattern in date_patterns)
            
            # ตรวจสอบว่ามีรูปแบบเวลาหรือไม่
            has_time = any(re.search(pattern, ocr_text) for pattern in time_patterns)
            
            # ต้องมีอย่างน้อยรูปแบบวันที่หรือเวลา
            format_valid = has_date or has_time
            
            print(f"🔍 DATE TIME VALIDATION: has_date={has_date}, has_time={has_time}, format_valid={format_valid}")
            
            # ถ้ารูปแบบไม่ถูกต้อง ให้ return False
            if not format_valid:
                return False
            
            # ถ้าเป็นโหมดปกติ (ไม่คัดกรองวัน) ให้ return True
            if self.current_expiry_mode == "normal":
                print("🔍 DATE TIME VALIDATION: Normal mode - no expiry filtering")
                return True
            
            # ถ้าเป็นโหมดคัดกรองวัน ให้ตรวจสอบวันหมดอายุ
            if self.current_expiry_mode == "filter":
                print("🔍 DATE TIME VALIDATION: Filter mode - checking expiry date")
                
                # หาวันหมดอายุจากข้อความ OCR (BBF format)
                expiry_date = self.extract_expiry_date_from_text(ocr_text)
                if expiry_date is None:
                    print("🔍 DATE TIME VALIDATION: No expiry date found in text")
                    return False
                
                # ตรวจสอบการคัดกรองวันหมดอายุ
                filter_result = self.check_expiry_date_filter(expiry_date)
                print(f"🔍 DATE TIME VALIDATION: Expiry date {expiry_date} filter result: {filter_result}")
                
                return filter_result
            
            return format_valid
            
        except Exception as e:
            print(f"❌ DATE TIME VALIDATION ERROR: {e}")
            return False
    
    def extract_expiry_date_from_text(self, ocr_text):
        """
        สกัดวันหมดอายุจากข้อความ OCR (BBF format)
        
        Args:
            ocr_text: ข้อความที่อ่านได้จาก OCR
            
        Returns:
            datetime.date: วันหมดอายุ หรือ None ถ้าไม่พบ
        """
        try:
            import re
            from datetime import datetime
            
            # รูปแบบ BBF ที่ยอมรับ: BBF DD/MM/YY, BBF DD/MM/YYYY, BBF DD-MM-YY, BBF DD-MM-YYYY
            bbf_patterns = [
                r'BBF\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # BBF DD/MM/YY หรือ BBF DD/MM/YYYY
                r'BBF\s*(\d{1,2})\.(\d{1,2})\.(\d{2,4})',      # BBF DD.MM.YY หรือ BBF DD.MM.YYYY
            ]
            
            for pattern in bbf_patterns:
                match = re.search(pattern, ocr_text, re.IGNORECASE)
                if match:
                    day = int(match.group(1))
                    month = int(match.group(2))
                    year = int(match.group(3))
                    
                    # แปลงปี 2 หลักเป็น 4 หลัก (ถ้าปี < 50 ให้เป็น 20xx, ถ้า >= 50 ให้เป็น 19xx)
                    if year < 100:
                        if year < 50:
                            year += 2000
                        else:
                            year += 1900
                    
                    try:
                        expiry_date = datetime(year, month, day).date()
                        print(f"🔍 EXPIRY EXTRACTION: Found BBF date: {expiry_date}")
                        return expiry_date
                    except ValueError:
                        print(f"🔍 EXPIRY EXTRACTION: Invalid date: {day}/{month}/{year}")
                        continue
            
            print("🔍 EXPIRY EXTRACTION: No BBF date found in text")
            return None
            
        except Exception as e:
            print(f"❌ EXPIRY EXTRACTION ERROR: {e}")
            return None
    
    def check_expiry_date_filter(self, expiry_date):
        """
        ตรวจสอบวันหมดอายุตามการคัดกรองที่ตั้งค่าไว้
        
        Args:
            expiry_date: วันหมดอายุที่สกัดได้
            
        Returns:
            bool: True ถ้าผ่านการคัดกรอง, False ถ้าไม่ผ่าน
        """
        try:
            start_date, end_date, specific_date = self.get_expiry_filter_dates()
            
            if self.expiry_filter_type == "range":
                if start_date is None or end_date is None:
                    print("🔍 EXPIRY FILTER: Range dates not set")
                    return False
                
                # ตรวจสอบว่าวันหมดอายุอยู่ในช่วงที่กำหนดหรือไม่
                is_in_range = start_date <= expiry_date <= end_date
                print(f"🔍 EXPIRY FILTER: {expiry_date} in range {start_date} - {end_date}: {is_in_range}")
                return is_in_range
                
            elif self.expiry_filter_type == "specific":
                if specific_date is None:
                    print("🔍 EXPIRY FILTER: Specific date not set")
                    return False
                
                # ตรวจสอบว่าวันหมดอายุตรงกับวันที่เฉพาะหรือไม่
                is_specific = expiry_date == specific_date
                print(f"🔍 EXPIRY FILTER: {expiry_date} matches specific date {specific_date}: {is_specific}")
                return is_specific
            
            return False
            
        except Exception as e:
            print(f"❌ EXPIRY FILTER ERROR: {e}")
            return False
    
    def on_bottle_type_after_cap_validation(self):
        """ON M100/M110/M120 หลังจากตรวจสอบฝาผ่านแล้ว"""
        try:
            print(f"🏷️ ON BOTTLE TYPE AFTER CAP VALIDATION: {self.current_bottle_type}")
            
            success = False
            if self.current_bottle_type == "M100":
                success = self.modbus_thread.on_m100()
                print("🏷️ ON M100 (พบคำว่า 'เดิม' - ฝาผ่าน)")
                if success:
                    self.update_coil_lamp("m100", True)
            elif self.current_bottle_type == "M110":
                success = self.modbus_thread.on_m110()
                print("🏷️ ON M110 (พบคำว่า '2%' - ฝาผ่าน)")
                if success:
                    self.update_coil_lamp("m110", True)
            elif self.current_bottle_type == "M120":
                success = self.modbus_thread.on_m120()
                print("🏷️ ON M120 (พบคำว่า 'ลัก' - ฝาผ่าน)")
                if success:
                    self.update_coil_lamp("m120", True)
            
            if success:
                # ON M600 หลังจาก ON M100/M110/M120
                print("⏳ ON M600 และรอ D6004=200 เพื่อ reset ทั้งหมด...")
                m600_success = self.modbus_thread.on_m600()
                if m600_success:
                    self.update_coil_lamp("m600", True)
                
                # อัปเดตสถานะ
                self.status_label.setText(f'✅ ฝาผ่าน → ON {self.current_bottle_type} + M600 (รอ reset)')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                
                print(f"✅ BOTTLE TYPE ON SUCCESS: {self.current_bottle_type} + M600")
            else:
                print(f"❌ ไม่สามารถ ON {self.current_bottle_type} ได้")
                self.status_label.setText(f'❌ ไม่สามารถ ON {self.current_bottle_type} ได้')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                
        except Exception as e:
            print(f"❌ ERROR in on_bottle_type_after_cap_validation: {e}")
            self.status_label.setText(f'❌ ข้อผิดพลาด: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
    
    def display_cap_detection_results(self, result):
        """Display cap detection results with step-by-step images"""
        try:
            # Debug: Check result type
            print(f"🔍 DEBUG: display_cap_detection_results called with result type: {type(result)}")
            if isinstance(result, str):
                print(f"🔍 DEBUG: Result is string: {result}")
                self.cap_detection_text.setText(f"Error: Received string instead of dictionary: {result}")
                return
            elif not isinstance(result, dict):
                print(f"🔍 DEBUG: Result is not dict: {type(result)}")
                self.cap_detection_text.setText(f"Error: Expected dictionary, got {type(result)}")
                return
            
            # print("🔍 DEBUG: Starting to display cap detection results")  # Reduced spam
            # print(f"🔍 DEBUG: Result keys: {list(result.keys())}")  # Reduced spam
            
            # Clear previous results
            for i in reversed(range(self.cap_results_layout.count())):
                self.cap_results_layout.itemAt(i).widget().setParent(None)
            
            # print("🔍 DEBUG: Cleared previous results")  # Reduced spam
            
            # Step 1: Original Image
            # print("🔍 DEBUG: Creating Step 1 - Original Image")  # Reduced spam
            original_container = QWidget()
            original_container.setStyleSheet("border: 2px solid #2980b9; margin: 5px; padding: 5px; background-color: white;")
            original_layout = QVBoxLayout(original_container)
            
            original_title = QLabel("📸 Step 1: ภาพต้นฉบับ (Original Image)")
            original_title.setStyleSheet("font-weight: bold; color: #2980b9; font-size: 12px;")
            original_title.setAlignment(Qt.AlignCenter)
            original_layout.addWidget(original_title)
            
            # Display original image
            if result.get('image') is not None:
                # print("🔍 DEBUG: Found original image in result")  # Reduced spam
                original_image = result['image']
                if hasattr(original_image, 'shape'):
                    # print(f"🔍 DEBUG: Original image shape: {original_image.shape}")  # Reduced spam
                    # Resize for display
                    h, w = original_image.shape[:2]
                    max_size = 200
                    if h > max_size or w > max_size:
                        scale = max_size / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        original_image = cv2.resize(original_image, (new_w, new_h))
                    
                    # Convert to RGB for Qt
                    rgb_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    h, w, c = rgb_original.shape
                    bytes_per_line = c * w
                    
                    qimg = QtGui.QImage(rgb_original.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qimg)
                    
                    original_label = QLabel()
                    original_label.setPixmap(pixmap)
                    original_label.setAlignment(Qt.AlignCenter)
                    original_layout.addWidget(original_label)
                    # print("🔍 DEBUG: Added original image to layout")  # Reduced spam
                else:
                    # print("🔍 DEBUG: Original image has no shape attribute")  # Reduced spam
                    pass
            else:
                # print("🔍 DEBUG: No original image found in result")  # Reduced spam
                pass
            
            self.cap_results_layout.addWidget(original_container)
            # print("🔍 DEBUG: Added original container to cap_results_layout")  # Reduced spam
            
            # Step 2: Processed Image (with cap detection)
            if result.get('processed_image') is not None:
                # print("🔍 DEBUG: Creating Step 2 - Processed Image")  # Reduced spam
                processed_container = QWidget()
                processed_container.setStyleSheet("border: 2px solid #e67e22; margin: 5px; padding: 5px; background-color: white;")
                processed_layout = QVBoxLayout(processed_container)
                
                processed_title = QLabel("🔍 Step 2: ภาพที่ประมวลผลแล้ว (Processed Image)")
                processed_title.setStyleSheet("font-weight: bold; color: #e67e22; font-size: 12px;")
                processed_title.setAlignment(Qt.AlignCenter)
                processed_layout.addWidget(processed_title)
                
                # Display processed image
                processed_image = result['processed_image']
                if hasattr(processed_image, 'shape'):
                    # print(f"🔍 DEBUG: Processed image shape: {processed_image.shape}")  # Reduced spam
                    # Resize for display
                    h, w = processed_image.shape[:2]
                    max_size = 200
                    if h > max_size or w > max_size:
                        scale = max_size / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        processed_image = cv2.resize(processed_image, (new_w, new_h))
                    
                    # Convert to RGB for Qt
                    rgb_processed = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    h, w, c = rgb_processed.shape
                    bytes_per_line = c * w
                    
                    qimg = QtGui.QImage(rgb_processed.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qimg)
                    
                    processed_label = QLabel()
                    processed_label.setPixmap(pixmap)
                    processed_label.setAlignment(Qt.AlignCenter)
                    processed_layout.addWidget(processed_label)
                    # print("🔍 DEBUG: Added processed image to layout")  # Reduced spam
                else:
                    # print("🔍 DEBUG: Processed image has no shape attribute")  # Reduced spam
                    pass
                
                self.cap_results_layout.addWidget(processed_container)
                # print("🔍 DEBUG: Added processed container to cap_results_layout")  # Reduced spam
            else:
                # print("🔍 DEBUG: No processed image found in result")  # Reduced spam
                pass
            
            # Display cap detection results
            if result.get('cropped_images'):
                # Check if we have full pipeline results for caps
                if result.get('cap_processing_results'):
                    print(f"🔍 DEBUG: Found {len(result['cap_processing_results'])} cap processing results")
                    # Display full pipeline results for each cap
                    for cap_index, cap_result in enumerate(result['cap_processing_results']):
                        print(f"🔍 DEBUG: Cap {cap_index+1} result keys: {list(cap_result.keys())}")
                        print(f"🔍 DEBUG: Cap {cap_index+1} has line_detection_result: {'line_detection_result' in cap_result}")
                        print(f"🔍 DEBUG: Cap {cap_index+1} has ocr_results: {'ocr_results' in cap_result}")
                        if 'ocr_results' in cap_result:
                            print(f"🔍 DEBUG: Cap {cap_index+1} OCR results count: {len(cap_result['ocr_results'])}")
                        if 'line_detection_result' in cap_result:
                            line_result = cap_result['line_detection_result']
                            print(f"🔍 DEBUG: Cap {cap_index+1} line detection result type: {type(line_result)}")
                            if isinstance(line_result, dict):
                                print(f"🔍 DEBUG: Cap {cap_index+1} line detection keys: {list(line_result.keys())}")
                                print(f"🔍 DEBUG: Cap {cap_index+1} has error: {'error' in line_result}")
                                if 'cropped_lines' in line_result:
                                    print(f"🔍 DEBUG: Cap {cap_index+1} cropped_lines count: {len(line_result['cropped_lines'])}")
                                    if line_result['cropped_lines']:
                                        print(f"🔍 DEBUG: Cap {cap_index+1} first cropped_line type: {type(line_result['cropped_lines'][0])}")
                                        if isinstance(line_result['cropped_lines'][0], dict):
                                            print(f"🔍 DEBUG: Cap {cap_index+1} first cropped_line keys: {list(line_result['cropped_lines'][0].keys())}")
                        
                        cap_index = cap_result['cap_index']
                        
                        # Original cropped cap
                        crop_container = QWidget()
                        crop_container.setStyleSheet("border: 2px solid #27ae60; margin: 5px; padding: 5px; background-color: white;")
                        crop_layout = QVBoxLayout(crop_container)
                        
                        crop_title = QLabel(f"✂️ Step 3: ฝาที่ตรวจจับได้ {cap_index+1} (Cropped Cap {cap_index+1})")
                        crop_title.setStyleSheet("font-weight: bold; color: #27ae60; font-size: 12px;")
                        crop_title.setAlignment(Qt.AlignCenter)
                        crop_layout.addWidget(crop_title)
                        
                        # Display original cropped image
                        original_crop = cap_result['original_crop']
                        if hasattr(original_crop, 'shape'):
                            # Resize for display
                            h, w = original_crop.shape[:2]
                            max_size = 200
                            if h > max_size or w > max_size:
                                scale = max_size / max(h, w)
                                new_w, new_h = int(w * scale), int(h * scale)
                                original_crop = cv2.resize(original_crop, (new_w, new_h))
                            
                            # Convert to RGB for Qt
                            rgb_crop = cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB)
                            h, w, c = rgb_crop.shape
                            bytes_per_line = c * w
                            
                            qimg = QtGui.QImage(rgb_crop.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                            pixmap = QtGui.QPixmap.fromImage(qimg)
                            
                            crop_label = QLabel()
                            crop_label.setPixmap(pixmap)
                            crop_label.setAlignment(Qt.AlignCenter)
                            crop_layout.addWidget(crop_label)
                        
                        self.cap_results_layout.addWidget(crop_container)
                        
                        # Faded text detection results (แสดงก่อน CRAFT rotation)
                        if cap_result.get('faded_text_result'):
                            faded_result = cap_result['faded_text_result']
                            status = faded_result.get('status', 'unknown')
                            num_chars = faded_result.get('num_chars', 0)
                            total_area = faded_result.get('total_area', 0)
                            
                            # Choose color based on status
                            if status == 'faded':
                                color = "#e74c3c"  # Red for faded
                                emoji = "⚠️"
                            elif status == 'normal':
                                color = "#27ae60"  # Green for normal
                                emoji = "✅"
                            else:
                                color = "#95a5a6"  # Gray for unknown
                                emoji = "❓"
                            
                            # Main faded text detection container
                            faded_container = QWidget()
                            faded_container.setStyleSheet("border: 2px solid #f39c12; margin: 5px; padding: 5px; background-color: white;")
                            faded_layout = QVBoxLayout(faded_container)
                            
                            faded_title = QLabel(f"{emoji} Step 2a.5: ตรวจสอบรอยจางฝาที่ {cap_index+1} (ก่อน CRAFT)")
                            faded_title.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 12px;")
                            faded_title.setAlignment(Qt.AlignCenter)
                            faded_layout.addWidget(faded_title)
                            
                            # Display faded text detection info
                            faded_info = QLabel(f"สถานะ: {status.upper()}\nจำนวนตัวอักษร: {num_chars}\nพื้นที่รวม: {total_area}")
                            faded_info.setStyleSheet(f"color: {color}; font-size: 10px; padding: 5px;")
                            faded_info.setAlignment(Qt.AlignCenter)
                            faded_layout.addWidget(faded_info)
                            
                            self.cap_results_layout.addWidget(faded_container)
                            
                            # Display debug images if available
                            if 'debug_images' in faded_result and faded_result['debug_images']:
                                debug_images = faded_result['debug_images']
                                
                                # Create debug images container
                                debug_container = QWidget()
                                debug_container.setStyleSheet("border: 1px solid #95a5a6; margin: 5px; padding: 5px; background-color: #f8f9fa;")
                                debug_layout = QVBoxLayout(debug_container)
                                
                                debug_title = QLabel("🔍 Debug Images - Faded Text Detection")
                                debug_title.setStyleSheet("font-weight: bold; color: #34495e; font-size: 11px;")
                                debug_title.setAlignment(Qt.AlignCenter)
                                debug_layout.addWidget(debug_title)
                                
                                # Create horizontal layout for debug images
                                debug_images_layout = QHBoxLayout()
                                
                                # Display key debug images
                                debug_image_names = ['roi_show', 'gray', 'mask_circle', 'roi_masked', 'edges', 'text_edges']
                                for img_name in debug_image_names:
                                    if img_name in debug_images and debug_images[img_name] is not None:
                                        img = debug_images[img_name]
                                        
                                        # Resize for display
                                        h, w = img.shape[:2]
                                        max_size = 120
                                        if h > max_size or w > max_size:
                                            scale = max_size / max(h, w)
                                            new_w, new_h = int(w * scale), int(h * scale)
                                            img = cv2.resize(img, (new_w, new_h))
                                        
                                        # Convert to RGB for Qt
                                        if len(img.shape) == 2:  # Grayscale
                                            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                                        else:  # BGR
                                            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        
                                        h, w, c = rgb_img.shape
                                        bytes_per_line = c * w
                                        
                                        qimg = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                                        pixmap = QtGui.QPixmap.fromImage(qimg)
                                        
                                        # Create image container
                                        img_container = QWidget()
                                        img_container.setStyleSheet("border: 1px solid #bdc3c7; margin: 2px; padding: 2px; background-color: white;")
                                        img_layout = QVBoxLayout(img_container)
                                        
                                        # Image title
                                        img_title = QLabel(img_name.replace('_', ' ').title())
                                        img_title.setStyleSheet("font-size: 9px; color: #2c3e50; font-weight: bold;")
                                        img_title.setAlignment(Qt.AlignCenter)
                                        img_layout.addWidget(img_title)
                                        
                                        # Image
                                        img_label = QLabel()
                                        img_label.setPixmap(pixmap)
                                        img_label.setAlignment(Qt.AlignCenter)
                                        img_layout.addWidget(img_label)
                                        
                                        debug_images_layout.addWidget(img_container)
                                
                                debug_layout.addLayout(debug_images_layout)
                                self.cap_results_layout.addWidget(debug_container)
                        
                        # CRAFT rotated image
                        if cap_result.get('craft_rotated_image') is not None:
                            craft_container = QWidget()
                            craft_container.setStyleSheet("border: 2px solid #8e44ad; margin: 5px; padding: 5px; background-color: white;")
                            craft_layout = QVBoxLayout(craft_container)
                            
                            craft_title = QLabel(f"🔄 Step 2a: ฝาที่ {cap_index+1} หมุนด้วย CRAFT")
                            craft_title.setStyleSheet("font-weight: bold; color: #8e44ad; font-size: 12px;")
                            craft_title.setAlignment(Qt.AlignCenter)
                            craft_layout.addWidget(craft_title)
                            
                            # Display CRAFT rotated image
                            craft_image = cap_result['craft_rotated_image']
                            if hasattr(craft_image, 'shape'):
                                # Resize for display
                                h, w = craft_image.shape[:2]
                                max_size = 200
                                if h > max_size or w > max_size:
                                    scale = max_size / max(h, w)
                                    new_w, new_h = int(w * scale), int(h * scale)
                                    craft_image = cv2.resize(craft_image, (new_w, new_h))
                                
                                # Convert to RGB for Qt
                                rgb_craft = cv2.cvtColor(craft_image, cv2.COLOR_BGR2RGB)
                                h, w, c = rgb_craft.shape
                                bytes_per_line = c * w
                                
                                qimg = QtGui.QImage(rgb_craft.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                                pixmap = QtGui.QPixmap.fromImage(qimg)
                                
                                craft_label = QLabel()
                                craft_label.setPixmap(pixmap)
                                craft_label.setAlignment(Qt.AlignCenter)
                                craft_layout.addWidget(craft_label)
                            
                            self.cap_results_layout.addWidget(craft_container)
                        
                        # AI rotated image
                        if cap_result.get('ai_rotated_image') is not None:
                            ai_container = QWidget()
                            ai_container.setStyleSheet("border: 2px solid #e74c3c; margin: 5px; padding: 5px; background-color: white;")
                            ai_layout = QVBoxLayout(ai_container)
                            
                            ai_title = QLabel(f"🤖 Step 2b: ฝาที่ {cap_index+1} หมุนด้วย AI")
                            ai_title.setStyleSheet("font-weight: bold; color: #e74c3c; font-size: 12px;")
                            ai_title.setAlignment(Qt.AlignCenter)
                            ai_layout.addWidget(ai_title)
                            
                            # Display AI rotated image
                            ai_image = cap_result['ai_rotated_image']
                            
                            # Manual rotation removed
                            
                            if hasattr(ai_image, 'shape'):
                                # Resize for display
                                h, w = ai_image.shape[:2]
                                max_size = 200
                                if h > max_size or w > max_size:
                                    scale = max_size / max(h, w)
                                    new_w, new_h = int(w * scale), int(h * scale)
                                    ai_image = cv2.resize(ai_image, (new_w, new_h))
                                
                                # Convert to RGB for Qt
                                rgb_ai = cv2.cvtColor(ai_image, cv2.COLOR_BGR2RGB)
                                h, w, c = rgb_ai.shape
                                bytes_per_line = c * w
                                
                                qimg = QtGui.QImage(rgb_ai.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                                pixmap = QtGui.QPixmap.fromImage(qimg)
                                
                                ai_label = QLabel()
                                ai_label.setPixmap(pixmap)
                                ai_label.setAlignment(Qt.AlignCenter)
                                ai_layout.addWidget(ai_label)
                            
                            self.cap_results_layout.addWidget(ai_container)
                        
                        # Line detection results
                        if cap_result.get('line_detection_result') and 'error' not in cap_result.get('line_detection_result', {}):
                            print(f"🔍 DEBUG: Cap {cap_index+1} has line_detection_result")
                            line_container = QWidget()
                            line_container.setStyleSheet("border: 2px solid #9b59b6; margin: 5px; padding: 5px; background-color: white;")
                            line_layout = QVBoxLayout(line_container)
                            
                            line_title = QLabel(f"📏 Step 3: ตรวจจับบรรทัดข้อความฝาที่ {cap_index+1}")
                            line_title.setStyleSheet("font-weight: bold; color: #9b59b6; font-size: 12px;")
                            line_title.setAlignment(Qt.AlignCenter)
                            line_layout.addWidget(line_title)
                            
                            # Display line detection results
                            line_result = cap_result['line_detection_result']
                            if isinstance(line_result, dict):
                                # Show number of detected lines
                                num_lines = len(line_result.get('cropped_lines', []))
                                print(f"🔍 DEBUG: Line detection result keys: {list(line_result.keys())}")
                                print(f"🔍 DEBUG: Number of cropped lines: {num_lines}")
                                line_info = QLabel(f"📊 จำนวนบรรทัดที่ตรวจจับได้: {num_lines}")
                                line_info.setStyleSheet("color: #2c3e50; font-size: 11px; background-color: #ecf0f1; padding: 5px;")
                                line_info.setAlignment(Qt.AlignCenter)
                                line_layout.addWidget(line_info)
                                
                                # Display cropped line images if available
                                if 'cropped_lines' in line_result and line_result['cropped_lines']:
                                    print(f"🔍 DEBUG: Found {len(line_result['cropped_lines'])} cropped lines")
                                    print(f"🔍 DEBUG: First cropped line type: {type(line_result['cropped_lines'][0])}")
                                    if isinstance(line_result['cropped_lines'][0], dict):
                                        print(f"🔍 DEBUG: First cropped line keys: {list(line_result['cropped_lines'][0].keys())}")
                                    
                                    lines_title = QLabel("🖼️ ภาพบรรทัดที่ตรวจจับได้:")
                                    lines_title.setStyleSheet("font-weight: bold; color: #8e44ad; font-size: 10px;")
                                    line_layout.addWidget(lines_title)
                                    
                                    for line_idx, line_data in enumerate(line_result['cropped_lines'][:3]):  # Show max 3 lines
                                        if isinstance(line_data, dict) and 'image' in line_data:
                                            line_image = line_data['image']
                                        elif hasattr(line_data, 'shape'):  # Fallback for direct image array
                                            line_image = line_data
                                        else:
                                            continue
                                        
                                        if line_image is not None and hasattr(line_image, 'shape'):
                                            print(f"🔍 DEBUG: Processing line {line_idx+1}, shape: {line_image.shape}")
                                            # Resize for display
                                            h, w = line_image.shape[:2]
                                            max_size = 150
                                            if h > max_size or w > max_size:
                                                scale = max_size / max(h, w)
                                                new_w, new_h = int(w * scale), int(h * scale)
                                                line_image = cv2.resize(line_image, (new_w, new_h))
                                            
                                            # Convert to RGB for Qt
                                            rgb_line = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
                                            h, w, c = rgb_line.shape
                                            bytes_per_line = c * w
                                            
                                            qimg = QtGui.QImage(rgb_line.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                                            pixmap = QtGui.QPixmap.fromImage(qimg)
                                            
                                            line_label = QLabel()
                                            line_label.setPixmap(pixmap)
                                            line_label.setAlignment(Qt.AlignCenter)
                                            line_label.setStyleSheet("border: 1px solid #9b59b6; margin: 2px;")
                                            line_layout.addWidget(line_label)
                                            print(f"🔍 DEBUG: Added line {line_idx+1} to UI")
                                        else:
                                            print(f"🔍 DEBUG: Line {line_idx+1} is None or has no shape attribute")
                                    
                                    if len(line_result['cropped_lines']) > 3:
                                        more_lines = QLabel(f"... และอีก {len(line_result['cropped_lines']) - 3} บรรทัด")
                                        more_lines.setStyleSheet("color: #7f8c8d; font-size: 10px; font-style: italic;")
                                        more_lines.setAlignment(Qt.AlignCenter)
                                        line_layout.addWidget(more_lines)
                            
                            self.cap_results_layout.addWidget(line_container)
                        
                        # OCR results
                        if cap_result.get('ocr_results'):
                            ocr_container = QWidget()
                            ocr_container.setStyleSheet("border: 2px solid #f39c12; margin: 5px; padding: 5px; background-color: white;")
                            ocr_layout = QVBoxLayout(ocr_container)
                            
                            ocr_title = QLabel(f"📝 Step 4: OCR ผลลัพธ์ฝาที่ {cap_index+1}")
                            ocr_title.setStyleSheet("font-weight: bold; color: #f39c12; font-size: 12px;")
                            ocr_title.setAlignment(Qt.AlignCenter)
                            ocr_layout.addWidget(ocr_title)
                            
                            # Display OCR results
                            ocr_results = cap_result['ocr_results']
                            
                            # Check if ocr_results is a dictionary (from recognize_text_from_craft_lines)
                            if isinstance(ocr_results, dict):
                                # Handle the dictionary format from recognize_text_from_craft_lines
                                if 'line_results' in ocr_results:
                                    line_results = ocr_results['line_results']
                                    total_lines = ocr_results.get('total_lines', 0)
                                    
                                    # Show summary
                                    summary_label = QLabel(f"📊 จำนวนบรรทัดที่อ่านได้: {total_lines}")
                                    summary_label.setStyleSheet("color: #2c3e50; font-size: 11px; background-color: #ecf0f1; padding: 5px; font-weight: bold;")
                                    summary_label.setAlignment(Qt.AlignCenter)
                                    ocr_layout.addWidget(summary_label)
                                    
                                    # Display each line result
                                    for j, line_result in enumerate(line_results):
                                        if isinstance(line_result, dict):
                                            recognized_text = line_result.get('recognized_text', 'N/A')
                                            confidence = line_result.get('confidence_score', 0.0)
                                            line_index = line_result.get('line_index', j)
                                            
                                            ocr_text = f"  บรรทัด {line_index+1}: '{recognized_text}' (ความเชื่อมั่น: {confidence:.3f})"
                                            
                                            ocr_label = QLabel(ocr_text)
                                            ocr_label.setStyleSheet("color: #2c3e50; font-size: 10px; background-color: #ecf0f1; padding: 2px;")
                                            ocr_label.setWordWrap(True)
                                            ocr_layout.addWidget(ocr_label)
                                    
                            # Show total recognized text if available
                            if 'total_recognized_text' in ocr_results:
                                total_text = ocr_results['total_recognized_text']
                                if total_text:
                                    total_label = QLabel(f"📝 ข้อความรวม: '{total_text}'")
                                    total_label.setStyleSheet("color: #27ae60; font-size: 11px; background-color: #d5f4e6; padding: 5px; font-weight: bold;")
                                    total_label.setWordWrap(True)
                                    ocr_layout.addWidget(total_label)
                            
                            # Show retry information if available
                            if 'format_valid' in cap_result and 'score' in cap_result:
                                format_valid = cap_result['format_valid']
                                score = cap_result['score']
                                rotation_attempt = cap_result.get('rotation_attempt', 1)
                                
                                if format_valid:
                                    retry_label = QLabel(f"✅ Format Valid (Score: {score:.1f}, Attempt: {rotation_attempt}/5)")
                                    retry_label.setStyleSheet("color: #27ae60; font-size: 10px; background-color: #d5f4e6; padding: 3px; font-weight: bold;")
                                else:
                                    retry_label = QLabel(f"⚠️ Format Invalid (Score: {score:.1f}, Attempt: {rotation_attempt}/5)")
                                    retry_label.setStyleSheet("color: #e74c3c; font-size: 10px; background-color: #fadbd8; padding: 3px; font-weight: bold;")
                                
                                retry_label.setAlignment(Qt.AlignCenter)
                                ocr_layout.addWidget(retry_label)
                            else:
                                # Fallback for other dictionary formats
                                ocr_text = f"  ผลลัพธ์ OCR: {str(ocr_results)}"
                                ocr_label = QLabel(ocr_text)
                                ocr_label.setStyleSheet("color: #2c3e50; font-size: 10px; background-color: #ecf0f1; padding: 2px;")
                                ocr_label.setWordWrap(True)
                                ocr_layout.addWidget(ocr_label)
                        else:
                            # Handle list format (legacy)
                            for j, ocr_result in enumerate(ocr_results):
                                # Check if ocr_result is a dictionary
                                if isinstance(ocr_result, dict):
                                    ocr_text = f"  {j+1}. '{ocr_result.get('text', 'N/A')}' (ความเชื่อมั่น: {ocr_result.get('confidence', 0):.3f})"
                                else:
                                    # If it's a string or other type, display as is
                                    ocr_text = f"  {j+1}. '{str(ocr_result)}'"
                                
                                ocr_label = QLabel(ocr_text)
                                ocr_label.setStyleSheet("color: #2c3e50; font-size: 10px; background-color: #ecf0f1; padding: 2px;")
                                ocr_label.setWordWrap(True)
                                ocr_layout.addWidget(ocr_label)
                        
                        self.cap_results_layout.addWidget(ocr_container)
                else:
                    # Fallback: Display only cropped images (old behavior)
                    for i, crop in enumerate(result['cropped_images']):
                        crop_container = QWidget()
                        crop_container.setStyleSheet("border: 2px solid #27ae60; margin: 5px; padding: 5px; background-color: white;")
                        crop_layout = QVBoxLayout(crop_container)
                        
                        crop_title = QLabel(f"✂️ Step 3: ฝาที่ตรวจจับได้ {i+1} (Cropped Cap {i+1})")
                        crop_title.setStyleSheet("font-weight: bold; color: #27ae60; font-size: 12px;")
                        crop_title.setAlignment(Qt.AlignCenter)
                        crop_layout.addWidget(crop_title)
                        
                        # Display cropped image
                        if hasattr(crop, 'shape'):
                            # Resize for display
                            h, w = crop.shape[:2]
                            max_size = 200
                            if h > max_size or w > max_size:
                                scale = max_size / max(h, w)
                                new_w, new_h = int(w * scale), int(h * scale)
                                crop = cv2.resize(crop, (new_w, new_h))
                            
                            # Convert to RGB for Qt
                            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            h, w, c = rgb_crop.shape
                            bytes_per_line = c * w
                            
                            qimg = QtGui.QImage(rgb_crop.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                            pixmap = QtGui.QPixmap.fromImage(qimg)
                            
                            crop_label = QLabel()
                            crop_label.setPixmap(pixmap)
                            crop_label.setAlignment(Qt.AlignCenter)
                            crop_layout.addWidget(crop_label)
                        
                        self.cap_results_layout.addWidget(crop_container)
            else:
                # No caps detected, show text detection results
                if result.get('craft_rotated_image') is not None:
                    # Step 2a: CRAFT Rotated Image
                    craft_container = QWidget()
                    craft_container.setStyleSheet("border: 2px solid #8e44ad; margin: 5px; padding: 5px; background-color: white;")
                    craft_layout = QVBoxLayout(craft_container)
                    
                    craft_title = QLabel("🔄 Step 2a: ภาพที่ CRAFT หมุนแล้ว (CRAFT Rotated)")
                    craft_title.setStyleSheet("font-weight: bold; color: #8e44ad; font-size: 12px;")
                    craft_title.setAlignment(Qt.AlignCenter)
                    craft_layout.addWidget(craft_title)
                    
                    # Display CRAFT rotated image
                    craft_image = result['craft_rotated_image']
                    if hasattr(craft_image, 'shape'):
                        # Resize for display
                        h, w = craft_image.shape[:2]
                        max_size = 200
                        if h > max_size or w > max_size:
                            scale = max_size / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            craft_image = cv2.resize(craft_image, (new_w, new_h))
                        
                        # Convert to RGB for Qt
                        rgb_craft = cv2.cvtColor(craft_image, cv2.COLOR_BGR2RGB)
                        h, w, c = rgb_craft.shape
                        bytes_per_line = c * w
                        
                        qimg = QtGui.QImage(rgb_craft.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(qimg)
                        
                        craft_label = QLabel()
                        craft_label.setPixmap(pixmap)
                        craft_label.setAlignment(Qt.AlignCenter)
                        craft_layout.addWidget(craft_label)
                    
                    self.cap_results_layout.addWidget(craft_container)
                
                # Step 2b: AI Rotated Image
                if result.get('rotated_image') is not None:
                    ai_rotated_container = QWidget()
                    ai_rotated_container.setStyleSheet("border: 2px solid #e74c3c; margin: 5px; padding: 5px; background-color: white;")
                    ai_rotated_layout = QVBoxLayout(ai_rotated_container)
                    
                    ai_rotated_title = QLabel("🤖 Step 2b: ภาพที่ AI หมุนแล้ว (AI Rotated)")
                    ai_rotated_title.setStyleSheet("font-weight: bold; color: #e74c3c; font-size: 12px;")
                    ai_rotated_title.setAlignment(Qt.AlignCenter)
                    ai_rotated_layout.addWidget(ai_rotated_title)
                    
                    # Display AI rotated image
                    ai_rotated_image = result['rotated_image']
                    if hasattr(ai_rotated_image, 'shape'):
                        # Resize for display
                        h, w = ai_rotated_image.shape[:2]
                        max_size = 200
                        if h > max_size or w > max_size:
                            scale = max_size / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            ai_rotated_image = cv2.resize(ai_rotated_image, (new_w, new_h))
                        
                        # Convert to RGB for Qt
                        rgb_ai = cv2.cvtColor(ai_rotated_image, cv2.COLOR_BGR2RGB)
                        h, w, c = rgb_ai.shape
                        bytes_per_line = c * w
                        
                        qimg = QtGui.QImage(rgb_ai.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(qimg)
                        
                        ai_rotated_label = QLabel()
                        ai_rotated_label.setPixmap(pixmap)
                        ai_rotated_label.setAlignment(Qt.AlignCenter)
                        ai_rotated_layout.addWidget(ai_rotated_label)
                    
                    self.cap_results_layout.addWidget(ai_rotated_container)
                
                # Step 3: Cropped Image from AI Rotation
                if result.get('cropped_image') is not None:
                    cropped_container = QWidget()
                    cropped_container.setStyleSheet("border: 2px solid #f39c12; margin: 5px; padding: 5px; background-color: white;")
                    cropped_layout = QVBoxLayout(cropped_container)
                    
                    cropped_title = QLabel("✂️ Step 3: ภาพที่ครอปแล้ว (Cropped Image)")
                    cropped_title.setStyleSheet("font-weight: bold; color: #f39c12; font-size: 12px;")
                    cropped_title.setAlignment(Qt.AlignCenter)
                    cropped_layout.addWidget(cropped_title)
                    
                    # Display cropped image
                    cropped_image = result['cropped_image']
                    if hasattr(cropped_image, 'shape'):
                        # Resize for display
                        h, w = cropped_image.shape[:2]
                        max_size = 200
                        if h > max_size or w > max_size:
                            scale = max_size / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            cropped_image = cv2.resize(cropped_image, (new_w, new_h))
                        
                        # Convert to RGB for Qt
                        rgb_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                        h, w, c = rgb_cropped.shape
                        bytes_per_line = c * w
                        
                        qimg = QtGui.QImage(rgb_cropped.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(qimg)
                        
                        cropped_label = QLabel()
                        cropped_label.setPixmap(pixmap)
                        cropped_label.setAlignment(Qt.AlignCenter)
                        cropped_layout.addWidget(cropped_label)
                    
                    self.cap_results_layout.addWidget(cropped_container)
            
            # Update cap detection text with step-by-step process
            cap_text = f"""
=== ผลการตรวจจับฝาและข้อความ ===
📊 ภาพต้นฉบับ: {result.get('image_shape', 'N/A')}

🔄 ขั้นตอนการประมวลผล:
✅ Step 1: ตรวจจับฝา - เสร็จสิ้น
✅ Step 2: ตัดภาพ - เสร็จสิ้น
✅ Step 2a.5: ตรวจสอบรอยจาง - เสร็จสิ้น
            """
            
            if result.get('cropped_images'):
                cap_text += f"\n🔍 ฝาที่ตรวจจับได้: {len(result['cropped_images'])} ฝา"
                cap_text += f"\n✅ พบฝา {len(result['cropped_images'])} ฝา"
                
                # Check if we have full pipeline results for caps
                if result.get('cap_processing_results'):
                    cap_text += f"\n\n🔄 ประมวลผลฝาแต่ละใบผ่าน Pipeline เต็ม:"
                    for cap_result in result['cap_processing_results']:
                        cap_index = cap_result['cap_index'] + 1
                        cap_text += f"\n\n📋 ฝาที่ {cap_index}:"
                        
                        # Faded text detection (แสดงก่อน CRAFT rotation)
                        if cap_result.get('faded_text_result'):
                            faded_result = cap_result['faded_text_result']
                            status = faded_result.get('status', 'unknown')
                            num_chars = faded_result.get('num_chars', 0)
                            total_area = faded_result.get('total_area', 0)
                            
                            if status == 'faded':
                                cap_text += f"\n  ⚠️ Step 2a.5: ตรวจสอบรอยจาง - พบรอยจาง (ตัวอักษร: {num_chars}, พื้นที่: {total_area})"
                            elif status == 'normal':
                                cap_text += f"\n  ✅ Step 2a.5: ตรวจสอบรอยจาง - ปกติ (ตัวอักษร: {num_chars}, พื้นที่: {total_area})"
                            else:
                                cap_text += f"\n  ❓ Step 2a.5: ตรวจสอบรอยจาง - ไม่ทราบสถานะ (ตัวอักษร: {num_chars}, พื้นที่: {total_area})"
                        
                        # CRAFT rotation
                        if cap_result.get('craft_rotated_image') is not None:
                            cap_text += f"\n  ✅ Step 2a: CRAFT rotation - สำเร็จ"
                            
                            # AI rotation
                            if cap_result.get('ai_rotated_image') is not None:
                                cap_text += f"\n  ✅ Step 2b: AI rotation - สำเร็จ"
                                
                                # Line detection
                                if cap_result.get('line_detection_result') and 'error' not in cap_result.get('line_detection_result', {}):
                                    cap_text += f"\n  ✅ Step 3: Line detection - สำเร็จ"
                                    
                                    # OCR
                                    if cap_result.get('ocr_results'):
                                        cap_text += f"\n  ✅ Step 4: OCR - สำเร็จ"
                                        
                                        ocr_results = cap_result['ocr_results']
                                        if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                                            # Handle dictionary format from recognize_text_from_craft_lines
                                            line_results = ocr_results['line_results']
                                            total_lines = ocr_results.get('total_lines', 0)
                                            cap_text += f"\n  📝 ข้อความที่อ่านได้: {total_lines} บรรทัด"
                                            
                                            for j, line_result in enumerate(line_results):
                                                if isinstance(line_result, dict):
                                                    recognized_text = line_result.get('recognized_text', 'N/A')
                                                    confidence = line_result.get('confidence_score', 0.0)
                                                    line_index = line_result.get('line_index', j)
                                                    cap_text += f"\n    บรรทัด {line_index+1}: '{recognized_text}' (ความเชื่อมั่น: {confidence:.3f})"
                                            
                                            # Show total recognized text if available
                                            if 'total_recognized_text' in ocr_results:
                                                total_text = ocr_results['total_recognized_text']
                                                if total_text:
                                                    cap_text += f"\n  📝 ข้อความรวม: '{total_text}'"
                                        else:
                                            # Handle list format (legacy)
                                            cap_text += f"\n  📝 ข้อความที่อ่านได้: {len(ocr_results)} รายการ"
                                            for j, ocr_result in enumerate(ocr_results):
                                                # Check if ocr_result is a dictionary
                                                if isinstance(ocr_result, dict):
                                                    cap_text += f"\n    {j+1}. '{ocr_result.get('text', 'N/A')}' (ความเชื่อมั่น: {ocr_result.get('confidence', 0):.3f})"
                                                else:
                                                    # If it's a string or other type, display as is
                                                    cap_text += f"\n    {j+1}. '{str(ocr_result)}'"
                                        
                                        cap_text += f"\n  🎯 ฝาที่ {cap_index}: ประมวลผลเสร็จสมบูรณ์"
                                    else:
                                        cap_text += f"\n  ❌ Step 4: OCR - ไม่สำเร็จ"
                                        cap_text += f"\n  ⚠️ ฝาที่ {cap_index}: ประมวลผลไม่สมบูรณ์"
                                else:
                                    cap_text += f"\n  ❌ Step 3: Line detection - ไม่สำเร็จ"
                                    cap_text += f"\n  ⚠️ ฝาที่ {cap_index}: ประมวลผลไม่สมบูรณ์"
                            else:
                                cap_text += f"\n  ❌ Step 2b: AI rotation - ไม่สำเร็จ"
                                cap_text += f"\n  ⚠️ ฝาที่ {cap_index}: ประมวลผลไม่สมบูรณ์"
                        else:
                            cap_text += f"\n  ❌ Step 2a: CRAFT rotation - ไม่สำเร็จ"
                            cap_text += f"\n  ⚠️ ฝาที่ {cap_index}: ประมวลผลไม่สมบูรณ์"
                    
                    cap_text += f"\n\n🎯 ผลลัพธ์: ระบบประมวลผลฝาเสร็จสิ้น"
                else:
                    cap_text += f"\n\n🎯 ผลลัพธ์: ระบบตรวจจับฝาเสร็จสิ้น (ไม่มีการประมวลผลเพิ่มเติม)"
            else:
                cap_text += f"\n⚠️ ไม่พบฝา เริ่มประมวลผลข้อความ..."
                
                # Step 2a: CRAFT
                if result.get('craft_rotated_image') is not None:
                    cap_text += f"\n✅ Step 2a: ตรวจจับข้อความด้วย CRAFT - เสร็จสิ้น"
                    
                    # Step 2b: AI Rotation
                    if result.get('rotated_image') is not None:
                        cap_text += f"\n✅ Step 2b: ประมวลผลด้วย AI rotation - เสร็จสิ้น"
                        
                        # Step 3: Line Detection
                        if result.get('line_detection_result') and 'error' not in result.get('line_detection_result', {}):
                            cap_text += f"\n✅ Step 3: ตรวจจับบรรทัดข้อความ - เสร็จสิ้น"
                            
                            # Step 4: OCR
                            if result.get('final_ocr_results'):
                                cap_text += f"\n✅ Step 4: อ่านข้อความด้วย OCR - เสร็จสิ้น"
                                
                                ocr_results = result['final_ocr_results']
                                if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                                    # Handle dictionary format from recognize_text_from_craft_lines
                                    line_results = ocr_results['line_results']
                                    total_lines = ocr_results.get('total_lines', 0)
                                    cap_text += f"\n📝 ข้อความที่อ่านได้: {total_lines} บรรทัด"
                                    
                                    for i, line_result in enumerate(line_results):
                                        if isinstance(line_result, dict):
                                            recognized_text = line_result.get('recognized_text', 'N/A')
                                            confidence = line_result.get('confidence_score', 0.0)
                                            line_index = line_result.get('line_index', i)
                                            cap_text += f"\n  บรรทัด {line_index+1}: '{recognized_text}' (ความเชื่อมั่น: {confidence:.3f})"
                                    
                                    # Show total recognized text if available
                                    if 'total_recognized_text' in ocr_results:
                                        total_text = ocr_results['total_recognized_text']
                                        if total_text:
                                            cap_text += f"\n📝 ข้อความรวม: '{total_text}'"
                                else:
                                    # Handle list format (legacy)
                                    cap_text += f"\n📝 ข้อความที่อ่านได้: {len(ocr_results)} รายการ"
                                    for i, ocr_result in enumerate(ocr_results):
                                        # Check if ocr_result is a dictionary
                                        if isinstance(ocr_result, dict):
                                            cap_text += f"\n  {i+1}. '{ocr_result.get('text', 'N/A')}' (ความเชื่อมั่น: {ocr_result.get('confidence', 0):.3f})"
                                        else:
                                            # If it's a string or other type, display as is
                                            cap_text += f"\n  {i+1}. '{str(ocr_result)}'"
                                
                                # แสดงผลลัพธ์ OCR ที่อ่านได้สำหรับการตรวจสอบ
                                cap_text += f"\n\n🔍 ผลลัพธ์ OCR สำหรับการตรวจสอบ:"
                                
                                # ใช้ข้อมูลจาก combined_result (retry mechanism)
                                combined_result = result.get('combined_result', {})
                                
                                # แสดงมุมการหมุนที่ใช้
                                if 'ai_rotation_angle' in combined_result:
                                    rotation_angle = combined_result['ai_rotation_angle']
                                    cap_text += f"\n🔄 มุมการหมุนที่ใช้: {rotation_angle}°"
                                
                                # แสดงคะแนนการหมุน
                                if 'score' in combined_result:
                                    score = combined_result['score']
                                    cap_text += f"\n📊 คะแนนการหมุน: {score:.1f}"
                                
                                # แสดงจำนวนครั้งที่หมุน
                                if 'rotation_attempt' in combined_result:
                                    attempt = combined_result['rotation_attempt']
                                    cap_text += f"\n🔄 ครั้งที่หมุน: {attempt}/5"
                                
                                # แสดงข้อมูลการหมุนเพิ่มเติม
                                if 'format_valid' in combined_result:
                                    format_valid = combined_result['format_valid']
                                    cap_text += f"\n✅ ฟอร์มถูกต้อง: {'ใช่' if format_valid else 'ไม่ใช่'}"
                                
                                # แสดงข้อมูลการใช้ภาพจาก CRAFT
                                if 'used_craft_image' in combined_result:
                                    cap_text += f"\n🔄 ใช้ภาพจาก: CRAFT (หมุน {combined_result['ai_rotation_angle']}°)"
                                else:
                                    cap_text += f"\n🔄 ใช้ภาพจาก: AI Rotation"
                                
                                # ตรวจสอบรูปแบบ MFG บรรทัดแรก
                                if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                                    line_results = ocr_results['line_results']
                                    if line_results:
                                        first_line = line_results[0]
                                        first_text = first_line.get('recognized_text', '').strip()
                                        
                                        # ตรวจสอบว่าบรรทัดแรกมี MFG หรือไม่
                                        if any(prefix in first_text.upper() for prefix in ['MFG', 'BBF']):
                                            cap_text += f"\n✅ บรรทัดแรกมี MFG/BBF: '{first_text}'"
                                        else:
                                            cap_text += f"\n❌ บรรทัดแรกไม่มี MFG/BBF: '{first_text}'"
                                            cap_text += f"\n🔄 ระบบจะหมุนภาพ 5 มุมเพื่อหาฟอร์ม MFG ที่ถูกต้อง"
                                
                                if self.current_bottle_type in ["M100", "M110", "M120"]:
                                    is_cap_valid = self.validate_cap_result(result)
                                    if is_cap_valid:
                                        cap_text += f"\n✅ ฝาผ่าน - ตรงกับฟอร์มที่คาดหวัง"
                                    else:
                                        cap_text += f"\n❌ ฝาไม่ผ่าน - ไม่ตรงกับฟอร์มที่คาดหวัง"
                                else:
                                    cap_text += f"\nℹ️ ไม่ต้องตรวจสอบฟอร์ม (ไม่ใช่ M100/M110/M120)"
                                
                                cap_text += f"\n\n🎯 ผลลัพธ์: ระบบประมวลผลข้อความเสร็จสิ้น"
                            else:
                                cap_text += f"\n❌ Step 4: OCR - ไม่สำเร็จ"
                                cap_text += f"\n\n⚠️ ผลลัพธ์: ระบบประมวลผลข้อความไม่สมบูรณ์"
                        else:
                            cap_text += f"\n❌ Step 3: ตรวจจับบรรทัดข้อความ - ไม่สำเร็จ"
                            cap_text += f"\n\n⚠️ ผลลัพธ์: ระบบประมวลผลข้อความไม่สมบูรณ์"
                    else:
                        cap_text += f"\n❌ Step 2b: AI rotation - ไม่สำเร็จ"
                        cap_text += f"\n\n⚠️ ผลลัพธ์: ระบบประมวลผลข้อความไม่สมบูรณ์"
                else:
                    cap_text += f"\n❌ Step 2a: CRAFT - ไม่สำเร็จ"
                    cap_text += f"\n\n⚠️ ผลลัพธ์: ระบบประมวลผลข้อความไม่สมบูรณ์"
            
            self.cap_detection_text.setText(cap_text)
            
        except Exception as e:
            print(f"❌ Error displaying cap detection results: {e}")
            self.cap_detection_text.setText(f"ข้อผิดพลาดในการแสดงผล: {str(e)}")
    
    def display_cap_processing_ui(self):
        """Display cap processing UI when starting cap detection"""
        try:
            # Clear previous results
            for i in reversed(range(self.cap_results_layout.count())):
                self.cap_results_layout.itemAt(i).widget().setParent(None)
            
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
            
            self.cap_results_layout.addWidget(processing_container)
            
            # Update cap detection text
            if hasattr(self, 'cap_detection_text') and self.cap_detection_text:
                self.cap_detection_text.setText("🔄 กำลังประมวลผลฝา... กรุณารอสักครู่")
                self.cap_detection_text.setStyleSheet("""
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
            # Don't clear previous results - just add completion status
            # for i in reversed(range(self.cap_results_layout.count())):
            #     self.cap_results_layout.itemAt(i).widget().setParent(None)
            
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
            if self.current_bottle_type in ["M100", "M110", "M120"]:
                status_info = QLabel(f"📋 ฝาผ่านการตรวจสอบ - กำลังส่งสัญญาณ {self.current_bottle_type}")
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
            
            self.cap_results_layout.addWidget(complete_container)
            
            # Update cap detection text
            if hasattr(self, 'cap_detection_text') and self.cap_detection_text:
                if self.current_bottle_type in ["M100", "M110", "M120"]:
                    self.cap_detection_text.setText(f"✅ การประมวลผลฝาเสร็จสิ้น - กำลังส่งสัญญาณ {self.current_bottle_type}")
                else:
                    self.cap_detection_text.setText("✅ การประมวลผลฝาเสร็จสิ้น")
                self.cap_detection_text.setStyleSheet("""
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
    
    def on_taste_mode_changed(self):
        """Handle taste mode selection change"""
        try:
            # Reset all taste mode signals first
            if hasattr(self, 'modbus_thread') and self.modbus_thread:
                self.modbus_thread.reset_m720()
                self.modbus_thread.reset_m721()
                self.modbus_thread.reset_m722()
                
                # Reset all taste combination signals
                self.modbus_thread.reset_m730()
                self.modbus_thread.reset_m731()
                self.modbus_thread.reset_m732()
                self.modbus_thread.reset_m733()
                self.modbus_thread.reset_m734()
                self.modbus_thread.reset_m735()
                
                # Reset all 1-taste signals
                self.modbus_thread.reset_m740()
                self.modbus_thread.reset_m741()
                self.modbus_thread.reset_m742()
                
                # Reset lamp status
                self.update_coil_lamp("m720", False)
                self.update_coil_lamp("m721", False)
                self.update_coil_lamp("m722", False)
                self.update_coil_lamp("m730", False)
                self.update_coil_lamp("m731", False)
                self.update_coil_lamp("m732", False)
                self.update_coil_lamp("m733", False)
                self.update_coil_lamp("m734", False)
                self.update_coil_lamp("m735", False)
                self.update_coil_lamp("m740", False)
                self.update_coil_lamp("m741", False)
                self.update_coil_lamp("m742", False)
                
                print("🔄 TASTE MODE: Reset all taste mode and combination signals")
            
            # Get the checked mode
            if self.taste_mode_1.isChecked():
                self.current_taste_mode = 1
                self.taste_mode_2.setChecked(False)
                self.taste_mode_3.setChecked(False)
                self.show_taste_selection()
                self.taste_mode_status.setText("โหมดปัจจุบัน: 1 รสชาติ - กรุณาเลือกรสชาติ")
                self.taste_mode_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px; font-weight: bold;")
                
                # Send M722 signal
                if hasattr(self, 'modbus_thread') and self.modbus_thread:
                    success = self.modbus_thread.on_m722()
                    if success:
                        print("🏷️ ON M722 (โหมด 1 รสชาติ)")
                        self.update_coil_lamp("m722", True)
                    else:
                        print("❌ ไม่สามารถ ON M722 ได้")
                
            elif self.taste_mode_2.isChecked():
                self.current_taste_mode = 2
                self.taste_mode_1.setChecked(False)
                self.taste_mode_3.setChecked(False)
                self.show_taste_selection()
                self.taste_mode_status.setText("โหมดปัจจุบัน: 2 รสชาติ - กรุณาเลือกรสชาติ")
                self.taste_mode_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px; font-weight: bold;")
                
                # Send M721 signal
                if hasattr(self, 'modbus_thread') and self.modbus_thread:
                    success = self.modbus_thread.on_m721()
                    if success:
                        print("🏷️ ON M721 (โหมด 2 รสชาติ)")
                        self.update_coil_lamp("m721", True)
                    else:
                        print("❌ ไม่สามารถ ON M721 ได้")
                
            elif self.taste_mode_3.isChecked():
                self.current_taste_mode = 3
                self.taste_mode_1.setChecked(False)
                self.taste_mode_2.setChecked(False)
                self.hide_taste_selection()
                self.selected_tastes = ["M100", "M110", "M120"]  # All tastes
                self.taste_mode_status.setText("โหมดปัจจุบัน: 3 รสชาติ (M100, M110, M120)")
                self.taste_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px; font-weight: bold;")
                
                # Send M720 signal
                if hasattr(self, 'modbus_thread') and self.modbus_thread:
                    success = self.modbus_thread.on_m720()
                    if success:
                        print("🏷️ ON M720 (โหมด 3 รสชาติ)")
                        self.update_coil_lamp("m720", True)
                    else:
                        print("❌ ไม่สามารถ ON M720 ได้")
            
            print(f"🔄 TASTE MODE: Changed to {self.current_taste_mode}-taste mode")
            print(f"🔄 TASTE MODE: Selected tastes: {self.selected_tastes}")
            
        except Exception as e:
            print(f"❌ Error in on_taste_mode_changed: {e}")
    
    def show_taste_selection(self):
        """Show taste selection checkboxes"""
        try:
            # Show all taste checkboxes
            for i in range(self.taste_selection_layout.count()):
                widget = self.taste_selection_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
            
            # Reset all taste selections
            self.taste_m100.setChecked(False)
            self.taste_m110.setChecked(False)
            self.taste_m120.setChecked(False)
            self.selected_tastes = []
            
            print("🔄 TASTE SELECTION: Showing taste selection checkboxes")
            
        except Exception as e:
            print(f"❌ Error in show_taste_selection: {e}")
    
    def hide_taste_selection(self):
        """Hide taste selection checkboxes"""
        try:
            # Hide taste selection labels and checkboxes
            for i in range(self.taste_selection_layout.count()):
                widget = self.taste_selection_layout.itemAt(i).widget()
                if widget and isinstance(widget, QCheckBox):
                    widget.setVisible(False)
                elif widget and isinstance(widget, QLabel) and "เลือกรสชาติ" in widget.text():
                    widget.setVisible(False)
            
            print("🔄 TASTE SELECTION: Hiding taste selection checkboxes")
            
        except Exception as e:
            print(f"❌ Error in hide_taste_selection: {e}")
    
    def update_selected_tastes(self):
        """Update selected tastes based on checkboxes"""
        try:
            self.selected_tastes = []
            
            if self.taste_m100.isChecked():
                self.selected_tastes.append("M100")
            if self.taste_m110.isChecked():
                self.selected_tastes.append("M110")
            if self.taste_m120.isChecked():
                self.selected_tastes.append("M120")
            
            # Validate selection count
            if self.current_taste_mode == 1 and len(self.selected_tastes) != 1:
                if len(self.selected_tastes) > 1:
                    QMessageBox.warning(self, "คำเตือน", "กรุณาเลือกรสชาติเพียง 1 รสชาติ")
                    # Uncheck the last selected
                    if self.taste_m100.isChecked() and len(self.selected_tastes) > 1:
                        self.taste_m100.setChecked(False)
                    elif self.taste_m110.isChecked() and len(self.selected_tastes) > 1:
                        self.taste_m110.setChecked(False)
                    elif self.taste_m120.isChecked() and len(self.selected_tastes) > 1:
                        self.taste_m120.setChecked(False)
                    self.update_selected_tastes()  # Recursive call
                    return
            elif self.current_taste_mode == 2 and len(self.selected_tastes) != 2:
                if len(self.selected_tastes) > 2:
                    QMessageBox.warning(self, "คำเตือน", "กรุณาเลือกรสชาติเพียง 2 รสชาติ")
                    # Uncheck the last selected
                    if self.taste_m100.isChecked() and len(self.selected_tastes) > 2:
                        self.taste_m100.setChecked(False)
                    elif self.taste_m110.isChecked() and len(self.selected_tastes) > 2:
                        self.taste_m110.setChecked(False)
                    elif self.taste_m120.isChecked() and len(self.selected_tastes) > 2:
                        self.taste_m120.setChecked(False)
                    self.update_selected_tastes()  # Recursive call
                    return
            
            # Update status and send signals
            if self.current_taste_mode == 1:
                self.taste_mode_status.setText(f"โหมดปัจจุบัน: 1 รสชาติ ({', '.join(self.selected_tastes)})")
                
                # Send appropriate signal for 1-taste mode
                if len(self.selected_tastes) == 1:
                    self.send_one_taste_signal()
            elif self.current_taste_mode == 2:
                self.taste_mode_status.setText(f"โหมดปัจจุบัน: 2 รสชาติ ({', '.join(self.selected_tastes)})")
                
                # Send appropriate signal for 2-taste mode
                if len(self.selected_tastes) == 2:
                    self.send_two_taste_signal()
            
            print(f"🔄 TASTE SELECTION: Updated selected tastes: {self.selected_tastes}")
            
        except Exception as e:
            print(f"❌ Error in update_selected_tastes: {e}")
    
    def send_two_taste_signal(self):
        """Send appropriate signal for 2-taste mode selection"""
        try:
            if not hasattr(self, 'modbus_thread') or not self.modbus_thread:
                print("❌ TASTE SIGNAL: Modbus thread not available")
                return
            
            # Reset all 2-taste signals first
            self.modbus_thread.reset_m730()
            self.modbus_thread.reset_m731()
            self.modbus_thread.reset_m732()
            self.modbus_thread.reset_m733()
            self.modbus_thread.reset_m734()
            self.modbus_thread.reset_m735()
            
            # Reset lamp status
            self.update_coil_lamp("m730", False)
            self.update_coil_lamp("m731", False)
            self.update_coil_lamp("m732", False)
            self.update_coil_lamp("m733", False)
            self.update_coil_lamp("m734", False)
            self.update_coil_lamp("m735", False)
            
            print("🔄 TASTE SIGNAL: Reset all 2-taste signals and lamps")
            
            # Determine which signal to send based on selected tastes
            if "M100" in self.selected_tastes and "M110" in self.selected_tastes:
                # ขวดดั้งเดิม + ขวดน้ำตาล 2%
                success = self.modbus_thread.on_m730()
                if success:
                    print("🏷️ ON M730 (ขวดดั้งเดิม + ขวดน้ำตาล 2%)")
                    self.update_coil_lamp("m730", True)
                else:
                    print("❌ ไม่สามารถ ON M730 ได้")
                    
            elif "M100" in self.selected_tastes and "M120" in self.selected_tastes:
                # ขวดดั้งเดิม + ขวดผสมแมงลัก
                success = self.modbus_thread.on_m731()
                if success:
                    print("🏷️ ON M731 (ขวดดั้งเดิม + ขวดผสมแมงลัก)")
                    self.update_coil_lamp("m731", True)
                else:
                    print("❌ ไม่สามารถ ON M731 ได้")
                    
            elif "M110" in self.selected_tastes and "M120" in self.selected_tastes:
                # ขวดน้ำตาล 2% + ขวดผสมแมงลัก
                success = self.modbus_thread.on_m732()
                if success:
                    print("🏷️ ON M732 (ขวดน้ำตาล 2% + ขวดผสมแมงลัก)")
                    self.update_coil_lamp("m732", True)
                else:
                    print("❌ ไม่สามารถ ON M732 ได้")
                    
            elif "M110" in self.selected_tastes and "M100" in self.selected_tastes:
                # ขวดน้ำตาล 2% + ขวดดั้งเดิม
                success = self.modbus_thread.on_m733()
                if success:
                    print("🏷️ ON M733 (ขวดน้ำตาล 2% + ขวดดั้งเดิม)")
                    self.update_coil_lamp("m733", True)
                else:
                    print("❌ ไม่สามารถ ON M733 ได้")
                    
            elif "M120" in self.selected_tastes and "M110" in self.selected_tastes:
                # ขวดผสมแมงลัก + ขวดน้ำตาล 2%
                success = self.modbus_thread.on_m734()
                if success:
                    print("🏷️ ON M734 (ขวดผสมแมงลัก + ขวดน้ำตาล 2%)")
                    self.update_coil_lamp("m734", True)
                else:
                    print("❌ ไม่สามารถ ON M734 ได้")
                    
            elif "M120" in self.selected_tastes and "M100" in self.selected_tastes:
                # ขวดผสมแมงลัก + ขวดดั้งเดิม
                success = self.modbus_thread.on_m735()
                if success:
                    print("🏷️ ON M735 (ขวดผสมแมงลัก + ขวดดั้งเดิม)")
                    self.update_coil_lamp("m735", True)
                else:
                    print("❌ ไม่สามารถ ON M735 ได้")
            
        except Exception as e:
            print(f"❌ Error in send_two_taste_signal: {e}")
    
    def send_one_taste_signal(self):
        """Send appropriate signal for 1-taste mode selection"""
        try:
            if not hasattr(self, 'modbus_thread') or not self.modbus_thread:
                print("❌ TASTE SIGNAL: Modbus thread not available")
                return
            
            # Reset all 1-taste signals first
            self.modbus_thread.reset_m740()
            self.modbus_thread.reset_m741()
            self.modbus_thread.reset_m742()
            
            # Reset lamp status
            self.update_coil_lamp("m740", False)
            self.update_coil_lamp("m741", False)
            self.update_coil_lamp("m742", False)
            
            print("🔄 TASTE SIGNAL: Reset all 1-taste signals and lamps")
            
            # Determine which signal to send based on selected taste
            if "M100" in self.selected_tastes:
                # ขวดดั้งเดิม
                success = self.modbus_thread.on_m740()
                if success:
                    print("🏷️ ON M740 (ขวดดั้งเดิม)")
                    self.update_coil_lamp("m740", True)
                else:
                    print("❌ ไม่สามารถ ON M740 ได้")
                    
            elif "M110" in self.selected_tastes:
                # ขวดน้ำตาล 2%
                success = self.modbus_thread.on_m741()
                if success:
                    print("🏷️ ON M741 (ขวดน้ำตาล 2%)")
                    self.update_coil_lamp("m741", True)
                else:
                    print("❌ ไม่สามารถ ON M741 ได้")
                    
            elif "M120" in self.selected_tastes:
                # ขวดผสมแมงลัก
                success = self.modbus_thread.on_m742()
                if success:
                    print("🏷️ ON M742 (ขวดผสมแมงลัก)")
                    self.update_coil_lamp("m742", True)
                else:
                    print("❌ ไม่สามารถ ON M742 ได้")
            
        except Exception as e:
            print(f"❌ Error in send_one_taste_signal: {e}")
    
    def on_expiry_mode_changed(self):
        """Handle expiry mode selection change"""
        try:
            # Get the checked mode
            if self.expiry_mode_normal.isChecked():
                self.current_expiry_mode = "normal"
                self.expiry_mode_filter.setChecked(False)
                self.hide_expiry_filter_controls()
                self.expiry_mode_status.setText("โหมดปัจจุบัน: ปกติ (ไม่คัดกรองวันหมดอายุ)")
                self.expiry_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px; font-weight: bold;")
            elif self.expiry_mode_filter.isChecked():
                self.current_expiry_mode = "filter"
                self.expiry_mode_normal.setChecked(False)
                self.show_expiry_filter_controls()
                self.update_expiry_mode_status()
            
            print(f"🔄 EXPIRY MODE: Changed to {self.current_expiry_mode} mode")
            
        except Exception as e:
            print(f"❌ Error in on_expiry_mode_changed: {e}")
    
    def on_expiry_filter_type_changed(self):
        """Handle expiry filter type selection change"""
        try:
            # Get the checked filter type
            if self.expiry_filter_range.isChecked():
                self.expiry_filter_type = "range"
                self.expiry_filter_specific.setChecked(False)
                self.show_range_date_selection()
                self.hide_specific_date_selection()
            elif self.expiry_filter_specific.isChecked():
                self.expiry_filter_type = "specific"
                self.expiry_filter_range.setChecked(False)
                self.hide_range_date_selection()
                self.show_specific_date_selection()
            
            self.update_expiry_mode_status()
            print(f"🔄 EXPIRY FILTER TYPE: Changed to {self.expiry_filter_type}")
            
        except Exception as e:
            print(f"❌ Error in on_expiry_filter_type_changed: {e}")
    
    def show_expiry_filter_controls(self):
        """Show expiry filter controls"""
        try:
            # Show filter type selection
            for i in range(self.expiry_filter_type_layout.count()):
                widget = self.expiry_filter_type_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
            
            # Show date selection based on current filter type
            if self.expiry_filter_type == "range":
                self.show_range_date_selection()
                self.hide_specific_date_selection()
            else:
                self.hide_range_date_selection()
                self.show_specific_date_selection()
            
            print("🔄 EXPIRY FILTER: Showing expiry filter controls")
            
        except Exception as e:
            print(f"❌ Error in show_expiry_filter_controls: {e}")
    
    def hide_expiry_filter_controls(self):
        """Hide expiry filter controls"""
        try:
            # Hide filter type selection
            for i in range(self.expiry_filter_type_layout.count()):
                widget = self.expiry_filter_type_layout.itemAt(i).widget()
                if widget and isinstance(widget, QCheckBox):
                    widget.setVisible(False)
                elif widget and isinstance(widget, QLabel) and "ประเภทการกรอง" in widget.text():
                    widget.setVisible(False)
            
            # Hide all date selections
            self.hide_range_date_selection()
            self.hide_specific_date_selection()
            
            print("🔄 EXPIRY FILTER: Hiding expiry filter controls")
            
        except Exception as e:
            print(f"❌ Error in hide_expiry_filter_controls: {e}")
    
    def show_range_date_selection(self):
        """Show range date selection controls"""
        try:
            for i in range(self.range_date_layout.count()):
                widget = self.range_date_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
            
            print("🔄 EXPIRY FILTER: Showing range date selection")
            
        except Exception as e:
            print(f"❌ Error in show_range_date_selection: {e}")
    
    def hide_range_date_selection(self):
        """Hide range date selection controls"""
        try:
            for i in range(self.range_date_layout.count()):
                widget = self.range_date_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)
            
            print("🔄 EXPIRY FILTER: Hiding range date selection")
            
        except Exception as e:
            print(f"❌ Error in hide_range_date_selection: {e}")
    
    def show_specific_date_selection(self):
        """Show specific date selection controls"""
        try:
            for i in range(self.specific_date_layout.count()):
                widget = self.specific_date_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
            
            print("🔄 EXPIRY FILTER: Showing specific date selection")
            
        except Exception as e:
            print(f"❌ Error in show_specific_date_selection: {e}")
    
    def hide_specific_date_selection(self):
        """Hide specific date selection controls"""
        try:
            for i in range(self.specific_date_layout.count()):
                widget = self.specific_date_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)
            
            print("🔄 EXPIRY FILTER: Hiding specific date selection")
            
        except Exception as e:
            print(f"❌ Error in hide_specific_date_selection: {e}")
    
    def update_expiry_mode_status(self):
        """Update expiry mode status label"""
        try:
            if self.current_expiry_mode == "normal":
                self.expiry_mode_status.setText("โหมดปัจจุบัน: ปกติ (ไม่คัดกรองวันหมดอายุ)")
                self.expiry_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px; font-weight: bold;")
            elif self.current_expiry_mode == "filter":
                if self.expiry_filter_type == "range":
                    start_date = self.expiry_start_date_edit.date().toString("dd/MM/yyyy")
                    end_date = self.expiry_end_date_edit.date().toString("dd/MM/yyyy")
                    self.expiry_mode_status.setText(f"โหมดปัจจุบัน: คัดกรองช่วงวันที่ ({start_date} - {end_date})")
                else:  # specific
                    specific_date = self.expiry_specific_date_edit.date().toString("dd/MM/yyyy")
                    self.expiry_mode_status.setText(f"โหมดปัจจุบัน: คัดกรองวันที่เฉพาะ ({specific_date})")
                self.expiry_mode_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px; font-weight: bold;")
            
        except Exception as e:
            print(f"❌ Error in update_expiry_mode_status: {e}")
    
    def get_expiry_filter_dates(self):
        """Get expiry filter dates for validation"""
        try:
            if self.current_expiry_mode == "normal":
                return None, None, None
            
            if self.expiry_filter_type == "range":
                start_date = self.expiry_start_date_edit.date().toPyDate()
                end_date = self.expiry_end_date_edit.date().toPyDate()
                return start_date, end_date, None
            else:  # specific
                specific_date = self.expiry_specific_date_edit.date().toPyDate()
                return None, None, specific_date
                
        except Exception as e:
            print(f"❌ Error in get_expiry_filter_dates: {e}")
            return None, None, None
    
    def select_ocr_test_image(self):
        """Select image for OCR testing"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "เลือกภาพสำหรับทดสอบ OCR", 
                "", 
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
            )
            
            if file_path:
                print(f"🔤 OCR TEST: Selected image: {file_path}")
                
                # Load and display image
                image = cv2.imread(file_path)
                if image is not None:
                    self.ocr_test_image = image
                    self.ocr_test_image_path = file_path
                    
                    # Display image
                    self.display_ocr_test_image(image)
                    
                    # Enable process buttons
                    self.btn_process_ocr.setEnabled(True)
                    self.btn_process_ocr_enhanced.setEnabled(True)
                    
                    # Update status
                    self.ocr_status_label.setText(f'✅ เลือกภาพแล้ว: {os.path.basename(file_path)}')
                    self.ocr_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
                    
                    # Clear previous results
                    self.ocr_results_text.clear()
                    
                    print(f"🔤 OCR TEST: Image loaded successfully, shape: {image.shape}")
                else:
                    QMessageBox.warning(self, "ข้อผิดพลาด", "ไม่สามารถโหลดภาพได้")
                    print(f"❌ OCR TEST: Failed to load image: {file_path}")
            else:
                print("🔤 OCR TEST: No image selected")
                
        except Exception as e:
            print(f"❌ Error in select_ocr_test_image: {e}")
            QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเลือกภาพ: {str(e)}")
    
    def display_ocr_test_image(self, image):
        """Display image in OCR test tab"""
        try:
            # Resize image for display
            h, w = image.shape[:2]
            max_size = 400
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                display_image = cv2.resize(image, (new_w, new_h))
            else:
                display_image = image
            
            # Convert to RGB for Qt
            rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_image.shape
            bytes_per_line = c * w
            
            qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            
            self.ocr_image_label.setPixmap(pixmap)
            self.ocr_image_label.setAlignment(Qt.AlignCenter)
            
        except Exception as e:
            print(f"❌ Error in display_ocr_test_image: {e}")
    
    def process_ocr_test(self):
        """Process OCR on selected image"""
        if self.ocr_test_image is None:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาเลือกภาพก่อน")
            return
        
        try:
            print("🔤 OCR TEST: Starting OCR processing...")
            
            # Update UI
            self.btn_process_ocr.setEnabled(False)
            self.ocr_progress_bar.setVisible(True)
            self.ocr_progress_bar.setValue(0)
            self.ocr_status_label.setText('🔄 กำลังประมวลผล OCR...')
            self.ocr_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            
            # Clear previous results
            self.ocr_results_text.clear()
            
            # Process OCR
            self.ocr_progress_bar.setValue(20)
            QtWidgets.QApplication.processEvents()
            
            # Use DeepOCR for text recognition
            if self.ocr_model is not None:
                print("🔤 OCR TEST: Using DeepOCR for text recognition...")
                
                self.ocr_progress_bar.setValue(40)
                QtWidgets.QApplication.processEvents()
                
                # Preprocess image for better number recognition
                processed_image = self.preprocess_image_for_ocr(self.ocr_test_image)
                
                # Perform OCR using DeepOCR
                ocr_result = self.ocr_model.recognize_text_from_image_array(
                    processed_image, 
                    self.ocr_test_image_path
                )
                
                self.ocr_progress_bar.setValue(80)
                QtWidgets.QApplication.processEvents()
                
                # Process results
                if 'error' not in ocr_result and ocr_result.get('recognized_text'):
                    recognized_text = ocr_result['recognized_text']
                    confidence_score = ocr_result.get('confidence_score', 0.0)
                    
                    print(f"🔤 OCR TEST: OCR completed successfully")
                    print(f"🔤 OCR TEST: Recognized text: '{recognized_text}'")
                    print(f"🔤 OCR TEST: Confidence: {confidence_score:.3f}")
                    
                    # Format results
                    result_text = f"🔤 ผลลัพธ์ OCR (DeepOCR)\n"
                    result_text += f"📊 สถานะ: สำเร็จ\n"
                    result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                    result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n\n"
                    
                    result_text += "📝 ข้อความที่อ่านได้:\n"
                    result_text += "=" * 50 + "\n"
                    result_text += f"'{recognized_text}'\n\n"
                    
                    result_text += "📊 สถิติ:\n"
                    result_text += "=" * 50 + "\n"
                    result_text += f"ความเชื่อมั่น: {confidence_score:.3f}\n"
                    result_text += f"จำนวนตัวอักษร: {len(recognized_text)}\n"
                    result_text += f"จำนวนคำ: {len(recognized_text.split())}\n"
                    
                    # Show additional info if available
                    if 'processing_time' in ocr_result:
                        result_text += f"เวลาประมวลผล: {ocr_result['processing_time']:.2f} วินาที\n"
                    
                    if 'model_info' in ocr_result:
                        result_text += f"โมเดล: {ocr_result['model_info']}\n"
                    
                else:
                    error_msg = ocr_result.get('error', 'ไม่สามารถอ่านข้อความได้')
                    result_text = f"❌ ไม่พบข้อความในภาพ\n\n"
                    result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                    result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n"
                    result_text += f"❌ ข้อผิดพลาด: {error_msg}\n"
                    print(f"🔤 OCR TEST: OCR failed - {error_msg}")
                
                # Display results
                self.ocr_results_text.setText(result_text)
                
            else:
                result_text = "❌ ข้อผิดพลาด: DeepOCR ไม่พร้อมใช้งาน\n\n"
                result_text += "กรุณาตรวจสอบการติดตั้ง DeepOCR model"
                self.ocr_results_text.setText(result_text)
                print("❌ OCR TEST: DeepOCR not available")
            
            self.ocr_progress_bar.setValue(100)
            QtWidgets.QApplication.processEvents()
            
            # Update UI
            self.btn_process_ocr.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('✅ ประมวลผล OCR เสร็จสิ้น')
            self.ocr_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            
            print("🔤 OCR TEST: OCR processing completed")
            
        except Exception as e:
            print(f"❌ Error in process_ocr_test: {e}")
            
            # Reset UI
            self.btn_process_ocr.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('❌ เกิดข้อผิดพลาดในการประมวลผล OCR')
            self.ocr_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            
            # Show error in results
            error_text = f"❌ ข้อผิดพลาดในการประมวลผล OCR\n\n"
            error_text += f"รายละเอียด: {str(e)}\n\n"
            error_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path) if self.ocr_test_image_path else 'ไม่ระบุ'}\n"
            self.ocr_results_text.setText(error_text)
    
    def process_ocr_test_enhanced(self):
        """
        ประมวลผล OCR แบบปรับปรุงสำหรับการอ่านเลข
        """
        if self.ocr_test_image is None:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาเลือกภาพก่อน")
            return
        
        try:
            print("🔤 OCR TEST ENHANCED: Starting enhanced OCR processing...")
            
            # Update UI
            self.btn_process_ocr.setEnabled(False)
            self.btn_process_ocr_enhanced.setEnabled(False)
            self.ocr_progress_bar.setVisible(True)
            self.ocr_progress_bar.setValue(0)
            self.ocr_status_label.setText('🔄 กำลังประมวลผล OCR แบบปรับปรุง...')
            self.ocr_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            
            # Clear previous results
            self.ocr_results_text.clear()
            
            # Process OCR
            self.ocr_progress_bar.setValue(20)
            QtWidgets.QApplication.processEvents()
            
            # Use DeepOCR with enhanced preprocessing
            if self.ocr_model is not None:
                print("🔤 OCR TEST ENHANCED: Using DeepOCR with enhanced preprocessing...")
                
                self.ocr_progress_bar.setValue(40)
                QtWidgets.QApplication.processEvents()
                
                # Enhanced preprocessing for better number recognition
                processed_image = self.preprocess_image_for_ocr(self.ocr_test_image)
                
                # Additional preprocessing specifically for numbers
                enhanced_image = self.enhance_image_for_numbers(processed_image)
                
                self.ocr_progress_bar.setValue(60)
                QtWidgets.QApplication.processEvents()
                
                # Perform OCR using DeepOCR
                ocr_result = self.ocr_model.recognize_text_from_image_array(
                    enhanced_image, 
                    self.ocr_test_image_path
                )
                
                self.ocr_progress_bar.setValue(80)
                QtWidgets.QApplication.processEvents()
                
                # Process results
                if 'error' not in ocr_result and ocr_result.get('recognized_text'):
                    recognized_text = ocr_result['recognized_text']
                    confidence_score = ocr_result.get('confidence_score', 0.0)
                    
                    print(f"🔤 OCR TEST ENHANCED: OCR completed successfully")
                    print(f"🔤 OCR TEST ENHANCED: Recognized text: '{recognized_text}'")
                    print(f"🔤 OCR TEST ENHANCED: Confidence: {confidence_score:.3f}")
                    
                    # Format results
                    result_text = f"🔤 ผลลัพธ์ OCR แบบปรับปรุง (DeepOCR)\n"
                    result_text += f"📊 สถานะ: สำเร็จ\n"
                    result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                    result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n"
                    result_text += f"🔧 การปรับปรุง: เพิ่มขนาดภาพ, ปรับความคมชัด, CLAHE\n\n"
                    
                    result_text += "📝 ข้อความที่อ่านได้:\n"
                    result_text += "=" * 50 + "\n"
                    result_text += f"'{recognized_text}'\n\n"
                    
                    result_text += "📊 สถิติ:\n"
                    result_text += "=" * 50 + "\n"
                    result_text += f"ความเชื่อมั่น: {confidence_score:.3f}\n"
                    result_text += f"จำนวนตัวอักษร: {len(recognized_text)}\n"
                    result_text += f"จำนวนคำ: {len(recognized_text.split())}\n"
                    
                    # Show additional info if available
                    if 'processing_time' in ocr_result:
                        result_text += f"เวลาประมวลผล: {ocr_result['processing_time']:.2f} วินาที\n"
                    
                    if 'model_info' in ocr_result:
                        result_text += f"โมเดล: {ocr_result['model_info']}\n"
                    
                else:
                    error_msg = ocr_result.get('error', 'ไม่สามารถอ่านข้อความได้')
                    result_text = f"❌ ไม่พบข้อความในภาพ\n\n"
                    result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                    result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n"
                    result_text += f"❌ ข้อผิดพลาด: {error_msg}\n"
                    print(f"🔤 OCR TEST ENHANCED: OCR failed - {error_msg}")
                
                # Display results
                self.ocr_results_text.setText(result_text)
                
            else:
                result_text = "❌ ข้อผิดพลาด: DeepOCR ไม่พร้อมใช้งาน\n\n"
                result_text += "กรุณาตรวจสอบการติดตั้ง DeepOCR model"
                self.ocr_results_text.setText(result_text)
                print("❌ OCR TEST ENHANCED: DeepOCR not available")
            
            self.ocr_progress_bar.setValue(100)
            QtWidgets.QApplication.processEvents()
            
            # Update UI
            self.btn_process_ocr.setEnabled(True)
            self.btn_process_ocr_enhanced.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('✅ ประมวลผล OCR แบบปรับปรุงเสร็จสิ้น')
            self.ocr_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            
            print("🔤 OCR TEST ENHANCED: Enhanced OCR processing completed")
            
        except Exception as e:
            print(f"❌ Error in process_ocr_test_enhanced: {e}")
            
            # Reset UI
            self.btn_process_ocr.setEnabled(True)
            self.btn_process_ocr_enhanced.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('❌ เกิดข้อผิดพลาดในการประมวลผล OCR แบบปรับปรุง')
            self.ocr_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            
            # Show error in results
            error_text = f"❌ ข้อผิดพลาดในการประมวลผล OCR แบบปรับปรุง\n\n"
            error_text += f"รายละเอียด: {str(e)}\n\n"
            error_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path) if self.ocr_test_image_path else 'ไม่ระบุ'}\n"
            self.ocr_results_text.setText(error_text)
    
    def enhance_image_for_numbers(self, image):
        """
        ปรับปรุงภาพเพิ่มเติมสำหรับการอ่านเลข
        """
        try:
            if image is None:
                return image
            
            print("🔧 Enhancing image specifically for number recognition...")
            
            # แปลงเป็น grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # เพิ่มขนาดภาพอีกครั้งเพื่อให้เลขชัดขึ้น
            height, width = gray.shape
            scale_factor = 1.5  # เพิ่มขนาดอีก 1.5 เท่า
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize ด้วย INTER_LANCZOS4 เพื่อให้ภาพชัดที่สุด
            resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # ปรับปรุงความคมชัดด้วย unsharp mask
            blurred = cv2.GaussianBlur(resized, (0, 0), 1.5)
            sharpened = cv2.addWeighted(resized, 2.0, blurred, -1.0, 0)
            
            # ปรับปรุงความคมชัดด้วย CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened)
            
            # ปรับปรุงความคมชัดด้วย morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # ปรับปรุงความคมชัดด้วย bilateral filter
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # แปลงกลับเป็น BGR format สำหรับ DeepOCR
            if len(image.shape) == 3:
                result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                result = enhanced
            
            print(f"✅ Image enhanced for numbers: {width}x{height} -> {new_width}x{new_height}")
            return result
            
        except Exception as e:
            print(f"❌ Error enhancing image for numbers: {e}")
            return image
            
            QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการประมวลผล OCR: {str(e)}")
    
    def start_system(self):
        """Start system by turning ON M700"""
        if self.modbus_thread:
            success = self.modbus_thread.on_m700()
            if success:
                self.status_label.setText('▶️ ระบบเริ่มทำงาน (M700 ON)')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                self.update_coil_lamp("m700", True)
                print("✅ START: M700 ON - ระบบเริ่มทำงาน")
            else:
                self.status_label.setText('❌ ไม่สามารถเริ่มระบบได้')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                print("❌ START: ไม่สามารถ ON M700 ได้")
        else:
            self.status_label.setText('❌ Modbus ไม่พร้อมใช้งาน')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            print("❌ START: Modbus thread ไม่พร้อมใช้งาน")
    
    def stop_system(self):
        """Stop system by turning ON M701 and reset to initial state"""
        if self.modbus_thread:
            success = self.modbus_thread.on_m701()
            if success:
                self.status_label.setText('⏹️ ระบบหยุดทำงาน (M701 ON - รอ 1.5 วินาที)')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                self.update_coil_lamp("m701", True)
                print("✅ STOP: M701 ON - ระบบหยุดทำงาน (รอ 1.5 วินาที)")
                
                # อัปเดตสถานะหลัง 1.5 วินาที และ reset ทั้งหมด
                QTimer.singleShot(1500, self.reset_to_initial_state)
            else:
                self.status_label.setText('❌ ไม่สามารถหยุดระบบได้')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                print("❌ STOP: ไม่สามารถ ON M701 ได้")
        else:
            self.status_label.setText('❌ Modbus ไม่พร้อมใช้งาน')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            print("❌ STOP: Modbus thread ไม่พร้อมใช้งาน")
    
    def update_stop_status(self):
        """Update status after M701 reset"""
        self.status_label.setText('⏹️ ระบบหยุดทำงาน (M701 Reset)')
        self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        print("✅ STOP: M701 Reset - ระบบหยุดทำงานเสร็จสิ้น")
    
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
    
    def reset_lamp_status(self):
        """Reset all lamp status to OFF state"""
        try:
            print("🔄 Resetting lamp status...")
            
            # Reset connection status
            self.update_modbus_connection_status(False)
            
            # Reset all coil lamps to OFF
            coil_names = ["m100", "m110", "m120", "m130", "m140", "m600", "m700", "m701", "m750", "m850", "m503", "m505", "m720", "m721", "m722", "m730", "m731", "m732", "m733", "m734", "m735", "m740", "m741", "m742"]
            for coil_name in coil_names:
                self.update_coil_lamp(coil_name, False)
            
            # Reset register lamps
            self.update_register_lamp("d6004", "รอค่า", False)
            self.update_register_lamp("d6006", "รอค่า", False)
            self.update_register_lamp("d6007", "รอค่า", False)
            
            print("✅ Lamp status reset completed")
            
        except Exception as e:
            print(f"❌ Error resetting lamp status: {e}")
    
    def reset_modbus_to_initial_state(self):
        """Reset Modbus state to initial values"""
        try:
            if self.modbus_thread:
                print("🔄 Resetting Modbus state...")
                
                # Reset all bottle type coils
                self.modbus_thread.reset_m100()
                self.modbus_thread.reset_m110() 
                self.modbus_thread.reset_m120()
                self.modbus_thread.reset_m130()
                self.modbus_thread.reset_m750()
                self.modbus_thread.reset_m850()
                self.modbus_thread.reset_m140()
                self.modbus_thread.reset_m600()
                
                # Reset M700, M701
                self.modbus_thread.reset_m700()
                self.modbus_thread.reset_m701()
                
                # Reset gripper coils
                self.modbus_thread.reset_m503()
                self.modbus_thread.reset_m505()
                
                # Reset taste mode signals
                self.modbus_thread.reset_m720()
                self.modbus_thread.reset_m721()
                self.modbus_thread.reset_m722()
                
                # Reset 2-taste combination signals
                self.modbus_thread.reset_m730()
                self.modbus_thread.reset_m731()
                self.modbus_thread.reset_m732()
                self.modbus_thread.reset_m733()
                self.modbus_thread.reset_m734()
                self.modbus_thread.reset_m735()
                
                # Reset 1-taste signals
                self.modbus_thread.reset_m740()
                self.modbus_thread.reset_m741()
                self.modbus_thread.reset_m742()
                
                # Reset queue count
                self.reset_queue_count()
                
                # Stop D6006 monitoring for M130
                self.modbus_thread.d6006_monitoring = False
                print("✅ Stopped D6006 monitoring")
                
                # Reset D6007 tracking
                self.modbus_thread.last_d6007_value = None
                print("✅ Reset D6007 tracking")
                
                print("✅ Modbus state reset completed")
                
        except Exception as e:
            print(f"❌ Error resetting Modbus state: {e}")
    
    def reset_queue_count(self):
        """Reset queue count to zero"""
        try:
            if self.modbus_thread:
                print("🔄 Resetting queue count...")
                self.modbus_thread.pending_m301_count = 0
                self.modbus_thread.m600_reset_pending = False
                self.modbus_thread.d6006_monitoring = False
                print("✅ Queue count and monitoring reset to 0")
        except Exception as e:
            print(f"❌ Error resetting queue count: {e}")
    
    def reset_ui_to_initial_state(self):
        """Reset all UI elements to initial state"""
        try:
            print("🔄 Resetting UI elements...")
            
            # Reset main status labels
            self.status_label.setText('⏸️ โปรแกรมพร้อมทำงาน (รอ M511 เพื่อเริ่มการทำงาน)')
            self.status_label.setStyleSheet("color: #f39c12; padding: 5px;")
            
            self.combined_status_label.setText('📊 สถานะรวม: กล้อง USB + Sentech พร้อมทำงาน')
            self.combined_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-weight: bold;")
            
            # Reset Modbus status
            self.modbus_status_label.setText('📡 Modbus: พร้อมใช้งาน')
            self.modbus_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            # Reset queue status
            self.queue_status_label.setText('📋 คิว: 0')
            self.queue_status_label.setStyleSheet("color: #e67e22; padding: 5px; font-weight: bold;")
            
            # Reset queue info label
            self.queue_info_label.setText('📋 สถานะคิว: ไม่มีคิวรอ')
            self.queue_info_label.setStyleSheet("color: #e67e22; padding: 5px; font-size: 11px;")
            
            # Reset D6004 status
            self.d6004_status_label.setText('🔍 D6004: รอค่า')
            self.d6004_status_label.setStyleSheet("color: #9b59b6; padding: 5px; font-size: 11px;")
            
            # Reset D6007 status
            self.d6007_status_label.setText('🔍 D6007: รอค่า')
            self.d6007_status_label.setStyleSheet("color: #9b59b6; padding: 5px; font-size: 11px;")
            
            # Reset gripper status
            self.gripper_status_label.setText('🤖 สถานะ Gripper: รอคำสั่ง')
            self.gripper_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-weight: bold;")
            
            # Reset camera status labels
            self.camera_status_label.setText('📷 กล้อง: พร้อมใช้งาน')
            self.camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            self.sentech_camera_status_label.setText('📷 Sentech: พร้อมใช้งาน')
            self.sentech_camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            # Clear cap detection UI displays
            if hasattr(self, 'cap_detection_text') and self.cap_detection_text:
                self.cap_detection_text.clear()
            
            # Clear cap results layout (if it exists)
            if hasattr(self, 'cap_results_layout') and self.cap_results_layout:
                for i in reversed(range(self.cap_results_layout.count())):
                    widget = self.cap_results_layout.itemAt(i).widget()
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
            self.current_image = None
            self.current_result = None
            self.current_sentech_image = None
            self.current_cap_result = None
            self.original_cap_image = None
            self.ai_rotated_image = None
            self.current_rotation_angle = 0
            
            # Clear batch results
            self.batch_results = []
            
            # Clear image displays (if they exist)
            if hasattr(self, 'image_label') and self.image_label:
                self.image_label.clear()
                self.image_label.setText("📷 ไม่มีภาพ")
            
            # Clear cap detection UI displays
            if hasattr(self, 'cap_detection_text') and self.cap_detection_text:
                self.cap_detection_text.clear()
            
            # Clear cap results layout (if it exists)
            if hasattr(self, 'cap_results_layout') and self.cap_results_layout:
                for i in reversed(range(self.cap_results_layout.count())):
                    widget = self.cap_results_layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)
                self.image_label.setStyleSheet("color: #7f8c8d; padding: 20px; border: 2px dashed #7f8c8d;")
            
            if hasattr(self, 'sentech_image_label') and self.sentech_image_label:
                self.sentech_image_label.clear()
                self.sentech_image_label.setText("📷 ไม่มีภาพ Sentech")
                self.sentech_image_label.setStyleSheet("color: #7f8c8d; padding: 20px; border: 2px dashed #7f8c8d;")
            
            print("✅ Current data cleared")
            
        except Exception as e:
            print(f"❌ Error clearing current data: {e}")
    
    def reset_progress_bars(self):
        """Reset and hide all progress bars"""
        try:
            print("🔄 Resetting progress bars...")
            
            # Reset and hide main progress bar
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(False)
            
            # Reset and hide cap progress bar
            if hasattr(self, 'cap_progress_bar'):
                self.cap_progress_bar.setValue(0)
                self.cap_progress_bar.setVisible(False)
            
            print("✅ Progress bars reset")
            
        except Exception as e:
            print(f"❌ Error resetting progress bars: {e}")
    
    def reset_camera_states(self):
        """Reset camera states"""
        try:
            print("🔄 Resetting camera states...")
            
            # Reset camera status indicators
            if self.usb_camera:
                self.camera_status_label.setText('📷 กล้อง: พร้อมใช้งาน')
                self.camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            if self.sentech_camera:
                self.sentech_camera_status_label.setText('📷 Sentech: พร้อมใช้งาน')
                self.sentech_camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            
            print("✅ Camera states reset")
            
        except Exception as e:
            print(f"❌ Error resetting camera states: {e}")
            
    def display_single_results(self, result):
        """Display single image results"""
        summary = result.get('summary', {})
        detections = result.get('detections', [])
        
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
            
        self.results_text.setText(detail_text)
        
    def display_result_image(self, image):
        """Display result image with detections"""
        try:
            # Resize image for display
            h, w = image.shape[:2]
            max_size = 400
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
            
            # Convert to RGB for Qt
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_image.shape
            bytes_per_line = c * w
            
            qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error displaying result image: {e}")
    
    def open_gripper(self):
        """เปิด Gripper โดยส่งสัญญาณ M503"""
        if self.modbus_thread:
            success = self.modbus_thread.on_m503()
            if success:
                self.gripper_status_label.setText('🟢 Gripper: เปิด (M503 ON)')
                self.gripper_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-weight: bold;")
                self.update_coil_lamp("m503", True)
                print("✅ GRIPPER: M503 ON - เปิด Gripper")
                
                # Reset M503 หลังจาก 0.5 วินาที (Push button behavior)
                QTimer.singleShot(500, self.reset_gripper_open)
            else:
                self.gripper_status_label.setText('❌ Gripper: ไม่สามารถเปิดได้')
                self.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
                print("❌ GRIPPER: ไม่สามารถ ON M503 ได้")
        else:
            self.gripper_status_label.setText('❌ Modbus ไม่พร้อมใช้งาน')
            self.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
            print("❌ GRIPPER: Modbus thread ไม่พร้อมใช้งาน")
    
    def close_gripper(self):
        """ปิด Gripper โดยส่งสัญญาณ M505"""
        if self.modbus_thread:
            success = self.modbus_thread.on_m505()
            if success:
                self.gripper_status_label.setText('🔴 Gripper: ปิด (M505 ON)')
                self.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
                self.update_coil_lamp("m505", True)
                print("✅ GRIPPER: M505 ON - ปิด Gripper")
                
                # Reset M505 หลังจาก 0.5 วินาที (Push button behavior)
                QTimer.singleShot(500, self.reset_gripper_close)
            else:
                self.gripper_status_label.setText('❌ Gripper: ไม่สามารถปิดได้')
                self.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
                print("❌ GRIPPER: ไม่สามารถ ON M505 ได้")
        else:
            self.gripper_status_label.setText('❌ Modbus ไม่พร้อมใช้งาน')
            self.gripper_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-weight: bold;")
            print("❌ GRIPPER: Modbus thread ไม่พร้อมใช้งาน")
    
    def reset_gripper_open(self):
        """Reset M503 หลังจากเปิด Gripper"""
        if self.modbus_thread:
            success = self.modbus_thread.reset_m503()
            if success:
                self.gripper_status_label.setText('🤖 Gripper: รอคำสั่ง (M503 Reset)')
                self.gripper_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-weight: bold;")
                self.update_coil_lamp("m503", False)
                print("✅ GRIPPER: M503 Reset - Push button completed")
            else:
                print("❌ GRIPPER: ไม่สามารถ Reset M503 ได้")
    
    def reset_gripper_close(self):
        """Reset M505 หลังจากปิด Gripper"""
        if self.modbus_thread:
            success = self.modbus_thread.reset_m505()
            if success:
                self.gripper_status_label.setText('🤖 Gripper: รอคำสั่ง (M505 Reset)')
                self.gripper_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-weight: bold;")
                self.update_coil_lamp("m505", False)
                print("✅ GRIPPER: M505 Reset - Push button completed")
            else:
                print("❌ GRIPPER: ไม่สามารถ Reset M505 ได้")
    
    def check_if_jetson(self):
        """Check if running on Jetson device"""
        try:
            # ตรวจสอบ Jetson โดยดูจาก system info
            import platform
            system_info = platform.platform().lower()
            
            # ตรวจสอบ Jetson-specific indicators
            if 'jetson' in system_info or 'tegra' in system_info:
                return True
            
            # ตรวจสอบ ARM architecture (Jetson ใช้ ARM)
            if platform.machine().startswith('aarch64') or platform.machine().startswith('arm'):
                return True
            
            # ตรวจสอบ OS (Jetson ใช้ Linux)
            if platform.system().lower() == 'linux':
                # ตรวจสอบ GPU info
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                    if 'Jetson' in result.stdout or 'Tegra' in result.stdout:
                        return True
                except:
                    pass
            
            return False
        except:
            return False
    
    def setup_jetson_shortcuts(self):
        """Setup keyboard shortcuts for Jetson fullscreen operation"""
        try:
            # Fullscreen toggle (F11)
            fullscreen_shortcut = QtWidgets.QShortcut(Qt.Key_F11, self)
            fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
            
            # Exit fullscreen (Escape)
            exit_fullscreen_shortcut = QtWidgets.QShortcut(Qt.Key_Escape, self)
            exit_fullscreen_shortcut.activated.connect(self.exit_fullscreen)
            
            # Exit application (Ctrl+Q)
            exit_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self)
            exit_shortcut.activated.connect(self.close)
            
            print("✅ Jetson keyboard shortcuts configured")
            print("   F11: Toggle fullscreen")
            print("   Escape: Exit fullscreen")
            print("   Ctrl+Q: Exit application")
            
        except Exception as e:
            print(f"⚠️ Could not setup Jetson shortcuts: {e}")
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and normal window mode"""
        if self.windowState() == Qt.WindowFullScreen:
            self.showNormal()
            print("🖥️ Switched to normal window mode")
        else:
            self.showFullScreen()
            print("🖥️ Switched to fullscreen mode")
    
    def exit_fullscreen(self):
        """Exit fullscreen mode"""
        if self.windowState() == Qt.WindowFullScreen:
            self.showNormal()
            print("🖥️ Exited fullscreen mode")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop status timer
        if hasattr(self, 'status_timer') and self.status_timer.isActive():
            print("🔄 CLOSE EVENT: Stopping status timer...")
            self.status_timer.stop()
        
        # Stop Modbus thread
        if self.modbus_thread:
            self.modbus_thread.stop()
            self.modbus_thread.wait(1000)
        
        # Close USB camera
        if self.usb_camera:
            self.usb_camera.close_camera()
        
        # Close Sentech camera
        if self.sentech_camera:
            self.sentech_camera.cleanup()
        
        # Stop Sentech service
        if hasattr(self, 'sentech_service_process') and self.sentech_service_process:
            print("🔄 CLOSE EVENT: Stopping Sentech service...")
            try:
                self.sentech_service_process.terminate()
                self.sentech_service_process.wait(timeout=5)
            except:
                try:
                    self.sentech_service_process.kill()
                except:
                    pass
        
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = BottleDetectionGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

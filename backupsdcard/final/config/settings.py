# -*- coding: utf-8 -*-
"""
Configuration settings for bottle detection application
เก็บที่อยู่ไฟล์ (paths), model names, ค่าคงที่ (constants)
"""

import os
import sys
import platform

# CRITICAL: Setup CUDA paths BEFORE importing any libs modules
# This must be done at module level before any libs imports
try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    # Fallback if cuda_setup is not available
    pass

# Get project root directory (directory containing this config file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# MODEL PATHS - แก้ไขที่นี่เพื่อเปลี่ยน path ของโมเดล
# จัดระเบียบโมเดลอยู่ใน models/ directory
# =============================================================================
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Detection models
CAP_MODEL_PATH = os.path.join(MODELS_DIR, "detection", "cap.pt")
BOTTLE_MODEL_PATH = os.path.join(MODELS_DIR, "detection", "bottle.pt")
# ค่า confidence ต่ำสุดที่ YOLO ขวดจะนับว่าเจอ (ยิ่งต่ำยิ่งเจอง่าย แต่อาจได้ false positive)
# ถ้าตรวจแล้วไม่เจออะไร จะลองรันอีกครั้งด้วย BOTTLE_YOLO_CONF_FALLBACK
BOTTLE_YOLO_CONF = 0.5
BOTTLE_YOLO_CONF_FALLBACK = 0.25
# ความเร็ว YOLO ขวด: YOLO_IMGSZ เล็กลง = เร็วขึ้น (อาจแม่นน้อยลง), YOLO_MAX_EDGE = ย่อภาพก่อนรันถ้าใหญ่กว่านี้
YOLO_IMGSZ = 640
YOLO_MAX_EDGE = 1280
# Bottle preprocessing: เพิ่มความคมก่อนเข้าโมเดลขวด
BOTTLE_ENABLE_SHARPEN = True
# unsharp amount (0.0-3.0): ยิ่งสูงยิ่งคม แต่อาจเกิด noise/halo
BOTTLE_SHARPEN_AMOUNT = 1.0
# Gaussian blur sigma สำหรับ unsharp (0.5-3.0 แนะนำ)
BOTTLE_SHARPEN_SIGMA = 1.2
# Sentech: จำนวนเฟรมที่ flush ก่อนดึงเฟรมล่าสุด (น้อยลง = เร็วขึ้น), เวลารอเฟรมใหม่ (วินาที)
SENTECH_FLUSH_MAX = 20
SENTECH_WAIT_NEW_FRAME = 0.35

# Rotation models
ROTATION_MODEL_PATH = os.path.join(MODELS_DIR, "rotation", "text_rotation_model.pth")
BEST_ACCURACY_MODEL_PATH = os.path.join(MODELS_DIR, "rotation", "best_accuracy.pth")

# External dependencies directory
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external")

# CRAFT models - ตรวจสอบ path หลายที่
# 1. ตรวจสอบที่ root level (C:\Users\Win 10 Home\Desktop\project_final\CRAFT-pytorch\)
CRAFT_PYTORCH_DIR_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "CRAFT-pytorch")
# 2. ตรวจสอบที่ external directory (final\external\CRAFT-pytorch\)
CRAFT_PYTORCH_DIR = os.path.join(EXTERNAL_DIR, "CRAFT-pytorch")

# ใช้ path ที่มีไฟล์จริง
if os.path.exists(os.path.join(CRAFT_PYTORCH_DIR_ROOT, "craft_mlt_25k.pth")):
    CRAFT_PYTORCH_DIR = CRAFT_PYTORCH_DIR_ROOT
    print(f"✅ Found CRAFT models at: {CRAFT_PYTORCH_DIR}")
elif os.path.exists(os.path.join(CRAFT_PYTORCH_DIR, "craft_mlt_25k.pth")):
    print(f"✅ Found CRAFT models at: {CRAFT_PYTORCH_DIR}")
else:
    print(f"⚠️ CRAFT models not found at either location:")
    print(f"   - {CRAFT_PYTORCH_DIR_ROOT}")
    print(f"   - {CRAFT_PYTORCH_DIR}")

CRAFT_MODEL_PATH = os.path.join(CRAFT_PYTORCH_DIR, "craft_mlt_25k.pth")
CRAFT_REFINER_PATH = os.path.join(CRAFT_PYTORCH_DIR, "craft_refiner_CTW1500.pth")

# Deep text recognition (external dependency - linked to external/)
DEEP_TEXT_RECOGNITION_DIR = os.path.join(EXTERNAL_DIR, "deep-text-recognition-benchmark")

# OCR models
OCR_MODEL_PATH = os.path.join(MODELS_DIR, "ocr", "ocr.pth")

# =============================================================================
# MODEL PATH VALIDATION
# =============================================================================
# Validate that model files exist
def validate_model_paths():
    """Validate that all model paths exist"""
    paths_to_check = {
        'CAP_MODEL_PATH': CAP_MODEL_PATH,
        'BOTTLE_MODEL_PATH': BOTTLE_MODEL_PATH,
        'ROTATION_MODEL_PATH': ROTATION_MODEL_PATH,
        'OCR_MODEL_PATH': OCR_MODEL_PATH,
    }
    
    missing_paths = []
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            missing_paths.append(f"{name}: {path}")
            print(f"⚠️ Warning: {name} not found at {path}")
    
    if missing_paths:
        print(f"⚠️ Warning: {len(missing_paths)} model file(s) not found")
        return False
    
    return True

# Validate paths on import
_ = validate_model_paths()

# =============================================================================
# FADED TEXT DETECTION CONFIGURATION
# =============================================================================
FADED_TEXT_CONFIG = {
    'MIN_AREA': -1,           # เกณฑ์ตัด noise ของ component
    'AREA_THRESH': 0,         # เกณฑ์ตัดสิน faded/normal (0 = ปิดชั่วคราว, ถือว่าไม่จางเสมอ)
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
    },
    # Image enhancement parameters (from brightness analysis)
    # ใช้ค่าที่วิเคราะห์ได้จาก cap_brightness_analyzer (brightness: 33)
    'IMAGE_ENHANCEMENT': {
        'brightness': 33,     # ค่ากลางที่วิเคราะห์ได้ (เพื่อให้ Mean intensity ≈ 128)
        'contrast': 1.8,      # ความคมชัด (ปรับตาม Contrast Ratio ที่วิเคราะห์ได้)
        'gamma': 1.0         # Gamma correction
    },
    # Mapping จาก bottle_type (รส) -> brightness/contrast
    # รสแต่ละรสต้องการค่า brightness/contrast ที่แตกต่างกัน
    'TASTE_ENHANCEMENT_MAP': {
        'M100': {  # รสดั้งเดิม
            'brightness': 16,
            'contrast': 1.7,
            'gamma': 1.0
        },
        'M110': {  # รสน้ำตาลน้อย (สีฟ้า)
            'brightness': 16,
            'contrast': 2.7,
            'gamma': 1.0
        },
        'M120': {  # รสผสมแมงลัก
            'brightness': 16,
            'contrast': 2.7,
            'gamma': 1.0
        }
    },
    'USE_ADAPTIVE_THRESHOLD': True,  # ใช้ Adaptive Threshold แทน Canny (เหมาะกับตัวอักษรสีดำ)
    'USE_EDGES_FOR_ADAPTIVE': False,  # ถ้า True: ใช้ edges มาทำ Adaptive Threshold, ถ้า False: ใช้ roi_masked (วิธีเดิม)
    'USE_EDGES_DIRECTLY': True,  # ถ้า True: ใช้ edges โดยตรงมาทำ text_edges (ไม่ผ่าน Adaptive Threshold) - แนะนำ
    # Histogram Matching for Fade Detection (ใช้แทนการเพิ่มแสงสว่างแบบเดิมโดยอัตโนมัติ)
    'USE_HISTOGRAM_MATCHING': True,  # ใช้ Histogram Matching แทนการเพิ่มแสงสว่างแบบเดิม (เปิดใช้งานเสมอ)
    'REFERENCE_CAP_IMAGE_PATH': r'C:\Users\User\Desktop\projectnon\project\final\captured_images\sentech_camera\sentech_20260117_162643.png',  # Path to reference cap image
    'HISTOGRAM_MATCHING_METHOD': 'clahe',  # 'histogram_matching', 'clahe', 'brightness', 'mixed'
    'PREVENT_SATURATION': True  # ป้องกันการขาวเกิน (saturation)
}

# =============================================================================
# MODBUS CONFIGURATION
# =============================================================================
# โรบอต / PLC — อ่านเขียน coils, D5500 (Program ID), D5001/D5002 ฯลฯ ผ่าน TCP ที่ IP นี้
MODBUS_IP = "192.168.1.5"
MODBUS_PORT = 502

# =============================================================================
# SIMULATION CONFIGURATION (ใช้ตอนทดสอบร่วมกับ Modbus Simulator)
# =============================================================================
# ถ้าใช้ Modbus Simulator และไม่ต่อกล้องจริง:
#   - ตั้ง USE_SIMULATED_IMAGES = True
#   - โปรแกรมจะโหลดภาพจากโฟลเดอร์ด้านล่าง แบบไล่ไฟล์เรียงตามชื่อ (round-robin)
USE_SIMULATED_IMAGES = False

# โฟลเดอร์ภาพขวด (usb) และภาพฝา (sentech) — แต่ละครั้งที่ถ่ายจะใช้ไฟล์ถัดไปในโฟลเดอร์
SIMULATED_USB_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "captured_images", "usb_camera")
SIMULATED_SENTECH_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "captured_images", "sentech_camera")

# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================
SENTECH_GENTL_PATH = '/opt/sentech/lib/libstgentl.cti'

# USB Camera paths
if platform.system() == "Windows":
    USBCAMERA_PATH = r"C:\Users\Win 10 Home\Desktop\Myproject\straure_code\usbcamra.py"
else:
    USBCAMERA_PATH = r"/home/nvidia/Desktop/final_boss/backupsdcard/final/usbcamra.py"

# =============================================================================
# SYSTEM PATHS
# =============================================================================
PYQT5_SYSTEM_PATH = '/usr/lib/python3/dist-packages'

# Add paths to sys.path if not already there
# Add CRAFT-pytorch path (from external directory)
if CRAFT_PYTORCH_DIR not in sys.path and os.path.exists(CRAFT_PYTORCH_DIR):
    sys.path.append(CRAFT_PYTORCH_DIR)
    # Also add basenet subdirectory
    basenet_path = os.path.join(CRAFT_PYTORCH_DIR, "basenet")
    if basenet_path not in sys.path and os.path.exists(basenet_path):
        sys.path.insert(0, basenet_path)

# Add deep-text-recognition-benchmark path (from external directory)
if DEEP_TEXT_RECOGNITION_DIR not in sys.path and os.path.exists(DEEP_TEXT_RECOGNITION_DIR):
    sys.path.insert(0, DEEP_TEXT_RECOGNITION_DIR)

if PYQT5_SYSTEM_PATH not in sys.path:
    sys.path.insert(0, PYQT5_SYSTEM_PATH)

if USBCAMERA_PATH not in sys.path:
    sys.path.append(os.path.dirname(USBCAMERA_PATH))

# =============================================================================
# LIBRARY AVAILABILITY FLAGS
# =============================================================================
# Check for optional libraries
try:
    from harvesters.core import Harvester
    HARVESTERS_AVAILABLE = True
    print("✅ Harvesters imported successfully")
except ImportError:
    HARVESTERS_AVAILABLE = False
    print("⚠️ Harvesters not available")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ Ultralytics YOLO imported successfully")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"⚠️ Ultralytics YOLO not available: {e}")

try:
    from libs.detection.capmodel import initialize_detector, detect_caps_from_path
    from libs.processing.rotationCRAFT import initialize_detector as initialize_craft_detector, detect_text_and_rotate_from_array
    from libs.processing.rotationmodel import (
        initialize_model, 
        get_rotated_image_from_array,
        process_craft_rotated_from_array,
        get_cropped_from_craft_rotated_array,
        process_craft_rotated_with_verification_from_array,
        get_cropped_with_verification_from_craft_rotated_array
    )
    from libs.processing.craft_line_detection import (
        initialize_detector as initialize_line_detector,
        detect_lines_from_rotation_result,
        get_cropped_lines_from_rotation_result
    )
    from libs.processing.deep_ocr import DeepOCRModel, initialize_ocr_model, recognize_text_from_craft_lines
    CAP_DETECTION_AVAILABLE = True
    print("✅ Cap detection modules imported successfully")
except ImportError as e:
    CAP_DETECTION_AVAILABLE = False
    print(f"⚠️ Cap detection modules not available: {e}")

# =============================================================================
# PIL COMPATIBILITY FIX
# =============================================================================
try:
    import PIL.Image
    if not hasattr(PIL.Image, 'ANTIALIAS'):
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
        print("✅ Applied PIL ANTIALIAS compatibility fix")
except Exception as e:
    print(f"Warning: Could not apply PIL compatibility fix: {e}")


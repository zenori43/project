# -*- coding: utf-8 -*-
"""
Image Processor - ฟังก์ชันประมวลผลภาพ, การโหลด/รัน Model (ถ้ามี)
"""

# IMPORTANT: Setup CUDA paths BEFORE importing OpenCV or PyTorch
from core.cuda_setup import setup_cuda_paths
setup_cuda_paths()

import cv2
import numpy as np
import easyocr
import os
from difflib import SequenceMatcher
from typing import List, Dict, Optional

# Import CUDA image utilities
from core.cuda_image_utils import cuda_resize, cuda_cvtColor, cuda_gaussianBlur

from config.settings import FADED_TEXT_CONFIG, YOLO_AVAILABLE

# Initialize EasyOCR reader for bottle detection (Thai + English)
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    bottle_reader = easyocr.Reader(["th", "en"], gpu=CUDA_AVAILABLE)
    print(f"✅ EasyOCR initialized with GPU: {CUDA_AVAILABLE}")
except Exception as e:
    print(f"❌ EasyOCR initialization failed: {e}")
    bottle_reader = None

# Import bottle detection module
from libs.detection.bottledetect import (
    detect_bottle_and_crop_type, 
    process_bottle_image_simple,
    get_type_crops_only,
    get_detection_summary,
    draw_detections_on_image,
    save_cropped_type
)


# =============================================================================
# TEXT MATCHING FUNCTIONS
# =============================================================================

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
    Check bottle type using keyword matching from OCR text only.
    ไม่กรองตาม selected_tastes — ทุกโหมด (1/2/3 รส) แมปตามที่อ่านได้ แล้ว ON M ตามผล (Good → ON M ที่อ่านได้, NG → NG).

    Args:
        ocr_text: combined OCR text
        selected_tastes: ไม่ใช้แล้ว (เก็บไว้เพื่อ backward compatibility)

    Returns:
        str: M100, M110, M120, or None
    """
    if not ocr_text:
        return None

    # แมปจากข้อความที่อ่านได้เท่านั้น — โหมด 1 รส / 2 รส / 3 รส ล้วนประมวลและ ON M ตามที่อ่านได้
    if "เดิม" in ocr_text:
        return "M100"
    if "2%" in ocr_text:
        return "M110"
    if "ลัก" in ocr_text:
        return "M120"
    return None


# =============================================================================
# IMAGE UTILITY FUNCTIONS
# =============================================================================

def safe_show_img(img, to_bgr=False):
    """รับภาพ 1 หรือ 3 แชนเนล แล้วทำให้ปลอดภัยต่อการแสดงผล"""
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if len(img.shape) == 2:
        disp = cuda_cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        disp = img.copy()
        if to_bgr:
            disp = cuda_cvtColor(disp, cv2.COLOR_RGB2BGR)
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
        resized = cuda_resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
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


# =============================================================================
# FADED TEXT DETECTION FUNCTIONS
# =============================================================================

def extract_bottommost_cap_from_image(image_path):
    """
    Extract the bottommost cap from an image (for reference cap selection)
    
    Args:
        image_path: str, path to image file
    
    Returns:
        numpy array: Cropped cap image, or None if not found
    """
    try:
        from libs.detection.capmodel import initialize_detector
        from config.settings import CAP_MODEL_PATH
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot load image: {image_path}")
            return None
        
        # Initialize cap detector with model path
        cap_detector = initialize_detector(CAP_MODEL_PATH)
        if cap_detector is None or cap_detector.model is None:
            print(f"❌ Cannot initialize cap detector")
            return None
        
        # Detect caps
        result = cap_detector.detect_caps(image_array=image)
        if 'error' in result:
            print(f"❌ Cap detection error: {result['error']}")
            return None
        
        detections = result.get('detections', [])
        if not detections:
            print(f"⚠️ No caps found in reference image")
            return None
        
        # Crop all detections
        all_cropped_caps = cap_detector.crop_detections(
            image=image,
            result=result,
            margin=10
        )
        
        if not all_cropped_caps:
            return None
        
        # Find bottommost cap (highest y-coordinate)
        bottommost_cap = None
        max_y = 0
        
        for i, detection in enumerate(detections):
            if 'bbox' in detection:
                bbox = detection['bbox']
                y_center = (bbox[1] + bbox[3]) / 2
                
                if y_center > max_y:
                    max_y = y_center
                    bottommost_cap = all_cropped_caps[i]
        
        if bottommost_cap is None:
            # Fallback: use last cap
            bottommost_cap = all_cropped_caps[-1]
        
        print(f"✅ Extracted bottommost cap from reference image (y_center: {max_y:.1f})")
        return bottommost_cap
        
    except Exception as e:
        print(f"❌ Error extracting bottommost cap: {e}")
        import traceback
        traceback.print_exc()
        return None


def match_cap_histogram_to_reference(cap_image, reference_image, method='clahe', prevent_saturation=True):
    """
    Match histogram of cap image to reference image using specified method
    
    Args:
        cap_image: numpy array image to adjust
        reference_image: numpy array reference image
        method: str, matching method ('histogram_matching', 'clahe', 'brightness', 'mixed')
        prevent_saturation: bool, prevent saturation (limit max value to 95%)
    
    Returns:
        numpy array: Adjusted image
    """
    if cap_image is None or reference_image is None:
        return cap_image
    
    try:
        # Convert to grayscale
        if len(cap_image.shape) == 3:
            cap_gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)
        else:
            cap_gray = cap_image.copy()
        
        if len(reference_image.shape) == 3:
            ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference_image.copy()
        
        if method == 'histogram_matching':
            # Traditional histogram matching
            ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
            src_hist = cv2.calcHist([cap_gray], [0], None, [256], [0, 256])
            
            ref_cdf = ref_hist.cumsum()
            ref_cdf_normalized = (ref_cdf * 255 / ref_cdf.max()).astype(np.uint8)
            
            src_cdf = src_hist.cumsum()
            src_cdf_normalized = (src_cdf * 255 / src_cdf.max()).astype(np.uint8)
            
            # Create mapping
            mapping = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                diff = np.abs(ref_cdf_normalized - src_cdf_normalized[i])
                mapped_value = np.argmin(diff)
                if prevent_saturation:
                    mapping[i] = min(mapped_value, int(255 * 0.95))
                else:
                    mapping[i] = mapped_value
            
            matched_gray = mapping[cap_gray]
            
        elif method == 'clahe':
            # CLAHE method
            ref_mean = np.mean(ref_gray)
            src_mean = np.mean(cap_gray)
            
            brightness_diff = abs(ref_mean - src_mean)
            clip_limit = 2.0 + (brightness_diff / 128.0) * 2.0
            clip_limit = min(clip_limit, 4.0)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(cap_gray)
            
            brightness_shift = ref_mean - np.mean(enhanced)
            matched_gray = cv2.add(enhanced, int(brightness_shift))
            
            if prevent_saturation:
                matched_gray = np.clip(matched_gray, 0, int(255 * 0.95))
            else:
                matched_gray = np.clip(matched_gray, 0, 255)
            
            matched_gray = matched_gray.astype(np.uint8)
            
        elif method == 'brightness':
            # Simple brightness adjustment
            ref_mean = np.mean(ref_gray)
            src_mean = np.mean(cap_gray)
            brightness_diff = ref_mean - src_mean
            
            matched_gray = cv2.add(cap_gray, int(brightness_diff))
            
            if prevent_saturation:
                matched_gray = np.clip(matched_gray, 0, int(255 * 0.95))
            else:
                matched_gray = np.clip(matched_gray, 0, 255)
            
            matched_gray = matched_gray.astype(np.uint8)
            
        elif method == 'mixed':
            # Histogram matching + CLAHE
            ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
            src_hist = cv2.calcHist([cap_gray], [0], None, [256], [0, 256])
            
            ref_cdf = ref_hist.cumsum()
            ref_cdf_normalized = (ref_cdf * 255 / ref_cdf.max()).astype(np.uint8)
            
            src_cdf = src_hist.cumsum()
            src_cdf_normalized = (src_cdf * 255 / src_cdf.max()).astype(np.uint8)
            
            mapping = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                diff = np.abs(ref_cdf_normalized - src_cdf_normalized[i])
                mapped_value = np.argmin(diff)
                mapping[i] = min(mapped_value, int(255 * 0.95))
            
            hist_matched = mapping[cap_gray]
            
            # Apply CLAHE on top
            ref_mean = np.mean(ref_gray)
            src_mean = np.mean(hist_matched)
            brightness_diff = abs(ref_mean - src_mean)
            clip_limit = 2.0 + (brightness_diff / 128.0) * 2.0
            clip_limit = min(clip_limit, 4.0)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(hist_matched)
            
            brightness_shift = ref_mean - np.mean(enhanced)
            matched_gray = cv2.add(enhanced, int(brightness_shift))
            
            if prevent_saturation:
                matched_gray = np.clip(matched_gray, 0, int(255 * 0.95))
            else:
                matched_gray = np.clip(matched_gray, 0, 255)
            
            matched_gray = matched_gray.astype(np.uint8)
        else:
            # Default: use original image
            matched_gray = cap_gray
        
        # Convert back to BGR if original was color
        if len(cap_image.shape) == 3:
            matched_image = cv2.cvtColor(matched_gray, cv2.COLOR_GRAY2BGR)
        else:
            matched_image = matched_gray.copy()
        
        return matched_image
        
    except Exception as e:
        print(f"❌ Error in histogram matching: {e}")
        import traceback
        traceback.print_exc()
        return cap_image


def enhance_cap_image_for_fade_detection(image, bottle_type=None):
    """
    Enhance cap image with histogram matching or brightness/contrast adjustment
    for better text detection (especially black text on light background)
    
    Args:
        image: numpy array image to enhance
        bottle_type: str, bottle type (M100, M110, M120) - ไม่ใช้แล้ว (เก็บไว้เพื่อ backward compatibility)
    """
    if image is None:
        return None
    
    # Check if histogram matching is enabled
    use_histogram_matching = FADED_TEXT_CONFIG.get('USE_HISTOGRAM_MATCHING', True)
    reference_cap_path = FADED_TEXT_CONFIG.get('REFERENCE_CAP_IMAGE_PATH')
    
    if use_histogram_matching and reference_cap_path and os.path.exists(reference_cap_path):
        try:
            # Extract bottommost cap from reference image (if it's a full image, not already cropped)
            reference_cap_image = extract_bottommost_cap_from_image(reference_cap_path)
            
            if reference_cap_image is None:
                # Fallback: try loading as direct cap image
                reference_cap_image = cv2.imread(reference_cap_path)
            
            if reference_cap_image is not None:
                method = FADED_TEXT_CONFIG.get('HISTOGRAM_MATCHING_METHOD', 'clahe')
                prevent_saturation = FADED_TEXT_CONFIG.get('PREVENT_SATURATION', True)
                
                enhanced = match_cap_histogram_to_reference(
                    image, 
                    reference_cap_image, 
                    method=method,
                    prevent_saturation=prevent_saturation
                )
                
                print(f"🔍 ENHANCE IMAGE: ใช้ Histogram Matching (วิธี: {method}) แทนการเพิ่มแสงสว่างแบบเดิม")
                return enhanced
            else:
                print(f"⚠️ ENHANCE IMAGE: ไม่สามารถโหลดหรือตรวจจับฝาจากรูปอ้างอิงได้: {reference_cap_path}")
        except Exception as e:
            print(f"❌ ENHANCE IMAGE: เกิดข้อผิดพลาดในการใช้ Histogram Matching: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to old method
    
    # Fallback to old brightness/contrast method
    enhanced = image.copy()
    
    # ใช้ค่า default จาก IMAGE_ENHANCEMENT โดยตรง (ไม่ต้องรอ bottle_type)
    # ค่านี้มาจากการวิเคราะห์ค่ากลางจาก cap_brightness_analyzer
    enhancement = FADED_TEXT_CONFIG.get('IMAGE_ENHANCEMENT', {})
    print(f"🔍 ENHANCE IMAGE: ใช้ค่ากลางที่วิเคราะห์ได้ - brightness={enhancement.get('brightness')}, contrast={enhancement.get('contrast')}")
    
    # Apply brightness
    brightness = enhancement.get('brightness', 0)
    if brightness != 0:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1, beta=brightness)
    
    # Apply contrast
    contrast = enhancement.get('contrast', 1.0)
    if contrast != 1.0:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=0)
    
    # Apply gamma correction
    gamma = enhancement.get('gamma', 1.0)
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                        for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
    
    return enhanced


def detect_faded_text_in_unified_region(unified_region, show_debug=False):
    """
    Detect faded text in unified region from CRAFT (simplified version)
    - ไม่ต้อง Enhance Image, Grayscale+Blur, Detect Circle
    - ใช้เฉพาะ Canny edges, morphological operations, และ CCA
    
    Args:
        unified_region: numpy array of the unified region from CRAFT (already cropped)
        show_debug: whether to show debug visualization
        
    Returns:
        dict: {
            'status': 'faded' or 'normal' or 'unknown',
            'num_chars': int,
            'total_area': int,
            'debug_images': dict (if show_debug=True)
        }
    """
    if unified_region is None:
        return {'status': 'unknown', 'num_chars': 0, 'total_area': 0}
    
    try:
        # Convert to BGR if needed
        if len(unified_region.shape) == 3 and unified_region.shape[2] == 3:
            img = unified_region.copy()
        else:
            img = cuda_cvtColor(unified_region, cv2.COLOR_GRAY2BGR)
        
        # Initialize result
        result = {
            'status': 'unknown',
            'num_chars': 0,
            'total_area': 0
        }
        
        if show_debug:
            result['debug_images'] = {}
        
        roi = img.copy()
        roi_show = roi.copy()
        
        # Convert to grayscale (ไม่ blur)
        gray = cuda_cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # เพิ่ม brightness/contrast เพื่อให้เห็น edges ในภาพมืด
        # คำนวณค่าเฉลี่ยของภาพ
        mean_brightness = np.mean(gray)
        if mean_brightness < 100:  # ถ้าภาพมืดเกินไป
            # เพิ่ม brightness
            brightness_adj = int(100 - mean_brightness)
            gray_enhanced = cv2.convertScaleAbs(gray, alpha=1, beta=brightness_adj)
            # เพิ่ม contrast
            gray_enhanced = cv2.convertScaleAbs(gray_enhanced, alpha=1.5, beta=0)
            print(f"🔍 เพิ่ม brightness/contrast - mean_brightness: {mean_brightness:.1f}, adjusted")
        else:
            gray_enhanced = gray.copy()
            print(f"🔍 ไม่ต้องปรับ brightness - mean_brightness: {mean_brightness:.1f}")
        
        # ตัวเลือกวิธีการสร้าง text_edges
        use_edges_directly = FADED_TEXT_CONFIG.get('USE_EDGES_DIRECTLY', False)
        use_adaptive = FADED_TEXT_CONFIG.get('USE_ADAPTIVE_THRESHOLD', False)
        use_edges_for_adaptive = FADED_TEXT_CONFIG.get('USE_EDGES_FOR_ADAPTIVE', False)
        
        # สร้าง edges จากภาพที่ปรับแล้ว (ใช้ threshold ที่ต่ำกว่าเพื่อให้เจอ edges ในภาพมืด)
        canny_low = FADED_TEXT_CONFIG['CANNY_PARAMS']['low_threshold']
        canny_high = FADED_TEXT_CONFIG['CANNY_PARAMS']['high_threshold']
        
        # ปรับ threshold ให้ต่ำลงสำหรับภาพมืด เพื่อให้จับ edges ได้ครบ
        if mean_brightness < 100:
            canny_low = max(20, canny_low // 3)  # ลด threshold ลงมากกว่าเดิม เพื่อให้จับ edges ได้ครบ
            canny_high = max(50, canny_high // 3)  # ลด threshold ลงมากกว่าเดิม
            print(f"🔍 ปรับ Canny thresholds สำหรับภาพมืด - low: {canny_low}, high: {canny_high}")
        else:
            # สำหรับภาพปกติ ก็ลด threshold ลงเล็กน้อยเพื่อให้จับ edges ได้ครบ
            canny_low = max(30, canny_low // 2)
            canny_high = max(60, canny_high // 2)
            print(f"🔍 ปรับ Canny thresholds - low: {canny_low}, high: {canny_high}")
        
        edges = cv2.Canny(gray_enhanced, canny_low, canny_high)
        
        if use_edges_directly:
            # ใช้ edges โดยตรงมาทำ text_edges
            print("🔍 ใช้ Edges โดยตรงมาทำ Text Edges")
            text_edges = edges.copy()
            
            # ใช้ closing เพื่อเชื่อมต่อ edges ที่ขาด (เพิ่ม kernel size เล็กน้อยเพื่อเชื่อมต่อได้ดีขึ้น)
            kernel_close = np.ones((3, 3), np.uint8)  # เพิ่มจาก 2x2 เป็น 3x3 เพื่อเชื่อมต่อ edges ที่ขาด
            text_edges = cv2.morphologyEx(text_edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            
        elif use_adaptive:
            if use_edges_for_adaptive:
                # ใช้ edges มาทำ Adaptive Threshold
                print("🔍 ใช้ Edges มาทำ Adaptive Threshold")
                
                # ลด dilation - ใช้ iterations น้อยลง
                kernel_dilate = np.ones((3, 3), np.uint8)
                edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)  # ลดจาก 2 เป็น 1
                
                # ใช้ edges_dilated มาทำ Adaptive Threshold
                adaptive_thresh = cv2.adaptiveThreshold(
                    edges_dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
            else:
                # ใช้ gray_enhanced มาทำ Adaptive Threshold (ใช้ภาพที่ปรับแล้ว)
                print("🔍 ใช้ Gray Enhanced มาทำ Adaptive Threshold")
                adaptive_thresh = cv2.adaptiveThreshold(
                    gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
            
            # ลด morphological operations - ใช้แค่ closing ครั้งเดียว
            kernel = np.ones((2, 2), np.uint8)
            text_edges = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        else:
            # Use Canny edge detection (original method)
            print("🔍 ใช้ Edges โดยตรง (Canny)")
            text_edges = edges.copy()
        
        # ทำความสะอาดและเชื่อมต่อ edges ที่ขาด
        # ใช้ closing เพื่อเชื่อมต่อ edges ที่ขาด (เพิ่ม kernel size เพื่อเชื่อมต่อได้ดีขึ้น)
        kernel_close = np.ones((3, 3), np.uint8)  # ใช้ 3x3 เพื่อเชื่อมต่อ edges ที่ขาด
        text_edges_cleaned = cv2.morphologyEx(text_edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # ตรวจสอบและลบขอบ (circular edges) ถ้ามี
        # ตรวจสอบว่ามีเส้นโค้งใกล้ขอบของ unified region หรือไม่
        h, w = text_edges_cleaned.shape
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        
        # สร้าง mask สำหรับขอบ (เฉพาะส่วนที่อยู่ใกล้ขอบของ unified region)
        # ใช้เฉพาะส่วนที่อยู่ใกล้ขอบ (ประมาณ 10-15% จากขอบ)
        edge_margin = min(h, w) // 10  # ประมาณ 10% ของขนาดภาพ
        
        # สร้าง mask สำหรับขอบบน-ล่าง
        edge_mask[:edge_margin, :] = 255  # ขอบบน
        edge_mask[-edge_margin:, :] = 255  # ขอบล่าง
        
        # สร้าง mask สำหรับขอบซ้าย-ขวา
        edge_mask[:, :edge_margin] = 255  # ขอบซ้าย
        edge_mask[:, -edge_margin:] = 255  # ขอบขวา
        
        # ตรวจสอบว่ามี edges อยู่ในขอบหรือไม่
        edges_in_border = cv2.bitwise_and(text_edges_cleaned, edge_mask)
        border_edge_count = np.sum(edges_in_border > 0)
        total_edge_count = np.sum(text_edges_cleaned > 0)
        
        # ถ้ามี edges ในขอบมากกว่า 20% ของ edges ทั้งหมด → น่าจะมีขอบ (circular edge)
        # และถ้า edges ในขอบมีความยาวมาก (อาจเป็นเส้นโค้ง) → ลบออก
        if total_edge_count > 0 and border_edge_count / total_edge_count > 0.2:
            # ตรวจสอบว่า edges ในขอบเป็นเส้นโค้งหรือไม่ (ใช้ HoughCircles)
            try:
                # หาเส้นโค้งในขอบ
                circles = cv2.HoughCircles(
                    edges_in_border,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=min(h, w) // 2,
                    param1=50,
                    param2=20,
                    minRadius=min(h, w) // 4,
                    maxRadius=min(h, w) // 2
                )
                
                if circles is not None:
                    # พบเส้นโค้ง → ลบ edges ในขอบออก
                    print(f"🔍 พบเส้นโค้งใกล้ขอบ → ลบ edges ในขอบออก (border edges: {border_edge_count}/{total_edge_count})")
                    # สร้าง mask สำหรับลบขอบ (ขยายเล็กน้อยเพื่อให้แน่ใจว่าลบหมด)
                    border_remove_mask = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8), iterations=1)
                    text_edges_cleaned = cv2.bitwise_and(text_edges_cleaned, cv2.bitwise_not(border_remove_mask))
                else:
                    # ไม่พบเส้นโค้ง แต่มี edges ในขอบมาก → อาจเป็นขอบจริงๆ → ลบออก
                    if border_edge_count / total_edge_count > 0.3:
                        print(f"🔍 พบ edges ในขอบมาก → ลบ edges ในขอบออก (border edges: {border_edge_count}/{total_edge_count})")
                        border_remove_mask = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8), iterations=1)
                        text_edges_cleaned = cv2.bitwise_and(text_edges_cleaned, cv2.bitwise_not(border_remove_mask))
                    else:
                        print(f"🔍 พบ edges ในขอบเล็กน้อย → ไม่ลบ (border edges: {border_edge_count}/{total_edge_count})")
            except Exception as e:
                print(f"⚠️ เกิดข้อผิดพลาดในการตรวจสอบขอบ: {e}")
        else:
            print(f"🔍 ไม่พบ edges ในขอบ → ไม่ต้องลบ (border edges: {border_edge_count}/{total_edge_count})")
        
        # CCA (Connected Component Analysis)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_edges_cleaned, connectivity=8)
        stat_fg = stats[1:, :]
        
        if stat_fg.size > 0:
            areas_all = stat_fg[:, cv2.CC_STAT_AREA]
            min_area = FADED_TEXT_CONFIG['MIN_AREA']
            if min_area <= 0:
                min_area = 20
            
            # กรองตามขนาด
            keep_mask = areas_all > min_area
            
            # กรองตาม aspect ratio
            h, w = text_edges_cleaned.shape
            center_x, center_y = w // 2, h // 2
            
            for i in range(len(stats)):
                if i == 0:
                    continue
                if not keep_mask[i-1]:
                    continue
                
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > 10:
                        keep_mask[i-1] = False
                    
                    # กรองตามความหนาแน่น
                    bbox_area = width * height
                    if bbox_area > 0:
                        density = areas_all[i-1] / bbox_area
                        if density < 0.15 and areas_all[i-1] < 100:
                            keep_mask[i-1] = False
            
            areas_keep = areas_all[keep_mask]
            num_chars = int(keep_mask.sum())
            total_area = int(areas_keep.sum())
        else:
            num_chars, total_area = 0, 0
        
        # Normalize พื้นที่ตามขนาดของ unified region เพื่อให้เปรียบเทียบได้
        # คำนวณพื้นที่ต่อหน่วย (normalized area) = total_area / (width * height)
        unified_region_area = h * w
        normalized_area = (total_area / unified_region_area * 100) if unified_region_area > 0 else 0  # แปลงเป็นเปอร์เซ็นต์
        
        print(f"🔍 Unified Region Size: {w}x{h} = {unified_region_area} pixels")
        print(f"🔍 Total Text Area: {total_area} pixels ({normalized_area:.2f}% of unified region)")
        
        # Determine status
        area_thresh = FADED_TEXT_CONFIG['AREA_THRESH']
        if area_thresh <= 0:
            status = "normal"
            print(f"🔍 AREA_THRESH = 0 — ปิดเกณฑ์จาง (ถือว่า normal)")
        else:
            # ใช้ normalized area เปรียบเทียบ (แปลง area_thresh เป็นเปอร์เซ็นต์)
            # สมมติว่า area_thresh เป็นพื้นที่ขั้นต่ำสำหรับ unified region ขนาดมาตรฐาน (เช่น 200x100 = 20000 pixels)
            # คำนวณ normalized threshold
            standard_region_size = 20000  # ขนาดมาตรฐาน (ประมาณ 200x100)
            normalized_thresh = (area_thresh / standard_region_size * 100) if standard_region_size > 0 else 0
            
            print(f"🔍 ใช้เกณฑ์ area_thresh = {area_thresh} (normalized: {normalized_thresh:.2f}%)")
            
            # เปรียบเทียบ normalized area กับ normalized threshold
            if normalized_area < normalized_thresh:
                status = "faded"
                print(f"🔍 Normalized area ({normalized_area:.2f}%) < threshold ({normalized_thresh:.2f}%) → FADED")
            else:
                status = "normal"
                print(f"🔍 Normalized area ({normalized_area:.2f}%) >= threshold ({normalized_thresh:.2f}%) → NORMAL")
        
        result.update({
            'status': status,
            'num_chars': num_chars,
            'total_area': total_area
        })
        
        if show_debug:
            debug_images = {
                'roi_show': roi_show,
                'gray': gray,
                'gray_enhanced': gray_enhanced,
                'edges': edges,
                'text_edges': text_edges_cleaned
            }
            result['debug_images'] = debug_images
        
        print(f"Faded text detection (unified region) - Components: {num_chars}, Area: {total_area}, Status: {status}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in faded text detection (unified region): {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'unknown', 'num_chars': 0, 'total_area': 0, 'error': str(e)}


def detect_faded_text_in_cap(cap_image, yolo_model=None, show_debug=False, bottle_type=None):
    """
    Detect faded text in cap image using the provided algorithm
    
    Args:
        cap_image: numpy array of the cropped cap image (from Step 2: Crop detected regions)
        yolo_model: YOLO model for cap detection (ไม่ใช้แล้ว - ใช้รูปที่ crop แล้วโดยตรง)
        show_debug: whether to show debug visualization
        bottle_type: str, bottle type (M100, M110, M120) - ไม่ใช้แล้ว (เก็บไว้เพื่อ backward compatibility)
        
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
            img = cuda_cvtColor(cap_image, cv2.COLOR_GRAY2BGR)
        
        # Enhance image before detection (เพิ่มแสงและ contrast เพื่อตรวจจับตัวอักษรสีดำได้ดีขึ้น)
        # ใช้ค่ากลางที่วิเคราะห์ได้ (brightness: 33) โดยตรง ไม่ต้องรอ bottle_type
        img = enhance_cap_image_for_fade_detection(img, bottle_type=None)
        
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
        gray = cuda_cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cuda_gaussianBlur(gray, 
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
        # เพิ่มความหนาของ circle_edges เพื่อลบเส้นวงกลมให้สมบูรณ์ (จาก 2 เป็น 5)
        cv2.circle(circle_edges, (int(cx), int(cy)), int(r), 255, 5)  # เพิ่มความหนาเพื่อลบเส้นวงกลมให้ดีขึ้น
        cv2.circle(roi_show, (int(cx), int(cy)), int(r), (0,255,0), 2)
        
        # Mask ROI
        roi_masked = cv2.bitwise_and(gray, gray, mask=mask_circle)
        
        # ตัวเลือกวิธีการสร้าง text_edges
        use_edges_directly = FADED_TEXT_CONFIG.get('USE_EDGES_DIRECTLY', False)  # ใช้ edges โดยตรงมาทำ text_edges
        use_adaptive = FADED_TEXT_CONFIG.get('USE_ADAPTIVE_THRESHOLD', False)
        use_edges_for_adaptive = FADED_TEXT_CONFIG.get('USE_EDGES_FOR_ADAPTIVE', False)  # ใช้ edges มาทำ Adaptive Threshold
        
        # สร้าง edges ก่อน (ใช้สำหรับทุกกรณี)
        edges = cv2.Canny(roi_masked, 
                         FADED_TEXT_CONFIG['CANNY_PARAMS']['low_threshold'], 
                         FADED_TEXT_CONFIG['CANNY_PARAMS']['high_threshold'])
        
        # ลบเส้นวงกลมออกจาก edges (เพื่อแสดงใน debug images)
        # วิธีที่ 1: ใช้ bitwise_and (วิธีเดิม)
        edges_no_circle = cv2.bitwise_and(edges, cv2.bitwise_not(circle_edges))
        
        # วิธีที่ 2: ใช้ HoughCircles เพื่อหาเส้นวงกลมใน edges แล้วลบ (เพิ่มเติม)
        # หาเส้นวงกลมใน edges เพื่อลบให้สมบูรณ์
        circles_in_edges = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, 
            dp=1, minDist=int(r*2), 
            param1=50, param2=20,
            minRadius=int(r*0.9), maxRadius=int(r*1.1)
        )
        
        if circles_in_edges is not None:
            circles_in_edges = np.uint16(np.around(circles_in_edges))
            for circle in circles_in_edges[0, :]:
                cx_edge, cy_edge, r_edge = circle
                # ลบเส้นวงกลมที่พบใน edges (ใช้ความหนา 5 เพื่อให้แน่ใจว่าลบหมด)
                cv2.circle(edges_no_circle, (int(cx_edge), int(cy_edge)), int(r_edge), 0, 5)
        
        # วิธีที่ 3: ใช้ distance transform เพื่อลบเส้นที่อยู่ใกล้ขอบวงกลม (เพิ่มเติม)
        # สร้าง mask สำหรับส่วนกลาง (ลดเส้นที่อยู่ใกล้ขอบวงกลม)
        h, w = edges_no_circle.shape
        center_mask = np.zeros((h, w), dtype=np.uint8)
        # สร้างวงกลมที่เล็กกว่าเล็กน้อยเพื่อลบเส้นที่อยู่ใกล้ขอบ
        cv2.circle(center_mask, (int(cx), int(cy)), int(r * 0.95), 255, -1)  # 95% ของรัศมี
        
        # ใช้ mask เพื่อลดเส้นที่อยู่ใกล้ขอบ (อาจเป็นเส้นวงกลม)
        # แต่ไม่ลบทั้งหมด - เก็บเฉพาะส่วนที่อยู่ภายในวงกลม 95%
        edges_no_circle = cv2.bitwise_and(edges_no_circle, center_mask)
        
        if use_edges_directly:
            # ใช้ edges โดยตรงมาทำ text_edges (ไม่ผ่าน Adaptive Threshold)
            print("🔍 ใช้ Edges โดยตรงมาทำ Text Edges")
            
            # เริ่มจาก edges ที่ลบเส้นวงกลมแล้ว
            text_edges = edges_no_circle.copy()
            
            # ใช้ morphological operations เพื่อขยาย edges ให้เป็นพื้นที่ (เหมือน Adaptive Threshold)
            # Dilation เพื่อขยาย edges ให้เป็นพื้นที่ (ลดการขยายลงมาก - ใช้ kernel เล็กและ iterations น้อย)
            kernel_dilate = np.ones((1, 1), np.uint8)  # kernel ขนาด 1x1 (ไม่ขยายเลย)
            # ไม่ใช้ dilation หรือใช้แค่ closing เพื่อเชื่อมต่อ edges โดยไม่ขยายมาก
            # text_edges = cv2.dilate(text_edges, kernel_dilate, iterations=0)  # ไม่ขยายเลย
            # ใช้ closing แทน dilation เพื่อเชื่อมต่อ edges โดยไม่ขยายมาก
            kernel_close = np.ones((2, 2), np.uint8)
            text_edges = cv2.morphologyEx(text_edges, cv2.MORPH_CLOSE, kernel_close)  # ปิดช่องว่างเล็กๆ โดยไม่ขยายมาก
            
            # Apply morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            text_edges = cv2.morphologyEx(text_edges, cv2.MORPH_CLOSE, kernel)  # ปิดช่องว่าง
            text_edges = cv2.morphologyEx(text_edges, cv2.MORPH_OPEN, kernel)  # ลบจุดเล็กๆ
            
        elif use_adaptive:
            if use_edges_for_adaptive:
                # ใช้ edges มาทำ Adaptive Threshold
                print("🔍 ใช้ Edges มาทำ Adaptive Threshold")
                
                # Dilation edges เพื่อสร้างพื้นที่
                kernel_dilate = np.ones((3, 3), np.uint8)
                edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=2)  # ขยาย edges ให้เป็นพื้นที่
                
                # ใช้ edges_dilated มาทำ Adaptive Threshold
                adaptive_thresh = cv2.adaptiveThreshold(
                    edges_dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2  # ไม่ต้อง INV เพราะ edges_dilated เป็นขาวบนดำ
                )
            else:
                # ใช้ roi_masked มาทำ Adaptive Threshold (วิธีเดิม)
                print("🔍 ใช้ roi_masked มาทำ Adaptive Threshold (วิธีเดิม)")
                adaptive_thresh = cv2.adaptiveThreshold(
                    roi_masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
            
            # Apply morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            text_edges = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            text_edges = cv2.morphologyEx(text_edges, cv2.MORPH_OPEN, kernel)
            
            # ลบเส้นวงกลม
            text_edges = cv2.bitwise_and(text_edges, cv2.bitwise_not(circle_edges))
        else:
            # Use Canny edge detection (original method) - ใช้ edges โดยตรง
            print("🔍 ใช้ Edges โดยตรง (Canny) - ไม่ผ่าน Adaptive Threshold")
            
            # ลบเส้นวงกลม
            text_edges = edges_no_circle.copy()
        
        # กรอง noise และจุดรอบๆ (ไม่สนใจจุดเล็กๆ รอบๆ) - เหมือนใน cap_fade_detector
        # ใช้ morphological operations เพื่อลบจุดเล็กๆ
        kernel_small = np.ones((2, 2), np.uint8)
        # ลบจุดเล็กๆ ด้วย opening (ทำแค่ครั้งเดียวเพื่อไม่ให้ตัวอักษรหาย)
        text_edges_cleaned = cv2.morphologyEx(text_edges, cv2.MORPH_OPEN, kernel_small)
        # ปิดช่องว่างเล็กๆ ในตัวอักษรด้วย closing
        kernel_close = np.ones((3, 3), np.uint8)
        text_edges_cleaned = cv2.morphologyEx(text_edges_cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        # กรองตามตำแหน่ง - เน้นส่วนกลางของฝา (ลดจุดรอบๆ ขอบ)
        h, w = text_edges_cleaned.shape
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 2
        
        # สร้าง mask สำหรับส่วนกลาง (ลดน้ำหนักจุดรอบๆ ขอบ)
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(center_mask, (center_x, center_y), int(radius * 0.9), 255, -1)
        
        # ใช้ mask เพื่อลดจุดรอบๆ ขอบ
        text_edges_cleaned = cv2.bitwise_and(text_edges_cleaned, center_mask)
        
        # CCA (Connected Component Analysis) - ใช้ text_edges_cleaned
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_edges_cleaned, connectivity=8)
        stat_fg = stats[1:, :]  # ลบแบ็คกราวด์
        if stat_fg.size > 0:
            areas_all = stat_fg[:, cv2.CC_STAT_AREA]
            min_area = FADED_TEXT_CONFIG['MIN_AREA']
            if min_area <= 0:
                min_area = 20  # เพิ่มจาก 15 เป็น 20 (ไม่มากเกินไปเพื่อไม่ให้ตัวอักษรหาย)
            
            # กรองตามขนาด (ลบจุดเล็กๆ)
            keep_mask = areas_all > min_area
            
            # กรองตามตำแหน่ง - เน้นส่วนกลาง (ลดจุดรอบๆ ขอบ)
            max_dist_from_center = min(w, h) * 0.4  # ระยะสูงสุดจากจุดศูนย์กลาง (40% ของรัศมี)
            
            # คำนวณระยะห่างจากจุดศูนย์กลาง
            for i in range(len(centroids)):
                if i == 0:  # ข้าม background
                    continue
                if not keep_mask[i-1]:
                    continue
                cx, cy = centroids[i]
                dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                if dist_from_center > max_dist_from_center:
                    keep_mask[i-1] = False  # ลบจุดที่อยู่ไกลจากจุดศูนย์กลาง
            
            # กรองตาม aspect ratio - ตัวอักษรมักมี aspect ratio ที่เหมาะสม
            for i in range(len(stats)):
                if i == 0:  # ข้าม background
                    continue
                if not keep_mask[i-1]:
                    continue
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    # ลบ component ที่ aspect ratio ผิดปกติมาก (อาจเป็น noise)
                    if aspect_ratio > 10:  # ถ้ายาวมากเกินไป อาจเป็นเส้น noise
                        keep_mask[i-1] = False
                    
                    # กรองตามความหนาแน่นของ pixels (density) - ตัวอักษรมักมีความหนาแน่นสูง
                    # แต่ใช้ threshold ต่ำเพื่อไม่ให้ตัวอักษรหาย
                    bbox_area = width * height
                    if bbox_area > 0:
                        density = areas_all[i-1] / bbox_area
                        # ลบ component ที่มีความหนาแน่นต่ำมากๆ เท่านั้น (อาจเป็น noise ที่กระจายมาก)
                        # ใช้ threshold ต่ำ (0.15) เพื่อไม่ให้ตัวอักษรหาย
                        if density < 0.15 and areas_all[i-1] < 100:  # กรองเฉพาะ noise ที่เล็กและกระจาย
                            keep_mask[i-1] = False
                    
                    # กรองตามขนาด - ตัวอักษรมักมีขนาดที่เหมาะสม (ไม่เล็กเกินไป ไม่ใหญ่เกินไป)
                    # ใช้ threshold สูงขึ้นเพื่อไม่ให้ตัวอักษรหาย
                    max_char_size = min(w, h) * 0.5  # เพิ่มจาก 30% เป็น 50% ของขนาดภาพ
                    if max(width, height) > max_char_size:
                        # ตรวจสอบว่าเป็น noise จริงๆ หรือไม่ (ขนาดใหญ่แต่ area น้อย = noise)
                        if areas_all[i-1] < max_char_size * 2:  # ถ้า area น้อยกว่าขนาดที่ควรจะเป็น
                            keep_mask[i-1] = False  # ลบ component ที่ใหญ่เกินไปแต่ area น้อย (อาจเป็น noise)
            
            areas_keep = areas_all[keep_mask]
            num_chars = int(keep_mask.sum())
            total_area = int(areas_keep.sum())
        else:
            num_chars, total_area = 0, 0
        
        # ใช้ text_edges_cleaned สำหรับการวิเคราะห์ต่อไป
        text_edges = text_edges_cleaned
        
        # Determine status - ใช้ threshold เดียวกันสำหรับทุกรส
        area_thresh = FADED_TEXT_CONFIG['AREA_THRESH']
        if area_thresh <= 0:
            status = "normal"  # 0 = ปิดเกณฑ์จางชั่วคราว (ถือว่าไม่จางเสมอ)
            print(f"🔍 AREA_THRESH = 0 — ปิดเกณฑ์จาง (ถือว่า normal)")
        else:
            print(f"🔍 ใช้เกณฑ์ area_thresh = {area_thresh} สำหรับทุกรส")
            status = "faded" if total_area < area_thresh else "normal"
        
        result.update({
            'status': status,
            'num_chars': num_chars,
            'total_area': total_area
        })
        
        if show_debug:
            debug_images = {
                'roi_show': roi_show,
                'gray': gray,
                'mask_circle': mask_circle,
                'roi_masked': roi_masked,
                'edges': edges_no_circle,  # แสดง edges ที่ลบเส้นวงกลมออกแล้ว
                'text_edges': text_edges  # แสดง text_edges ที่ทำความสะอาดแล้ว (กรอง noise แล้ว)
            }
            # Add enhanced image if available
            if img is not None and not np.array_equal(img, cap_image):
                debug_images['enhanced_image'] = img
            result['debug_images'] = debug_images
        
        print(f"Faded text detection - Components: {num_chars}, Area: {total_area}, Status: {status}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in faded text detection: {e}")
        return {'status': 'unknown', 'num_chars': 0, 'total_area': 0, 'error': str(e)}


def enhance_cap_sharpness(image):
    """
    Enhance sharpness of cap image to make text clearer
    
    Args:
        image: numpy array image (BGR or RGB)
        
    Returns:
        numpy array: Enhanced image with improved sharpness
    """
    if image is None:
        return None
    
    try:
        print("🔧 Enhancing cap image sharpness...")
        
        # Convert to RGB if needed (handle both BGR and RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it's BGR (OpenCV default) or RGB
            # We'll assume BGR and convert to RGB for processing
            image_rgb = cuda_cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            # Grayscale, convert to RGB
            image_rgb = cuda_cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Already RGB or unknown format, use as is
            image_rgb = image.copy()
        
        # Convert to grayscale for sharpening
        gray = cuda_cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Unsharp Mask (high-quality sharpening)
        # Create Gaussian blur
        blurred = cuda_gaussianBlur(gray, (0, 0), 2.0)
        # Apply unsharp mask: sharpened = original + (original - blurred) * amount
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        # Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Improves local contrast and sharpness
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        
        # Method 3: Additional sharpening using kernel
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened_final = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original sharpened with kernel sharpened for better result
        final_gray = cv2.addWeighted(enhanced, 0.7, sharpened_final, 0.3, 0)
        
        # Convert back to RGB
        final_image = cuda_cvtColor(final_gray, cv2.COLOR_GRAY2RGB)
        
        # Convert back to BGR if original was BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if original was likely BGR by comparing first pixel
            # For simplicity, we'll convert back to BGR
            final_image = cuda_cvtColor(final_image, cv2.COLOR_RGB2BGR)
        
        print("✅ Cap image sharpness enhanced")
        return final_image
        
    except Exception as e:
        print(f"❌ Error enhancing cap sharpness: {e}")
        import traceback
        traceback.print_exc()
        # Return original image if enhancement fails
        return image


# ย่อครอปก่อน OCR เพื่อเร่งความเร็ว (max ด้านไม่เกินค่านี้)
OCR_MAX_EDGE = 480

def perform_ocr_on_image(image):
    """
    Perform OCR on an image using EasyOCR
    
    Args:
        image: numpy array image
        
    Returns:
        List[Dict]: List of OCR results with text, confidence, and bbox
    """
    if bottle_reader is None:
        return []
    
    try:
        # Ensure image is in the correct format
        if image is None:
            print("⚠️ OCR Error: Image is None")
            return []
        
        # ย่อภาพถ้าใหญ่เกินไป เพื่อเร่ง OCR
        if max(image.shape[:2]) > OCR_MAX_EDGE:
            h, w = image.shape[:2]
            r = OCR_MAX_EDGE / max(h, w)
            new_w, new_h = int(round(w * r)), int(round(h * r))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB if needed (EasyOCR expects RGB)
        # OpenCV images are typically BGR, so convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB conversion (OpenCV default is BGR)
            image_rgb = cuda_cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            # Grayscale, convert to RGB
            image_rgb = cuda_cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Already RGB or unknown format, use as is
            image_rgb = image
        
        # Perform OCR with detail=1 to get bounding boxes and confidence
        results = bottle_reader.readtext(image_rgb, detail=1)
        ocr_results = []
        
        # Parse results: EasyOCR returns [(bbox, text, confidence), ...]
        for (bbox, text, confidence) in results:
            ocr_results.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        print(f"✅ OCR: Found {len(ocr_results)} text regions")
        if ocr_results:
            all_texts = " ".join([r['text'] for r in ocr_results])
            print(f"✅ OCR Text: {all_texts}")
        
        return ocr_results
        
    except Exception as e:
        print(f"❌ Error performing OCR: {e}")
        import traceback
        traceback.print_exc()
        # Try fallback: return empty list instead of empty string
        try:
            # Try with original image if RGB conversion failed
            if bottle_reader is not None:
                results = bottle_reader.readtext(image, detail=1)
                ocr_results = []
                for (bbox, text, confidence) in results:
                    ocr_results.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                return ocr_results
        except Exception as e2:
            print(f"❌ OCR Error (fallback): {e2}")
        return []


def aggregate_easyocr_confidences_from_type_crops(type_crops):
    """
    หลัง perform_ocr_on_image ใส่ใน crop['ocr_results'] แล้ว — รวมค่าความมั่นใจของ EasyOCR ต่อบรรทัดในทุก type crop

    Returns:
        dict พร้อม merge เข้า result: type_easyocr_mean_confidence, min, max, line_count
    """
    out = {
        "type_easyocr_mean_confidence": None,
        "type_easyocr_min_confidence": None,
        "type_easyocr_max_confidence": None,
        "type_easyocr_line_count": 0,
    }
    if not type_crops:
        return out
    all_confs = []
    for crop in type_crops:
        if not isinstance(crop, dict):
            continue
        for item in crop.get("ocr_results") or []:
            if isinstance(item, dict) and item.get("confidence") is not None:
                try:
                    all_confs.append(float(item["confidence"]))
                except (TypeError, ValueError):
                    pass
    if not all_confs:
        return out
    out["type_easyocr_line_count"] = len(all_confs)
    out["type_easyocr_mean_confidence"] = sum(all_confs) / len(all_confs)
    out["type_easyocr_min_confidence"] = min(all_confs)
    out["type_easyocr_max_confidence"] = max(all_confs)
    return out


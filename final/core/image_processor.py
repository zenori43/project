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
            img = cuda_cvtColor(cap_image, cv2.COLOR_GRAY2BGR)
        
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


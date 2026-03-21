# -*- coding: utf-8 -*-
"""
โปรแกรมตรวจจับตัวอักษรด้วย CRAFT
- ใช้ CRAFT จับตัวอักษร
- ใช้กรอบเล็ก (small boxes) แต่ละกรอบแยกกัน ไม่รวมกรอบ
"""

import sys
import os
import random
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# PyQt5 imports
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                  QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                                  QMessageBox, QGroupBox, QTextEdit, QGridLayout, QScrollArea, QCheckBox)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap, QFont
    PYQT5_AVAILABLE = True
except ImportError:
    print("⚠️ PyQt5 not available. Please install: pip install PyQt5")
    PYQT5_AVAILABLE = False

# Import from project
project_root = os.path.dirname(os.path.abspath(__file__))
final_dir = os.path.join(project_root, 'final')

# Add paths
if final_dir not in sys.path:
    sys.path.insert(0, final_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    os.chdir(final_dir)
    from libs.detection.capmodel import initialize_detector, CapDetector
    from libs.processing.rotationCRAFT import initialize_detector as initialize_craft_detector
    from config.settings import CAP_MODEL_PATH, CRAFT_MODEL_PATH, CRAFT_REFINER_PATH
    os.chdir(project_root)
    CAP_DETECTION_AVAILABLE = True
except ImportError as e:
    os.chdir(project_root)
    print(f"⚠️ Modules not available: {e}")
    CAP_DETECTION_AVAILABLE = False


def detect_text_with_craft(cap_image: np.ndarray, craft_detector, show_debug: bool = False) -> Dict:
    """
    ตรวจจับตัวอักษรด้วย CRAFT และสร้างกรอบเล็กแยกกัน
    1. ใช้ CRAFT จับตัวอักษร
    2. สร้างกรอบเล็ก (small boxes) สำหรับแต่ละ text box แยกกัน (ไม่รวมกรอบ)
    
    Args:
        cap_image: numpy array of the cropped cap image
        craft_detector: CRAFT detector instance
        show_debug: whether to return debug images
        
    Returns:
        dict with detection results (text_boxes: list of polygons from CRAFT)
    """
    if cap_image is None:
        return {'num_text_regions': 0, 'text_boxes': [], 'message': 'ไม่มีภาพ'}
    
    try:
        # Convert to RGB if needed (CRAFT expects RGB)
        if len(cap_image.shape) == 3:
            if cap_image.shape[2] == 3:
                # Assume BGR, convert to RGB
                image_rgb = cv2.cvtColor(cap_image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cap_image.copy()
        else:
            image_rgb = cv2.cvtColor(cap_image, cv2.COLOR_GRAY2RGB)
        
        # Step 1: ใช้ CRAFT จับตัวอักษร
        craft_result = craft_detector.detect_text_and_rotate(image_array=image_rgb)
        
        if not craft_result or 'text_boxes' not in craft_result or len(craft_result['text_boxes']) == 0:
            return {
                'num_text_regions': 0,
                'text_boxes': [],
                'message': 'CRAFT ไม่พบตัวอักษร'
            }
        
        text_boxes = craft_result['text_boxes']
        num_text_regions = len(text_boxes)
        h, w = cap_image.shape[:2]
        
        # ใช้กรอบ polygon จาก CRAFT โดยตรง ไม่ต้อง padding หรือแปลงเป็น bbox
        # Prepare result (ใช้ polygon จาก CRAFT โดยตรง)
        result = {
            'num_text_regions': num_text_regions,
            'craft_result': craft_result,
            'text_boxes': text_boxes  # ใช้ polygon โดยตรง
        }
        
        if show_debug:
            if len(cap_image.shape) == 3:
                original_bgr = cap_image.copy()
            else:
                original_bgr = cv2.cvtColor(cap_image, cv2.COLOR_GRAY2BGR)
            
            # Draw CRAFT boxes (polygon จาก CRAFT โดยตรง)
            craft_boxes_image = original_bgr.copy()
            for box in text_boxes:
                box_i = np.array(box, dtype=np.int32)
                cv2.polylines(craft_boxes_image, [box_i], True, (0, 255, 0), 2)
            
            # Mask ของ polygon ทั้งหมด (สีขาว = บริเวณใน polygon)
            text_boxes_mask = np.zeros((h, w), dtype=np.uint8)
            for box in text_boxes:
                box_i = np.array(box, dtype=np.int32)
                cv2.fillPoly(text_boxes_mask, [box_i], 255)
            text_boxes_mask_img = cv2.cvtColor(text_boxes_mask, cv2.COLOR_GRAY2BGR)
            
            # Cropped region แสดงเป็นตัวอย่าง polygon แรก (ใช้ bounding box ของ polygon)
            if len(text_boxes) > 0:
                first_box = np.array(text_boxes[0], dtype=np.int32)
                sx = int(np.clip(np.min(first_box[:, 0]), 0, w - 1))
                sy = int(np.clip(np.min(first_box[:, 1]), 0, h - 1))
                sxx = int(np.clip(np.max(first_box[:, 0]) + 1, 1, w))
                syy = int(np.clip(np.max(first_box[:, 1]) + 1, 1, h))
                cropped_region = cap_image[sy:syy, sx:sxx].copy() if (sxx > sx and syy > sy) else np.zeros((1, 1, 3), dtype=np.uint8)
            else:
                cropped_region = np.zeros((1, 1, 3), dtype=np.uint8)
            
            debug_images = {
                'original': original_bgr,
                'craft_boxes': craft_boxes_image,
                'text_boxes_mask': text_boxes_mask_img,
                'cropped_region': cropped_region
            }
            result['debug_images'] = debug_images
        
        print(f"CRAFT Detection - Text Regions: {num_text_regions}, Polygons: {len(text_boxes)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in CRAFT detection: {e}")
        import traceback
        traceback.print_exc()
        return {'num_text_regions': 0, 'text_boxes': [], 'error': str(e)}


def apply_alpha_blending_darken(foreground: np.ndarray, darken_factor: float = 0.3) -> np.ndarray:
    """
    Alpha Blending Darken - ทำให้ตัวอักษรเข้มขึ้น
    I_out = I_foreground * (1 - darken_factor) + dark_color * darken_factor
    
    Args:
        foreground: ภาพตัวอักษร (BGR)
        darken_factor: ปัจจัยการทำให้เข้ม (0.0-1.0, สูง = เข้มขึ้น)
    
    Returns:
        ภาพที่เข้มขึ้น (BGR)
    """
    dark_color = np.array([0, 0, 0], dtype=np.float32)  # สีดำ
    foreground_f = foreground.astype(np.float32)
    darkened = foreground_f * (1 - darken_factor) + dark_color * darken_factor
    return np.clip(darkened, 0, 255).astype(np.uint8)


def apply_alpha_blending(foreground: np.ndarray, background: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Alpha Blending - การผสมค่าความโปร่งใส
    I_out = α * I_foreground + (1 - α) * I_background
    
    Args:
        foreground: ภาพตัวอักษร (BGR)
        background: ภาพพื้นหลัง (BGR)
        alpha: ค่าความโปร่งใส (0.0-1.0)
    
    Returns:
        ภาพที่ผสมแล้ว (BGR)
    """
    foreground = foreground.astype(np.float32)
    background = background.astype(np.float32)
    
    # Resize background to match foreground if needed
    if background.shape[:2] != foreground.shape[:2]:
        background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
    
    blended = alpha * foreground + (1 - alpha) * background
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_sharpening(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    Sharpening - ทำให้ตัวอักษรคมชัดขึ้น (เข้มขึ้น)
    ใช้ Unsharp Mask technique
    
    Args:
        image: ภาพต้นฉบับ (BGR)
        strength: ความแรงของการ sharpen (1.0-3.0)
    
    Returns:
        ภาพที่คมชัดขึ้น (BGR)
    """
    # สร้าง Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
    # Unsharp mask: original - blurred
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 15, sigma: float = 5.0) -> np.ndarray:
    """
    Gaussian Blurring - การทำให้เบลอ
    ใช้ Kernel คำนวณค่าเฉลี่ยของพิกเซลรอบข้าง
    
    Args:
        image: ภาพที่ต้องการทำให้เบลอ (BGR)
        kernel_size: ขนาดของ kernel (ต้องเป็นเลขคี่)
        sigma: ค่า standard deviation ของ Gaussian
    
    Returns:
        ภาพที่เบลอแล้ว (BGR)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred


def apply_inpainting(image: np.ndarray, mask: np.ndarray, method: int = cv2.INPAINT_TELEA) -> np.ndarray:
    """
    Inpainting - การลบและเติมเต็ม
    วิเคราะห์พิกเซลรอบๆ แล้วนำสี/Texture มาเขียนทับ
    
    Args:
        image: ภาพต้นฉบับ (BGR)
        mask: หน้ากากที่ระบุบริเวณที่ต้องการลบ (grayscale, 255 = บริเวณที่ลบ)
        method: วิธีการ inpainting (INPAINT_TELEA หรือ INPAINT_NS)
    
    Returns:
        ภาพที่ลบตัวอักษรแล้ว (BGR)
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    inpainted = cv2.inpaint(image, mask, 3, method)
    return inpainted


def adjust_contrast_brightness_darken(image: np.ndarray, contrast: float = 1.5, brightness: int = -20) -> np.ndarray:
    """
    Adjusting Contrast & Brightness Darken - ทำให้ตัวอักษรเข้มขึ้น
    เพิ่ม Contrast และลด Brightness
    
    Args:
        image: ภาพต้นฉบับ (BGR)
        contrast: ค่า contrast (1.0-2.5, สูง = contrast มาก = เข้มขึ้น)
        brightness: ค่า brightness ที่ลด (-50 ถึง 0)
    
    Returns:
        ภาพที่เข้มขึ้น (BGR)
    """
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted


def adjust_contrast_brightness(image: np.ndarray, contrast: float = 0.3, brightness: int = 30) -> np.ndarray:
    """
    Adjusting Contrast & Brightness - การปรับความต่างสี
    ลดค่า Contrast หรือปรับค่า Brightness ให้ใกล้เคียงกับพื้นหลัง
    
    Args:
        image: ภาพต้นฉบับ (BGR)
        contrast: ค่า contrast (0.0-1.0, ต่ำ = contrast น้อย)
        brightness: ค่า brightness ที่เพิ่ม/ลด (-100 ถึง 100)
    
    Returns:
        ภาพที่ปรับแล้ว (BGR)
    """
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted


def apply_gradient_masking_darken(image: np.ndarray, direction: str = 'horizontal',
                                 darken_start: float = 0.0, darken_end: float = 0.5) -> np.ndarray:
    """
    Gradient Masking Darken - ทำให้ตัวอักษรเข้มขึ้นแบบไล่เฉด
    สร้าง Mask แบบไล่เฉดแล้วทำให้เข้มขึ้นตาม mask
    
    Args:
        image: ภาพต้นฉบับ (BGR)
        direction: ทิศทางของ gradient ('horizontal' หรือ 'vertical')
        darken_start: ปัจจัยการทำให้เข้มเริ่มต้น (0.0 = ไม่เข้ม)
        darken_end: ปัจจัยการทำให้เข้มสิ้นสุด (0.5 = เข้มขึ้น 50%)
    
    Returns:
        ภาพที่เข้มขึ้นแบบไล่เฉด (BGR)
    """
    h, w = image.shape[:2]
    
    # สร้าง gradient mask สำหรับ darkening
    if direction == 'horizontal':
        mask = np.linspace(darken_start, darken_end, w)
        mask = np.tile(mask, (h, 1))
    else:  # vertical
        mask = np.linspace(darken_start, darken_end, h)
        mask = np.tile(mask.reshape(-1, 1), (1, w))
    
    # แปลงเป็น 3 channels และทำให้เข้มขึ้นตาม mask
    mask_3d = np.stack([mask] * 3, axis=2)
    # Darken: multiply by (1 - mask_factor), where mask_factor is how much to darken
    result = image.astype(np.float32) * (1 - mask_3d)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_gradient_masking(image: np.ndarray, direction: str = 'horizontal', 
                          start_opacity: float = 1.0, end_opacity: float = 0.0) -> np.ndarray:
    """
    Masking with Gradients - ใช้หน้ากากไล่เฉด
    สร้าง Mask แบบไล่เฉดแล้วนำไปคูณกับภาพ
    
    Args:
        image: ภาพต้นฉบับ (BGR)
        direction: ทิศทางของ gradient ('horizontal' หรือ 'vertical')
        start_opacity: ความโปร่งใสเริ่มต้น (1.0 = ไม่โปร่งใส)
        end_opacity: ความโปร่งใสสิ้นสุด (0.0 = โปร่งใสเต็มที่)
    
    Returns:
        ภาพที่ไล่เฉดแล้ว (BGR)
    """
    h, w = image.shape[:2]
    
    # สร้าง gradient mask
    if direction == 'horizontal':
        # จากซ้ายไปขวา
        mask = np.linspace(start_opacity, end_opacity, w)
        mask = np.tile(mask, (h, 1))
    else:  # vertical
        # จากบนลงล่าง
        mask = np.linspace(start_opacity, end_opacity, h)
        mask = np.tile(mask.reshape(-1, 1), (1, w))
    
    # แปลงเป็น 3 channels
    mask_3d = np.stack([mask] * 3, axis=2)
    
    # คำนวณค่าเฉลี่ยของพื้นหลัง (ใช้ขอบของภาพ)
    if direction == 'horizontal':
        bg_color = np.mean(image[:, :w//10], axis=(0, 1))  # สีจากขอบซ้าย
    else:
        bg_color = np.mean(image[:h//10, :], axis=(0, 1))  # สีจากขอบบน
    
    # ผสมภาพกับพื้นหลังตาม mask
    result = (image.astype(np.float32) * mask_3d + 
              bg_color.astype(np.float32) * (1 - mask_3d))
    
    return np.clip(result, 0, 255).astype(np.uint8)


def _get_background_color_around_box(image: np.ndarray, min_x: int, min_y: int, max_x: int, max_y: int, margin: int = 15) -> np.ndarray:
    """คำนวณสีเฉลี่ยของพื้นหลังรอบๆ กรอบ (ไม่รวมบริเวณในกรอบ)"""
    h, w = image.shape[:2]
    # บริเวณขยายออกมานอกกรอบ
    y1 = max(0, min_y - margin)
    y2 = min(h, max_y + margin)
    x1 = max(0, min_x - margin)
    x2 = min(w, max_x + margin)
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return np.mean(image, axis=(0, 1))
    return np.mean(region, axis=(0, 1))


def apply_fade_techniques_craft_polygons(cap_image: np.ndarray, text_boxes: List[np.ndarray],
                                         select_ratio: float = 0.5, strong_fade: bool = False) -> Dict:
    """
    ประมวลผลเฉพาะบาง polygon จาก CRAFT (สุ่มเลือก) ด้วยเทคนิคต่างๆ
    ใช้ polygon จาก CRAFT โดยตรง ไม่ต้อง padding หรือแปลงเป็น bbox
    
    Args:
        cap_image: ภาพฝาทั้งหมด (BGR)
        text_boxes: รายการ polygon จาก CRAFT (แต่ละ polygon เป็น array ของ points)
        select_ratio: สัดส่วนกรอบที่สุ่มเลือก (0.0-1.0) เช่น 0.5 = ประมาณครึ่งหนึ่ง
        strong_fade: ถ้า True จะทำให้ตัวอักษรจางมากขึ้น (ใช้พารามิเตอร์รุนแรงขึ้น)
    
    Returns:
        dict with processed full cap images (แต่ละเทคนิคแก้เฉพาะ polygon ที่ถูกสุ่มเลือก)
    """
    results = {}
    if text_boxes is None or len(text_boxes) == 0:
        return results
    
    n = len(text_boxes)
    # สุ่มเลือกบาง polygon: ลดจำนวนลง (ประมาณ 20-30% แทน 50%)
    # อย่างน้อย 1 polygon แต่ไม่เกิน 30% ของทั้งหมด
    adjusted_ratio = min(select_ratio, 0.3)  # จำกัดสูงสุดที่ 30%
    k = max(1, min(n, int(round(n * adjusted_ratio)) or 1))
    k = min(k, n)
    selected_indices = sorted(random.sample(range(n), k))  # Sort เพื่อให้ง่ายต่อการ debug
    
    img_h, img_w = cap_image.shape[:2]
    
    def get_polygon_bbox(polygon: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of polygon"""
        pts = np.array(polygon, dtype=np.float32)
        sx = int(np.clip(np.min(pts[:, 0]), 0, img_w - 1))
        sy = int(np.clip(np.min(pts[:, 1]), 0, img_h - 1))
        sxx = int(np.clip(np.max(pts[:, 0]) + 1, 1, img_w))
        syy = int(np.clip(np.max(pts[:, 1]) + 1, 1, img_h))
        return sx, sy, sxx, syy
    
    def create_polygon_mask(polygon: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Create mask for polygon within its bounding box"""
        sx, sy, sxx, syy = bbox
        mask = np.zeros((syy - sy, sxx - sx), dtype=np.uint8)
        # Adjust polygon coordinates relative to bbox
        adjusted_poly = np.array(polygon, dtype=np.int32) - [sx, sy]
        cv2.fillPoly(mask, [adjusted_poly], 255)
        return mask
    
    def paste_processed_with_mask(out: np.ndarray, polygon: np.ndarray, processed_patch: np.ndarray) -> None:
        """Paste processed patch back using polygon mask (only within polygon)"""
        sx, sy, sxx, syy = get_polygon_bbox(polygon)
        
        if sxx <= sx or syy <= sy:
            return
        
        # Create mask for this polygon
        polygon_mask = create_polygon_mask(polygon, (sx, sy, sxx, syy))
        
        # Resize processed patch to match bbox size
        ph, pw = processed_patch.shape[:2]
        bbox_h, bbox_w = syy - sy, sxx - sx
        if bbox_w != pw or bbox_h != ph:
            processed_patch = cv2.resize(processed_patch, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
        
        # Paste only within polygon mask
        mask_3d = np.stack([polygon_mask] * 3, axis=2) / 255.0
        roi = out[sy:syy, sx:sxx]
        roi[:] = (roi.astype(np.float32) * (1 - mask_3d) + 
                  processed_patch.astype(np.float32) * mask_3d).astype(np.uint8)
    
    # พารามิเตอร์ตามความรุนแรง: ปกติ vs จางมากขึ้น (รุนแรงขึ้น)
    if strong_fade:
        alpha_val = 0.12           # จางมาก (เหลือแค่ 12% ตัวอักษร)
        grad_start, grad_end = 0.6, 0.0           # ไล่จางเร็ว
    else:
        alpha_val = 0.25           # จางปานกลาง
        grad_start, grad_end = 1.0, 0.0
    
    try:
        # 1. Alpha Blending (เฉพาะ polygon ที่เลือก) - จางลงด้วยการผสมกับพื้นหลัง
        out_alpha = cap_image.copy()
        for i in selected_indices:
            polygon = text_boxes[i]
            sx, sy, sxx, syy = get_polygon_bbox(polygon)
            if sxx <= sx or syy <= sy:
                continue
            patch = cap_image[sy:syy, sx:sxx].copy()
            bg_color = _get_background_color_around_box(cap_image, sx, sy, sxx, syy)
            bg_patch = np.full((patch.shape[0], patch.shape[1], 3), bg_color, dtype=np.uint8)
            processed = apply_alpha_blending(patch, bg_patch, alpha=alpha_val)
            paste_processed_with_mask(out_alpha, polygon, processed)
        results['alpha_blending'] = out_alpha
        
        # 2. Inpainting
        out_inpaint = cap_image.copy()
        for i in selected_indices:
            polygon = text_boxes[i]
            sx, sy, sxx, syy = get_polygon_bbox(polygon)
            if sxx <= sx or syy <= sy:
                continue
            patch = cap_image[sy:syy, sx:sxx].copy()
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            processed = apply_inpainting(patch, mask, method=cv2.INPAINT_TELEA)
            paste_processed_with_mask(out_inpaint, polygon, processed)
        results['inpainting'] = out_inpaint
        
        # 3. Gradient horizontal - ไล่จาง
        out_gh = cap_image.copy()
        for i in selected_indices:
            polygon = text_boxes[i]
            sx, sy, sxx, syy = get_polygon_bbox(polygon)
            if sxx <= sx or syy <= sy:
                continue
            patch = cap_image[sy:syy, sx:sxx].copy()
            processed = apply_gradient_masking(patch, direction='horizontal', start_opacity=grad_start, end_opacity=grad_end)
            paste_processed_with_mask(out_gh, polygon, processed)
        results['gradient_horizontal'] = out_gh
        
        # 4. Gradient vertical - ไล่จาง
        out_gv = cap_image.copy()
        for i in selected_indices:
            polygon = text_boxes[i]
            sx, sy, sxx, syy = get_polygon_bbox(polygon)
            if sxx <= sx or syy <= sy:
                continue
            patch = cap_image[sy:syy, sx:sxx].copy()
            processed = apply_gradient_masking(patch, direction='vertical', start_opacity=grad_start, end_opacity=grad_end)
            paste_processed_with_mask(out_gv, polygon, processed)
        results['gradient_vertical'] = out_gv
        
        print(f"Fade techniques: สุ่มเลือก {k}/{n} polygon จาก CRAFT (indices: {selected_indices})")
        
        # เพิ่มข้อมูล polygon ที่เลือกในผลลัพธ์
        results['selected_indices'] = selected_indices
        results['selected_polygons'] = [text_boxes[i] for i in selected_indices]
        
    except Exception as e:
        print(f"❌ Error applying fade techniques (small boxes): {e}")
        import traceback
        traceback.print_exc()
    
    return results


class CapDetectionThread(QThread):
    """Thread for processing cap detection"""
    result_ready = pyqtSignal(dict)
    progress_updated = pyqtSignal(str)
    
    def __init__(self, image_path: str, cap_detector: CapDetector, craft_detector, strong_fade: bool = False):
        super().__init__()
        self.image_path = image_path
        self.cap_detector = cap_detector
        self.craft_detector = craft_detector
        self.strong_fade = strong_fade
    
    def run(self):
        try:
            self.progress_updated.emit("กำลังตรวจจับฝา...")
            
            # Step 1: Detect caps
            cap_result = self.cap_detector.detect_caps(self.image_path)
            
            if cap_result['total_detections'] == 0:
                self.result_ready.emit({
                    'error': 'no_caps',
                    'message': 'ไม่พบฝาในภาพ'
                })
                return
            
            self.progress_updated.emit("กำลังตัดภาพฝา...")
            
            # Step 2: Crop detected caps
            image = cv2.imread(self.image_path)
            cropped_images = self.cap_detector.crop_detections(image=image, result=cap_result, margin=20)
            
            if not cropped_images:
                self.result_ready.emit({
                    'error': 'no_crops',
                    'message': 'ไม่สามารถตัดภาพฝาได้'
                })
                return
            
            # Step 3: Select bottommost cap
            detections = cap_result['detections']
            if len(detections) > 1:
                max_y = -1
                bottommost_index = 0
                for i, det in enumerate(detections):
                    bbox = det['bbox']
                    y_center = (bbox[1] + bbox[3]) / 2
                    if y_center > max_y:
                        max_y = y_center
                        bottommost_index = i
            else:
                bottommost_index = 0
            
            self.progress_updated.emit("กำลังตรวจจับตัวอักษรด้วย CRAFT...")
            
            # Step 4: Detect text with CRAFT
            bottommost_cap = cropped_images[bottommost_index]
            craft_result = detect_text_with_craft(bottommost_cap, self.craft_detector, show_debug=True)
            
            # Step 5: Apply fade techniques เฉพาะบาง polygon จาก CRAFT (สุ่มเลือก)
            fade_results = {}
            text_boxes = craft_result.get('text_boxes', [])
            if text_boxes is not None and len(text_boxes) > 0:
                mode_text = "จางมากขึ้น (รุนแรง)" if self.strong_fade else "จางปานกลาง"
                self.progress_updated.emit(f"กำลังประมวลผลด้วยเทคนิคต่างๆ ({mode_text}, สุ่มบาง polygon)...")
                fade_results = apply_fade_techniques_craft_polygons(bottommost_cap, text_boxes, select_ratio=0.25, strong_fade=self.strong_fade)
            
            self.result_ready.emit({
                'cap_result': cap_result,
                'cropped_images': cropped_images,
                'bottommost_index': bottommost_index,
                'craft_result': craft_result,
                'fade_results': fade_results
            })
            
        except Exception as e:
            self.result_ready.emit({
                'error': 'processing_error',
                'message': f'เกิดข้อผิดพลาด: {str(e)}'
            })


class BatchProcessingThread(QThread):
    """Thread for processing multiple images in batch"""
    result_ready = pyqtSignal(dict)  # สำหรับแต่ละไฟล์ที่เสร็จ
    progress_updated = pyqtSignal(str)
    finished = pyqtSignal(int, int)  # (success_count, total_count)
    
    def __init__(self, image_files: List[str], output_folder: str, 
                 cap_detector: CapDetector, craft_detector, strong_fade: bool = False):
        super().__init__()
        self.image_files = image_files
        self.output_folder = output_folder
        self.cap_detector = cap_detector
        self.craft_detector = craft_detector
        self.strong_fade = strong_fade
    
    def run(self):
        """ประมวลผลทุกไฟล์และบันทึกผลลัพธ์"""
        from datetime import datetime
        success_count = 0
        
        for idx, image_path in enumerate(self.image_files):
            try:
                self.progress_updated.emit(f"กำลังประมวลผลไฟล์ {idx+1}/{len(self.image_files)}: {os.path.basename(image_path)}")
                
                # ประมวลผลแต่ละไฟล์
                result = self.process_single_image(image_path)
                
                if 'error' not in result:
                    # บันทึกผลลัพธ์
                    self.save_results_for_image(image_path, result, idx)
                    success_count += 1
                    self.result_ready.emit({
                        'file_path': image_path,
                        'index': idx + 1,
                        'total': len(self.image_files),
                        'success': True
                    })
                else:
                    self.result_ready.emit({
                        'file_path': image_path,
                        'index': idx + 1,
                        'total': len(self.image_files),
                        'success': False,
                        'error': result.get('error'),
                        'message': result.get('message')
                    })
                    
            except Exception as e:
                self.result_ready.emit({
                    'file_path': image_path,
                    'index': idx + 1,
                    'total': len(self.image_files),
                    'success': False,
                    'error': 'exception',
                    'message': str(e)
                })
        
        self.finished.emit(success_count, len(self.image_files))
    
    def process_single_image(self, image_path: str) -> dict:
        """ประมวลผลไฟล์เดียว (ใช้โค้ดเดียวกับ CapDetectionThread)"""
        try:
            # Step 1: Detect caps
            cap_result = self.cap_detector.detect_caps(image_path)
            
            if cap_result['total_detections'] == 0:
                return {'error': 'no_caps', 'message': 'ไม่พบฝาในภาพ'}
            
            # Step 2: Crop detected caps
            image = cv2.imread(image_path)
            cropped_images = self.cap_detector.crop_detections(image=image, result=cap_result, margin=20)
            
            if not cropped_images:
                return {'error': 'no_crops', 'message': 'ไม่สามารถตัดภาพฝาได้'}
            
            # Step 3: Select bottommost cap
            detections = cap_result['detections']
            if len(detections) > 1:
                max_y = -1
                bottommost_index = 0
                for i, det in enumerate(detections):
                    bbox = det['bbox']
                    y_center = (bbox[1] + bbox[3]) / 2
                    if y_center > max_y:
                        max_y = y_center
                        bottommost_index = i
            else:
                bottommost_index = 0
            
            # Step 4: Detect text with CRAFT
            bottommost_cap = cropped_images[bottommost_index]
            craft_result = detect_text_with_craft(bottommost_cap, self.craft_detector, show_debug=False)
            
            # Step 5: Apply fade techniques
            fade_results = {}
            text_boxes = craft_result.get('text_boxes', [])
            if text_boxes is not None and len(text_boxes) > 0:
                fade_results = apply_fade_techniques_craft_polygons(
                    bottommost_cap, text_boxes, select_ratio=0.25, strong_fade=self.strong_fade
                )
            
            return {
                'cap_result': cap_result,
                'cropped_images': cropped_images,
                'bottommost_index': bottommost_index,
                'craft_result': craft_result,
                'fade_results': fade_results
            }
            
        except Exception as e:
            return {'error': 'processing_error', 'message': f'เกิดข้อผิดพลาด: {str(e)}'}
    
    def save_results_for_image(self, image_path: str, result: dict, index: int):
        """บันทึกผลลัพธ์สำหรับแต่ละภาพ"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # เพิ่ม index เพื่อไม่ให้ชื่อไฟล์ซ้ำ
        base_name = f"{index+1:03d}_{base_name}"
        
        # บันทึกภาพต้นฉบับ (cap)
        cropped_images = result.get('cropped_images', [])
        bottommost_index = result.get('bottommost_index', 0)
        if cropped_images and bottommost_index < len(cropped_images):
            cap_image = cropped_images[bottommost_index]
            original_path = os.path.join(self.output_folder, f"{base_name}_original.png")
            cv2.imwrite(original_path, cap_image)
        
        # บันทึกภาพผลลัพธ์จากเทคนิคต่างๆ
        fade_results = result.get('fade_results', {})
        technique_names = {
            'alpha_blending': '01_AlphaBlending',
            'inpainting': '03_Inpainting',
            'gradient_horizontal': '05_GradientHorizontal',
            'gradient_vertical': '06_GradientVertical'
        }
        
        for key, name_prefix in technique_names.items():
            if key in fade_results:
                image = fade_results[key]
                save_path = os.path.join(self.output_folder, f"{base_name}_{name_prefix}.png")
                cv2.imwrite(save_path, image)


class CRAFTDetectorGUI(QMainWindow):
    """GUI สำหรับตรวจจับตัวอักษรด้วย CRAFT"""
    
    def __init__(self):
        super().__init__()
        self.cap_detector = None
        self.craft_detector = None
        self.current_image_path = None
        self.current_result = None
        self.image_files = []  # รายการไฟล์สำหรับประมวลผลหลายไฟล์
        self.output_folder = None  # โฟลเดอร์สำหรับบันทึกผลลัพธ์
        self.init_ui()
        self.load_models()
    
    def init_ui(self):
        self.setWindowTitle('CRAFT Text Detector - ตรวจจับตัวอักษรด้วย CRAFT')
        self.setGeometry(100, 100, 1600, 1000)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_group = QGroupBox("ควบคุม")
        control_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("📁 เลือกภาพ")
        self.btn_load.clicked.connect(self.load_image)
        control_layout.addWidget(self.btn_load)
        
        self.btn_load_folder = QPushButton("📂 เลือกโฟลเดอร์")
        self.btn_load_folder.clicked.connect(self.load_folder)
        control_layout.addWidget(self.btn_load_folder)
        
        self.btn_process = QPushButton("🔍 ตรวจจับตัวอักษรด้วย CRAFT")
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setEnabled(False)
        control_layout.addWidget(self.btn_process)
        
        # Strong fade checkbox - ทำให้จางมากขึ้น (รุนแรงขึ้น)
        self.strong_fade_checkbox = QCheckBox("ทำให้จางมากขึ้น (รุนแรงขึ้น)")
        self.strong_fade_checkbox.setToolTip("เลือกเพื่อใช้พารามิเตอร์ที่รุนแรงขึ้น ตัวอักษรในกรอบจะจางลงมากกว่าเดิม")
        self.strong_fade_checkbox.setChecked(False)
        control_layout.addWidget(self.strong_fade_checkbox)
        
        # Save button
        self.btn_save = QPushButton("💾 บันทึกภาพผลลัพธ์")
        self.btn_save.clicked.connect(self.save_results)
        self.btn_save.setEnabled(False)
        control_layout.addWidget(self.btn_save)
        
        self.status_label = QLabel("พร้อมใช้งาน")
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Image display area
        image_group = QGroupBox("ภาพผลลัพธ์")
        image_layout = QHBoxLayout()
        
        # Original image container
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_layout.setContentsMargins(5, 5, 5, 5)
        original_title = QLabel("📷 ภาพต้นฉบับ (Original Image)")
        original_title.setAlignment(Qt.AlignCenter)
        original_title.setStyleSheet("font-weight: bold; font-size: 11pt; color: #1976D2; padding: 5px;")
        original_layout.addWidget(original_title)
        self.original_label = QLabel("ยังไม่มีภาพ")
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        original_layout.addWidget(self.original_label)
        image_layout.addWidget(original_container)
        
        # Result image container
        result_container = QWidget()
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(5, 5, 5, 5)
        result_title = QLabel("🎯 ภาพฝาที่เลือก (Selected Cap)")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("font-weight: bold; font-size: 11pt; color: #388E3C; padding: 5px;")
        result_layout.addWidget(result_title)
        self.result_label = QLabel("ยังไม่มีผลลัพธ์")
        self.result_label.setMinimumSize(400, 400)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        result_layout.addWidget(self.result_label)
        image_layout.addWidget(result_container)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # Debug images
        debug_group = QGroupBox("Debug Images - CRAFT Detection")
        debug_layout = QHBoxLayout()
        
        self.debug_labels = []
        debug_info = [
            ('Original', 'ภาพต้นฉบับ'),
            ('CRAFT Polygons', 'แสดง Polygon ทั้งหมด\n(เขียว = ไม่เลือก, แดง = เลือก)'),
            ('Mask Polygons', 'Mask ของ Polygon ทั้งหมด\n(ขาว = บริเวณ Polygon)'),
            ('ตัวอย่าง Polygon', 'ตัวอย่าง Polygon แรก\n(แสดงเฉพาะ 1 Polygon)')
        ]
        for title, description in debug_info:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(3, 3, 3, 3)
            container_layout.setSpacing(3)
            
            # Title
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 10pt; color: #1976D2;")
            container_layout.addWidget(title_label)
            
            # Image
            image_label = QLabel("(ยังไม่มี)")
            image_label.setMinimumSize(200, 180)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
            container_layout.addWidget(image_label)
            
            # Description
            desc_label = QLabel(description)
            desc_label.setAlignment(Qt.AlignCenter)
            desc_label.setStyleSheet("font-size: 8pt; color: #666;")
            desc_label.setWordWrap(True)
            container_layout.addWidget(desc_label)
            
            debug_layout.addWidget(container)
            self.debug_labels.append(image_label)
        
        debug_group.setLayout(debug_layout)
        layout.addWidget(debug_group)
        
        # Fade techniques results - Grid layout for multiple techniques
        fade_group = QGroupBox("ผลลัพธ์เทคนิคการทำให้ตัวอักษรจาง")
        fade_scroll = QScrollArea()
        fade_scroll_widget = QWidget()
        fade_grid = QGridLayout(fade_scroll_widget)
        
        # Technique names and their display labels
        self.fade_labels = {}
        self.fade_title_labels = {}
        technique_names = {
            'alpha_blending': ('1. Alpha Blending', 'ผสมกับพื้นหลัง\n(รุนแรง = จางมาก)'),
            'inpainting': ('3. Inpainting', 'การลบและเติมเต็ม\nใช้ Texture จากพื้นหลังมาเขียนทับ'),
            'gradient_horizontal': ('5. Gradient Masking', 'ไล่จางจากซ้าย→ขวา\n(รุนแรง = ไล่จางเร็ว)'),
            'gradient_vertical': ('6. Gradient Masking', 'ไล่จางจากบน→ล่าง\n(รุนแรง = ไล่จางเร็ว)')
        }
        
        row = 0
        col = 0
        for key, (title, description) in technique_names.items():
            # สร้าง container สำหรับแต่ละเทคนิค
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(5, 5, 5, 5)
            container_layout.setSpacing(5)
            
            # Title label
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #2E7D32; padding: 3px;")
            container_layout.addWidget(title_label)
            
            # Image label
            image_label = QLabel("(ยังไม่มีภาพ)")
            image_label.setMinimumSize(250, 200)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("border: 2px solid #4CAF50; background-color: #f0f0f0; padding: 5px;")
            container_layout.addWidget(image_label)
            
            # Description label
            desc_label = QLabel(description)
            desc_label.setAlignment(Qt.AlignCenter)
            desc_label.setStyleSheet("font-size: 9pt; color: #555; padding: 3px;")
            desc_label.setWordWrap(True)
            container_layout.addWidget(desc_label)
            
            fade_grid.addWidget(container, row, col)
            self.fade_labels[key] = image_label
            self.fade_title_labels[key] = title_label
            
            col += 1
            if col >= 3:  # 3 columns per row
                col = 0
                row += 1
        
        fade_scroll.setWidget(fade_scroll_widget)
        fade_scroll.setWidgetResizable(True)
        fade_scroll.setMinimumHeight(300)
        fade_group_layout = QVBoxLayout()
        fade_group_layout.addWidget(fade_scroll)
        fade_group.setLayout(fade_group_layout)
        layout.addWidget(fade_group)
        
        # Results text
        results_group = QGroupBox("ผลลัพธ์")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
    
    def load_models(self):
        """โหลด models"""
        if not CAP_DETECTION_AVAILABLE:
            QMessageBox.warning(self, "ข้อผิดพลาด", "ไม่สามารถโหลด modules ได้")
            return
        
        try:
            # Load cap detector
            model_path = os.path.join(final_dir, 'models', 'detection', 'cap.pt')
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "ข้อผิดพลาด", f"ไม่พบไฟล์โมเดล: {model_path}")
                self.status_label.setText("❌ ไม่พบไฟล์โมเดล")
                return
            
            self.cap_detector = initialize_detector(model_path)
            
            # Load CRAFT detector
            craft_model_path = os.path.join(project_root, 'CRAFT-pytorch', 'craft_mlt_25k.pth')
            craft_refiner_path = os.path.join(project_root, 'CRAFT-pytorch', 'craft_refiner_CTW1500.pth')
            
            if not os.path.exists(craft_model_path):
                QMessageBox.warning(self, "ข้อผิดพลาด", f"ไม่พบไฟล์ CRAFT model: {craft_model_path}")
                self.status_label.setText("❌ ไม่พบไฟล์ CRAFT model")
                return
            
            self.craft_detector = initialize_craft_detector(craft_model_path, craft_refiner_path)
            
            self.status_label.setText("✅ โหลดโมเดลสำเร็จ - พร้อมใช้งาน")
        except Exception as e:
            QMessageBox.warning(self, "ข้อผิดพลาด", f"ไม่สามารถโหลดโมเดลได้: {str(e)}")
            self.status_label.setText("❌ โหลดโมเดลล้มเหลว")
    
    def load_image(self):
        """เลือกภาพ (ไฟล์เดียว)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "เลือกภาพ", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_files = [file_path]  # เก็บเป็นรายการเดียว
            self.output_folder = None  # Reset output folder (โหมดไฟล์เดียว)
            self.display_image(file_path, self.original_label)
            self.btn_process.setEnabled(True)
            self.status_label.setText(f"✅ โหลดภาพ: {os.path.basename(file_path)}")
    
    def load_folder(self):
        """เลือกโฟลเดอร์และโฟลเดอร์สำหรับบันทึกผลลัพธ์"""
        folder_path = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ที่มีภาพ")
        if not folder_path:
            return
        
        # เลือกโฟลเดอร์สำหรับบันทึกผลลัพธ์
        output_folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์สำหรับบันทึกผลลัพธ์")
        if not output_folder:
            return
        
        # หาไฟล์ภาพทั้งหมดใน folder
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']
        image_files = []
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            QMessageBox.warning(self, "ข้อผิดพลาด", "ไม่พบไฟล์ภาพในโฟลเดอร์ที่เลือก")
            return
        
        self.image_files = sorted(image_files)
        self.output_folder = output_folder
        
        # แสดงภาพแรก
        if self.image_files:
            self.current_image_path = self.image_files[0]
            self.display_image(self.current_image_path, self.original_label)
        
        self.btn_process.setEnabled(True)
        self.status_label.setText(f"✅ โหลด {len(self.image_files)} ไฟล์ | บันทึกที่: {os.path.basename(output_folder)}")
    
    def display_image(self, image_path: str, label: QLabel):
        """แสดงภาพใน label"""
        image = cv2.imread(image_path)
        if image is None:
            return
        
        h, w = image.shape[:2]
        max_size = 400
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)
    
    def display_numpy_image(self, image: np.ndarray, label: QLabel):
        """แสดง numpy image ใน label"""
        if image is None or image.size == 0:
            return
        
        h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
        max_size = 250
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)
    
    def process_image(self):
        """ประมวลผลภาพ (รองรับทั้งไฟล์เดียวและหลายไฟล์)"""
        if not self.image_files or not self.cap_detector or not self.craft_detector:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาเลือกภาพ/โฟลเดอร์และตรวจสอบว่าโมเดลโหลดสำเร็จ")
            return
        
        # ถ้ามี output_folder แสดงว่าเลือก folder (ประมวลผลหลายไฟล์)
        if self.output_folder:
            self.btn_process.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.status_label.setText(f"กำลังประมวลผล {len(self.image_files)} ไฟล์...")
            
            # สร้าง thread สำหรับประมวลผลหลายไฟล์
            strong_fade = self.strong_fade_checkbox.isChecked()
            self.batch_thread = BatchProcessingThread(
                self.image_files, self.output_folder, 
                self.cap_detector, self.craft_detector, strong_fade
            )
            self.batch_thread.result_ready.connect(self.on_batch_result_ready)
            self.batch_thread.progress_updated.connect(self.status_label.setText)
            self.batch_thread.finished.connect(self.on_batch_finished)
            self.batch_thread.start()
        else:
            # ประมวลผลไฟล์เดียว (โหมดเดิม)
            self.btn_process.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.status_label.setText("กำลังประมวลผล...")
            
            strong_fade = self.strong_fade_checkbox.isChecked()
            self.thread = CapDetectionThread(self.current_image_path, self.cap_detector, self.craft_detector, strong_fade)
            self.thread.result_ready.connect(self.on_result_ready)
            self.thread.progress_updated.connect(self.status_label.setText)
            self.thread.start()
    
    def on_result_ready(self, result: dict):
        """รับผลลัพธ์จากการประมวลผล"""
        self.btn_process.setEnabled(True)
        
        if 'error' in result:
            QMessageBox.warning(self, "ข้อผิดพลาด", result['message'])
            self.status_label.setText(f"❌ {result['message']}")
            return
        
        self.current_result = result
        
        # Display cropped cap
        cropped_images = result.get('cropped_images', [])
        bottommost_index = result.get('bottommost_index', 0)
        
        if cropped_images and bottommost_index < len(cropped_images):
            cap_image = cropped_images[bottommost_index]
            self.display_numpy_image(cap_image, self.result_label)
        
        # Display debug images
        craft_result = result.get('craft_result', {})
        debug_images = craft_result.get('debug_images', {}).copy()
        
        # สร้าง debug image ที่แสดง polygon ที่ถูกเลือก (ถ้ามี)
        fade_results = result.get('fade_results', {})
        if fade_results and 'selected_indices' in fade_results:
            selected_indices = fade_results['selected_indices']
            text_boxes = craft_result.get('text_boxes', [])
            if len(text_boxes) > 0 and 'original' in debug_images:
                selected_polygons_img = debug_images['original'].copy()
                # วาด polygon ทั้งหมด
                for idx, polygon in enumerate(text_boxes):
                    box_i = np.array(polygon, dtype=np.int32)
                    if idx in selected_indices:
                        # Polygon ที่ถูกเลือก: สีแดง
                        cv2.polylines(selected_polygons_img, [box_i], True, (0, 0, 255), 3)
                    else:
                        # Polygon ที่ไม่ถูกเลือก: สีเขียวอ่อน
                        cv2.polylines(selected_polygons_img, [box_i], True, (0, 255, 0), 1)
                debug_images['selected_polygons'] = selected_polygons_img
        
        debug_names = ['original', 'craft_boxes', 'text_boxes_mask', 'cropped_region']
        for i, name in enumerate(debug_names):
            if i < len(self.debug_labels) and name in debug_images:
                self.display_numpy_image(debug_images[name], self.debug_labels[i])
        
        # แสดง selected_polygons ถ้ามี (แทนที่ text_boxes_mask)
        if 'selected_polygons' in debug_images and len(self.debug_labels) > 2:
            self.display_numpy_image(debug_images['selected_polygons'], self.debug_labels[2])
        
        # Display fade technique results
        fade_results = result.get('fade_results', {})
        for key, image_label in self.fade_labels.items():
            if key in fade_results:
                self.display_numpy_image(fade_results[key], image_label)
            else:
                # แสดงข้อความว่าไม่มีผลลัพธ์
                image_label.setText("(ไม่มีผลลัพธ์)")
                image_label.setAlignment(Qt.AlignCenter)
        
        # Display results
        cap_result = result.get('cap_result', {})
        detections = cap_result.get('detections', [])
        
        results_text = "=== ผลลัพธ์การตรวจจับฝา ===\n\n"
        results_text += f"พบฝาทั้งหมด: {len(detections)} ฝา\n\n"
        
        for i, det in enumerate(detections):
            conf = det.get('confidence', 0)
            bbox = det.get('bbox', [])
            results_text += f"ฝาที่ {i+1}:\n"
            results_text += f"  - ความเชื่อมั่น: {conf:.2%}\n"
            results_text += f"  - ตำแหน่ง: {bbox}\n"
            if i == bottommost_index:
                results_text += f"  - ⭐ ฝาที่เลือก (ฝาล่างสุด)\n"
            results_text += "\n"
        
        results_text += "=== ผลลัพธ์การตรวจจับตัวอักษรด้วย CRAFT ===\n\n"
        num_regions = craft_result.get('num_text_regions', 0)
        text_boxes = craft_result.get('text_boxes', [])
        results_text += f"จำนวน Polygon จาก CRAFT: {len(text_boxes)}\n"
        
        # แสดงข้อมูล polygon ที่ถูกเลือก
        if fade_results:
            selected_indices = fade_results.get('selected_indices', [])
            selected_polygons = fade_results.get('selected_polygons', [])
            if selected_indices:
                results_text += f"  → สุ่มเลือก {len(selected_indices)}/{len(text_boxes)} polygon เพื่อประมวลผล\n"
                results_text += f"  → Polygon ที่เลือก (indices): {selected_indices}\n"
        
        message = craft_result.get('message', '')
        if message:
            results_text += f"หมายเหตุ: {message}\n\n"
        
        # Add fade techniques info
        if fade_results:
            results_text += "=== เทคนิคการทำให้ตัวอักษรจาง (เฉพาะ polygon ที่สุ่มเลือก) ===\n\n"
            technique_descriptions = {
                'alpha_blending': '1. Alpha Blending: ผสมค่าความโปร่งใสกับพื้นหลัง',
                'inpainting': '3. Inpainting: ลบและเติมเต็มด้วย texture จากพื้นหลัง',
                'gradient_horizontal': '5. Gradient Masking (แนวนอน): ค่อยๆ จางจากซ้ายไปขวา',
                'gradient_vertical': '6. Gradient Masking (แนวตั้ง): ค่อยๆ จางจากบนลงล่าง'
            }
            
            for key, desc in technique_descriptions.items():
                if key in fade_results:
                    results_text += f"✅ {desc}\n"
                else:
                    results_text += f"❌ {desc.split(':')[0]}: ไม่สามารถประมวลผลได้\n"
        
        self.results_text.setText(results_text)
        self.status_label.setText("✅ ประมวลผลเสร็จสิ้น")
        self.btn_save.setEnabled(True)  # เปิดใช้งานปุ่มบันทึก
    
    def on_batch_result_ready(self, result: dict):
        """รับผลลัพธ์จากการประมวลผลแต่ละไฟล์ (batch mode)"""
        # แสดงผลลัพธ์ของไฟล์ล่าสุดที่ประมวลผลเสร็จ
        if result.get('success'):
            # อัปเดต UI ด้วยไฟล์ล่าสุด (ถ้าต้องการ)
            pass
        else:
            # Log error แต่ไม่แสดง popup เพื่อไม่รบกวนการประมวลผล
            print(f"❌ ไฟล์ {result.get('file_path')}: {result.get('message', '')}")
    
    def on_batch_finished(self, success_count: int, total_count: int):
        """เมื่อประมวลผลทุกไฟล์เสร็จสิ้น"""
        self.btn_process.setEnabled(True)
        self.btn_save.setEnabled(True)
        
        message = f"✅ ประมวลผลเสร็จสิ้น: {success_count}/{total_count} ไฟล์สำเร็จ"
        if success_count < total_count:
            message += f" ({total_count - success_count} ไฟล์ล้มเหลว)"
        
        self.status_label.setText(message)
        QMessageBox.information(self, "ประมวลผลเสร็จสิ้น", 
                               f"ประมวลผล {total_count} ไฟล์\n"
                               f"สำเร็จ: {success_count} ไฟล์\n"
                               f"ล้มเหลว: {total_count - success_count} ไฟล์\n\n"
                               f"ผลลัพธ์บันทึกไว้ที่:\n{self.output_folder}")
    
    def save_results(self):
        """บันทึกภาพผลลัพธ์ทั้งหมดลง folder"""
        if not self.current_result:
            QMessageBox.warning(self, "ข้อผิดพลาด", "ยังไม่มีผลลัพธ์ให้บันทึก")
            return
        
        # เลือก folder สำหรับบันทึก
        save_dir = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์สำหรับบันทึกภาพผลลัพธ์")
        if not save_dir:
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0] if self.current_image_path else "result"
            
            saved_count = 0
            
            # บันทึกภาพต้นฉบับ
            cropped_images = self.current_result.get('cropped_images', [])
            bottommost_index = self.current_result.get('bottommost_index', 0)
            if cropped_images and bottommost_index < len(cropped_images):
                cap_image = cropped_images[bottommost_index]
                original_path = os.path.join(save_dir, f"{base_name}_original_{timestamp}.png")
                cv2.imwrite(original_path, cap_image)
                saved_count += 1
            
            # บันทึกภาพผลลัพธ์จากเทคนิคต่างๆ
            fade_results = self.current_result.get('fade_results', {})
            technique_names = {
                'alpha_blending': '01_AlphaBlending',
                'inpainting': '03_Inpainting',
                'gradient_horizontal': '05_GradientHorizontal',
                'gradient_vertical': '06_GradientVertical'
            }
            
            for key, name_prefix in technique_names.items():
                if key in fade_results:
                    image = fade_results[key]
                    save_path = os.path.join(save_dir, f"{base_name}_{name_prefix}_{timestamp}.png")
                    cv2.imwrite(save_path, image)
                    saved_count += 1
            
            # บันทึก debug images
            craft_result = self.current_result.get('craft_result', {})
            debug_images = craft_result.get('debug_images', {})
            if 'craft_boxes' in debug_images:
                debug_path = os.path.join(save_dir, f"{base_name}_debug_craft_boxes_{timestamp}.png")
                cv2.imwrite(debug_path, debug_images['craft_boxes'])
                saved_count += 1
            
            if 'selected_polygons' in debug_images:
                selected_path = os.path.join(save_dir, f"{base_name}_debug_selected_polygons_{timestamp}.png")
                cv2.imwrite(selected_path, debug_images['selected_polygons'])
                saved_count += 1
            
            QMessageBox.information(self, "บันทึกสำเร็จ", 
                                   f"บันทึกภาพ {saved_count} ไฟล์ลงโฟลเดอร์:\n{save_dir}")
            self.status_label.setText(f"✅ บันทึก {saved_count} ไฟล์สำเร็จ")
            
        except Exception as e:
            QMessageBox.warning(self, "ข้อผิดพลาด", f"ไม่สามารถบันทึกภาพได้: {str(e)}")
            self.status_label.setText(f"❌ บันทึกล้มเหลว: {str(e)}")


def main():
    if not PYQT5_AVAILABLE:
        print("❌ PyQt5 ไม่ได้ติดตั้ง กรุณาติดตั้งด้วย: pip install PyQt5")
        return
    
    if not CAP_DETECTION_AVAILABLE:
        print("❌ Modules ไม่พร้อมใช้งาน")
        return
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = CRAFTDetectorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

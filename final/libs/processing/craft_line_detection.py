#!/usr/bin/env python3
"""
CRAFT Line Detection Tool
ใช้ CRAFT เพื่อจับบรรทัดข้อความในภาพ และครอปเป็นบรรทัดๆ
"""

import sys
import os

# CRITICAL: Setup CUDA paths BEFORE importing PyTorch
# This must be done at module level before any torch imports
try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    # Fallback if cuda_setup is not available
    pass

# Now import PyTorch (after CUDA paths are set)
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image, ImageTk
import json
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from collections import OrderedDict
from typing import Dict, Optional, Union, Tuple, List

# Import CUDA image utilities
try:
    from core.cuda_image_utils import cuda_resize, cuda_cvtColor, cuda_gaussianBlur
    CUDA_IMAGE_UTILS_AVAILABLE = True
except ImportError:
    # Fallback if not available
    CUDA_IMAGE_UTILS_AVAILABLE = False
    def cuda_resize(img, size, **kwargs):
        return cv2.resize(img, size, **kwargs)
    def cuda_cvtColor(img, code):
        return cv2.cvtColor(img, code)
    def cuda_gaussianBlur(img, ksize, sigma):
        return cv2.GaussianBlur(img, ksize, sigma)

# เพิ่ม path ของ CRAFT-pytorch
# Get path from config if available, otherwise use default
try:
    from config.settings import CRAFT_PYTORCH_DIR
    craft_path = CRAFT_PYTORCH_DIR
except ImportError:
    # Fallback to old path if config not available
    craft_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch"

if craft_path not in sys.path:
    sys.path.append(craft_path)

try:
    import craft_utils
    import imgproc
    import file_utils
    from craft import CRAFT
except ImportError as e:
    print(f"Error importing CRAFT modules: {e}")
    print("Please make sure CRAFT-pytorch is properly installed")
    sys.exit(1)

def copyStateDict(state_dict):
    """Copy state dict for compatibility"""
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None, show_time=False):
    """Test network on image and return detection results"""
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    if show_time:
        print(f"infer/postproc time : {t0:.3f}/{t1:.3f}")

    return boxes, polys, score_text

def load_craft_model(model_path, refiner_path=None, use_cuda=True):
    """Load CRAFT model and optional refiner"""
    net = CRAFT()
    
    print(f'Loading main model weights from checkpoint ({model_path})')
    if use_cuda:
        net.load_state_dict(copyStateDict(torch.load(model_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    
    # Load refiner model if provided
    refine_net = None
    if refiner_path:
        if not os.path.exists(refiner_path):
            print(f"⚠️ Warning: Refiner file not found: {refiner_path}")
            print("   Continuing without refiner model...")
        else:
            try:
                # Import refiner model
                from refinenet import RefineNet
                
                refine_net = RefineNet()
                print(f'Loading refiner weights from checkpoint ({refiner_path})')
                
                if use_cuda:
                    refine_net.load_state_dict(copyStateDict(torch.load(refiner_path)))
                    refine_net = refine_net.cuda()
                    refine_net = torch.nn.DataParallel(refine_net)
                else:
                    refine_net.load_state_dict(copyStateDict(torch.load(refiner_path, map_location='cpu')))
                
                refine_net.eval()
                print(f"✓ Refiner model loaded successfully from: {refiner_path}")
                
            except ImportError as e:
                print(f"⚠️ Warning: Could not import RefineNet: {e}")
                print("   Continuing without refiner model...")
            except Exception as e:
                print(f"⚠️ Warning: Error loading refiner model: {e}")
                print("   Continuing without refiner model...")
    
    return net, refine_net

def group_lines_advanced(polys, tolerance=10):
    """จัดกลุ่ม polys ด้วยวิธี clustering ที่ซับซ้อนกว่า"""
    if polys is None or (hasattr(polys, '__len__') and len(polys) == 0):
        return []
    
    # สร้าง list ของ (y_center, poly)
    poly_with_y = []
    for poly in polys:
        if poly is not None:
            try:
                y_center = np.mean(poly[:, 1])
                poly_with_y.append((y_center, poly))
            except (IndexError, TypeError):
                continue
    
    if not poly_with_y:
        return []
    
    # เรียงตาม y_center
    poly_with_y.sort(key=lambda x: x[0])
    
    # หา gaps ระหว่าง polys
    y_centers = [y for y, _ in poly_with_y]
    gaps = []
    for i in range(1, len(y_centers)):
        gap = y_centers[i] - y_centers[i-1]
        gaps.append(gap)
    
    # หา median gap เพื่อใช้เป็นเกณฑ์
    if gaps:
        median_gap = np.median(gaps)
        # ใช้ median_gap * 1.5 เป็น tolerance
        adaptive_tolerance = min(tolerance, median_gap * 1.5)
    else:
        adaptive_tolerance = tolerance
    
    # จัดกลุ่มด้วย adaptive tolerance
    lines = []
    current_line = []
    current_y = None
    
    for y_center, poly in poly_with_y:
        if current_y is None:
            current_y = y_center
            current_line.append(poly)
        elif abs(y_center - current_y) <= adaptive_tolerance:
            current_line.append(poly)
        else:
            if current_line:
                lines.append(current_line)
            current_line = [poly]
            current_y = y_center
    
    if current_line:
        lines.append(current_line)
    
    return lines

def group_lines_by_y_position(polys, tolerance=10):
    """จัดกลุ่ม polys ตามตำแหน่ง Y เพื่อแยกเป็นบรรทัด"""
    if polys is None or (hasattr(polys, '__len__') and len(polys) == 0):
        return []
    
    # สร้าง list ของ (y_center, poly)
    poly_with_y = []
    for poly in polys:
        if poly is not None:
            try:
                y_center = np.mean(poly[:, 1])
                poly_with_y.append((y_center, poly))
            except (IndexError, TypeError):
                # ข้าม poly ที่ไม่ถูกต้อง
                continue
    
    if not poly_with_y:
        return []
    
    # เรียงตาม y_center
    poly_with_y.sort(key=lambda x: x[0])
    
    # ใช้ clustering แบบง่ายเพื่อแยกบรรทัด
    lines = []
    current_line = []
    current_y = None
    
    for y_center, poly in poly_with_y:
        if current_y is None:
            current_y = y_center
            current_line.append(poly)
        elif abs(y_center - current_y) <= tolerance:
            # อยู่ในบรรทัดเดียวกัน
            current_line.append(poly)
        else:
            # เริ่มบรรทัดใหม่
            if current_line:
                lines.append(current_line)
            current_line = [poly]
            current_y = y_center
    
    # เพิ่มบรรทัดสุดท้าย
    if current_line:
        lines.append(current_line)
    
    # ถ้ายังมีบรรทัดน้อยเกินไป ให้ลองแยกเพิ่มเติม
    if len(lines) < 3 and len(poly_with_y) >= 3:
        # ลองใช้ tolerance ที่เล็กลง
        return group_lines_by_y_position(polys, tolerance=max(5, tolerance // 2))
    
    return lines

def group_lines_without_overlap(polys, tolerance=10, max_lines=3):
    """จัดกลุ่ม polys โดยไม่ให้กรอบทับกัน และจำกัดจำนวนบรรทัด"""
    if polys is None or (hasattr(polys, '__len__') and len(polys) == 0):
        return []
    
    # สร้าง list ของ (y_center, poly)
    poly_with_y = []
    for poly in polys:
        if poly is not None:
            try:
                y_center = np.mean(poly[:, 1])
                poly_with_y.append((y_center, poly))
            except (IndexError, TypeError):
                continue
    
    if not poly_with_y:
        return []
    
    # เรียงตาม y_center
    poly_with_y.sort(key=lambda x: x[0])
    
    # หา gaps ระหว่าง polys
    y_centers = [y for y, _ in poly_with_y]
    gaps = []
    for i in range(1, len(y_centers)):
        gap = y_centers[i] - y_centers[i-1]
        gaps.append(gap)
    
    # หา median gap เพื่อใช้เป็นเกณฑ์
    if gaps:
        median_gap = np.median(gaps)
        # ใช้ median_gap * 0.8 เป็น tolerance (ลดลงจาก 1.2)
        adaptive_tolerance = min(tolerance, median_gap * 0.8)
    else:
        adaptive_tolerance = tolerance
    
    # จัดกลุ่มด้วย adaptive tolerance
    lines = []
    current_line = []
    current_y = None
    
    for y_center, poly in poly_with_y:
        if current_y is None:
            current_y = y_center
            current_line.append(poly)
        elif abs(y_center - current_y) <= adaptive_tolerance:
            current_line.append(poly)
        else:
            if current_line:
                lines.append(current_line)
            current_line = [poly]
            current_y = y_center
    
    if current_line:
        lines.append(current_line)
    
    # ถ้ามีบรรทัดมากเกินไป ให้รวมบรรทัดที่ใกล้กัน
    if len(lines) > max_lines:
        lines = merge_close_lines(lines, max_lines, adaptive_tolerance)
    
    # ถ้ายังมีบรรทัดน้อยเกินไป ให้แยกเพิ่มเติม
    elif len(lines) < max_lines and len(poly_with_y) >= max_lines:
        # ลองใช้ tolerance ที่เล็กลง
        return group_lines_without_overlap(polys, tolerance=max(3, tolerance // 2), max_lines=max_lines)
    
    return lines

def group_lines_fine_tuned(polys, tolerance=8, max_lines=3):
    """จัดกลุ่ม polys แบบละเอียดมากขึ้นเพื่อแยกบรรทัดให้ดีขึ้น"""
    if polys is None or (hasattr(polys, '__len__') and len(polys) == 0):
        return []
    
    # สร้าง list ของ (y_center, poly)
    poly_with_y = []
    for poly in polys:
        if poly is not None:
            try:
                y_center = np.mean(poly[:, 1])
                poly_with_y.append((y_center, poly))
            except (IndexError, TypeError):
                continue
    
    if not poly_with_y:
        return []
    
    # เรียงตาม y_center
    poly_with_y.sort(key=lambda x: x[0])
    
    # หา gaps ระหว่าง polys
    y_centers = [y for y, _ in poly_with_y]
    gaps = []
    for i in range(1, len(y_centers)):
        gap = y_centers[i] - y_centers[i-1]
        gaps.append(gap)
    
    # หา median gap และใช้ค่าเล็กกว่า
    if gaps:
        median_gap = np.median(gaps)
        # ใช้ median_gap * 0.6 เป็น tolerance (ลดลงอีก)
        adaptive_tolerance = min(tolerance, median_gap * 0.6)
    else:
        adaptive_tolerance = tolerance
    
    # จัดกลุ่มด้วย adaptive tolerance
    lines = []
    current_line = []
    current_y = None
    
    for y_center, poly in poly_with_y:
        if current_y is None:
            current_y = y_center
            current_line.append(poly)
        elif abs(y_center - current_y) <= adaptive_tolerance:
            current_line.append(poly)
        else:
            if current_line:
                lines.append(current_line)
            current_line = [poly]
            current_y = y_center
    
    if current_line:
        lines.append(current_line)
    
    # ถ้ายังมีบรรทัดน้อยเกินไป ให้แยกเพิ่มเติม
    if len(lines) < max_lines and len(poly_with_y) >= max_lines:
        # ลองใช้ tolerance ที่เล็กลงมาก
        return group_lines_fine_tuned(polys, tolerance=max(2, tolerance // 3), max_lines=max_lines)
    
    # ถ้ามีบรรทัดมากเกินไป ให้รวมบรรทัดที่ใกล้กัน
    if len(lines) > max_lines:
        lines = merge_close_lines(lines, max_lines, adaptive_tolerance)
    
    return lines

def merge_close_lines(lines, target_lines, tolerance):
    """รวมบรรทัดที่ใกล้กันจนได้จำนวนที่ต้องการ"""
    if len(lines) <= target_lines:
        return lines
    
    # คำนวณระยะห่างระหว่างบรรทัด
    line_centers = []
    for line in lines:
        if line:
            y_centers = [np.mean(poly[:, 1]) for poly in line if poly is not None]
            if y_centers:
                line_centers.append(np.mean(y_centers))
    
    # หาคู่บรรทัดที่ใกล้กันที่สุด
    while len(lines) > target_lines:
        min_distance = float('inf')
        merge_idx = -1
        
        for i in range(len(lines) - 1):
            if i < len(line_centers) - 1:
                distance = abs(line_centers[i+1] - line_centers[i])
                if distance < min_distance:
                    min_distance = distance
                    merge_idx = i
        
        if merge_idx >= 0:
            # รวมบรรทัดที่ใกล้กันที่สุด
            lines[merge_idx].extend(lines[merge_idx + 1])
            lines.pop(merge_idx + 1)
            line_centers.pop(merge_idx + 1)
        else:
            break
    
    return lines

def check_line_overlap(line1_polys, line2_polys):
    """ตรวจสอบว่าบรรทัดทับกันหรือไม่"""
    if not line1_polys or not line2_polys:
        return False
    
    # หา bounding box ของแต่ละบรรทัด
    def get_line_bbox(polys):
        valid_polys = [p for p in polys if p is not None]
        if not valid_polys:
            return None
        
        all_points = np.vstack(valid_polys)
        x_min = int(np.min(all_points[:, 0]))
        y_min = int(np.min(all_points[:, 1]))
        x_max = int(np.max(all_points[:, 0]))
        y_max = int(np.max(all_points[:, 1]))
        return (x_min, y_min, x_max, y_max)
    
    bbox1 = get_line_bbox(line1_polys)
    bbox2 = get_line_bbox(line2_polys)
    
    if bbox1 is None or bbox2 is None:
        return False
    
    # ตรวจสอบการทับกัน
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # ตรวจสอบการทับกันในแนวแกน Y
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    # ถ้าทับกันมากกว่า 30% ของความสูงของบรรทัดที่เล็กกว่า
    line1_height = y1_max - y1_min
    line2_height = y2_max - y2_min
    min_height = min(line1_height, line2_height)
    
    return y_overlap > (min_height * 0.3)

def crop_line(image, line_polys, padding=0, max_height_ratio=0.08, uniform_height=True, shrink_x_percent=0.02, shrink_y_percent=0.10):
    """ครอปบรรทัดจาก polys ที่อยู่ในบรรทัดเดียวกัน"""
    if line_polys is None or (hasattr(line_polys, '__len__') and len(line_polys) == 0):
        return None, None
    
    # กรอง polys ที่ไม่เป็น None
    valid_polys = [p for p in line_polys if p is not None]
    if not valid_polys:
        return None, None
    
    # หา bounding box ของบรรทัด
    all_points = np.vstack(valid_polys)
    # แกน X (แนวนอน): ครอปเข้าไปใน bounding box เล็กน้อยเพื่อให้ใกล้ข้อความมากขึ้น
    # แกน Y (แนวตั้ง): ใช้ padding ตามเดิมเพื่อให้ครอปเหมาะสม
    x_min_raw = np.min(all_points[:, 0])
    x_max_raw = np.max(all_points[:, 0])
    y_min_raw = np.min(all_points[:, 1])
    y_max_raw = np.max(all_points[:, 1])
    
    # คำนวณความกว้างและความสูงของ bounding box
    width = x_max_raw - x_min_raw
    height = y_max_raw - y_min_raw
    
    # ครอปเข้าไปใน bounding box เล็กน้อย
    shrink_x = max(1, int(width * shrink_x_percent))  # ใช้ shrink_x_percent แทน hardcode
    shrink_y = max(2, int(height * shrink_y_percent))  # ใช้ shrink_y_percent แทน hardcode
    
    # ปรับ bounding box แบบเดิมก่อน (ไม่ใช้ shrink_y ตอนนี้)
    x_min = max(0, int(np.floor(x_min_raw + shrink_x)))  # ขยับขอบซ้ายเข้าไป
    x_max = min(image.shape[1], int(np.ceil(x_max_raw - shrink_x)))  # ขยับขอบขวาเข้าไป
    y_min = max(0, int(np.floor(y_min_raw)))  # ยังไม่ใช้ shrink_y
    y_max = min(image.shape[0], int(np.ceil(y_max_raw)))  # ยังไม่ใช้ shrink_y
    
    # จำกัดความสูงของกรอบ
    current_height = y_max - y_min
    max_height = int(image.shape[0] * max_height_ratio)
    
    if current_height > max_height:
        # ลดความสูงโดยปรับจากจุดกึ่งกลาง
        center_y = (y_min + y_max) // 2
        y_min = max(0, center_y - max_height // 2)
        y_max = min(image.shape[0], center_y + max_height // 2)
    
    # ถ้าต้องการความสูงเท่ากัน ให้ใช้ความสูงที่กำหนด
    if uniform_height:
        # ใช้ความสูงคงที่สำหรับทุกกรอบ
        fixed_height = int(image.shape[0] * max_height_ratio)  # ใช้ max_height_ratio แทน hardcode 0.1
        center_y = (y_min + y_max) // 2
        y_min = max(0, center_y - fixed_height // 2)
        y_max = min(image.shape[0], center_y + fixed_height // 2)
    
    # ใช้ shrink_y หลังจากที่คำนวณ y_min และ y_max แล้ว เพื่อให้แน่ใจว่า shrink ยังคงมีผล
    current_height_after = y_max - y_min
    shrink_y_actual = max(1, int(current_height_after * shrink_y_percent))  # คำนวณ shrink_y จากความสูงปัจจุบัน
    y_min = max(0, y_min + shrink_y_actual)  # ขยับขอบบนเข้าไป
    y_max = min(image.shape[0], y_max - shrink_y_actual)  # ขยับขอบล่างเข้าไป
    
    # ครอปภาพ
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped, (x_min, y_min, x_max, y_max)

def crop_detected_boxes(image, lines, output_folder, base_filename):
    """ครอปรูปจากกรอบที่ตรวจจับได้และบันทึกเป็นไฟล์"""
    cropped_files = []
    
    try:
        for line_idx, line_polys in enumerate(lines):
            if line_polys is None or (hasattr(line_polys, '__len__') and len(line_polys) == 0):
                continue
                
            # ครอปบรรทัด
            cropped_image, bbox = crop_line(image, line_polys)
            
            if cropped_image is not None:
                # บันทึกไฟล์
                crop_filename = f"{base_filename}_line_{line_idx+1}.png"
                crop_path = os.path.join(output_folder, crop_filename)
                cv2.imwrite(crop_path, cuda_cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                cropped_files.append(crop_path)
                
                print(f"บันทึกบรรทัด {line_idx+1}: {crop_filename}")
                
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการครอปกรอบ: {str(e)}")
    
    return cropped_files

def draw_detection_result(image, lines, output_path=None):
    """วาดผลการตรวจจับบนภาพ"""
    result_image = image.copy()
    
    # สีสำหรับแต่ละบรรทัด
    colors = [
        (255, 0, 0),    # แดง
        (0, 255, 0),    # เขียว
        (0, 0, 255),    # น้ำเงิน
        (255, 255, 0),  # เหลือง
        (255, 0, 255),  # ม่วง
        (0, 255, 255),  # ฟ้า
        (128, 0, 0),    # แดงเข้ม
        (0, 128, 0),    # เขียวเข้ม
        (0, 0, 128),    # น้ำเงินเข้ม
        (128, 128, 0),  # เขียวเหลือง
    ]
    
    # เก็บ bounding boxes ของบรรทัดที่วาดแล้ว
    drawn_bboxes = []
    
    # คำนวณความสูงคงที่สำหรับทุกกรอบ
    image_height = image.shape[0]
    uniform_height = int(image_height * 0.1)  # 10% ของความสูงของภาพ
    
    # แสดง 3 บรรทัด
    for line_idx, line_polys in enumerate(lines):
        if line_idx >= 3:  # แสดงแค่ 3 บรรทัดแรก
            break
            
        color = colors[line_idx % len(colors)]
        
        # ไม่วาด polys - แค่กรอบอย่างเดียว
        
        # หา bounding box ของบรรทัด
        if line_polys:
            valid_polys = [p for p in line_polys if p is not None]
            if valid_polys:
                all_points = np.vstack(valid_polys)
                x_min = int(np.min(all_points[:, 0]))
                y_min = int(np.min(all_points[:, 1]))
                x_max = int(np.max(all_points[:, 0]))
                y_max = int(np.max(all_points[:, 1]))
                
                # วาดกรอบบรรทัดอย่างเดียว (ไม่ปรับ uniform height)
                cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), color, 3)
    
    if output_path:
        cv2.imwrite(output_path, cuda_cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    return result_image

def adjust_bbox_to_avoid_overlap(bbox, existing_bboxes, min_gap=5):
    """ปรับขนาด bbox เพื่อไม่ให้ทับกัน"""
    x_min, y_min, x_max, y_max = bbox
    
    for existing_bbox in existing_bboxes:
        ex_x_min, ex_y_min, ex_x_max, ex_y_max = existing_bbox
        
        # ตรวจสอบการทับกันในแนวแกน Y
        y_overlap = max(0, min(y_max, ex_y_max) - max(y_min, ex_y_min))
        
        if y_overlap > 0:
            # ปรับขนาดเพื่อไม่ให้ทับกัน
            if y_min < ex_y_min:
                # bbox อยู่ด้านบน
                y_max = ex_y_min - min_gap
            else:
                # bbox อยู่ด้านล่าง
                y_min = ex_y_max + min_gap
    
    return (x_min, y_min, x_max, y_max)

class CRAFTLineDetector:
    """
    CRAFT Line Detection Class สำหรับเรียกใช้จาก code อื่น
    รับรูปจาก rotationmodel.py มาตรวจจับบรรทัด
    """
    
    def __init__(self, model_path: str = None, refiner_path: str = None, cuda: bool = True):
        """
        Initialize CRAFT Line Detector
        
        Args:
            model_path: Path to CRAFT model (.pth file)
            refiner_path: Path to CRAFT refiner model (.pth file) - optional
            cuda: Use CUDA if available
        """
        self.net = None
        self.refine_net = None
        # Use CUDA if available (CUDA paths already set at module level)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"🚀 PyTorch CUDA enabled - using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ PyTorch CUDA not available - using CPU for craft line detection")
        
        # Default settings จาก code เดิม
        self.settings = {
            'text_threshold': 0.7,
            'low_text': 0.4,
            'link_threshold': 0.4,
            'padding': 0,
            'tolerance': 10,
            'uniform_height': True,
            'max_height_ratio': 0.08,  # 8% ของความสูง
            'shrink_x_percent': 0.02,  # 2% ของความกว้าง
            'shrink_y_percent': 0.10,  # 10% ของความสูง
            'horizontal_splitting': True,
            'min_width': 65,
            'min_height': 30,
            'max_lines': 3
        }
        
        # Global variables for storing results
        self.last_detection_result = None
        self.last_processed_image = None
        self.detection_history = []
        
        if model_path:
            self.load_model(model_path, refiner_path)
    
    def load_model(self, model_path: str, refiner_path: str = None) -> bool:
        """
        Load CRAFT model and refiner from paths
        
        Args:
            model_path: Path to .pth model file
            refiner_path: Path to .pth refiner file (optional)
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"Loading CRAFT model from {model_path}...")
            self.net, self.refine_net = load_craft_model(model_path, refiner_path, use_cuda=self.device.type == 'cuda')
            print("✅ CRAFT model loaded successfully.")
            
            if self.refine_net is not None:
                print("✓ Refiner model loaded successfully.")
            else:
                print("✓ Using main model only.")
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading CRAFT model: {e}")
            return False
    
    def update_settings(self, **kwargs):
        """
        Update detection settings
        
        Args:
            **kwargs: Settings to update
        """
        self.settings.update(kwargs)
    
    def detect_lines_from_image_path(self, image_path: str) -> Dict:
        """
        Detect text lines from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict containing line detection results
        """
        try:
            if self.net is None:
                raise ValueError("CRAFT model not loaded. Please load model first.")
            
            # Load image
            image = imgproc.loadImage(image_path)
            if image is None:
                raise ValueError(f"Could not read image {image_path}")
            
            return self.detect_lines_from_image_array(image, image_path)
            
        except Exception as e:
            print(f"Error detecting lines from image path: {e}")
            return {
                'error': str(e),
                'lines': [],
                'total_lines': 0,
                'image_path': image_path
            }
    
    def detect_lines_from_image_array(self, image: np.ndarray, image_path: str = None) -> Dict:
        """
        Detect text lines from image array (รับรูปจาก rotationmodel.py)
        
        Args:
            image: Image as numpy array (RGB format)
            image_path: Optional image path for reference
            
        Returns:
            Dict containing line detection results
        """
        try:
            if self.net is None:
                raise ValueError("CRAFT model not loaded. Please load model first.")
            
            if image is None:
                raise ValueError("Image array is None")
            
            # Store original image
            self.last_processed_image = image.copy()
            
            # Detect text regions
            boxes, polys, score_text = test_net(
                self.net, image,
                self.settings['text_threshold'],
                self.settings['link_threshold'],
                self.settings['low_text'],
                self.device.type == 'cuda',
                False,  # poly
                self.refine_net,  # refiner
                show_time=False
            )
            
            # Check if polys is empty or None
            if polys is None or (hasattr(polys, '__len__') and len(polys) == 0):
                result = {
                    'image_path': image_path,
                    'original_image': image,
                    'lines': [],
                    'total_lines': 0,
                    'total_text_regions': 0,
                    'text_boxes': [],
                    'text_polys': [],
                    'cropped_lines': [],
                    'timestamp': str(datetime.now())
                }
                
                self.last_detection_result = result
                self.detection_history.append(result)
                return result
            
            # Apply horizontal splitting if enabled
            if self.settings['horizontal_splitting']:
                split_polys = self.split_large_regions_horizontally(polys)
                lines = group_lines_fine_tuned(split_polys, self.settings['tolerance'], max_lines=self.settings['max_lines'])
            else:
                lines = group_lines_fine_tuned(polys, self.settings['tolerance'], max_lines=self.settings['max_lines'])
            
            # Crop lines
            cropped_lines = []
            for line_idx, line_polys in enumerate(lines):
                cropped_line, bbox = crop_line(
                    image, line_polys, 
                    self.settings['padding'], 
                    self.settings['max_height_ratio'], 
                    self.settings['uniform_height'],
                    self.settings.get('shrink_x_percent', 0.02),
                    self.settings.get('shrink_y_percent', 0.10)
                )
                
                if cropped_line is not None:
                    cropped_lines.append({
                        'line_index': line_idx,
                        'image': cropped_line,
                        'bbox': bbox,
                        'num_characters': len(line_polys)
                    })
            
            # Create result dictionary
            result = {
                'image_path': image_path,
                'original_image': image,
                'lines': lines,
                'total_lines': len(lines),
                'total_text_regions': len(polys),
                'text_boxes': boxes,
                'text_polys': polys,
                'cropped_lines': cropped_lines,
                'settings_used': self.settings.copy(),
                'timestamp': str(datetime.now())
            }
            
            # Store in global variables
            self.last_detection_result = result
            self.detection_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"Error detecting lines from image array: {e}")
            return {
                'error': str(e),
                'lines': [],
                'total_lines': 0,
                'image_path': image_path
            }
    
    def detect_lines_from_rotation_result(self, rotation_result: Dict) -> Dict:
        """
        Detect text lines from rotation model result
        
        Args:
            rotation_result: Result from rotationmodel.py
            
        Returns:
            Dict containing line detection results
        """
        try:
            # Extract image from rotation result
            if 'rotated_image' in rotation_result and rotation_result['rotated_image'] is not None:
                image = rotation_result['rotated_image']
                image_path = rotation_result.get('image_path')
            elif 'cropped_image' in rotation_result and rotation_result['cropped_image'] is not None:
                image = rotation_result['cropped_image']
                image_path = rotation_result.get('image_path')
            # Backward compatibility: support old ai_rotated_image key
            elif 'ai_rotated_image' in rotation_result and rotation_result['ai_rotated_image'] is not None:
                image = rotation_result['ai_rotated_image']
                image_path = rotation_result.get('image_path')
            else:
                raise ValueError("No valid image found in rotation result")
            
            # Detect lines
            detection_result = self.detect_lines_from_image_array(image, image_path)
            
            # Add rotation info to result
            detection_result['rotation_info'] = {
                'rotation_angle': rotation_result.get('rotation_angle', rotation_result.get('ai_rotation_angle', 0)),  # Support both old and new key names
                'verification_passed': rotation_result.get('verification_passed', False),
                'total_rotations': rotation_result.get('total_rotations', 1)
            }
            
            return detection_result
            
        except Exception as e:
            print(f"Error detecting lines from rotation result: {e}")
            return {
                'error': str(e),
                'lines': [],
                'total_lines': 0
            }
    
    def save_detection_result(self, output_path: str, result: Dict = None) -> bool:
        """
        Save detection result with cropped lines
        
        Args:
            output_path: Directory to save results
            result: Detection result (if None, uses last result)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if result is None:
                result = self.last_detection_result
            
            if result is None or 'error' in result:
                print("⚠️ No valid detection result to save")
                return False
            
            os.makedirs(output_path, exist_ok=True)
            
            # Save cropped lines
            saved_files = []
            for line_data in result['cropped_lines']:
                line_idx = line_data['line_index']
                line_image = line_data['image']
                bbox = line_data['bbox']
                
                # Generate filename
                base_name = "detected_line"
                if result.get('image_path'):
                    base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
                
                line_filename = f"{base_name}_line_{line_idx+1}.png"
                line_path = os.path.join(output_path, line_filename)
                
                # Save image
                cv2.imwrite(line_path, cuda_cvtColor(line_image, cv2.COLOR_RGB2BGR))
                saved_files.append(line_path)
                
                print(f"✅ Saved line {line_idx+1}: {line_filename}")
            
            # Save metadata
            metadata_path = os.path.join(output_path, f"{base_name}_detection_metadata.json")
            metadata = result.copy()
            
            # Remove image data from metadata
            for key in ['original_image']:
                if key in metadata:
                    del metadata[key]
            
            # Remove image data from cropped_lines
            for line_data in metadata['cropped_lines']:
                if 'image' in line_data:
                    del line_data['image']
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Detection metadata saved: {metadata_path}")
            print(f"✅ Total files saved: {len(saved_files)} lines + 1 metadata")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving detection result: {e}")
            return False
    
    def save_annotated_image(self, output_path: str, result: Dict = None) -> bool:
        """
        Save image with detection annotations
        
        Args:
            output_path: Path to save annotated image
            result: Detection result (if None, uses last result)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if result is None:
                result = self.last_detection_result
            
            if result is None or 'error' in result:
                print("⚠️ No valid detection result to annotate")
                return False
            
            # Draw detection results
            annotated_image = draw_detection_result(result['original_image'], result['lines'])
            
            # Save annotated image
            cv2.imwrite(output_path, cuda_cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            print(f"✅ Annotated image saved: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving annotated image: {e}")
            return False
    
    def get_last_result(self) -> Optional[Dict]:
        """Get last detection result"""
        return self.last_detection_result
    
    def get_last_image(self) -> Optional[np.ndarray]:
        """Get last processed image"""
        return self.last_processed_image
    
    def get_detection_history(self) -> list:
        """Get detection history"""
        return self.detection_history
    
    def clear_history(self):
        """Clear detection history"""
        self.detection_history = []
    
    def split_large_regions_horizontally(self, polys, max_height_ratio=0.15):
        """แบ่งกรอบใหญ่ตามแนวนอน - เฉพาะกรอบใหญ่จริงๆ"""
        if not self.settings['horizontal_splitting']:
            return polys
        
        min_height = self.settings['min_height']
        min_width = self.settings['min_width']
        
        split_polys = []
        
        for poly in polys:
            if poly is not None:
                try:
                    # คำนวณขนาดของ poly
                    x_coords = poly[:, 0]
                    y_coords = poly[:, 1]
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # ตรวจสอบว่าต้องแบ่งหรือไม่ - เพิ่มเงื่อนไขความกว้าง
                    should_split = (height > min_height and 
                                  width > min_width and
                                  height > (y_max - y_min) * max_height_ratio)
                    
                    if should_split:
                        # แบ่งเป็น 2 ส่วนตามแนวนอน
                        mid_y = (y_min + y_max) // 2
                        
                        # สร้าง poly สำหรับส่วนบน
                        top_poly = poly.copy()
                        top_poly[:, 1] = np.clip(top_poly[:, 1], y_min, mid_y)
                        split_polys.append(top_poly)
                        
                        # สร้าง poly สำหรับส่วนล่าง
                        bottom_poly = poly.copy()
                        bottom_poly[:, 1] = np.clip(bottom_poly[:, 1], mid_y, y_max)
                        split_polys.append(bottom_poly)
                        
                        print(f"Split large region: {width}x{height} -> 2 parts (min_width={min_width}, min_height={min_height})")
                    else:
                        # ไม่ต้องแบ่ง
                        split_polys.append(poly)
                        
                except Exception as e:
                    print(f"Error splitting region: {e}")
                    split_polys.append(poly)
        
        return split_polys
    

# Global instance for easy access
_global_line_detector = None

def get_global_detector() -> CRAFTLineDetector:
    """Get global line detector instance"""
    global _global_line_detector
    if _global_line_detector is None:
        _global_line_detector = CRAFTLineDetector()
    return _global_line_detector

def initialize_detector(model_path: str, refiner_path: str = None, cuda: bool = True) -> CRAFTLineDetector:
    """
    Initialize global line detector with model and optional refiner
    
    Args:
        model_path: Path to CRAFT model (.pth file)
        refiner_path: Path to CRAFT refiner model (optional)
        cuda: Use CUDA if available
        
    Returns:
        CRAFTLineDetector instance
    """
    global _global_line_detector
    _global_line_detector = CRAFTLineDetector(model_path, refiner_path, cuda)
    return _global_line_detector

def detect_lines_from_path(image_path: str) -> Dict:
    """
    Quick function to detect lines from image path using global detector
    
    Args:
        image_path: Path to image file
        
    Returns:
        Line detection result dictionary
    """
    detector = get_global_detector()
    if detector.net is None:
        raise ValueError("Global detector not initialized. Call initialize_detector() first.")
    
    return detector.detect_lines_from_image_path(image_path)

def detect_lines_from_array(image_array: np.ndarray) -> Dict:
    """
    Quick function to detect lines from image array using global detector
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        Line detection result dictionary
    """
    detector = get_global_detector()
    if detector.net is None:
        raise ValueError("Global detector not initialized. Call initialize_detector() first.")
    
    return detector.detect_lines_from_image_array(image_array)

def detect_lines_from_rotation_result(rotation_result: Dict) -> Dict:
    """
    Quick function to detect lines from rotation result using global detector
    
    Args:
        rotation_result: Result from rotationmodel.py
        
    Returns:
        Line detection result dictionary
    """
    detector = get_global_detector()
    if detector.net is None:
        raise ValueError("Global detector not initialized. Call initialize_detector() first.")
    
    return detector.detect_lines_from_rotation_result(rotation_result)

def get_cropped_lines_from_path(image_path: str) -> list:
    """
    Quick function to get cropped lines from image path
    
    Args:
        image_path: Path to image file
        
    Returns:
        List of cropped line images
    """
    result = detect_lines_from_path(image_path)
    return result.get('cropped_lines', []) if 'error' not in result else []

def get_cropped_lines_from_array(image_array: np.ndarray) -> list:
    """
    Quick function to get cropped lines from image array
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        List of cropped line images
    """
    result = detect_lines_from_array(image_array)
    return result.get('cropped_lines', []) if 'error' not in result else []

def get_cropped_lines_from_rotation_result(rotation_result: Dict) -> list:
    """
    Quick function to get cropped lines from rotation result
    
    Args:
        rotation_result: Result from rotationmodel.py
        
    Returns:
        List of cropped line images
    """
    result = detect_lines_from_rotation_result(rotation_result)
    return result.get('cropped_lines', []) if 'error' not in result else []
    
if __name__ == "__main__":
    print("CRAFT Line Detection Tool")
    print("This module provides line detection functionality for CRAFT text detection.")
    print("Use the CRAFTLineDetector class or global functions to detect text lines.")

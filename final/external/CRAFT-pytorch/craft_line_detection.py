#!/usr/bin/env python3
"""
CRAFT Line Detection Tool
ใช้ CRAFT เพื่อจับบรรทัดข้อความในภาพ และครอปเป็นบรรทัดๆ
"""

import sys
import os
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

# เพิ่ม path ของ CRAFT-pytorch
sys.path.append(r"C:\Users\Win 10 Home\CRAFT-pytorch")

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

def load_craft_model(model_path, use_cuda=True):
    """Load CRAFT model"""
    net = CRAFT()
    
    print(f'Loading weights from checkpoint ({model_path})')
    if use_cuda:
        net.load_state_dict(copyStateDict(torch.load(model_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    return net

def group_lines_advanced(polys, tolerance=10):
    """จัดกลุ่ม polys ด้วยวิธี clustering ที่ซับซ้อนกว่า"""
    if not polys:
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
    if not polys:
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
    if not polys:
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
    if not polys:
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

def crop_line(image, line_polys, padding=10, max_height_ratio=0.12, uniform_height=True):
    """ครอปบรรทัดจาก polys ที่อยู่ในบรรทัดเดียวกัน"""
    if not line_polys:
        return None, None
    
    # กรอง polys ที่ไม่เป็น None
    valid_polys = [p for p in line_polys if p is not None]
    if not valid_polys:
        return None, None
    
    # หา bounding box ของบรรทัด
    all_points = np.vstack(valid_polys)
    x_min = max(0, int(np.min(all_points[:, 0])) - padding)
    y_min = max(0, int(np.min(all_points[:, 1])) - padding)
    x_max = min(image.shape[1], int(np.max(all_points[:, 0])) + padding)
    y_max = min(image.shape[0], int(np.max(all_points[:, 1])) + padding)
    
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
        fixed_height = int(image.shape[0] * 0.1)  # 10% ของความสูงของภาพ
        center_y = (y_min + y_max) // 2
        y_min = max(0, center_y - fixed_height // 2)
        y_max = min(image.shape[0], center_y + fixed_height // 2)
    
    # ครอปภาพ
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped, (x_min, y_min, x_max, y_max)

def crop_detected_boxes(image, lines, output_folder, base_filename):
    """ครอปรูปจากกรอบที่ตรวจจับได้และบันทึกเป็นไฟล์"""
    cropped_files = []
    
    try:
        for line_idx, line_polys in enumerate(lines):
            if not line_polys:
                continue
                
            # ครอปบรรทัด
            cropped_image, bbox = crop_line(image, line_polys)
            
            if cropped_image is not None:
                # บันทึกไฟล์
                crop_filename = f"{base_filename}_line_{line_idx+1}.png"
                crop_path = os.path.join(output_folder, crop_filename)
                cv2.imwrite(crop_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
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
    
    for line_idx, line_polys in enumerate(lines):
        color = colors[line_idx % len(colors)]
        
        # วาด polys ในบรรทัดนี้
        for poly in line_polys:
            if poly is not None:
                poly_int = np.array(poly, dtype=np.int32)
                cv2.polylines(result_image, [poly_int], True, color, 2)
        
        # หา bounding box ของบรรทัด
        if line_polys:
            valid_polys = [p for p in line_polys if p is not None]
            if valid_polys:
                all_points = np.vstack(valid_polys)
                x_min = int(np.min(all_points[:, 0]))
                y_min = int(np.min(all_points[:, 1]))
                x_max = int(np.max(all_points[:, 0]))
                y_max = int(np.max(all_points[:, 1]))
                
                # ปรับความสูงให้เท่ากัน
                center_y = (y_min + y_max) // 2
                y_min = max(0, center_y - uniform_height // 2)
                y_max = min(image_height, center_y + uniform_height // 2)
                
                # ตรวจสอบการทับกันกับบรรทัดที่วาดแล้ว
                adjusted_bbox = adjust_bbox_to_avoid_overlap(
                    (x_min, y_min, x_max, y_max), drawn_bboxes
                )
                
                if adjusted_bbox:
                    x_min, y_min, x_max, y_max = adjusted_bbox
                    
                    # วาดกรอบบรรทัด
                    cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), color, 3)
                    
                    # เพิ่มข้อความแสดงหมายเลขบรรทัด
                    cv2.putText(result_image, f"Line {line_idx + 1}", 
                               (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, color, 2)
                    
                    # เก็บ bbox ที่วาดแล้ว
                    drawn_bboxes.append((x_min, y_min, x_max, y_max))
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
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

class CRAFTLineDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CRAFT Line Detection Tool")
        self.root.geometry("1400x900")
        
        # Model variables
        self.net = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # UI variables
        self.selected_folder = tk.StringVar()
        self.processing = False
        self.current_image = None
        self.current_image_path = None
        
        # Image navigation variables
        self.image_files = []
        self.current_image_index = -1
        self.detection_results = {}  # Store detection results for each image
        
        # Settings
        self.settings = {
            'text_threshold': 0.7,
            'low_text': 0.4,
            'link_threshold': 0.4,
            'padding': 10,
            'tolerance': 10,  # ลดจาก 20 เป็น 10
            'save_results': True,
            'output_folder': 'line_detection_results',
            'model_path': r'C:\Users\Win 10 Home\CRAFT-pytorch\craft_mlt_25k.pth',
            'uniform_height': True,
            'max_height_ratio': 0.1
        }
        
        self.setup_ui()
        self.setup_keyboard_events()
        self.load_model()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="CRAFT Line Detection Tool", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model path
        ttk.Label(settings_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.model_path_var = tk.StringVar(value=self.settings['model_path'])
        model_path_entry = ttk.Entry(settings_frame, textvariable=self.model_path_var, width=50)
        model_path_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(settings_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5, pady=2)
        
        # Thresholds
        ttk.Label(settings_frame, text="Text Threshold:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.text_threshold_var = tk.DoubleVar(value=self.settings['text_threshold'])
        text_threshold_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.text_threshold_var, 
                                       orient=tk.HORIZONTAL, length=150)
        text_threshold_scale.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(settings_frame, textvariable=self.text_threshold_var).grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Link Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.link_threshold_var = tk.DoubleVar(value=self.settings['link_threshold'])
        link_threshold_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.link_threshold_var, 
                                       orient=tk.HORIZONTAL, length=150)
        link_threshold_scale.grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(settings_frame, textvariable=self.link_threshold_var).grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Padding:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.padding_var = tk.IntVar(value=self.settings['padding'])
        padding_scale = ttk.Scale(settings_frame, from_=0, to=50, variable=self.padding_var, 
                                orient=tk.HORIZONTAL, length=150)
        padding_scale.grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(settings_frame, textvariable=self.padding_var).grid(row=3, column=2, padx=5, pady=2)

        # Tolerance
        ttk.Label(settings_frame, text="Line Tolerance:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.tolerance_var = tk.IntVar(value=self.settings['tolerance'])
        tolerance_scale = ttk.Scale(settings_frame, from_=5, to=50, variable=self.tolerance_var, 
                                    orient=tk.HORIZONTAL, length=150)
        tolerance_scale.grid(row=4, column=1, padx=5, pady=2)
        ttk.Label(settings_frame, textvariable=self.tolerance_var).grid(row=4, column=2, padx=5, pady=2)
        
        # Checkboxes
        self.save_results_var = tk.BooleanVar(value=self.settings['save_results'])
        ttk.Checkbutton(settings_frame, text="Save Results", 
                       variable=self.save_results_var).grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.uniform_height_var = tk.BooleanVar(value=self.settings['uniform_height'])
        ttk.Checkbutton(settings_frame, text="Uniform Height", 
                       variable=self.uniform_height_var).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Output folder
        ttk.Label(settings_frame, text="Output Folder:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        self.output_folder_var = tk.StringVar(value=self.settings['output_folder'])
        output_folder_entry = ttk.Entry(settings_frame, textvariable=self.output_folder_var, width=50)
        output_folder_entry.grid(row=6, column=1, padx=5, pady=2)
        ttk.Button(settings_frame, text="Browse", command=self.browse_output_folder).grid(row=6, column=2, padx=5, pady=2)
        
        # Folder selection
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        folder_frame.columnconfigure(1, weight=1)
        
        ttk.Label(folder_frame, text="Select Image Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        folder_entry = ttk.Entry(folder_frame, textvariable=self.selected_folder, width=50)
        folder_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.process_button = ttk.Button(button_frame, text="Process Folder", 
                                       command=self.process_folder, state="disabled")
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Processing", 
                                    command=self.stop_processing, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=(0, 10))
        
        # Navigation buttons
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        self.prev_button = ttk.Button(nav_frame, text="← Previous", command=self.previous_image, state="disabled")
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.image_info_label = ttk.Label(nav_frame, text="No images loaded")
        self.image_info_label.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(nav_frame, text="Next →", command=self.next_image, state="disabled")
        self.next_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        results_frame.columnconfigure(0, weight=2)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Text results
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        ttk.Label(text_frame, text="Detection Results:").grid(row=0, column=0, sticky=tk.W)
        self.text_results = scrolledtext.ScrolledText(text_frame, height=20, width=70)
        self.text_results.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image preview
        image_frame = ttk.Frame(results_frame)
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(1, weight=1)
        
        ttk.Label(image_frame, text="Image Preview:").grid(row=0, column=0, sticky=tk.W)
        self.image_label = ttk.Label(image_frame, text="No image selected", 
                                   border=1, relief="solid")
        self.image_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure results frame weights
        results_frame.columnconfigure(0, weight=2)
        results_frame.columnconfigure(1, weight=1)
    
    def setup_keyboard_events(self):
        """Setup keyboard event handlers"""
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Up>', lambda e: self.previous_image())
        self.root.bind('<Down>', lambda e: self.next_image())
        self.root.bind('<space>', lambda e: self.process_current_image())
        self.root.focus_set()
    
    def browse_model(self):
        """Browse for model file"""
        model_file = filedialog.askopenfilename(
            title="Select CRAFT Model File",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if model_file:
            self.model_path_var.set(model_file)
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.selected_folder.set(folder)
            self.load_image_files()
            self.process_button.config(state="normal")
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)
            self.settings['output_folder'] = folder
    
    def load_model(self):
        """Load CRAFT model"""
        try:
            model_path = self.settings['model_path']
            if os.path.exists(model_path):
                self.net = load_craft_model(model_path, use_cuda=self.device.type == 'cuda')
                self.status_var.set("Model loaded successfully")
            else:
                self.status_var.set("Model file not found")
        except Exception as e:
            self.status_var.set(f"Failed to load model: {str(e)}")
    
    def load_image_files(self):
        """Load image files from selected folder"""
        folder_path = self.selected_folder.get()
        if not folder_path:
            return
        
        self.image_files = self.get_image_files(folder_path)
        self.current_image_index = -1
        self.detection_results = {}
        
        if self.image_files:
            self.current_image_index = 0
            self.update_navigation_buttons()
            self.show_current_image()
        else:
            self.image_info_label.config(text="No images found")
            self.update_navigation_buttons()
    
    def get_image_files(self, folder_path):
        """Get all image files from folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file in os.listdir(folder_path):
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(folder_path, file))
        
        return sorted(image_files)
    
    def update_navigation_buttons(self):
        """Update navigation button states"""
        if not self.image_files:
            self.prev_button.config(state="disabled")
            self.next_button.config(state="disabled")
            return
        
        if self.current_image_index <= 0:
            self.prev_button.config(state="disabled")
        else:
            self.prev_button.config(state="normal")
        
        if self.current_image_index >= len(self.image_files) - 1:
            self.next_button.config(state="disabled")
        else:
            self.next_button.config(state="normal")
        
        if self.image_files:
            self.image_info_label.config(text=f"Image {self.current_image_index + 1} of {len(self.image_files)}")
    
    def show_current_image(self):
        """Show the current image with detection results"""
        if not self.image_files or self.current_image_index < 0:
            return
        
        current_image_path = self.image_files[self.current_image_index]
        
        # Show detection results if available
        if current_image_path in self.detection_results:
            self.display_results(current_image_path)
            self.update_image_preview_with_detection(current_image_path)
        else:
            self.update_image_preview(current_image_path)
            self.text_results.delete(1.0, tk.END)
            self.text_results.insert(tk.END, "No detection results available. Press Space to process this image.")
    
    def previous_image(self):
        """Go to previous image"""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_navigation_buttons()
            self.show_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.update_navigation_buttons()
            self.show_current_image()
    
    def process_current_image(self):
        """Process the current image"""
        if not self.image_files or self.current_image_index < 0:
            return
        
        # Update settings from UI
        self.settings['text_threshold'] = self.text_threshold_var.get()
        self.settings['link_threshold'] = self.link_threshold_var.get()
        self.settings['padding'] = self.padding_var.get()
        self.settings['tolerance'] = self.tolerance_var.get()
        self.settings['save_results'] = self.save_results_var.get()
        self.settings['uniform_height'] = self.uniform_height_var.get()
        self.settings['output_folder'] = self.output_folder_var.get()
        
        current_image_path = self.image_files[self.current_image_index]
        result_text = self.process_single_image(current_image_path)
        
        # Store results
        self.detection_results[current_image_path] = result_text
        
        # Display results
        self.display_results(current_image_path)
        self.update_image_preview_with_detection(current_image_path)
    
    def display_results(self, image_path):
        """Display results for a specific image"""
        if image_path in self.detection_results:
            self.text_results.delete(1.0, tk.END)
            self.text_results.insert(tk.END, f"=== {os.path.basename(image_path)} ===\n")
            self.text_results.insert(tk.END, self.detection_results[image_path] + "\n")
    
    def process_folder(self):
        """Process all images in the selected folder"""
        if not self.selected_folder.get():
            messagebox.showerror("Error", "Please select a folder first")
            return
        
        if self.processing:
            return
        
        # Update settings from UI
        self.settings['text_threshold'] = self.text_threshold_var.get()
        self.settings['link_threshold'] = self.link_threshold_var.get()
        self.settings['padding'] = self.padding_var.get()
        self.settings['tolerance'] = self.tolerance_var.get() # Update tolerance
        self.settings['save_results'] = self.save_results_var.get()
        self.settings['model_path'] = self.model_path_var.get()
        self.settings['uniform_height'] = self.uniform_height_var.get()
        self.settings['output_folder'] = self.output_folder_var.get()
        
        self.processing = True
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self._process_folder_thread)
        thread.daemon = True
        thread.start()
    
    def _process_folder_thread(self):
        """Thread function for processing folder"""
        try:
            folder_path = self.selected_folder.get()
            image_files = self.get_image_files(folder_path)
            
            if not image_files:
                self.status_var.set("No image files found in folder")
                return
            
            self.text_results.delete(1.0, tk.END)
            total_files = len(image_files)
            
            # Create output folder if saving results
            if self.settings['save_results']:
                os.makedirs(self.settings['output_folder'], exist_ok=True)
            
            for i, image_file in enumerate(image_files):
                if not self.processing:  # Check if stopped
                    break
                
                self.status_var.set(f"Processing {i+1}/{total_files}: {os.path.basename(image_file)}")
                self.progress_var.set((i / total_files) * 100)
                self.root.update()
                
                # Process single image
                result_text = self.process_single_image(image_file)
                
                # Store results
                self.detection_results[image_file] = result_text
                
                # Update results
                self.text_results.insert(tk.END, f"=== {os.path.basename(image_file)} ===\n")
                self.text_results.insert(tk.END, result_text + "\n\n")
                self.text_results.see(tk.END)
                self.root.update()
            
            self.progress_var.set(100)
            self.status_var.set("Processing completed")
            
            # Show completion message with output folder
            if self.settings['save_results']:
                output_folder_abs = os.path.abspath(self.settings['output_folder'])
                self.text_results.insert(tk.END, f"\n=== Processing Completed ===\n")
                self.text_results.insert(tk.END, f"Results saved in: {output_folder_abs}\n")
                self.text_results.see(tk.END)
            
            # Show first image after processing
            if self.image_files:
                self.current_image_index = 0
                self.update_navigation_buttons()
                self.show_current_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Processing failed")
        finally:
            self.processing = False
            self.process_button.config(state="normal")
            self.stop_button.config(state="disabled")
    
    def process_single_image(self, image_path):
        """Process a single image with CRAFT line detection"""
        try:
            if self.net is None:
                return "Error: Model not loaded"
            
            # Load image
            image = imgproc.loadImage(image_path)
            if image is None:
                return f"Error: Could not read image {image_path}"
            
            # Detect text regions
            boxes, polys, score_text = test_net(
                self.net, image,
                self.settings['text_threshold'],
                self.settings['link_threshold'],
                self.settings['low_text'],
                self.device.type == 'cuda',
                False,  # poly
                show_time=False
            )
            
            if len(polys) == 0:
                return "No text detected in image"
            
            # Group polys into lines
            lines = group_lines_fine_tuned(polys, self.settings['tolerance'], max_lines=3)
            
            result_text = []
            result_text.append(f"Detected {len(polys)} text regions")
            result_text.append(f"Grouped into {len(lines)} lines")
            result_text.append("")
            
            # Process each line
            for line_idx, line_polys in enumerate(lines):
                result_text.append(f"Line {line_idx + 1}: {len(line_polys)} characters")
                
                # Crop line
                cropped_line, bbox = crop_line(image, line_polys, self.settings['padding'], 
                                             self.settings['max_height_ratio'], 
                                             self.settings['uniform_height'])
                
                if cropped_line is not None:
                    # Save cropped line if enabled
                    if self.settings['save_results']:
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        line_filename = f"{base_name}_line_{line_idx + 1}.png"
                        line_path = os.path.join(self.settings['output_folder'], line_filename)
                        cv2.imwrite(line_path, cv2.cvtColor(cropped_line, cv2.COLOR_RGB2BGR))
                        result_text.append(f"  Saved: {line_filename}")
                    
                    result_text.append(f"  BBox: {bbox}")
                
                result_text.append("")
            
            # Save image with detection results if enabled
            if self.settings['save_results'] and len(lines) > 0:
                # Draw detection results
                result_image = draw_detection_result(image, lines)
                
                # Save annotated image
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                annotated_filename = f"{base_name}_annotated.png"
                annotated_path = os.path.join(self.settings['output_folder'], annotated_filename)
                cv2.imwrite(annotated_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                result_text.append(f"Saved annotated image: {annotated_filename}")
                
                # Crop and save detected boxes
                cropped_files = crop_detected_boxes(image, lines, self.settings['output_folder'], base_name)
                if cropped_files:
                    result_text.append(f"Cropped {len(cropped_files)} detected boxes")
                    for crop_file in cropped_files:
                        result_text.append(f"  Saved: {os.path.basename(crop_file)}")
                
                result_text.append(f"Output folder: {os.path.abspath(self.settings['output_folder'])}")
                result_text.append("")
            
            return "\n".join(result_text)
            
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    def update_image_preview(self, image_path):
        """Update the image preview"""
        try:
            image = cv2.imread(image_path)
            if image is not None:
                # Resize to fit preview area
                height, width = image.shape[:2]
                max_size = 400
                
                if height > max_size or width > max_size:
                    scale = min(max_size / height, max_size / width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                
                # Convert to PIL Image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update label
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
        except Exception as e:
            print(f"Error updating image preview: {e}")
    
    def update_image_preview_with_detection(self, image_path):
        """Update the image preview with detection results"""
        try:
            if image_path not in self.detection_results:
                self.update_image_preview(image_path)
                return
            
            # Load original image
            image = imgproc.loadImage(image_path)
            if image is None:
                self.update_image_preview(image_path)
                return
            
            # Get detection results
            if self.net is None:
                self.update_image_preview(image_path)
                return
            
            # Detect text regions
            boxes, polys, score_text = test_net(
                self.net, image,
                self.settings['text_threshold'],
                self.settings['link_threshold'],
                self.settings['low_text'],
                self.device.type == 'cuda',
                False,
                show_time=False
            )
            
            # ตรวจสอบ polys ว่าถูกต้องหรือไม่
            if polys is None or len(polys) == 0:
                self.update_image_preview(image_path)
                return
            
            # ตรวจสอบ polys แต่ละตัว
            valid_polys = []
            for poly in polys:
                if poly is not None and len(poly) > 0:
                    try:
                        # ตรวจสอบว่า poly มี shape ที่ถูกต้อง
                        if poly.shape[0] >= 4:  # ต้องมีจุดอย่างน้อย 4 จุด
                            valid_polys.append(poly)
                    except (IndexError, AttributeError):
                        continue
            
            if len(valid_polys) == 0:
                self.update_image_preview(image_path)
                return
            
            # Group into lines
            lines = group_lines_fine_tuned(valid_polys, self.settings['tolerance'], max_lines=3)
            
            if len(lines) > 0:
                # Draw detection results
                result_image = draw_detection_result(image, lines)
                
                # Save annotated image if enabled
                if self.settings['save_results']:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    annotated_filename = f"{base_name}_annotated.png"
                    annotated_path = os.path.join(self.settings['output_folder'], annotated_filename)
                    cv2.imwrite(annotated_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                
                # Resize for preview
                height, width = result_image.shape[:2]
                max_size = 400
                
                if height > max_size or width > max_size:
                    scale = min(max_size / height, max_size / width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    result_image = cv2.resize(result_image, (new_width, new_height))
                
                # Convert to PIL Image
                pil_image = Image.fromarray(result_image)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update label
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
            else:
                self.update_image_preview(image_path)
                
        except Exception as e:
            print(f"Error updating image preview with detection: {e}")
            self.update_image_preview(image_path)
    
    def stop_processing(self):
        """Stop the processing"""
        self.processing = False
        self.status_var.set("Processing stopped")
    
    def clear_results(self):
        """Clear the results text area"""
        self.text_results.delete(1.0, tk.END)
        self.status_var.set("Results cleared")

def main():
    root = tk.Tk()
    app = CRAFTLineDetectionUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

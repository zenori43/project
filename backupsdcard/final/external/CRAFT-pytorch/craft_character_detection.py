#!/usr/bin/env python3
"""
CRAFT Character Detection Tool
ใช้ CRAFT เพื่อจับตัวอักษรแต่ละตัวในภาพ และครอปเป็นรูปๆ ละ
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
from PIL import Image
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

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

from collections import OrderedDict

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

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
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

    if args.show_time:
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

def crop_character(image, poly, padding=5):
    """ครอปตัวอักษรจากกรอบที่กำหนด"""
    # แปลง poly เป็น box
    poly = np.array(poly, dtype=np.int32)
    
    # หา bounding box
    x_min = max(0, int(np.min(poly[:, 0])) - padding)
    y_min = max(0, int(np.min(poly[:, 1])) - padding)
    x_max = min(image.shape[1], int(np.max(poly[:, 0])) + padding)
    y_max = min(image.shape[0], int(np.max(poly[:, 1])) + padding)
    
    # ครอปภาพ
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped, (x_min, y_min, x_max, y_max)

def process_folder(folder_path, output_dir="character_crops"):
    """ประมวลผลทุกไฟล์ในโฟลเดอร์"""
    
    # สร้างโฟลเดอร์ผลลัพธ์
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # หาไฟล์ภาพในโฟลเดอร์
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        print(f"ไม่พบไฟล์ภาพในโฟลเดอร์: {folder_path}")
        return
    
    print(f"พบไฟล์ภาพ {len(image_files)} ไฟล์")
    
    # โหลดโมเดล CRAFT
    model_path = r"C:\Users\Win 10 Home\CRAFT-pytorch\craft_mlt_25k.pth"
    if not os.path.exists(model_path):
        print(f"ไม่พบไฟล์โมเดล: {model_path}")
        print("กรุณาตรวจสอบว่าไฟล์โมเดลอยู่ในตำแหน่งที่ถูกต้อง")
        return
    
    try:
        net = load_craft_model(model_path, use_cuda=args.cuda)
    except Exception as e:
        print(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return
    
    total_characters = 0
    
    # ประมวลผลแต่ละไฟล์
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] กำลังประมวลผล: {os.path.basename(image_path)}")
        
        # โหลดภาพ
        image = imgproc.loadImage(image_path)
        if image is None:
            print(f"ไม่สามารถโหลดภาพได้: {image_path}")
            continue
        
        # ตรวจจับตัวอักษร
        boxes, polys, score_text = test_net(
            net, image, 
            args.text_threshold, 
            args.link_threshold, 
            args.low_text, 
            args.cuda, 
            args.poly
        )
        
        if len(polys) > 0:
            print(f"พบตัวอักษร {len(polys)} ตัว")
            
            # สร้างโฟลเดอร์สำหรับไฟล์นี้
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            file_output_dir = os.path.join(output_dir, base_name)
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)
            
            # ครอปและบันทึกแต่ละตัวอักษร
            for j, poly in enumerate(polys):
                if poly is None:
                    continue
                
                try:
                    # ครอปตัวอักษร
                    cropped_char, bbox = crop_character(image, poly, padding=args.padding)
                    
                    if cropped_char.size > 0:  # ตรวจสอบว่าครอปได้หรือไม่
                        # บันทึกรูปครอป
                        char_filename = f"char_{j+1:03d}.png"
                        char_path = os.path.join(file_output_dir, char_filename)
                        cv2.imwrite(char_path, cv2.cvtColor(cropped_char, cv2.COLOR_RGB2BGR))
                        
                        # บันทึกข้อมูล
                        char_info = {
                            'id': j+1,
                            'original_image': os.path.basename(image_path),
                            'cropped_file': char_filename,
                            'bbox': bbox,
                            'polygon': poly.tolist(),
                            'area': cv2.contourArea(np.array(poly, dtype=np.int32))
                        }
                        
                        # บันทึกข้อมูล JSON สำหรับตัวอักษรนี้
                        info_filename = f"char_{j+1:03d}.json"
                        info_path = os.path.join(file_output_dir, info_filename)
                        with open(info_path, 'w', encoding='utf-8') as f:
                            json.dump(char_info, f, indent=2, ensure_ascii=False)
                        
                        print(f"  บันทึกตัวอักษร {j+1}: {char_filename}")
                        total_characters += 1
                
                except Exception as e:
                    print(f"  เกิดข้อผิดพลาดในการครอปตัวอักษร {j+1}: {e}")
        else:
            print("ไม่พบตัวอักษรในภาพ")
    
    print(f"\nสรุป: ครอปตัวอักษรทั้งหมด {total_characters} ตัว")
    print(f"ผลลัพธ์อยู่ในโฟลเดอร์: {output_dir}")

def main():
    """Main function"""
    global args
    
    # ตั้งค่า arguments
    parser = argparse.ArgumentParser(description='CRAFT Character Detection and Cropping')
    parser.add_argument('--trained_model', default=r'C:\Users\tiwat\CRAFT-pytorch\weights\craft_mlt_25k.pth', 
                       type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False , type=str2bool, help='Use cuda for inference')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type result')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--padding', default=5, type=int, help='padding around cropped characters')
    parser.add_argument('--folder_path', type=str, help='path to input folder')
    
    args = parser.parse_args()
    
    # ถ้าไม่ได้ระบุ folder_path ให้ใช้ GUI เลือกโฟลเดอร์
    if not args.folder_path:
        root = tk.Tk()
        root.withdraw()
        
        folder_path = filedialog.askdirectory(
            title="เลือกโฟลเดอร์ที่มีรูปภาพ"
        )
        
        if not folder_path:
            print("ไม่ได้เลือกโฟลเดอร์, ยกเลิกการทำงาน")
            return
    else:
        folder_path = args.folder_path
    
    print("="*50)
    print("CRAFT Character Detection and Cropping Tool")
    print("="*50)
    print(f"โฟลเดอร์ภาพ: {folder_path}")
    print(f"ใช้ CUDA: {args.cuda}")
    print(f"Padding: {args.padding} pixels")
    print("="*50)
    
    # ประมวลผลโฟลเดอร์
    process_folder(folder_path)

if __name__ == '__main__':
    main()

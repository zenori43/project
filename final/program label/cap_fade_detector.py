# -*- coding: utf-8 -*-
"""
Cap Fade Detector
โปรแกรมตรวจจับการ fade ของฝา พร้อมการครอปฝา
"""

import os
import sys

# CRITICAL: Setup CUDA paths BEFORE importing any libs modules
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    pass

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Dict
import copy

# Import cap detection and faded text detection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.detection.capmodel import CapDetector
from core.image_processor import detect_faded_text_in_cap
from config.settings import CAP_MODEL_PATH, FADED_TEXT_CONFIG
from core.cuda_image_utils import cuda_cvtColor, cuda_gaussianBlur


class CapFadeDetector:
    """โปรแกรมตรวจจับการ fade ของฝา"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Cap Fade Detector - ตรวจจับการ Fade ของฝา")
        self.root.geometry("1600x1000")
        
        # Initialize detector
        self.cap_detector = None
        self.current_image = None
        self.current_image_path = None
        self.cropped_caps = []  # List of cropped cap images
        self.fade_results = []  # List of fade detection results
        
        # Settings
        self.conf_threshold = 0.5
        self.crop_margin = 10
        
        # Fade detection config (copy from settings, can be modified)
        self.fade_config = copy.deepcopy(FADED_TEXT_CONFIG)
        
        self.setup_gui()
        self.load_cap_model()
    
    def setup_gui(self):
        """Setup GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Cap Fade Detector - ตรวจจับการ Fade ของฝา", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.configure(width=350)
        
        # Control buttons frame
        control_frame = ttk.LabelFrame(left_panel, text="ตัวเลือก", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Image selection button
        self.select_btn = ttk.Button(control_frame, text="เลือกรูปภาพ", 
                                    command=self.select_image)
        self.select_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Folder selection button
        self.select_folder_btn = ttk.Button(control_frame, text="เลือกโฟลเดอร์", 
                                          command=self.select_folder)
        self.select_folder_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Detect and crop button
        self.detect_btn = ttk.Button(control_frame, text="ตรวจจับและครอปฝา", 
                                    command=self.detect_and_crop, state=tk.DISABLED)
        self.detect_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Detect fade button
        self.detect_fade_btn = ttk.Button(control_frame, text="ตรวจจับการ Fade", 
                                         command=self.detect_fade, state=tk.DISABLED)
        self.detect_fade_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Settings frame - Cap Detection
        cap_settings_frame = ttk.LabelFrame(left_panel, text="ตั้งค่าการตรวจจับฝา", padding="10")
        cap_settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence threshold
        ttk.Label(cap_settings_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(cap_settings_frame, from_=0.1, to=1.0, 
                               variable=self.conf_var, orient=tk.HORIZONTAL)
        conf_scale.pack(fill=tk.X, pady=(0, 5))
        self.conf_label = ttk.Label(cap_settings_frame, text="0.50")
        self.conf_label.pack(anchor=tk.W)
        conf_scale.configure(command=lambda v: self.conf_label.config(text=f"{float(v):.2f}"))
        
        # Crop margin
        ttk.Label(cap_settings_frame, text="Crop Margin (pixels):").pack(anchor=tk.W)
        self.margin_var = tk.IntVar(value=10)
        margin_scale = ttk.Scale(cap_settings_frame, from_=0, to=50, 
                                variable=self.margin_var, orient=tk.HORIZONTAL)
        margin_scale.pack(fill=tk.X, pady=(0, 5))
        self.margin_label = ttk.Label(cap_settings_frame, text="10")
        self.margin_label.pack(anchor=tk.W)
        margin_scale.configure(command=lambda v: self.margin_label.config(text=f"{int(v)}"))
        
        # Settings frame - Fade Detection
        fade_settings_frame = ttk.LabelFrame(left_panel, text="ตั้งค่าการตรวจจับ Fade", padding="10")
        fade_settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # MIN_AREA
        ttk.Label(fade_settings_frame, text="MIN_AREA (noise filter):").pack(anchor=tk.W)
        self.min_area_var = tk.IntVar(value=self.fade_config.get('MIN_AREA', -1))
        min_area_entry = ttk.Entry(fade_settings_frame, textvariable=self.min_area_var, width=15)
        min_area_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # AREA_THRESH
        ttk.Label(fade_settings_frame, text="AREA_THRESH (fade threshold):").pack(anchor=tk.W)
        self.area_thresh_var = tk.IntVar(value=self.fade_config.get('AREA_THRESH', -1))
        area_thresh_entry = ttk.Entry(fade_settings_frame, textvariable=self.area_thresh_var, width=15)
        area_thresh_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # CIRCLE_FALLBACK_MARGIN
        ttk.Label(fade_settings_frame, text="CIRCLE_FALLBACK_MARGIN:").pack(anchor=tk.W)
        self.circle_margin_var = tk.IntVar(value=self.fade_config.get('CIRCLE_FALLBACK_MARGIN', 6))
        circle_margin_entry = ttk.Entry(fade_settings_frame, textvariable=self.circle_margin_var, width=15)
        circle_margin_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # Image Enhancement Settings
        enhance_frame = ttk.LabelFrame(fade_settings_frame, text="ปรับปรุงภาพ", padding="5")
        enhance_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Brightness adjustment
        ttk.Label(enhance_frame, text="Brightness (เพิ่มแสง):").pack(anchor=tk.W)
        self.brightness_var = tk.IntVar(value=0)
        brightness_scale = ttk.Scale(enhance_frame, from_=-100, to=100, 
                                    variable=self.brightness_var, orient=tk.HORIZONTAL)
        brightness_scale.pack(fill=tk.X, pady=(0, 5))
        self.brightness_label = ttk.Label(enhance_frame, text="0")
        self.brightness_label.pack(anchor=tk.W)
        brightness_scale.configure(command=lambda v: self.brightness_label.config(text=f"{int(v)}"))
        
        # Contrast adjustment
        ttk.Label(enhance_frame, text="Contrast (ความคมชัด):").pack(anchor=tk.W)
        # ใช้ค่า contrast จาก config (1.57) เป็น default
        default_contrast = self.fade_config.get('IMAGE_ENHANCEMENT', {}).get('contrast', 1.57)
        self.contrast_var = tk.DoubleVar(value=default_contrast)
        contrast_scale = ttk.Scale(enhance_frame, from_=0.5, to=3.0, 
                                  variable=self.contrast_var, orient=tk.HORIZONTAL)
        contrast_scale.pack(fill=tk.X, pady=(0, 5))
        self.contrast_label = ttk.Label(enhance_frame, text=f"{default_contrast:.2f}")
        self.contrast_label.pack(anchor=tk.W)
        contrast_scale.configure(command=lambda v: self.contrast_label.config(text=f"{float(v):.2f}"))
        
        # Gamma correction
        ttk.Label(enhance_frame, text="Gamma Correction:").pack(anchor=tk.W)
        self.gamma_var = tk.DoubleVar(value=1.0)
        gamma_scale = ttk.Scale(enhance_frame, from_=0.3, to=3.0, 
                               variable=self.gamma_var, orient=tk.HORIZONTAL)
        gamma_scale.pack(fill=tk.X, pady=(0, 5))
        self.gamma_label = ttk.Label(enhance_frame, text="1.0")
        self.gamma_label.pack(anchor=tk.W)
        gamma_scale.configure(command=lambda v: self.gamma_label.config(text=f"{float(v):.2f}"))
        
        # Use adaptive threshold checkbox
        # ใช้ค่า USE_ADAPTIVE_THRESHOLD จาก config เป็น default
        default_use_adaptive = self.fade_config.get('USE_ADAPTIVE_THRESHOLD', True)
        self.use_adaptive_var = tk.BooleanVar(value=default_use_adaptive)
        adaptive_check = ttk.Checkbutton(enhance_frame, text="ใช้ Adaptive Threshold", 
                                        variable=self.use_adaptive_var)
        adaptive_check.pack(anchor=tk.W, pady=(5, 0))
        
        # Canny thresholds
        ttk.Label(fade_settings_frame, text="Canny Low Threshold:").pack(anchor=tk.W)
        self.canny_low_var = tk.IntVar(value=self.fade_config['CANNY_PARAMS']['low_threshold'])
        canny_low_entry = ttk.Entry(fade_settings_frame, textvariable=self.canny_low_var, width=15)
        canny_low_entry.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(fade_settings_frame, text="Canny High Threshold:").pack(anchor=tk.W)
        self.canny_high_var = tk.IntVar(value=self.fade_config['CANNY_PARAMS']['high_threshold'])
        canny_high_entry = ttk.Entry(fade_settings_frame, textvariable=self.canny_high_var, width=15)
        canny_high_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # Gaussian Blur
        ttk.Label(fade_settings_frame, text="Gaussian Blur Kernel Size:").pack(anchor=tk.W)
        self.blur_kernel_var = tk.IntVar(value=self.fade_config['GAUSSIAN_BLUR_PARAMS']['kernel_size'][0])
        blur_kernel_entry = ttk.Entry(fade_settings_frame, textvariable=self.blur_kernel_var, width=15)
        blur_kernel_entry.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(fade_settings_frame, text="Gaussian Blur Sigma:").pack(anchor=tk.W)
        self.blur_sigma_var = tk.DoubleVar(value=self.fade_config['GAUSSIAN_BLUR_PARAMS']['sigma'])
        blur_sigma_entry = ttk.Entry(fade_settings_frame, textvariable=self.blur_sigma_var, width=15)
        blur_sigma_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # Hough Circles params
        hough_frame = ttk.LabelFrame(fade_settings_frame, text="Hough Circles", padding="5")
        hough_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(hough_frame, text="dp:").pack(anchor=tk.W)
        self.hough_dp_var = tk.DoubleVar(value=self.fade_config['HOUGH_CIRCLES_PARAMS']['dp'])
        ttk.Entry(hough_frame, textvariable=self.hough_dp_var, width=10).pack(anchor=tk.W, pady=(0, 2))
        
        ttk.Label(hough_frame, text="minDist:").pack(anchor=tk.W)
        self.hough_mindist_var = tk.IntVar(value=self.fade_config['HOUGH_CIRCLES_PARAMS']['minDist'])
        ttk.Entry(hough_frame, textvariable=self.hough_mindist_var, width=10).pack(anchor=tk.W, pady=(0, 2))
        
        ttk.Label(hough_frame, text="param1:").pack(anchor=tk.W)
        self.hough_param1_var = tk.IntVar(value=self.fade_config['HOUGH_CIRCLES_PARAMS']['param1'])
        ttk.Entry(hough_frame, textvariable=self.hough_param1_var, width=10).pack(anchor=tk.W, pady=(0, 2))
        
        ttk.Label(hough_frame, text="param2:").pack(anchor=tk.W)
        self.hough_param2_var = tk.IntVar(value=self.fade_config['HOUGH_CIRCLES_PARAMS']['param2'])
        ttk.Entry(hough_frame, textvariable=self.hough_param2_var, width=10).pack(anchor=tk.W, pady=(0, 2))
        
        ttk.Label(hough_frame, text="minRadius:").pack(anchor=tk.W)
        self.hough_minr_var = tk.IntVar(value=self.fade_config['HOUGH_CIRCLES_PARAMS']['minRadius'])
        ttk.Entry(hough_frame, textvariable=self.hough_minr_var, width=10).pack(anchor=tk.W, pady=(0, 2))
        
        ttk.Label(hough_frame, text="maxRadius:").pack(anchor=tk.W)
        self.hough_maxr_var = tk.IntVar(value=self.fade_config['HOUGH_CIRCLES_PARAMS']['maxRadius'])
        ttk.Entry(hough_frame, textvariable=self.hough_maxr_var, width=10).pack(anchor=tk.W, pady=(0, 2))
        
        # Results summary
        summary_frame = ttk.LabelFrame(left_panel, text="สรุปผล", padding="10")
        summary_frame.pack(fill=tk.BOTH, expand=True)
        
        self.summary_text = tk.Text(summary_frame, height=15, width=40, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.configure(yscrollcommand=scrollbar.set)
        
        # Right panel - Display
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Image display tab
        image_tab = ttk.Frame(self.notebook)
        self.notebook.add(image_tab, text="ภาพ")
        
        image_scroll = ttk.Scrollbar(image_tab, orient=tk.VERTICAL)
        image_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_canvas = tk.Canvas(image_tab, yscrollcommand=image_scroll.set)
        image_scroll.config(command=self.image_canvas.yview)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Cropped caps tab
        caps_tab = ttk.Frame(self.notebook)
        self.notebook.add(caps_tab, text="ฝาที่ครอป")
        
        caps_scroll = ttk.Scrollbar(caps_tab, orient=tk.VERTICAL)
        caps_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.caps_canvas = tk.Canvas(caps_tab, yscrollcommand=caps_scroll.set)
        caps_scroll.config(command=self.caps_canvas.yview)
        self.caps_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Fade detection results tab
        fade_tab = ttk.Frame(self.notebook)
        self.notebook.add(fade_tab, text="ผลการตรวจจับ Fade")
        
        fade_scroll = ttk.Scrollbar(fade_tab, orient=tk.VERTICAL)
        fade_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.fade_canvas = tk.Canvas(fade_tab, yscrollcommand=fade_scroll.set)
        fade_scroll.config(command=self.fade_canvas.yview)
        self.fade_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Debug images tab
        debug_tab = ttk.Frame(self.notebook)
        self.notebook.add(debug_tab, text="Debug Images")
        
        debug_scroll = ttk.Scrollbar(debug_tab, orient=tk.VERTICAL)
        debug_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.debug_canvas = tk.Canvas(debug_tab, yscrollcommand=debug_scroll.set)
        debug_scroll.config(command=self.debug_canvas.yview)
        self.debug_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def load_cap_model(self):
        """Load cap detection model"""
        try:
            model_path = CAP_MODEL_PATH
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return False
            
            self.cap_detector = CapDetector(model_path=model_path, conf_threshold=self.conf_threshold)
            self.log_info("✅ โหลด Cap Detection Model สำเร็จ")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.log_info(f"❌ ไม่สามารถโหลด Model ได้: {str(e)}")
            return False
    
    def select_image(self):
        """Select single image file"""
        file_path = filedialog.askopenfilename(
            title="เลือกรูปภาพ",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            self.load_image(file_path)
            self.detect_btn.config(state=tk.NORMAL)
    
    def select_folder(self):
        """Select folder with images"""
        folder_path = filedialog.askdirectory(title="เลือกโฟลเดอร์")
        if folder_path:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(folder_path).glob(f"*{ext}"))
                image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
            
            if not image_files:
                messagebox.showwarning("Warning", "ไม่พบไฟล์ภาพในโฟลเดอร์นี้")
                return
            
            self.current_image_path = str(image_files[0])
            self.load_image(self.current_image_path)
            self.detect_btn.config(state=tk.NORMAL)
            self.log_info(f"พบ {len(image_files)} ไฟล์ในโฟลเดอร์ (กำลังใช้ไฟล์แรก)")
    
    def load_image(self, image_path):
        """Load image from file"""
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("ไม่สามารถโหลดภาพได้")
            
            self.display_image(self.current_image)
            self.log_info(f"✅ โหลดภาพ: {os.path.basename(image_path)}")
            self.log_info(f"   ขนาด: {self.current_image.shape[1]}x{self.current_image.shape[0]}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display image in canvas"""
        display_width = 800
        h, w = image.shape[:2]
        scale = display_width / w if w > display_width else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        display_image = cv2.resize(image, (new_w, new_h))
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        try:
            from PIL import Image, ImageTk
        except ImportError:
            import Image
            import ImageTk
        
        pil_image = Image.fromarray(display_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.image = photo
        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
    
    def detect_and_crop(self):
        """Detect caps and crop them"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "กรุณาเลือกรูปภาพก่อน")
            return
        
        if self.cap_detector is None:
            messagebox.showerror("Error", "Cap detector not loaded")
            return
        
        try:
            self.log_info("🔍 กำลังตรวจจับฝา...")
            
            self.cap_detector.conf_threshold = self.conf_var.get()
            self.crop_margin = self.margin_var.get()
            
            result = self.cap_detector.detect_caps(image_array=self.current_image)
            
            if 'error' in result:
                raise Exception(result['error'])
            
            num_caps = result.get('total_detections', 0)
            self.log_info(f"✅ พบฝา {num_caps} ฝา")
            
            if num_caps == 0:
                messagebox.showinfo("Info", "ไม่พบฝาในภาพ")
                return
            
            self.cropped_caps = self.cap_detector.crop_detections(
                image=self.current_image,
                result=result,
                margin=self.crop_margin
            )
            
            self.log_info(f"✅ ครอปฝาได้ {len(self.cropped_caps)} ฝา")
            self.display_cropped_caps()
            self.detect_fade_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.log_info(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    
    def display_cropped_caps(self):
        """Display cropped cap images"""
        self.caps_canvas.delete("all")
        
        if not self.cropped_caps:
            return
        
        y_offset = 10
        max_width = 400
        
        for i, cap_image in enumerate(self.cropped_caps):
            h, w = cap_image.shape[:2]
            scale = max_width / w if w > max_width else 1.0
            new_w, new_h = int(w * scale), int(h * scale)
            display_image = cv2.resize(cap_image, (new_w, new_h))
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            
            try:
                from PIL import Image, ImageTk
            except ImportError:
                import Image
                import ImageTk
            
            pil_image = Image.fromarray(display_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            label_text = f"ฝาที่ {i+1} ({w}x{h})"
            self.caps_canvas.create_text(10, y_offset, anchor=tk.NW, text=label_text, 
                                         font=("Arial", 12, "bold"))
            y_offset += 25
            
            self.caps_canvas.create_image(10, y_offset, anchor=tk.NW, image=photo)
            self.caps_canvas.image_refs = getattr(self.caps_canvas, 'image_refs', [])
            self.caps_canvas.image_refs.append(photo)
            
            y_offset += new_h + 20
        
        self.caps_canvas.config(scrollregion=self.caps_canvas.bbox("all"))
    
    def detect_with_adaptive_threshold(self, cap_image):
        """Detect text using adaptive thresholding (better for low contrast)"""
        try:
            # Convert to grayscale if needed
            if len(cap_image.shape) == 3:
                gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cap_image.copy()
            
            # Apply Gaussian blur
            blur = cv2.GaussianBlur(gray, 
                                   (self.blur_kernel_var.get(), self.blur_kernel_var.get()),
                                   self.blur_sigma_var.get())
            
            # Use adaptive threshold instead of Canny
            # This works better for black text on light background
            adaptive_thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Connected Component Analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cleaned, connectivity=8
            )
            
            stat_fg = stats[1:, :]
            if stat_fg.size > 0:
                areas_all = stat_fg[:, cv2.CC_STAT_AREA]
                min_area = self.min_area_var.get() if self.min_area_var.get() > 0 else 10
                keep_mask = areas_all > min_area
                areas_keep = areas_all[keep_mask]
                num_chars = int(keep_mask.sum())
                total_area = int(areas_keep.sum())
            else:
                num_chars, total_area = 0, 0
            
            # Determine status
            area_thresh = self.area_thresh_var.get() if self.area_thresh_var.get() > 0 else 1000
            status = "faded" if total_area < area_thresh else "normal"
            
            return {
                'status': status,
                'num_chars': num_chars,
                'total_area': total_area,
                'adaptive_thresh': adaptive_thresh,
                'cleaned': cleaned
            }
        except Exception as e:
            self.log_info(f"❌ Error in adaptive threshold: {str(e)}")
            return {'status': 'unknown', 'num_chars': 0, 'total_area': 0}
    
    def update_fade_config(self):
        """Update fade detection config from GUI"""
        self.fade_config['MIN_AREA'] = self.min_area_var.get()
        self.fade_config['AREA_THRESH'] = self.area_thresh_var.get()
        self.fade_config['CIRCLE_FALLBACK_MARGIN'] = self.circle_margin_var.get()
        self.fade_config['CANNY_PARAMS']['low_threshold'] = self.canny_low_var.get()
        self.fade_config['CANNY_PARAMS']['high_threshold'] = self.canny_high_var.get()
        self.fade_config['GAUSSIAN_BLUR_PARAMS']['kernel_size'] = (
            self.blur_kernel_var.get(), self.blur_kernel_var.get()
        )
        self.fade_config['GAUSSIAN_BLUR_PARAMS']['sigma'] = self.blur_sigma_var.get()
        self.fade_config['HOUGH_CIRCLES_PARAMS']['dp'] = self.hough_dp_var.get()
        self.fade_config['HOUGH_CIRCLES_PARAMS']['minDist'] = self.hough_mindist_var.get()
        self.fade_config['HOUGH_CIRCLES_PARAMS']['param1'] = self.hough_param1_var.get()
        self.fade_config['HOUGH_CIRCLES_PARAMS']['param2'] = self.hough_param2_var.get()
        self.fade_config['HOUGH_CIRCLES_PARAMS']['minRadius'] = self.hough_minr_var.get()
        self.fade_config['HOUGH_CIRCLES_PARAMS']['maxRadius'] = self.hough_maxr_var.get()
    
    def enhance_image(self, image):
        """Enhance image with brightness, contrast, and gamma correction"""
        if image is None:
            return None
        
        enhanced = image.copy()
        
        # Apply brightness
        brightness = self.brightness_var.get()
        if brightness != 0:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1, beta=brightness)
        
        # Apply contrast
        contrast = self.contrast_var.get()
        if contrast != 1.0:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=0)
        
        # Apply gamma correction
        gamma = self.gamma_var.get()
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def detect_fade(self):
        """Detect fade in cropped caps"""
        if not self.cropped_caps:
            messagebox.showwarning("Warning", "กรุณาตรวจจับและครอปฝาก่อน")
            return
        
        try:
            self.log_info("🔍 กำลังตรวจจับการ Fade...")
            
            # Update config from GUI
            self.update_fade_config()
            
            # Temporarily modify global config
            import config.settings as settings
            original_config = settings.FADED_TEXT_CONFIG.copy()
            settings.FADED_TEXT_CONFIG.update(self.fade_config)
            
            self.fade_results = []
            
            for i, cap_image in enumerate(self.cropped_caps):
                self.log_info(f"   กำลังตรวจสอบฝาที่ {i+1}...")
                
                # Enhance image before detection
                enhanced_cap = self.enhance_image(cap_image)
                
                # Detect faded text with enhanced image
                fade_result = detect_faded_text_in_cap(
                    enhanced_cap, 
                    yolo_model=None, 
                    show_debug=True
                )
                
                fade_result['cap_index'] = i + 1
                fade_result['cap_image'] = cap_image
                fade_result['enhanced_image'] = enhanced_cap
                self.fade_results.append(fade_result)
                
                # If using adaptive threshold, try alternative detection
                if self.use_adaptive_var.get():
                    self.log_info(f"   ใช้ Adaptive Threshold สำหรับฝาที่ {i+1}...")
                    adaptive_result = self.detect_with_adaptive_threshold(enhanced_cap)
                    fade_result['adaptive_result'] = adaptive_result
                
                status = fade_result.get('status', 'unknown')
                num_chars = fade_result.get('num_chars', 0)
                total_area = fade_result.get('total_area', 0)
                
                self.log_info(f"   ฝาที่ {i+1}: Status={status}, Chars={num_chars}, Area={total_area}")
            
            # Restore original config
            settings.FADED_TEXT_CONFIG = original_config
            
            # Display results
            self.display_fade_results()
            self.display_debug_images()
            self.update_summary()
            
            self.log_info("✅ ตรวจจับการ Fade เสร็จสิ้น")
            
        except Exception as e:
            messagebox.showerror("Error", f"Fade detection failed: {str(e)}")
            self.log_info(f"❌ เกิดข้อผิดพลาด: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_fade_results(self):
        """Display fade detection results"""
        self.fade_canvas.delete("all")
        
        if not self.fade_results:
            return
        
        y_offset = 10
        max_width = 400
        
        for result in self.fade_results:
            cap_image = result['cap_image']
            cap_idx = result['cap_index']
            status = result.get('status', 'unknown')
            num_chars = result.get('num_chars', 0)
            total_area = result.get('total_area', 0)
            
            # Resize for display
            h, w = cap_image.shape[:2]
            scale = max_width / w if w > max_width else 1.0
            new_w, new_h = int(w * scale), int(h * scale)
            display_image = cv2.resize(cap_image.copy(), (new_w, new_h))
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            
            # Draw status label
            status_color = (255, 0, 0) if status == 'faded' else (0, 255, 0)
            status_text = "FADED" if status == 'faded' else "NORMAL"
            
            # Convert back to BGR for drawing
            display_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
            cv2.putText(display_bgr, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            display_image = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            
            try:
                from PIL import Image, ImageTk
            except ImportError:
                import Image
                import ImageTk
            
            pil_image = Image.fromarray(display_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Label
            label_text = f"ฝาที่ {cap_idx}: {status_text} (Chars: {num_chars}, Area: {total_area})"
            self.fade_canvas.create_text(10, y_offset, anchor=tk.NW, text=label_text, 
                                         font=("Arial", 12, "bold"))
            y_offset += 25
            
            self.fade_canvas.create_image(10, y_offset, anchor=tk.NW, image=photo)
            self.fade_canvas.image_refs = getattr(self.fade_canvas, 'image_refs', [])
            self.fade_canvas.image_refs.append(photo)
            
            y_offset += new_h + 30
        
        self.fade_canvas.config(scrollregion=self.fade_canvas.bbox("all"))
    
    def display_debug_images(self):
        """Display debug images from fade detection"""
        self.debug_canvas.delete("all")
        
        if not self.fade_results:
            return
        
        y_offset = 10
        max_width = 300
        
        for result in self.fade_results:
            if 'debug_images' not in result:
                continue
            
            cap_idx = result['cap_index']
            debug_images = result['debug_images']
            
            # Title
            title_text = f"ฝาที่ {cap_idx} - Debug Images"
            self.debug_canvas.create_text(10, y_offset, anchor=tk.NW, text=title_text, 
                                         font=("Arial", 14, "bold"))
            y_offset += 30
            
            # Display enhanced image if available
            if 'enhanced_image' in result:
                enhanced_img = result['enhanced_image']
                if len(enhanced_img.shape) == 3:
                    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                elif len(enhanced_img.shape) == 2:
                    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
                
                h, w = enhanced_img.shape[:2]
                scale = max_width / w if w > max_width else 1.0
                new_w, new_h = int(w * scale), int(h * scale)
                display_img = cv2.resize(enhanced_img, (new_w, new_h))
                
                try:
                    from PIL import Image, ImageTk
                except ImportError:
                    import Image
                    import ImageTk
                
                pil_image = Image.fromarray(display_img)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.debug_canvas.create_text(10, y_offset, anchor=tk.NW, text="Enhanced Image", 
                                             font=("Arial", 10, "bold"))
                y_offset += 20
                self.debug_canvas.create_image(10, y_offset, anchor=tk.NW, image=photo)
                self.debug_canvas.image_refs = getattr(self.debug_canvas, 'image_refs', [])
                self.debug_canvas.image_refs.append(photo)
                y_offset += new_h + 20
            
            # Display adaptive threshold result if available
            if 'adaptive_result' in result:
                adaptive_result = result['adaptive_result']
                if 'adaptive_thresh' in adaptive_result:
                    adaptive_img = adaptive_result['adaptive_thresh']
                    if len(adaptive_img.shape) == 2:
                        adaptive_img = cv2.cvtColor(adaptive_img, cv2.COLOR_GRAY2RGB)
                    
                    h, w = adaptive_img.shape[:2]
                    scale = max_width / w if w > max_width else 1.0
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(adaptive_img, (new_w, new_h))
                    
                    try:
                        from PIL import Image, ImageTk
                    except ImportError:
                        import Image
                        import ImageTk
                    
                    pil_image = Image.fromarray(display_img)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    self.debug_canvas.create_text(10, y_offset, anchor=tk.NW, 
                                                text=f"Adaptive Threshold (Status: {adaptive_result.get('status', 'unknown')})", 
                                                font=("Arial", 10, "bold"))
                    y_offset += 20
                    self.debug_canvas.create_image(10, y_offset, anchor=tk.NW, image=photo)
                    self.debug_canvas.image_refs = getattr(self.debug_canvas, 'image_refs', [])
                    self.debug_canvas.image_refs.append(photo)
                    y_offset += new_h + 20
            
            # Display each debug image
            debug_names = ['roi_show', 'gray', 'mask_circle', 'roi_masked', 'edges', 'text_edges']
            
            for name in debug_names:
                if name not in debug_images:
                    continue
                
                debug_img = debug_images[name]
                
                # Convert grayscale to RGB if needed
                if len(debug_img.shape) == 2:
                    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
                elif len(debug_img.shape) == 3 and debug_img.shape[2] == 3:
                    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                
                # Resize
                h, w = debug_img.shape[:2]
                scale = max_width / w if w > max_width else 1.0
                new_w, new_h = int(w * scale), int(h * scale)
                display_img = cv2.resize(debug_img, (new_w, new_h))
                
                try:
                    from PIL import Image, ImageTk
                except ImportError:
                    import Image
                    import ImageTk
                
                pil_image = Image.fromarray(display_img)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Label
                self.debug_canvas.create_text(10, y_offset, anchor=tk.NW, text=name, 
                                             font=("Arial", 10))
                y_offset += 20
                
                self.debug_canvas.create_image(10, y_offset, anchor=tk.NW, image=photo)
                self.debug_canvas.image_refs = getattr(self.debug_canvas, 'image_refs', [])
                self.debug_canvas.image_refs.append(photo)
                
                y_offset += new_h + 20
            
            y_offset += 20
        
        self.debug_canvas.config(scrollregion=self.debug_canvas.bbox("all"))
    
    def update_summary(self):
        """Update summary text"""
        if not self.fade_results:
            self.summary_text.delete(1.0, tk.END)
            return
        
        summary = "=== สรุปผลการตรวจจับ Fade ===\n\n"
        
        faded_count = sum(1 for r in self.fade_results if r.get('status') == 'faded')
        normal_count = sum(1 for r in self.fade_results if r.get('status') == 'normal')
        unknown_count = sum(1 for r in self.fade_results if r.get('status') == 'unknown')
        
        summary += f"📊 สรุป:\n"
        summary += f"  - Faded: {faded_count} ฝา\n"
        summary += f"  - Normal: {normal_count} ฝา\n"
        summary += f"  - Unknown: {unknown_count} ฝา\n\n"
        
        summary += "=== รายละเอียดแต่ละฝา ===\n\n"
        
        for result in self.fade_results:
            cap_idx = result['cap_index']
            status = result.get('status', 'unknown')
            num_chars = result.get('num_chars', 0)
            total_area = result.get('total_area', 0)
            
            summary += f"ฝาที่ {cap_idx}:\n"
            summary += f"  Status: {status}\n"
            summary += f"  Components: {num_chars}\n"
            summary += f"  Total Area: {total_area}\n"
            
            if status == 'faded':
                summary += f"  ⚠️ ตรวจพบการ Fade!\n"
            else:
                summary += f"  ✅ ปกติ\n"
            
            summary += "\n"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
    
    def log_info(self, message):
        """Log information message"""
        print(message)


def main():
    """Main function"""
    root = tk.Tk()
    app = CapFadeDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Cap Histogram Analyzer & Matcher
โปรแกรมวิเคราะห์และปรับ histogram ของภาพฝาให้เท่ากัน เพื่อให้ความสว่างเท่ากันและใช้เกณฑ์เดียวกันได้
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
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import cap detection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.detection.capmodel import CapDetector
from config.settings import CAP_MODEL_PATH


class CapIntensityAnalyzer:
    """โปรแกรมวิเคราะห์และปรับ histogram ของภาพฝาให้เท่ากัน"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Cap Histogram Analyzer & Matcher - วิเคราะห์และปรับ Histogram ฝา")
        self.root.geometry("1600x1000")
        
        # Initialize detector
        self.cap_detector = None
        self.current_image = None
        self.current_image_path = None
        self.reference_image = None  # Reference image for histogram matching
        self.reference_image_path = None
        self.reference_cap_image = None  # Cropped reference cap
        self.cropped_caps = []  # List of cropped cap images (only one selected cap)
        self.selected_cap_index = None  # Index of selected cap
        self.matched_caps = []  # List of histogram-matched cap images
        self.intensity_results = []  # List of intensity analysis results
        self.matched_results = []  # List of matched intensity results
        self.detection_result = None  # Store detection result for drawing labels
        self.labeled_image = None  # Image with color labels
        
        # Settings
        self.conf_threshold = 0.5
        self.crop_margin = 10
        self.use_histogram_matching = True  # Enable histogram matching by default
        
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
        title_label = ttk.Label(main_frame, text="Cap Histogram Analyzer & Matcher - วิเคราะห์และปรับ Histogram ฝา", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
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
        
        # Reference image selection button
        self.select_reference_btn = ttk.Button(control_frame, text="เลือกรูปอ้างอิง (Reference)", 
                                              command=self.select_reference_image, state=tk.DISABLED)
        self.select_reference_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Detect and crop button
        self.detect_btn = ttk.Button(control_frame, text="ตรวจจับและครอปฝา", 
                                    command=self.detect_and_crop, state=tk.DISABLED)
        self.detect_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Analyze intensity button
        self.analyze_btn = ttk.Button(control_frame, text="วิเคราะห์ Histogram", 
                                     command=self.analyze_intensity, state=tk.DISABLED)
        self.analyze_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Histogram matching button
        self.match_histogram_btn = ttk.Button(control_frame, text="ปรับ Histogram ให้เท่ากัน", 
                                             command=self.match_histograms, state=tk.DISABLED)
        self.match_histogram_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Save matched images button
        self.save_matched_btn = ttk.Button(control_frame, text="บันทึกภาพที่ปรับแล้ว", 
                                          command=self.save_matched_images, state=tk.DISABLED)
        self.save_matched_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(left_panel, text="ตั้งค่า", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Histogram matching method
        ttk.Label(settings_frame, text="วิธีปรับ Histogram:").pack(anchor=tk.W)
        self.matching_method_var = tk.StringVar(value="histogram_matching")
        matching_method_frame = ttk.Frame(settings_frame)
        matching_method_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Radiobutton(matching_method_frame, text="Histogram Matching (แบบเดิม)", 
                       variable=self.matching_method_var, value="histogram_matching").pack(anchor=tk.W)
        ttk.Radiobutton(matching_method_frame, text="CLAHE (ปรับความสว่างแบบอัตโนมัติ)", 
                       variable=self.matching_method_var, value="clahe").pack(anchor=tk.W)
        ttk.Radiobutton(matching_method_frame, text="Brightness Adjustment (ปรับความสว่าง)", 
                       variable=self.matching_method_var, value="brightness").pack(anchor=tk.W)
        ttk.Radiobutton(matching_method_frame, text="Histogram Matching + CLAHE (ผสม)", 
                       variable=self.matching_method_var, value="mixed").pack(anchor=tk.W)
        
        # Saturation prevention
        self.prevent_saturation_var = tk.BooleanVar(value=True)
        saturation_check = ttk.Checkbutton(settings_frame, text="ป้องกันการขาวเกิน (Saturation)", 
                                          variable=self.prevent_saturation_var)
        saturation_check.pack(anchor=tk.W, pady=(5, 0))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                               variable=self.conf_var, orient=tk.HORIZONTAL)
        conf_scale.pack(fill=tk.X, pady=(0, 5))
        self.conf_label = ttk.Label(settings_frame, text="0.50")
        self.conf_label.pack(anchor=tk.W)
        conf_scale.configure(command=lambda v: self.conf_label.config(text=f"{float(v):.2f}"))
        
        # Crop margin
        ttk.Label(settings_frame, text="Crop Margin (pixels):").pack(anchor=tk.W)
        self.margin_var = tk.IntVar(value=10)
        margin_scale = ttk.Scale(settings_frame, from_=0, to=50, 
                                variable=self.margin_var, orient=tk.HORIZONTAL)
        margin_scale.pack(fill=tk.X, pady=(0, 5))
        self.margin_label = ttk.Label(settings_frame, text="10")
        self.margin_label.pack(anchor=tk.W)
        margin_scale.configure(command=lambda v: self.margin_label.config(text=f"{int(v)}"))
        
        
        # Results summary
        summary_frame = ttk.LabelFrame(left_panel, text="สรุปผล", padding="10")
        summary_frame.pack(fill=tk.BOTH, expand=True)
        
        self.summary_text = tk.Text(summary_frame, height=10, width=40, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.configure(yscrollcommand=scrollbar.set)
        
        # Right panel - Display with tabs
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Images (ฝาอ้างอิงและฝาที่ต้องการปรับ)
        images_tab = ttk.Frame(self.notebook)
        self.notebook.add(images_tab, text="ภาพ")
        images_tab.columnconfigure(0, weight=1)
        images_tab.columnconfigure(1, weight=1)
        images_tab.rowconfigure(0, weight=1)
        
        # Reference cap display (left side)
        ref_frame = ttk.LabelFrame(images_tab, text="ฝาอ้างอิง (Reference)", padding="10")
        ref_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        ref_scroll = ttk.Scrollbar(ref_frame, orient=tk.VERTICAL)
        ref_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.reference_cap_canvas = tk.Canvas(ref_frame, yscrollcommand=ref_scroll.set, bg="white")
        ref_scroll.config(command=self.reference_cap_canvas.yview)
        self.reference_cap_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Target cap display (right side)
        target_frame = ttk.LabelFrame(images_tab, text="ฝาที่ต้องการปรับ", padding="10")
        target_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        target_scroll = ttk.Scrollbar(target_frame, orient=tk.VERTICAL)
        target_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.target_cap_canvas = tk.Canvas(target_frame, yscrollcommand=target_scroll.set, bg="white")
        target_scroll.config(command=self.target_cap_canvas.yview)
        self.target_cap_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Tab 2: Histogram (กราฟ)
        histogram_tab = ttk.Frame(self.notebook)
        self.notebook.add(histogram_tab, text="Histogram")
        
        # Create notebook for histogram comparison
        hist_notebook = ttk.Notebook(histogram_tab)
        hist_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Before matching tab
        before_tab = ttk.Frame(hist_notebook)
        hist_notebook.add(before_tab, text="ก่อน Matching")
        self.intensity_fig = Figure(figsize=(10, 6), dpi=100)
        self.intensity_ax = self.intensity_fig.add_subplot(111)
        self.intensity_canvas = FigureCanvasTkAgg(self.intensity_fig, before_tab)
        self.intensity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # After matching tab
        after_tab = ttk.Frame(hist_notebook)
        hist_notebook.add(after_tab, text="หลัง Matching")
        self.matched_fig = Figure(figsize=(10, 6), dpi=100)
        self.matched_ax = self.matched_fig.add_subplot(111)
        self.matched_canvas = FigureCanvasTkAgg(self.matched_fig, after_tab)
        self.matched_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Comparison tab
        comparison_tab = ttk.Frame(hist_notebook)
        hist_notebook.add(comparison_tab, text="เปรียบเทียบ")
        self.comparison_fig = Figure(figsize=(12, 6), dpi=100)
        self.comparison_ax1 = self.comparison_fig.add_subplot(121)
        self.comparison_ax2 = self.comparison_fig.add_subplot(122)
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_fig, comparison_tab)
        self.comparison_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
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
            self.select_reference_btn.config(state=tk.NORMAL)
    
    def select_folder(self):
        """Select folder with images"""
        folder_path = filedialog.askdirectory(title="เลือกโฟลเดอร์")
        if folder_path:
            # Find all image files
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(folder_path).glob(f"*{ext}"))
                image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
            
            if not image_files:
                messagebox.showwarning("Warning", "ไม่พบไฟล์ภาพในโฟลเดอร์นี้")
                return
            
            # Process first image for now (can extend to batch processing)
            self.current_image_path = str(image_files[0])
            self.load_image(self.current_image_path)
            self.detect_btn.config(state=tk.NORMAL)
            self.select_reference_btn.config(state=tk.NORMAL)
            self.log_info(f"พบ {len(image_files)} ไฟล์ในโฟลเดอร์ (กำลังใช้ไฟล์แรก)")
    
    def select_reference_image(self):
        """Select reference image for histogram matching"""
        file_path = filedialog.askopenfilename(
            title="เลือกรูปอ้างอิง (Reference Image)",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.reference_image_path = file_path
            self.reference_image = cv2.imread(file_path)
            if self.reference_image is None:
                messagebox.showerror("Error", "ไม่สามารถโหลดรูปอ้างอิงได้")
                return
            
            self.log_info(f"✅ โหลดรูปอ้างอิง: {os.path.basename(file_path)}")
            
            # Auto-detect and crop cap from reference image
            if self.cap_detector is None:
                messagebox.showwarning("Warning", "Cap detector ยังไม่ได้โหลด กรุณารอสักครู่")
                return
            
            try:
                self.log_info("🔍 กำลังตรวจจับฝาจากรูปอ้างอิง...")
                
                # Update settings
                self.cap_detector.conf_threshold = self.conf_var.get()
                crop_margin = self.margin_var.get()
                
                # Detect caps from reference image
                ref_result = self.cap_detector.detect_caps(image_array=self.reference_image)
                
                if 'error' in ref_result:
                    messagebox.showerror("Error", f"ไม่สามารถตรวจจับฝาจากรูปอ้างอิงได้: {ref_result['error']}")
                    return
                
                num_caps = ref_result.get('total_detections', 0)
                if num_caps == 0:
                    messagebox.showwarning("Warning", "ไม่พบฝาในรูปอ้างอิง")
                    return
                
                self.log_info(f"✅ พบฝา {num_caps} ฝาในรูปอ้างอิง")
                
                # Crop detections
                all_ref_caps = self.cap_detector.crop_detections(
                    image=self.reference_image,
                    result=ref_result,
                    margin=crop_margin
                )
                
                if not all_ref_caps:
                    messagebox.showwarning("Warning", "ไม่สามารถครอปฝาจากรูปอ้างอิงได้")
                    return
                
                # Select bottommost cap from reference image (same logic as main image)
                selected_ref_cap = None
                selected_ref_index = 0
                
                if 'detections' in ref_result:
                    detections = ref_result['detections']
                    max_y = 0
                    
                    # Find the cap with the highest y-coordinate (bottom of image)
                    for i, detection in enumerate(detections):
                        if 'bbox' in detection:
                            bbox = detection['bbox']
                            y_center = (bbox[1] + bbox[3]) / 2
                            conf = detection.get('confidence', 0.0)
                            self.log_info(f"   ฝาอ้างอิงที่ {i+1} - y_center: {y_center:.1f}, ความเชื่อมั่น: {conf:.4f} ({conf*100:.2f}%)")
                            
                            if y_center > max_y:
                                max_y = y_center
                                selected_ref_cap = all_ref_caps[i]
                                selected_ref_index = i
                    
                    if selected_ref_cap is not None:
                        selected_conf = detections[selected_ref_index].get('confidence', 0.0) if selected_ref_index < len(detections) else 0.0
                        self.log_info(f"✅ เลือกฝาอ้างอิงที่ {selected_ref_index + 1} (ฝาล่างสุด, y_center: {max_y:.1f}, ความเชื่อมั่น: {selected_conf:.4f} ({selected_conf*100:.2f}%))")
                else:
                    # Fallback: use the last cap
                    selected_ref_cap = all_ref_caps[-1]
                    selected_ref_index = len(all_ref_caps) - 1
                    self.log_info(f"✅ ไม่มีข้อมูลตำแหน่ง ใช้ฝาอ้างอิงลำดับสุดท้าย (ฝาที่ {selected_ref_index + 1})")
                
                if selected_ref_cap is not None:
                    self.reference_cap_image = selected_ref_cap.copy()
                    self.log_info("✅ ตั้งฝาอ้างอิงสำเร็จ - พร้อมใช้สำหรับ Histogram Matching")
                    self.display_reference_cap()
                    self.match_histogram_btn.config(state=tk.NORMAL)
                else:
                    messagebox.showwarning("Warning", "ไม่สามารถเลือกฝาอ้างอิงได้")
                    
            except Exception as e:
                messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการตรวจจับฝาจากรูปอ้างอิง: {str(e)}")
                self.log_info(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def load_image(self, image_path):
        """Load image from file"""
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("ไม่สามารถโหลดภาพได้")
            
            self.log_info(f"✅ โหลดภาพ: {os.path.basename(image_path)}")
            self.log_info(f"   ขนาด: {self.current_image.shape[1]}x{self.current_image.shape[0]}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
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
            
            # Update settings
            self.cap_detector.conf_threshold = self.conf_var.get()
            self.crop_margin = self.margin_var.get()
            
            # Detect caps
            result = self.cap_detector.detect_caps(image_array=self.current_image)
            
            if 'error' in result:
                raise Exception(result['error'])
            
            # Store detection result for later use
            self.detection_result = result
            
            num_caps = result.get('total_detections', 0)
            self.log_info(f"✅ พบฝา {num_caps} ฝา")
            
            if num_caps == 0:
                messagebox.showinfo("Info", "ไม่พบฝาในภาพ")
                return
            
            # Crop detections
            all_cropped_caps = self.cap_detector.crop_detections(
                image=self.current_image,
                result=result,
                margin=self.crop_margin
            )
            
            self.log_info(f"✅ ครอปฝาได้ {len(all_cropped_caps)} ฝา")
            
            # Select only the bottommost cap (highest y-coordinate) - same as main program
            selected_cap = None
            selected_index = 0
            
            if all_cropped_caps and 'detections' in result:
                detections = result['detections']
                max_y = 0
                
                # Find the cap with the highest y-coordinate (bottom of image)
                for i, detection in enumerate(detections):
                    if 'bbox' in detection:
                        # bbox format: [x1, y1, x2, y2]
                        bbox = detection['bbox']
                        y_center = (bbox[1] + bbox[3]) / 2  # Use center y-coordinate
                        conf = detection.get('confidence', 0.0)
                        self.log_info(f"   ฝาที่ {i+1} - y_center: {y_center:.1f}, ความเชื่อมั่น: {conf:.4f} ({conf*100:.2f}%)")
                        
                        if y_center > max_y:
                            max_y = y_center
                            selected_cap = all_cropped_caps[i]
                            selected_index = i
                
                if selected_cap is not None:
                    selected_conf = detections[selected_index].get('confidence', 0.0) if selected_index < len(detections) else 0.0
                    self.log_info(f"✅ เลือกฝาที่ {selected_index + 1} (ฝาล่างสุด, y_center: {max_y:.1f}, ความเชื่อมั่น: {selected_conf:.4f} ({selected_conf*100:.2f}%))")
            elif all_cropped_caps:
                # Fallback: use the last cap if no bounding box info available
                selected_cap = all_cropped_caps[-1]
                selected_index = len(all_cropped_caps) - 1
                self.log_info(f"✅ ไม่มีข้อมูลตำแหน่ง ใช้ฝาลำดับสุดท้าย (ฝาที่ {selected_index + 1})")
            
            # Store only the selected cap
            if selected_cap is not None:
                self.cropped_caps = [selected_cap]
                self.selected_cap_index = selected_index
                self.log_info(f"✅ เลือกฝาเดียว: ฝาที่ {selected_index + 1} (ฝาล่างสุด) จาก {len(all_cropped_caps)} ฝา")
            else:
                self.cropped_caps = []
                self.selected_cap_index = None
                messagebox.showwarning("Warning", "ไม่สามารถเลือกฝาได้")
                return
            
            # Enable analyze button
            self.analyze_btn.config(state=tk.NORMAL)
            
            # Display target cap
            self.display_target_cap()
            
            # Note: User needs to select reference image separately
            if self.reference_cap_image is None:
                self.log_info("💡 หมายเหตุ: กรุณาเลือกรูปอ้างอิงเพื่อทำ Histogram Matching")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.log_info(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    
    def display_reference_cap(self):
        """Display reference cap image"""
        self.reference_cap_canvas.delete("all")
        
        if self.reference_cap_image is None:
            self.reference_cap_canvas.create_text(200, 200, text="ยังไม่มีฝาอ้างอิง", 
                                                   font=("Arial", 14), fill="gray")
            return
        
        # Resize for display
        h, w = self.reference_cap_image.shape[:2]
        max_width = 500
        max_height = 500
        scale = min(max_width / w, max_height / h) if w > max_width or h > max_height else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        display_image = cv2.resize(self.reference_cap_image, (new_w, new_h))
        
        # Convert BGR to RGB
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        try:
            from PIL import Image, ImageTk
        except ImportError:
            import Image
            import ImageTk
        
        pil_image = Image.fromarray(display_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Center image
        canvas_width = self.reference_cap_canvas.winfo_width()
        canvas_height = self.reference_cap_canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
        else:
            x, y = 10, 10
        
        # Display image
        self.reference_cap_canvas.create_image(x, y, anchor=tk.NW, image=photo)
        self.reference_cap_canvas.image_ref = photo  # Keep reference
        
        # Add info label
        info_text = f"ขนาด: {w}x{h}"
        self.reference_cap_canvas.create_text(x + new_w // 2, y + new_h + 20, 
                                             text=info_text, font=("Arial", 10), fill="black")
        
        self.reference_cap_canvas.config(scrollregion=self.reference_cap_canvas.bbox("all"))
    
    def display_target_cap(self):
        """Display target cap image (cap to be adjusted)"""
        self.target_cap_canvas.delete("all")
        
        if not self.cropped_caps:
            self.target_cap_canvas.create_text(200, 200, text="ยังไม่มีฝาที่ต้องการปรับ", 
                                               font=("Arial", 14), fill="gray")
            return
        
        cap_image = self.cropped_caps[0]  # Only one cap
        
        # Resize for display
        h, w = cap_image.shape[:2]
        max_width = 500
        max_height = 500
        scale = min(max_width / w, max_height / h) if w > max_width or h > max_height else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        display_image = cv2.resize(cap_image, (new_w, new_h))
        
        # Convert BGR to RGB
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        try:
            from PIL import Image, ImageTk
        except ImportError:
            import Image
            import ImageTk
        
        pil_image = Image.fromarray(display_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Center image
        canvas_width = self.target_cap_canvas.winfo_width()
        canvas_height = self.target_cap_canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
        else:
            x, y = 10, 10
        
        # Display image
        self.target_cap_canvas.create_image(x, y, anchor=tk.NW, image=photo)
        self.target_cap_canvas.image_ref = photo  # Keep reference
        
        # Add info label
        if hasattr(self, 'selected_cap_index') and self.selected_cap_index is not None:
            info_text = f"ฝาที่ {self.selected_cap_index + 1} (ฝาล่างสุด)\nขนาด: {w}x{h}"
        else:
            info_text = f"ขนาด: {w}x{h}"
        self.target_cap_canvas.create_text(x + new_w // 2, y + new_h + 30, 
                                         text=info_text, font=("Arial", 10), fill="black")
        
        self.target_cap_canvas.config(scrollregion=self.target_cap_canvas.bbox("all"))
    
    def analyze_intensity(self):
        """Analyze intensity of selected cap"""
        if not self.cropped_caps:
            messagebox.showwarning("Warning", "กรุณาตรวจจับและครอปฝาก่อน")
            return
        
        try:
            self.log_info("📊 กำลังวิเคราะห์ Histogram...")
            self.intensity_results = []
            
            # Analyze only the selected cap (should be only one)
            for i, cap_image in enumerate(self.cropped_caps):
                # Convert to grayscale if needed
                if len(cap_image.shape) == 3:
                    gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = cap_image
                
                # Calculate intensity statistics
                mean_intensity = np.mean(gray)
                std_intensity = np.std(gray)
                min_intensity = np.min(gray)
                max_intensity = np.max(gray)
                median_intensity = np.median(gray)
                
                # Calculate histogram
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                
                result = {
                    'cap_index': i + 1,
                    'mean': mean_intensity,
                    'std': std_intensity,
                    'min': min_intensity,
                    'max': max_intensity,
                    'median': median_intensity,
                    'histogram': hist.flatten(),
                    'image': cap_image,
                    'gray': gray
                }
                
                self.intensity_results.append(result)
                
                cap_label = f"ฝาที่เลือก (Index: {self.selected_cap_index + 1})" if hasattr(self, 'selected_cap_index') and self.selected_cap_index is not None else f"ฝาที่ {i+1}"
                self.log_info(f"   {cap_label}: Mean={mean_intensity:.2f}, Std={std_intensity:.2f}, "
                            f"Min={min_intensity:.0f}, Max={max_intensity:.0f}, Median={median_intensity:.2f}")
            
            # Update summary
            self.update_summary()
            
            # Plot histogram
            self.plot_intensity_histogram()
            
            self.log_info("✅ วิเคราะห์ Histogram เสร็จสิ้น")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.log_info(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    
    def match_histograms(self):
        """Match histogram of selected cap to reference cap"""
        if not self.cropped_caps:
            messagebox.showwarning("Warning", "กรุณาตรวจจับและครอปฝาก่อน")
            return
        
        if self.reference_cap_image is None:
            messagebox.showwarning("Warning", "กรุณาเลือกรูปอ้างอิงหรือใช้ฝาที่เลือกเป็น Reference")
            return
        
        if not self.intensity_results:
            messagebox.showwarning("Warning", "กรุณาวิเคราะห์ Histogram ก่อน")
            return
        
        try:
            self.log_info("🔄 กำลังปรับ Histogram ให้เท่ากัน...")
            self.matched_caps = []
            self.matched_results = []
            
            # Convert reference cap to grayscale
            if len(self.reference_cap_image.shape) == 3:
                ref_gray = cv2.cvtColor(self.reference_cap_image, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = self.reference_cap_image.copy()
            
            # Calculate reference histogram
            ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
            ref_cdf = self.calculate_cdf(ref_hist)
            
            # Match the selected cap to reference (should be only one cap)
            for i, cap_image in enumerate(self.cropped_caps):
                # Convert to grayscale
                if len(cap_image.shape) == 3:
                    gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = cap_image.copy()
                
                # Apply selected matching method
                method = self.matching_method_var.get()
                
                if method == "histogram_matching":
                    # Traditional histogram matching
                    src_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    src_cdf = self.calculate_cdf(src_hist)
                    mapping = self.create_histogram_mapping(src_cdf, ref_cdf)
                    matched_gray = mapping[gray]
                    
                elif method == "clahe":
                    # CLAHE method
                    matched_gray = self.apply_clahe_matching(gray, ref_gray)
                    
                elif method == "brightness":
                    # Simple brightness adjustment
                    matched_gray = self.apply_brightness_adjustment(gray, ref_gray)
                    
                elif method == "mixed":
                    # Histogram matching + CLAHE
                    src_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    src_cdf = self.calculate_cdf(src_hist)
                    mapping = self.create_histogram_mapping(src_cdf, ref_cdf)
                    hist_matched = mapping[gray]
                    # Apply CLAHE on top
                    matched_gray = self.apply_clahe_matching(hist_matched, ref_gray)
                else:
                    # Default to histogram matching
                    src_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    src_cdf = self.calculate_cdf(src_hist)
                    mapping = self.create_histogram_mapping(src_cdf, ref_cdf)
                    matched_gray = mapping[gray]
                
                # Convert back to BGR if original was color
                if len(cap_image.shape) == 3:
                    matched_image = cv2.cvtColor(matched_gray, cv2.COLOR_GRAY2BGR)
                else:
                    matched_image = matched_gray.copy()
                
                self.matched_caps.append(matched_image)
                
                # Calculate matched intensity statistics
                matched_mean = np.mean(matched_gray)
                matched_std = np.std(matched_gray)
                matched_min = np.min(matched_gray)
                matched_max = np.max(matched_gray)
                matched_median = np.median(matched_gray)
                matched_hist = cv2.calcHist([matched_gray], [0], None, [256], [0, 256])
                
                result = {
                    'cap_index': i + 1,
                    'mean': matched_mean,
                    'std': matched_std,
                    'min': matched_min,
                    'max': matched_max,
                    'median': matched_median,
                    'histogram': matched_hist.flatten(),
                    'image': matched_image,
                    'gray': matched_gray,
                    'original_mean': self.intensity_results[i]['mean'],
                    'original_std': self.intensity_results[i]['std']
                }
                
                self.matched_results.append(result)
                
                cap_label = f"ฝาที่เลือก (Index: {self.selected_cap_index + 1})" if hasattr(self, 'selected_cap_index') and self.selected_cap_index is not None else f"ฝาที่ {i+1}"
                self.log_info(f"   {cap_label}: Mean {self.intensity_results[i]['mean']:.2f} -> {matched_mean:.2f}, "
                            f"Std {self.intensity_results[i]['std']:.2f} -> {matched_std:.2f}")
            
            # Display matched cap (update target cap display)
            if len(self.matched_caps) > 0:
                # Update target cap canvas with matched image
                matched_image = self.matched_caps[0]
                h, w = matched_image.shape[:2]
                max_width = 500
                max_height = 500
                scale = min(max_width / w, max_height / h) if w > max_width or h > max_height else 1.0
                new_w, new_h = int(w * scale), int(h * scale)
                display_image = cv2.resize(matched_image, (new_w, new_h))
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                
                try:
                    from PIL import Image, ImageTk
                except ImportError:
                    import Image
                    import ImageTk
                
                pil_image = Image.fromarray(display_image)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.target_cap_canvas.delete("all")
                canvas_width = self.target_cap_canvas.winfo_width()
                canvas_height = self.target_cap_canvas.winfo_height()
                if canvas_width > 1 and canvas_height > 1:
                    x = (canvas_width - new_w) // 2
                    y = (canvas_height - new_h) // 2
                else:
                    x, y = 10, 10
                
                self.target_cap_canvas.create_image(x, y, anchor=tk.NW, image=photo)
                self.target_cap_canvas.image_ref = photo
                
                # Add label showing it's matched
                info_text = f"ฝาที่ปรับแล้ว (หลัง Matching)\nขนาด: {w}x{h}"
                self.target_cap_canvas.create_text(x + new_w // 2, y + new_h + 30, 
                                                 text=info_text, font=("Arial", 10, "bold"), fill="green")
                self.target_cap_canvas.config(scrollregion=self.target_cap_canvas.bbox("all"))
            
            # Plot matched histogram
            self.plot_matched_histogram()
            
            # Plot comparison
            self.plot_histogram_comparison()
            
            # Update summary
            self.update_matched_summary()
            
            # Enable save button
            self.save_matched_btn.config(state=tk.NORMAL)
            
            method_name = {
                "histogram_matching": "Histogram Matching",
                "clahe": "CLAHE",
                "brightness": "Brightness Adjustment",
                "mixed": "Histogram Matching + CLAHE"
            }.get(self.matching_method_var.get(), "Histogram Matching")
            
            self.log_info(f"✅ ปรับ Histogram เสร็จสิ้น (วิธี: {method_name}) - ดูผลลัพธ์ที่ฝาที่ต้องการปรับ")
            
        except Exception as e:
            messagebox.showerror("Error", f"Histogram matching failed: {str(e)}")
            self.log_info(f"❌ เกิดข้อผิดพลาด: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def calculate_cdf(self, histogram):
        """Calculate Cumulative Distribution Function from histogram"""
        cdf = histogram.cumsum()
        cdf_normalized = cdf * float(histogram.max()) / cdf.max()
        return cdf_normalized
    
    def create_histogram_mapping(self, src_cdf, ref_cdf):
        """Create mapping function from source CDF to reference CDF with saturation prevention"""
        mapping = np.zeros(256, dtype=np.uint8)
        
        # Normalize CDFs to 0-255 range
        src_cdf_norm = (src_cdf / src_cdf.max() * 255).astype(np.uint8)
        ref_cdf_norm = (ref_cdf / ref_cdf.max() * 255).astype(np.uint8)
        
        for i in range(256):
            # Find closest value in reference CDF
            diff = np.abs(ref_cdf_norm - src_cdf_norm[i])
            mapped_value = np.argmin(diff)
            
            # Prevent saturation - limit maximum value
            if self.prevent_saturation_var.get():
                # Limit to 95% of max to prevent saturation
                max_allowed = int(255 * 0.95)
                mapping[i] = min(mapped_value, max_allowed)
            else:
                mapping[i] = mapped_value
        
        return mapping
    
    def apply_clahe_matching(self, gray_image, reference_gray):
        """Apply CLAHE to match brightness/contrast"""
        # Calculate target brightness from reference
        ref_mean = np.mean(reference_gray)
        src_mean = np.mean(gray_image)
        
        # Adjust clip limit based on brightness difference
        brightness_diff = abs(ref_mean - src_mean)
        clip_limit = 2.0 + (brightness_diff / 128.0) * 2.0  # 2.0 to 4.0
        clip_limit = min(clip_limit, 4.0)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_image)
        
        # Adjust brightness to match reference mean
        brightness_shift = ref_mean - np.mean(enhanced)
        adjusted = cv2.add(enhanced, int(brightness_shift))
        
        # Prevent saturation
        if self.prevent_saturation_var.get():
            adjusted = np.clip(adjusted, 0, int(255 * 0.95))
        
        return adjusted.astype(np.uint8)
    
    def apply_brightness_adjustment(self, gray_image, reference_gray):
        """Apply simple brightness adjustment to match reference"""
        ref_mean = np.mean(reference_gray)
        src_mean = np.mean(gray_image)
        
        # Calculate brightness adjustment
        brightness_diff = ref_mean - src_mean
        
        # Apply adjustment
        adjusted = cv2.add(gray_image, int(brightness_diff))
        
        # Prevent saturation
        if self.prevent_saturation_var.get():
            adjusted = np.clip(adjusted, 0, int(255 * 0.95))
        else:
            adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)
    
    def display_matched_caps(self):
        """Display histogram-matched cap images"""
        self.matched_caps_canvas.delete("all")
        
        if not self.matched_caps:
            return
        
        y_offset = 10
        max_width = 400
        
        for i, (original, matched) in enumerate(zip(self.cropped_caps, self.matched_caps)):
            # Create side-by-side comparison
            h, w = original.shape[:2]
            scale = max_width / w if w > max_width else 1.0
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize both images
            orig_display = cv2.resize(original, (new_w, new_h))
            matched_display = cv2.resize(matched, (new_w, new_h))
            
            # Convert to RGB
            orig_display = cv2.cvtColor(orig_display, cv2.COLOR_BGR2RGB)
            matched_display = cv2.cvtColor(matched_display, cv2.COLOR_BGR2RGB)
            
            # Combine side by side
            combined = np.hstack([orig_display, matched_display])
            
            # Convert to PhotoImage
            try:
                from PIL import Image, ImageTk
            except ImportError:
                import Image
                import ImageTk
            
            pil_image = Image.fromarray(combined)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Add label
            orig_mean = self.intensity_results[i]['mean']
            matched_mean = self.matched_results[i]['mean'] if i < len(self.matched_results) else 0
            label_text = f"ฝาที่ {i+1}: Mean {orig_mean:.1f} -> {matched_mean:.1f}"
            self.matched_caps_canvas.create_text(10, y_offset, anchor=tk.NW, text=label_text, 
                                                 font=("Arial", 12, "bold"))
            y_offset += 25
            
            # Display image
            self.matched_caps_canvas.create_image(10, y_offset, anchor=tk.NW, image=photo)
            self.matched_caps_canvas.image_refs = getattr(self.matched_caps_canvas, 'image_refs', [])
            self.matched_caps_canvas.image_refs.append(photo)  # Keep reference
            
            y_offset += new_h + 20
        
        self.matched_caps_canvas.config(scrollregion=self.matched_caps_canvas.bbox("all"))
    
    def plot_matched_histogram(self):
        """Plot histogram after matching"""
        self.matched_ax.clear()
        
        if not self.matched_results:
            return
        
        # Plot histogram for matched cap (only one)
        result = self.matched_results[0]
        cap_label = f"ฝาที่เลือก (Index: {self.selected_cap_index + 1})" if hasattr(self, 'selected_cap_index') and self.selected_cap_index is not None else "ฝาที่เลือก"
        
        self.matched_ax.plot(result['histogram'], 
                          label=f"{cap_label} หลัง Matching (Mean={result['mean']:.1f})",
                          color='green', alpha=0.7, linewidth=2)
        
        # Plot reference histogram
        if self.reference_cap_image is not None:
            if len(self.reference_cap_image.shape) == 3:
                ref_gray = cv2.cvtColor(self.reference_cap_image, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = self.reference_cap_image.copy()
            ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
            self.matched_ax.plot(ref_hist.flatten(), 
                               label="Reference (อ้างอิง)",
                               color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        self.matched_ax.set_xlabel('Intensity (0-255)')
        self.matched_ax.set_ylabel('Frequency')
        self.matched_ax.set_title('Histogram หลัง Matching - ฮิสโตแกรมหลังการปรับ')
        self.matched_ax.legend()
        self.matched_ax.grid(True, alpha=0.3)
        
        self.matched_canvas.draw()
    
    def plot_histogram_comparison(self):
        """Plot before/after histogram comparison"""
        self.comparison_ax1.clear()
        self.comparison_ax2.clear()
        
        if not self.intensity_results or not self.matched_results:
            return
        
        cap_label = f"ฝาที่เลือก (Index: {self.selected_cap_index + 1})" if hasattr(self, 'selected_cap_index') and self.selected_cap_index is not None else "ฝาที่เลือก"
        
        # Before matching
        result_before = self.intensity_results[0]
        self.comparison_ax1.plot(result_before['histogram'], 
                               label=f"{cap_label} (Mean={result_before['mean']:.1f})",
                               color='blue', alpha=0.7, linewidth=2)
        
        # Plot reference if available
        if self.reference_cap_image is not None:
            if len(self.reference_cap_image.shape) == 3:
                ref_gray = cv2.cvtColor(self.reference_cap_image, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = self.reference_cap_image.copy()
            ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
            self.comparison_ax1.plot(ref_hist.flatten(), 
                                   label="Reference (อ้างอิง)",
                                   color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        self.comparison_ax1.set_xlabel('Intensity (0-255)')
        self.comparison_ax1.set_ylabel('Frequency')
        self.comparison_ax1.set_title('ก่อน Matching')
        self.comparison_ax1.legend()
        self.comparison_ax1.grid(True, alpha=0.3)
        
        # After matching
        result_after = self.matched_results[0]
        self.comparison_ax2.plot(result_after['histogram'], 
                                label=f"{cap_label} หลัง Matching (Mean={result_after['mean']:.1f})",
                                color='green', alpha=0.7, linewidth=2)
        
        # Plot reference
        if self.reference_cap_image is not None:
            if len(self.reference_cap_image.shape) == 3:
                ref_gray = cv2.cvtColor(self.reference_cap_image, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = self.reference_cap_image.copy()
            ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
            self.comparison_ax2.plot(ref_hist.flatten(), 
                                    label="Reference (อ้างอิง)",
                                    color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        self.comparison_ax2.set_xlabel('Intensity (0-255)')
        self.comparison_ax2.set_ylabel('Frequency')
        self.comparison_ax2.set_title('หลัง Matching')
        self.comparison_ax2.legend()
        self.comparison_ax2.grid(True, alpha=0.3)
        
        self.comparison_canvas.draw()
    
    def update_matched_summary(self):
        """Update summary with matched results"""
        if not self.matched_results:
            return
        
        summary = "=== สรุปผลหลัง Matching ===\n\n"
        
        for result in self.matched_results:
            summary += f"ฝาที่ {result['cap_index']}:\n"
            summary += f"  Mean: {result['original_mean']:.2f} -> {result['mean']:.2f}\n"
            summary += f"  Std: {result['original_std']:.2f} -> {result['std']:.2f}\n"
            summary += f"  Min: {result['min']:.0f}, Max: {result['max']:.0f}\n"
            summary += f"  Median: {result['median']:.2f}\n\n"
        
        # Overall statistics
        all_means = [r['mean'] for r in self.matched_results]
        summary += "=== สถิติรวมหลัง Matching ===\n"
        summary += f"Mean ของทุกฝา: {np.mean(all_means):.2f}\n"
        summary += f"Std ของทุกฝา: {np.std(all_means):.2f}\n"
        summary += f"Min Mean: {np.min(all_means):.2f}\n"
        summary += f"Max Mean: {np.max(all_means):.2f}\n"
        summary += f"\n✅ ความสว่างเท่ากันแล้ว - ใช้เกณฑ์เดียวกันได้\n"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
    
    def save_matched_images(self):
        """Save histogram-matched images"""
        if not self.matched_caps:
            messagebox.showwarning("Warning", "ไม่มีภาพที่ปรับแล้ว")
            return
        
        # Ask for save directory
        save_dir = filedialog.askdirectory(title="เลือกโฟลเดอร์บันทึก")
        if not save_dir:
            return
        
        try:
            saved_count = 0
            for i, matched_image in enumerate(self.matched_caps):
                # Generate filename
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                filename = f"{base_name}_cap_{i+1}_matched.png"
                filepath = os.path.join(save_dir, filename)
                
                # Save image
                cv2.imwrite(filepath, matched_image)
                saved_count += 1
            
            messagebox.showinfo("Success", f"บันทึกภาพ {saved_count} ภาพเรียบร้อยแล้ว\nที่: {save_dir}")
            self.log_info(f"✅ บันทึกภาพ {saved_count} ภาพที่ {save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save images: {str(e)}")
            self.log_info(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    
    def plot_intensity_histogram(self):
        """Plot intensity histogram"""
        self.intensity_ax.clear()
        
        if not self.intensity_results:
            return
        
        # Plot histogram for selected cap (only one)
        result = self.intensity_results[0]
        cap_label = f"ฝาที่เลือก (Index: {self.selected_cap_index + 1})" if hasattr(self, 'selected_cap_index') and self.selected_cap_index is not None else "ฝาที่เลือก"
        
        self.intensity_ax.plot(result['histogram'], 
                              label=f"{cap_label} (Mean={result['mean']:.1f})",
                              color='blue', alpha=0.7, linewidth=2)
        
        # Plot reference histogram if available
        if self.reference_cap_image is not None:
            if len(self.reference_cap_image.shape) == 3:
                ref_gray = cv2.cvtColor(self.reference_cap_image, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = self.reference_cap_image.copy()
            ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
            self.intensity_ax.plot(ref_hist.flatten(), 
                                 label="Reference (อ้างอิง)",
                                 color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        self.intensity_ax.set_xlabel('Intensity (0-255)')
        self.intensity_ax.set_ylabel('Frequency')
        self.intensity_ax.set_title('Histogram ก่อน Matching - ฮิสโตแกรมก่อนการปรับ')
        self.intensity_ax.legend()
        self.intensity_ax.grid(True, alpha=0.3)
        
        self.intensity_canvas.draw()
    
    def update_summary(self):
        """Update summary text"""
        if not self.intensity_results:
            self.summary_text.delete(1.0, tk.END)
            return
        
        summary = "=== สรุปผลการวิเคราะห์ความเข้ม ===\n\n"
        
        for result in self.intensity_results:
            summary += f"ฝาที่ {result['cap_index']}:\n"
            summary += f"  Mean: {result['mean']:.2f}\n"
            summary += f"  Std: {result['std']:.2f}\n"
            summary += f"  Min: {result['min']:.0f}\n"
            summary += f"  Max: {result['max']:.0f}\n"
            summary += f"  Median: {result['median']:.2f}\n\n"
        
        # Overall statistics
        all_means = [r['mean'] for r in self.intensity_results]
        summary += "=== สถิติรวม ===\n"
        summary += f"Mean ของทุกฝา: {np.mean(all_means):.2f}\n"
        summary += f"Std ของทุกฝา: {np.std(all_means):.2f}\n"
        summary += f"Min Mean: {np.min(all_means):.2f}\n"
        summary += f"Max Mean: {np.max(all_means):.2f}\n"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
    
    def update_results_text(self):
        """Update results text with detailed analysis"""
        if not self.intensity_results:
            self.results_text.delete(1.0, tk.END)
            return
        
        results = "=== ผลการวิเคราะห์ความเข้ม ===\n\n"
        
        threshold = self.threshold_var.get()
        
        for result in self.intensity_results:
            results += f"ฝาที่ {result['cap_index']}:\n"
            results += f"  Mean Intensity: {result['mean']:.2f}\n"
            results += f"  Std Deviation: {result['std']:.2f}\n"
            results += f"  Min: {result['min']:.0f}, Max: {result['max']:.0f}\n"
            results += f"  Median: {result['median']:.2f}\n"
            
            # Classification based on threshold
            if result['mean'] < threshold:
                classification = "ความเข้มต่ำ (ต่ำกว่าเกณฑ์)"
            else:
                classification = "ความเข้มสูง (สูงกว่าเกณฑ์)"
            
            results += f"  การแยก: {classification}\n"
            results += f"  เปอร์เซ็นต์ที่ต่ำกว่าเกณฑ์: {(result['histogram'][:threshold].sum() / result['histogram'].sum() * 100):.1f}%\n\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
    
    def classify_and_label_caps(self):
        """Classify selected cap into color (Yellow, Blue, Gray) based on mean intensity and draw label"""
        if not self.intensity_results:
            messagebox.showwarning("Warning", "กรุณาวิเคราะห์ความเข้มก่อน")
            return
        
        if self.current_image is None or self.detection_result is None:
            messagebox.showwarning("Warning", "ไม่มีภาพหรือผลการตรวจจับ")
            return
        
        try:
            # Since we only have one cap, classify it based on mean intensity
            result = self.intensity_results[0]
            mean = result['mean']
            
            # Use fixed thresholds for classification (can be adjusted)
            # These thresholds are based on typical cap brightness values
            threshold_high = 150  # High brightness -> Yellow
            threshold_low = 100   # Low brightness -> Gray
            
            yellow_caps = []
            blue_caps = []
            gray_caps = []
            
            if mean >= threshold_high:
                yellow_caps = [result]
                result['color'] = 'Yellow'
                result['color_thai'] = 'เหลือง'
                result['color_bgr'] = (0, 255, 255)  # Yellow in BGR
            elif mean >= threshold_low:
                blue_caps = [result]
                result['color'] = 'Blue'
                result['color_thai'] = 'ฟ้า'
                result['color_bgr'] = (255, 0, 0)  # Blue in BGR
            else:
                gray_caps = [result]
                result['color'] = 'Gray'
                result['color_thai'] = 'เทา'
                result['color_bgr'] = (128, 128, 128)  # Gray in BGR
            
            # Calculate histogram-based features for refinement
            std = result['std']
            median = result['median']
            hist = result['histogram']
            # Percentage of bright pixels (>150)
            bright_ratio = hist[150:].sum() / hist.sum() * 100
            # Percentage of dark pixels (<80)
            dark_ratio = hist[:80].sum() / hist.sum() * 100
            
            # Refine classification using additional criteria
            # If bright_ratio is very high (>40%), likely yellow
            # If dark_ratio is very high (>40%), likely gray
            if bright_ratio > 40 and result['color'] != 'Yellow':
                # Move to yellow if bright enough
                if result in blue_caps:
                    blue_caps.remove(result)
                if result in gray_caps:
                    gray_caps.remove(result)
                yellow_caps = [result]
                result['color'] = 'Yellow'
                result['color_thai'] = 'เหลือง'
                result['color_bgr'] = (0, 255, 255)
            elif dark_ratio > 40 and result['color'] != 'Gray':
                # Move to gray if dark enough
                if result in yellow_caps:
                    yellow_caps.remove(result)
                if result in blue_caps:
                    blue_caps.remove(result)
                gray_caps = [result]
                result['color'] = 'Gray'
                result['color_thai'] = 'เทา'
                result['color_bgr'] = (128, 128, 128)
            
            # Store additional metrics
            result['bright_ratio'] = bright_ratio
            result['dark_ratio'] = dark_ratio
            
            # Draw labels on image
            self.labeled_image = self.draw_labels_on_image(yellow_caps, blue_caps, gray_caps)
            
            # Display labeled image
            self.display_labeled_image(self.labeled_image)
            
            # Update results text
            cap_label = f"ฝาที่เลือก (Index: {self.selected_cap_index + 1})" if hasattr(self, 'selected_cap_index') and self.selected_cap_index is not None else "ฝาที่เลือก"
            results = "=== ผลการแยกฝาตามสี ===\n\n"
            
            if yellow_caps:
                results += f"🟡 สีเหลือง: {cap_label}\n"
                r = yellow_caps[0]
                results += f"  Mean={r['mean']:.2f}, Std={r['std']:.2f}, "
                results += f"Median={r['median']:.2f}, Bright%={r['bright_ratio']:.1f}%, Dark%={r['dark_ratio']:.1f}%\n"
            elif blue_caps:
                results += f"🔵 สีฟ้า: {cap_label}\n"
                r = blue_caps[0]
                results += f"  Mean={r['mean']:.2f}, Std={r['std']:.2f}, "
                results += f"Median={r['median']:.2f}, Bright%={r['bright_ratio']:.1f}%, Dark%={r['dark_ratio']:.1f}%\n"
            elif gray_caps:
                results += f"⚫ สีเทา: {cap_label}\n"
                r = gray_caps[0]
                results += f"  Mean={r['mean']:.2f}, Std={r['std']:.2f}, "
                results += f"Median={r['median']:.2f}, Bright%={r['bright_ratio']:.1f}%, Dark%={r['dark_ratio']:.1f}%\n"
            
            results += "\n=== เกณฑ์ที่ใช้ ===\n"
            results += "1. Mean Intensity (หลัก)\n"
            results += "2. Standard Deviation (ความแปรปรวน)\n"
            results += "3. Median Intensity (ค่ามัธยฐาน)\n"
            results += "4. Bright Pixel Ratio (เปอร์เซ็นต์พิกเซลสว่าง >150)\n"
            results += "5. Dark Pixel Ratio (เปอร์เซ็นต์พิกเซลมืด <80)\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            
            # Update summary
            cap_label = f"ฝาที่เลือก (Index: {self.selected_cap_index + 1})" if hasattr(self, 'selected_cap_index') and self.selected_cap_index is not None else "ฝาที่เลือก"
            summary = "=== สรุปผลการแยกฝา ===\n\n"
            
            if yellow_caps:
                summary += f"🟡 สีเหลือง: {cap_label}\n"
                summary += f"   Mean: {yellow_caps[0]['mean']:.2f}\n"
            elif blue_caps:
                summary += f"🔵 สีฟ้า: {cap_label}\n"
                summary += f"   Mean: {blue_caps[0]['mean']:.2f}\n"
            elif gray_caps:
                summary += f"⚫ สีเทา: {cap_label}\n"
                summary += f"   Mean: {gray_caps[0]['mean']:.2f}\n"
            
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(1.0, summary)
            
            color_name = yellow_caps[0]['color_thai'] if yellow_caps else (blue_caps[0]['color_thai'] if blue_caps else gray_caps[0]['color_thai'])
            self.log_info(f"✅ แยกฝาแล้ว: {color_name} ({cap_label})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.log_info(f"❌ เกิดข้อผิดพลาด: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def draw_labels_on_image(self, yellow_caps, blue_caps, gray_caps):
        """Draw color labels on the original image"""
        if self.current_image is None or self.detection_result is None:
            return None
        
        # Create a copy of the image
        labeled_img = self.current_image.copy()
        
        # Get all detections
        detections = self.detection_result.get('detections', [])
        
        # Create a mapping from cap index to color (only for selected cap)
        color_map = {}
        if self.intensity_results and hasattr(self, 'selected_cap_index') and self.selected_cap_index is not None:
            result = self.intensity_results[0]
            color_map[self.selected_cap_index] = {
                'color': result.get('color', 'Unknown'),
                'color_thai': result.get('color_thai', 'ไม่ทราบ'),
                'color_bgr': result.get('color_bgr', (255, 255, 255)),
                'mean': result['mean']
            }
        
        # Draw bounding box and label only for selected cap
        if self.selected_cap_index is not None and self.selected_cap_index < len(detections):
            detection = detections[self.selected_cap_index]
            if self.selected_cap_index in color_map:
                color_info = color_map[self.selected_cap_index]
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box with color
                color_bgr = color_info['color_bgr']
                cv2.rectangle(labeled_img, (x1, y1), (x2, y2), color_bgr, 3)
                
                # Prepare label text
                label_text = f"{color_info['color_thai']} (Mean:{color_info['mean']:.1f})"
                
                # Calculate text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # Draw background rectangle for text
                text_x = x1
                text_y = y1 - 10 if y1 > 30 else y2 + text_height + 10
                
                # Ensure text is within image bounds
                if text_y < text_height:
                    text_y = y2 + text_height + 10
                if text_y + text_height > labeled_img.shape[0]:
                    text_y = y1 - 10
                
                # Draw filled rectangle for text background
                cv2.rectangle(labeled_img, 
                             (text_x, text_y - text_height - 5),
                             (text_x + text_width + 10, text_y + baseline + 5),
                             color_bgr, -1)
                
                # Draw text
                cv2.putText(labeled_img, label_text, (text_x + 5, text_y),
                           font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        
        return labeled_img
    
    def display_labeled_image(self, image):
        """Display labeled image in canvas"""
        if image is None:
            return
        
        # Resize for display
        display_width = 800
        h, w = image.shape[:2]
        scale = display_width / w if w > display_width else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        display_image = cv2.resize(image, (new_w, new_h))
        
        # Convert BGR to RGB
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        try:
            from PIL import Image, ImageTk
        except ImportError:
            import Image
            import ImageTk
        
        pil_image = Image.fromarray(display_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Clear and display
        self.labeled_canvas.delete("all")
        self.labeled_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.labeled_canvas.image = photo  # Keep a reference
        self.labeled_canvas.config(scrollregion=self.labeled_canvas.bbox("all"))
    
    def log_info(self, message):
        """Log information message"""
        print(message)
        # Can add to log widget if needed


def main():
    """Main function"""
    root = tk.Tk()
    app = CapIntensityAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

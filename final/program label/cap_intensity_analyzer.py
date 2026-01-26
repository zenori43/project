# -*- coding: utf-8 -*-
"""
Cap Intensity Analyzer
โปรแกรมวิเคราะห์ความเข้มในภาพฝาเพื่อหาเกณฑ์การแยกฝา
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
    """โปรแกรมวิเคราะห์ความเข้มในภาพฝา"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Cap Intensity Analyzer - วิเคราะห์ความเข้มฝา")
        self.root.geometry("1400x900")
        
        # Initialize detector
        self.cap_detector = None
        self.current_image = None
        self.current_image_path = None
        self.cropped_caps = []  # List of cropped cap images
        self.intensity_results = []  # List of intensity analysis results
        self.detection_result = None  # Store detection result for drawing labels
        self.labeled_image = None  # Image with color labels
        
        # Settings
        self.conf_threshold = 0.5
        self.crop_margin = 10
        
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
        title_label = ttk.Label(main_frame, text="Cap Intensity Analyzer - วิเคราะห์ความเข้มฝา", 
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
        
        # Detect and crop button
        self.detect_btn = ttk.Button(control_frame, text="ตรวจจับและครอปฝา", 
                                    command=self.detect_and_crop, state=tk.DISABLED)
        self.detect_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Analyze intensity button
        self.analyze_btn = ttk.Button(control_frame, text="วิเคราะห์ความเข้ม", 
                                     command=self.analyze_intensity, state=tk.DISABLED)
        self.analyze_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(left_panel, text="ตั้งค่า", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
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
        
        # Intensity threshold for separation
        threshold_frame = ttk.LabelFrame(left_panel, text="เกณฑ์ความเข้ม", padding="10")
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(threshold_frame, text="Threshold (0-255):").pack(anchor=tk.W)
        self.threshold_var = tk.IntVar(value=128)
        threshold_scale = ttk.Scale(threshold_frame, from_=0, to=255, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X, pady=(0, 5))
        self.threshold_label = ttk.Label(threshold_frame, text="128")
        self.threshold_label.pack(anchor=tk.W)
        threshold_scale.configure(command=lambda v: self.threshold_label.config(text=f"{int(v)}"))
        
        # Apply threshold button (now for 3-color classification)
        self.apply_threshold_btn = ttk.Button(threshold_frame, text="แยกฝาและ Label (3 สี)", 
                                            command=self.classify_and_label_caps, state=tk.DISABLED)
        self.apply_threshold_btn.pack(fill=tk.X, pady=(5, 0))
        
        # Results summary
        summary_frame = ttk.LabelFrame(left_panel, text="สรุปผล", padding="10")
        summary_frame.pack(fill=tk.BOTH, expand=True)
        
        self.summary_text = tk.Text(summary_frame, height=10, width=40, wrap=tk.WORD)
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
        
        # Image display with scroll
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
        
        # Intensity analysis tab
        intensity_tab = ttk.Frame(self.notebook)
        self.notebook.add(intensity_tab, text="วิเคราะห์ความเข้ม")
        
        # Matplotlib figure for intensity histogram
        self.intensity_fig = Figure(figsize=(10, 6), dpi=100)
        self.intensity_ax = self.intensity_fig.add_subplot(111)
        self.intensity_canvas = FigureCanvasTkAgg(self.intensity_fig, intensity_tab)
        self.intensity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Labeled image tab
        labeled_tab = ttk.Frame(self.notebook)
        self.notebook.add(labeled_tab, text="ภาพที่มี Label")
        
        labeled_scroll = ttk.Scrollbar(labeled_tab, orient=tk.VERTICAL)
        labeled_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.labeled_canvas = tk.Canvas(labeled_tab, yscrollcommand=labeled_scroll.set)
        labeled_scroll.config(command=self.labeled_canvas.yview)
        self.labeled_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Results tab
        results_tab = ttk.Frame(self.notebook)
        self.notebook.add(results_tab, text="ผลลัพธ์")
        
        results_scroll = ttk.Scrollbar(results_tab, orient=tk.VERTICAL)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(results_tab, wrap=tk.WORD, yscrollcommand=results_scroll.set)
        results_scroll.config(command=self.results_text.yview)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
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
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.image = photo  # Keep a reference
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
            self.cropped_caps = self.cap_detector.crop_detections(
                image=self.current_image,
                result=result,
                margin=self.crop_margin
            )
            
            self.log_info(f"✅ ครอปฝาได้ {len(self.cropped_caps)} ฝา")
            
            # Display cropped caps
            self.display_cropped_caps()
            
            # Enable analyze button
            self.analyze_btn.config(state=tk.NORMAL)
            self.apply_threshold_btn.config(state=tk.NORMAL)
            
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
            # Resize for display
            h, w = cap_image.shape[:2]
            scale = max_width / w if w > max_width else 1.0
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
            
            # Add label
            label_text = f"ฝาที่ {i+1} ({w}x{h})"
            self.caps_canvas.create_text(10, y_offset, anchor=tk.NW, text=label_text, 
                                         font=("Arial", 12, "bold"))
            y_offset += 25
            
            # Display image
            self.caps_canvas.create_image(10, y_offset, anchor=tk.NW, image=photo)
            self.caps_canvas.image_refs = getattr(self.caps_canvas, 'image_refs', [])
            self.caps_canvas.image_refs.append(photo)  # Keep reference
            
            y_offset += new_h + 20
        
        self.caps_canvas.config(scrollregion=self.caps_canvas.bbox("all"))
    
    def analyze_intensity(self):
        """Analyze intensity of cropped caps"""
        if not self.cropped_caps:
            messagebox.showwarning("Warning", "กรุณาตรวจจับและครอปฝาก่อน")
            return
        
        try:
            self.log_info("📊 กำลังวิเคราะห์ความเข้ม...")
            self.intensity_results = []
            
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
                
                self.log_info(f"   ฝาที่ {i+1}: Mean={mean_intensity:.2f}, Std={std_intensity:.2f}, "
                            f"Min={min_intensity:.0f}, Max={max_intensity:.0f}, Median={median_intensity:.2f}")
            
            # Update summary
            self.update_summary()
            
            # Plot histogram
            self.plot_intensity_histogram()
            
            # Update results text
            self.update_results_text()
            
            self.log_info("✅ วิเคราะห์ความเข้มเสร็จสิ้น")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.log_info(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    
    def plot_intensity_histogram(self):
        """Plot intensity histogram"""
        self.intensity_ax.clear()
        
        if not self.intensity_results:
            return
        
        # Plot histogram for each cap
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.intensity_results)))
        
        for i, result in enumerate(self.intensity_results):
            self.intensity_ax.plot(result['histogram'], 
                                  label=f"ฝาที่ {result['cap_index']} (Mean={result['mean']:.1f})",
                                  color=colors[i], alpha=0.7)
        
        # Add threshold line
        threshold = self.threshold_var.get()
        self.intensity_ax.axvline(x=threshold, color='red', linestyle='--', 
                                label=f'Threshold={threshold}')
        
        self.intensity_ax.set_xlabel('Intensity (0-255)')
        self.intensity_ax.set_ylabel('Frequency')
        self.intensity_ax.set_title('Intensity Histogram - ฮิสโตแกรมความเข้ม')
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
        """Classify caps into 3 colors (Yellow, Blue, Gray) based on mean intensity and draw labels"""
        if not self.intensity_results:
            messagebox.showwarning("Warning", "กรุณาวิเคราะห์ความเข้มก่อน")
            return
        
        if self.current_image is None or self.detection_result is None:
            messagebox.showwarning("Warning", "ไม่มีภาพหรือผลการตรวจจับ")
            return
        
        try:
            # Sort caps by mean intensity
            sorted_results = sorted(self.intensity_results, key=lambda x: x['mean'], reverse=True)
            
            # Classify into 3 groups based on mean intensity
            num_caps = len(sorted_results)
            yellow_caps = []  # Highest mean
            blue_caps = []    # Middle mean
            gray_caps = []    # Lowest mean
            
            if num_caps == 1:
                # Only one cap - assign to middle (blue)
                blue_caps = sorted_results
            elif num_caps == 2:
                # Two caps - highest = yellow, lowest = gray
                yellow_caps = [sorted_results[0]]
                gray_caps = [sorted_results[1]]
            else:
                # Three or more caps - divide into 3 groups
                # Use dynamic thresholds based on mean values
                means = [r['mean'] for r in sorted_results]
                mean_min = min(means)
                mean_max = max(means)
                mean_range = mean_max - mean_min
                
                # Calculate thresholds (33% and 67% of range)
                threshold_high = mean_min + mean_range * 0.67
                threshold_low = mean_min + mean_range * 0.33
                
                for result in sorted_results:
                    mean = result['mean']
                    if mean >= threshold_high:
                        yellow_caps.append(result)
                    elif mean >= threshold_low:
                        blue_caps.append(result)
                    else:
                        gray_caps.append(result)
            
            # Additional criteria for accuracy - refine classification
            # Use std, median, and histogram distribution to verify/refine
            for result in sorted_results:
                mean = result['mean']
                std = result['std']
                median = result['median']
                
                # Calculate histogram-based features
                hist = result['histogram']
                # Percentage of bright pixels (>150)
                bright_ratio = hist[150:].sum() / hist.sum() * 100
                # Percentage of dark pixels (<80)
                dark_ratio = hist[:80].sum() / hist.sum() * 100
                
                # Determine initial classification based on mean
                initial_group = None
                if result in yellow_caps:
                    initial_group = 'yellow'
                elif result in blue_caps:
                    initial_group = 'blue'
                else:
                    initial_group = 'gray'
                
                # Refine classification using additional criteria
                # If bright_ratio is very high (>40%), likely yellow
                # If dark_ratio is very high (>40%), likely gray
                # Otherwise, likely blue
                if bright_ratio > 40 and initial_group != 'yellow':
                    # Move to yellow if bright enough
                    if result in blue_caps:
                        blue_caps.remove(result)
                    if result in gray_caps:
                        gray_caps.remove(result)
                    yellow_caps.append(result)
                    final_group = 'yellow'
                elif dark_ratio > 40 and initial_group != 'gray':
                    # Move to gray if dark enough
                    if result in yellow_caps:
                        yellow_caps.remove(result)
                    if result in blue_caps:
                        blue_caps.remove(result)
                    gray_caps.append(result)
                    final_group = 'gray'
                else:
                    final_group = initial_group
                
                # Store classification info
                if final_group == 'yellow' or result in yellow_caps:
                    result['color'] = 'Yellow'
                    result['color_thai'] = 'เหลือง'
                    result['color_bgr'] = (0, 255, 255)  # Yellow in BGR
                elif final_group == 'blue' or result in blue_caps:
                    result['color'] = 'Blue'
                    result['color_thai'] = 'ฟ้า'
                    result['color_bgr'] = (255, 0, 0)  # Blue in BGR
                else:
                    result['color'] = 'Gray'
                    result['color_thai'] = 'เทา'
                    result['color_bgr'] = (128, 128, 128)  # Gray in BGR
                
                # Store additional metrics
                result['bright_ratio'] = bright_ratio
                result['dark_ratio'] = dark_ratio
            
            # Draw labels on image
            self.labeled_image = self.draw_labels_on_image(yellow_caps, blue_caps, gray_caps)
            
            # Display labeled image
            self.display_labeled_image(self.labeled_image)
            
            # Update results text
            results = "=== ผลการแยกฝาตามสี (3 สี) ===\n\n"
            results += f"🟡 สีเหลือง (Mean สูงสุด): {len(yellow_caps)} ฝา\n"
            for r in yellow_caps:
                results += f"  - ฝาที่ {r['cap_index']}: Mean={r['mean']:.2f}, Std={r['std']:.2f}, "
                results += f"Median={r['median']:.2f}, Bright%={r['bright_ratio']:.1f}%, Dark%={r['dark_ratio']:.1f}%\n"
            
            results += f"\n🔵 สีฟ้า (Mean กลาง): {len(blue_caps)} ฝา\n"
            for r in blue_caps:
                results += f"  - ฝาที่ {r['cap_index']}: Mean={r['mean']:.2f}, Std={r['std']:.2f}, "
                results += f"Median={r['median']:.2f}, Bright%={r['bright_ratio']:.1f}%, Dark%={r['dark_ratio']:.1f}%\n"
            
            results += f"\n⚫ สีเทา (Mean ต่ำสุด): {len(gray_caps)} ฝา\n"
            for r in gray_caps:
                results += f"  - ฝาที่ {r['cap_index']}: Mean={r['mean']:.2f}, Std={r['std']:.2f}, "
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
            summary = "=== สรุปผลการแยกฝา ===\n\n"
            summary += f"🟡 สีเหลือง: {len(yellow_caps)} ฝา\n"
            if yellow_caps:
                means = [r['mean'] for r in yellow_caps]
                summary += f"   Mean range: {np.min(means):.2f} - {np.max(means):.2f}\n"
            
            summary += f"\n🔵 สีฟ้า: {len(blue_caps)} ฝา\n"
            if blue_caps:
                means = [r['mean'] for r in blue_caps]
                summary += f"   Mean range: {np.min(means):.2f} - {np.max(means):.2f}\n"
            
            summary += f"\n⚫ สีเทา: {len(gray_caps)} ฝา\n"
            if gray_caps:
                means = [r['mean'] for r in gray_caps]
                summary += f"   Mean range: {np.min(means):.2f} - {np.max(means):.2f}\n"
            
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(1.0, summary)
            
            self.log_info(f"✅ แยกฝาแล้ว: เหลือง {len(yellow_caps)} ฝา, ฟ้า {len(blue_caps)} ฝา, เทา {len(gray_caps)} ฝา")
            
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
        
        # Create a mapping from cap index to color
        color_map = {}
        for result in self.intensity_results:
            cap_idx = result['cap_index'] - 1  # Convert to 0-based index
            if cap_idx < len(detections):
                color_map[cap_idx] = {
                    'color': result.get('color', 'Unknown'),
                    'color_thai': result.get('color_thai', 'ไม่ทราบ'),
                    'color_bgr': result.get('color_bgr', (255, 255, 255)),
                    'mean': result['mean']
                }
        
        # Draw bounding boxes and labels
        for i, detection in enumerate(detections):
            if i not in color_map:
                continue
            
            color_info = color_map[i]
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

# -*- coding: utf-8 -*-
"""
Cap Brightness Analyzer
วิเคราะห์ค่ากลางของ brightness/contrast จากภาพฝาเพื่อปรับภาพให้แสงเท่ากัน
"""

import os
import sys

# Handle X11 display issues (fix X Error: BadLength)
os.environ['QT_X11_NO_MITSHM'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
# Additional X11 fixes
os.environ['XLIB_SKIP_ARGB_VISUALS'] = '1'
# Disable X11 RENDER extension to avoid RenderAddGlyphs error
os.environ['TK_SILENCE_DEPRECATION'] = '1'
# Disable font rendering acceleration
os.environ['_TKINTER_NO_FONT_RENDERING'] = '1'
# Force disable X11 RENDER extension
os.environ['X11_NO_RENDER'] = '1'
# Use simple bitmap fonts only
os.environ['TK_USE_BITMAP_FONTS'] = '1'

# CRITICAL: Setup CUDA paths BEFORE importing any libs modules
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    pass

import cv2
import numpy as np

# Try to import tkinter with error handling
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    GUI_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"⚠️ GUI not available: {e}")
    print("   Running in headless mode...")
    GUI_AVAILABLE = False
    # Create dummy classes for headless mode
    class tk:
        class Tk:
            def __init__(self, *args, **kwargs):
                pass
            def mainloop(self):
                pass
        class StringVar:
            def __init__(self, *args, **kwargs):
                self.value = kwargs.get('value', '')
            def get(self):
                return self.value
            def set(self, value):
                self.value = value
    class ttk:
        class Frame:
            def __init__(self, *args, **kwargs):
                pass
            def grid(self, *args, **kwargs):
                pass
        class LabelFrame:
            def __init__(self, *args, **kwargs):
                pass
            def grid(self, *args, **kwargs):
                pass
        class Button:
            def __init__(self, *args, **kwargs):
                pass
            def grid(self, *args, **kwargs):
                pass
            def config(self, *args, **kwargs):
                pass
        class Label:
            def __init__(self, *args, **kwargs):
                pass
            def grid(self, *args, **kwargs):
                pass
        class Entry:
            def __init__(self, *args, **kwargs):
                pass
            def grid(self, *args, **kwargs):
                pass
        class Scrollbar:
            def __init__(self, *args, **kwargs):
                pass
            def grid(self, *args, **kwargs):
                pass
            def config(self, *args, **kwargs):
                pass
        class Text:
            def __init__(self, *args, **kwargs):
                pass
            def grid(self, *args, **kwargs):
                pass
            def insert(self, *args, **kwargs):
                pass
            def delete(self, *args, **kwargs):
                pass
            def see(self, *args, **kwargs):
                pass
    class filedialog:
        @staticmethod
        def askdirectory(*args, **kwargs):
            return None
        @staticmethod
        def askopenfilenames(*args, **kwargs):
            return None
    class messagebox:
        @staticmethod
        def showwarning(*args, **kwargs):
            print(f"WARNING: {args[1] if len(args) > 1 else ''}")
        @staticmethod
        def showerror(*args, **kwargs):
            print(f"ERROR: {args[1] if len(args) > 1 else ''}")

from pathlib import Path
from typing import List, Dict
import statistics

# Import cap detection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.detection.capmodel import CapDetector
from config.settings import CAP_MODEL_PATH


class CapBrightnessAnalyzer:
    """โปรแกรมวิเคราะห์ค่ากลางของ brightness/contrast จากภาพฝา"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Cap Brightness Analyzer - วิเคราะห์ค่ากลางของ Brightness/Contrast")
        self.root.geometry("1200x800")
        
        # Initialize detector
        self.cap_detector = None
        self.image_paths = []
        self.analysis_results = []
        
        self.setup_gui()
        self.load_cap_model()
    
    def setup_gui(self):
        """Setup GUI components"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title (use simple font to avoid X11 issues)
        title_label = ttk.Label(main_frame, text="Cap Brightness Analyzer")
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection (English only to avoid X11 RENDER issues)
        file_frame = ttk.LabelFrame(main_frame, text="Select Folder or Files", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="Select Folder", 
                  command=self.select_folder).grid(row=0, column=0, padx=5)
        ttk.Button(file_frame, text="Select Files", 
                  command=self.select_files).grid(row=0, column=1, padx=5)
        
        self.folder_path_var = tk.StringVar(value="/home/nvidia/Desktop/final_boss/backupsdcard/final/captured_images/sentech_camera")
        folder_entry = ttk.Entry(file_frame, textvariable=self.folder_path_var, width=60)
        folder_entry.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Analysis button (English only to avoid X11 RENDER issues)
        self.analyze_btn = ttk.Button(main_frame, text="Analyze Brightness", 
                                      command=self.analyze_brightness, state=tk.DISABLED)
        self.analyze_btn.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Scrollable text area
        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.results_text = tk.Text(results_frame, height=20, width=80, 
                                    yscrollcommand=scrollbar.set)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.results_text.yview)
        
        # Summary frame
        summary_frame = ttk.LabelFrame(main_frame, text="Summary", padding="10")
        summary_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.summary_text = tk.Text(summary_frame, height=8, width=80)
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=5, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def load_cap_model(self):
        """Load cap detection model"""
        try:
            self.log_info("[*] Loading Cap Detection Model...")
            self.cap_detector = CapDetector(CAP_MODEL_PATH)
            self.log_info("[OK] Cap Detection Model loaded successfully")
            self.analyze_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.log_info(f"[ERROR] Failed to load Model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load Model: {str(e)}")
    
    def log_info(self, message):
        """Log message to log area"""
        print(message)  # Always print to console
        if GUI_AVAILABLE and hasattr(self, 'log_text'):
            try:
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
                self.root.update_idletasks()  # Use update_idletasks instead of update
            except Exception as e:
                # If GUI update fails, just print
                print(f"GUI update failed: {e}")
    
    def select_folder(self):
        """Select folder containing images"""
        folder = filedialog.askdirectory(
            initialdir=self.folder_path_var.get(),
            title="เลือกโฟลเดอร์ที่มีภาพฝา"
        )
        if folder:
            self.folder_path_var.set(folder)
            # Find all image files
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            self.image_paths = []
            for ext in image_extensions:
                self.image_paths.extend(Path(folder).glob(f'*{ext}'))
                self.image_paths.extend(Path(folder).glob(f'*{ext.upper()}'))
            self.image_paths = sorted([str(p) for p in self.image_paths])
            self.log_info(f"✅ พบภาพ {len(self.image_paths)} ไฟล์")
    
    def select_files(self):
        """Select image files"""
        files = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if files:
            self.image_paths = list(files)
            self.folder_path_var.set(os.path.dirname(files[0]))
            self.log_info(f"[OK] Selected {len(self.image_paths)} images")
    
    def find_bottommost_cap(self, detections):
        """Find the cap with highest y coordinate (bottommost)"""
        if not detections:
            return None
        
        bottommost = None
        max_y = -1
        
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) >= 4:
                # bbox format: [x1, y1, x2, y2]
                y_center = (bbox[1] + bbox[3]) / 2  # Use center y or bottom y
                y_bottom = bbox[3]  # Use bottom y coordinate
                
                if y_bottom > max_y:
                    max_y = y_bottom
                    bottommost = detection
        
        return bottommost
    
    def analyze_cap_brightness(self, cap_image):
        """Analyze brightness/contrast/intensity of a cap image"""
        if cap_image is None:
            return None
        
        # Convert to grayscale if needed
        if len(cap_image.shape) == 3:
            gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cap_image.copy()
        
        # Calculate statistics
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        min_intensity = int(np.min(gray))
        max_intensity = int(np.max(gray))
        median_intensity = float(np.median(gray))
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate brightness adjustment needed (target: ~128)
        target_mean = 128.0
        brightness_adjustment = target_mean - mean_intensity
        
        # Calculate contrast (std deviation relative to mean)
        contrast_ratio = std_intensity / mean_intensity if mean_intensity > 0 else 1.0
        
        return {
            'mean': mean_intensity,
            'std': std_intensity,
            'min': min_intensity,
            'max': max_intensity,
            'median': median_intensity,
            'brightness_adjustment': brightness_adjustment,
            'contrast_ratio': contrast_ratio,
            'histogram': hist
        }
    
    def analyze_brightness(self):
        """Analyze brightness from all images"""
        if not self.image_paths:
            messagebox.showwarning("Warning", "Please select folder or image files first")
            return
        
        if self.cap_detector is None:
            messagebox.showerror("Error", "Cap detector not loaded")
            return
        
        try:
            self.log_info("[*] Starting brightness analysis...")
            self.results_text.delete(1.0, tk.END)
            self.summary_text.delete(1.0, tk.END)
            
            self.analysis_results = []
            all_brightness_adjustments = []
            all_contrast_ratios = []
            all_means = []
            all_stds = []
            all_medians = []
            
            total_images = len(self.image_paths)
            processed = 0
            
            for image_path in self.image_paths:
                try:
                    self.log_info(f"[*] Processing: {os.path.basename(image_path)}")
                    
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        self.log_info(f"[WARN] Cannot load image: {image_path}")
                        continue
                    
                    # Detect caps
                    result = self.cap_detector.detect_caps(image_path)
                    if 'error' in result:
                        self.log_info(f"[WARN] Cap detection failed: {result['error']}")
                        continue
                    
                    detections = result.get('detections', [])
                    if not detections:
                        self.log_info(f"[WARN] No caps found in image: {os.path.basename(image_path)}")
                        continue
                    
                    # Find bottommost cap (highest y coordinate)
                    bottommost_cap = self.find_bottommost_cap(detections)
                    if bottommost_cap is None:
                        self.log_info(f"[WARN] No bottommost cap found: {os.path.basename(image_path)}")
                        continue
                    
                    # Crop bottommost cap
                    bbox = bottommost_cap.get('bbox', [])
                    if len(bbox) < 4:
                        continue
                    
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    margin = 10
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(image.shape[1], x2 + margin)
                    y2 = min(image.shape[0], y2 + margin)
                    
                    cropped_cap = image[y1:y2, x1:x2]
                    
                    # Analyze brightness
                    analysis = self.analyze_cap_brightness(cropped_cap)
                    if analysis:
                        analysis['image_path'] = image_path
                        analysis['image_name'] = os.path.basename(image_path)
                        self.analysis_results.append(analysis)
                        
                        all_brightness_adjustments.append(analysis['brightness_adjustment'])
                        all_contrast_ratios.append(analysis['contrast_ratio'])
                        all_means.append(analysis['mean'])
                        all_stds.append(analysis['std'])
                        all_medians.append(analysis['median'])
                        
                        self.log_info(f"✅ วิเคราะห์สำเร็จ: Mean={analysis['mean']:.2f}, Brightness Adj={analysis['brightness_adjustment']:.2f}")
                    
                    processed += 1
                    
                except Exception as e:
                    self.log_info(f"[ERROR] Error occurred: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Calculate central values (median of all adjustments)
            if all_brightness_adjustments:
                central_brightness = statistics.median(all_brightness_adjustments)
                central_contrast_ratio = statistics.median(all_contrast_ratios)
                central_mean = statistics.median(all_means)
                central_std = statistics.median(all_stds)
                central_median = statistics.median(all_medians)
                
                # Calculate average as alternative
                avg_brightness = statistics.mean(all_brightness_adjustments)
                avg_contrast_ratio = statistics.mean(all_contrast_ratios)
                avg_mean = statistics.mean(all_means)
                
                # Display results
                self.display_results(central_brightness, central_contrast_ratio, 
                                   central_mean, central_std, central_median,
                                   avg_brightness, avg_contrast_ratio, avg_mean,
                                   processed, total_images)
                
                self.log_info(f"[OK] Analysis completed: Processed {processed}/{total_images} images")
            else:
                self.log_info("[ERROR] No analysis data found")
                messagebox.showwarning("Warning", "No analysis data found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred: {str(e)}")
            self.log_info(f"[ERROR] Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_results(self, central_brightness, central_contrast_ratio,
                       central_mean, central_std, central_median,
                       avg_brightness, avg_contrast_ratio, avg_mean,
                       processed, total):
        """Display analysis results"""
        # Detailed results (English only to avoid X11 RENDER issues)
        results = f"=== Brightness Analysis Results ===\n\n"
        results += f"Processed images: {processed}/{total}\n\n"
        
        results += "=== Central Values (Median) ===\n"
        results += f"Brightness Adjustment: {central_brightness:.2f}\n"
        results += f"Contrast Ratio: {central_contrast_ratio:.3f}\n"
        results += f"Mean Intensity: {central_mean:.2f}\n"
        results += f"Std Intensity: {central_std:.2f}\n"
        results += f"Median Intensity: {central_median:.2f}\n\n"
        
        results += "=== Average Values ===\n"
        results += f"Brightness Adjustment: {avg_brightness:.2f}\n"
        results += f"Contrast Ratio: {avg_contrast_ratio:.3f}\n"
        results += f"Mean Intensity: {avg_mean:.2f}\n\n"
        
        results += "=== Per Image Details ===\n"
        for i, result in enumerate(self.analysis_results, 1):
            results += f"\n{i}. {result['image_name']}\n"
            results += f"   Mean: {result['mean']:.2f}\n"
            results += f"   Std: {result['std']:.2f}\n"
            results += f"   Brightness Adj: {result['brightness_adjustment']:.2f}\n"
            results += f"   Contrast Ratio: {result['contrast_ratio']:.3f}\n"
        
        self.results_text.insert(1.0, results)
        
        # Summary
        summary = f"=== Recommended Values for Image Adjustment ===\n\n"
        summary += f"[OK] Recommended Brightness: {central_brightness:.0f}\n"
        summary += f"   (Average: {avg_brightness:.0f})\n\n"
        summary += f"[OK] Recommended Contrast Ratio: {central_contrast_ratio:.3f}\n"
        summary += f"   (Average: {avg_contrast_ratio:.3f})\n\n"
        summary += f"[NOTE] Notes:\n"
        summary += f"   - Brightness: Value to add/subtract to make Mean ≈ 128\n"
        summary += f"   - Contrast: Ratio of Std/Mean\n"
        summary += f"   - Use Median to reduce impact of outliers\n"
        
        self.summary_text.insert(1.0, summary)


def main():
    """Main function with X11 error handling"""
    if not GUI_AVAILABLE:
        print("[ERROR] GUI is not available. Cannot run in interactive mode.")
        print("   Please ensure tkinter is installed and X11 display is available.")
        return
    
    try:
        root = tk.Tk()
        
        # Disable X11 RENDER extension to avoid RenderAddGlyphs error
        try:
            # Force use of bitmap fonts only (no X11 RENDER)
            root.option_add('*font', 'fixed')
            root.option_add('*Label.font', 'fixed')
            root.option_add('*Button.font', 'fixed')
            root.option_add('*Entry.font', 'fixed')
            root.option_add('*Text.font', 'fixed')
            # Disable antialiasing
            root.tk.call('option', 'add', '*font', 'fixed')
        except:
            pass  # Ignore if option_add fails
        
        app = CapBrightnessAnalyzer(root)
        
        # Try to run mainloop with error handling
        try:
            root.mainloop()
        except Exception as e:
            print(f"[WARN] Error during mainloop: {e}")
            # Try to continue without GUI updates
            if hasattr(app, 'analyze_brightness'):
                print("   You can still use the analyzer programmatically.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize GUI: {e}")
        print("   This might be an X11 display issue.")
        print("   Try:")
        print("   1. Check DISPLAY environment: echo $DISPLAY")
        print("   2. Use Xvfb for headless: xvfb-run -a python3 ...")
        print("   3. Or run with: export QT_X11_NO_MITSHM=1")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

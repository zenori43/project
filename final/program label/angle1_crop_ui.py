"""
โปรแกรมสำหรับครอป angle1 จากภาพหลายไฟล์
ใช้โมเดล bottle.pt ตรวจจับและครอป angle1 แล้วบันทึกเป็น folder
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from datetime import datetime
import threading
import sys
from typing import List, Dict

# เพิ่ม path สำหรับ import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ultralytics import YOLO
    import torch
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Error importing modules: {e}")
    try:
        import tkinter.messagebox as messagebox
        messagebox.showerror("Error", f"ไม่สามารถ import modules ได้: {e}")
    except:
        pass
    sys.exit(1)


class Angle1CropUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Angle1 Crop Tool")
        self.root.geometry("800x600")
        
        # Model path
        self.model_path = r"C:\Users\Win 10 Home\Desktop\project_final\final\models\detection\bottle.pt"
        self.model = None
        self.model_loaded = False
        
        # Data
        self.selected_files = []
        self.output_folder = ""
        self.processing = False
        
        # Confidence threshold
        self.confidence_threshold = 0.5
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        """สร้าง UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Model path
        ttk.Label(main_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_path_var = tk.StringVar(value=self.model_path)
        model_entry = ttk.Entry(main_frame, textvariable=self.model_path_var, width=60)
        model_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, pady=5)
        
        # Model status
        self.model_status_var = tk.StringVar(value="กำลังโหลดโมเดล...")
        ttk.Label(main_frame, textvariable=self.model_status_var, foreground="blue").grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=5
        )
        
        # Confidence threshold
        ttk.Label(main_frame, text="Confidence Threshold:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.confidence_var = tk.DoubleVar(value=self.confidence_threshold)
        confidence_scale = ttk.Scale(
            main_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.confidence_var,
            orient=tk.HORIZONTAL
        )
        confidence_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        self.confidence_label = ttk.Label(main_frame, text=f"{self.confidence_threshold:.2f}")
        self.confidence_label.grid(row=2, column=2, pady=5)
        confidence_scale.configure(command=self.update_confidence_label)
        
        # File selection
        ttk.Label(main_frame, text="Selected Files:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Button(main_frame, text="Select Images", command=self.select_files).grid(
            row=3, column=1, sticky=tk.W, pady=5, padx=5
        )
        
        # File list
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=10)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # Output folder
        ttk.Label(main_frame, text="Output Folder:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.output_folder_var = tk.StringVar()
        output_entry = ttk.Entry(main_frame, textvariable=self.output_folder_var, width=60)
        output_entry.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_folder).grid(row=5, column=2, pady=5)
        
        # Progress
        self.progress_var = tk.StringVar(value="พร้อมใช้งาน")
        ttk.Label(main_frame, textvariable=self.progress_var, foreground="green").grid(
            row=6, column=0, columnspan=3, sticky=tk.W, pady=5
        )
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Process button
        self.process_button = ttk.Button(
            main_frame, 
            text="Process & Save", 
            command=self.process_images,
            state=tk.DISABLED
        )
        self.process_button.grid(row=8, column=0, columnspan=3, pady=10)
        
    def update_confidence_label(self, value):
        """อัปเดต label ของ confidence threshold"""
        self.confidence_threshold = float(value)
        self.confidence_label.config(text=f"{self.confidence_threshold:.2f}")
        
    def browse_model(self):
        """เลือกไฟล์โมเดล"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
            self.model_path = file_path
            self.load_model()
            
    def load_model(self):
        """โหลดโมเดล YOLO"""
        def load_in_thread():
            try:
                model_path = self.model_path_var.get()
                if not os.path.exists(model_path):
                    self.model_status_var.set(f"❌ ไม่พบไฟล์โมเดล: {model_path}")
                    return
                    
                self.model_status_var.set("กำลังโหลดโมเดล...")
                self.model = YOLO(model_path)
                
                # ตรวจสอบ CUDA
                if torch.cuda.is_available():
                    try:
                        self.model.to('cuda')
                        self.model_status_var.set("✅ โมเดลโหลดสำเร็จ (CUDA)")
                    except:
                        self.model.to('cpu')
                        self.model_status_var.set("✅ โมเดลโหลดสำเร็จ (CPU)")
                else:
                    self.model.to('cpu')
                    self.model_status_var.set("✅ โมเดลโหลดสำเร็จ (CPU)")
                    
                self.model_loaded = True
                self.check_ready()
                
            except Exception as e:
                self.model_status_var.set(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                self.model_loaded = False
                self.check_ready()
        
        threading.Thread(target=load_in_thread, daemon=True).start()
        
    def select_files(self):
        """เลือกไฟล์ภาพหลายไฟล์"""
        files = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All Files", "*.*")
            ]
        )
        if files:
            self.selected_files = list(files)
            self.file_listbox.delete(0, tk.END)
            for file in self.selected_files:
                self.file_listbox.insert(tk.END, os.path.basename(file))
            self.check_ready()
            
    def browse_output_folder(self):
        """เลือกโฟลเดอร์สำหรับบันทึกผลลัพธ์"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)
            self.output_folder = folder
            self.check_ready()
            
    def check_ready(self):
        """ตรวจสอบว่าพร้อมประมวลผลหรือไม่"""
        if self.model_loaded and self.selected_files and self.output_folder_var.get():
            self.process_button.config(state=tk.NORMAL)
        else:
            self.process_button.config(state=tk.DISABLED)
            
    def process_images(self):
        """ประมวลผลภาพทั้งหมด"""
        if self.processing:
            messagebox.showwarning("Warning", "กำลังประมวลผลอยู่ กรุณารอสักครู่")
            return
            
        if not self.model_loaded:
            messagebox.showerror("Error", "โมเดลยังไม่โหลด")
            return
            
        if not self.selected_files:
            messagebox.showerror("Error", "กรุณาเลือกไฟล์ภาพ")
            return
            
        output_folder = self.output_folder_var.get()
        if not output_folder:
            messagebox.showerror("Error", "กรุณาเลือกโฟลเดอร์สำหรับบันทึกผลลัพธ์")
            return
            
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        os.makedirs(output_folder, exist_ok=True)
        
        # เริ่มประมวลผลใน thread แยก
        self.processing = True
        self.process_button.config(state=tk.DISABLED)
        self.progress_bar['maximum'] = len(self.selected_files)
        self.progress_bar['value'] = 0
        
        threading.Thread(target=self.process_images_thread, args=(output_folder,), daemon=True).start()
        
    def detect_bottles(self, image: np.ndarray) -> List[Dict]:
        """ตรวจจับ bottles ในภาพ"""
        results = self.model(image, conf=self.confidence_threshold, imgsz=640)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = self.model.names[class_id]
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'label': label,
                    'confidence': confidence,
                    'type': 'bottle'
                }
                detections.append(detection)
        
        return detections
    
    def crop_angle1_regions(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """ครอป angle1 regions จาก detections"""
        angle1_crops = []
        for detection in detections:
            if detection['label'] == "angle1":
                x1, y1, x2, y2 = detection['bbox']
                roi = image[y1:y2, x1:x2].copy()
                angle1_crops.append({
                    'original_crop': roi,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'label': detection['label']
                })
        return angle1_crops
    
    def process_images_thread(self, output_folder):
        """ประมวลผลภาพใน thread แยก"""
        try:
            total_files = len(self.selected_files)
            success_count = 0
            error_count = 0
            
            for idx, image_path in enumerate(self.selected_files):
                try:
                    self.progress_var.set(f"กำลังประมวลผล: {os.path.basename(image_path)} ({idx+1}/{total_files})")
                    
                    # โหลดภาพ
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"ไม่สามารถโหลดภาพได้: {image_path}")
                    
                    # ตรวจจับ bottles
                    detections = self.detect_bottles(image)
                    
                    # ครอป angle1
                    angle1_crops = self.crop_angle1_regions(image, detections)
                    
                    if not angle1_crops:
                        print(f"ไม่พบ angle1 ในภาพ: {image_path}")
                        continue
                    
                    # บันทึก angle1 crops
                    base_name = Path(image_path).stem
                    for crop_idx, crop_data in enumerate(angle1_crops):
                        crop_image = crop_data['original_crop']
                        
                        # สร้างชื่อไฟล์
                        if len(angle1_crops) == 1:
                            output_filename = f"{base_name}_angle1.jpg"
                        else:
                            output_filename = f"{base_name}_angle1_{crop_idx+1}.jpg"
                        
                        output_path = os.path.join(output_folder, output_filename)
                        cv2.imwrite(output_path, crop_image)
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    error_count += 1
                    
                # อัปเดต progress
                self.progress_bar['value'] = idx + 1
                self.root.update_idletasks()
            
            # แสดงผลลัพธ์
            self.progress_var.set(
                f"เสร็จสิ้น! สำเร็จ: {success_count}, ผิดพลาด: {error_count}"
            )
            messagebox.showinfo(
                "เสร็จสิ้น",
                f"ประมวลผลเสร็จสิ้น\nสำเร็จ: {success_count} ไฟล์\nผิดพลาด: {error_count} ไฟล์\n\nบันทึกไว้ที่: {output_folder}"
            )
            
        except Exception as e:
            self.progress_var.set(f"เกิดข้อผิดพลาด: {str(e)}")
            messagebox.showerror("Error", f"เกิดข้อผิดพลาด: {str(e)}")
            
        finally:
            self.processing = False
            self.process_button.config(state=tk.NORMAL)
            self.progress_bar['value'] = 0


def main():
    root = tk.Tk()
    app = Angle1CropUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

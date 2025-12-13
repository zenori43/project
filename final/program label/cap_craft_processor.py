import cv2
import numpy as np
import os
import threading
import time
from pathlib import Path

# Handle X11 display issues
import os
os.environ['QT_X11_NO_MITSHM'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ GUI not available: {e}")
    print("   Running in headless mode...")
    GUI_AVAILABLE = False

# Import our custom modules
from capmodel import CapDetector, initialize_detector as init_cap_detector
from rotationCRAFT import CRAFTTextDetector, initialize_detector as init_craft_detector

class CapCraftProcessor:
    """
    GUI Application สำหรับประมวลผลภาพฝา
    - เลือกรูปภาพ
    - จับฝาด้วย YOLOv8
    - ครอปภาพฝา
    - หมุนภาพด้วย CRAFT
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cap Detection & CRAFT Rotation Processor")
        self.root.geometry("1200x800")
        
        # Initialize detectors
        self.cap_detector = None
        self.craft_detector = None
        
        # Current image and results
        self.current_image = None
        self.current_image_path = None
        self.cap_detection_result = None
        self.craft_results = []
        
        # Setup GUI
        self.setup_gui()
        self.initialize_detectors()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Cap Detection & CRAFT Rotation Processor", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Image selection button
        self.select_btn = ttk.Button(control_frame, text="เลือกรูปภาพ", 
                                    command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Process button
        self.process_btn = ttk.Button(control_frame, text="ประมวลผล", 
                                     command=self.process_image, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Save results button
        self.save_btn = ttk.Button(control_frame, text="บันทึกผลลัพธ์", 
                                  command=self.save_results, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, padx=(10, 0))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="พร้อมใช้งาน")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Image display frame
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Original image display
        orig_frame = ttk.LabelFrame(image_frame, text="รูปภาพต้นฉบับ", padding="5")
        orig_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        orig_frame.columnconfigure(0, weight=1)
        orig_frame.rowconfigure(0, weight=1)
        
        self.orig_canvas = tk.Canvas(orig_frame, bg="white", width=400, height=300)
        self.orig_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results display
        results_frame = ttk.LabelFrame(image_frame, text="ผลลัพธ์การประมวลผล", padding="5")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Cap detection tab
        self.cap_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.cap_tab, text="การจับฝา")
        
        self.cap_canvas = tk.Canvas(self.cap_tab, bg="white", width=400, height=300)
        self.cap_canvas.pack(fill=tk.BOTH, expand=True)
        
        # CRAFT rotation tab
        self.craft_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.craft_tab, text="การหมุน CRAFT")
        
        self.craft_canvas = tk.Canvas(self.craft_tab, bg="white", width=400, height=300)
        self.craft_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Results info frame
        info_frame = ttk.LabelFrame(main_frame, text="ข้อมูลผลลัพธ์", padding="5")
        info_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        info_frame.columnconfigure(0, weight=1)
        
        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for info text
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
    def initialize_detectors(self):
        """Initialize the detection models"""
        try:
            self.update_status("กำลังโหลดโมเดล...")
            
            # Initialize cap detector
            cap_model_path = "/home/nvidia/Desktop/final_boss/backupsdcard/final/cap.pt"
            if os.path.exists(cap_model_path):
                self.cap_detector = init_cap_detector(cap_model_path, conf_threshold=0.5)
                self.log_info("✅ โมเดลจับฝาโหลดสำเร็จ")
            else:
                self.log_info("❌ ไม่พบไฟล์โมเดลจับฝา")
                
            # Initialize CRAFT detector
            craft_model_path = "/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch/craft_mlt_25k.pth"
            craft_refiner_path = "/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch/craft_refiner_CTW1500.pth"
            
            if os.path.exists(craft_model_path):
                self.craft_detector = init_craft_detector(craft_model_path, craft_refiner_path, cuda=False)
                self.log_info("✅ โมเดล CRAFT โหลดสำเร็จ")
            else:
                self.log_info("❌ ไม่พบไฟล์โมเดล CRAFT")
                
            self.update_status("พร้อมใช้งาน")
            
        except Exception as e:
            self.log_info(f"❌ ข้อผิดพลาดในการโหลดโมเดล: {e}")
            self.update_status("เกิดข้อผิดพลาด")
            
    def select_image(self):
        """Select image file"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="เลือกรูปภาพ",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self.load_and_display_image(file_path)
            self.process_btn.config(state=tk.NORMAL)
            self.log_info(f"เลือกรูปภาพ: {os.path.basename(file_path)}")
            
    def load_and_display_image(self, image_path):
        """Load and display image in the original canvas"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("ไม่สามารถโหลดรูปภาพได้")
                
            self.current_image = image
            
            # Resize image for display
            display_image = self.resize_for_display(image, max_width=400, max_height=300)
            
            # Convert to PIL Image
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(display_image_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Display in canvas
            self.orig_canvas.delete("all")
            self.orig_canvas.create_image(200, 150, image=photo)
            self.orig_canvas.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถโหลดรูปภาพ: {e}")
            
    def resize_for_display(self, image, max_width=400, max_height=300):
        """Resize image for display while maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(image, (new_w, new_h))
        
    def process_image(self):
        """Process the selected image"""
        if self.current_image is None:
            messagebox.showwarning("คำเตือน", "กรุณาเลือกรูปภาพก่อน")
            return
            
        if self.cap_detector is None or self.craft_detector is None:
            messagebox.showerror("ข้อผิดพลาด", "โมเดลยังไม่พร้อมใช้งาน")
            return
            
        # Start processing in a separate thread
        self.process_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.update_status("กำลังประมวลผล...")
        
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
        
    def _process_image_thread(self):
        """Process image in separate thread"""
        try:
            # Step 1: Detect caps
            self.log_info("🔍 กำลังจับฝา...")
            self.cap_detection_result = self.cap_detector.detect_caps(image_array=self.current_image)
            
            if 'error' in self.cap_detection_result:
                raise Exception(f"การจับฝาล้มเหลว: {self.cap_detection_result['error']}")
                
            num_caps = self.cap_detection_result['total_detections']
            self.log_info(f"✅ พบฝา {num_caps} ฝา")
            
            # Display cap detection results
            self.root.after(0, self.display_cap_detection)
            
            if num_caps == 0:
                self.root.after(0, self._processing_complete)
                return
                
            # Step 2: Crop cap regions and process with CRAFT
            self.craft_results = []
            cropped_images = self.cap_detector.crop_detections(image=self.current_image, 
                                                             result=self.cap_detection_result, 
                                                             margin=20)
            
            for i, cropped_img in enumerate(cropped_images):
                self.log_info(f"🔄 กำลังประมวลผลฝาที่ {i+1} ด้วย CRAFT...")
                
                # Process with CRAFT
                craft_result = self.craft_detector.detect_text_and_rotate(image_array=cropped_img)
                
                if 'error' not in craft_result:
                    self.craft_results.append({
                        'cap_index': i,
                        'cropped_image': cropped_img,
                        'craft_result': craft_result
                    })
                    angle = craft_result['rotation_angle']
                    self.log_info(f"✅ ฝาที่ {i+1}: มุมหมุน {angle:.2f} องศา")
                else:
                    self.log_info(f"❌ ฝาที่ {i+1}: CRAFT ประมวลผลล้มเหลว")
                    
            # Display CRAFT results
            self.root.after(0, self.display_craft_results)
            self.root.after(0, self._processing_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self._processing_error(str(e)))
            
    def display_cap_detection(self):
        """Display cap detection results"""
        if self.cap_detection_result is None:
            return
            
        # Draw detections on image
        result_image = self.cap_detector.draw_detections(
            image=self.current_image, 
            result=self.cap_detection_result
        )
        
        # Resize for display
        display_image = self.resize_for_display(result_image, max_width=400, max_height=300)
        
        # Convert to PIL Image
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display in canvas
        self.cap_canvas.delete("all")
        self.cap_canvas.create_image(200, 150, image=photo)
        self.cap_canvas.image = photo  # Keep a reference
        
    def display_craft_results(self):
        """Display CRAFT rotation results"""
        if not self.craft_results:
            return
            
        # Create a combined display of all CRAFT results
        if len(self.craft_results) == 1:
            # Single result - show the rotated image
            result = self.craft_results[0]
            craft_result = result['craft_result']
            
            if 'rotated_image' in craft_result and craft_result['rotated_image'] is not None:
                rotated_img = craft_result['rotated_image']
                # Convert RGB to BGR for display
                display_img = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR)
            else:
                display_img = result['cropped_image']
                
        else:
            # Multiple results - create a grid
            num_results = len(self.craft_results)
            cols = min(2, num_results)
            rows = (num_results + cols - 1) // cols
            
            # Calculate individual image size
            img_width = 400 // cols
            img_height = 300 // rows
            
            display_img = np.zeros((300, 400, 3), dtype=np.uint8)
            
            for i, result in enumerate(self.craft_results):
                row = i // cols
                col = i % cols
                
                craft_result = result['craft_result']
                if 'rotated_image' in craft_result and craft_result['rotated_image'] is not None:
                    img = craft_result['rotated_image']
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img = result['cropped_image']
                    
                # Resize image
                img_resized = cv2.resize(img, (img_width, img_height))
                
                # Place in grid
                y_start = row * img_height
                y_end = y_start + img_height
                x_start = col * img_width
                x_end = x_start + img_width
                
                display_img[y_start:y_end, x_start:x_end] = img_resized
                
                # Add text label
                angle = craft_result['rotation_angle']
                cv2.putText(display_img, f"Cap {i+1}: {angle:.1f}°", 
                           (x_start + 5, y_start + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Resize for display
        display_image = self.resize_for_display(display_img, max_width=400, max_height=300)
        
        # Convert to PIL Image
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display in canvas
        self.craft_canvas.delete("all")
        self.craft_canvas.create_image(200, 150, image=photo)
        self.craft_canvas.image = photo  # Keep a reference
        
    def _processing_complete(self):
        """Called when processing is complete"""
        self.progress.stop()
        self.process_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.update_status("ประมวลผลเสร็จสิ้น")
        
    def _processing_error(self, error_msg):
        """Called when processing encounters an error"""
        self.progress.stop()
        self.process_btn.config(state=tk.NORMAL)
        self.update_status("เกิดข้อผิดพลาด")
        self.log_info(f"❌ ข้อผิดพลาด: {error_msg}")
        messagebox.showerror("ข้อผิดพลาด", f"การประมวลผลล้มเหลว:\n{error_msg}")
        
    def save_results(self):
        """Save processing results"""
        if not self.cap_detection_result and not self.craft_results:
            messagebox.showwarning("คำเตือน", "ไม่มีผลลัพธ์ให้บันทึก")
            return
            
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="เลือกโฟลเดอร์บันทึกผลลัพธ์")
        if not output_dir:
            return
            
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            
            # Save cap detection result
            if self.cap_detection_result:
                cap_result_path = os.path.join(output_dir, f"{base_name}_cap_detection_{timestamp}.jpg")
                self.cap_detector.save_detection_result(
                    cap_result_path, 
                    image=self.cap_detector.draw_detections(image=self.current_image, result=self.cap_detection_result),
                    result=self.cap_detection_result
                )
                
            # Save CRAFT results
            for i, result in enumerate(self.craft_results):
                craft_result = result['craft_result']
                
                # Save rotated image
                if 'rotated_image' in craft_result and craft_result['rotated_image'] is not None:
                    rotated_path = os.path.join(output_dir, f"{base_name}_cap_{i+1}_rotated_{timestamp}.jpg")
                    self.craft_detector.save_rotated_image(rotated_path, craft_result)
                    
                # Save CRAFT detection result
                craft_detection_path = os.path.join(output_dir, f"{base_name}_cap_{i+1}_craft_{timestamp}.jpg")
                self.craft_detector.save_detection_result(
                    craft_detection_path,
                    image=self.craft_detector.draw_detections(image=result['cropped_image'], result=craft_result),
                    result=craft_result
                )
                
            messagebox.showinfo("สำเร็จ", f"บันทึกผลลัพธ์เรียบร้อยแล้ว\nโฟลเดอร์: {output_dir}")
            self.log_info(f"✅ บันทึกผลลัพธ์เรียบร้อย: {output_dir}")
            
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถบันทึกผลลัพธ์: {e}")
            self.log_info(f"❌ ข้อผิดพลาดในการบันทึก: {e}")
            
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def log_info(self, message):
        """Log information to the info text widget"""
        self.info_text.insert(tk.END, f"{message}\n")
        self.info_text.see(tk.END)
        self.root.update_idletasks()
        
    def run(self):
        """Run the application"""
        if GUI_AVAILABLE:
            self.root.mainloop()
        else:
            print("❌ GUI not available - cannot run interactive mode")

def main():
    """Main function"""
    try:
        app = CapCraftProcessor()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถเริ่มแอปพลิเคชัน: {e}")

if __name__ == "__main__":
    main()

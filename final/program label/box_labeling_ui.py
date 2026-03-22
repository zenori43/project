
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
from PIL import Image, ImageTk, ImageDraw
import glob
from pathlib import Path
import cv2
import numpy as np
import threading
from datetime import datetime

# Import deep OCR
try:
    from deep_ocr import DeepOCRModel, initialize_ocr_model
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Deep OCR not available: {e}")
    OCR_AVAILABLE = False

class BoxLabelingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Box Labeling Tool")
        self.root.geometry("1600x1000")
        
        # Make window resizable
        self.root.resizable(True, True)
        
        # Data storage
        self.current_folder = ""
        self.image_files = []
        self.current_index = 0
        self.ocr_results_data = {}  # {image_path: {"file_name": str, "ocr_results": list}}
        self.presets = []  # List of previous text labels for quick selection

        
        # Image zoom control
        self.zoom_factor = 1.0
        self.base_zoom_factor = 1.0
        self.original_image = None
        self.display_image = None
        self.original_size = None  # Store original image size
        self.display_size = None   # Store display image size
        self.image_offset_x = 0    # Image offset in canvas
        self.image_offset_y = 0
        
        # Load existing data if available
        self.ocr_results_file = "ocr_results.json"
        self.presets_file = "box_presets.json"
        self.ocr_config_file = "ocr_config.json"
        self.load_data()
        
        # OCR model
        self.ocr_model = None
        self.ocr_model_path = "best_accuracy.pth"
        self.auto_label_enabled = False
        
        self.setup_ui()
        
    def load_data(self):
        """Load existing OCR results and presets from files"""
        try:
            if os.path.exists(self.ocr_results_file):
                with open(self.ocr_results_file, 'r', encoding='utf-8') as f:
                    self.ocr_results_data = json.load(f)
        except Exception as e:
            print(f"Error loading OCR results: {e}")
            
        try:
            if os.path.exists(self.presets_file):
                with open(self.presets_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.presets = data.get('text_presets', [])
        except Exception as e:
            print(f"Error loading presets: {e}")
        
        # Load OCR configuration
        try:
            if os.path.exists(self.ocr_config_file):
                with open(self.ocr_config_file, 'r', encoding='utf-8') as f:
                    ocr_config = json.load(f)
                    saved_model_path = ocr_config.get('model_path', '')
                    if saved_model_path and os.path.exists(saved_model_path):
                        self.ocr_model_path = saved_model_path
        except Exception as e:
            print(f"Error loading OCR config: {e}")
    
    def save_data(self):
        """Save OCR results and presets to files"""
        try:
            with open(self.ocr_results_file, 'w', encoding='utf-8') as f:
                json.dump(self.ocr_results_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving OCR results: {e}")
            
        try:
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                data = {
                    'text_presets': self.presets
                }
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving presets: {e}")
        
        # Save OCR configuration
        try:
            with open(self.ocr_config_file, 'w', encoding='utf-8') as f:
                ocr_config = {
                    'model_path': self.ocr_model_path
                }
                json.dump(ocr_config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving OCR config: {e}")
    
    def setup_ui(self):
        """Setup the main UI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Folder selection
        folder_frame = ttk.LabelFrame(main_frame, text="โฟลเดอร์", padding="5")
        folder_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=50)
        folder_entry.grid(row=0, column=0, padx=(0, 5))
        
        browse_btn = ttk.Button(folder_frame, text="เลือกโฟลเดอร์", command=self.browse_folder)
        browse_btn.grid(row=0, column=1)
        
        # Navigation frame
        nav_frame = ttk.LabelFrame(main_frame, text="นำทาง", padding="5")
        nav_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.prev_btn = ttk.Button(nav_frame, text="← ก่อนหน้า", command=self.previous_image)
        self.prev_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.next_btn = ttk.Button(nav_frame, text="ถัดไป →", command=self.next_image)
        self.next_btn.grid(row=0, column=1, padx=(0, 5))
        
        self.status_label = ttk.Label(nav_frame, text="0 / 0")
        self.status_label.grid(row=0, column=2, padx=10)
        
        # Image size control
        ttk.Label(nav_frame, text="ขนาดภาพ:").grid(row=0, column=3, padx=(20, 5))
        
        self.zoom_in_btn = ttk.Button(nav_frame, text="ขยาย +", command=self.zoom_in)
        self.zoom_in_btn.grid(row=0, column=4, padx=(0, 5))
        
        self.zoom_out_btn = ttk.Button(nav_frame, text="ย่อ -", command=self.zoom_out)
        self.zoom_out_btn.grid(row=0, column=5, padx=(0, 5))
        
        self.reset_zoom_btn = ttk.Button(nav_frame, text="ขนาดปกติ", command=self.reset_zoom)
        self.reset_zoom_btn.grid(row=0, column=6, padx=(0, 5))

        ttk.Label(nav_frame, text="หมุน:").grid(row=0, column=7, padx=(12, 4))
        self.rotate_180_btn = ttk.Button(nav_frame, text="180°", width=6, command=lambda: self.rotate_current_image(180))
        self.rotate_180_btn.grid(row=0, column=8, padx=(0, 4))
        self.rotate_m180_btn = ttk.Button(nav_frame, text="−180°", width=6, command=lambda: self.rotate_current_image(-180))
        self.rotate_m180_btn.grid(row=0, column=9, padx=(0, 5))
        
        self.zoom_label = ttk.Label(nav_frame, text="100%")
        self.zoom_label.grid(row=0, column=10, padx=(5, 0))
        
        # Label statistics
        self.label_stats_label = ttk.Label(nav_frame, text="", font=('Arial', 8))
        self.label_stats_label.grid(row=0, column=11, padx=(10, 0))
        
        # OCR instructions
        instructions_label = ttk.Label(nav_frame, text="(คลิกปุ่ม OCR เพื่ออ่านข้อความจากรูปภาพ)", font=('Arial', 8))
        instructions_label.grid(row=0, column=12, padx=(10, 0))
        
        # OCR controls
        ocr_frame = ttk.LabelFrame(nav_frame, text="OCR Auto-Label", padding="5")
        ocr_frame.grid(row=1, column=0, columnspan=13, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # OCR model path
        ttk.Label(ocr_frame, text="OCR Model:").grid(row=0, column=0, padx=(0, 5))
        self.ocr_path_var = tk.StringVar(value=self.ocr_model_path)
        ocr_path_entry = ttk.Entry(ocr_frame, textvariable=self.ocr_path_var, width=30)
        ocr_path_entry.grid(row=0, column=1, padx=(0, 5))
        
        # Bind to update status when path changes
        self.ocr_path_var.trace('w', lambda *args: self.update_model_status())
        
        browse_ocr_btn = ttk.Button(ocr_frame, text="Browse", command=self.browse_ocr_model)
        browse_ocr_btn.grid(row=0, column=2, padx=(0, 5))
        
        self.load_ocr_btn = ttk.Button(ocr_frame, text="Load OCR Model", command=self.load_ocr_model)
        self.load_ocr_btn.grid(row=0, column=3, padx=(0, 5))
        
        reset_ocr_btn = ttk.Button(ocr_frame, text="Reset", command=self.reset_ocr_path)
        reset_ocr_btn.grid(row=0, column=4, padx=(0, 5))
        
        self.ocr_status_label = ttk.Label(ocr_frame, text="OCR: Not Loaded", font=('Arial', 8))
        self.ocr_status_label.grid(row=0, column=5, padx=(5, 0))
        
        # Model file status
        self.model_status_label = ttk.Label(ocr_frame, text="", font=('Arial', 8))
        self.model_status_label.grid(row=0, column=6, padx=(5, 0))
        
        # OCR controls
        self.ocr_image_btn = ttk.Button(ocr_frame, text="OCR Full Image", command=self.ocr_full_image)
        self.ocr_image_btn.grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        
        self.ocr_all_btn = ttk.Button(ocr_frame, text="OCR All Images", command=self.ocr_all_images)
        self.ocr_all_btn.grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        
        self.ocr_status_btn = ttk.Button(ocr_frame, text="OCR Status", command=self.show_ocr_status)
        self.ocr_status_btn.grid(row=1, column=2, padx=(0, 5), pady=(5, 0))
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="ภาพ", padding="5")
        image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 10))
        
        # Configure image frame to expand
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Create a canvas for image display
        self.image_canvas = tk.Canvas(image_frame, bg='white', relief='sunken', bd=1)
        self.image_canvas.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.image_label = ttk.Label(image_frame, text="เลือกโฟลเดอร์เพื่อเริ่มต้น")
        self.image_label.grid(row=1, column=0, padx=10, pady=5)
        
        # Labeling frame
        label_frame = ttk.LabelFrame(main_frame, text="การ Label", padding="5")
        label_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Label Editing Frame (moved to top)
        edit_frame = ttk.LabelFrame(label_frame, text="แก้ไข Label", padding="5")
        edit_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(edit_frame, text="ข้อความ:").grid(row=0, column=0, padx=(0, 5))
        
        # Create entry for editing label
        self.label_var = tk.StringVar()
        self.label_entry = ttk.Entry(edit_frame, textvariable=self.label_var, width=50)
        self.label_entry.grid(row=0, column=1, padx=(0, 5))
        
        # Bind focus events to handle keyboard shortcuts properly
        # These are now handled by the visual feedback bindings above
        
        # Apply edited label button
        apply_label_btn = ttk.Button(edit_frame, text="ใช้ Label นี้", command=self.apply_edited_label)
        apply_label_btn.grid(row=0, column=2, padx=(0, 5))
        
        # Clear label entry button
        clear_label_btn = ttk.Button(edit_frame, text="ล้าง", command=self.clear_label_entry)
        clear_label_btn.grid(row=0, column=3, padx=(0, 5))
        
        # Copy from OCR button
        copy_from_ocr_btn = ttk.Button(edit_frame, text="คัดลอกจาก OCR", command=self.copy_from_ocr)
        copy_from_ocr_btn.grid(row=0, column=4, padx=(0, 5))
        
        # Label status indicator
        self.label_status_label = ttk.Label(edit_frame, text="", font=('Arial', 9))
        self.label_status_label.grid(row=0, column=5, padx=(5, 0))
        
        # OCR Results
        results_frame = ttk.Frame(label_frame)
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(results_frame, text="OCR Results:").grid(row=0, column=0, padx=(0, 5))
        
        # Create text widget for OCR results
        self.ocr_results_text = tk.Text(results_frame, height=4, width=60, wrap=tk.WORD)
        self.ocr_results_text.grid(row=0, column=1, padx=(0, 5))
        
        # Scrollbar for text widget
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.ocr_results_text.yview)
        scrollbar.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.ocr_results_text.configure(yscrollcommand=scrollbar.set)
        
        # Buttons
        save_results_btn = ttk.Button(results_frame, text="Save Results", command=self.save_ocr_results)
        save_results_btn.grid(row=0, column=3, padx=(0, 5))
        
        clear_results_btn = ttk.Button(results_frame, text="Clear Results", command=self.clear_ocr_results)
        clear_results_btn.grid(row=0, column=4, padx=(0, 5))
        
        # Add to Presets button
        add_preset_btn = ttk.Button(results_frame, text="Add to Presets", command=self.add_to_presets)
        add_preset_btn.grid(row=0, column=5, padx=(0, 5))
        
        # Clear Presets button
        clear_presets_btn = ttk.Button(results_frame, text="Clear Presets", command=self.clear_presets)
        clear_presets_btn.grid(row=0, column=6, padx=(0, 5))
        
        # Presets frame
        presets_frame = ttk.LabelFrame(label_frame, text="Presets ข้อความ (คลิกเพื่อใช้)", padding="5")
        presets_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.presets_frame = ttk.Frame(presets_frame)
        self.presets_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Keyboard shortcuts
        def handle_key(event):
            # Check if the focus is on the label entry widget or any text widget
            focused_widget = self.root.focus_get()
            
            # If focus is on label entry or OCR results text, allow normal text editing behavior
            if focused_widget in [self.label_entry, self.ocr_results_text]:
                # Allow all normal text editing keys (arrows, backspace, etc.)
                return
            
            # Otherwise, handle navigation shortcuts
            if event.keysym == 'plus':
                self.zoom_in()
            elif event.keysym == 'minus':
                self.zoom_out()
            elif event.keysym in ['r', 'R']:
                self.reset_zoom()
            elif event.keysym == 'o':
                self.ocr_full_image()
            elif event.keysym == 'a':
                self.ocr_all_images()
            elif event.keysym == 'c':
                self.copy_from_ocr()
            elif event.keysym in ['e', 'E']:
                self.toggle_edit_mode()
            elif event.keysym == 'Return':
                # Enter key behavior depends on focus
                if focused_widget == self.label_entry:
                    self.apply_edited_label()
                else:
                    # If not in edit mode, enter edit mode
                    self.toggle_edit_mode()
        
        # Bind keyboard events to root window
        self.root.bind('<Key>', handle_key)
        
        # Additional binding for specific key combinations to avoid conflicts
        self.root.bind('<Control-Left>', lambda e: self.previous_image())
        self.root.bind('<Control-Right>', lambda e: self.next_image())
        
        # Alternative navigation keys that work even when editing text
        self.root.bind('<Alt-Left>', lambda e: self.previous_image())
        self.root.bind('<Alt-Right>', lambda e: self.next_image())
        
        # Text widgets handle arrow keys automatically, so we don't need special bindings
        
        # Bind specific arrow key events to avoid conflicts with text editing
        # These will only trigger when not in text widgets
        self.root.bind('<Left>', self.handle_left_key)
        self.root.bind('<Right>', self.handle_right_key)
        
        # Visual feedback is now handled in the focus event handlers below
        
        # Add status indicator for text editing mode
        def on_focus_in(e):
            self.label_entry.config(relief='solid', borderwidth=2)
            self.label_status_label.config(text="✏️ แก้ไขข้อความ (ลูกศรเลื่อนเคอร์เซอร์, Enter=ใช้ label, E=ออก)", foreground='blue')
            print("🔍 Label field focused - Arrow keys will move cursor, Enter to apply and next, E to exit")
        
        def on_focus_out(e):
            self.label_entry.config(relief='flat', borderwidth=1)
            self.update_label_status()
            print("🔍 Label field unfocused - Arrow keys will navigate images, E to enter edit mode")
        
        self.label_entry.bind('<FocusIn>', on_focus_in)
        self.label_entry.bind('<FocusOut>', on_focus_out)
        
        # Add Enter key binding for label entry
        self.label_entry.bind('<Return>', lambda e: self.apply_edited_label())
        
        # Add Escape key to exit edit mode
        self.label_entry.bind('<Escape>', lambda e: self.root.focus_set())
        
        # Add E key binding to label entry for quick exit
        self.label_entry.bind('<Key>', lambda e: self.handle_label_entry_key(e))
        
        # Add instructions for keyboard shortcuts
        instructions_text = """
Keyboard Shortcuts:
- Left/Right Arrow: Navigate images (when not editing text)
- Ctrl+Left/Ctrl+Right: Navigate images (always)
- Alt+Left/Alt+Right: Navigate images (always)
- +/-: Zoom in/out
- R: Reset zoom
- O: OCR current image
- A: OCR all images
- C: Copy from OCR
- E: Toggle edit mode (enter/exit label editing)
- Enter: Apply label and go to next image (when editing) / Enter edit mode (when not editing)

Text Editing:
- Press E to enter edit mode, E again to exit
- When editing, use arrow keys to move cursor
- Use Ctrl+Left/Ctrl+Right to navigate images while editing text
- Press Enter to apply label and go to next image
- Press Escape to exit edit mode
- The label field will show visual feedback when focused
        """
        print(instructions_text)
        
        # Presets frame
        presets_frame = ttk.LabelFrame(label_frame, text="Presets ข้อความ (คลิกเพื่อใช้)", padding="5")
        presets_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.presets_frame = ttk.Frame(presets_frame)
        self.presets_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))        

        
        # Export frame
        export_frame = ttk.Frame(main_frame)
        export_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        export_btn = ttk.Button(export_frame, text="Export JSON", command=self.export_json)
        export_btn.grid(row=0, column=0, padx=(0, 5))
        
        export_gt_btn = ttk.Button(export_frame, text="Export gt.txt (ล่าสุด)", command=self.export_gt_txt)
        export_gt_btn.grid(row=0, column=1, padx=(0, 5))
        
        export_gt_all_btn = ttk.Button(export_frame, text="Export gt.txt (ทั้งหมด)", command=self.export_gt_txt_all)
        export_gt_all_btn.grid(row=0, column=2, padx=(0, 5))
        
        clear_all_btn = ttk.Button(export_frame, text="ล้าง Labels ทั้งหมด", command=self.clear_all_labels)
        clear_all_btn.grid(row=0, column=3, padx=(0, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(export_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Presets frame
        presets_frame = ttk.LabelFrame(main_frame, text="Presets ข้อความ (คลิกเพื่อใช้)", padding="5")
        presets_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.presets_frame = ttk.Frame(presets_frame)
        self.presets_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Update UI
        self.update_ui()
        
        # Try to load OCR model if available
        if OCR_AVAILABLE:
            self.try_load_default_ocr_model()
    
    def browse_folder(self):
        """Browse for image folder"""
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ที่มีภาพ")
        if folder:
            self.current_folder = folder
            self.folder_var.set(folder)
            # Reset zoom factor when loading new folder
            self.zoom_factor = 1.0
            self.load_images()
            self.update_ui()
    
    def browse_ocr_model(self):
        """Browse for OCR model file"""
        model_file = filedialog.askopenfilename(
            title="เลือกไฟล์ OCR Model",
            filetypes=[
                ("PyTorch model files", "*.pth"),
                ("All files", "*.*")
            ]
        )
        if model_file:
            self.ocr_path_var.set(model_file)
            # Update the default model path
            self.ocr_model_path = model_file
            # Save the configuration
            self.save_data()
            # Update model status
            self.update_model_status()
    
    def reset_ocr_path(self):
        """Reset OCR model path to default"""
        default_path = "best_accuracy.pth"
        self.ocr_path_var.set(default_path)
        self.ocr_model_path = default_path
        # Save the configuration
        self.save_data()
        # Update model status
        self.update_model_status()
    
    def load_images(self):
        """Load all image files from the selected folder"""
        if not self.current_folder:
            return
            
        # Supported image extensions
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
        self.image_files = []
        
        # Use pathlib for better file handling
        folder_path = Path(self.current_folder)
        
        # Get all files in the folder
        all_files = list(folder_path.iterdir())
        
        # Filter for image files
        for file_path in all_files:
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                if file_ext in extensions:
                    self.image_files.append(str(file_path))
        
        # Sort files
        self.image_files.sort()
        self.current_index = 0
        
        print(f"Found {len(self.image_files)} images in {self.current_folder}")
        
        # Update status
        self.update_status()
    
    def update_status(self):
        """Update navigation status"""
        if self.image_files:
            self.status_label.config(text=f"{self.current_index + 1} / {len(self.image_files)}")
            self.progress_var.set((self.current_index + 1) / len(self.image_files) * 100)
        else:
            self.status_label.config(text="0 / 0")
            self.progress_var.set(0)
    
    def previous_image(self):
        """Go to previous image"""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            # Reset zoom factor when changing images
            self.zoom_factor = 1.0
            self.update_ui()
    
    def next_image(self):
        """Go to next image"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            # Reset zoom factor when changing images
            self.zoom_factor = 1.0
            self.update_ui()
    

    
    def update_ui(self):
        """Update the UI with current image and labels"""
        if not self.image_files:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(
                self.image_canvas.winfo_width()//2, 
                self.image_canvas.winfo_height()//2,
                text="เลือกโฟลเดอร์เพื่อเริ่มต้น",
                font=('Arial', 14),
                fill='gray'
            )
            self.rotate_180_btn.config(state="disabled")
            self.rotate_m180_btn.config(state="disabled")
            return
        
        current_image_path = self.image_files[self.current_index]
        
        # Load and display image
        try:
            self.original_image = Image.open(current_image_path)
            
            # Get original image size
            original_width, original_height = self.original_image.size
            self.original_size = (original_width, original_height)
            
            # Calculate display size while maintaining aspect ratio
            max_width = 800
            max_height = 600
            
            # Calculate scaling factor to fit within max dimensions
            scale_x = max_width / original_width
            scale_y = max_height / original_height
            scale = min(scale_x, scale_y)  # Use smaller scale to fit both dimensions
            
            # Calculate new dimensions
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Apply zoom factor
            new_width = int(new_width * self.zoom_factor)
            new_height = int(new_height * self.zoom_factor)
            self.display_size = (new_width, new_height)
            
            # Resize image
            self.display_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(self.display_image)
            
            # Clear canvas and display image
            self.image_canvas.delete("all")
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Center the image in canvas
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2
                self.image_offset_x = x
                self.image_offset_y = y
                self.image_canvas.create_image(x, y, anchor='nw', image=photo)
                self.image_canvas.image = photo  # Keep a reference
            else:
                # Fallback to label if canvas not ready
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
            
        except Exception as e:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(
                self.image_canvas.winfo_width()//2, 
                self.image_canvas.winfo_height()//2,
                text=f"ไม่สามารถโหลดภาพได้: {e}",
                font=('Arial', 12),
                fill='red'
            )
        
        # Load existing OCR results for current image
        self.load_image_ocr_results(current_image_path)
        
        # Update navigation buttons
        self.prev_btn.config(state='normal' if self.current_index > 0 else 'disabled')
        self.next_btn.config(state='normal' if self.current_index < len(self.image_files) - 1 else 'disabled')
        rot_state = 'normal' if self.image_files else 'disabled'
        self.rotate_180_btn.config(state=rot_state)
        self.rotate_m180_btn.config(state=rot_state)
        
        # Update status
        self.update_status()
        
        # Update zoom label
        zoom_percentage = int(self.zoom_factor * 100)
        self.zoom_label.config(text=f"{zoom_percentage}%")
        
        # Update image info
        if self.image_files:
            current_image_path = self.image_files[self.current_index]
            filename = os.path.basename(current_image_path)
            self.image_label.config(text=f"ไฟล์: {filename}")
        
        # Update OCR statistics
        if self.image_files:
            ocr_count = sum(1 for img in self.image_files if img in self.ocr_results_data)
            total_count = len(self.image_files)
            
            if current_image_path in self.ocr_results_data:
                results_count = len(self.ocr_results_data[current_image_path].get("ocr_results", []))
                self.label_stats_label.config(text=f"Labeled: {ocr_count}/{total_count} | ✓ {results_count} results", foreground='green')
            else:
                self.label_stats_label.config(text=f"Labeled: {ocr_count}/{total_count} | ✗ ยังไม่ได้ Label", foreground='red')
        

        
        # Update model file status
        self.update_model_status()
        
        # Update presets
        self.update_presets()
        
        # Reset label status if no OCR results
        if self.image_files:
            current_image_path = self.image_files[self.current_index]
            if current_image_path not in self.ocr_results_data:
                self.label_status_label.config(text="✗ ยังไม่ได้ Label", foreground='red')
    
    def display_to_original_coords(self, x1, y1, x2, y2):
        """Convert display coordinates to original image coordinates"""
        if not self.original_size or not self.display_size:
            return [x1, y1, x2, y2]
        
        # Remove image offset
        x1 = x1 - self.image_offset_x
        y1 = y1 - self.image_offset_y
        x2 = x2 - self.image_offset_x
        y2 = y2 - self.image_offset_y
        
        # Convert display coordinates to original coordinates
        orig_w, orig_h = self.original_size
        disp_w, disp_h = self.display_size
        
        # Scale coordinates
        orig_x1 = int(x1 * orig_w / disp_w)
        orig_y1 = int(y1 * orig_h / disp_h)
        orig_x2 = int(x2 * orig_w / disp_w)
        orig_y2 = int(y2 * orig_h / disp_h)
        
        return [orig_x1, orig_y1, orig_x2, orig_y2]
    
    def original_to_display_coords(self, x1, y1, x2, y2):
        """Convert original image coordinates to display coordinates"""
        if not self.original_size or not self.display_size:
            return [x1, y1, x2, y2]
        
        # Convert original coordinates to display coordinates
        orig_w, orig_h = self.original_size
        disp_w, disp_h = self.display_size
        
        # Scale coordinates
        disp_x1 = int(x1 * disp_w / orig_w)
        disp_y1 = int(y1 * disp_h / orig_h)
        disp_x2 = int(x2 * disp_w / orig_w)
        disp_y2 = int(y2 * disp_h / orig_h)
        
        # Add image offset
        disp_x1 = disp_x1 + self.image_offset_x
        disp_y1 = disp_y1 + self.image_offset_y
        disp_x2 = disp_x2 + self.image_offset_x
        disp_y2 = disp_y2 + self.image_offset_y
        
        return [disp_x1, disp_y1, disp_x2, disp_y2]
    
    def load_image_ocr_results(self, image_path):
        """Load existing OCR results for current image"""
        if image_path in self.ocr_results_data:
            results = self.ocr_results_data[image_path].get("ocr_results", [])
            if results:
                # Display latest result in text widget
                latest_result = results[-1]
                result_text = f"Image: {os.path.basename(image_path)}\n"
                result_text += f"Text: {latest_result.get('text', '')}\n"
                result_text += f"Confidence: {latest_result.get('confidence', 0.0):.3f}\n"
                result_text += "-" * 50 + "\n"
                
                self.ocr_results_text.delete(1.0, tk.END)
                self.ocr_results_text.insert(tk.END, result_text)
                
                # Also load text into label entry for editing
                self.label_var.set(latest_result.get('text', ''))
                
                # Update label status
                self.label_status_label.config(text="✓ Label แล้ว", foreground='green')
            else:
                # Clear both OCR results and label entry
                self.ocr_results_text.delete(1.0, tk.END)
                self.label_var.set("")
                self.label_status_label.config(text="✗ ยังไม่ได้ Label", foreground='red')
        else:
            # Clear both OCR results and label entry
            self.ocr_results_text.delete(1.0, tk.END)
            self.label_var.set("")
            self.label_status_label.config(text="✗ ยังไม่ได้ Label", foreground='red')
    

    

    
    def update_model_status(self):
        """Update model file status"""
        model_path = self.ocr_path_var.get()
        if model_path:
            if os.path.exists(model_path):
                filename = os.path.basename(model_path)
                self.model_status_label.config(text=f"✓ {filename}", foreground='green')
            else:
                self.model_status_label.config(text="✗ File not found", foreground='red')
        else:
            self.model_status_label.config(text="", foreground='black')
    
    def update_presets(self):
        """Update the text presets buttons"""
        # Clear existing preset buttons
        for widget in self.presets_frame.winfo_children():
            widget.destroy()
        
        # Create preset buttons
        for i, preset in enumerate(self.presets):
            # Create frame for each preset
            preset_frame = ttk.Frame(self.presets_frame)
            preset_frame.grid(row=i//3, column=i%3, padx=2, pady=2)
            
            # Preset button
            btn = ttk.Button(preset_frame, text=preset[:20] + "..." if len(preset) > 20 else preset, 
                           command=lambda p=preset: self.use_preset(p))
            btn.grid(row=0, column=0, padx=(0, 2))
            
            # Delete button for this preset
            del_btn = ttk.Button(preset_frame, text="X", width=3,
                               command=lambda p=preset: self.delete_preset(p))
            del_btn.grid(row=0, column=1)
    
    def use_preset(self, preset):
        """Use a preset text"""
        # Clear current OCR results and insert preset text
        self.ocr_results_text.delete(1.0, tk.END)
        
        # Create a formatted result with preset text
        if self.image_files:
            current_image_path = self.image_files[self.current_index]
            result_text = f"Image: {os.path.basename(current_image_path)}\n"
            result_text += f"Text: {preset}\n"
            result_text += f"Confidence: 1.000 (Preset)\n"
            result_text += "-" * 50 + "\n"
            
            self.ocr_results_text.insert(tk.END, result_text)
            
            # Also set text in label entry for editing
            self.label_var.set(preset)
            
            # Update label status
            self.label_status_label.config(text="✓ Label แล้ว", foreground='green')
            
            # Save this as an OCR result
            self.save_ocr_result(current_image_path, preset, 1.0)
    
    def add_to_presets(self):
        """Add current OCR result to presets"""
        # Get current OCR result text
        current_text = self.ocr_results_text.get(1.0, tk.END).strip()
        if current_text:
            # Extract text from OCR result (remove metadata)
            lines = current_text.split('\n')
            text_line = None
            for line in lines:
                if line.startswith('Text: '):
                    text_line = line.replace('Text: ', '').strip()
                    break
            
            if text_line and text_line not in self.presets:
                self.presets.append(text_line)
                self.update_presets()
                self.save_data()
                messagebox.showinfo("Success", f"Added '{text_line}' to presets")
            elif text_line in self.presets:
                messagebox.showinfo("Info", "This text is already in presets")
            else:
                messagebox.showwarning("Warning", "No OCR text found to add to presets")
        else:
            messagebox.showwarning("Warning", "No OCR results to add to presets")
    
    def clear_presets(self):
        """Clear all presets"""
        if messagebox.askyesno("Confirm", "Do you want to clear all presets?"):
            self.presets.clear()
            self.update_presets()
            self.save_data()
            messagebox.showinfo("Success", "All presets cleared")
    
    def delete_preset(self, preset):
        """Delete a specific preset"""
        if messagebox.askyesno("Confirm", f"Do you want to delete preset '{preset}'?"):
            self.presets.remove(preset)
            self.update_presets()
            self.save_data()
            messagebox.showinfo("Success", f"Preset '{preset}' deleted")
    

    
    def export_json(self):
        """Export OCR results to JSON file"""
        if not self.ocr_results_data:
            messagebox.showinfo("ข้อมูล", "ไม่มี OCR Results ที่จะ Export")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="บันทึกไฟล์ OCR Results JSON"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.ocr_results_data, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("สำเร็จ", f"Export OCR Results เรียบร้อยแล้ว\nไฟล์: {filename}")
                
            except Exception as e:
                messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถ Export ได้: {e}")
    
    def export_gt_txt(self):
        """Export OCR results to gt.txt format for deep-text-recognition-benchmark"""
        if not self.ocr_results_data:
            messagebox.showinfo("ข้อมูล", "ไม่มี OCR Results ที่จะ Export")
            return
        
        # Ask for save location
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์สำหรับสร้างโครงสร้าง data")
        
        if folder:
            try:
                # Create data folder structure
                data_folder = os.path.join(folder, "data")
                test_folder = os.path.join(data_folder, "test")
                
                # Create directories
                os.makedirs(data_folder, exist_ok=True)
                os.makedirs(test_folder, exist_ok=True)
                
                # Copy images to test folder
                copied_images = []
                for image_path in self.ocr_results_data.keys():
                    if os.path.exists(image_path):
                        filename = os.path.basename(image_path)
                        dest_path = os.path.join(test_folder, filename)
                        
                        # Copy image file
                        import shutil
                        shutil.copy2(image_path, dest_path)
                        copied_images.append(filename)
                
                # Create gt.txt file - ใช้ข้อมูลล่าสุดที่แก้ไขแล้ว
                gt_file_path = os.path.join(data_folder, "gt.txt")
                exported_count = 0
                
                with open(gt_file_path, 'w', encoding='utf-8') as f:
                    for image_path, result_data in self.ocr_results_data.items():
                        filename = os.path.basename(image_path)
                        
                        # ใช้ข้อมูลล่าสุดจาก OCR results (ข้อมูลที่แก้ไขแล้ว)
                        results = result_data.get("ocr_results", [])
                        if results:
                            # ใช้ข้อมูลล่าสุด (รายการสุดท้าย) ที่แก้ไขแล้ว
                            latest_result = results[-1]
                            text = latest_result.get("text", "").strip()
                            if text:  # Only include non-empty results
                                # Format: test/filename.png\ttext\n
                                f.write(f"test/{filename}\t{text}\n")
                                exported_count += 1
                
                # Show success message with structure info
                structure_info = f"""โครงสร้างไฟล์ที่สร้าง:
{data_folder}/
├── gt.txt ({exported_count} entries)
└── test/
    ├── {filename if copied_images else "..."}
    └── ... ({len(copied_images)} ไฟล์)

รูปแบบ gt.txt:
test/filename.png\ttext (ข้อมูลล่าสุดที่แก้ไขแล้ว)

หมายเหตุ: Export ข้อมูลล่าสุดที่แก้ไขแล้ว ไม่ใช่ข้อมูลเดิมจาก OCR"""
                
                messagebox.showinfo("สำเร็จ", f"Export gt.txt เรียบร้อยแล้ว\n\n{structure_info}")
                
            except Exception as e:
                messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถ Export ได้: {e}")
    
    def export_gt_txt_all(self):
        """Export all OCR results to gt.txt format (including all versions)"""
        if not self.ocr_results_data:
            messagebox.showinfo("ข้อมูล", "ไม่มี OCR Results ที่จะ Export")
            return
        
        # Ask for save location
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์สำหรับสร้างโครงสร้าง data")
        
        if folder:
            try:
                # Create data folder structure
                data_folder = os.path.join(folder, "data")
                test_folder = os.path.join(data_folder, "test")
                
                # Create directories
                os.makedirs(data_folder, exist_ok=True)
                os.makedirs(test_folder, exist_ok=True)
                
                # Copy images to test folder
                copied_images = []
                for image_path in self.ocr_results_data.keys():
                    if os.path.exists(image_path):
                        filename = os.path.basename(image_path)
                        dest_path = os.path.join(test_folder, filename)
                        
                        # Copy image file
                        import shutil
                        shutil.copy2(image_path, dest_path)
                        copied_images.append(filename)
                
                # Create gt.txt file - ใช้ข้อมูลทั้งหมด
                gt_file_path = os.path.join(data_folder, "gt.txt")
                exported_count = 0
                
                with open(gt_file_path, 'w', encoding='utf-8') as f:
                    for image_path, result_data in self.ocr_results_data.items():
                        filename = os.path.basename(image_path)
                        
                        # ใช้ข้อมูลทั้งหมดจาก OCR results
                        results = result_data.get("ocr_results", [])
                        for result_info in results:
                            text = result_info.get("text", "").strip()
                            if text:  # Only include non-empty results
                                # Format: test/filename.png\ttext\n
                                f.write(f"test/{filename}\t{text}\n")
                                exported_count += 1
                
                # Show success message with structure info
                structure_info = f"""โครงสร้างไฟล์ที่สร้าง:
{data_folder}/
├── gt.txt ({exported_count} entries)
└── test/
    ├── {filename if copied_images else "..."}
    └── ... ({len(copied_images)} ไฟล์)

รูปแบบ gt.txt:
test/filename.png\ttext (ข้อมูลทั้งหมดจาก OCR)

หมายเหตุ: Export ข้อมูลทั้งหมดจาก OCR รวมถึงข้อมูลเดิมและที่แก้ไขแล้ว"""
                
                messagebox.showinfo("สำเร็จ", f"Export gt.txt (ทั้งหมด) เรียบร้อยแล้ว\n\n{structure_info}")
                
            except Exception as e:
                messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถ Export ได้: {e}")
    
    def zoom_in(self):
        """Zoom in the image"""
        self.zoom_factor = min(self.zoom_factor * 1.2, 3.0)
        self.update_ui()
    
    def zoom_out(self):
        """Zoom out the image"""
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.3)
        self.update_ui()
    
    def reset_zoom(self):
        """Reset zoom to normal size"""
        self.zoom_factor = 1.0
        self.update_ui()

    def rotate_current_image(self, angle_deg: float):
        """หมุนภาพปัจจุบันแล้วบันทึกทับไฟล์เดิม (+180° กับ −180° ได้ผลเหมือนกัน — กลับหัว)"""
        if not self.image_files:
            messagebox.showwarning("คำเตือน", "กรุณาเลือกโฟลเดอร์และภาพก่อน")
            return
        path = self.image_files[self.current_index]
        try:
            img = Image.open(path)
            # 180° / −180°: ใช้ transpose ให้คมชัดและคงโหมดสี
            if abs(angle_deg) == 180:
                rotated = img.transpose(Image.ROTATE_180)
            else:
                rotated = img.rotate(-angle_deg, expand=True, fillcolor="white")

            ext = os.path.splitext(path)[1].lower()
            save_kwargs = {}
            if ext in (".jpg", ".jpeg"):
                if rotated.mode in ("RGBA", "P"):
                    rotated = rotated.convert("RGB")
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True
            elif ext == ".webp":
                if rotated.mode not in ("RGB", "RGBA"):
                    rotated = rotated.convert("RGBA")

            rotated.save(path, **save_kwargs)
            self.update_ui()
        except Exception as e:
            messagebox.showerror("หมุนภาพไม่สำเร็จ", str(e))
    
    def clear_all_labels(self):
        """Clear all OCR results"""
        if messagebox.askyesno("ยืนยัน", "คุณต้องการล้าง OCR Results ทั้งหมดหรือไม่?"):
            self.ocr_results_data.clear()
            self.save_data()
            self.update_ui()
            messagebox.showinfo("สำเร็จ", "ล้าง OCR Results ทั้งหมดแล้ว")
    
    # OCR Functions
    def try_load_default_ocr_model(self):
        """Try to load default OCR model"""
        if not OCR_AVAILABLE:
            return
        
        default_path = self.ocr_model_path
        if os.path.exists(default_path):
            try:
                # Set the path in the entry field first
                self.ocr_path_var.set(default_path)
                self.load_ocr_model()
            except Exception as e:
                print(f"Failed to load default OCR model: {e}")
    
    def load_ocr_model(self):
        """Load OCR model"""
        if not OCR_AVAILABLE:
            messagebox.showerror("Error", "Deep OCR not available. Please install deep_ocr.py")
            return
        
        model_path = self.ocr_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", f"OCR model not found: {model_path}")
            return
        
        try:
            self.ocr_status_label.config(text="Loading OCR model...", foreground='orange')
            self.root.update()
            
            # Load model in a separate thread to avoid blocking UI
            def load_model_thread():
                try:
                    self.ocr_model = DeepOCRModel(model_path)
                    if self.ocr_model.model is not None:
                        self.root.after(0, lambda: self.ocr_status_label.config(
                            text="OCR: Loaded ✓", foreground='green'))
                        self.root.after(0, lambda: self.ocr_image_btn.config(state='normal'))
                        self.root.after(0, lambda: self.ocr_status_btn.config(state='normal'))
                    else:
                        self.root.after(0, lambda: self.ocr_status_label.config(
                            text="OCR: Failed to load", foreground='red'))
                except Exception as e:
                    self.root.after(0, lambda: self.ocr_status_label.config(
                        text=f"OCR: Error - {str(e)[:20]}", foreground='red'))
                    print(f"Error loading OCR model: {e}")
            
            thread = threading.Thread(target=load_model_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            self.ocr_status_label.config(text=f"OCR: Error - {str(e)[:20]}", foreground='red')
            print(f"Error loading OCR model: {e}")
    

    
    def ocr_full_image(self):
        """OCR the full current image"""
        if self.ocr_model is None:
            messagebox.showwarning("Warning", "Please load OCR model first")
            return
        
        if not self.image_files:
            messagebox.showwarning("Warning", "No image available")
            return
        
        # Get current image path
        current_image_path = self.image_files[self.current_index]
        
        # Perform OCR
        try:
            self.ocr_status_label.config(text="OCR: Processing full image...", foreground='orange')
            self.root.update()
            
            # Run OCR in separate thread
            def ocr_thread():
                try:
                    # Load image
                    image = cv2.imread(current_image_path)
                    if image is None:
                        self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load image"))
                        self.root.after(0, lambda: self.ocr_status_label.config(text="OCR: Error", foreground='red'))
                        return
                    
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Perform OCR
                    ocr_result = self.ocr_model.recognize_text_from_image_array(image_rgb)
                    
                    if 'error' in ocr_result:
                        self.root.after(0, lambda: messagebox.showerror("OCR Error", ocr_result['error']))
                        self.root.after(0, lambda: self.ocr_status_label.config(text="OCR: Error", foreground='red'))
                        return
                    
                    recognized_text = ocr_result.get('recognized_text', '').strip()
                    confidence = ocr_result.get('confidence_score', 0.0)
                    
                    if recognized_text:
                        # Display result in text widget
                        result_text = f"Image: {os.path.basename(current_image_path)}\n"
                        result_text += f"Text: {recognized_text}\n"
                        result_text += f"Confidence: {confidence:.3f}\n"
                        result_text += "-" * 50 + "\n"
                        
                        self.root.after(0, lambda: self.ocr_results_text.delete(1.0, tk.END))
                        self.root.after(0, lambda: self.ocr_results_text.insert(tk.END, result_text))
                        self.root.after(0, lambda: self.ocr_results_text.see(tk.END))
                        
                        # Also set text in label entry for editing
                        self.root.after(0, lambda: self.label_var.set(recognized_text))
                        
                        # Update label status
                        self.root.after(0, lambda: self.label_status_label.config(text="✓ Label แล้ว", foreground='green'))
                        
                        # Save result
                        self.save_ocr_result(current_image_path, recognized_text, confidence)
                        
                        # Show status
                        self.root.after(0, lambda: self.ocr_status_label.config(
                            text=f"OCR: Found text (Confidence: {confidence:.3f})", foreground='green'))
                    else:
                        self.root.after(0, lambda: self.ocr_status_label.config(text="OCR: No text found", foreground='orange'))
                        
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("OCR Error", str(e)))
                    self.root.after(0, lambda: self.ocr_status_label.config(text="OCR: Error", foreground='red'))
            
            thread = threading.Thread(target=ocr_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"OCR failed: {e}")
            self.ocr_status_label.config(text="OCR: Error", foreground='red')
    
    def show_ocr_status(self):
        """Show OCR status and model information"""
        if self.ocr_model is None:
            messagebox.showinfo("OCR Status", "OCR model is not loaded")
        else:
            model_path = self.ocr_path_var.get()
            status_text = f"OCR Model: {os.path.basename(model_path)}\n"
            status_text += f"Model Path: {model_path}\n"
            status_text += f"Status: Loaded ✓"
            messagebox.showinfo("OCR Status", status_text)
    
    def ocr_all_images(self):
        """Perform OCR on all images in the folder"""
        if not self.ocr_model:
            messagebox.showwarning("Warning", "OCR model not loaded")
            return
        
        if not self.image_files:
            messagebox.showwarning("Warning", "No images loaded")
            return
        
        # Count images that need OCR
        images_to_process = [img for img in self.image_files if img not in self.ocr_results_data]
        
        if not images_to_process:
            messagebox.showinfo("Info", "All images already have OCR results")
            return
        
        # Confirm with user
        result = messagebox.askyesno("Confirm", f"Process OCR for {len(images_to_process)} images?\nThis may take some time.")
        if not result:
            return
        
        # Disable button during processing
        self.ocr_all_btn.config(state='disabled')
        self.ocr_all_btn.config(text="Processing...")
        
        # Start processing in separate thread
        import threading
        thread = threading.Thread(target=self._process_all_images_ocr, args=(images_to_process,))
        thread.daemon = True
        thread.start()
    
    def _process_all_images_ocr(self, images_to_process):
        """Process OCR for all images in background thread"""
        try:
            total_images = len(images_to_process)
            processed = 0
            
            for i, image_path in enumerate(images_to_process):
                try:
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Failed to load image: {image_path}")
                        continue
                    
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Perform OCR
                    ocr_result = self.ocr_model.recognize_text_from_image_array(image_rgb)
                    
                    # Check for errors in OCR result
                    if 'error' in ocr_result:
                        print(f"OCR error for {image_path}: {ocr_result['error']}")
                        continue
                    
                    # Extract text and confidence from result
                    recognized_text = ocr_result.get('recognized_text', '')
                    confidence = ocr_result.get('confidence_score', 0.0)
                    
                    # Save result
                    self.save_ocr_result(image_path, recognized_text, confidence)
                    
                    processed += 1
                    
                    # Update progress in main thread
                    progress = f"Processing... {processed}/{total_images}"
                    self.root.after(0, lambda p=progress: self.ocr_all_btn.config(text=p))
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    # Continue with next image instead of stopping
                    continue
            
            # Update UI in main thread
            self.root.after(0, self._finish_all_ocr_processing, processed, total_images)
            
        except Exception as e:
            print(f"Error in batch OCR processing: {str(e)}")
            self.root.after(0, self._finish_all_ocr_processing, 0, 0, error=str(e))
    
    def _finish_all_ocr_processing(self, processed, total, error=None):
        """Finish batch OCR processing and update UI"""
        # Re-enable button
        self.ocr_all_btn.config(state='normal')
        self.ocr_all_btn.config(text="OCR All Images")
        
        if error:
            messagebox.showerror("Error", f"Error during batch OCR: {error}")
        else:
            failed = total - processed
            message = f"OCR processing completed!\n"
            message += f"Successfully processed: {processed} images\n"
            if failed > 0:
                message += f"Failed: {failed} images\n"
                message += f"Check console for error details"
            messagebox.showinfo("Complete", message)
            
            # Update current image display if it was processed
            if self.image_files and self.current_index < len(self.image_files):
                current_image_path = self.image_files[self.current_index]
                if current_image_path in self.ocr_results_data:
                    self.load_image_ocr_results(current_image_path)
            
            # Update statistics
            self.update_ui()
    
    def save_ocr_result(self, image_path, text, confidence):
        """Save OCR result to data"""
        filename = os.path.basename(image_path)
        
        if image_path not in self.ocr_results_data:
            self.ocr_results_data[image_path] = {
                "file_name": filename,
                "ocr_results": []
            }
        
        result = {
            "text": text,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        self.ocr_results_data[image_path]["ocr_results"].append(result)
        self.save_data()
    
    def save_ocr_results(self):
        """Save OCR results to file"""
        if not self.ocr_results_data:
            messagebox.showinfo("Info", "No OCR results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save OCR Results"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.ocr_results_data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("Success", f"OCR results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save OCR results: {e}")
    
    def clear_ocr_results(self):
        """Clear OCR results text widget"""
        self.ocr_results_text.delete(1.0, tk.END)
        self.label_var.set("")
        self.label_status_label.config(text="✗ ยังไม่ได้ Label", foreground='red')
    
    def apply_edited_label(self):
        """Apply the edited label from entry to OCR results"""
        edited_text = self.label_var.get().strip()
        if not edited_text:
            messagebox.showwarning("Warning", "Please enter text in the label field")
            return
        
        if not self.image_files:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        current_image_path = self.image_files[self.current_index]
        
        # Create formatted result with edited text
        result_text = f"Image: {os.path.basename(current_image_path)}\n"
        result_text += f"Text: {edited_text}\n"
        result_text += f"Confidence: 1.000 (Edited)\n"
        result_text += "-" * 50 + "\n"
        
        # Update OCR results text widget
        self.ocr_results_text.delete(1.0, tk.END)
        self.ocr_results_text.insert(tk.END, result_text)
        
        # Save the edited result
        self.save_ocr_result(current_image_path, edited_text, 1.0)
        
        # Update label status
        self.label_status_label.config(text="✓ Label แล้ว", foreground='green')
        
        # Clear the entry
        self.label_var.set("")
        
        # Auto-advance to next image if available
        if self.current_index < len(self.image_files) - 1:
            self.next_image()
        else:
            messagebox.showinfo("Complete", "All images have been labeled!")
    
    def clear_label_entry(self):
        """Clear the label entry field"""
        self.label_var.set("")
    
    def copy_from_ocr(self):
        """Copy text from OCR results to label entry"""
        current_text = self.ocr_results_text.get(1.0, tk.END).strip()
        if current_text:
            # Extract text from OCR result
            lines = current_text.split('\n')
            text_line = None
            for line in lines:
                if line.startswith('Text: '):
                    text_line = line.replace('Text: ', '').strip()
                    break
            
            if text_line:
                self.label_var.set(text_line)
                messagebox.showinfo("Success", f"Copied text: '{text_line}'")
            else:
                messagebox.showwarning("Warning", "No text found in OCR results")
        else:
            messagebox.showwarning("Warning", "No OCR results to copy from")
    
    def on_label_entry_focus_in(self):
        """Handle when label entry gets focus"""
        # This method is kept for compatibility but not used
        pass
    
    def on_label_entry_focus_out(self):
        """Handle when label entry loses focus"""
        # This method is kept for compatibility but not used
        pass
    
    def handle_left_key(self, event):
        """Handle left arrow key press"""
        focused_widget = self.root.focus_get()
        if focused_widget in [self.label_entry, self.ocr_results_text]:
            # Let the widget handle the key normally for text editing
            return
        # Otherwise, navigate to previous image
        self.previous_image()
    
    def handle_right_key(self, event):
        """Handle right arrow key press"""
        focused_widget = self.root.focus_get()
        if focused_widget in [self.label_entry, self.ocr_results_text]:
            # Let the widget handle the key normally for text editing
            return
        # Otherwise, navigate to next image
        self.next_image()
    
    def update_label_status(self):
        """Update label status based on current state"""
        if not self.image_files:
            return
        
        current_image_path = self.image_files[self.current_index]
        if current_image_path in self.ocr_results_data:
            results_count = len(self.ocr_results_data[current_image_path].get("ocr_results", []))
            if results_count > 0:
                self.label_status_label.config(text="✓ Label แล้ว", foreground='green')
            else:
                self.label_status_label.config(text="✗ ยังไม่ได้ Label", foreground='red')
        else:
            self.label_status_label.config(text="✗ ยังไม่ได้ Label", foreground='red')
    
    def toggle_edit_mode(self):
        """Toggle edit mode - focus/unfocus label entry"""
        focused_widget = self.root.focus_get()
        
        if focused_widget == self.label_entry:
            # Currently in edit mode, exit edit mode
            self.root.focus_set()  # Remove focus from entry
            print("🔍 Exited edit mode - Arrow keys will navigate images, E to enter edit mode")
        else:
            # Not in edit mode, enter edit mode
            self.label_entry.focus_set()
            # Select all text if there's any
            if self.label_var.get():
                self.label_entry.select_range(0, tk.END)
            print("🔍 Entered edit mode - Arrow keys will move cursor, Enter to apply and next, E to exit")
    
    def handle_label_entry_key(self, event):
        """Handle key events in label entry"""
        if event.keysym in ['e', 'E']:
            # Exit edit mode when E is pressed
            self.root.focus_set()
            return "break"  # Prevent the E from being typed
        # Let other keys work normally
        return None

def main():
    root = tk.Tk()
    app = BoxLabelingUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

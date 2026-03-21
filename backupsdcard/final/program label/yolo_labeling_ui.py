# -*- coding: utf-8 -*-
"""
YOLO Labeling Tool with Auto Detection
โปรแกรมสำหรับ labeling ภาพด้วย YOLO model พร้อม auto detect และแก้ไขได้
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
import torch
import glob
import json
from typing import List, Dict, Tuple, Optional

# YOLO class names
CLASS_NAMES = {
    0: "barcode",
    1: "bottle",
    2: "angle1",
    3: "type",
    4: "nutri",
    5: "angle2",
    6: "angle3",
    7: "angle4"
}

# Colors for each class
CLASS_COLORS = {
    0: "#FF0000",  # barcode - Red
    1: "#00FF00",  # bottle - Green
    2: "#0000FF",  # angle1 - Blue
    3: "#FFFF00",  # type - Yellow
    4: "#FF00FF",  # nutri - Magenta
    5: "#00FFFF",  # angle2 - Cyan
    6: "#FFA500",  # angle3 - Orange
    7: "#800080",  # angle4 - Purple
}


class YOLOLabelingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Labeling Tool with Auto Detection")
        self.root.geometry("1600x1000")
        self.root.resizable(True, True)
        
        # Model path (relative to this script -> project/final/models/detection/bottle.pt)
        script_dir = Path(__file__).resolve().parent
        final_dir = script_dir.parent  # .../project/final
        self.model_path = str(final_dir / "models" / "detection" / "bottle.pt")
        self.model = None
        self.model_loaded = False
        
        # Data storage
        self.current_folder = ""
        self.image_files = []
        self.current_index = 0
        self.labels_data = {}  # {image_path: [{"class_id": int, "bbox": [x_center, y_center, width, height]}]}
        
        # Current image
        self.current_image_path = None
        self.original_image = None
        self.display_image = None
        self.image_scale = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0
        
        # Bounding boxes
        self.boxes = []  # [{"class_id": int, "bbox": [x1, y1, x2, y2], "selected": bool}]
        self.selected_box_index = None
        self.drag_start = None
        self.drag_mode = None  # "move", "resize_tl", "resize_tr", "resize_bl", "resize_br", "draw"
        self.drawing_box = None  # Temporary box being drawn
        
        # UI setup
        self.setup_ui()
        
        # Load model
        self.load_model()
        
    def setup_ui(self):
        """Setup user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls (with vertical scrollbar)
        left_panel = ttk.Frame(main_frame, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_panel.pack_propagate(False)

        # Canvas + scrollbar for left tools
        left_canvas = tk.Canvas(left_panel, borderwidth=0, highlightthickness=0, bg="#f0f0f0")
        left_scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        scroll_frame = ttk.Frame(left_canvas)
        canvas_window = left_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def _on_frame_configure(event):
            # Update scroll region
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            # Update canvas window width
            canvas_width = event.width
            left_canvas.itemconfig(canvas_window, width=canvas_width)

        def _on_canvas_configure(event):
            # Update scroll frame width when canvas is resized
            canvas_width = event.width
            left_canvas.itemconfig(canvas_window, width=canvas_width)

        scroll_frame.bind("<Configure>", _on_frame_configure)
        left_canvas.bind("<Configure>", _on_canvas_configure)
        
        # Mouse wheel support
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def _bind_to_mousewheel(event):
            left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            left_canvas.unbind_all("<MouseWheel>")
        
        left_canvas.bind('<Enter>', _bind_to_mousewheel)
        left_canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Store reference for later use
        self.left_canvas = left_canvas
        self.scroll_frame = scroll_frame
        
        # Model section
        model_frame = ttk.LabelFrame(scroll_frame, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_status_label = ttk.Label(model_frame, text="Model: ไม่ได้โหลด", foreground="red")
        self.model_status_label.pack(anchor=tk.W)
        
        ttk.Button(model_frame, text="โหลด Model", command=self.load_model).pack(fill=tk.X, pady=(5, 0))
        
        # Folder section
        folder_frame = ttk.LabelFrame(scroll_frame, text="โฟลเดอร์", padding=10)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(folder_frame, text="เลือกโฟลเดอร์ภาพ", command=self.select_folder).pack(fill=tk.X)
        
        self.folder_label = ttk.Label(folder_frame, text="ยังไม่ได้เลือกโฟลเดอร์", wraplength=200)
        self.folder_label.pack(fill=tk.X, pady=(5, 0))
        
        # Navigation section
        nav_frame = ttk.LabelFrame(scroll_frame, text="Navigation", padding=10)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        nav_buttons = ttk.Frame(nav_frame)
        nav_buttons.pack(fill=tk.X)
        
        ttk.Button(nav_buttons, text="◀ ก่อนหน้า", command=self.prev_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(nav_buttons, text="ถัดไป ▶", command=self.next_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        
        self.image_info_label = ttk.Label(nav_frame, text="0 / 0")
        self.image_info_label.pack(pady=(5, 0))
        
        # Detection section
        detect_frame = ttk.LabelFrame(scroll_frame, text="Auto Detection", padding=10)
        detect_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(detect_frame, text="Confidence:").pack(anchor=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.25)
        confidence_scale = ttk.Scale(detect_frame, from_=0.1, to=1.0, variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.pack(fill=tk.X)
        
        self.confidence_label = ttk.Label(detect_frame, text="0.25")
        self.confidence_label.pack()
        confidence_scale.configure(command=lambda v: self.confidence_label.config(text=f"{float(v):.2f}"))
        
        ttk.Button(detect_frame, text="🔍 Auto Detect (ภาพปัจจุบัน)", command=self.auto_detect).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(detect_frame, text="🗂 Auto Detect ทุกภาพในโฟลเดอร์", command=self.auto_detect_all).pack(fill=tk.X, pady=(5, 0))
        
        # Class selection
        class_frame = ttk.LabelFrame(scroll_frame, text="Class Selection", padding=10)
        class_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.selected_class_var = tk.IntVar(value=0)
        for class_id, class_name in CLASS_NAMES.items():
            color = CLASS_COLORS[class_id]
            rb = ttk.Radiobutton(
                class_frame,
                text=f"{class_id}: {class_name}",
                variable=self.selected_class_var,
                value=class_id
            )
            rb.pack(anchor=tk.W)
            # Add color indicator
            color_label = tk.Label(class_frame, bg=color, width=2, height=1)
            color_label.place(in_=rb, relx=1.0, x=5, rely=0.5, anchor=tk.W)
        
        # Box management
        box_frame = ttk.LabelFrame(scroll_frame, text="Box Management", padding=10)
        box_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Box list with scrollbar
        list_container = ttk.Frame(box_frame)
        list_container.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.box_listbox = tk.Listbox(list_container, height=8, font=("Courier", 9))
        box_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.box_listbox.yview)
        self.box_listbox.configure(yscrollcommand=box_scrollbar.set)
        
        self.box_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        box_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind listbox selection
        self.box_listbox.bind('<<ListboxSelect>>', self.on_box_list_select)
        self.box_listbox.bind('<Double-Button-1>', self.on_box_list_double_click)
        
        # Buttons
        button_frame = ttk.Frame(box_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="ลบ Box ที่เลือก", command=self.delete_selected_box_from_list).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(button_frame, text="เปลี่ยน Class", command=self.change_selected_box_class).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        
        ttk.Button(box_frame, text="ลบทั้งหมด", command=self.clear_all_boxes).pack(fill=tk.X, pady=(5, 0))
        
        # Box count
        self.box_count_label = ttk.Label(box_frame, text="Boxes: 0")
        self.box_count_label.pack(pady=(10, 0))
        
        # Export section
        export_frame = ttk.LabelFrame(scroll_frame, text="Export", padding=10)
        export_frame.pack(fill=tk.X)
        
        ttk.Button(export_frame, text="💾 บันทึก Label ปัจจุบัน", command=self.save_current_labels).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(export_frame, text="💾 Export YOLO Format", command=self.export_yolo_format).pack(fill=tk.X)
        
        # Help section
        help_frame = ttk.LabelFrame(scroll_frame, text="คีย์ลัด", padding=10)
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        help_text = (
            "← → : เปลี่ยนภาพ\n"
            "Delete : ลบ box\n"
            "D : Auto detect\n"
            "C : เปลี่ยน class\n"
            "S : บันทึก label"
        )
        ttk.Label(help_frame, text=help_text, font=("Courier", 9)).pack(anchor=tk.W)
        
        # Right panel - Image canvas
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        canvas_frame = ttk.Frame(right_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="crosshair")
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_hover)
        self.canvas.bind("<Button-3>", self.on_right_click)  # Right click to delete
        
        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Delete>", lambda e: self.delete_selected_box())
        self.root.bind("<d>", lambda e: self.auto_detect())
        self.root.bind("<c>", lambda e: self.change_selected_box_class())
        self.root.bind("<s>", lambda e: self.save_current_labels())
        self.root.focus_set()
        
    def load_model(self):
        """Load YOLO model"""
        try:
            if not os.path.exists(self.model_path):
                messagebox.showerror("Error", f"ไม่พบไฟล์ model:\n{self.model_path}")
                return
            
            self.model = YOLO(self.model_path)
            if torch.cuda.is_available():
                self.model.to('cuda')
                print("✅ YOLO model using CUDA")
            else:
                self.model.to('cpu')
                print("⚠️ YOLO model using CPU")
            
            self.model_loaded = True
            self.model_status_label.config(text="Model: โหลดแล้ว", foreground="green")
            messagebox.showinfo("Success", "โหลด YOLO model สำเร็จ!")
        except Exception as e:
            messagebox.showerror("Error", f"ไม่สามารถโหลด model:\n{str(e)}")
            self.model_loaded = False
            self.model_status_label.config(text="Model: ผิดพลาด", foreground="red")
    
    def select_folder(self):
        """Select folder containing images"""
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ภาพ")
        if folder:
            self.current_folder = folder
            self.folder_label.config(text=os.path.basename(folder))
            
            # Find all image files
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
            self.image_files = []
            for ext in extensions:
                self.image_files.extend(glob.glob(os.path.join(folder, ext)))
                self.image_files.extend(glob.glob(os.path.join(folder, ext.upper())))
            
            self.image_files.sort()
            
            if self.image_files:
                self.current_index = 0
                self.load_image(self.image_files[0])
                self.update_image_info()
            else:
                messagebox.showwarning("Warning", "ไม่พบไฟล์ภาพในโฟลเดอร์นี้")
    
    def load_image(self, image_path):
        """Load image and display"""
        try:
            self.current_image_path = image_path
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                messagebox.showerror("Error", f"ไม่สามารถโหลดภาพ: {image_path}")
                return
            
            # Convert BGR to RGB
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Load existing labels
            self.load_labels_for_image(image_path)
            
            # Display image
            self.display_image_on_canvas()
            self.update_box_count()
            self.update_box_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการโหลดภาพ:\n{str(e)}")
    
    def load_labels_for_image(self, image_path):
        """Load YOLO format labels for current image"""
        self.boxes = []
        
        # Get label file path
        label_path = self.get_label_path(image_path)
        
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                h, w = self.original_image.shape[:2]
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO format to xyxy
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                        
                        self.boxes.append({
                            "class_id": class_id,
                            "bbox": [x1, y1, x2, y2],
                            "selected": False
                        })
            except Exception as e:
                print(f"Error loading labels: {e}")
    
    def get_label_path(self, image_path):
        """Get corresponding label file path"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        folder = os.path.dirname(image_path)
        label_folder = os.path.join(folder, "labels")
        os.makedirs(label_folder, exist_ok=True)
        return os.path.join(label_folder, f"{base_name}.txt")
    
    def display_image_on_canvas(self):
        """Display image on canvas with bounding boxes"""
        if self.original_image is None:
            return
        
        # Calculate scale to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.display_image_on_canvas)
            return
        
        img_h, img_w = self.original_image.shape[:2]
        scale_w = canvas_width / img_w
        scale_h = canvas_height / img_h
        self.image_scale = min(scale_w, scale_h, 1.0)  # Don't scale up
        
        # Resize image for display
        display_w = int(img_w * self.image_scale)
        display_h = int(img_h * self.image_scale)
        
        display_img = cv2.resize(self.original_image, (display_w, display_h))
        display_img = Image.fromarray(display_img)
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(display_img)
        
        # Draw existing boxes
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box["bbox"]
            class_id = box["class_id"]
            selected = box.get("selected", False)
            
            # Scale coordinates
            x1_d = int(x1 * self.image_scale)
            y1_d = int(y1 * self.image_scale)
            x2_d = int(x2 * self.image_scale)
            y2_d = int(y2 * self.image_scale)
            
            # Get color
            color = CLASS_COLORS.get(class_id, "#FFFFFF")
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            
            # Draw rectangle
            width = 3 if selected else 2
            draw.rectangle([x1_d, y1_d, x2_d, y2_d], outline=color_rgb, width=width)
            
            # Draw label
            label_text = f"{class_id}: {CLASS_NAMES.get(class_id, 'unknown')}"
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Draw background for text
            bbox = draw.textbbox((x1_d, y1_d - 15), label_text, font=font)
            draw.rectangle(bbox, fill=color_rgb)
            draw.text((x1_d, y1_d - 15), label_text, fill="white", font=font)
            
            # Draw resize handles if selected
            if selected:
                handle_size = 8
                handles = [
                    (x1_d, y1_d),  # Top-left
                    (x2_d, y1_d),  # Top-right
                    (x1_d, y2_d),  # Bottom-left
                    (x2_d, y2_d),  # Bottom-right
                ]
                for hx, hy in handles:
                    draw.ellipse([hx-handle_size, hy-handle_size, hx+handle_size, hy+handle_size], 
                               fill="white", outline="black", width=2)
        
        # Draw temporary box being drawn
        if self.drawing_box is not None:
            x1, y1, x2, y2 = self.drawing_box
            x1_d = int(x1 * self.image_scale)
            y1_d = int(y1 * self.image_scale)
            x2_d = int(x2 * self.image_scale)
            y2_d = int(y2 * self.image_scale)
            
            class_id = self.selected_class_var.get()
            color = CLASS_COLORS.get(class_id, "#FFFFFF")
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            
            # Draw dashed rectangle
            draw.rectangle([x1_d, y1_d, x2_d, y2_d], outline=color_rgb, width=2)
        
        # Convert to PhotoImage
        self.display_image = ImageTk.PhotoImage(display_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_click(self, event):
        """Handle canvas click"""
        if self.original_image is None:
            return
        
        # Get click position in image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        img_x = int(canvas_x / self.image_scale)
        img_y = int(canvas_y / self.image_scale)
        
        # Check if clicking on a box
        clicked_box_index = None
        clicked_handle = None
        
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box["bbox"]
            
            # Check resize handles
            handle_size = 8 / self.image_scale
            if abs(img_x - x1) < handle_size and abs(img_y - y1) < handle_size:
                clicked_box_index = i
                clicked_handle = "resize_tl"
                break
            elif abs(img_x - x2) < handle_size and abs(img_y - y1) < handle_size:
                clicked_box_index = i
                clicked_handle = "resize_tr"
                break
            elif abs(img_x - x1) < handle_size and abs(img_y - y2) < handle_size:
                clicked_box_index = i
                clicked_handle = "resize_bl"
                break
            elif abs(img_x - x2) < handle_size and abs(img_y - y2) < handle_size:
                clicked_box_index = i
                clicked_handle = "resize_br"
                break
            # Check if clicking inside box
            elif x1 <= img_x <= x2 and y1 <= img_y <= y2:
                clicked_box_index = i
                clicked_handle = "move"
                break
        
        if clicked_box_index is not None:
            # Select box
            for i in range(len(self.boxes)):
                self.boxes[i]["selected"] = (i == clicked_box_index)
            self.selected_box_index = clicked_box_index
            self.drag_mode = clicked_handle
            self.drag_start = (img_x, img_y)
            self.update_box_list()  # Update listbox selection
        else:
            # Start drawing new box
            for box in self.boxes:
                box["selected"] = False
            self.selected_box_index = None
            self.drag_mode = "draw"
            self.drag_start = (img_x, img_y)
            self.drawing_box = None
            self.update_box_list()  # Clear listbox selection
        
        self.display_image_on_canvas()
    
    def on_canvas_drag(self, event):
        """Handle canvas drag"""
        if self.drag_start is None:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        img_x = int(canvas_x / self.image_scale)
        img_y = int(canvas_y / self.image_scale)
        
        if self.drag_mode == "draw":
            # Drawing new box
            start_x, start_y = self.drag_start
            x1 = min(start_x, img_x)
            y1 = min(start_y, img_y)
            x2 = max(start_x, img_x)
            y2 = max(start_y, img_y)
            
            # Ensure within image bounds
            h, w = self.original_image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            self.drawing_box = [x1, y1, x2, y2]
        elif self.selected_box_index is not None:
            # Moving/resizing existing box
            dx = img_x - self.drag_start[0]
            dy = img_y - self.drag_start[1]
            
            box = self.boxes[self.selected_box_index]
            x1, y1, x2, y2 = box["bbox"]
            
            if self.drag_mode == "move":
                # Move entire box
                box["bbox"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
            elif self.drag_mode == "resize_tl":
                box["bbox"] = [x1 + dx, y1 + dy, x2, y2]
            elif self.drag_mode == "resize_tr":
                box["bbox"] = [x1, y1 + dy, x2 + dx, y2]
            elif self.drag_mode == "resize_bl":
                box["bbox"] = [x1 + dx, y1, x2, y2 + dy]
            elif self.drag_mode == "resize_br":
                box["bbox"] = [x1, y1, x2 + dx, y2 + dy]
            
            # Ensure valid bbox
            x1, y1, x2, y2 = box["bbox"]
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Ensure within image bounds
            h, w = self.original_image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            box["bbox"] = [x1, y1, x2, y2]
            self.drag_start = (img_x, img_y)
        
        self.display_image_on_canvas()
    
    def on_canvas_release(self, event):
        """Handle canvas release"""
        if self.drag_mode == "draw" and self.drawing_box is not None:
            # Create new box
            x1, y1, x2, y2 = self.drawing_box
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:  # Minimum size
                class_id = self.selected_class_var.get()
                self.boxes.append({
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                    "selected": False
                })
                # Select the newly created box
                self.selected_box_index = len(self.boxes) - 1
                self.boxes[self.selected_box_index]["selected"] = True
        
        self.drag_start = None
        self.drag_mode = None
        self.drawing_box = None
        self.update_box_count()
        self.update_box_list()
        self.display_image_on_canvas()
    
    def on_canvas_hover(self, event):
        """Handle canvas hover"""
        if self.selected_box_index is not None:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            img_x = int(canvas_x / self.image_scale)
            img_y = int(canvas_y / self.image_scale)
            
            box = self.boxes[self.selected_box_index]
            x1, y1, x2, y2 = box["bbox"]
            
            handle_size = 8 / self.image_scale
            if (abs(img_x - x1) < handle_size and abs(img_y - y1) < handle_size) or \
               (abs(img_x - x2) < handle_size and abs(img_y - y2) < handle_size):
                self.canvas.config(cursor="sizing")
            else:
                self.canvas.config(cursor="fleur")
        else:
            self.canvas.config(cursor="crosshair")
    
    def on_right_click(self, event):
        """Handle right click - delete box"""
        if self.selected_box_index is not None:
            self.delete_selected_box()
    
    def delete_selected_box(self):
        """Delete selected box"""
        if self.selected_box_index is not None:
            self.boxes.pop(self.selected_box_index)
            self.selected_box_index = None
            self.update_box_count()
            self.update_box_list()
            self.display_image_on_canvas()
    
    def change_selected_box_class(self):
        """Change class of selected box"""
        if self.selected_box_index is not None:
            new_class = self.selected_class_var.get()
            self.boxes[self.selected_box_index]["class_id"] = new_class
            self.update_box_list()
            self.display_image_on_canvas()
    
    def update_box_count(self):
        """Update box count display"""
        total = len(self.boxes)
        class_counts = {}
        for box in self.boxes:
            class_id = box["class_id"]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        count_text = f"Boxes: {total}"
        if class_counts:
            details = []
            for class_id, count in sorted(class_counts.items()):
                class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
                details.append(f"{class_name}: {count}")
            count_text += "\n" + ", ".join(details)
        
        self.box_count_label.config(text=count_text)
    
    def update_box_list(self):
        """Update the box listbox display"""
        self.box_listbox.delete(0, tk.END)
        
        for i, box in enumerate(self.boxes):
            class_id = box["class_id"]
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            x1, y1, x2, y2 = box["bbox"]
            confidence = box.get("confidence", None)
            
            # Format display text
            if confidence is not None:
                text = f"[{i}] {class_id}:{class_name} ({x1},{y1})-({x2},{y2}) conf:{confidence:.2f}"
            else:
                text = f"[{i}] {class_id}:{class_name} ({x1},{y1})-({x2},{y2})"
            
            self.box_listbox.insert(tk.END, text)
            
            # Highlight selected box
            if box.get("selected", False):
                self.box_listbox.selection_set(i)
    
    def on_box_list_select(self, event):
        """Handle listbox selection"""
        selection = self.box_listbox.curselection()
        if selection:
            index = selection[0]
            if 0 <= index < len(self.boxes):
                # Deselect all
                for i in range(len(self.boxes)):
                    self.boxes[i]["selected"] = False
                
                # Select the chosen one
                self.boxes[index]["selected"] = True
                self.selected_box_index = index
                
                # Update display
                self.display_image_on_canvas()
    
    def on_box_list_double_click(self, event):
        """Handle double click on listbox - focus on box"""
        selection = self.box_listbox.curselection()
        if selection:
            index = selection[0]
            if 0 <= index < len(self.boxes):
                # Center view on selected box
                box = self.boxes[index]
                x1, y1, x2, y2 = box["bbox"]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Scroll canvas to center (if needed)
                # This is a simple implementation - you can enhance it
                self.display_image_on_canvas()
    
    def delete_selected_box_from_list(self):
        """Delete box selected in listbox"""
        selection = self.box_listbox.curselection()
        if selection:
            index = selection[0]
            if 0 <= index < len(self.boxes):
                self.boxes.pop(index)
                self.selected_box_index = None
                self.update_box_list()
                self.update_box_count()
                self.display_image_on_canvas()
        elif self.selected_box_index is not None:
            # Fallback to canvas selection
            self.delete_selected_box()
    
    def clear_all_boxes(self):
        """Clear all boxes"""
        if messagebox.askyesno("Confirm", "ต้องการลบ boxes ทั้งหมดหรือไม่?"):
            self.boxes = []
            self.selected_box_index = None
            self.update_box_count()
            self.update_box_list()
            self.display_image_on_canvas()
    
    def auto_detect(self):
        """Auto detect objects using YOLO model"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "กรุณาโหลด model ก่อน")
            return
        
        if self.original_image is None:
            messagebox.showwarning("Warning", "กรุณาโหลดภาพก่อน")
            return
        
        try:
            # Convert RGB to BGR for YOLO
            img_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
            
            # Run detection
            confidence = self.confidence_var.get()
            results = self.model(img_bgr, conf=confidence, imgsz=640)
            
            # Clear existing boxes
            self.boxes = []
            
            # Process results
            h, w = self.original_image.shape[:2]
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0].item())
                    confidence_score = float(box.conf[0].item())
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    if x2 > x1 and y2 > y1:
                        self.boxes.append({
                            "class_id": class_id,
                            "bbox": [x1, y1, x2, y2],
                            "selected": False,
                            "confidence": confidence_score
                        })
            
            self.selected_box_index = None
            self.update_box_count()
            self.update_box_list()
            self.display_image_on_canvas()
            
            messagebox.showinfo("Success", f"ตรวจจับได้ {len(self.boxes)} objects")
            
        except Exception as e:
            messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการตรวจจับ:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def auto_detect_all(self):
        """Auto detect objects for all images in current folder (batch)"""
        import threading

        if not self.model_loaded:
            messagebox.showwarning("Warning", "กรุณาโหลด model ก่อน")
            return

        if not self.image_files:
            messagebox.showwarning("Warning", "กรุณาเลือกโฟลเดอร์ภาพก่อน")
            return

        if not messagebox.askyesno("ยืนยัน", "ต้องการให้ Auto Detect ทุกภาพในโฟลเดอร์นี้หรือไม่?\nระบบจะสร้าง/เขียนทับไฟล์ label (.txt) ของแต่ละภาพ"):
            return

        def worker():
            total_images = len(self.image_files)
            success_images = 0
            total_boxes = 0

            confidence = self.confidence_var.get()

            for idx, image_path in enumerate(self.image_files, 1):
                try:
                    img_bgr = cv2.imread(image_path)
                    if img_bgr is None:
                        print(f"❌ AUTO DETECT ALL: โหลดภาพไม่ได้: {image_path}")
                        continue

                    # Run detection
                    results = self.model(img_bgr, conf=confidence, imgsz=640)

                    h, w = img_bgr.shape[:2]
                    boxes_for_image = []

                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls[0].item())
                            # confidence_score = float(box.conf[0].item())

                            # Ensure coordinates are within image bounds
                            x1 = max(0, min(x1, w))
                            y1 = max(0, min(y1, h))
                            x2 = max(0, min(x2, w))
                            y2 = max(0, min(y2, h))

                            if x2 > x1 and y2 > y1:
                                boxes_for_image.append((class_id, x1, y1, x2, y2))

                    # Save labels for this image
                    label_path = self.get_label_path(image_path)
                    with open(label_path, "w") as f:
                        for class_id, x1, y1, x2, y2 in boxes_for_image:
                            x_center = ((x1 + x2) / 2) / w
                            y_center = ((y1 + y2) / 2) / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                    success_images += 1
                    total_boxes += len(boxes_for_image)

                    print(f"✅ AUTO DETECT ALL [{idx}/{total_images}] {os.path.basename(image_path)} -> {len(boxes_for_image)} boxes")

                except Exception as ex:
                    print(f"❌ AUTO DETECT ALL ERROR ({image_path}): {ex}")

            def show_done():
                msg = (
                    f"Auto Detect ทุกภาพเสร็จสิ้น\n\n"
                    f"ภาพทั้งหมด: {total_images}\n"
                    f"ประมวลผลสำเร็จ: {success_images}\n"
                    f"จำนวน boxes ที่ตรวจจับได้รวม: {total_boxes}\n\n"
                    f"ไฟล์ label (.txt) ถูกบันทึกไว้ในโฟลเดอร์ labels ของแต่ละภาพแล้ว"
                )
                messagebox.showinfo("เสร็จสิ้น", msg)

                # รีเฟรชภาพปัจจุบันให้ใช้ label ใหม่ (ถ้ามี)
                if self.current_image_path:
                    self.load_image(self.current_image_path)

            self.root.after(0, show_done)

        threading.Thread(target=worker, daemon=True).start()
    
    def prev_image(self):
        """Load previous image"""
        if self.image_files and self.current_index > 0:
            self.save_current_labels()
            self.current_index -= 1
            self.load_image(self.image_files[self.current_index])
            self.update_image_info()
    
    def next_image(self):
        """Load next image"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.save_current_labels()
            self.current_index += 1
            self.load_image(self.image_files[self.current_index])
            self.update_image_info()
    
    def on_closing(self):
        """Handle window closing"""
        self.save_current_labels()
        self.root.destroy()
    
    def update_image_info(self):
        """Update image info label"""
        if self.image_files:
            self.image_info_label.config(text=f"{self.current_index + 1} / {len(self.image_files)}")
    
    def save_current_labels(self):
        """Save labels for current image in YOLO format"""
        if self.current_image_path is None:
            return
        
        label_path = self.get_label_path(self.current_image_path)
        h, w = self.original_image.shape[:2]
        
        try:
            with open(label_path, 'w') as f:
                for box in self.boxes:
                    x1, y1, x2, y2 = box["bbox"]
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    class_id = box["class_id"]
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Show brief confirmation
            self.root.after(100, lambda: None)  # Small delay to update UI
        except Exception as e:
            messagebox.showerror("Error", f"ไม่สามารถบันทึก label:\n{str(e)}")
            print(f"Error saving labels: {e}")
    
    def export_yolo_format(self):
        """Export all images and labels in YOLO format"""
        if not self.image_files:
            messagebox.showwarning("Warning", "กรุณาเลือกโฟลเดอร์ภาพก่อน")
            return
        
        # Ask for output folder
        output_folder = filedialog.askdirectory(title="เลือกโฟลเดอร์สำหรับ Export")
        if not output_folder:
            return
        
        try:
            # Save current labels first
            self.save_current_labels()
            
            # Create folders
            images_folder = os.path.join(output_folder, "images")
            labels_folder = os.path.join(output_folder, "labels")
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(labels_folder, exist_ok=True)
            
            # Copy images and labels
            copied_count = 0
            
            for image_path in self.image_files:
                # Copy image
                image_name = os.path.basename(image_path)
                dest_image = os.path.join(images_folder, image_name)
                
                import shutil
                shutil.copy2(image_path, dest_image)
                
                # Copy label if exists
                label_path = self.get_label_path(image_path)
                if os.path.exists(label_path):
                    label_name = os.path.basename(label_path)
                    dest_label = os.path.join(labels_folder, label_name)
                    shutil.copy2(label_path, dest_label)
                    copied_count += 1
            
            messagebox.showinfo("Success", 
                f"Export สำเร็จ!\n\n"
                f"Images: {len(self.image_files)} ไฟล์\n"
                f"Labels: {copied_count} ไฟล์\n\n"
                f"โฟลเดอร์: {output_folder}")
            
        except Exception as e:
            messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการ export:\n{str(e)}")
            import traceback
            traceback.print_exc()


def main():
    root = tk.Tk()
    app = YOLOLabelingUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

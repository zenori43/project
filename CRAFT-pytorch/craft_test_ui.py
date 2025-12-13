"""
CRAFT Test UI - A tool based on test.py for drawing three equal lines within detected text regions
"""

import sys
import os
import time
import argparse
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
from skimage import io
import numpy as np

# Add CRAFT-pytorch to path
sys.path.append(r'c:\Users\Win 10 Home\CRAFT-pytorch')
import craft_utils
import imgproc
import file_utils
from craft import CRAFT
from collections import OrderedDict

def copyStateDict(state_dict):
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

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None, canvas_size=1280, mag_ratio=1.5, show_time=False):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
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

    # render results
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def draw_three_lines_in_box(image, box, line_color=(0, 255, 0), line_thickness=2):
    """Draw three equal horizontal lines within a detected text box"""
    if box is None or len(box) < 4:
        return image, []
    
    try:
        # Convert box to rectangle if it's a polygon
        if len(box) == 4:
            # Get bounding rectangle
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
        else:
            # Handle polygon case
            box_array = np.array(box, dtype=np.int32)
            x_min, y_min, w, h = cv2.boundingRect(box_array)
            x_max = x_min + w
            y_max = y_min + h
        
        # Ensure coordinates are integers and within image bounds
        x_min = max(0, int(x_min))
        x_max = min(image.shape[1], int(x_max))
        y_min = max(0, int(y_min))
        y_max = min(image.shape[0], int(y_max))
        
        # Check if box is valid
        if x_min >= x_max or y_min >= y_max:
            return image, []
        
        # Calculate three equal lines
        box_height = y_max - y_min
        if box_height < 4:  # Too small to draw lines
            return image, []
            
        line_spacing = box_height // 4  # Divide into 4 parts, lines at 1/4, 2/4, 3/4
        
        line_positions = [
            y_min + line_spacing,
            y_min + 2 * line_spacing,
            y_min + 3 * line_spacing
        ]
        
        # Store crop regions (3 rectangles between lines)
        crop_regions = []
        
        # Draw the three lines and create crop regions
        for i in range(len(line_positions) - 1):
            y_pos_int = int(line_positions[i])
            next_y_pos_int = int(line_positions[i + 1])
            
            if 0 <= y_pos_int < image.shape[0] and 0 <= next_y_pos_int < image.shape[0]:
                # Draw line
                cv2.line(image, (x_min, y_pos_int), (x_max, y_pos_int), line_color, line_thickness)
                
                # Create crop region (rectangle between this line and next line)
                crop_region = {
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_pos_int,
                    'y_max': next_y_pos_int
                }
                crop_regions.append(crop_region)
        
        # Draw the last line
        if len(line_positions) > 0:
            last_y_pos_int = int(line_positions[-1])
            if 0 <= last_y_pos_int < image.shape[0]:
                cv2.line(image, (x_min, last_y_pos_int), (x_max, last_y_pos_int), line_color, line_thickness)
        
        return image, crop_regions
        
    except Exception as e:
        print(f"Error drawing lines in box: {str(e)}")
        return image, []

def crop_regions_from_image(image, crop_regions, output_folder, base_filename):
    """Crop regions from image and save them"""
    cropped_files = []
    
    try:
        for i, region in enumerate(crop_regions):
            # Crop the region
            cropped_image = image[region['y_min']:region['y_max'], region['x_min']:region['x_max']]
            
            # Save the cropped image
            crop_filename = f"{base_filename}_crop_{i+1}.png"
            crop_path = os.path.join(output_folder, crop_filename)
            cv2.imwrite(crop_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
            cropped_files.append(crop_path)
            
            print(f"Saved cropped region {i+1}: {crop_filename}")
            
    except Exception as e:
        print(f"Error cropping regions: {str(e)}")
    
    return cropped_files

def process_image_with_lines(image_path, net, settings):
    """Process a single image and draw three lines in detected boxes"""
    try:
        # Load image
        image = imgproc.loadImage(image_path)
        image_for_vis = image[:,:,::-1].copy()  # Convert to BGR for OpenCV
        
        # Detect text regions
        bboxes, polys, score_text = test_net(
            net, image, 
            settings['text_threshold'], 
            settings['link_threshold'], 
            settings['low_text'], 
            settings['cuda'], 
            settings['poly'], 
            None,  # refine_net
            settings['canvas_size'],
            settings['mag_ratio'],
            settings['show_time']
        )
        
        # Draw three lines in each detected box
        result_image = image_for_vis.copy()
        detected_boxes = []
        
        if len(polys) > 0:
            try:
                # Filter out None values from polys
                valid_polys = [poly for poly in polys if poly is not None and len(poly) > 0]
                
                if len(valid_polys) > 0:
                    # Group all polys into a single large bounding box
                    all_points = np.concatenate(valid_polys, axis=0)
                    rect = cv2.minAreaRect(all_points)
                    main_box = cv2.boxPoints(rect)
                    main_box = np.int32(main_box)
                    detected_boxes.append(main_box)
                    
                    # Draw the main bounding box outline
                    cv2.polylines(result_image, [main_box], True, (0, 255, 0), 3)
                    
                    # Draw three lines in the main detected region only and get crop regions
                    result_image, crop_regions = draw_three_lines_in_box(result_image, main_box, (255, 0, 0), 2)
                    
                    # Also draw individual small boxes for each detected text region (for reference)
                    for i, poly in enumerate(valid_polys):
                        try:
                            # Draw the individual box outline (thinner)
                            poly_int = poly.astype(np.int32)
                            cv2.polylines(result_image, [poly_int], True, (0, 255, 255), 1)
                        except Exception as e:
                            print(f"Error processing individual poly {i}: {str(e)}")
                            continue
                            
            except Exception as e:
                print(f"Error processing polys: {str(e)}")
                # If grouping fails, try to process individual polys
                for i, poly in enumerate(polys):
                    if poly is not None and len(poly) > 0:
                        try:
                            result_image = draw_three_lines_in_box(result_image, poly, (0, 255, 255), 1)
                        except Exception as e:
                            print(f"Error processing poly {i}: {str(e)}")
                            continue
        
        return result_image, detected_boxes, score_text, crop_regions
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, [], None

class CRAFTTestUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CRAFT Test UI - Three Lines Detection")
        self.root.geometry("1200x800")
        
        # Initialize CRAFT model
        self.net = None
        self.model_loaded = False
        
        # Settings
        self.settings = {
            'trained_model': r'C:\Users\Win 10 Home\CRAFT-pytorch\craft_mlt_25k.pth',
            'text_threshold': 0.7,
            'low_text': 0.4,
            'link_threshold': 0.4,
            'cuda': False,
            'canvas_size': 1280,
            'mag_ratio': 1.5,
            'poly': False,
            'show_time': False,
            'output_folder': './craft_test_results/'
        }
        
        # Image navigation
        self.current_folder = ""
        self.image_files = []
        self.current_image_index = 0
        self.current_image_path = ""
        
        # UI elements
        self.setup_ui()
        
        # Load model
        self.load_model()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Folder selection
        ttk.Label(main_frame, text="Image Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(main_frame, textvariable=self.folder_var, width=50)
        folder_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, padx=5, pady=5)
        
        # Navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Button(nav_frame, text="← Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        self.image_info_label = ttk.Label(nav_frame, text="No image selected")
        self.image_info_label.pack(side=tk.LEFT, padx=20)
        ttk.Button(nav_frame, text="Next →", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=3, padx=10)
        
        ttk.Button(control_frame, text="Process Single Image", command=self.process_single_image).pack(pady=2)
        ttk.Button(control_frame, text="Process All Images", command=self.process_folder).pack(pady=2)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Settings controls
        ttk.Label(settings_frame, text="Text Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.text_threshold_var = tk.DoubleVar(value=self.settings['text_threshold'])
        ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.text_threshold_var, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(settings_frame, textvariable=tk.StringVar(value=f"{self.settings['text_threshold']:.1f}")).grid(row=0, column=2)
        
        ttk.Label(settings_frame, text="Link Threshold:").grid(row=1, column=0, sticky=tk.W)
        self.link_threshold_var = tk.DoubleVar(value=self.settings['link_threshold'])
        ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.link_threshold_var, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(settings_frame, textvariable=tk.StringVar(value=f"{self.settings['link_threshold']:.1f}")).grid(row=1, column=2)
        
        # Output folder
        ttk.Label(settings_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W)
        self.output_folder_var = tk.StringVar(value=self.settings['output_folder'])
        ttk.Entry(settings_frame, textvariable=self.output_folder_var, width=30).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(settings_frame, text="Browse", command=self.browse_output_folder).grid(row=2, column=2, padx=5)
        
        # Image display
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="5")
        image_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg="white")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=6, column=0, columnspan=3, pady=5)
        
        # Bind keyboard events
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
    
    def load_model(self):
        """Load the CRAFT model"""
        try:
            self.status_label.config(text="Loading CRAFT model...")
            self.root.update()
            
            # Initialize CRAFT
            self.net = CRAFT()
            
            print('Loading weights from checkpoint (' + self.settings['trained_model'] + ')')
            if self.settings['cuda']:
                self.net.load_state_dict(copyStateDict(torch.load(self.settings['trained_model'])))
            else:
                self.net.load_state_dict(copyStateDict(torch.load(self.settings['trained_model'], map_location='cpu')))
            
            if self.settings['cuda']:
                self.net = self.net.cuda()
                self.net = torch.nn.DataParallel(self.net)
                cudnn.benchmark = False
            
            self.net.eval()
            self.model_loaded = True
            
            self.status_label.config(text="Model loaded successfully")
            print("CRAFT model loaded successfully")
            
        except Exception as e:
            self.status_label.config(text=f"Error loading model: {str(e)}")
            print(f"Error loading CRAFT model: {str(e)}")
    
    def browse_folder(self):
        """Browse for image folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)
            self.current_folder = folder
            self.load_image_files()
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder_var.set(folder)
            self.settings['output_folder'] = folder
    
    def load_image_files(self):
        """Load image files from the selected folder"""
        try:
            self.image_files = []
            if self.current_folder:
                # Use file_utils.get_files which returns (imgs, masks, xmls)
                imgs, masks, xmls = file_utils.get_files(self.current_folder)
                self.image_files = imgs  # Use only image files
                
                self.current_image_index = 0
                if self.image_files:
                    self.current_image_path = self.image_files[0]
                    self.update_image_info()
                    self.load_current_image()
                else:
                    self.image_info_label.config(text="No images found")
                    self.current_image_path = ""
        except Exception as e:
            print(f"Error loading image files: {str(e)}")
    
    def update_image_info(self):
        """Update image info label"""
        if self.image_files:
            self.image_info_label.config(text=f"Image {self.current_image_index + 1} of {len(self.image_files)}")
    
    def load_current_image(self):
        """Load and display current image"""
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            return
        
        try:
            # Load image
            image = cv2.imread(self.current_image_path)
            if image is None:
                return
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            height, width = image_rgb.shape[:2]
            max_size = 600
            
            if height > max_size or width > max_size:
                scale = min(max_size / height, max_size / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
    
    def previous_image(self):
        """Go to previous image"""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.current_image_path = self.image_files[self.current_image_index]
            self.update_image_info()
            self.load_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.current_image_path = self.image_files[self.current_image_index]
            self.update_image_info()
            self.load_current_image()
    
    def process_single_image(self):
        """Process current image"""
        if not self.current_image_path or not self.model_loaded:
            messagebox.showwarning("Warning", "Please select an image and ensure model is loaded")
            return
        
        # Update settings
        self.settings['text_threshold'] = self.text_threshold_var.get()
        self.settings['link_threshold'] = self.link_threshold_var.get()
        self.settings['output_folder'] = self.output_folder_var.get()
        
        # Create output folder
        os.makedirs(self.settings['output_folder'], exist_ok=True)
        
        # Process in thread
        thread = threading.Thread(target=self._process_single_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_single_image_thread(self):
        """Process single image in background thread"""
        try:
            self.status_label.config(text="Processing image...")
            self.progress_var.set(50)
            self.root.update()
            
            # Process image
            result_image, detected_boxes, score_text, crop_regions = process_image_with_lines(
                self.current_image_path, self.net, self.settings
            )
            
            if result_image is not None:
                # Save result
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                result_filename = f"{base_name}_three_lines.png"
                result_path = os.path.join(self.settings['output_folder'], result_filename)
                cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                
                # Crop and save regions if crop_regions exist
                cropped_files = []
                if crop_regions:
                    # Load original image for cropping
                    original_image = imgproc.loadImage(self.current_image_path)
                    cropped_files = crop_regions_from_image(original_image, crop_regions, self.settings['output_folder'], base_name)
                
                # Display result
                self.display_result_image(result_image)
                
                # Show success message with crop info
                crop_info = f"\nCropped {len(cropped_files)} regions" if cropped_files else ""
                self.status_label.config(text=f"Processed successfully. Saved: {result_filename}{crop_info}")
                messagebox.showinfo("Success", f"Image processed successfully!\nSaved as: {result_filename}{crop_info}")
            else:
                self.status_label.config(text="Error processing image")
                messagebox.showerror("Error", "Failed to process image")
            
            self.progress_var.set(100)
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            self.progress_var.set(0)
    
    def process_folder(self):
        """Process all images in folder"""
        if not self.image_files or not self.model_loaded:
            messagebox.showwarning("Warning", "Please select a folder with images and ensure model is loaded")
            return
        
        # Update settings
        self.settings['text_threshold'] = self.text_threshold_var.get()
        self.settings['link_threshold'] = self.link_threshold_var.get()
        self.settings['output_folder'] = self.output_folder_var.get()
        
        # Create output folder
        os.makedirs(self.settings['output_folder'], exist_ok=True)
        
        # Process in thread
        thread = threading.Thread(target=self._process_folder_thread)
        thread.daemon = True
        thread.start()
    
    def _process_folder_thread(self):
        """Process all images in background thread"""
        try:
            total_images = len(self.image_files)
            processed_count = 0
            
            self.status_label.config(text="Processing all images...")
            self.progress_var.set(0)
            self.root.update()
            
            for i, image_path in enumerate(self.image_files):
                try:
                    self.status_label.config(text=f"Processing {i+1}/{total_images}: {os.path.basename(image_path)}")
                    self.progress_var.set((i / total_images) * 100)
                    self.root.update()
                    
                    # Process image
                    result_image, detected_boxes, score_text, crop_regions = process_image_with_lines(
                        image_path, self.net, self.settings
                    )
                    
                    if result_image is not None:
                        # Save result
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        result_filename = f"{base_name}_three_lines.png"
                        result_path = os.path.join(self.settings['output_folder'], result_filename)
                        cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                        
                        # Crop and save regions if crop_regions exist
                        if crop_regions:
                            original_image = imgproc.loadImage(image_path)
                            crop_regions_from_image(original_image, crop_regions, self.settings['output_folder'], base_name)
                        
                        processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
            
            self.progress_var.set(100)
            self.status_label.config(text=f"Completed! Processed {processed_count}/{total_images} images")
            messagebox.showinfo("Success", f"Processing completed!\nProcessed {processed_count}/{total_images} images\nResults saved in: {self.settings['output_folder']}")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error processing folder: {str(e)}")
            self.progress_var.set(0)
    
    def display_result_image(self, result_image):
        """Display result image in canvas"""
        try:
            # Resize for display
            height, width = result_image.shape[:2]
            max_size = 600
            
            if height > max_size or width > max_size:
                scale = min(max_size / height, max_size / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                result_image = cv2.resize(result_image, (new_width, new_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(result_image)
            self.result_photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.result_photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            print(f"Error displaying result image: {str(e)}")

def main():
    root = tk.Tk()
    app = CRAFTTestUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

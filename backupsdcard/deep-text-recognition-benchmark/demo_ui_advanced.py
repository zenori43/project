import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from datetime import datetime
from PIL import Image, ImageTk
import string
import json
import subprocess
import tempfile

# Import modules from the current project
from utils import CTCLabelConverter, AttnLabelConverter
from model import Model

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AdvancedDemoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced OCR Demo with Text Recognition")
        self.root.geometry("1400x900")
        
        # Model variables
        self.model = None
        self.converter = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # UI variables
        self.selected_folder = tk.StringVar()
        self.processing = False
        self.current_image = None
        self.current_image_path = None
        
        # Image navigation variables
        self.image_files = []
        self.current_image_index = -1
        self.processed_results = {}  # Store results for each image
        
        # Settings
        self.settings = {
            'show_model_results': True,
            'save_results': True,
            'output_folder': 'demo_results',
            'demo_model_path': 'best_accuracy.pth'
        }
        
        self.setup_ui()
        self.setup_keyboard_events()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Advanced OCR Demo with Text Recognition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model path
        ttk.Label(settings_frame, text="Demo Model Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.model_path_var = tk.StringVar(value=self.settings['demo_model_path'])
        model_path_entry = ttk.Entry(settings_frame, textvariable=self.model_path_var, width=40)
        model_path_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(settings_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5, pady=2)
        
        # Checkboxes
        self.show_model_var = tk.BooleanVar(value=self.settings['show_model_results'])
        ttk.Checkbutton(settings_frame, text="Show Demo Results", 
                       variable=self.show_model_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.save_results_var = tk.BooleanVar(value=self.settings['save_results'])
        ttk.Checkbutton(settings_frame, text="Save Results", 
                       variable=self.save_results_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Folder selection
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        folder_frame.columnconfigure(1, weight=1)
        
        ttk.Label(folder_frame, text="Select Image Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        folder_entry = ttk.Entry(folder_frame, textvariable=self.selected_folder, width=50)
        folder_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.process_button = ttk.Button(button_frame, text="Process Folder", 
                                       command=self.process_folder, state="disabled")
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Processing", 
                                    command=self.stop_processing, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=(0, 10))
        
        # Navigation buttons
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        self.prev_button = ttk.Button(nav_frame, text="← Previous", command=self.previous_image, state="disabled")
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.image_info_label = ttk.Label(nav_frame, text="No images loaded")
        self.image_info_label.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(nav_frame, text="Next →", command=self.next_image, state="disabled")
        self.next_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        results_frame.columnconfigure(0, weight=2)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Text results
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        ttk.Label(text_frame, text="Recognized Text:").grid(row=0, column=0, sticky=tk.W)
        self.text_results = scrolledtext.ScrolledText(text_frame, height=20, width=70)
        self.text_results.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image preview
        image_frame = ttk.Frame(results_frame)
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(1, weight=1)
        
        ttk.Label(image_frame, text="Image Preview:").grid(row=0, column=0, sticky=tk.W)
        self.image_label = ttk.Label(image_frame, text="No image selected", 
                                   border=1, relief="solid")
        self.image_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure results frame weights
        results_frame.columnconfigure(0, weight=2)
        results_frame.columnconfigure(1, weight=1)
    
    def setup_keyboard_events(self):
        """Setup keyboard event handlers"""
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Up>', lambda e: self.previous_image())
        self.root.bind('<Down>', lambda e: self.next_image())
        self.root.bind('<space>', lambda e: self.process_current_image())
        self.root.focus_set()  # Set focus to main window
    
    def browse_model(self):
        """Browse for demo model file"""
        model_file = filedialog.askopenfilename(
            title="Select Demo Model File",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if model_file:
            self.model_path_var.set(model_file)
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.selected_folder.set(folder)
            self.load_image_files()
            self.process_button.config(state="normal")
    
    def load_image_files(self):
        """Load image files from selected folder"""
        folder_path = self.selected_folder.get()
        if not folder_path:
            return
        
        self.image_files = self.get_image_files(folder_path)
        self.current_image_index = -1
        self.processed_results = {}
        
        if self.image_files:
            self.current_image_index = 0
            self.update_navigation_buttons()
            self.show_current_image()
        else:
            self.image_info_label.config(text="No images found")
            self.update_navigation_buttons()
    
    def update_navigation_buttons(self):
        """Update navigation button states"""
        if not self.image_files:
            self.prev_button.config(state="disabled")
            self.next_button.config(state="disabled")
            return
        
        # Update previous button
        if self.current_image_index <= 0:
            self.prev_button.config(state="disabled")
        else:
            self.prev_button.config(state="normal")
        
        # Update next button
        if self.current_image_index >= len(self.image_files) - 1:
            self.next_button.config(state="disabled")
        else:
            self.next_button.config(state="normal")
        
        # Update image info
        if self.image_files:
            self.image_info_label.config(text=f"Image {self.current_image_index + 1} of {len(self.image_files)}")
    
    def show_current_image(self):
        """Show the current image"""
        if not self.image_files or self.current_image_index < 0:
            return
        
        current_image_path = self.image_files[self.current_image_index]
        self.update_image_preview(current_image_path)
        
        # Show results if available
        if current_image_path in self.processed_results:
            self.display_results(current_image_path)
        else:
            self.text_results.delete(1.0, tk.END)
            self.text_results.insert(tk.END, "No results available. Press Space to process this image.")
    
    def previous_image(self):
        """Go to previous image"""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_navigation_buttons()
            self.show_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.update_navigation_buttons()
            self.show_current_image()
    
    def process_current_image(self):
        """Process the current image"""
        if not self.image_files or self.current_image_index < 0:
            return
        
        current_image_path = self.image_files[self.current_image_index]
        result_text = self.process_single_image(current_image_path)
        
        # Store results
        self.processed_results[current_image_path] = result_text
        
        # Display results
        self.display_results(current_image_path)
    
    def display_results(self, image_path):
        """Display results for a specific image"""
        if image_path in self.processed_results:
            self.text_results.delete(1.0, tk.END)
            self.text_results.insert(tk.END, f"=== {os.path.basename(image_path)} ===\n")
            self.text_results.insert(tk.END, self.processed_results[image_path] + "\n")
    
    def process_folder(self):
        """Process all images in the selected folder"""
        if not self.selected_folder.get():
            messagebox.showerror("Error", "Please select a folder first")
            return
        
        if self.processing:
            return
        
        # Update settings from UI
        self.settings['show_model_results'] = self.show_model_var.get()
        self.settings['save_results'] = self.save_results_var.get()
        self.settings['demo_model_path'] = self.model_path_var.get()
        
        self.processing = True
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self._process_folder_thread)
        thread.daemon = True
        thread.start()
    
    def _process_folder_thread(self):
        """Thread function for processing folder"""
        try:
            folder_path = self.selected_folder.get()
            image_files = self.get_image_files(folder_path)
            
            if not image_files:
                self.status_var.set("No image files found in folder")
                return
            
            self.text_results.delete(1.0, tk.END)
            total_files = len(image_files)
            
            # Create output folder if saving results
            if self.settings['save_results']:
                os.makedirs(self.settings['output_folder'], exist_ok=True)
            
            for i, image_file in enumerate(image_files):
                if not self.processing:  # Check if stopped
                    break
                
                self.status_var.set(f"Processing {i+1}/{total_files}: {os.path.basename(image_file)}")
                self.progress_var.set((i / total_files) * 100)
                self.root.update()
                
                # Process single image
                result_text = self.process_single_image(image_file)
                
                # Store results
                self.processed_results[image_file] = result_text
                
                # Update results
                self.text_results.insert(tk.END, f"=== {os.path.basename(image_file)} ===\n")
                self.text_results.insert(tk.END, result_text + "\n\n")
                self.text_results.see(tk.END)
                self.root.update()
            
            self.progress_var.set(100)
            self.status_var.set("Processing completed")
            
            # Show first image after processing
            if self.image_files:
                self.current_image_index = 0
                self.update_navigation_buttons()
                self.show_current_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Processing failed")
        finally:
            self.processing = False
            self.process_button.config(state="normal")
            self.stop_button.config(state="disabled")
    
    def get_image_files(self, folder_path):
        """Get all image files from folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file in os.listdir(folder_path):
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(folder_path, file))
        
        return sorted(image_files)
    
    def process_single_image(self, image_path):
        """Process a single image with demo.py recognition"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return f"Error: Could not read image {image_path}"
            
            # Use demo.py for recognition
            recognized_text = self.recognize_text_with_demo(image_path)
            
            if recognized_text:
                return f"Recognized Text: '{recognized_text}'"
            else:
                return "No text recognized"
            
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    def recognize_text_with_demo(self, image_path):
        """Recognize text using demo.py"""
        try:
            if not os.path.exists(image_path):
                return ""
            
            # Get demo.py path (assuming it's in the same directory)
            demo_script = "demo.py"
            if not os.path.exists(demo_script):
                # Try to find demo.py in parent directories
                current_dir = os.getcwd()
                for root, dirs, files in os.walk(current_dir):
                    if "demo.py" in files:
                        demo_script = os.path.join(root, "demo.py")
                        break
            
            if not os.path.exists(demo_script):
                return "Demo script not found"
            
            # Create temp folder for single image
            temp_folder = "temp_single_image"
            os.makedirs(temp_folder, exist_ok=True)
            
            # Copy image to temp folder
            import shutil
            temp_image_path = os.path.join(temp_folder, os.path.basename(image_path))
            shutil.copy2(image_path, temp_image_path)
            
            # Run demo.py
            model_path = self.settings['demo_model_path']
            cmd = [
                "python", demo_script,
                "--Transformation", "TPS",
                "--FeatureExtraction", "ResNet",
                "--SequenceModeling", "BiLSTM",
                "--Prediction", "Attn",
                "--image_folder", temp_folder,
                "--saved_model", model_path,
                "--sensitive"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Clean up temp folder
            shutil.rmtree(temp_folder, ignore_errors=True)
            
            if result.returncode == 0:
                # Parse output to extract recognized text
                lines = result.stdout.split('\n')
                for line in lines:
                    if temp_image_path in line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            return parts[1].strip()
                return "No text recognized"
            else:
                return f"Demo error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Demo timeout"
        except Exception as e:
            return f"Demo error: {str(e)}"
    
    def update_image_preview(self, image_path):
        """Update the image preview"""
        try:
            # Load and resize image for preview
            image = cv2.imread(image_path)
            if image is not None:
                # Resize to fit preview area
                height, width = image.shape[:2]
                max_size = 300
                
                if height > max_size or width > max_size:
                    scale = min(max_size / height, max_size / width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                
                # Convert to PIL Image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update label
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
                
        except Exception as e:
            print(f"Error updating image preview: {e}")
    
    def save_image_results(self, image_path, result_text):
        """Save results for a single image"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save text results
            text_filename = f"{base_name}_results_{timestamp}.txt"
            text_path = os.path.join(self.settings['output_folder'], text_filename)
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"Image: {image_path}\n")
                f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")
                f.write(result_text)
                
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            settings_file = "demo_settings.json"
            with open(settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            messagebox.showinfo("Success", "Settings saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def stop_processing(self):
        """Stop the processing"""
        self.processing = False
        self.status_var.set("Processing stopped")
    
    def clear_results(self):
        """Clear the results text area"""
        self.text_results.delete(1.0, tk.END)
        self.status_var.set("Results cleared")

def main():
    root = tk.Tk()
    app = AdvancedDemoUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

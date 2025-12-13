#!/usr/bin/env python3
"""
Rotation Model Library
รับรูปจาก CRAFT มาแล้วหมุนให้ตรง แล้วส่งรูปที่หมุนแล้วไปใช้ต่อได้
ใช้หลักการเดียวกับ correct_rotation.py
"""

import os
import sys

# CRITICAL: Setup CUDA paths BEFORE importing PyTorch
# This must be done at module level before any torch imports
try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    # Fallback if cuda_setup is not available
    pass

# Now import PyTorch (after CUDA paths are set)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np
from typing import Dict, Optional, Union, Tuple
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib
from collections import OrderedDict

# Import CUDA image utilities
try:
    from core.cuda_image_utils import cuda_resize, cuda_cvtColor, cuda_gaussianBlur
    CUDA_IMAGE_UTILS_AVAILABLE = True
except ImportError:
    # Fallback if not available
    CUDA_IMAGE_UTILS_AVAILABLE = False
    def cuda_resize(img, size, **kwargs):
        return cv2.resize(img, size, **kwargs)
    def cuda_cvtColor(img, code):
        return cv2.cvtColor(img, code)
    def cuda_gaussianBlur(img, ksize, sigma):
        return cv2.GaussianBlur(img, ksize, sigma)

# Import CRAFT detector
try:
    from libs.processing.rotationCRAFT import CRAFTTextDetector, initialize_detector, get_global_detector
    CRAFT_AVAILABLE = True
except ImportError:
    # Fallback to relative import if absolute import fails
    try:
        from .rotationCRAFT import CRAFTTextDetector, initialize_detector, get_global_detector
        CRAFT_AVAILABLE = True
    except ImportError:
        CRAFT_AVAILABLE = False
        print("⚠️ CRAFT detector not available. Install rotationCRAFT.py for text detection features.")

class RotationModel:
    """
    Library สำหรับหมุนภาพให้ตรงโดยใช้ AI model
    รับรูปจาก CRAFT มาแล้วหมุนให้ตรง แล้วส่งรูปที่หมุนแล้วไปใช้ต่อได้
    """
    
    def __init__(self, model_path: str = None, cuda: bool = True, craft_model_path: str = None):
        """
        Initialize RotationModel
        
        Args:
            model_path: Path to rotation model (.pth file)
            cuda: Use CUDA if available
            craft_model_path: Path to CRAFT model for text detection
        """
        self.model = None
        # Use CUDA if available (CUDA paths already set at module level)
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        if self.cuda:
            print(f"🚀 PyTorch CUDA enabled - using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ PyTorch CUDA not available - using CPU for rotation model")
        
        # Model parameters
        self.IMAGE_SIZE = (224, 224)
        self.ANGLE_CLASSES = {
            0: -180,  # Class 0 -> Rotate by -180 degrees to correct
            1: -90,   # Class 1 -> Rotate by -90 degrees to correct
            2: 0,     # Class 2 -> No rotation needed
            3: 90,    # Class 3 -> Rotate by 90 degrees to correct
            4: 180    # Class 4 -> Rotate by 180 degrees to correct
        }
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Global variables for storing results
        self.last_rotation_result = None
        self.last_processed_image = None
        self.last_rotated_image = None
        self.rotation_history = []
        
        # Cache for storing rotation results (for repeated images)
        self.cache = OrderedDict()  # {image_hash: result}
        self.cache_max_size = 50  # Maximum cache size
        
        # CRAFT detector
        self.craft_detector = None
        if CRAFT_AVAILABLE and craft_model_path:
            self.init_craft_detector(craft_model_path)
        
        if model_path:
            self.load_model(model_path)
    
    def init_craft_detector(self, craft_model_path: str, craft_refiner_path: str = None) -> bool:
        """
        Initialize CRAFT text detector
        
        Args:
            craft_model_path: Path to CRAFT model
            craft_refiner_path: Path to CRAFT refiner model (optional)
            
        Returns:
            bool: True if initialized successfully
        """
        try:
            if not CRAFT_AVAILABLE:
                print("❌ CRAFT detector not available")
                return False
            
            self.craft_detector = CRAFTTextDetector(craft_model_path, craft_refiner_path, self.cuda)
            print("✅ CRAFT detector initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing CRAFT detector: {e}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """
        Load rotation model from path
        
        Args:
            model_path: Path to .pth model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"Loading rotation model from {model_path}...")
            print("Using torchvision's EfficientNet-B0 architecture.")
            
            # Load the EfficientNet-B0 architecture without pre-trained weights
            model = models.efficientnet_b0(weights=None)

            # The original classifier in EfficientNet-B0 outputs 1000 classes.
            # We adapt the final layer for our 5-class task.
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 5)
            
            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            self.model = model
            print("✅ Rotation model loaded successfully.")
            return True
            
        except Exception as e:
            print(f"❌ Error loading rotation model: {e}")
            return False
    
    def rotate_image(self, image_path: str = None, image_array: np.ndarray = None) -> Dict:
        """
        Rotate image to correct orientation
        
        Args:
            image_path: Path to image file
            image_array: Image as numpy array (if image_path is None)
            
        Returns:
            Dict containing rotation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load model first.")
        
        try:
            # Load image
            if image_path:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                # Load with PIL for preprocessing
                pil_image = Image.open(image_path).convert("RGB")
                # Convert to numpy for storage
                image = np.array(pil_image)
            elif image_array is not None:
                # Convert numpy array to PIL
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    # Convert BGR to RGB if needed
                    if image_array.dtype == np.uint8:
                        image_rgb = cuda_cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    else:
                        image_rgb = image_array
                else:
                    image_rgb = image_array
                pil_image = Image.fromarray(image_rgb)
                image = image_rgb
            else:
                raise ValueError("Either image_path or image_array must be provided")
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Store original image
            self.last_processed_image = image.copy()
            
            # Preprocess for model
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Predict rotation class
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_angle = self.ANGLE_CLASSES[predicted_class.item()]
            confidence_score = confidence.item()
            
            # Rotate image if necessary
            rotated_image = None
            if predicted_angle != 0:
                # Convert RGB to BGR for OpenCV rotation
                image_bgr = cuda_cvtColor(image, cv2.COLOR_RGB2BGR)
                (h, w) = image_bgr.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, predicted_angle, 1.0)
                rotated_bgr = cv2.warpAffine(image_bgr, M, (w, h),
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                # Convert back to RGB
                rotated_image = cuda_cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
                self.last_rotated_image = rotated_image.copy()
            else:
                # No rotation needed
                rotated_image = image.copy()
                self.last_rotated_image = rotated_image.copy()
            
            # Create result dictionary
            result = {
                'image_path': image_path,
                'original_image': image,
                'rotated_image': rotated_image,
                'predicted_class': predicted_class.item(),
                'rotation_angle': predicted_angle,
                'confidence_score': confidence_score,
                'image_shape': image.shape,
                'timestamp': str(datetime.now())
            }
            
            # Store in global variables
            self.last_rotation_result = result
            self.rotation_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"Error during rotation: {e}")
            return {
                'error': str(e),
                'rotation_angle': 0,
                'rotated_image': None
            }
    
    def rotate_image_with_verification(self, image_path: str = None, image_array: np.ndarray = None) -> Dict:
        """
        Rotate image and verify if it's correctly oriented, rotate again if needed
        
        Args:
            image_path: Path to image file
            image_array: Image as numpy array (if image_path is None)
            
        Returns:
            Dict containing rotation results with verification
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load model first.")
        
        try:
            print("🔄 Starting rotation with verification...")
            
            # First rotation
            first_result = self.rotate_image(image_path=image_path, image_array=image_array)
            
            if 'error' in first_result:
                print(f"❌ First rotation failed: {first_result['error']}")
                return first_result
            
            first_angle = first_result['rotation_angle']
            first_rotated_image = first_result['rotated_image']
            
            print(f"✅ First rotation applied: {first_angle} degrees")
            
            # Check if first rotation was successful (predicted class should be 2 for no rotation needed)
            if first_result['predicted_class'] == 2:
                print("✅ Image is correctly oriented after first rotation")
                result = first_result.copy()
                result['verification_passed'] = True
                result['total_rotations'] = 1
                result['verification_angle'] = 0
                return result
            
            # If first rotation didn't result in class 2, try one more rotation
            print("🔄 First rotation didn't achieve correct orientation, trying second rotation...")
            
            # Use the rotated image for second prediction
            pil_rotated = Image.fromarray(first_rotated_image)
            input_tensor = self.transform(pil_rotated).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            second_angle = self.ANGLE_CLASSES[predicted_class.item()]
            second_confidence_score = confidence.item()
            
            print(f"🔍 Second prediction: class {predicted_class.item()} -> angle {second_angle} degrees")
            
            # Apply second rotation if needed
            final_rotated_image = first_rotated_image
            if second_angle != 0:
                # Convert RGB to BGR for OpenCV rotation
                image_bgr = cv2.cvtColor(first_rotated_image, cv2.COLOR_RGB2BGR)
                (h, w) = image_bgr.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, second_angle, 1.0)
                rotated_bgr = cv2.warpAffine(image_bgr, M, (w, h),
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                # Convert back to RGB
                final_rotated_image = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
                print(f"✅ Second rotation applied: {second_angle} degrees")
            else:
                print("✅ No second rotation needed")
            
            # Create comprehensive result
            result = {
                'image_path': image_path,
                'original_image': first_result['original_image'],
                'rotated_image': final_rotated_image,
                'first_predicted_class': first_result['predicted_class'],
                'first_rotation_angle': first_angle,
                'first_confidence_score': first_result.get('confidence_score', 0.0),
                'second_predicted_class': predicted_class.item(),
                'second_rotation_angle': second_angle,
                'second_confidence_score': second_confidence_score,
                'total_rotation_angle': first_angle + second_angle,
                'verification_passed': predicted_class.item() == 2,
                'total_rotations': 2,
                'verification_angle': second_angle,
                'image_shape': first_result['image_shape'],
                'timestamp': str(datetime.now())
            }
            
            # Store in global variables
            self.last_rotation_result = result
            self.last_processed_image = first_result['original_image']
            self.last_rotated_image = final_rotated_image
            self.rotation_history.append(result)
            
            if result['verification_passed']:
                print("✅ Verification passed: Image is correctly oriented")
            else:
                print("⚠️ Verification failed: Image may still need rotation")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during rotation with verification: {e}")
            return {
                'error': str(e),
                'rotation_angle': 0,
                'rotated_image': None,
                'verification_passed': False
            }
    
    def validate_ocr_format(self, ocr_results: Dict) -> bool:
        """
        Validate if OCR results match expected format:
        - MFG02/06/25 (manufacturing date)
        - BBF20/06/25 (best before date) 
        - 09:10S02 (time and batch code)
        
        Args:
            ocr_results: OCR results from recognize_text_from_craft_lines
            
        Returns:
            bool: True if format matches expected pattern
        """
        try:
            if not isinstance(ocr_results, dict) or 'line_results' not in ocr_results:
                return False
            
            line_results = ocr_results['line_results']
            if not line_results:
                return False
            
            # Check if any line matches expected patterns
            for line_result in line_results:
                text = line_result.get('recognized_text', '').strip()
                if not text:
                    continue
                
                # Pattern 1: MFG/BBF date format (MM/DD/YY or DD/MM/YY)
                if any(prefix in text.upper() for prefix in ['MFG', 'BBF']):
                    if '/' in text and len(text.split('/')) >= 3:
                        return True
                
                # Pattern 2: Time and batch format (HH:MMS##)
                if ':' in text and 'S' in text:
                    parts = text.split(':')
                    if len(parts) == 2 and 'S' in parts[1]:
                        return True
                
                # Pattern 3: Date format (DD/MM/YY or MM/DD/YY)
                if '/' in text and len(text.split('/')) == 3:
                    date_parts = text.split('/')
                    if all(part.isdigit() for part in date_parts):
                        return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error validating OCR format: {e}")
            return False
    
    def is_first_line_time_format(self, ocr_results: Dict) -> bool:
        """
        Check if the first line is in time format (HH:MMS##) instead of MFG date format
        
        Args:
            ocr_results: OCR results from recognize_text_from_craft_lines
            
        Returns:
            bool: True if first line is time format, False if MFG format
        """
        try:
            if not isinstance(ocr_results, dict) or 'line_results' not in ocr_results:
                return False
            
            line_results = ocr_results['line_results']
            if not line_results:
                return False
            
            # Check first line only
            first_line = line_results[0]
            text = first_line.get('recognized_text', '').strip()
            if not text:
                return False
            
            # Check if first line is time format (HH:MMS##)
            if ':' in text and 'S' in text:
                parts = text.split(':')
                if len(parts) == 2 and 'S' in parts[1]:
                    print(f"🔄 TIME FORMAT DETECTED: First line is time format: {text}")
                    return True
            
            # Check if first line is MFG format
            if any(prefix in text.upper() for prefix in ['MFG', 'BBF']):
                print(f"🔄 MFG FORMAT DETECTED: First line is MFG format: {text}")
                return False
            
            return False
            
        except Exception as e:
            print(f"❌ Error checking first line format: {e}")
            return False
    
    def has_mfg_in_first_line(self, ocr_results: Dict) -> bool:
        """
        Check if the first line contains MFG or BBF prefix
        
        Args:
            ocr_results: OCR results from recognize_text_from_craft_lines
            
        Returns:
            bool: True if first line contains MFG/BBF, False otherwise
        """
        try:
            if not isinstance(ocr_results, dict) or 'line_results' not in ocr_results:
                return False
            
            line_results = ocr_results['line_results']
            if not line_results:
                return False
            
            # Check first line only
            first_line = line_results[0]
            text = first_line.get('recognized_text', '').strip()
            if not text:
                return False
            
            # Check if first line contains MFG or BBF
            has_mfg = any(prefix in text.upper() for prefix in ['MFG', 'BBF'])
            if has_mfg:
                print(f"✅ MFG FOUND: First line contains MFG/BBF: '{text}'")
            else:
                print(f"❌ NO MFG: First line does not contain MFG/BBF: '{text}'")
            
            return has_mfg
            
        except Exception as e:
            print(f"❌ Error checking MFG in first line: {e}")
            return False
    
    def _test_single_rotation(self, angle: int, craft_rotated_image: np.ndarray, line_detector, ocr_model, rotation_callback=None, attempt_num: int = 0) -> Optional[Dict]:
        """
        Test a single rotation angle (helper function for parallel processing)
        
        Args:
            angle: Rotation angle to test
            craft_rotated_image: Original CRAFT rotated image
            line_detector: Line detection model
            ocr_model: OCR model
            rotation_callback: Callback function for progress updates
            attempt_num: Attempt number for callback
            
        Returns:
            Dict with result or None if failed
        """
        try:
            # Rotate CRAFT image by the specified angle
            if angle == 0:
                rotated_image = craft_rotated_image.copy()
            else:
                h, w = craft_rotated_image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # For 90° and 270° rotations, we need to adjust the image size
                if angle in [90, 270, -90]:
                    # Swap width and height for 90° rotations
                    rotated_image = cv2.warpAffine(craft_rotated_image, rotation_matrix, (h, w))
                else:
                    # Keep original size for 180° rotation
                    rotated_image = cv2.warpAffine(craft_rotated_image, rotation_matrix, (w, h))
            
            # Callback for rotation attempt
            if rotation_callback:
                rotation_callback(attempt_num, 5, rotated_image, None, None)
            
            # Detect lines directly from rotated CRAFT image
            line_detection_result = line_detector.detect_lines_from_rotation_result({
                'rotated_image': rotated_image,
                'cropped_image': rotated_image
            })
            
            if line_detection_result and 'error' not in line_detection_result:
                # Perform OCR on rotated CRAFT image
                ocr_results = ocr_model.recognize_text_from_craft_lines(line_detection_result)
                
                # Check MFG first (skip format validation if no MFG)
                has_mfg = self.has_mfg_in_first_line(ocr_results)
                
                if not has_mfg:
                    if rotation_callback:
                        rotation_callback(attempt_num, 5, rotated_image, ocr_results, False)
                    return None  # Skip this rotation
                
                # Validate OCR format (only when MFG exists)
                format_valid = self.validate_ocr_format(ocr_results)
                
                # Calculate score based on format validity and confidence
                score = 0
                if format_valid:
                    score += 100  # Base score for valid format
                    
                    # Add confidence bonus
                    if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                        for line_result in ocr_results['line_results']:
                            confidence = line_result.get('confidence_score', 0)
                            score += confidence * 10
                
                # Return result
                result = {
                    'rotation_angle': angle,
                    'rotated_image': rotated_image,
                    'cropped_image': rotated_image,
                    'line_detection_result': line_detection_result,
                    'ocr_results': ocr_results,
                    'format_valid': format_valid,
                    'has_mfg': has_mfg,
                    'score': score,
                    'rotation_attempt': attempt_num,
                    'used_craft_image': True
                }
                
                # Callback for OCR results
                if rotation_callback:
                    rotation_callback(attempt_num, 5, rotated_image, ocr_results, format_valid)
                
                return result
            else:
                # Line detection failed
                if rotation_callback:
                    rotation_callback(attempt_num, 5, rotated_image, None, False)
                return None
                
        except Exception as e:
            # Skip this rotation on error
            return None
    
    def _image_hash(self, image: np.ndarray) -> str:
        """
        Generate hash from image for cache key
        
        Args:
            image: Image array
            
        Returns:
            str: MD5 hash of image
        """
        try:
            # Use image bytes for hashing
            return hashlib.md5(image.tobytes()).hexdigest()
        except Exception as e:
            # Fallback: use shape and mean for hash
            return hashlib.md5(f"{image.shape}_{image.mean()}".encode()).hexdigest()
    
    def process_craft_rotated_image_with_retry(self, craft_rotated_image: np.ndarray, line_detector, ocr_model, rotation_callback=None, use_parallel: bool = True, use_cache: bool = True) -> Dict:
        """
        Process CRAFT rotated image with 5-rotation retry mechanism for better OCR results
        ทำแค่ Line Detection → OCR เท่านั้น (ไม่ทำ CRAFT ซ้ำ)
        ตรวจสอบ: ถ้า OCR อ่านไม่ได้ MFG ในบรรทัดแรก → ไม่เอาการหมุนนั้น แล้วไปลองอันอื่นเลย
        
        Args:
            craft_rotated_image: CRAFT rotated image (numpy array)
            line_detector: Line detection model
            ocr_model: OCR model
            
        Returns:
            Dict containing the best rotation result
        """
        # Optimized: ตรวจสอบ cache ก่อน (ถ้าเปิดใช้งาน)
        if use_cache:
            img_hash = self._image_hash(craft_rotated_image)
            if img_hash in self.cache:
                cached_result = self.cache[img_hash]
                print(f"✅ CACHE HIT: Found cached result for image hash {img_hash[:8]}...")
                # Move to end (LRU - Least Recently Used)
                self.cache.move_to_end(img_hash)
                return cached_result
            else:
                print(f"🔄 CACHE MISS: Processing new image (hash: {img_hash[:8]}...)")
        
        # Optimized: ใช้ Parallel Processing สำหรับ Jetson AGX Xavier
        if use_parallel:
            print("🚀 ROTATION RETRY: Starting parallel processing (5 rotations simultaneously)...")
            result = self._process_with_parallel(craft_rotated_image, line_detector, ocr_model, rotation_callback)
        else:
            # Sequential processing (fallback)
            print("🔄 ROTATION RETRY: Starting sequential processing (5 rotations one by one)...")
            
            best_result = None
            best_score = 0
            best_rotation = 0
            
            # Optimized: เรียงลำดับการหมุน - ลอง 0° ก่อน (ภาพอาจถูกต้องอยู่แล้ว)
            # ลำดับ: 0°, 90°, -90°, 180°, 270° (ลองมุมที่ใช้บ่อยก่อน)
            rotation_angles = [0, 90, -90, 180, 270]
            
            for i, angle in enumerate(rotation_angles):
                rotation_result = self._test_single_rotation(angle, craft_rotated_image, line_detector, ocr_model, rotation_callback, i + 1)
                
                if rotation_result:
                    # Update best result if this is better
                    if rotation_result['score'] > best_score:
                        best_score = rotation_result['score']
                        best_rotation = angle
                        best_result = rotation_result
                    
                    # Early Exit - หยุดทันทีถ้าเจอ MFG และ format valid แล้ว
                    if rotation_result.get('has_mfg', False) and rotation_result.get('format_valid', False):
                        print(f"✅ ROTATION RETRY: Found valid MFG at {angle}° (attempt {i+1}/5) - Stopping")
                        result = best_result
                        break
            
            # Return best result found
            if best_result:
                print(f"✅ ROTATION RETRY: Best result at {best_rotation}° (score: {best_score:.1f})")
                result = best_result
            else:
                print("❌ ROTATION RETRY: No valid results found")
                result = {
                    'error': 'No valid OCR results found in any rotation',
                    'rotation_angle': 0,
                    'rotated_image': None,
                    'cropped_image': None,
                    'format_valid': False,
                    'score': 0
                }
        
        # Optimized: เก็บผลลัพธ์ไว้ใน cache (ถ้าเปิดใช้งาน)
        if use_cache and result and 'error' not in result:
            img_hash = self._image_hash(craft_rotated_image)
            
            # ลบ cache เก่าถ้าเต็ม (LRU - Least Recently Used)
            if len(self.cache) >= self.cache_max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                print(f"🗑️ CACHE: Removed oldest entry (cache size: {len(self.cache)})")
            
            # เก็บผลลัพธ์ไว้ใน cache
            self.cache[img_hash] = result
            print(f"💾 CACHE: Stored result for image hash {img_hash[:8]}... (cache size: {len(self.cache)})")
        
        return result
    
    def _process_with_parallel(self, craft_rotated_image: np.ndarray, line_detector, ocr_model, rotation_callback=None) -> Dict:
        """
        Process rotations in parallel using ThreadPoolExecutor
        Optimized for Jetson AGX Xavier with CUDA support
        
        Note: PyTorch models release GIL during GPU operations, so parallel processing
        works well even with threading. CUDA can handle multiple tasks via streams.
        
        Args:
            craft_rotated_image: CRAFT rotated image (numpy array)
            line_detector: Line detection model (thread-safe for CUDA operations)
            ocr_model: OCR model (thread-safe for CUDA operations)
            rotation_callback: Callback function for progress updates
            
        Returns:
            Dict containing the best rotation result
        """
        # Optimized: เรียงลำดับการหมุน - ลอง 0° ก่อน
        rotation_angles = [0, 90, -90, 180, 270]
        
        best_result = None
        best_score = 0
        best_rotation = 0
        results_lock = threading.Lock()  # Thread-safe lock for results
        
        # Check CUDA availability using self.cuda (already checked in __init__)
        # Use self.cuda instead of torch.cuda.is_available() to ensure consistency
        if self.cuda:
            print(f"🚀 PARALLEL: CUDA available - GPU: {torch.cuda.get_device_name(0)}")
            # Jetson AGX Xavier can handle multiple CUDA streams
            # Threading works well because PyTorch releases GIL during GPU ops
            max_workers = min(5, len(rotation_angles))  # Test up to 5 angles simultaneously
        else:
            print("⚠️ PARALLEL: CUDA not available - using CPU")
            # CPU: limit workers to avoid overload
            max_workers = min(3, len(rotation_angles))  # Test up to 3 angles simultaneously
        
        print(f"🚀 PARALLEL: Testing {len(rotation_angles)} rotations with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all rotation tests
            future_to_angle = {
                executor.submit(
                    self._test_single_rotation, 
                    angle, 
                    craft_rotated_image, 
                    line_detector, 
                    ocr_model, 
                    rotation_callback,
                    i + 1
                ): angle
                for i, angle in enumerate(rotation_angles)
            }
            
            # Process results as they complete (early exit when found valid result)
            for future in as_completed(future_to_angle):
                angle = future_to_angle[future]
                try:
                    result = future.result()
                    
                    if result:
                        # Thread-safe update
                        with results_lock:
                            # Early Exit - หยุดทันทีถ้าเจอ MFG และ format valid
                            if result.get('has_mfg', False) and result.get('format_valid', False):
                                print(f"✅ PARALLEL: Found valid MFG at {angle}° - Stopping all other tasks")
                                # Cancel remaining tasks (best effort)
                                for f in future_to_angle:
                                    if f != future and not f.done():
                                        f.cancel()
                                return result
                            
                            # Update best result if this is better
                            if result['score'] > best_score:
                                best_score = result['score']
                                best_rotation = angle
                                best_result = result
                                
                except Exception as e:
                    # Skip this rotation on error
                    continue
        
        # Return best result found
        if best_result:
            print(f"✅ PARALLEL: Best result at {best_rotation}° (score: {best_score:.1f})")
            return best_result
        else:
            print("❌ PARALLEL: No valid results found")
            return {
                'error': 'No valid OCR results found in any rotation',
                'rotation_angle': 0,
                'rotated_image': None,
                'cropped_image': None,
                'format_valid': False,
                'score': 0
            }
    
    def process_craft_rotated_image_with_models(self, craft_rotated_image: np.ndarray, line_detector, ocr_model) -> Dict:
        """
        Process CRAFT rotated image with external line detector and OCR model
        ใช้ retry mechanism หมุน 5 ครั้งเลย (ไม่ใช้ AI model ทำนาย)
        
        Args:
            craft_rotated_image: CRAFT rotated image (numpy array)
            line_detector: External line detector instance
            ocr_model: External OCR model instance
            
        Returns:
            Dict containing rotation, text detection, and cropping results
        """
        try:
            # Optimized: ใช้ retry mechanism หมุน 5 ครั้งเลย (ไม่ใช้ AI model)
            retry_result = self.process_craft_rotated_image_with_retry(
                craft_rotated_image, line_detector, ocr_model
            )
            
            # Return retry result
            if retry_result and 'error' not in retry_result:
                return retry_result
            else:
                return retry_result
                
        except Exception as e:
            print(f"❌ Error in process_craft_rotated_image_with_models: {e}")
            return {
                'error': f'Error processing with external models: {e}',
                'rotated_image': None,
                'cropped_image': None,
                'line_detection_result': None,
                'ocr_results': None
            }
    
    def process_craft_rotated_image(self, craft_rotated_image: np.ndarray) -> Dict:
        """
        รับรูปที่หมุนแล้วจาก CRAFT มาแล้วหมุนด้วย AI อีกที ใช้ CRAFT ตรวจจับข้อความ แล้วครอป
        
        Args:
            craft_rotated_image: รูปที่หมุนแล้วจาก CRAFT (numpy array)
            
        Returns:
            Dict containing AI rotation, text detection, and cropping results
        """
        if self.model is None:
            raise ValueError("AI rotation model not loaded. Please load model first.")
        if self.craft_detector is None:
            raise ValueError("CRAFT detector not initialized. Call init_craft_detector() first.")
        
        try:
            print("🔄 Processing CRAFT rotated image with AI rotation...")
            
            # Step 1: หมุนรูปที่ CRAFT หมุนแล้วด้วย AI model
            ai_rotation_result = self.rotate_image(image_array=craft_rotated_image)
            
            if 'error' in ai_rotation_result:
                print(f"❌ AI rotation failed: {ai_rotation_result['error']}")
                return ai_rotation_result
            
            # ได้รูปที่หมุนด้วย AI แล้ว
            ai_rotated_image = ai_rotation_result['rotated_image']
            ai_rotation_angle = ai_rotation_result['rotation_angle']
            ai_confidence_score = ai_rotation_result.get('confidence_score', 0.0)
            
            print(f"✅ AI rotation applied: {ai_rotation_angle} degrees")
            
            # Step 2: ใช้ CRAFT ตรวจจับข้อความในรูปที่หมุนด้วย AI แล้ว
            print("🔍 Detecting text with CRAFT...")
            detection_result = self.craft_detector.detect_text_and_rotate(image_array=ai_rotated_image)
            
            if 'error' in detection_result:
                print(f"❌ CRAFT detection failed: {detection_result['error']}")
                return detection_result
            
            # Step 3: คำนวณการครอปเฉพาะส่วนที่มีข้อความ
            print("✂️ Calculating unified crop...")
            cropped_image, crop_box = self._calculate_unified_crop(ai_rotated_image, detection_result)
            
            # Step 4: รวมผลลัพธ์
            combined_result = {
                'craft_rotated_image': craft_rotated_image,  # รูปที่ CRAFT หมุนแล้ว
                'ai_rotated_image': ai_rotated_image,        # รูปที่ AI หมุนแล้ว
                'cropped_image': cropped_image,              # รูปที่ครอปแล้ว
                'ai_rotation_angle': ai_rotation_angle,      # มุมที่ AI หมุน
                'ai_confidence_score': ai_confidence_score,  # ความเชื่อมั่นของ AI
                'ai_predicted_class': ai_rotation_result['predicted_class'],
                'text_boxes': detection_result.get('text_boxes', []),
                'text_polys': detection_result.get('text_polys', []),
                'total_text_regions': detection_result.get('total_text_regions', 0),
                'crop_box': crop_box,
                'crop_success': cropped_image is not None,
                'image_shape': craft_rotated_image.shape,
                'timestamp': str(datetime.now())
            }
            
            # เก็บผลลัพธ์ล่าสุด
            self.last_rotation_result = combined_result
            self.last_processed_image = craft_rotated_image
            self.last_rotated_image = ai_rotated_image
            self.rotation_history.append(combined_result)
            
            return combined_result
            
        except Exception as e:
            print(f"❌ Error in process_craft_rotated_image: {e}")
            return {
                'error': str(e),
                'ai_rotation_angle': 0,
                'ai_rotated_image': None,
                'cropped_image': None
            }
    
    def process_craft_rotated_image_with_verification(self, craft_rotated_image: np.ndarray) -> Dict:
        """
        รับรูปที่หมุนแล้วจาก CRAFT มาแล้วหมุนด้วย AI พร้อม verification ใช้ CRAFT ตรวจจับข้อความ แล้วครอป
        
        Args:
            craft_rotated_image: รูปที่หมุนแล้วจาก CRAFT (numpy array)
            
        Returns:
            Dict containing AI rotation with verification, text detection, and cropping results
        """
        if self.model is None:
            raise ValueError("AI rotation model not loaded. Please load model first.")
        if self.craft_detector is None:
            raise ValueError("CRAFT detector not initialized. Call init_craft_detector() first.")
        
        try:
            print("🔄 Processing CRAFT rotated image with AI rotation and verification...")
            
            # Step 1: หมุนรูปที่ CRAFT หมุนแล้วด้วย AI model พร้อม verification
            ai_rotation_result = self.rotate_image_with_verification(image_array=craft_rotated_image)
            
            if 'error' in ai_rotation_result:
                print(f"❌ AI rotation with verification failed: {ai_rotation_result['error']}")
                return ai_rotation_result
            
            # ได้รูปที่หมุนด้วย AI แล้ว
            ai_rotated_image = ai_rotation_result['rotated_image']
            ai_rotation_angle = ai_rotation_result.get('total_rotation_angle', ai_rotation_result['rotation_angle'])
            ai_confidence_score = ai_rotation_result.get('second_confidence_score', ai_rotation_result.get('confidence_score', 0.0))
            
            print(f"✅ AI rotation with verification applied: {ai_rotation_angle} degrees total")
            print(f"   Verification passed: {ai_rotation_result.get('verification_passed', False)}")
            
            # Step 2: ใช้ CRAFT ตรวจจับข้อความในรูปที่หมุนด้วย AI แล้ว
            print("🔍 Detecting text with CRAFT...")
            detection_result = self.craft_detector.detect_text_and_rotate(image_array=ai_rotated_image)
            
            if 'error' in detection_result:
                print(f"❌ CRAFT detection failed: {detection_result['error']}")
                return detection_result
            
            # Step 3: คำนวณการครอปเฉพาะส่วนที่มีข้อความ
            print("✂️ Calculating unified crop...")
            cropped_image, crop_box = self._calculate_unified_crop(ai_rotated_image, detection_result)
            
            # Step 4: รวมผลลัพธ์
            combined_result = {
                'craft_rotated_image': craft_rotated_image,  # รูปที่ CRAFT หมุนแล้ว
                'ai_rotated_image': ai_rotated_image,        # รูปที่ AI หมุนแล้ว
                'cropped_image': cropped_image,              # รูปที่ครอปแล้ว
                'ai_rotation_angle': ai_rotation_angle,      # มุมที่ AI หมุนรวม
                'ai_confidence_score': ai_confidence_score,  # ความเชื่อมั่นของ AI
                'ai_predicted_class': ai_rotation_result.get('second_predicted_class', ai_rotation_result['predicted_class']),
                'verification_passed': ai_rotation_result.get('verification_passed', False),
                'total_rotations': ai_rotation_result.get('total_rotations', 1),
                'first_rotation_angle': ai_rotation_result.get('first_rotation_angle', ai_rotation_angle),
                'second_rotation_angle': ai_rotation_result.get('second_rotation_angle', 0),
                'first_confidence_score': ai_rotation_result.get('first_confidence_score', 0.0),
                'second_confidence_score': ai_rotation_result.get('second_confidence_score', 0.0),
                'text_boxes': detection_result.get('text_boxes', []),
                'text_polys': detection_result.get('text_polys', []),
                'total_text_regions': detection_result.get('total_text_regions', 0),
                'crop_box': crop_box,
                'crop_success': cropped_image is not None,
                'image_shape': craft_rotated_image.shape,
                'timestamp': str(datetime.now())
            }
            
            # เก็บผลลัพธ์ล่าสุด
            self.last_rotation_result = combined_result
            self.last_processed_image = craft_rotated_image
            self.last_rotated_image = ai_rotated_image
            self.rotation_history.append(combined_result)
            
            return combined_result
            
        except Exception as e:
            print(f"❌ Error in process_craft_rotated_image_with_verification: {e}")
            return {
                'error': str(e),
                'ai_rotation_angle': 0,
                'ai_rotated_image': None,
                'cropped_image': None,
                'verification_passed': False
            }
    
    def process_craft_rotated_from_path(self, craft_rotated_image_path: str) -> Dict:
        """
        รับรูปที่หมุนแล้วจาก CRAFT จากไฟล์ มาแล้วหมุนด้วย AI อีกที
        
        Args:
            craft_rotated_image_path: Path ไปยังรูปที่ CRAFT หมุนแล้ว
            
        Returns:
            Dict containing AI rotation, text detection, and cropping results
        """
        try:
            # โหลดรูปที่ CRAFT หมุนแล้ว
            if not os.path.exists(craft_rotated_image_path):
                raise FileNotFoundError(f"CRAFT rotated image not found: {craft_rotated_image_path}")
            
            # โหลดรูปด้วย PIL แล้วแปลงเป็น numpy array
            pil_image = Image.open(craft_rotated_image_path).convert("RGB")
            craft_rotated_image = np.array(pil_image)
            
            print(f"📸 Loaded CRAFT rotated image: {craft_rotated_image_path}")
            print(f"   Image shape: {craft_rotated_image.shape}")
            
            # ประมวลผล
            return self.process_craft_rotated_image(craft_rotated_image)
            
        except Exception as e:
            print(f"❌ Error loading CRAFT rotated image: {e}")
            return {
                'error': str(e),
                'ai_rotation_angle': 0,
                'ai_rotated_image': None,
                'cropped_image': None
            }
    
    def process_craft_rotated_image_with_verification_from_path(self, craft_rotated_image_path: str) -> Dict:
        """
        รับรูปที่หมุนแล้วจาก CRAFT จากไฟล์ มาแล้วหมุนด้วย AI พร้อม verification
        
        Args:
            craft_rotated_image_path: Path ไปยังรูปที่ CRAFT หมุนแล้ว
            
        Returns:
            Dict containing AI rotation with verification, text detection, and cropping results
        """
        try:
            # โหลดรูปที่ CRAFT หมุนแล้ว
            if not os.path.exists(craft_rotated_image_path):
                raise FileNotFoundError(f"CRAFT rotated image not found: {craft_rotated_image_path}")
            
            # โหลดรูปด้วย PIL แล้วแปลงเป็น numpy array
            pil_image = Image.open(craft_rotated_image_path).convert("RGB")
            craft_rotated_image = np.array(pil_image)
            
            print(f"📸 Loaded CRAFT rotated image: {craft_rotated_image_path}")
            print(f"   Image shape: {craft_rotated_image.shape}")
            
            # ประมวลผลด้วย verification
            return self.process_craft_rotated_image_with_verification(craft_rotated_image)
            
        except Exception as e:
            print(f"❌ Error loading CRAFT rotated image: {e}")
            return {
                'error': str(e),
                'ai_rotation_angle': 0,
                'ai_rotated_image': None,
                'cropped_image': None,
                'verification_passed': False
            }
    
    def _calculate_unified_crop(self, image: np.ndarray, detection_result: Dict) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """
        Calculate unified bounding box from all detected text regions
        
        Args:
            image: Input image
            detection_result: Detection result from CRAFT
            
        Returns:
            Tuple of (cropped_image, crop_box)
        """
        try:
            if 'text_boxes' not in detection_result or detection_result['text_boxes'] is None or len(detection_result['text_boxes']) == 0:
                print("⚠️ No text regions detected")
                return None, None
            
            # Get all text boxes
            text_boxes = detection_result['text_boxes']
            
            # Calculate bounding box that encompasses all text regions
            all_points = np.concatenate(text_boxes, axis=0)
            
            # Get min/max coordinates
            min_x = int(np.min(all_points[:, 0]))
            max_x = int(np.max(all_points[:, 0]))
            min_y = int(np.min(all_points[:, 1]))
            max_y = int(np.max(all_points[:, 1]))
            
            # Add padding (10% of box size)
            width = max_x - min_x
            height = max_y - min_y
            padding_x = int(width * 0.1)
            padding_y = int(height * 0.1)
            
            # Apply padding with image boundary checks
            min_x = max(0, min_x - padding_x)
            max_x = min(image.shape[1], max_x + padding_x)
            min_y = max(0, min_y - padding_y)
            max_y = min(image.shape[0], max_y + padding_y)
            
            # Crop image
            cropped_image = image[min_y:max_y, min_x:max_x]
            
            crop_box = [min_x, min_y, max_x, max_y]
            
            print(f"✅ Cropped image from {image.shape} to {cropped_image.shape}")
            print(f"   Crop box: {crop_box}")
            
            return cropped_image, crop_box
            
        except Exception as e:
            print(f"Error calculating unified crop: {e}")
            return None, None
    
    def save_cropped_result(self, output_path: str, result: Dict = None) -> bool:
        """
        Save cropped image from combined rotation and detection result
        
        Args:
            output_path: Path to save the cropped image
            result: Combined result (if None, uses last result)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if result is None:
                # Try to get from last rotation result
                result = self.last_rotation_result
            
            if result and 'cropped_image' in result and result['cropped_image'] is not None:
                # Convert RGB to BGR for saving
                cropped_bgr = cuda_cvtColor(result['cropped_image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, cropped_bgr)
                
                # Save metadata
                json_path = output_path.replace('.jpg', '.json').replace('.png', '.json')
                json_result = result.copy()
                
                # Remove image data from JSON
                for key in ['original_image', 'rotated_image', 'cropped_image']:
                    if key in json_result:
                        del json_result[key]
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Cropped image saved to: {output_path}")
                return True
            else:
                print("⚠️ No cropped image available")
                return False
                
        except Exception as e:
            print(f"❌ Error saving cropped image: {e}")
            return False
    
    def save_rotated_image(self, output_path: str, result: Dict = None) -> bool:
        """
        Save rotated image
        
        Args:
            output_path: Path to save the rotated image
            result: Rotation result (if None, uses last rotation result)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if result is None:
                result = self.last_rotation_result
            
            if result and 'rotated_image' in result and result['rotated_image'] is not None:
                # Convert RGB to BGR for saving
                rotated_bgr = cuda_cvtColor(result['rotated_image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, rotated_bgr)
                print(f"✅ Rotated image saved to: {output_path}")
                return True
            else:
                print("⚠️ No rotated image available")
                return False
                
        except Exception as e:
            print(f"❌ Error saving rotated image: {e}")
            return False
    
    def save_rotation_result(self, output_path: str, result: Dict = None) -> bool:
        """
        Save rotation result with metadata
        
        Args:
            output_path: Path to save the result image
            result: Rotation result (if None, uses last rotation result)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if result is None:
                result = self.last_rotation_result
            
            if result and 'rotated_image' in result and result['rotated_image'] is not None:
                # Save rotated image
                rotated_bgr = cuda_cvtColor(result['rotated_image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, rotated_bgr)
                
                # Save metadata as JSON
                json_path = output_path.replace('.jpg', '.json').replace('.png', '.json')
                json_result = result.copy()
                
                # Remove image data from JSON
                if 'original_image' in json_result:
                    del json_result['original_image']
                if 'rotated_image' in json_result:
                    del json_result['rotated_image']
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Rotation result saved to: {output_path}")
                return True
            else:
                print("⚠️ No rotation result available")
                return False
                
        except Exception as e:
            print(f"❌ Error saving rotation result: {e}")
            return False
    
    def get_last_result(self) -> Optional[Dict]:
        """Get last rotation result"""
        return self.last_rotation_result
    
    def get_last_image(self) -> Optional[np.ndarray]:
        """Get last processed image"""
        return self.last_processed_image
    
    def get_last_rotated_image(self) -> Optional[np.ndarray]:
        """Get last rotated image"""
        return self.last_rotated_image
    
    def get_rotation_history(self) -> list:
        """Get rotation history"""
        return self.rotation_history
    
    def clear_history(self):
        """Clear rotation history"""
        self.rotation_history = []
    
    def clear_cache(self):
        """Clear rotation result cache"""
        self.cache.clear()
        print("🗑️ CACHE: Cleared all cached results")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'cache_max_size': self.cache_max_size,
            'cache_usage_percent': (len(self.cache) / self.cache_max_size * 100) if self.cache_max_size > 0 else 0
        }
    
    def set_cache_size(self, max_size: int):
        """Set maximum cache size"""
        self.cache_max_size = max_size
        # Remove oldest entries if cache exceeds new max size
        while len(self.cache) > max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        print(f"💾 CACHE: Set cache max size to {max_size} (current size: {len(self.cache)})")
    
    def batch_rotate_images(self, image_paths: list, output_dir: str = "rotated_images") -> list:
        """
        Rotate multiple images in batch
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save rotated images
            
        Returns:
            List of rotation results
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            results = []
            
            for i, image_path in enumerate(image_paths):
                print(f"🔄 Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                result = self.rotate_image(image_path)
                
                if 'error' not in result:
                    # Save rotated image
                    filename = os.path.basename(image_path)
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(output_dir, f"{name}_rotated{ext}")
                    
                    if self.save_rotated_image(output_path, result):
                        result['saved_path'] = output_path
                        results.append(result)
                        print(f"✅ Saved: {output_path}")
                else:
                    print(f"❌ Error processing: {result['error']}")
            
            print(f"🎉 Batch processing completed: {len(results)}/{len(image_paths)} images processed")
            return results
            
        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
            return []


# Global instance for easy access
_global_rotation_model = None

def get_global_model() -> RotationModel:
    """Get global rotation model instance"""
    global _global_rotation_model
    if _global_rotation_model is None:
        _global_rotation_model = RotationModel()
    return _global_rotation_model

def initialize_model(model_path: str, cuda: bool = True, craft_model_path: str = None) -> RotationModel:
    """
    Initialize global rotation model
    
    Args:
        model_path: Path to rotation model (.pth file)
        cuda: Use CUDA if available
        craft_model_path: Path to CRAFT model for text detection
        
    Returns:
        RotationModel instance
    """
    global _global_rotation_model
    _global_rotation_model = RotationModel(model_path, cuda, craft_model_path)
    return _global_rotation_model

def rotate_image_from_path(image_path: str) -> Dict:
    """
    Quick function to rotate image from path using global model
    
    Args:
        image_path: Path to image file
        
    Returns:
        Rotation result dictionary
    """
    model = get_global_model()
    if model.model is None:
        raise ValueError("Global model not initialized. Call initialize_model() first.")
    
    return model.rotate_image(image_path)

def rotate_image_from_array(image_array: np.ndarray) -> Dict:
    """
    Quick function to rotate image from array using global model
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        Rotation result dictionary
    """
    model = get_global_model()
    if model.model is None:
        raise ValueError("Global model not initialized. Call initialize_model() first.")
    
    return model.rotate_image(image_array=image_array)

def rotate_image_with_verification_from_path(image_path: str) -> Dict:
    """
    Quick function to rotate image from path with verification using global model
    
    Args:
        image_path: Path to image file
        
    Returns:
        Rotation result dictionary with verification
    """
    model = get_global_model()
    if model.model is None:
        raise ValueError("Global model not initialized. Call initialize_model() first.")
    
    return model.rotate_image_with_verification(image_path)

def rotate_image_with_verification_from_array(image_array: np.ndarray) -> Dict:
    """
    Quick function to rotate image from array with verification using global model
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        Rotation result dictionary with verification
    """
    model = get_global_model()
    if model.model is None:
        raise ValueError("Global model not initialized. Call initialize_model() first.")
    
    return model.rotate_image_with_verification(image_array=image_array)

def rotate_and_crop_text_from_path(image_path: str) -> Dict:
    """
    Quick function to rotate image and crop text regions from path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Combined result dictionary with rotation, detection, and cropping
    """
    model = get_global_model()
    if model.model is None:
        raise ValueError("Global model not initialized. Call initialize_model() first.")
    if model.craft_detector is None:
        raise ValueError("CRAFT detector not initialized. Call initialize_model() with craft_model_path.")
    
    return model.rotate_and_detect_text(image_path)

def rotate_and_crop_text_from_array(image_array: np.ndarray) -> Dict:
    """
    Quick function to rotate image and crop text regions from array
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        Combined result dictionary with rotation, detection, and cropping
    """
    model = get_global_model()
    if model.model is None:
        raise ValueError("Global model not initialized. Call initialize_model() first.")
    if model.craft_detector is None:
        raise ValueError("CRAFT detector not initialized. Call initialize_model() with craft_model_path.")
    
    return model.rotate_and_detect_text(image_array=image_array)

def get_rotated_image_from_path(image_path: str) -> Optional[np.ndarray]:
    """
    Quick function to get rotated image from path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Rotated image as numpy array, or None if error
    """
    result = rotate_image_from_path(image_path)
    return result.get('rotated_image') if 'error' not in result else None

def get_rotated_image_from_array(image_array: np.ndarray) -> Optional[np.ndarray]:
    """
    Quick function to get rotated image from array
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        Rotated image as numpy array, or None if error
    """
    result = rotate_image_from_array(image_array)
    return result.get('rotated_image') if 'error' not in result else None

def get_cropped_text_image_from_path(image_path: str) -> Optional[np.ndarray]:
    """
    Quick function to get cropped text image from path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Cropped text image as numpy array, or None if error
    """
    result = rotate_and_crop_text_from_path(image_path)
    return result.get('cropped_image') if 'error' not in result else None

def get_cropped_text_image_from_array(image_array: np.ndarray) -> Optional[np.ndarray]:
    """
    Quick function to get cropped text image from array
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        Cropped text image as numpy array, or None if error
    """
    result = rotate_and_crop_text_from_array(image_array)
    return result.get('cropped_image') if 'error' not in result else None

def process_craft_rotated_from_path(craft_rotated_image_path: str) -> Dict:
    """
    รับรูปที่หมุนแล้วจาก CRAFT จากไฟล์ มาแล้วหมุนด้วย AI อีกที
    
    Args:
        craft_rotated_image_path: Path ไปยังรูปที่ CRAFT หมุนแล้ว
        
    Returns:
        Dict containing AI rotation, text detection, and cropping results
    """
    model = get_global_model()
    if model.model is None:
        raise ValueError("Global model not initialized. Call initialize_model() first.")
    if model.craft_detector is None:
        raise ValueError("CRAFT detector not initialized. Call initialize_model() with craft_model_path.")
    
    return model.process_craft_rotated_from_path(craft_rotated_image_path)

def process_craft_rotated_from_array(craft_rotated_image: np.ndarray) -> Dict:
    """
    รับรูปที่หมุนแล้วจาก CRAFT จาก numpy array มาแล้วหมุนด้วย AI อีกที
    
    Args:
        craft_rotated_image: รูปที่หมุนแล้วจาก CRAFT (numpy array)
        
    Returns:
        Dict containing AI rotation, text detection, and cropping results
    """
    model = get_global_model()
    if model.model is None:
        raise ValueError("Global model not initialized. Call initialize_model() first.")
    if model.craft_detector is None:
        raise ValueError("CRAFT detector not initialized. Call initialize_model() with craft_model_path.")
    
    return model.process_craft_rotated_image(craft_rotated_image)

def process_craft_rotated_with_verification_from_path(craft_rotated_image_path: str) -> Dict:
    """
    รับรูปที่หมุนแล้วจาก CRAFT จากไฟล์ มาแล้วหมุนด้วย AI พร้อม verification
    
    Args:
        craft_rotated_image_path: Path ไปยังรูปที่ CRAFT หมุนแล้ว
        
    Returns:
        Dict containing AI rotation with verification, text detection, and cropping results
    """
    model = get_global_model()
    if model.model is None:
        raise ValueError("Global model not initialized. Call initialize_model() first.")
    if model.craft_detector is None:
        raise ValueError("CRAFT detector not initialized. Call initialize_model() with craft_model_path.")
    
    return model.process_craft_rotated_image_with_verification_from_path(craft_rotated_image_path)

def process_craft_rotated_with_verification_from_array(craft_rotated_image: np.ndarray) -> Dict:
    """
    รับรูปที่หมุนแล้วจาก CRAFT จาก numpy array มาแล้วหมุนด้วย AI พร้อม verification
    
    Args:
        craft_rotated_image: รูปที่หมุนแล้วจาก CRAFT (numpy array)
        
    Returns:
        Dict containing AI rotation with verification, text detection, and cropping results
    """
    model = get_global_model()
    if model.model is None:
        raise ValueError("Global model not initialized. Call initialize_model() first.")
    if model.craft_detector is None:
        raise ValueError("CRAFT detector not initialized. Call initialize_model() with craft_model_path.")
    
    return model.process_craft_rotated_image_with_verification(craft_rotated_image)

def get_cropped_from_craft_rotated_path(craft_rotated_image_path: str) -> Optional[np.ndarray]:
    """
    รับรูปที่ครอปแล้วจากรูปที่ CRAFT หมุนแล้ว (จากไฟล์)
    
    Args:
        craft_rotated_image_path: Path ไปยังรูปที่ CRAFT หมุนแล้ว
        
    Returns:
        รูปที่ครอปแล้วเป็น numpy array, หรือ None ถ้าเกิดข้อผิดพลาด
    """
    result = process_craft_rotated_from_path(craft_rotated_image_path)
    return result.get('cropped_image') if 'error' not in result else None

def get_cropped_from_craft_rotated_array(craft_rotated_image: np.ndarray) -> Optional[np.ndarray]:
    """
    รับรูปที่ครอปแล้วจากรูปที่ CRAFT หมุนแล้ว (จาก numpy array)
    
    Args:
        craft_rotated_image: รูปที่หมุนแล้วจาก CRAFT (numpy array)
        
    Returns:
        รูปที่ครอปแล้วเป็น numpy array, หรือ None ถ้าเกิดข้อผิดพลาด
    """
    result = process_craft_rotated_from_array(craft_rotated_image)
    return result.get('cropped_image') if 'error' not in result else None

def get_cropped_with_verification_from_craft_rotated_path(craft_rotated_image_path: str) -> Optional[np.ndarray]:
    """
    รับรูปที่ครอปแล้วจากรูปที่ CRAFT หมุนแล้ว พร้อม verification (จากไฟล์)
    
    Args:
        craft_rotated_image_path: Path ไปยังรูปที่ CRAFT หมุนแล้ว
        
    Returns:
        รูปที่ครอปแล้วเป็น numpy array, หรือ None ถ้าเกิดข้อผิดพลาด
    """
    result = process_craft_rotated_with_verification_from_path(craft_rotated_image_path)
    return result.get('cropped_image') if 'error' not in result else None

def get_cropped_with_verification_from_craft_rotated_array(craft_rotated_image: np.ndarray) -> Optional[np.ndarray]:
    """
    รับรูปที่ครอปแล้วจากรูปที่ CRAFT หมุนแล้ว พร้อม verification (จาก numpy array)
    
    Args:
        craft_rotated_image: รูปที่หมุนแล้วจาก CRAFT (numpy array)
        
    Returns:
        รูปที่ครอปแล้วเป็น numpy array, หรือ None ถ้าเกิดข้อผิดพลาด
    """
    result = process_craft_rotated_with_verification_from_array(craft_rotated_image)
    return result.get('cropped_image') if 'error' not in result else None


# Example usage and testing
if __name__ == "__main__":
    # Example: Initialize model with both rotation and CRAFT models
    rotation_model_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/final/text_rotation_model.pth"
    # Use CRAFT model path from config if available
    try:
        from config.settings import CRAFT_MODEL_PATH
        craft_model_path = CRAFT_MODEL_PATH
    except ImportError:
        # Fallback to old path if config not available
        craft_model_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch/craft_mlt_25k.pth"
    
    # Test image path - รูปที่ CRAFT หมุนแล้ว
    craft_rotated_image_path = r"D:\Sentech_Camera_Images-20250803T091245Z-1-001\Sentech_Camera_Images\sentech_20250801_171819.png"
    
    try:
        print("🚀 Testing CRAFT Rotated Image Processing with AI Rotation and Text Detection")
        print("=" * 70)
        
        # Check if files exist
        if not os.path.exists(rotation_model_path):
            print(f"❌ Rotation model not found: {rotation_model_path}")
            print("Please update the path to your rotation model")
            exit(1)
        
        if not os.path.exists(craft_model_path):
            print(f"❌ CRAFT model not found: {craft_model_path}")
            print("Please update the path to your CRAFT model")
            exit(1)
        
        if not os.path.exists(craft_rotated_image_path):
            print(f"❌ CRAFT rotated image not found: {craft_rotated_image_path}")
            print("Please update the path to your CRAFT rotated image")
            exit(1)
        
        # Initialize global model with both rotation and CRAFT
        print("🔧 Initializing models...")
        model = initialize_model(rotation_model_path, cuda=True, craft_model_path=craft_model_path)
        print("✅ Rotation and CRAFT models initialized successfully!")
        
        print(f"\n📸 Testing with CRAFT rotated image: {craft_rotated_image_path}")
        
        # Example 1: Process CRAFT rotated image from path
        print("\n🔄 Testing CRAFT rotated image processing from path...")
        result = process_craft_rotated_from_path(craft_rotated_image_path)
        
        if 'error' not in result:
            print(f"✅ Processing successful!")
            print(f"   - AI rotation angle: {result['ai_rotation_angle']} degrees")
            print(f"   - Text regions found: {result['total_text_regions']}")
            
            # Save results
            cv2.imwrite("craft_rotated_input.jpg", cuda_cvtColor(result['craft_rotated_image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite("ai_rotated_output.jpg", cuda_cvtColor(result['ai_rotated_image'], cv2.COLOR_RGB2BGR))
            
            # Save with text detection visualization
            if result['total_text_regions'] > 0:
                # Create visualization with text detection boxes
                vis_image = result['ai_rotated_image'].copy()
                for poly in result['text_polys']:
                    if poly is not None:
                        cv2.polylines(vis_image, [poly.astype(np.int32)], True, (255, 0, 0), 2)
                
                cv2.imwrite("ai_rotated_with_text_detection.jpg", cuda_cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                print("✅ Text detection visualization saved as 'ai_rotated_with_text_detection.jpg'")
            
            print("💾 Saved files:")
            print("   - craft_rotated_input.jpg (input from CRAFT)")
            print("   - ai_rotated_output.jpg (after AI rotation)")
            if result['total_text_regions'] > 0:
                print("   - ai_rotated_with_text_detection.jpg (with text detection boxes)")
        else:
            print(f"❌ Processing failed: {result['error']}")
        
        # Example 1.5: Process CRAFT rotated image with verification from path
        print("\n🔄 Testing CRAFT rotated image processing with verification from path...")
        result_verification = process_craft_rotated_with_verification_from_path(craft_rotated_image_path)
        
        if 'error' not in result_verification:
            print(f"✅ Verification processing successful!")
            print(f"   - Total AI rotation angle: {result_verification['ai_rotation_angle']} degrees")
            print(f"   - First rotation: {result_verification.get('first_rotation_angle', 0)} degrees")
            print(f"   - Second rotation: {result_verification.get('second_rotation_angle', 0)} degrees")
            print(f"   - Verification passed: {result_verification.get('verification_passed', False)}")
            print(f"   - Total rotations: {result_verification.get('total_rotations', 1)}")
            print(f"   - Text regions found: {result_verification['total_text_regions']}")
            
            # Save verification results
            cv2.imwrite("verification_ai_rotated_output.jpg", cuda_cvtColor(result_verification['ai_rotated_image'], cv2.COLOR_RGB2BGR))
            
            # Save with text detection visualization
            if result_verification['total_text_regions'] > 0:
                # Create visualization with text detection boxes
                vis_image = result_verification['ai_rotated_image'].copy()
                for poly in result_verification['text_polys']:
                    if poly is not None:
                        cv2.polylines(vis_image, [poly.astype(np.int32)], True, (0, 255, 0), 2)
                
                cv2.imwrite("verification_ai_rotated_with_text_detection.jpg", cuda_cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                print("✅ Verification text detection visualization saved")
            
            print("💾 Verification saved files:")
            print("   - verification_ai_rotated_output.jpg (after AI rotation with verification)")
            if result_verification['total_text_regions'] > 0:
                print("   - verification_ai_rotated_with_text_detection.jpg (with text detection boxes)")
        else:
            print(f"❌ Verification processing failed: {result_verification['error']}")
        
        # Example 2: Process CRAFT rotated image from array
        print("\n🔄 Testing CRAFT rotated image processing from array...")
        import cv2
        craft_rotated_array = cv2.imread(craft_rotated_image_path)
        if craft_rotated_array is not None:
            # Convert BGR to RGB
            craft_rotated_rgb = cuda_cvtColor(craft_rotated_array, cv2.COLOR_BGR2RGB)
            
            result = process_craft_rotated_from_array(craft_rotated_rgb)
            if 'error' not in result:
                cv2.imwrite("array_ai_rotated_result.jpg", cuda_cvtColor(result['ai_rotated_image'], cv2.COLOR_RGB2BGR))
                print("✅ Array processing successful: array_ai_rotated_result.jpg")
            else:
                print("❌ Array processing failed")
        else:
            print("❌ Failed to load image as array")
        
        # Example 2.5: Process CRAFT rotated image with verification from array
        print("\n🔄 Testing CRAFT rotated image processing with verification from array...")
        if craft_rotated_array is not None:
            # Convert BGR to RGB
            craft_rotated_rgb = cuda_cvtColor(craft_rotated_array, cv2.COLOR_BGR2RGB)
            
            result_verification_array = process_craft_rotated_with_verification_from_array(craft_rotated_rgb)
            if 'error' not in result_verification_array:
                cv2.imwrite("array_verification_ai_rotated_result.jpg", cuda_cvtColor(result_verification_array['ai_rotated_image'], cv2.COLOR_RGB2BGR))
                print("✅ Array verification processing successful: array_verification_ai_rotated_result.jpg")
                print(f"   - Verification passed: {result_verification_array.get('verification_passed', False)}")
                print(f"   - Total rotations: {result_verification_array.get('total_rotations', 1)}")
            else:
                print("❌ Array verification processing failed")
        else:
            print("❌ Failed to load image as array for verification")
        
        # Example 3: Get AI rotated image directly
        print("\n🎯 Testing direct AI rotated image retrieval...")
        ai_rotated_img = get_cropped_from_craft_rotated_path(craft_rotated_image_path)
        if ai_rotated_img is not None:
            cv2.imwrite("direct_ai_rotated_result.jpg", cuda_cvtColor(ai_rotated_img, cv2.COLOR_RGB2BGR))
            print("✅ Direct AI rotated image saved: direct_ai_rotated_result.jpg")
        else:
            print("❌ Failed to get direct AI rotated image")
        
        # Example 3.5: Get AI rotated image with verification directly
        print("\n🎯 Testing direct AI rotated image retrieval with verification...")
        ai_rotated_img_verification = get_cropped_with_verification_from_craft_rotated_path(craft_rotated_image_path)
        if ai_rotated_img_verification is not None:
            cv2.imwrite("direct_verification_ai_rotated_result.jpg", cuda_cvtColor(ai_rotated_img_verification, cv2.COLOR_RGB2BGR))
            print("✅ Direct verification AI rotated image saved: direct_verification_ai_rotated_result.jpg")
        else:
            print("❌ Failed to get direct verification AI rotated image")
        
        print("\n🎉 Testing completed!")
        print("\n📁 Generated files:")
        print("   - craft_rotated_input.jpg (input from CRAFT)")
        print("   - ai_rotated_output.jpg (after AI rotation)")
        print("   - ai_rotated_with_text_detection.jpg (with text detection boxes)")
        print("   - verification_ai_rotated_output.jpg (after AI rotation with verification)")
        print("   - verification_ai_rotated_with_text_detection.jpg (verification with text detection)")
        print("   - array_ai_rotated_result.jpg (array processing)")
        print("   - array_verification_ai_rotated_result.jpg (array verification processing)")
        print("   - direct_ai_rotated_result.jpg (direct retrieval)")
        print("   - direct_verification_ai_rotated_result.jpg (direct verification retrieval)")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

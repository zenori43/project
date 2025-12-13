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
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
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
import math
from typing import Dict, List, Optional, Union, Tuple
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

# Add deep-text-recognition-benchmark path
# Get path from config if available, otherwise use default
try:
    from config.settings import DEEP_TEXT_RECOGNITION_DIR
    deep_text_recognition_path = DEEP_TEXT_RECOGNITION_DIR
except ImportError:
    # Fallback to old path if config not available
    deep_text_recognition_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/deep-text-recognition-benchmark"

if deep_text_recognition_path not in sys.path and os.path.exists(deep_text_recognition_path):
    sys.path.insert(0, deep_text_recognition_path)

try:
    from utils import CTCLabelConverter, AttnLabelConverter
    from model import Model
    from dataset import RawDataset, AlignCollate, NormalizePAD, ResizeNormalize
except ImportError as e:
    print(f"Error importing deep-text-recognition modules: {e}")
    print("Please make sure deep-text-recognition-benchmark is properly installed at /home/nvidia/Desktop/final_boss/backupsdcard/deep-text-recognition-benchmark")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DeepOCRModel:
    """
    Deep OCR Model Class สำหรับเรียกใช้จาก code อื่น
    ใช้ deep-text-recognition-benchmark เพื่อทำ OCR
    """
    
    def __init__(self, model_path: str = None, cuda: bool = True):
        """
        Initialize Deep OCR Model
        
        Args:
            model_path: Path to trained model (.pth file)
            cuda: Use CUDA if available
        """
        self.model = None
        self.converter = None
        # Keep CUDA for Deep OCR (as requested)
        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
        print(f"✅ Deep OCR using device: {self.device}")
        
        # Default settings
        self.settings = {
            'Transformation': 'TPS',
            'FeatureExtraction': 'ResNet',
            'SequenceModeling': 'BiLSTM',
            'Prediction': 'Attn',
            'imgH': 32,
            'imgW': 100,
            'input_channel': 1,  # Use grayscale (1 channel)
            'output_channel': 512,
            'hidden_size': 256,
            'num_fiducial': 20,
            'batch_max_length': 25,
            'character': '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:/',
            'sensitive': True,
            'PAD': False,
            'batch_size': 1,
            'workers': 0,
            'num_class': 67
        }
        
        # Global variables for storing results
        self.last_ocr_result = None
        self.last_processed_image = None
        self.ocr_history = []
        
        if model_path:
            self.load_model(model_path)
    
    def _create_opt_object(self, settings: dict):
        """
        Create Opt object with all required attributes for deep-text-recognition
        
        Args:
            settings: Dictionary of settings
            
        Returns:
            Opt object with all required attributes
        """
        class Opt:
            def __init__(self, settings):
                for key, value in settings.items():
                    setattr(self, key, value)
                # Add missing attributes that RawDataset needs
                self.rgb = False  # Use grayscale images (1 channel)
                self.train_data = None  # Not used for inference
                self.select_data = None  # Not used for inference
                self.batch_ratio = None  # Not used for inference
                self.total_data_usage_ratio = 1.0  # Not used for inference
                self.exp_name = 'demo'  # Not used for inference
        
        return Opt(settings)
    
    def _detect_model_character_set(self, state_dict: dict) -> str:
        """
        Detect the character set used in the model from state dict
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            str: Detected character set
        """
        try:
            # Look for generator weight to determine number of classes
            generator_key = None
            for key in state_dict.keys():
                if 'generator.weight' in key:
                    generator_key = key
                    break
            
            if generator_key:
                num_classes = state_dict[generator_key].size(0)
                print(f"🔍 Detected {num_classes} classes in model")
                
                # Common character sets based on number of classes
                # Note: For AttnLabelConverter, the model has 2 extra tokens ([GO] and [s])
                # So if model has 96 classes, character set should be 94 characters
                if num_classes == 67:  # Custom character set with alphanumeric + :/
                    return '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:/'
                elif num_classes == 38:  # Basic alphanumeric
                    return '0123456789abcdefghijklmnopqrstuvwxyz'
                elif num_classes == 96:  # Extended ASCII with space (most common) - 94 chars + 2 tokens
                    return '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}'
                elif num_classes == 95:  # ASCII printable without space - 93 chars + 2 tokens
                    return '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
                elif num_classes == 97:  # Extended ASCII with space and special chars - 95 chars + 2 tokens
                    return '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
                elif num_classes == 98:  # Extended ASCII with space and special chars (padded) - 96 chars + 2 tokens
                    return '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
                elif num_classes == 37:  # Basic alphanumeric without 'z'
                    return '0123456789abcdefghijklmnopqrstuvwxy'
                elif num_classes == 36:  # Basic alphanumeric without 'y', 'z'
                    return '0123456789abcdefghijklmnopqrstuvwx'
                else:
                    # Try to infer from other parameters
                    print(f"⚠️ Unknown character set size: {num_classes}")
                    print("🔧 Trying to create compatible character set...")
                    
                    # Create a character set with the exact number of classes
                    if num_classes <= 26:
                        # Just lowercase letters
                        return 'abcdefghijklmnopqrstuvwxyz'[:num_classes]
                    elif num_classes <= 36:
                        # Numbers + lowercase letters
                        return '0123456789abcdefghijklmnopqrstuvwxyz'[:num_classes]
                    elif num_classes <= 62:
                        # Numbers + lowercase + uppercase letters
                        return '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[:num_classes]
                    elif num_classes <= 95:
                        # ASCII printable
                        ascii_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
                        return ascii_chars[:num_classes]
                    elif num_classes <= 96:
                        # Extended ASCII without space (for AttnLabelConverter with 2 tokens)
                        extended_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}'
                        return extended_chars[:num_classes]
                    else:
                        # Extended ASCII with space and pad
                        extended_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
                        if num_classes <= len(extended_chars):
                            return extended_chars[:num_classes]
                        else:
                            # Pad with spaces if needed
                            return extended_chars + ' ' * (num_classes - len(extended_chars))
            
            return self.settings['character']
            
        except Exception as e:
            print(f"⚠️ Error detecting character set: {e}")
            return self.settings['character']
    
    def load_model(self, model_path: str) -> bool:
        """
        Load OCR model from path
        
        Args:
            model_path: Path to .pth model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"Loading Deep OCR model from {model_path}...")
            
            # Load state dict first to detect character set
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle DataParallel state dict (remove 'module.' prefix if present)
            new_state_dict = {}
            has_module_prefix = False
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                    has_module_prefix = True
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            if has_module_prefix:
                print("🔧 Detected DataParallel state dict, removing 'module.' prefix...")
            
            # Debug: Print some key names to understand model structure
            print(f"🔍 Model state dict keys (first 10): {list(new_state_dict.keys())[:10]}")
            if 'Prediction.attention_cell.rnn.weight_ih' in new_state_dict:
                print(f"🔍 Found attention cell in model - this is an Attn model")
            elif 'Prediction.weight' in new_state_dict:
                print(f"🔍 Found Prediction.weight in model - this is a CTC model")
            
            # Detect character set from model
            detected_character_set = self._detect_model_character_set(new_state_dict)
            print(f"🔍 Detected character set: {detected_character_set}")
            print(f"🔍 Character set length: {len(detected_character_set)}")
            
            # Verify character set length matches model
            generator_key = None
            for key in new_state_dict.keys():
                if 'generator.weight' in key:
                    generator_key = key
                    break
            
            if generator_key:
                expected_classes = new_state_dict[generator_key].size(0)
                actual_length = len(detected_character_set)
                
                if actual_length != expected_classes:
                    print(f"⚠️ Character set length mismatch: expected {expected_classes}, got {actual_length}")
                    print("🔧 Adjusting character set length...")
                    
                    # Create a new character set with the exact number of classes
                    if expected_classes == 67:
                        # Use the exact 65-character set that matches the model (67 - 2 tokens)
                        detected_character_set = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:/'
                        print(f"🔧 Using 65-character set for 67-class model (with 2 tokens)")
                    elif expected_classes == 96:
                        # Use the exact 94-character set that matches the model (96 - 2 tokens)
                        detected_character_set = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}'
                        print(f"🔧 Using 94-character set for 96-class model (with 2 tokens)")
                    elif expected_classes == 95:
                        # Use the exact 95-character set that matches the model
                        detected_character_set = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
                        print(f"🔧 Using 95-character set (without space) for model")
                    elif expected_classes == 97:
                        # Use the exact 97-character set that matches the model
                        detected_character_set = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
                        print(f"🔧 Using 97-character set for model")
                    elif expected_classes == 98:
                        # Use the exact 98-character set that matches the model
                        detected_character_set = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
                        print(f"🔧 Using 98-character set for model")
                    else:
                        # For other sizes, use the original logic
                        if actual_length > expected_classes:
                            # Truncate character set
                            detected_character_set = detected_character_set[:expected_classes]
                            print(f"🔧 Truncated character set to {expected_classes} characters")
                        else:
                            # Pad character set
                            detected_character_set = detected_character_set + ' ' * (expected_classes - actual_length)
                            print(f"🔧 Padded character set to {expected_classes} characters")
                    
                    print(f"🔧 Adjusted character set: {detected_character_set}")
                    print(f"🔧 Adjusted length: {len(detected_character_set)}")
                    
                    # Verify the adjusted character set has the correct length
                    if len(detected_character_set) != expected_classes:
                        print(f"❌ Character set length verification failed: {len(detected_character_set)} != {expected_classes}")
                        # Force the correct length by truncating or padding
                        if len(detected_character_set) > expected_classes:
                            detected_character_set = detected_character_set[:expected_classes]
                        else:
                            detected_character_set = detected_character_set + ' ' * (expected_classes - len(detected_character_set))
                        print(f"🔧 Final character set length: {len(detected_character_set)}")
                    
                    # Remove any duplicate characters
                    unique_chars = []
                    for char in detected_character_set:
                        if char not in unique_chars:
                            unique_chars.append(char)
                    
                    if len(unique_chars) != len(detected_character_set):
                        print(f"⚠️ Duplicate characters detected, removing duplicates...")
                        detected_character_set = ''.join(unique_chars)
                        # Pad if needed
                        if len(detected_character_set) < expected_classes:
                            detected_character_set = detected_character_set + ' ' * (expected_classes - len(detected_character_set))
                        print(f"🔧 Final character set after removing duplicates: {len(detected_character_set)}")
            
            # Update settings with detected character set
            self.settings['character'] = detected_character_set
            self.settings['sensitive'] = True  # Most models are case-sensitive
            
            # Create model configuration
            opt = self._create_opt_object(self.settings)
            
            # Set number of classes based on the model's actual number of classes
            if generator_key:
                expected_classes = new_state_dict[generator_key].size(0)
                opt.num_class = expected_classes
                print(f"📊 Using model's actual number of classes: {expected_classes}")
                
                # Try to detect prediction type from model structure
                if 'attention_cell' in str(new_state_dict.keys()):
                    print(f"🔍 Detected Attn prediction type from model structure")
                    opt.Prediction = 'Attn'
                elif 'generator' in str(new_state_dict.keys()):
                    print(f"🔍 Detected CTC prediction type from model structure")
                    opt.Prediction = 'CTC'
                else:
                    print(f"⚠️ Could not detect prediction type, using default: {opt.Prediction}")
            else:
                # Fallback to character set length
                opt.num_class = len(detected_character_set)
                print(f"📊 Using character set length as number of classes: {opt.num_class}")
            
            # Create converter with the correct number of classes
            print(f"🔍 Creating converter with character set length: {len(opt.character)}")
            if 'CTC' in opt.Prediction:
                self.converter = CTCLabelConverter(opt.character)
                print(f"🔍 CTC converter created with {len(self.converter.character)} characters")
            else:
                self.converter = AttnLabelConverter(opt.character)
                print(f"🔍 AttnLabelConverter created with {len(self.converter.character)} characters")
                print(f"🔍 AttnLabelConverter tokens: {self.converter.character[:2]}")  # Show [GO] and [s] tokens
            
            # Ensure the converter has the correct number of classes
            print(f"🔍 Converter type: {type(self.converter)}")
            print(f"🔍 Converter character set length: {len(self.converter.character)}")
            if hasattr(self.converter, 'num_class'):
                print(f"🔍 Converter num_class: {self.converter.num_class}")
            
            # Check if converter has the correct number of classes
            if hasattr(self.converter, 'num_class') and self.converter.num_class != opt.num_class:
                print(f"⚠️ Converter has {self.converter.num_class} classes, but model expects {opt.num_class}")
                # Adjust the character set to match the model's expected number of classes
                if opt.num_class == 67:
                    # Use the exact 65-character set that matches the model (67 - 2 tokens)
                    opt.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:/'
                    self.settings['character'] = opt.character
                    print(f"🔧 Recreating converter with 65-character set for 67-class model: {opt.character}")
                    # Recreate converter with correct character set
                    if 'CTC' in opt.Prediction:
                        self.converter = CTCLabelConverter(opt.character)
                    else:
                        self.converter = AttnLabelConverter(opt.character)
                    print(f"🔍 New converter num_class: {self.converter.num_class}")
                elif opt.num_class == 96:
                    # Use the exact 94-character set that matches the model (96 - 2 tokens)
                    opt.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}'
                    self.settings['character'] = opt.character
                    print(f"🔧 Recreating converter with 94-character set for 96-class model: {opt.character}")
                    # Recreate converter with correct character set
                    if 'CTC' in opt.Prediction:
                        self.converter = CTCLabelConverter(opt.character)
                    else:
                        self.converter = AttnLabelConverter(opt.character)
                    print(f"🔍 New converter num_class: {self.converter.num_class}")
            
            # Update opt.num_class to match the converter's final number of classes
            if hasattr(self.converter, 'num_class'):
                opt.num_class = self.converter.num_class
                print(f"📊 Updated opt.num_class to match converter: {opt.num_class}")
            
            print(f"📊 Final model configuration: {opt.num_class} classes, character set length: {len(opt.character)}")
            
            # Create model
            self.model = Model(opt)
            print('Model input parameters:', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
                  opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
                  opt.SequenceModeling, opt.Prediction)
            
            # Load the cleaned state dict with strict=False to handle architecture mismatches
            try:
                self.model.load_state_dict(new_state_dict, strict=True)
                print("✅ Model weights loaded with strict=True")
            except RuntimeError as e:
                if "Missing key" in str(e) or "Unexpected key" in str(e):
                    print("⚠️ Architecture mismatch detected, loading with strict=False...")
                    self.model.load_state_dict(new_state_dict, strict=False)
                    print("✅ Model weights loaded with strict=False (some layers may not match)")
                else:
                    raise e
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model.eval()
            
            print("✅ Deep OCR model loaded successfully.")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Deep OCR model: {e}")
            return False
    
    def update_settings(self, **kwargs):
        """
        Update OCR settings
        
        Args:
            **kwargs: Settings to update
        """
        self.settings.update(kwargs)
    
    def recognize_text_from_image_path(self, image_path: str) -> Dict:
        """
        Recognize text from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict containing OCR results
        """
        try:
            if self.model is None:
                raise ValueError("OCR model not loaded. Please load model first.")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image {image_path}")
            
            return self.recognize_text_from_image_array(image, image_path)
            
        except Exception as e:
            print(f"Error recognizing text from image path: {e}")
            return {
                'error': str(e),
                'recognized_text': '',
                'confidence_score': 0.0,
                'image_path': image_path
            }
    
    def recognize_text_from_image_array(self, image: np.ndarray, image_path: str = None) -> Dict:
        """
        Recognize text from image array
        
        Args:
            image: Image as numpy array (BGR format)
            image_path: Optional image path for reference
            
        Returns:
            Dict containing OCR results
        """
        try:
            if self.model is None:
                raise ValueError("OCR model not loaded. Please load model first.")
            
            if image is None:
                raise ValueError("Image array is None")
            
            # Store original image
            self.last_processed_image = image.copy()
            
            # Convert to grayscale for model input
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to grayscale
                image_gray = cuda_cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image
            
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                # Save as grayscale
                cv2.imwrite(temp_path, image_gray)
            
            try:
                # Process with deep-text-recognition
                result = self._process_with_deep_recognition(temp_path)
                
                # Add image info to result
                result['image_path'] = image_path
                result['original_image'] = image
                result['timestamp'] = str(datetime.now())
                
                # Store in global variables
                self.last_ocr_result = result
                self.ocr_history.append(result)
                
                return result
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error recognizing text from image array: {e}")
            return {
                'error': str(e),
                'recognized_text': '',
                'confidence_score': 0.0,
                'image_path': image_path
            }
    
    def recognize_text_from_craft_lines(self, craft_result: Dict) -> Dict:
        """
        Recognize text from CRAFT line detection result
        
        Args:
            craft_result: Result from craft_line_detection.py
            
        Returns:
            Dict containing OCR results for each line
        """
        try:
            if self.model is None:
                raise ValueError("OCR model not loaded. Please load model first.")
            
            if 'error' in craft_result:
                raise ValueError(f"CRAFT detection error: {craft_result['error']}")
            
            cropped_lines = craft_result.get('cropped_lines', [])
            if not cropped_lines:
                return {
                    'error': 'No cropped lines found in CRAFT result',
                    'line_results': [],
                    'total_lines': 0
                }
            
            # Process each line
            line_results = []
            for line_data in cropped_lines:
                line_idx = line_data['line_index']
                line_image = line_data['image']
                
                # Recognize text for this line
                ocr_result = self.recognize_text_from_image_array(line_image)
                
                line_results.append({
                    'line_index': line_idx,
                    'image': line_image,
                    'bbox': line_data.get('bbox'),
                    'num_characters': line_data.get('num_characters', 0),
                    'recognized_text': ocr_result.get('recognized_text', ''),
                    'confidence_score': ocr_result.get('confidence_score', 0.0),
                    'error': ocr_result.get('error')
                })
            
            # Create combined result
            combined_result = {
                'line_results': line_results,
                'total_lines': len(line_results),
                'craft_result': craft_result,
                'timestamp': str(datetime.now())
            }
            
            # Calculate overall statistics
            valid_results = [r for r in line_results if 'error' not in r]
            if valid_results:
                combined_result['overall_confidence'] = np.mean([r['confidence_score'] for r in valid_results])
                combined_result['total_recognized_text'] = ' '.join([r['recognized_text'] for r in valid_results if r['recognized_text']])
            else:
                combined_result['overall_confidence'] = 0.0
                combined_result['total_recognized_text'] = ''
            
            return combined_result
            
        except Exception as e:
            print(f"Error recognizing text from CRAFT lines: {e}")
            return {
                'error': str(e),
                'line_results': [],
                'total_lines': 0
            }
    
    def _process_with_deep_recognition(self, image_path: str) -> Dict:
        """
        Process image with deep-text-recognition model
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict containing recognition results
        """
        try:
            # Create model configuration
            opt = self._create_opt_object(self.settings)
            
            # Prepare data - force grayscale input
            class GrayscaleAlignCollate(AlignCollate):
                def __call__(self, batch):
                    batch = filter(lambda x: x is not None, batch)
                    images, labels = zip(*batch)
                    
                    # Force all images to grayscale
                    grayscale_images = []
                    for image in images:
                        if image.mode != 'L':
                            image = image.convert('L')
                        grayscale_images.append(image)
                    
                    if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
                        resized_max_w = self.imgW
                        input_channel = 1  # Force grayscale
                        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

                        resized_images = []
                        for image in grayscale_images:
                            w, h = image.size
                            ratio = w / float(h)
                            if math.ceil(self.imgH * ratio) > self.imgW:
                                resized_w = self.imgW
                            else:
                                resized_w = math.ceil(self.imgH * ratio)

                            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                            resized_images.append(transform(resized_image))

                        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

                    else:
                        transform = ResizeNormalize((self.imgW, self.imgH))
                        image_tensors = [transform(image) for image in grayscale_images]
                        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

                    return image_tensors, labels
            
            AlignCollate_demo = GrayscaleAlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
            
            # Create temporary folder with single image
            temp_folder = tempfile.mkdtemp()
            temp_image_path = os.path.join(temp_folder, os.path.basename(image_path))
            
            try:
                # Copy image to temp folder
                import shutil
                shutil.copy2(image_path, temp_image_path)
                
                # Verify the image file exists and is readable
                if not os.path.exists(temp_image_path):
                    raise FileNotFoundError(f"Failed to create temporary image file: {temp_image_path}")
                
                # Test if image can be opened and convert to grayscale
                try:
                    test_img = Image.open(temp_image_path)
                    test_img.verify()
                    
                    # Ensure image is grayscale
                    if test_img.mode != 'L':
                        test_img = test_img.convert('L')
                        test_img.save(temp_image_path)
                        print(f"Converted image to grayscale: {temp_image_path}")
                        
                except Exception as img_error:
                    raise ValueError(f"Invalid image file: {img_error}")
                
                # Create dataset
                demo_data = RawDataset(root=temp_folder, opt=opt)
                
                # Check if dataset has any images
                if len(demo_data) == 0:
                    raise ValueError("No images found in dataset")
                
                demo_loader = torch.utils.data.DataLoader(
                    demo_data, batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=int(opt.workers),
                    collate_fn=AlignCollate_demo, pin_memory=True)
                
                # Predict
                if self.model is None:
                    raise ValueError("Model not loaded. Please load model first.")
                
                with torch.no_grad():
                    for image_tensors, image_path_list in demo_loader:
                        batch_size = image_tensors.size(0)
                        image = image_tensors.to(self.device)
                        
                        # Verify tensor shape is correct (should be [batch_size, 1, height, width])
                        if image.size(1) != 1:
                            print(f"Warning: Expected 1 channel, got {image.size(1)} channels. Converting...")
                            # If somehow we got 3 channels, convert to grayscale
                            if image.size(1) == 3:
                                # Convert RGB to grayscale using luminance formula
                                image = 0.299 * image[:, 0:1, :, :] + 0.587 * image[:, 1:2, :, :] + 0.114 * image[:, 2:3, :, :]
                        
                        print(f"Input tensor shape: {image.shape}")
                        
                        # For max length prediction
                        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(self.device)
                        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(self.device)
                        
                        if 'CTC' in opt.Prediction:
                            preds = self.model(image, text_for_pred)
                            
                            # Select max probability (greedy decoding) then decode index to character
                            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                            _, preds_index = preds.max(2)
                            preds_str = self.converter.decode(preds_index, preds_size)
                            
                        else:
                            preds = self.model(image, text_for_pred, is_train=False)
                            
                            # Select max probability (greedy decoding) then decode index to character
                            _, preds_index = preds.max(2)
                            preds_str = self.converter.decode(preds_index, length_for_pred)
                        
                        # Calculate confidence scores
                        preds_prob = F.softmax(preds, dim=2)
                        preds_max_prob, _ = preds_prob.max(dim=2)
                        
                        results = []
                        for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                            if 'Attn' in opt.Prediction:
                                pred_EOS = pred.find('[s]')
                                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                                pred_max_prob = pred_max_prob[:pred_EOS]
                            
                            # Calculate confidence score (= multiply of pred_max_prob)
                            confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
                            
                            results.append({
                                'image_path': img_name,
                                'recognized_text': pred,
                                'confidence_score': confidence_score
                            })
                        
                        # Return first result (should be only one)
                        if results:
                            return results[0]
                        else:
                            return {
                                'recognized_text': '',
                                'confidence_score': 0.0
                            }
                
            finally:
                # Clean up temporary folder
                shutil.rmtree(temp_folder, ignore_errors=True)
            
        except Exception as e:
            print(f"Error in deep recognition processing: {e}")
            import traceback
            traceback.print_exc()
            return {
                'recognized_text': '',
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    def save_ocr_result(self, output_path: str, result: Dict = None) -> bool:
        """
        Save OCR result to file
        
        Args:
            output_path: Path to save result
            result: OCR result (if None, uses last result)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if result is None:
                result = self.last_ocr_result
            
            if result is None or 'error' in result:
                print("⚠️ No valid OCR result to save")
                return False
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ OCR result saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving OCR result: {e}")
            return False
    
    def get_last_result(self) -> Optional[Dict]:
        """Get last OCR result"""
        return self.last_ocr_result
    
    def get_last_image(self) -> Optional[np.ndarray]:
        """Get last processed image"""
        return self.last_processed_image
    
    def get_ocr_history(self) -> list:
        """Get OCR history"""
        return self.ocr_history
    
    def clear_history(self):
        """Clear OCR history"""
        self.ocr_history = []


# Global instance for easy access
_global_ocr_model = None

def get_global_ocr_model() -> DeepOCRModel:
    """Get global OCR model instance"""
    global _global_ocr_model
    if _global_ocr_model is None:
        _global_ocr_model = DeepOCRModel()
    return _global_ocr_model

def initialize_ocr_model(model_path: str, cuda: bool = True) -> DeepOCRModel:
    """
    Initialize global OCR model
    
    Args:
        model_path: Path to OCR model (.pth file)
        cuda: Use CUDA if available
        
    Returns:
        DeepOCRModel instance
    """
    global _global_ocr_model
    _global_ocr_model = DeepOCRModel(model_path, cuda)
    return _global_ocr_model

def recognize_text_from_path(image_path: str) -> Dict:
    """
    Quick function to recognize text from image path using global OCR model
    
    Args:
        image_path: Path to image file
        
    Returns:
        OCR result dictionary
    """
    ocr_model = get_global_ocr_model()
    if ocr_model.model is None:
        raise ValueError("Global OCR model not initialized. Call initialize_ocr_model() first.")
    
    return ocr_model.recognize_text_from_image_path(image_path)

def recognize_text_from_array(image_array: np.ndarray) -> Dict:
    """
    Quick function to recognize text from image array using global OCR model
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        OCR result dictionary
    """
    ocr_model = get_global_ocr_model()
    if ocr_model.model is None:
        raise ValueError("Global OCR model not initialized. Call initialize_ocr_model() first.")
    
    return ocr_model.recognize_text_from_image_array(image_array)

def recognize_text_from_craft_lines(craft_result: Dict) -> Dict:
    """
    Quick function to recognize text from CRAFT lines using global OCR model
    
    Args:
        craft_result: Result from craft_line_detection.py
        
    Returns:
        OCR result dictionary
    """
    ocr_model = get_global_ocr_model()
    if ocr_model.model is None:
        raise ValueError("Global OCR model not initialized. Call initialize_ocr_model() first.")
    
    return ocr_model.recognize_text_from_craft_lines(craft_result)


class AdvancedDemoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced OCR Demo with Text Recognition")
        self.root.geometry("1400x900")
        
        # OCR model variables
        self.ocr_model = None
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
        """Process a single image with OCR recognition"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return f"Error: Could not read image {image_path}"
            
            # Use OCR model for recognition
            if self.ocr_model is None:
                # Initialize OCR model if not already done
                model_path = self.settings['demo_model_path']
                if os.path.exists(model_path):
                    self.ocr_model = DeepOCRModel(model_path)
                else:
                    return f"Error: Model file not found at {model_path}"
            
            # Recognize text
            ocr_result = self.ocr_model.recognize_text_from_image_path(image_path)
            
            if 'error' in ocr_result:
                return f"OCR Error: {ocr_result['error']}"
            
            recognized_text = ocr_result.get('recognized_text', '')
            confidence_score = ocr_result.get('confidence_score', 0.0)
            
            if recognized_text:
                return f"Recognized Text: '{recognized_text}' (Confidence: {confidence_score:.4f})"
            else:
                return "No text recognized"
            
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
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
                    image = cuda_resize(image, (new_width, new_height))
                
                # Convert to PIL Image
                image_rgb = cuda_cvtColor(image, cv2.COLOR_BGR2RGB)
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

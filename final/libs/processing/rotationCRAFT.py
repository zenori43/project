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
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Optional, Union
import logging
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

# Import CRAFT modules
import sys
# Get CRAFT path from config if available, otherwise use default
try:
    from config.settings import CRAFT_PYTORCH_DIR
    craft_path = CRAFT_PYTORCH_DIR
except ImportError:
    # Fallback to old path if config not available
    craft_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch"

if craft_path not in sys.path:
    sys.path.append(craft_path)

# Force clear any existing imports
import sys
import os

# Clear any existing CRAFT-related modules from cache (be more selective)
modules_to_clear = []
for mod in sys.modules.keys():
    if (('craft' in mod.lower() and 'rotationCRAFT' not in mod) or 
        'vgg16' in mod.lower() or 
        'basenet' in mod.lower()):
        modules_to_clear.append(mod)

for mod in modules_to_clear:
    if mod in sys.modules:
        try:
            del sys.modules[mod]
        except KeyError:
            pass  # Module might have been deleted already

# Add CRAFT paths (craft_path already set above)
basenet_path = os.path.join(craft_path, "basenet")

if craft_path not in sys.path:
    sys.path.insert(0, craft_path)
if basenet_path not in sys.path:
    sys.path.insert(0, basenet_path)

# CRAFT modules will be imported when needed to avoid import errors
CRAFT_MODULES_AVAILABLE = False
vgg16_bn = None
init_weights = None
CRAFT = None
craft_utils = None
file_utils = None
imgproc = None

def _import_craft_modules():
    """Import CRAFT modules when needed"""
    global CRAFT_MODULES_AVAILABLE, vgg16_bn, init_weights, CRAFT, craft_utils, file_utils, imgproc
    
    if CRAFT_MODULES_AVAILABLE:
        return True
    
    try:
        print("🔧 Importing CRAFT modules...")
        
        # Import basenet modules first with fallback
        try:
            from vgg16_bn import vgg16_bn, init_weights
            print("✅ vgg16_bn imported successfully")
        except ImportError as e:
            print(f"⚠️ vgg16_bn import failed: {e}")
            print("🔧 Trying fallback import...")
            # Try importing from basenet directory
            try:
                from basenet.vgg16_bn import vgg16_bn, init_weights
                print("✅ vgg16_bn imported from basenet directory")
            except ImportError as e2:
                print(f"❌ Fallback import failed: {e2}")
                raise e2
        
        # Import other CRAFT modules
        import craft_utils
        import file_utils
        
        # Import CRAFT with proper vgg16_bn handling
        try:
            # First, inject vgg16_bn into the craft module before importing CRAFT
            import craft
            craft.vgg16_bn = vgg16_bn
            craft.init_weights = init_weights
            CRAFT = craft.CRAFT
            print("✅ CRAFT imported with vgg16_bn injection")
        except Exception as e:
            print(f"⚠️ CRAFT import with injection failed: {e}")
            # Try direct import as fallback
            try:
                from craft import CRAFT
                print("✅ CRAFT imported directly")
            except Exception as e2:
                print(f"❌ Direct CRAFT import also failed: {e2}")
                raise e2
        
        # Import imgproc with error handling for numpy compatibility
        try:
            import imgproc
        except ValueError as e:
            if "numpy.dtype size changed" in str(e):
                print("⚠️ Warning: numpy/scikit-image compatibility issue detected")
                print("   Trying alternative import method...")
                # Try to import with specific numpy version handling
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning)
                import imgproc
            else:
                raise e
        
        # Make modules available globally
        globals()['vgg16_bn'] = vgg16_bn
        globals()['init_weights'] = init_weights
        globals()['CRAFT'] = CRAFT
        globals()['craft_utils'] = craft_utils
        globals()['file_utils'] = file_utils
        globals()['imgproc'] = imgproc
        
        CRAFT_MODULES_AVAILABLE = True
        print("✅ CRAFT modules imported successfully")
        print(f"✅ CRAFT class: {CRAFT}")
        print(f"✅ vgg16_bn class: {vgg16_bn}")
        return True
        
    except ImportError as e:
        print(f"❌ CRAFT import error: {e}")
        print("⚠️ CRAFT modules not available - some functions will be disabled")
        return False

def copyStateDict(state_dict):
    """Copy state dict for CRAFT model"""
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class CRAFTTextDetector:
    """
    Library สำหรับตรวจจับข้อความและหมุนภาพโดยใช้ CRAFT model
    สามารถเรียกใช้จาก code อื่นได้
    """
    
    def __init__(self, model_path: str = None, refiner_path: str = None, cuda: bool = True):
        """
        Initialize CRAFTTextDetector
        
        Args:
            model_path: Path to CRAFT model (.pth file)
            refiner_path: Path to CRAFT refiner model (.pth file) - optional
            cuda: Use CUDA if available
        """
        self.model = None
        self.refine_net = None
        # Use CUDA if available (CUDA paths already set at module level)
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        if self.cuda:
            print(f"🚀 PyTorch CUDA enabled - using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ PyTorch CUDA not available - using CPU for rotation CRAFT")
        
        # Detection parameters
        self.text_threshold = 0.7
        self.low_text = 0.4
        self.link_threshold = 0.4
        self.canvas_size = 1280
        self.mag_ratio = 1.5
        self.poly = False
        
        # Global variables for storing results
        self.last_detection_result = None
        self.last_processed_image = None
        self.last_rotated_image = None
        self.detection_history = []
        
        if model_path:
            self.load_model(model_path, refiner_path)
    
    def load_model(self, model_path: str, refiner_path: str = None) -> bool:
        """
        Load CRAFT model and refiner from paths
        
        Args:
            model_path: Path to .pth model file
            refiner_path: Path to .pth refiner file (optional)
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Import CRAFT modules if not already imported
            if not CRAFT_MODULES_AVAILABLE:
                print("🔧 CRAFT modules not imported yet, importing now...")
                if not _import_craft_modules():
                    raise Exception("Failed to import CRAFT modules")
            
            # Initialize CRAFT model with safe approach
            try:
                print("🔧 Creating CRAFT model...")
                
                # Check if CRAFT is available
                if CRAFT is None:
                    print("❌ CRAFT is None - modules not imported properly")
                    raise Exception("CRAFT module not available")
                
                # Create CRAFT model
                self.model = CRAFT()
                print("✅ CRAFT model created successfully")
                
                # Initialize weights if possible
                try:
                    if hasattr(self.model, 'apply') and init_weights is not None:
                        self.model.apply(init_weights)
                        print("✅ Model weights initialized")
                except Exception as e_init:
                    print(f"⚠️ Could not initialize weights: {e_init}")
                    print("   Continuing without weight initialization...")
                
            except Exception as e:
                print(f"❌ Error creating CRAFT model: {e}")
                print("🔧 Trying to re-import CRAFT modules...")
                
                # Try to re-import CRAFT modules
                try:
                    # Ensure vgg16_bn is available in craft module
                    import craft
                    craft.vgg16_bn = vgg16_bn
                    craft.init_weights = init_weights
                    CRAFT_NEW = craft.CRAFT
                    print("✅ CRAFT re-imported successfully")
                    
                    # Try creating model again
                    self.model = CRAFT_NEW()
                    print("✅ CRAFT model created after re-import")
                    
                except Exception as e2:
                    print(f"❌ Re-import failed: {e2}")
                    print("🔧 Trying direct import from craft module...")
                    
                    try:
                        # Try direct import with vgg16_bn injection
                        import craft
                        craft.vgg16_bn = vgg16_bn
                        craft.init_weights = init_weights
                        self.model = craft.CRAFT()
                        print("✅ CRAFT model created with direct import")
                        
                    except Exception as e3:
                        print(f"❌ Direct import failed: {e3}")
                        raise Exception("CRAFT model creation failed")
            
            print('Loading main model weights from checkpoint (' + model_path + ')')
            try:
                if self.cuda:
                    self.model.load_state_dict(copyStateDict(torch.load(model_path)))
                    self.model = self.model.cuda()
                    self.model = torch.nn.DataParallel(self.model)
                    cudnn.benchmark = False
                else:
                    self.model.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))
                print("✅ Model weights loaded successfully")
            except Exception as e:
                print(f"❌ Error loading model weights: {e}")
                print("🔧 Trying to load with strict=False...")
                
                # Try loading with strict=False
                try:
                    if self.cuda:
                        self.model.load_state_dict(copyStateDict(torch.load(model_path)), strict=False)
                        self.model = self.model.cuda()
                        self.model = torch.nn.DataParallel(self.model)
                        cudnn.benchmark = False
                    else:
                        self.model.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')), strict=False)
                    print("✅ Model weights loaded with strict=False")
                except Exception as e2:
                    print(f"❌ Still cannot load model weights: {e2}")
                    print("🔧 Trying to load without DataParallel...")
                    
                    # Try loading without DataParallel
                    try:
                        if self.cuda:
                            self.model.load_state_dict(copyStateDict(torch.load(model_path)), strict=False)
                            self.model = self.model.cuda()
                            cudnn.benchmark = False
                        else:
                            self.model.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')), strict=False)
                        print("✅ Model weights loaded without DataParallel")
                    except Exception as e3:
                        print(f"❌ Still cannot load model weights: {e3}")
                        raise e3
            
            self.model.eval()
            print(f"✓ Main model loaded successfully from: {model_path}")
            
            # Load refiner model if provided
            if refiner_path:
                if not os.path.exists(refiner_path):
                    print(f"⚠️ Warning: Refiner file not found: {refiner_path}")
                    print("   Continuing without refiner model...")
                else:
                    try:
                        # Import refiner model
                        from refinenet import RefineNet
                        
                        self.refine_net = RefineNet()
                        print('Loading refiner weights from checkpoint (' + refiner_path + ')')
                        
                        if self.cuda:
                            self.refine_net.load_state_dict(copyStateDict(torch.load(refiner_path)))
                            self.refine_net = self.refine_net.cuda()
                            self.refine_net = torch.nn.DataParallel(self.refine_net)
                        else:
                            self.refine_net.load_state_dict(copyStateDict(torch.load(refiner_path, map_location='cpu')))
                        
                        self.refine_net.eval()
                        print(f"✓ Refiner model loaded successfully from: {refiner_path}")
                        
                    except ImportError as e:
                        print(f"⚠️ Warning: Could not import RefineNet: {e}")
                        print("   Continuing without refiner model...")
                    except Exception as e:
                        print(f"⚠️ Warning: Error loading refiner model: {e}")
                        print("   Continuing without refiner model...")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect_text_and_rotate(self, image_path: str = None, image_array: np.ndarray = None) -> Dict:
        """
        Detect text and calculate rotation angle
        
        Args:
            image_path: Path to image file
            image_array: Image as numpy array (if image_path is None)
            
        Returns:
            Dict containing detection and rotation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load model first.")
        
        try:
            # Load image
            if image_path:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = imgproc.loadImage(image_path)  # RGB format
            elif image_array is not None:
                # Ensure image is contiguous and properly aligned
                if not image_array.flags['C_CONTIGUOUS']:
                    image_array = np.ascontiguousarray(image_array)
                
                # Convert BGR to RGB if needed
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    # Try CUDA first, fallback to CPU if it fails
                    try:
                        image = cuda_cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    except:
                        # Fallback to CPU
                        image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                else:
                    image = image_array.copy()
            else:
                raise ValueError("Either image_path or image_array must be provided")
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Store original image
            self.last_processed_image = image.copy()
            
            # Perform text detection
            t0 = time.time()
            
            # Resize image
            img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
                image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio
            )
            ratio_h = ratio_w = 1 / target_ratio
            
            # Preprocessing
            x = imgproc.normalizeMeanVariance(img_resized)
            x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
            x = x.unsqueeze(0)
            if self.cuda:
                x = x.cuda()
            
            # Forward pass
            with torch.no_grad():
                y, feature = self.model(x)
            
            # Make score and link map
            score_text = y[0,:,:,0].cpu().data.numpy()
            score_link = y[0,:,:,1].cpu().data.numpy()
            
            # Refine link if available
            if self.refine_net is not None:
                with torch.no_grad():
                    y_refiner = self.refine_net(y, feature)
                score_link = y_refiner[0,:,:,0].cpu().data.numpy()
            
            inference_time = time.time() - t0
            t1 = time.time()
            
            # Post-processing
            boxes, polys = craft_utils.getDetBoxes(
                score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly
            )
            
            # Coordinate adjustment
            boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
            polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
            for k in range(len(polys)):
                if polys[k] is None: 
                    polys[k] = boxes[k]
            
            postproc_time = time.time() - t1
            
            # Calculate rotation angle
            angle = 0
            final_polys = []
            if len(polys) > 0:
                try:
                    all_points = np.concatenate(polys, axis=0)
                    rect = cv2.minAreaRect(all_points)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    final_polys.append(box)
                    
                    # Calculate angle from bottom edge
                    sorted_corners = sorted(box, key=lambda p: p[1], reverse=True)
                    bottom_p1 = sorted_corners[0]
                    bottom_p2 = sorted_corners[1]
                    
                    if bottom_p1[0] < bottom_p2[0]:
                        origin_point = tuple(bottom_p1)
                        bottom_right_corner = tuple(bottom_p2)
                    else:
                        origin_point = tuple(bottom_p2)
                        bottom_right_corner = tuple(bottom_p1)
                    
                    dx = bottom_right_corner[0] - origin_point[0]
                    dy = bottom_right_corner[1] - origin_point[1]
                    angle = np.degrees(np.arctan2(-dy, dx))
                except Exception as e:
                    print(f"⚠️ CRAFT: Error calculating rotation angle: {e}")
                    angle = 0
            
            # Rotate image
            rotated_image = None
            if abs(angle) > 0.5:  # Only rotate if angle is significant (> 0.5 degrees)
                try:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                    # Ensure image is contiguous before rotation
                    if not image.flags['C_CONTIGUOUS']:
                        image = np.ascontiguousarray(image)
                    rotated_image = cv2.warpAffine(image, M, (w, h))
                    if rotated_image is not None:
                        self.last_rotated_image = rotated_image.copy()
                        print(f"✅ CRAFT: Rotated image by {angle:.2f} degrees")
                    else:
                        print(f"⚠️ CRAFT: Rotation failed - rotated_image is None")
                except Exception as e:
                    print(f"⚠️ CRAFT: Error rotating image: {e}")
                    rotated_image = None
            else:
                # No rotation needed (angle is too small)
                rotated_image = image.copy()
                print(f"ℹ️ CRAFT: No rotation needed (angle: {angle:.2f} degrees)")
            
            # Create result dictionary
            result = {
                'image_path': image_path,
                'text_boxes': boxes,
                'text_polys': polys,
                'final_polys': final_polys,
                'rotation_angle': angle,
                'total_text_regions': len(boxes),
                'image_shape': image.shape,
                'rotated_image': rotated_image,
                'inference_time': inference_time,
                'postproc_time': postproc_time,
                'timestamp': str(np.datetime64('now'))
            }
            
            # Store in global variables
            self.last_detection_result = result
            self.detection_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return {
                'error': str(e),
                'text_boxes': [],
                'text_polys': [],
                'rotation_angle': 0,
                'total_text_regions': 0
            }
    
    def draw_detections(self, image: np.ndarray = None, result: Dict = None) -> np.ndarray:
        """
        Draw text detection boxes on image
        
        Args:
            image: Input image (if None, uses last processed image)
            result: Detection result (if None, uses last detection result)
            
        Returns:
            Image with detection boxes drawn
        """
        if image is None:
            image = self.last_processed_image
        if result is None:
            result = self.last_detection_result
            
        if image is None or result is None:
            raise ValueError("No image or detection result available")
        
        # Create copy of image (convert RGB to BGR for OpenCV)
        output_image = cuda_cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw text detection boxes
        if 'final_polys' in result and result['final_polys']:
            for poly in result['final_polys']:
                cv2.polylines(output_image, [poly], True, (0, 255, 0), 2)
        
        # Draw angle information
        if 'rotation_angle' in result:
            angle = result['rotation_angle']
            cv2.putText(output_image, f"Angle: {angle:.2f} deg", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output_image
    
    def save_detection_result(self, output_path: str, image: np.ndarray = None, result: Dict = None) -> bool:
        """
        Save detection result image
        
        Args:
            output_path: Path to save the result image
            image: Image with detections drawn
            result: Detection result for JSON metadata
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if image is None:
                image = self.draw_detections()
            
            # Save image
            cv2.imwrite(output_path, image)
            
            # Save metadata as JSON
            if result is None:
                result = self.last_detection_result
            
            if result:
                json_path = output_path.replace('.jpg', '.json').replace('.png', '.json')
                # Convert numpy arrays to lists for JSON serialization
                json_result = result.copy()
                if 'text_boxes' in json_result:
                    json_result['text_boxes'] = [box.tolist() if hasattr(box, 'tolist') else box for box in json_result['text_boxes']]
                if 'text_polys' in json_result:
                    json_result['text_polys'] = [poly.tolist() if hasattr(poly, 'tolist') else poly for poly in json_result['text_polys']]
                if 'final_polys' in json_result:
                    json_result['final_polys'] = [poly.tolist() if hasattr(poly, 'tolist') else poly for poly in json_result['final_polys']]
                if 'rotated_image' in json_result:
                    del json_result['rotated_image']  # Don't save image data in JSON
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            print(f"Detection result saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving detection result: {e}")
            return False
    
    def save_rotated_image(self, output_path: str, result: Dict = None) -> bool:
        """
        Save rotated image
        
        Args:
            output_path: Path to save the rotated image
            result: Detection result (if None, uses last detection result)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if result is None:
                result = self.last_detection_result
            
            if result and 'rotated_image' in result and result['rotated_image'] is not None:
                # Convert RGB to BGR for saving
                rotated_bgr = cuda_cvtColor(result['rotated_image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, rotated_bgr)
                print(f"Rotated image saved to: {output_path}")
                return True
            else:
                print("No rotated image available")
                return False
                
        except Exception as e:
            print(f"Error saving rotated image: {e}")
            return False
    
    def get_last_result(self) -> Optional[Dict]:
        """Get last detection result"""
        return self.last_detection_result
    
    def get_last_image(self) -> Optional[np.ndarray]:
        """Get last processed image"""
        return self.last_processed_image
    
    def get_last_rotated_image(self) -> Optional[np.ndarray]:
        """Get last rotated image"""
        return self.last_rotated_image
    
    def get_detection_history(self) -> List[Dict]:
        """Get detection history"""
        return self.detection_history
    
    def clear_history(self):
        """Clear detection history"""
        self.detection_history = []
    
    def set_detection_parameters(self, text_threshold: float = 0.7, low_text: float = 0.4, 
                                link_threshold: float = 0.4, canvas_size: int = 1280, 
                                mag_ratio: float = 1.5):
        """Set detection parameters"""
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio


# Global instance for easy access
_global_craft_detector = None

def get_global_detector() -> CRAFTTextDetector:
    """Get global detector instance"""
    global _global_craft_detector
    if _global_craft_detector is None:
        _global_craft_detector = CRAFTTextDetector()
    return _global_craft_detector

def initialize_detector(model_path: str, refiner_path: str = None, cuda: bool = True) -> CRAFTTextDetector:
    """
    Initialize global detector with model and optional refiner
    
    Args:
        model_path: Path to CRAFT model
        refiner_path: Path to CRAFT refiner model (optional)
        cuda: Use CUDA if available
        
    Returns:
        CRAFTTextDetector instance
    """
    global _global_craft_detector
    _global_craft_detector = CRAFTTextDetector(model_path, refiner_path, cuda)
    return _global_craft_detector

def detect_text_and_rotate_from_path(image_path: str) -> Dict:
    """
    Quick function to detect text and rotate from image path using global detector
    
    Args:
        image_path: Path to image file
        
    Returns:
        Detection result dictionary
    """
    detector = get_global_detector()
    if detector.model is None:
        raise ValueError("Global detector not initialized. Call initialize_detector() first.")
    
    return detector.detect_text_and_rotate(image_path)

def detect_text_and_rotate_from_array(image_array: np.ndarray) -> Dict:
    """
    Quick function to detect text and rotate from image array using global detector
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        Detection result dictionary
    """
    detector = get_global_detector()
    if detector.model is None:
        raise ValueError("Global detector not initialized. Call initialize_detector() first.")
    
    return detector.detect_text_and_rotate(image_array=image_array)


# Example usage and testing
if __name__ == "__main__":
    # Example: Initialize detector with your models
    model_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch/craft_mlt_25k.pth"
    refiner_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/CRAFT-pytorch/craft_refiner_CTW1500.pth"
    
    try:
        # Initialize global detector with both models
        detector = initialize_detector(model_path, refiner_path, cuda=True)
        print("CRAFT detector initialized successfully!")
        
        # Check if refiner is loaded
        if detector.refine_net is not None:
            print("✓ Using both main model and refiner model")
        else:
            print("✓ Using main model only")
        
        # Example: Detect text and rotate from image path
        # result = detect_text_and_rotate_from_path("path/to/your/image.jpg")
        # print(f"Found {result['total_text_regions']} text regions")
        # print(f"Rotation angle: {result['rotation_angle']:.2f} degrees")
        
        # Example: Get detection history
        # history = detector.get_detection_history()
        # print(f"Detection history: {len(history)} entries")
        
    except Exception as e:
        print(f"Error during initialization: {e}")

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
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Union
import logging

# Try to import ultralytics for YOLOv8
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    print("✅ Ultralytics (YOLOv8) available")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ Ultralytics not available, please install: pip install ultralytics")

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("✅ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime not available")

class CapDetector:
    """
    Library สำหรับตรวจจับฝาโดยใช้ YOLOv8 model
    สามารถเรียกใช้จาก code อื่นได้
    """
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.5):
        """
        Initialize CapDetector
        
        Args:
            model_path: Path to YOLOv8 model (.pt file)
            conf_threshold: Confidence threshold for detection
        """
        self.model = None
        self.conf_threshold = conf_threshold
        # Use CUDA if available (CUDA paths already set at module level)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"🚀 PyTorch CUDA enabled - using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ PyTorch CUDA not available - using CPU for cap detection")
        
        # Global variables for storing results
        self.last_detection_result = None
        self.last_processed_image = None
        self.detection_history = []
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load YOLOv8 model from path
        
        Args:
            model_path: Path to YOLOv8 .pt model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if not ULTRALYTICS_AVAILABLE:
                print("❌ Ultralytics not available. Please install: pip install ultralytics")
                return False
            
            # Load YOLOv8 model using ultralytics
            self.model = YOLO(model_path)
            
            # Check if torchvision NMS supports CUDA (Jetson torchvision may not have CUDA NMS)
            # If not, we need to use CPU for YOLO model to avoid torchvision::nms errors
            try:
                import torchvision
                if self.device.type == 'cuda':
                    # Test if torchvision NMS can run on CUDA
                    test_boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], device='cuda', dtype=torch.float32)
                    test_scores = torch.tensor([0.9, 0.8], device='cuda', dtype=torch.float32)
                    try:
                        from torchvision.ops import nms
                        _ = nms(test_boxes, test_scores, 0.5)
                        # torchvision NMS works on CUDA, use CUDA
                        self.model.to('cuda')
                        print(f"✅ YOLOv8 model loaded successfully from: {model_path}")
                        print(f"🚀 Using CUDA (GPU) for YOLO model")
                    except Exception:
                        # torchvision NMS doesn't work on CUDA, use CPU
                        self.model.to('cpu')
                        self.device = torch.device('cpu')  # Update device to CPU
                        print(f"✅ YOLOv8 model loaded successfully from: {model_path}")
                        print(f"⚠️ Using CPU for YOLO model (torchvision NMS doesn't support CUDA on this system)")
                else:
                    self.model.to('cpu')
                    print(f"✅ YOLOv8 model loaded successfully from: {model_path}")
                    print(f"⚠️ Using CPU for YOLO model (CUDA not available)")
            except Exception as e:
                # Fallback: use CPU if torchvision check fails
                self.model.to('cpu')
                self.device = torch.device('cpu')
                print(f"✅ YOLOv8 model loaded successfully from: {model_path}")
                print(f"⚠️ Using CPU for YOLO model (torchvision check failed: {e})")
            
            print(f"Model info: {self.model.info()}")
            return True
            
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            return False
    
    def detect_caps(self, image_path: str = None, image_array: np.ndarray = None) -> Dict:
        """
        Detect caps in image using YOLOv8
        
        Args:
            image_path: Path to image file
            image_array: Image as numpy array (if image_path is None)
            
        Returns:
            Dict containing detection results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load model first.")
        
        try:
            # Load image
            if image_path:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = cv2.imread(image_path)
            elif image_array is not None:
                image = image_array.copy()
            else:
                raise ValueError("Either image_path or image_array must be provided")
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Store original image
            original_image = image.copy()
            self.last_processed_image = original_image.copy()
            
            # Resize image for faster detection (reduce 2K image to smaller size)
            # YOLO works well with smaller images and is much faster
            original_height, original_width = image.shape[:2]
            max_dimension = max(original_height, original_width)
            
            # If image is larger than 1280px, resize it for detection
            # Keep aspect ratio and resize to max 1280px on longest side
            if max_dimension > 1280:
                scale_factor = 1280.0 / max_dimension
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                # Use CUDA resize if available, otherwise CPU
                from core.cuda_image_utils import cuda_resize
                resized_image = cuda_resize(image, (new_width, new_height))
                print(f"📐 Resized image for detection: {original_width}x{original_height} -> {new_width}x{new_height} (scale: {scale_factor:.3f})")
            else:
                resized_image = image
                scale_factor = 1.0
            
            # Run YOLOv8 detection on resized image
            # If model is on CPU (due to torchvision NMS limitation), run on CPU
            # Otherwise run normally (model will use its current device)
            try:
                results = self.model(resized_image, conf=self.conf_threshold, verbose=False, device=str(self.device))
            except Exception as e:
                # If CUDA error occurs (e.g., torchvision::nms), fallback to CPU
                if 'torchvision::nms' in str(e) or 'CUDA' in str(e):
                    print(f"⚠️ CUDA error detected, falling back to CPU: {e}")
                    self.model.to('cpu')
                    self.device = torch.device('cpu')
                    results = self.model(resized_image, conf=self.conf_threshold, verbose=False, device='cpu')
                else:
                    raise
            
            # Process results
            detections = []
            
            if len(results) > 0:
                result = results[0]  # Get first result
                
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        # Get box coordinates (xyxy format) from resized image
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Scale bbox coordinates back to original image size
                        if scale_factor != 1.0:
                            x1 = int(x1 / scale_factor)
                            y1 = int(y1 / scale_factor)
                            x2 = int(x2 / scale_factor)
                            y2 = int(y2 / scale_factor)
                        else:
                            x1 = int(x1)
                            y1 = int(y1)
                            x2 = int(x2)
                            y2 = int(y2)
                        
                        # Get confidence and class
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[cls] if cls in self.model.names else f"class_{cls}"
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': class_name
                        }
                        detections.append(detection)
            
            # Create result dictionary
            result = {
                'image_path': image_path,
                'detections': detections,
                'total_detections': len(detections),
                'image_shape': image.shape,
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
                'detections': [],
                'total_detections': 0
            }
    
    def draw_detections(self, image: np.ndarray = None, result: Dict = None) -> np.ndarray:
        """
        Draw detection boxes on image
        
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
        
        # Create copy of image
        output_image = image.copy()
        
        # Draw detection boxes
        for detection in result['detections']:
            bbox = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            
            # Draw rectangle
            cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(output_image, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Detection result saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving detection result: {e}")
            return False
    
    def get_last_result(self) -> Optional[Dict]:
        """Get last detection result"""
        return self.last_detection_result
    
    def get_last_image(self) -> Optional[np.ndarray]:
        """Get last processed image"""
        return self.last_processed_image
    
    def get_detection_history(self) -> List[Dict]:
        """Get detection history"""
        return self.detection_history
    
    def clear_history(self):
        """Clear detection history"""
        self.detection_history = []
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold"""
        self.conf_threshold = max(0.0, min(1.0, threshold))
    
    def crop_detections(self, image: np.ndarray = None, result: Dict = None, margin: int = 10) -> List[np.ndarray]:
        """
        Crop detected regions from image
        
        Args:
            image: Input image (if None, uses last processed image)
            result: Detection result (if None, uses last detection result)
            margin: Extra margin around detection box in pixels
            
        Returns:
            List of cropped images
        """
        if image is None:
            image = self.last_processed_image
        if result is None:
            result = self.last_detection_result
            
        if image is None or result is None:
            raise ValueError("No image or detection result available")
        
        cropped_images = []
        
        for detection in result['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Add margin
            h, w = image.shape[:2]
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            # Crop the region
            cropped = image[y1:y2, x1:x2]
            cropped_images.append(cropped)
        
        return cropped_images
    
    def save_cropped_detections(self, output_dir: str = "cropped_detections", 
                               image: np.ndarray = None, result: Dict = None, 
                               margin: int = 10) -> List[str]:
        """
        Save cropped detection regions
        
        Args:
            output_dir: Directory to save cropped images
            image: Input image (if None, uses last processed image)
            result: Detection result (if None, uses last detection result)
            margin: Extra margin around detection box in pixels
            
        Returns:
            List of saved file paths
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get cropped images
            cropped_images = self.crop_detections(image, result, margin)
            
            saved_paths = []
            
            for i, cropped_img in enumerate(cropped_images):
                # Generate filename
                timestamp = str(np.datetime64('now')).replace(':', '-').replace('.', '-')
                filename = f"detection_{i+1}_{timestamp}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Save image
                cv2.imwrite(filepath, cropped_img)
                saved_paths.append(filepath)
                print(f"Saved cropped detection {i+1}: {filepath}")
            
            return saved_paths
            
        except Exception as e:
            print(f"Error saving cropped detections: {e}")
            return []


# Global instance for easy access
_global_cap_detector = None

def get_global_detector() -> CapDetector:
    """Get global detector instance"""
    global _global_cap_detector
    if _global_cap_detector is None:
        _global_cap_detector = CapDetector()
    return _global_cap_detector

def initialize_detector(model_path: str, conf_threshold: float = 0.5) -> CapDetector:
    """
    Initialize global detector with YOLOv8 model
    
    Args:
        model_path: Path to YOLOv8 model (.pt file)
        conf_threshold: Confidence threshold
        
    Returns:
        CapDetector instance
    """
    global _global_cap_detector
    _global_cap_detector = CapDetector(model_path, conf_threshold)
    return _global_cap_detector

def detect_caps_from_path(image_path: str) -> Dict:
    """
    Quick function to detect caps from image path using global detector
    
    Args:
        image_path: Path to image file
        
    Returns:
        Detection result dictionary
    """
    detector = get_global_detector()
    if detector.model is None:
        # Try to auto-initialize with default model path
        default_model_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/final/cap.pt"
        if os.path.exists(default_model_path):
            print(f"Auto-initializing detector with default model: {default_model_path}")
            initialize_detector(default_model_path, conf_threshold=0.5)
            detector = get_global_detector()
        else:
            raise ValueError("Global detector not initialized and default model not found. Call initialize_detector() first.")
    
    return detector.detect_caps(image_path)

def detect_caps_from_array(image_array: np.ndarray) -> Dict:
    """
    Quick function to detect caps from image array using global detector
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        Detection result dictionary
    """
    detector = get_global_detector()
    if detector.model is None:
        raise ValueError("Global detector not initialized. Call initialize_detector() first.")
    
    return detector.detect_caps(image_array=image_array)

def crop_detections_from_path(image_path: str, margin: int = 10) -> List[np.ndarray]:
    """
    Quick function to detect and crop caps from image path using global detector
    
    Args:
        image_path: Path to image file
        margin: Extra margin around detection box in pixels
        
    Returns:
        List of cropped images
    """
    detector = get_global_detector()
    if detector.model is None:
        # Try to auto-initialize with default model path
        default_model_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/final/cap.pt"
        if os.path.exists(default_model_path):
            print(f"Auto-initializing detector with default model: {default_model_path}")
            initialize_detector(default_model_path, conf_threshold=0.5)
            detector = get_global_detector()
        else:
            raise ValueError("Global detector not initialized and default model not found. Call initialize_detector() first.")
    
    # Detect caps first
    result = detector.detect_caps(image_path)
    
    if 'error' in result:
        raise ValueError(f"Detection failed: {result['error']}")
    
    # Crop detections
    return detector.crop_detections(margin=margin)

def save_cropped_detections_from_path(image_path: str, output_dir: str = "cropped_detections", 
                                     margin: int = 10) -> List[str]:
    """
    Quick function to detect and save cropped caps from image path using global detector
    
    Args:
        image_path: Path to image file
        output_dir: Directory to save cropped images
        margin: Extra margin around detection box in pixels
        
    Returns:
        List of saved file paths
    """
    detector = get_global_detector()
    if detector.model is None:
        # Try to auto-initialize with default model path
        default_model_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/final/cap.pt"
        if os.path.exists(default_model_path):
            print(f"Auto-initializing detector with default model: {default_model_path}")
            initialize_detector(default_model_path, conf_threshold=0.5)
            detector = get_global_detector()
        else:
            raise ValueError("Global detector not initialized and default model not found. Call initialize_detector() first.")
    
    # Detect caps first
    result = detector.detect_caps(image_path)
    
    if 'error' in result:
        raise ValueError(f"Detection failed: {result['error']}")
    
    # Save cropped detections
    return detector.save_cropped_detections(output_dir=output_dir, margin=margin)


# Example usage and testing
if __name__ == "__main__":
    # Example: Initialize detector with your YOLOv8 model
    model_path = r"/home/nvidia/Desktop/final_boss/backupsdcard/final/cap.pt"
    
    try:
        # Initialize global detector
        detector = initialize_detector(model_path, conf_threshold=0.5)
        print("✅ YOLOv8 detector initialized successfully!")
        
        # Example: Detect caps from image path
        # result = detect_caps_from_path("path/to/your/image.jpg")
        # print(f"Found {result['total_detections']} caps")
        
        # Example: Get detection history
        # history = detector.get_detection_history()
        # print(f"Detection history: {len(history)} entries")
        
    except Exception as e:
        print(f"Error during initialization: {e}")

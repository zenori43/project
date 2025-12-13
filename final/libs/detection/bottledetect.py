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
from ultralytics import YOLO
import warnings
from typing import List, Dict, Optional
import torch

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CUDA/GPU SUPPORT ---
CUDA_AVAILABLE = torch.cuda.is_available()

# Load YOLO bottle model from config
try:
    from config.settings import BOTTLE_MODEL_PATH
    BOTTLE_MODEL = BOTTLE_MODEL_PATH
except ImportError:
    # Fallback to old path if config not available
    BOTTLE_MODEL = r"/home/nvidia/Desktop/final_boss/backupsdcard/final/bottle.pt"

bottle_model = YOLO(BOTTLE_MODEL)
# Check if torchvision NMS supports CUDA (Jetson torchvision may not have CUDA NMS)
if CUDA_AVAILABLE:
    try:
        import torchvision
        # Test if torchvision NMS can run on CUDA
        test_boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], device='cuda', dtype=torch.float32)
        test_scores = torch.tensor([0.9, 0.8], device='cuda', dtype=torch.float32)
        try:
            from torchvision.ops import nms
            _ = nms(test_boxes, test_scores, 0.5)
            # torchvision NMS works on CUDA, use CUDA
            bottle_model.to('cuda')
            print("✅ Bottle YOLO model using CUDA")
        except Exception:
            # torchvision NMS doesn't work on CUDA, use CPU
            bottle_model.to('cpu')
            CUDA_AVAILABLE = False  # Update flag
            print("⚠️ Bottle YOLO model using CPU (torchvision NMS doesn't support CUDA)")
    except Exception:
        bottle_model.to('cpu')
        CUDA_AVAILABLE = False
        print("⚠️ Bottle YOLO model using CPU (torchvision check failed)")
else:
    bottle_model.to('cpu')

# ===== BOTTLE DETECTION FUNCTIONS =====
def detect_bottles(image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Detect bottles in the image using YOLO model
    
    Args:
        image: Input image (BGR format)
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        List of detection dictionaries with bbox, label, confidence
    """
    results = bottle_model(image, conf=confidence_threshold)
    detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = bottle_model.names[class_id]
            
            detection = {
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'confidence': confidence,
                'type': 'bottle'
            }
            detections.append(detection)
    
    return detections

def crop_type_regions(image: np.ndarray, detections: List[Dict]) -> List[Dict]:
    """
    Crop type regions from detected bottles
    
    Args:
        image: Original image
        detections: List of bottle detections
        
    Returns:
        List of cropped type regions with metadata
    """
    type_crops = []
    
    for detection in detections:
        if detection['label'] == "type":
            x1, y1, x2, y2 = detection['bbox']
            roi = image[y1:y2, x1:x2].copy()
            
            type_crop = {
                'original_crop': roi,
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'label': detection['label']
            }
            type_crops.append(type_crop)
    
    return type_crops

# ===== MAIN DETECTION FUNCTION =====
def detect_bottle_and_crop_type(image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
    """
    Main function to detect bottles and crop type regions
    
    Args:
        image: Input image (BGR format)
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        Dictionary containing detection results and type crops
    """
    # Detect bottles
    detections = detect_bottles(image, confidence_threshold)
    
    # Crop type regions
    type_crops = crop_type_regions(image, detections)
    
    # Prepare result summary
    result_summary = {
        'total_detections': len(detections),
        'type_detections': len(type_crops),
        'detected_labels': list(set([d['label'] for d in detections])),
        'has_type': len(type_crops) > 0
    }
    
    return {
        'detections': detections,
        'type_crops': type_crops,
        'summary': result_summary,
        'original_image': image
    }

# ===== UTILITY FUNCTIONS =====
def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image or None if failed
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    
    return image

def save_cropped_type(crop_data: Dict, output_path: str) -> bool:
    """
    Save cropped type region to file
    
    Args:
        crop_data: Dictionary containing crop data
        output_path: Path to save the cropped image
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        cv2.imwrite(output_path, crop_data['original_crop'])
        return True
    except Exception as e:
        print(f"Error saving cropped image: {e}")
        return False

def draw_detections_on_image(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw detection boxes on image
    
    Args:
        image: Original image
        detections: List of detections
        
    Returns:
        Image with detection boxes drawn
    """
    result_image = image.copy()
    
    # Draw detection boxes
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        
        # Choose color based on label
        if label == "type":
            color = (0, 255, 0)  # Green for type
        else:
            color = (255, 0, 0)  # Red for other labels
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f"{label} ({confidence:.2f})"
        cv2.putText(result_image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image

# ===== SIMPLE API FUNCTIONS =====
def process_bottle_image_simple(image_path: str) -> Dict:
    """
    Simple function to process bottle image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Detection results dictionary
    """
    image = load_image(image_path)
    if image is None:
        return {"error": "Could not load image"}
    
    return detect_bottle_and_crop_type(image)

def get_type_crops_only(image_path: str) -> List[np.ndarray]:
    """
    Get only the cropped type regions from an image
    
    Args:
        image_path: Path to image file
        
    Returns:
        List of cropped type images
    """
    result = process_bottle_image_simple(image_path)
    
    if "error" in result:
        return []
    
    return [crop['original_crop'] for crop in result['type_crops']]

def get_detection_summary(image_path: str) -> Dict:
    """
    Get detection summary from an image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Summary dictionary with detection counts and labels
    """
    result = process_bottle_image_simple(image_path)
    
    if "error" in result:
        return {"error": result["error"]}
    
    return result['summary']

# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    # Example usage
    print("Bottle Detection Module (Simplified)")
    print("Available functions:")
    print("- detect_bottle_and_crop_type(image)")
    print("- process_bottle_image_simple(image_path)")
    print("- get_type_crops_only(image_path)")
    print("- get_detection_summary(image_path)")
    
    # Test with a sample image if provided
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nTesting with image: {image_path}")
        
        result = process_bottle_image_simple(image_path)
        
        if "error" not in result:
            print(f"Detections: {result['summary']['total_detections']}")
            print(f"Type regions found: {result['summary']['type_detections']}")
            print(f"Detected labels: {result['summary']['detected_labels']}")
            
            if result['type_crops']:
                print(f"Type crops available: {len(result['type_crops'])}")
        else:
            print(f"Error: {result['error']}")

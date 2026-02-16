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

# ความเร็ว: ย่อภาพก่อน YOLO และใช้ imgsz เล็กลง
YOLO_MAX_EDGE = 1280  # ถ้าภาพใหญ่กว่านี้จะย่อก่อนรัน YOLO
YOLO_IMGSZ = 640      # ขนาดที่ส่งเข้า YOLO (เล็ก = เร็วขึ้น)

# ===== BOTTLE DETECTION FUNCTIONS =====
def detect_bottles(image: np.ndarray, confidence_threshold: float = 0.5, imgsz: int = None) -> List[Dict]:
    """
    Detect bottles in the image using YOLO model
    
    Args:
        image: Input image (BGR format)
        confidence_threshold: Minimum confidence for detection
        imgsz: YOLO input size (default YOLO_IMGSZ for speed)
        
    Returns:
        List of detection dictionaries with bbox, label, confidence
    """
    sz = imgsz if imgsz is not None else YOLO_IMGSZ
    results = bottle_model(image, conf=confidence_threshold, imgsz=sz)
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


def crop_angle1_regions(image: np.ndarray, detections: List[Dict]) -> List[Dict]:
    """
    Crop angle1 regions from detected bottles (สำหรับส่งเข้า defect model)
    
    Args:
        image: Original image
        detections: List of bottle detections
        
    Returns:
        List of cropped angle1 regions with metadata
    """
    angle1_crops = []
    for detection in detections:
        if detection['label'] == "angle1":
            x1, y1, x2, y2 = detection['bbox']
            roi = image[y1:y2, x1:x2].copy()
            angle1_crops.append({
                'original_crop': roi,
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'label': detection['label']
            })
    return angle1_crops

# ===== MAIN DETECTION FUNCTION =====
def detect_bottle_and_crop_type(image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
    """
    Main function to detect bottles and crop type regions.
    ตรวจ defect (angle1) ก่อน — ถ้า NG ไม่ครอป type และไม่รัน OCR (ข้ามไปเลย)
    เร่งความเร็ว: ย่อภาพก่อน YOLO ถ้าใหญ่กว่า YOLO_MAX_EDGE
    """
    h, w = image.shape[:2]
    scale_factor = 1.0
    img_for_yolo = image
    if max(h, w) > YOLO_MAX_EDGE:
        scale_factor = max(h, w) / float(YOLO_MAX_EDGE)
        new_w = int(round(w / scale_factor))
        new_h = int(round(h / scale_factor))
        img_for_yolo = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    detections = detect_bottles(img_for_yolo, confidence_threshold, imgsz=YOLO_IMGSZ)
    if scale_factor != 1.0:
        detections = [
            {**d, 'bbox': tuple(int(round(c * scale_factor)) for c in d['bbox'])}
            for d in detections
        ]
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            d['bbox'] = (max(0, min(x1, w)), max(0, min(y1, h)), max(0, min(x2, w)), max(0, min(y2, h)))
    # Crop angle1 ก่อน (ใช้สำหรับ defect inspection) — ใช้ภาพต้นฉบับ
    angle1_crops = crop_angle1_regions(image, detections)
    
    # เรียก defect model ล่วงหน้า (ใช้ภาพ angle1 ตัวแรก)
    defect_inspection = None
    if angle1_crops and len(angle1_crops) > 0:
        try:
            from libs.detection.defect_model import run_defect_inspection
            first_angle1 = angle1_crops[0]
            crop_img = first_angle1.get('original_crop')
            if crop_img is not None and crop_img.size > 0:
                defect_inspection = run_defect_inspection(crop_img)
                if defect_inspection:
                    print(f"🔍 Defect inspection (angle1): score={defect_inspection.get('score')} -> {defect_inspection.get('result')}")
        except Exception as e:
            print(f"⚠️ Defect inspection skipped: {e}")
    
    # ถ้า defect เป็น NG — ไม่ครอป type, ไม่อ่าน OCR (ไปเลย)
    if defect_inspection is not None and defect_inspection.get('result') == 'NG':
        print("⏭️ Defect NG — ข้าม type crop และ OCR")
        type_crops = []
    else:
        type_crops = crop_type_regions(image, detections)
    
    # Prepare result summary
    result_summary = {
        'total_detections': len(detections),
        'type_detections': len(type_crops),
        'detected_labels': list(set([d['label'] for d in detections])),
        'has_type': len(type_crops) > 0,
        'angle1_detections': len(angle1_crops),
    }
    
    result = {
        'detections': detections,
        'type_crops': type_crops,
        'angle1_crops': angle1_crops,
        'summary': result_summary,
        'original_image': image
    }
    if defect_inspection is not None:
        result['defect_inspection'] = defect_inspection
    return result

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

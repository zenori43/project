# -*- coding: utf-8 -*-
"""
Processing Modules - สำหรับการประมวลผลภาพ (rotation, CRAFT, line detection, OCR)
"""

from libs.processing.rotationmodel import (
    RotationModel,
    initialize_model,
    get_rotated_image_from_array,
    process_craft_rotated_from_array,
    get_cropped_from_craft_rotated_array,
    process_craft_rotated_with_verification_from_array,
    get_cropped_with_verification_from_craft_rotated_array
)

from libs.processing.rotationCRAFT import (
    CRAFTTextDetector,
    initialize_detector as initialize_craft_detector,
    detect_text_and_rotate_from_array
)

from libs.processing.craft_line_detection import (
    CRAFTLineDetector,
    initialize_detector as initialize_line_detector,
    detect_lines_from_rotation_result,
    get_cropped_lines_from_rotation_result
)

from libs.processing.deep_ocr import (
    DeepOCRModel,
    initialize_ocr_model,
    recognize_text_from_craft_lines
)

__all__ = [
    # Rotation
    'RotationModel',
    'initialize_model',
    'get_rotated_image_from_array',
    'process_craft_rotated_from_array',
    'get_cropped_from_craft_rotated_array',
    'process_craft_rotated_with_verification_from_array',
    'get_cropped_with_verification_from_craft_rotated_array',
    
    # CRAFT
    'CRAFTTextDetector',
    'initialize_craft_detector',
    'detect_text_and_rotate_from_array',
    
    # Line Detection
    'CRAFTLineDetector',
    'initialize_line_detector',
    'detect_lines_from_rotation_result',
    'get_cropped_lines_from_rotation_result',
    
    # OCR
    'DeepOCRModel',
    'initialize_ocr_model',
    'recognize_text_from_craft_lines',
]


# -*- coding: utf-8 -*-
"""
Detection Modules - สำหรับการตรวจจับวัตถุ (bottle, cap)
"""

from libs.detection.bottledetect import (
    detect_bottle_and_crop_type,
    process_bottle_image_simple,
    get_type_crops_only,
    get_detection_summary,
    draw_detections_on_image,
    save_cropped_type
)

from libs.detection.capmodel import (
    CapDetector,
    initialize_detector,
    detect_caps_from_path,
    detect_caps_from_array
)

__all__ = [
    'detect_bottle_and_crop_type',
    'process_bottle_image_simple',
    'get_type_crops_only',
    'get_detection_summary',
    'draw_detections_on_image',
    'save_cropped_type',
    'CapDetector',
    'initialize_detector',
    'detect_caps_from_path',
    'detect_caps_from_array',
]


# -*- coding: utf-8 -*-
"""
Simulated Capture - โหลดภาพจากโฟลเดอร์แบบไล่เรียง (round-robin)
ใช้เมื่อ USE_SIMULATED_IMAGES = True แทนการดึงจากกล้องจริง
"""
import os
import cv2

# รองรับนามสกุลภาพ
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

# state สำหรับ round-robin แยกโฟลเดอร์ขวด/ฝา
_usb_folder = None
_sentech_folder = None
_usb_files = []
_sentech_files = []
_usb_index = 0
_sentech_index = 0


def _list_images(folder):
    """คืนรายการ path ไฟล์ภาพในโฟลเดอร์ เรียงตามชื่อ"""
    if not folder or not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        lower = name.lower()
        if any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
            out.append(os.path.join(folder, name))
    return out


def _get_next_image(folder, storage_name):
    """
    โหลดภาพถัดไปจากโฟลเดอร์ (round-robin).
    storage_name = 'usb' หรือ 'sentech'
    คืน (image หรือ None, path ที่ใช้ หรือ None)
    """
    global _usb_folder, _sentech_folder, _usb_files, _sentech_files, _usb_index, _sentech_index
    folder = os.path.normpath(os.path.abspath(folder)) if folder else None
    if storage_name == 'usb':
        if _usb_folder != folder:
            _usb_folder = folder
            _usb_files[:] = _list_images(folder)
            _usb_index = 0
        files = _usb_files
        idx = _usb_index
    else:
        if _sentech_folder != folder:
            _sentech_folder = folder
            _sentech_files[:] = _list_images(folder)
            _sentech_index = 0
        files = _sentech_files
        idx = _sentech_index

    if not files:
        return None, None

    path = files[idx % len(files)]
    img = cv2.imread(path)
    if storage_name == 'usb':
        _usb_index = (idx + 1) % len(files)
    else:
        _sentech_index = (idx + 1) % len(files)
    return img, path


def get_next_usb_image(usb_folder):
    """
    โหลดภาพขวดถัดไปจากโฟลเดอร์ usb_camera (ไล่ไฟล์เรียงตามชื่อ).
    คืน (image หรือ None, path ที่ใช้ หรือ None)
    """
    return _get_next_image(usb_folder, 'usb')


def get_next_sentech_image(sentech_folder):
    """
    โหลดภาพฝาถัดไปจากโฟลเดอร์ sentech_camera (ไล่ไฟล์เรียงตามชื่อ).
    คืน (image หรือ None, path ที่ใช้ หรือ None)
    """
    return _get_next_image(sentech_folder, 'sentech')


def reset_simulated_indices():
    """รีเซ็ต index ให้เริ่มจากภาพแรกใหม่"""
    global _usb_index, _sentech_index
    _usb_index = 0
    _sentech_index = 0


def get_simulated_counts(usb_folder, sentech_folder):
    """คืนจำนวนไฟล์ภาพในแต่ละโฟลเดอร์ (สำหรับแสดงใน UI)"""
    return len(_list_images(usb_folder)), len(_list_images(sentech_folder))

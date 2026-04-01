# -*- coding: utf-8 -*-
"""
Camera Manager - จัดการการทำงานของกล้องทั้งสองตัว
- SentechCamera: ใช้ Harvesters library
- USBCamera: ใช้ OpenCV หรือ import จาก usbcamra.py
"""

# CRITICAL: Setup CUDA paths BEFORE importing any libs modules
try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    pass

import cv2
import time
import sys
import os
import platform
import numpy as np

from config.settings import (
    HARVESTERS_AVAILABLE,
    SENTECH_GENTL_PATH,
    USBCAMERA_PATH,
    SENTECH_FLUSH_MAX,
    SENTECH_WAIT_NEW_FRAME,
)

# Import USBCamera from libs
try:
    from libs.camera.usbcamra import USBCamera
    print("✅ Successfully imported USBCamera from libs.camera.usbcamra")
except ImportError as e:
    print(f"❌ Failed to import USBCamera: {e}")
    print("Using fallback USBCamera class...")
    
    # Fallback USBCamera class if import fails
    class USBCamera:
        def __init__(self):
            self.cap = None
            self.is_running = False
            self._thread = None
            
        def open_camera(self):
            try:
                # Try camera index 1 first (as detected by diagnostic)
                self.cap = cv2.VideoCapture(1)
                if not self.cap.isOpened():
                    # Fallback to camera index 0
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        return False
                return True
            except:
                return False
                
        def close_camera(self):
            if self.cap:
                self.cap.release()
                self.cap = None
                
        def capture_image(self):
            """
            ถ่ายภาพหนึ่งเฟรม - ดึงเฟรมล่าสุดมาใช้
            """
            if not self.cap or not self.cap.isOpened():
                return None
            
            # ล้าง buffer เพื่อดึงเฟรมล่าสุด
            for _ in range(10):  # ล้าง buffer 10 เฟรม
                self.cap.grab()
            
            # รอสักครู่เพื่อให้ได้เฟรมใหม่
            time.sleep(0.1)
            
            # ดึงเฟรมล่าสุด
            ret, frame = self.cap.read()
            if ret and frame is not None:
                print(f"📸 Captured latest frame: {frame.shape}")
                return frame
            
            return None


def _sentech_buffer_to_image_array(component):
    """
    แปลง payload จาก Harvesters เป็น numpy (H,W) Mono8 — ต้อง copy ก่อนออกจาก with fetch()
    เพราะบัฟเฟอร์จะถูก reuse ไม่งั้นภาพจะเพี้ยน/ดำทั้งกรอบหรือเหลือแถบบนๆ
    """
    image_data = component.data
    if image_data is None:
        return None
    height = int(component.height)
    width = int(component.width)
    if len(image_data.shape) == 1:
        need = height * width
        if image_data.size < need:
            print(f"⚠️ Sentech: buffer เล็กกว่า {width}x{height} ({image_data.size} < {need})")
            return None
        image = image_data.reshape((height, width))
    else:
        image = np.asarray(image_data)
    if image.ndim == 2 and image.shape[1] > width:
        image = image[:, :width]
    try:
        dih = int(component.delivered_image_height)
    except Exception:
        dih = 0
    if dih > 0 and image.ndim >= 2 and dih < image.shape[0]:
        image = image[:dih, ...]
    return np.ascontiguousarray(image).copy()


class SentechCamera:
    """Sentech camera class for cap detection - using Harvesters only"""
    
    def __init__(self, gentl_path=None):
        if gentl_path is None:
            gentl_path = SENTECH_GENTL_PATH
        self._is_initialized = False
        self._h = None  # Harvester instance
        self._ia = None  # Image acquirer
        self._gentl_path = gentl_path
        self._connected = False
        
    @property
    def image(self):
        """Get latest captured image"""
        if self._ia is not None and self._connected:
            try:
                # Wait a bit for camera to be ready
                time.sleep(0.1)
                
                with self._ia.fetch() as buffer:
                    component = buffer.payload.components[0]
                    return _sentech_buffer_to_image_array(component)
            except Exception as e:
                print(f"❌ Error getting image: {e}")
        return None
    
    @property 
    def callback_count(self):
        """Return callback count for compatibility"""
        return 1 if self._connected else 0

    def initialize(self):
        """Initialize Sentech camera using Harvesters only"""
        if self._is_initialized:
            return False
        
        if not HARVESTERS_AVAILABLE:
            print("❌ Harvesters not available")
            return False
        
        try:
            from harvesters.core import Harvester
            
            print("🔧 Initializing SENTECH camera with Harvesters...")
            
            # Create Harvester instance
            self._h = Harvester()
            
            # Add GenTL Producer
            self._h.add_file(self._gentl_path)
            self._h.update()
            
            if len(self._h.device_info_list) == 0:
                print("❌ No SENTECH devices found")
                return False
            
            print(f"✅ Found {len(self._h.device_info_list)} SENTECH device(s)")
            
            # Create image acquirer
            self._ia = self._h.create(0)
            
            # Configure camera (with error handling)
            try:
                # Try to configure camera parameters
                if hasattr(self._ia.device.node_map, 'Width'):
                    self._ia.device.node_map.Width.value = 2448
                if hasattr(self._ia.device.node_map, 'Height'):
                    self._ia.device.node_map.Height.value = 2048
                if hasattr(self._ia.device.node_map, 'PixelFormat'):
                    self._ia.device.node_map.PixelFormat.value = 'Mono8'
                if hasattr(self._ia.device.node_map, 'ExposureTime'):
                    self._ia.device.node_map.ExposureTime.value = 10000
                if hasattr(self._ia.device.node_map, 'Gain'):
                    self._ia.device.node_map.Gain.value = 1.0
                
                print("✅ Camera configured successfully")
            except Exception as config_error:
                print(f"⚠️ Camera configuration warning: {config_error}")
                print("📷 Using default camera settings")
            
            # Start acquisition
            self._ia.start()
            self._connected = True
            self._is_initialized = True
            
            print("✅ SENTECH camera initialized with Harvesters")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize SENTECH camera with Harvesters: {e}")
            return False
    
    def capture_image(self):
        """Capture image from Sentech camera using Harvesters only - ดึงเฟรมล่าสุด"""
        if not self._is_initialized:
            print("❌ Sentech camera not initialized")
            return None
        
        if self._ia is not None and self._connected:
            try:
                # ล้าง buffer เพื่อให้ได้เฟรมล่าสุด (ใช้ SENTECH_FLUSH_MAX จาก config)
                flushed_count = 0
                for i in range(SENTECH_FLUSH_MAX):
                    try:
                        buffer = self._ia.try_fetch(timeout=0.01)
                        if buffer:
                            buffer.queue()
                            flushed_count += 1
                    except Exception:
                        break
                if flushed_count > 0:
                    print(f"✅ Flushed {flushed_count} old frame(s) from buffer")
                # ลอง software trigger ถ้ากล้องรองรับ
                try:
                    if hasattr(self._ia.device.node_map, 'TriggerSoftware'):
                        self._ia.device.node_map.TriggerSoftware.execute()
                except Exception:
                    pass
                time.sleep(SENTECH_WAIT_NEW_FRAME)
                with self._ia.fetch(timeout=1.5) as buffer:
                    component = buffer.payload.components[0]
                    image_data = component.data
                    height = int(component.height)
                    width = int(component.width)
                    print(f"  Image data shape: {getattr(image_data, 'shape', None)}")
                    print(f"  Image dimensions: {width}x{height}")
                    if image_data is not None:
                        print(f"  Data type: {image_data.dtype}")
                        print(f"  Min value: {image_data.min()}, Max value: {image_data.max()}")
                    image = _sentech_buffer_to_image_array(component)
                    if image is not None:
                        print(f"  Output array: {image.shape} (copy จากบัฟเฟอร์แล้ว)")
                    return image
                    
            except Exception as e:
                print(f"❌ Error capturing image with Harvesters: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("❌ Camera not connected")
            return None
    
    def cleanup(self):
        """Cleanup Sentech camera using Harvesters only"""
        try:
            if self._is_initialized:
                try:
                    if self._ia is not None:
                        # Stop acquisition
                        self._ia.stop()
                        print("🛑 Stopped Harvesters acquisition")
                        
                        # Destroy image acquirer
                        self._ia.destroy()
                        self._ia = None
                        print("🧹 Destroyed image acquirer")
                    
                    if self._h is not None:
                        # Reset Harvester
                        self._h.reset()
                        self._h = None
                        print("🧹 Reset Harvester")
                    
                    self._connected = False
                    self._is_initialized = False
                    print("✅ SENTECH camera cleaned up with Harvesters")
                    
                except Exception as e:
                    print(f"❌ Error cleaning up Harvesters: {e}")
                    
        except Exception as e:
            print(f"❌ Error in cleanup: {e}")


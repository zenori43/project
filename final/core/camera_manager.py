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

from config.settings import (
    HARVESTERS_AVAILABLE,
    SENTECH_GENTL_PATH,
    USBCAMERA_PATH
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
                    # Get image data
                    component = buffer.payload.components[0]
                    image_data = component.data
                    
                    # Get dimensions
                    height = component.height
                    width = component.width
                    
                    # Reshape image data
                    if len(image_data.shape) == 1:
                        # 1D array - reshape to 2D
                        # For Mono8, reshape to 2D grayscale
                        image = image_data.reshape((height, width))
                    else:
                        image = image_data
                    
                    return image
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
                print("📸 Capturing latest frame with Harvesters...")
                
                # ล้าง buffer เพื่อให้ได้เฟรมล่าสุด - ดึงเฟรมเก่าออกมาเพื่อทิ้ง
                print("🔄 Flushing old frames from buffer...")
                flushed_count = 0
                max_flush = 50  # เพิ่มจำนวนการ flush เพื่อให้แน่ใจว่าล้างหมด
                
                # ดึงเฟรมเก่าออกมาเพื่อทิ้ง (flush buffer)
                # ใช้ try_fetch เพื่อดึงเฟรมที่มีอยู่แล้วทันทีโดยไม่รอเฟรมใหม่
                for i in range(max_flush):
                    try:
                        # ใช้ try_fetch เพื่อดึงเฟรมที่มีอยู่แล้วทันที (ไม่รอเฟรมใหม่)
                        # timeout สั้นๆ เพื่อดึงเฉพาะเฟรมที่มีอยู่แล้ว
                        buffer = self._ia.try_fetch(timeout=0.01)  # ลด timeout เพื่อให้เร็วขึ้น
                        if buffer:
                            # Queue buffer กลับไป - สำคัญมากเพื่อป้องกัน buffer leak
                            # แต่การ queue กลับไปจะทำให้เฟรมนี้ถูกใช้ซ้ำได้
                            # แต่เนื่องจากเราจะดึงเฟรมใหม่หลังจากนี้ เฟรมเก่าจะถูกใช้ก่อนเฟรมใหม่
                            buffer.queue()
                            flushed_count += 1
                            if flushed_count <= 5:
                                print(f"  Flushing buffer... (flushed {flushed_count} frames)")
                    except Exception as flush_error:
                        # ไม่มีเฟรมใน buffer แล้ว หรือเกิด error
                        break
                
                if flushed_count > 0:
                    print(f"  ✅ Flushed {flushed_count} old frame(s) from buffer")
                else:
                    print(f"  ℹ️ No old frames in buffer (already up to date)")
                
                # ลองใช้ Software Trigger ถ้ากล้องรองรับ (เพื่อให้ได้เฟรมใหม่จริงๆ)
                try:
                    if hasattr(self._ia.device.node_map, 'TriggerSoftware'):
                        print("  🔘 Triggering software trigger for new frame...")
                        self._ia.device.node_map.TriggerSoftware.execute()
                        print("  ✅ Software trigger executed")
                except Exception as trigger_error:
                    # กล้องอาจไม่รองรับ software trigger - ไม่เป็นไร
                    pass
                
                # รอสักครู่เพื่อให้ได้เฟรมใหม่ (สำคัญมาก - ให้กล้องสร้างเฟรมใหม่)
                # สำหรับ continuous acquisition ต้องรอให้เฟรมใหม่เข้ามา
                print("  ⏳ Waiting for new frame...")
                time.sleep(0.5)  # เพิ่มเวลาให้เฟรมใหม่เข้ามา (500ms - เพิ่มจาก 300ms)
                
                # ดึงเฟรมล่าสุด (รอเฟรมใหม่ถ้ายังไม่มี)
                print("📸 Fetching latest frame...")
                with self._ia.fetch(timeout=1.5) as buffer:
                    # Get image data
                    component = buffer.payload.components[0]
                    image_data = component.data
                    
                    # Get dimensions
                    height = component.height
                    width = component.width
                    
                    print(f"  Image data shape: {image_data.shape}")
                    print(f"  Image dimensions: {width}x{height}")
                    print(f"  Data type: {image_data.dtype}")
                    print(f"  Min value: {image_data.min()}, Max value: {image_data.max()}")
                    
                    # Reshape image data
                    if len(image_data.shape) == 1:
                        # 1D array - reshape to 2D
                        # For Mono8, reshape to 2D grayscale
                        image = image_data.reshape((height, width))
                    else:
                        image = image_data
                    
                    print(f"✅ Captured latest frame: {image.shape}")
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


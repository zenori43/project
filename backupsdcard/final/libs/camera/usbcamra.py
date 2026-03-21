import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple

class USBCamera:
    """
    คลาสสำหรับจัดการกล้อง USB แบบเรียบง่าย
    รองรับ USB Video Class
    """
    
    def __init__(self):
        """เริ่มต้นกล้อง USB"""
        self.cap = None
        self.is_running = False
        self._thread = None
        
    def open_camera(self) -> bool:
        """
        เปิดกล้อง USB (index 0)
        
        Returns:
            bool: True ถ้าเปิดสำเร็จ, False ถ้าไม่สำเร็จ
        """
        try:
            # สำหรับ Jetson - ลอง V4L2 backend ก่อน
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            # ถ้า V4L2 ไม่ได้ ลอง GStreamer
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0, cv2.CAP_GSTREAMER)
            
            # ถ้ายังไม่ได้ ลองแบบปกติ
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            
            # ถ้ายังไม่ได้ ลองใช้ V4L2 แบบอื่น
            if not self.cap.isOpened():
                print("🔄 ลองใช้ V4L2 แบบอื่น...")
                self.cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                print("❌ ไม่สามารถเปิดกล้อง USB ได้")
                return False
            
            # ตั้งค่าพื้นฐาน
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
            except:
                pass
            
            # ปรับความละเอียดให้สูงสุดเท่าที่กล้องทำได้
            # ลองความละเอียดสูงสุดที่พบบ่อย: 4K, 2K, Full HD, HD
            resolutions = [
                (3840, 2160),  # 4K UHD
                (2560, 1440),  # 2K QHD
                (1920, 1080),  # Full HD
                (1280, 720),   # HD
                (640, 480)     # VGA (fallback)
            ]
            
            best_width = 1920
            best_height = 1080
            resolution_set = False
            
            print("🔍 กำลังตรวจสอบความละเอียดสูงสุดที่กล้องรองรับ...")
            for width, height in resolutions:
                try:
                    # ลองตั้งค่าความละเอียด
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
                    
                    # ตรวจสอบว่าตั้งค่าได้จริงหรือไม่
                    actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    
                    # ถ้าค่าที่ได้ใกล้เคียงกับที่ตั้ง (ยอมรับความแตกต่าง ±10 pixels)
                    if abs(actual_width - width) <= 10 and abs(actual_height - height) <= 10:
                        best_width = int(actual_width)
                        best_height = int(actual_height)
                        print(f"✅ ตั้งค่าความละเอียดสำเร็จ: {best_width}x{best_height}")
                        resolution_set = True
                        break
                    else:
                        print(f"  ⚠️ {width}x{height} - ได้ {actual_width:.0f}x{actual_height:.0f} (ไม่ตรง)")
                except Exception as e:
                    print(f"  ⚠️ {width}x{height} - ไม่สามารถตั้งค่าได้: {e}")
                    continue
            
            if not resolution_set:
                # ถ้าไม่สามารถตั้งค่าความละเอียดสูงได้ ใช้ค่าเริ่มต้น
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920.0)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080.0)
                    best_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    best_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✅ ใช้ความละเอียดเริ่มต้น: {best_width}x{best_height}")
                except:
                    pass
            
            # ตั้งค่า FPS
            try:
                self.cap.set(cv2.CAP_PROP_FPS, 30.0)
            except:
                pass
            
            # ลองตั้งค่าเพิ่มเติมถ้าจำเป็น
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            except:
                pass  # ถ้าไม่สามารถตั้งค่าได้ ให้ข้ามไป
            
            # ตรวจสอบการตั้งค่าที่รายงาน
            reported_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            reported_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # ตรวจสอบ backend ที่ใช้
            backend = self.cap.getBackendName()
            print(f"✅ เปิดกล้อง USB สำเร็จ! (Jetson Platform - {backend})")
            print(f"📐 ขนาดที่รายงาน: {reported_width:.0f}x{reported_height:.0f}")
            print(f"⚡ เฟรมเรท: {fps:.1f} FPS")
            
            # ถ้า FPS ต่ำเกินไป ให้ลองปรับการตั้งค่าใหม่
            if fps < 15:
                print("⚠️ FPS ต่ำเกินไป กำลังลองปรับการตั้งค่าใหม่...")
                
                # ลองใช้ YUYV format แทน MJPG
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
                    self.cap.set(cv2.CAP_PROP_FPS, 30.0)
                except:
                    pass
                
                # ตรวจสอบ FPS ใหม่
                new_fps = self.cap.get(cv2.CAP_PROP_FPS)
                print(f"🔄 FPS ใหม่: {new_fps:.1f}")
                
                # ถ้ายังต่ำ ให้ลองลดความละเอียด
                if new_fps < 15:
                    print("⚠️ ลองลดความละเอียดเพื่อเพิ่ม FPS...")
                    try:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.0)
                        self.cap.set(cv2.CAP_PROP_FPS, 30.0)
                    except:
                        pass
                    
                    final_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    final_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    final_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"📐 ความละเอียดใหม่: {final_width:.0f}x{final_height:.0f}")
                    print(f"⚡ FPS สุดท้าย: {final_fps:.1f}")
            
            # ตรวจสอบความละเอียดจริงและ FPS จริงจากภาพ
            actual_width = 0
            actual_height = 0
            start_time = time.time()
            frame_count = 0
            
            # ลองอ่านเฟรมหลายครั้งเพื่อวัด FPS จริง
            for i in range(30):  # เพิ่มจำนวนเฟรมเพื่อวัด FPS ที่แม่นยำ
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    actual_height, actual_width = test_frame.shape[:2]
                    frame_count += 1
                    if i == 0:  # แสดงขนาดครั้งแรก
                        print(f"📐 ขนาดจริง: {actual_width}x{actual_height}")
                # ไม่ต้อง sleep เพื่อวัด FPS จริง
            
            # คำนวณ FPS จริง
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                actual_fps = frame_count / elapsed_time
                print(f"⚡ FPS จริงจากการทดสอบ: {actual_fps:.1f}")
                
                # แสดงคำแนะนำถ้า FPS ต่ำ
                if actual_fps < 20:
                    print("⚠️ FPS ต่ำ - ลองปรับการตั้งค่ากล้องหรือ driver")
                elif actual_fps < 25:
                    print("⚠️ FPS ปานกลาง - อาจปรับปรุงได้")
                else:
                    print("✅ FPS ดี - พร้อมใช้งาน")
            else:
                print("⚠️ ไม่สามารถวัด FPS จริงได้")
            
            if actual_width > 0:
                if actual_width >= 1920:
                    print("📸 ความละเอียด: 2K หรือสูงกว่า (ชัดมาก)")
                elif actual_width >= 1280:
                    print("📸 ความละเอียด: HD (ชัดดี)")
                elif actual_width >= 640:
                    print("📸 ความละเอียด: VGA (ชัดปานกลาง)")
                else:
                    print("📸 ความละเอียด: ต่ำ (ชัดน้อย)")
            else:
                print("⚠️ ไม่สามารถตรวจสอบขนาดภาพจริงได้ กำลังใช้ค่าเริ่มต้น")
            
            return True
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")
            return False
    
    def close_camera(self):
        """ปิดกล้อง USB"""
        self.is_running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
            print("🛑 ปิดกล้อง USB แล้ว")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        อ่านเฟรมจากกล้อง
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        return self.cap.read()
    
    def capture_image(self) -> Optional[np.ndarray]:
        """
        ถ่ายภาพหนึ่งเฟรม - ดึงเฟรมล่าสุดมาใช้
        
        Returns:
            Optional[np.ndarray]: ภาพที่ถ่าย หรือ None
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
    
    def save_image(self, filepath: str, frame: Optional[np.ndarray] = None, quality: int = 95) -> bool:
        """
        บันทึกภาพลงไฟล์
        
        Args:
            filepath (str): พาธของไฟล์
            frame (Optional[np.ndarray]): เฟรมที่จะบันทึก
            quality (int): คุณภาพการบีบอัด (0-100)
            
        Returns:
            bool: True ถ้าบันทึกสำเร็จ
        """
        try:
            if frame is None:
                ret, frame = self.read_frame()
                if not ret:
                    print("❌ ไม่สามารถอ่านภาพได้")
                    return False
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            cv2.imwrite(filepath, frame, encode_params)
            print(f"💾 บันทึกภาพ: {filepath}")
            return True
                
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")
            return False
    
    def start_streaming(self, callback: Callable[[np.ndarray], None], display_size: Tuple[int, int] = (1280, 720)):
        """
        เริ่มการสตรีมภาพ
        
        Args:
            callback: ฟังก์ชันที่จะเรียกเมื่อได้เฟรมใหม่
            display_size: ขนาดหน้าต่างแสดงผล
        """
        if self.is_running:
            print("⚠️ การสตรีมกำลังทำงานอยู่แล้ว")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._stream_loop, args=(callback, display_size))
        self._thread.daemon = True
        self._thread.start()
        print(f"🎬 เริ่มการสตรีมภาพ")
    
    def stop_streaming(self):
        """หยุดการสตรีมภาพ"""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("⏹️ หยุดการสตรีมภาพ")
    
    def _stream_loop(self, callback: Callable[[np.ndarray], None], display_size: Tuple[int, int]):
        """ลูปสำหรับการสตรีมภาพ"""
        while self.is_running:
            ret, frame = self.read_frame()
            if ret:
                try:
                    # ปรับขนาดภาพสำหรับแสดงผล
                    display_frame = cv2.resize(frame, display_size)
                    callback(display_frame)
                except Exception as e:
                    print(f"❌ เกิดข้อผิดพลาด: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        if self.open_camera():
            return self
        else:
            raise RuntimeError("ไม่สามารถเปิดกล้องได้")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_camera()


# ฟังก์ชันช่วยเหลือ
def open_usb_camera() -> Optional[USBCamera]:
    """
    เปิดกล้อง USB
    
    Returns:
        Optional[USBCamera]: ออบเจ็กต์กล้อง หรือ None
    """
    camera = USBCamera()
    if camera.open_camera():
        return camera
    return None


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    print("🎬 เริ่มการสตรีมภาพ (กด 'q' เพื่อออก, 's' เพื่อบันทึก)")
    
    with USBCamera() as camera:
        def frame_callback(frame):
            cv2.imshow('USB Camera', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                camera.stop_streaming()
            elif key == ord('s'):
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                # บันทึกภาพต้นฉบับไม่ใช่ภาพที่ปรับขนาด
                original_frame = camera.capture_image()
                if original_frame is not None:
                    camera.save_image(filename, original_frame, quality=95)
                    print(f"💾 บันทึกภาพต้นฉบับ: {original_frame.shape[1]}x{original_frame.shape[0]}")
                else:
                    print("❌ ไม่สามารถบันทึกภาพได้")
        
        camera.start_streaming(frame_callback, display_size=(1280, 720))
        
        while camera.is_running:
            time.sleep(0.1)
        
        cv2.destroyAllWindows()

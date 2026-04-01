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
        self._control_window_name = "USB Camera Controls"
        self._preview_window_name = "USB Camera Preview"
        self._trackbar_defs = []
        self._trackbar_values = {}

    # ---------------- Camera controls ----------------
    @staticmethod
    def _build_default_trackbar_defs():
        """
        กำหนดรายการ control ที่พยายามสร้างให้ปรับได้
        หมายเหตุ: บางกล้อง/driver ไม่รองรับบางค่า
        """
        return [
            {"name": "Brightness", "prop": cv2.CAP_PROP_BRIGHTNESS, "min": 0, "max": 255, "default": 128},
            {"name": "Contrast", "prop": cv2.CAP_PROP_CONTRAST, "min": 0, "max": 255, "default": 128},
            {"name": "Saturation", "prop": cv2.CAP_PROP_SATURATION, "min": 0, "max": 255, "default": 128},
            {"name": "Hue", "prop": cv2.CAP_PROP_HUE, "min": 0, "max": 255, "default": 0},
            {"name": "Gain", "prop": cv2.CAP_PROP_GAIN, "min": 0, "max": 255, "default": 64},
            {"name": "Sharpness", "prop": cv2.CAP_PROP_SHARPNESS, "min": 0, "max": 255, "default": 128},
            {"name": "Gamma", "prop": cv2.CAP_PROP_GAMMA, "min": 0, "max": 255, "default": 100},
            # Exposure/Fokus มักไม่เป็นสเกลเดียวกันทุกกล้อง จึงให้ช่วงกว้าง
            {"name": "Exposure", "prop": cv2.CAP_PROP_EXPOSURE, "min": 0, "max": 1000, "default": 100},
            # หลายกล้องรองรับช่วง focus กว้างกว่า 255 (เช่น 1023/4095)
            {"name": "Focus", "prop": cv2.CAP_PROP_FOCUS, "min": 0, "max": 4095, "default": 0},
            {"name": "WB Temp", "prop": cv2.CAP_PROP_WB_TEMPERATURE, "min": 2000, "max": 7500, "default": 4500},
            {"name": "Zoom", "prop": cv2.CAP_PROP_ZOOM, "min": 0, "max": 400, "default": 100},
        ]

    def _set_camera_prop_safe(self, prop_id: int, value: float, name: str = "") -> bool:
        """พยายาม set property และคืนผลสำเร็จ/ล้มเหลวแบบไม่ throw"""
        if not self.cap or not self.cap.isOpened():
            return False
        try:
            ok = self.cap.set(prop_id, float(value))
            if not ok and name:
                print(f"⚠️ ไม่รองรับการปรับ: {name}")
            return bool(ok)
        except Exception as e:
            if name:
                print(f"⚠️ ปรับ {name} ไม่ได้: {e}")
            return False

    def _get_camera_prop_safe(self, prop_id: int, default_value: float = 0.0) -> float:
        """อ่านค่า property แบบปลอดภัย"""
        if not self.cap or not self.cap.isOpened():
            return float(default_value)
        try:
            value = self.cap.get(prop_id)
            if value is None:
                return float(default_value)
            return float(value)
        except Exception:
            return float(default_value)

    def disable_auto_controls(self):
        """
        พยายามปิด auto controls ที่กระทบการปรับ manual
        (บางรุ่นกล้องไม่รองรับ)
        """
        if not self.cap or not self.cap.isOpened():
            return
        # V4L2/MJPEG บางตัวใช้ค่า 1=manual, 3=auto
        self._set_camera_prop_safe(cv2.CAP_PROP_AUTOFOCUS, 0.0, "Auto Focus")
        self._set_camera_prop_safe(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0, "Auto Exposure")

    def get_camera_controls(self) -> dict:
        """คืนค่า control สำคัญของกล้องในรูป dict"""
        controls = {}
        for d in self._build_default_trackbar_defs():
            controls[d["name"]] = self._get_camera_prop_safe(d["prop"], d["default"])
        return controls

    def set_camera_control(self, control_name: str, value: float) -> bool:
        """ตั้งค่า control ตามชื่อ (เช่น Brightness, Exposure, Focus)"""
        defs = self._build_default_trackbar_defs()
        for d in defs:
            if d["name"].lower() == control_name.lower():
                return self._set_camera_prop_safe(d["prop"], float(value), d["name"])
        print(f"⚠️ ไม่พบ control ชื่อ: {control_name}")
        return False

    def create_camera_control_panel(self, window_name: str = "USB Camera Controls") -> bool:
        """
        สร้าง panel ปรับกล้องด้วย OpenCV Trackbar
        เรียกได้หลัง open_camera()
        """
        if not self.cap or not self.cap.isOpened():
            print("❌ ยังไม่ได้เปิดกล้อง - ไม่สามารถสร้าง panel ได้")
            return False

        self._control_window_name = window_name
        cv2.namedWindow(self._control_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._control_window_name, 560, 520)
        self._trackbar_defs = self._build_default_trackbar_defs()
        self._trackbar_values = {}

        self.disable_auto_controls()

        def _make_callback(defn):
            def _on_change(v):
                self._trackbar_values[defn["name"]] = int(v)
                self._set_camera_prop_safe(defn["prop"], float(v), defn["name"])
            return _on_change

        for d in self._trackbar_defs:
            cur = int(round(self._get_camera_prop_safe(d["prop"], d["default"])))
            # ถ้ากล้องรายงานค่าปัจจุบันเกิน max ที่ตั้งไว้ ให้ขยายช่วง slider อัตโนมัติ
            slider_max = int(d["max"])
            if cur > slider_max:
                slider_max = min(65535, max(cur + 100, cur * 2))
            cur = max(d["min"], min(cur, slider_max))
            self._trackbar_values[d["name"]] = cur
            cv2.createTrackbar(d["name"], self._control_window_name, cur, slider_max, _make_callback(d))
            if d["min"] > 0:
                cv2.setTrackbarMin(d["name"], self._control_window_name, d["min"])

        print("🎛️ เปิด Camera Control Panel แล้ว (กด q เพื่อปิด preview)")
        return True

    def show_preview_with_control_panel(self, preview_size: Tuple[int, int] = (1280, 720)):
        """
        แสดง preview + panel ปรับกล้องแบบ realtime
        ใช้งานสะดวกสำหรับจูนค่าในหน้างาน
        """
        if not self.cap or not self.cap.isOpened():
            print("❌ ยังไม่ได้เปิดกล้อง")
            return

        self._preview_window_name = "USB Camera Preview"
        cv2.namedWindow(self._preview_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._preview_window_name, preview_size[0], preview_size[1])
        self.create_camera_control_panel(self._control_window_name)

        print("🎬 เริ่ม preview พร้อม panel ปรับกล้อง (กด q หรือ ESC เพื่อออก)")
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            display_frame = cv2.resize(frame, preview_size)
            cv2.imshow(self._preview_window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        cv2.destroyWindow(self._preview_window_name)
        cv2.destroyWindow(self._control_window_name)
        
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
    print("🎬 โหมดทดสอบกล้อง USB")
    print("   - กด 'q' ออก")
    print("   - กด 's' บันทึกภาพ")
    print("   - กด 'c' เปิด Camera Control Panel")
    
    with USBCamera() as camera:
        def frame_callback(frame):
            cv2.imshow('USB Camera', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                camera.stop_streaming()
            elif key == ord('c'):
                camera.show_preview_with_control_panel(display_size)
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
        
        display_size = (1280, 720)
        camera.start_streaming(frame_callback, display_size=display_size)
        
        while camera.is_running:
            time.sleep(0.1)
        
        cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Business Logic - Thread classes และ business logic
รวม Thread classes สำหรับการประมวลผลแบบ async
"""

# CRITICAL: Setup CUDA paths BEFORE importing any libs modules
try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    pass

import cv2
import os
import time
import sys
from typing import List, Dict

from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from pymodbus.client import ModbusTcpClient

# Import from our modules
from core.image_processor import (
    perform_ocr_on_image,
    check_bottle_type,
    detect_faded_text_in_cap,
    enhance_cap_sharpness
)

# Import bottle detection module
from libs.detection.bottledetect import process_bottle_image_simple

# Import cap detection modules
from config.settings import CAP_DETECTION_AVAILABLE
if CAP_DETECTION_AVAILABLE:
    from libs.processing.craft_line_detection import (
        detect_lines_from_rotation_result,
    )
    from libs.processing.deep_ocr import recognize_text_from_craft_lines


class CapDetectionThread(QThread):
    """Thread for processing cap detection to avoid GUI freezing"""
    result_ready = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    rotation_attempt_updated = pyqtSignal(int, int, object, object, object)  # attempt_num, total_attempts, rotated_image, ocr_results, format_valid
    
    def __init__(self, image, cap_detector, craft_detector, rotation_model, line_detector, ocr_model, faded_text_yolo_model=None, bottle_type=None):
        super().__init__()
        self.image = image
        self.cap_detector = cap_detector
        self.craft_detector = craft_detector
        self.rotation_model = rotation_model
        self.line_detector = line_detector
        self.ocr_model = ocr_model
        self.faded_text_yolo_model = faded_text_yolo_model
        self.bottle_type = bottle_type  # เก็บ bottle_type เพื่อใช้ในการปรับ brightness/contrast
        self.modbus_thread = None  # Will be set by GUI if needed
        
    def run(self):
        try:
            print("🔄 CAP DETECTION THREAD: Starting processing...")
            print(f"🔄 CAP DETECTION THREAD: Image shape: {self.image.shape}")
            self.status_updated.emit("กำลังประมวลผลฝา...")
            self.progress_updated.emit(5)
            
            # Save temporary image for processing
            temp_path = "temp_sentech_image.jpg"
            cv2.imwrite(temp_path, self.image)
            print("🔄 CAP DETECTION THREAD: Saved temp image")
            self.progress_updated.emit(10)
            
            # Step 1: Detect caps
            self.status_updated.emit("กำลังตรวจจับฝา...")
            self.progress_updated.emit(20)
            print("🔄 CAP DETECTION THREAD: Step 1 - ตรวจจับฝา")
            cap_result = self.cap_detector.detect_caps(temp_path)
            
            # แสดงความเชื่อมั่นของแต่ละฝาที่ตรวจจับได้
            if cap_result and 'detections' in cap_result and cap_result['detections']:
                total_caps = len(cap_result['detections'])
                print(f"✅ พบฝาทั้งหมด: {total_caps} ฝา")
                for idx, detection in enumerate(cap_result['detections'], 1):
                    conf = detection.get('confidence', 0.0)
                    bbox = detection.get('bbox', [])
                    class_name = detection.get('class_name', 'cap')
                    print(f"   📍 ฝาที่ {idx}: ความเชื่อมั่น = {conf:.4f} ({conf*100:.2f}%) | Class: {class_name} | BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            else:
                print("⚠️ ไม่พบฝาใดๆ ที่ตรวจจับได้")
            
            # Get the processed image from cap detector
            processed_image = self.cap_detector.last_processed_image
            if processed_image is None:
                raise ValueError("ไม่สามารถประมวลผลรูปภาพได้")
            
            self.progress_updated.emit(30)
            
            # Step 2: Crop detected regions and process each crop
            self.status_updated.emit("กำลังตัดภาพ...")
            self.progress_updated.emit(40)
            print("🔄 CAP DETECTION THREAD: Step 2 - ตัดภาพ")
            cropped_images = self.cap_detector.crop_detections(margin=20)
            
            result = {
                'cap_result': cap_result,
                'processed_image': processed_image,
                'cropped_images': cropped_images,
                'image': self.image,
                'image_shape': self.image.shape
            }
            
            if not cropped_images:
                self.status_updated.emit("ไม่พบฝา กำลังประมวลผลข้อความ...")
                self.progress_updated.emit(50)
                print("🔄 CAP DETECTION THREAD: ไม่พบฝา เริ่มประมวลผลข้อความ...")
                
                # Step 2a: Enhance sharpness before CRAFT
                self.status_updated.emit("กำลังปรับความคมชัด...")
                print("🔄 CAP DETECTION THREAD: Step 2a - ปรับความคมชัดก่อน CRAFT")
                enhanced_image = enhance_cap_sharpness(processed_image)
                if enhanced_image is None:
                    enhanced_image = processed_image  # Fallback to original if enhancement fails
                
                # Step 2b: Use CRAFT to detect text and rotate original image
                self.status_updated.emit("กำลังตรวจจับข้อความด้วย CRAFT...")
                self.progress_updated.emit(60)
                print("🔄 CAP DETECTION THREAD: Step 2b - ตรวจจับข้อความด้วย CRAFT")
                craft_result = self.craft_detector.detect_text_and_rotate(image_array=enhanced_image)
                
                if craft_result and 'rotated_image' in craft_result and craft_result['rotated_image'] is not None:
                    # CRAFT already rotated the image, now use rotation model on CRAFT's rotated image
                    craft_rotated_image = craft_result['rotated_image']
                    result['craft_rotated_image'] = craft_rotated_image.copy()
                    print("🔄 CAP DETECTION THREAD: CRAFT หมุนภาพสำเร็จ")
                    self.progress_updated.emit(65)
                    
                    # Step 2b: Use rotation model to process CRAFT rotated image (simple processing)
                    self.status_updated.emit("กำลังประมวลผลด้วย AI rotation...")
                    self.progress_updated.emit(70)
                    print("🔄 CAP DETECTION THREAD: Step 2b - ประมวลผลด้วย AI rotation")
                    print("🔄 CAP DETECTION THREAD: Calling process_craft_rotated_image_with_models...")
                    combined_result = self.rotation_model.process_craft_rotated_image_with_models(
                        craft_rotated_image, self.line_detector, self.ocr_model)
                    print(f"🔄 CAP DETECTION THREAD: Combined result type: {type(combined_result)}")
                    if combined_result:
                        print(f"🔄 CAP DETECTION THREAD: Combined result keys: {list(combined_result.keys()) if isinstance(combined_result, dict) else 'Not a dict'}")
                    
                    if combined_result and 'rotated_image' in combined_result and combined_result['rotated_image'] is not None:
                        rotated_image = combined_result['rotated_image']
                        result['rotated_image'] = rotated_image
                        print("🔄 CAP DETECTION THREAD: Rotation สำเร็จ")
                        self.progress_updated.emit(75)
                        
                        # Step 2c: Get cropped image from rotated image
                        if combined_result.get('cropped_image') is not None:
                            result['cropped_image'] = combined_result['cropped_image']
                            print("🔄 CAP DETECTION THREAD: ได้ภาพที่ครอปแล้ว")
                        
                        result['combined_result'] = combined_result
                        
                        # Step 3: Use results from combined_result (includes retry mechanism)
                        self.status_updated.emit("กำลังประมวลผลผลลัพธ์...")
                        self.progress_updated.emit(80)
                        print("🔄 CAP DETECTION THREAD: Step 3 - ใช้ผลลัพธ์จาก retry mechanism")
                        
                        # Use line detection and OCR results from combined_result (includes retry)
                        if 'line_detection_result' in combined_result:
                            result['line_detection_result'] = combined_result['line_detection_result']
                            print("🔄 CAP DETECTION THREAD: ใช้ผลลัพธ์ line detection จาก retry")
                        
                        if 'ocr_results' in combined_result:
                            result['ocr_results'] = combined_result['ocr_results']
                            print("🔄 CAP DETECTION THREAD: ใช้ผลลัพธ์ OCR จาก retry")
                            self.progress_updated.emit(85)
                            
                            # Display retry information
                            if 'rotation_angle' in combined_result:
                                rotation_angle = combined_result['rotation_angle']
                                print(f"🔄 CAP DETECTION THREAD: มุมการหมุนที่ใช้: {rotation_angle}°")
                            
                            if 'score' in combined_result:
                                score = combined_result['score']
                                print(f"🔄 CAP DETECTION THREAD: คะแนนการหมุน: {score:.1f}")
                            
                            if 'rotation_attempt' in combined_result:
                                attempt = combined_result['rotation_attempt']
                                print(f"🔄 CAP DETECTION THREAD: ครั้งที่หมุน: {attempt}/5")
                            
                            self.progress_updated.emit(95)
                        else:
                            print("⚠️ CAP DETECTION THREAD: ไม่สามารถตรวจจับบรรทัดข้อความได้")
                    else:
                        print("⚠️ CAP DETECTION THREAD: AI rotation ไม่สำเร็จ")
                else:
                    print("⚠️ CAP DETECTION THREAD: CRAFT ไม่สามารถหมุนภาพได้")
            else:
                print(f"🔄 CAP DETECTION THREAD: พบฝา {len(cropped_images)} ฝา")
                self.progress_updated.emit(50)
                
                # Select only the bottommost cap (highest y-coordinate)
                bottommost_cap = None
                bottommost_index = 0
                max_y = 0
                
                # Find the cap with the highest y-coordinate (bottom of image)
                if cropped_images and 'detections' in cap_result:
                    detections = cap_result['detections']
                    print(f"🔄 CAP DETECTION THREAD: ตรวจสอบตำแหน่งฝา {len(detections)} ฝา")
                    
                    # Find the cap with the highest y-coordinate (bottom of image)
                    for i, detection in enumerate(detections):
                        if 'bbox' in detection:
                            # bbox format: [x1, y1, x2, y2]
                            bbox = detection['bbox']
                            y_center = (bbox[1] + bbox[3]) / 2  # Use center y-coordinate
                            conf = detection.get('confidence', 0.0)
                            print(f"🔄 CAP DETECTION THREAD: ฝาที่ {i+1} - y_center: {y_center:.1f}, ความเชื่อมั่น: {conf:.4f} ({conf*100:.2f}%)")
                            
                            if y_center > max_y:
                                max_y = y_center
                                bottommost_cap = cropped_images[i]
                                bottommost_index = i
                    
                    # แสดงความเชื่อมั่นของฝาที่เลือก
                    selected_conf = detections[bottommost_index].get('confidence', 0.0) if bottommost_index < len(detections) else 0.0
                    print(f"🔄 CAP DETECTION THREAD: เลือกฝาที่ {bottommost_index + 1} (y_center: {max_y:.1f}, ความเชื่อมั่น: {selected_conf:.4f} ({selected_conf*100:.2f}%)) เป็นฝาล่างสุด")
                elif cropped_images:
                    # Fallback: use the last cap if no bounding box info available
                    bottommost_cap = cropped_images[-1]
                    bottommost_index = len(cropped_images) - 1
                    print(f"🔄 CAP DETECTION THREAD: ไม่มีข้อมูลตำแหน่ง ใช้ฝาลำดับสุดท้าย จาก {len(cropped_images)} ฝา")
                
                # แสดงสรุปข้อมูลฝาที่เลือก
                if bottommost_index < len(cropped_images):
                    selected_detection = detections[bottommost_index] if bottommost_index < len(detections) and 'detections' in cap_result else None
                    if selected_detection:
                        selected_conf = selected_detection.get('confidence', 0.0)
                        print(f"✅ CAP DETECTION THREAD: เลือกฝาที่ {bottommost_index + 1} (ฝาล่างสุด) จาก {len(cropped_images)} ฝา | ความเชื่อมั่น: {selected_conf:.4f} ({selected_conf*100:.2f}%)")
                    else:
                        print(f"✅ CAP DETECTION THREAD: เลือกฝาที่ {bottommost_index + 1} (ฝาล่างสุด) จาก {len(cropped_images)} ฝา")
                else:
                    print(f"✅ CAP DETECTION THREAD: เลือกฝาที่ {bottommost_index + 1} (ฝาล่างสุด) จาก {len(cropped_images)} ฝา")
                
                # Process only the bottommost cap through the full pipeline
                all_cap_results = []
                if bottommost_cap is not None:
                    self.status_updated.emit(f"กำลังประมวลผลฝาที่ {bottommost_index + 1} (ฝาล่างสุด)...")
                    self.progress_updated.emit(50)
                    selected_conf_str = ""
                    if 'detections' in cap_result and bottommost_index < len(cap_result['detections']):
                        selected_conf = cap_result['detections'][bottommost_index].get('confidence', 0.0)
                        selected_conf_str = f" (ความเชื่อมั่น: {selected_conf:.2%})"
                    print(f"🔄 CAP DETECTION THREAD: ประมวลผลฝาที่ {bottommost_index + 1} (ฝาล่างสุด){selected_conf_str}")
                    
                    # Step 2a: Enhance sharpness before CRAFT
                    self.status_updated.emit(f"กำลังปรับความคมชัดฝาที่ {bottommost_index + 1}...")
                    print(f"🔄 CAP DETECTION THREAD: Step 2a - ปรับความคมชัดฝาที่ {bottommost_index + 1} ก่อน CRAFT")
                    enhanced_cap = enhance_cap_sharpness(bottommost_cap)
                    if enhanced_cap is None:
                        enhanced_cap = bottommost_cap  # Fallback to original if enhancement fails
                    
                    # Step 2b: Use CRAFT to detect text and rotate cropped cap
                    self.status_updated.emit(f"กำลังตรวจจับข้อความในฝาที่ {bottommost_index + 1} ด้วย CRAFT...")
                    craft_result = self.craft_detector.detect_text_and_rotate(image_array=enhanced_cap)
                    
                    cap_result = {
                        'cap_index': bottommost_index,
                        'original_crop': bottommost_cap,
                        'craft_result': craft_result
                    }
                    
                    # Step 2a.5: Detect faded text in the cropped cap BEFORE CRAFT rotation
                    self.status_updated.emit(f"กำลังตรวจสอบรอยจางในฝาที่ {bottommost_index + 1}...")
                    print(f"🔄 CAP DETECTION THREAD: Step 2a.5 - ตรวจสอบรอยจางในฝาที่ {bottommost_index + 1} (ก่อน CRAFT)")
                    print(f"🔍 CAP DETECTION THREAD: ใช้ค่ากลางที่วิเคราะห์ได้ (brightness: 33) สำหรับการปรับภาพ")
                    faded_text_result = detect_faded_text_in_cap(
                        bottommost_cap, 
                        yolo_model=None,  # ไม่ใช้ YOLO model ใหม่ ใช้รูปที่ crop แล้วโดยตรง
                        show_debug=True,  # เปิด debug mode เพื่อแสดงภาพใน UI
                        bottle_type=None  # ไม่ใช้แล้ว - ใช้ threshold เดียวกันสำหรับทุกรส
                    )
                    cap_result['faded_text_result'] = faded_text_result
                    print(f"🔄 CAP DETECTION THREAD: ตรวจสอบรอยจางฝาที่ {bottommost_index + 1} สำเร็จ - สถานะ: {faded_text_result['status']}")
                    print(f"📊 CAP DETECTION THREAD: total_area = {faded_text_result.get('total_area', 0)}, num_chars = {faded_text_result.get('num_chars', 0)}")
                    
                    # เก็บผลลัพธ์ไว้ก่อน (แม้จะเป็น faded) เพื่อแสดงภาพและค่า area
                    all_cap_results.append(cap_result)
                    
                    # หยุดประมวลผลเมื่อเจอข้อความจาง แต่ยังแสดงผลลัพธ์
                    if faded_text_result.get('status') == 'faded':
                        print("❌ CAP DETECTION THREAD: ตรวจพบข้อความจาง - หยุดประมวลผล (แต่จะแสดงภาพและค่า area)")
                        print("🚨 CAP DETECTION THREAD: ส่งสัญญาณ M140 (ฝาไม่ผ่าน)")
                        print("🚀 CAP DETECTION THREAD: ส่งสัญญาณ M600 (ประมวลผลเสร็จสิ้น)")
                        
                        # ส่งสัญญาณ M140 (ฝาไม่ผ่าน)
                        if hasattr(self, 'modbus_thread') and self.modbus_thread:
                            self.modbus_thread.on_m140()
                            print("✅ M140 ส่งสัญญาณเรียบร้อย")
                            
                            # ส่งสัญญาณ M600 (ประมวลผลเสร็จสิ้น)
                            self.modbus_thread.on_m600()
                            print("✅ M600 ส่งสัญญาณเรียบร้อย")
                        
                        # สร้างผลลัพธ์ที่บ่งบอกว่าเป็นข้อความจาง แต่มีข้อมูลครบถ้วนเพื่อแสดงผล
                        result = {
                            'error': 'faded_text_detected',
                            'message': f'ตรวจพบข้อความจาง - total_area: {faded_text_result.get("total_area", 0)}, num_chars: {faded_text_result.get("num_chars", 0)}',
                            'faded_text_result': faded_text_result,
                            'cap_processing_results': all_cap_results,  # ใช้ all_cap_results ที่มี cap_result แล้ว
                            'image': self.image,  # เพิ่มภาพต้นฉบับ
                            'image_shape': self.image.shape if self.image is not None else None
                        }
                        
                        # Final progress update
                        self.progress_updated.emit(100)
                        self.status_updated.emit("ตรวจพบข้อความจาง - แสดงผลลัพธ์")
                        
                        print("🔄 CAP DETECTION THREAD: Emitting result with faded text (พร้อมแสดงภาพและค่า area)...")
                        self.result_ready.emit(result)
                        return
                    
                    if craft_result and 'rotated_image' in craft_result and craft_result['rotated_image'] is not None:
                        craft_rotated_image = craft_result['rotated_image']
                        cap_result['craft_rotated_image'] = craft_rotated_image
                        print(f"🔄 CAP DETECTION THREAD: CRAFT หมุนฝาที่ {bottommost_index + 1} สำเร็จ")
                        
                        # Step 2b: Use rotation model to process CRAFT rotated cap (simple processing)
                        self.status_updated.emit(f"กำลังประมวลผลฝาที่ {bottommost_index + 1} ด้วย rotation...")
                        combined_result = self.rotation_model.process_craft_rotated_image_with_models(
                            craft_rotated_image, self.line_detector, self.ocr_model)
                        
                        if combined_result and 'rotated_image' in combined_result and combined_result['rotated_image'] is not None:
                            cap_result['rotated_image'] = combined_result['rotated_image']
                            cap_result['combined_result'] = combined_result
                            print(f"🔄 CAP DETECTION THREAD: Rotation ฝาที่ {bottommost_index + 1} สำเร็จ")
                            
                            # Step 3: Detect text lines from rotation result
                            self.status_updated.emit(f"กำลังตรวจจับบรรทัดข้อความในฝาที่ {bottommost_index + 1}...")
                            line_detection_result = detect_lines_from_rotation_result(combined_result)
                            cap_result['line_detection_result'] = line_detection_result
                            
                            if line_detection_result and 'error' not in line_detection_result:
                                print(f"🔄 CAP DETECTION THREAD: ตรวจจับบรรทัดข้อความฝาที่ {bottommost_index + 1} สำเร็จ")
                                
                                # Step 4: Perform OCR on detected lines
                                self.status_updated.emit(f"กำลังอ่านข้อความในฝาที่ {bottommost_index + 1} ด้วย OCR...")
                                ocr_results = recognize_text_from_craft_lines(line_detection_result)
                                cap_result['ocr_results'] = ocr_results
                                print(f"🔄 CAP DETECTION THREAD: OCR ฝาที่ {bottommost_index + 1} สำเร็จ - อ่านได้ {len(ocr_results)} รายการ")
                            else:
                                print(f"⚠️ CAP DETECTION THREAD: ไม่สามารถตรวจจับบรรทัดข้อความฝาที่ {bottommost_index + 1} ได้")
                        else:
                            print(f"⚠️ CAP DETECTION THREAD: Rotation ฝาที่ {bottommost_index + 1} ไม่สำเร็จ")
                    else:
                        print(f"⚠️ CAP DETECTION THREAD: CRAFT ไม่สามารถหมุนฝาที่ {bottommost_index + 1} ได้")
                    
                    all_cap_results.append(cap_result)
                
                result['cap_processing_results'] = all_cap_results
                self.progress_updated.emit(90)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
                print("🔄 CAP DETECTION THREAD: ลบไฟล์ชั่วคราวแล้ว")
            except:
                pass
            
            # Final progress update
            self.progress_updated.emit(100)
            self.status_updated.emit("ประมวลผลเสร็จสิ้น")
            
            print("🔄 CAP DETECTION THREAD: Emitting result...")
            self.result_ready.emit(result)
                
        except Exception as e:
            print(f"❌ CAP DETECTION THREAD: Error: {e}")
            error_result = {
                'error': str(e),
                'image': self.image,
                'image_shape': self.image.shape if self.image is not None else None
            }
            self.result_ready.emit(error_result)

class ModbusThread(QThread):
    """Thread for Modbus communication"""
    modbus_status = pyqtSignal(str)
    trigger_detected = pyqtSignal()
    queue_trigger_detected = pyqtSignal()  # สำหรับการถ่ายภาพจากคิว
    silent_trigger_detected = pyqtSignal()  # สำหรับการถ่ายภาพและประมวลผลแบบเงียบ (ไม่แสดง UI, ไม่ส่งสัญญาณ)
    result_ready_to_display = pyqtSignal(dict, dict)  # (bottle_result, cap_result) - สำหรับแสดงผลลัพธ์จาก queue
    bottle_type_detected = pyqtSignal(str, str)  # (bottle_type, ocr_text)
    d6004_status_updated = pyqtSignal(int)  # D6004 value (deprecated - ใช้ M401 แทน)
    d6007_status_updated = pyqtSignal(int)  # D6007 value (deprecated - ใช้ M403-M406 แทน)
    m402_status_updated = pyqtSignal(bool)  # M402 value (coil) - ใช้แทน D6006
    m403_status_updated = pyqtSignal(bool)  # M403 value (coil) - น้ำเต้าหู้รสดั้งเดิม (แทน D6007=100)
    m404_status_updated = pyqtSignal(bool)  # M404 value (coil) - น้ำตาลน้อย 2% (แทน D6007=200)
    m405_status_updated = pyqtSignal(bool)  # M405 value (coil) - ผสมเม็ดแมงลัก (แทน D6007=300)
    m406_status_updated = pyqtSignal(bool)  # M406 value (coil) - NG เต็ม (แทน D6007=400)
    d5002_status_updated = pyqtSignal(int)  # D5002 value
    d5001_status_updated = pyqtSignal(int)  # D5001 value (error code)
    m401_status_updated = pyqtSignal(bool)  # M401 value (coil)
    m402_status_updated = pyqtSignal(bool)  # M402 value (coil) - ใช้แทน D6006
    reset_processing_signal = pyqtSignal()  # สำหรับ reset การประมวลผล
    m513_stop_signal = pyqtSignal()  # สำหรับ M513 stop program
    capture_only_trigger = pyqtSignal()  # ID 5: capture only mode (no processing)
    capture_limit_reached_signal = pyqtSignal()  # สำหรับหยุดระบบเมื่อถ่ายครบจำนวนแล้ว
    
    def __init__(self, modbus_ip="192.168.1.5", modbus_port=502):
        super().__init__()
        self.modbus_ip = modbus_ip
        self.modbus_port = modbus_port
        self.modbus_client = None
        self.is_running = False
        self.last_m301 = False
        self.last_m512 = False
        self.last_m513 = False
        self.last_m511 = False
        self.m512_ready = False
        self.stop_requested = False
        self.m600_reset_pending = False
        self.pending_m301_count = 0
        self.pending_results_queue = []  # เก็บผลลัพธ์ที่ประมวลผลแล้ว (bottle_result, cap_result)
        self.program_enabled = False  # เริ่มต้นปิดการทำงานของโปรแกรม (รอ M511)
        self.d6006_monitoring = False  # สำหรับ M130 monitoring (ใช้ M402 แทน D6006)
        self.angle3_retry_mode = False  # สำหรับการถ่ายภาพซ้ำหลัง M402 ON
        self.angle3_retry_count = 0  # นับจำนวนครั้งที่ถ่ายภาพซ้ำ
        self.last_m402_value = None  # เก็บค่า M402 ครั้งล่าสุดเพื่อตรวจสอบการเปลี่ยนแปลง
        self.max_angle3_retries = 10  # จำนวนครั้งสูงสุดที่ถ่ายภาพซ้ำ
        self.last_d6007_value = None  # เก็บค่า D6007 ครั้งล่าสุดเพื่อตรวจสอบการเปลี่ยนแปลง (deprecated)
        self.last_m403_value = None  # เก็บค่า M403 ครั้งล่าสุด
        self.last_m404_value = None  # เก็บค่า M404 ครั้งล่าสุด
        self.last_m405_value = None  # เก็บค่า M405 ครั้งล่าสุด
        self.last_m406_value = None  # เก็บค่า M406 ครั้งล่าสุด
        self.last_d5002_value = None  # เก็บค่า D5002 ครั้งล่าสุดเพื่อตรวจสอบการเปลี่ยนแปลง
        self.last_d5001_value = None  # เก็บค่า D5001 ครั้งล่าสุดเพื่อตรวจสอบการเปลี่ยนแปลง
        self.capture_only_mode = False  # ID 5: capture only mode (no processing)
        self.capture_image_count = 0  # นับจำนวนภาพที่ถ่ายใน capture only mode
        self.capture_with_limit_mode = False  # ID 6: capture with limit mode
        self.capture_limit_count = 10  # จำนวนครั้งที่ต้องการถ่าย
        self.capture_current_count = 0  # จำนวนครั้งที่ถ่ายแล้ว
        self.capture_limit_reached = False  # Flag สำหรับบอกว่าถ่ายครบแล้ว รอหยุด
        
    def run(self):
        super().__init__()
        self.modbus_ip = modbus_ip
        self.modbus_port = modbus_port
        self.modbus_client = None
        self.is_running = False
        self.last_m301 = False
        self.last_m512 = False
        self.last_m513 = False
        self.last_m511 = False
        self.m512_ready = False
        self.stop_requested = False
        self.m600_reset_pending = False
        self.pending_m301_count = 0
        self.program_enabled = False  # เริ่มต้นปิดการทำงานของโปรแกรม (รอ M511)
        self.d6006_monitoring = False  # สำหรับ M130 monitoring (ใช้ M402 แทน D6006)
        self.angle3_retry_mode = False  # สำหรับการถ่ายภาพซ้ำหลัง M402 ON
        self.angle3_retry_count = 0  # นับจำนวนครั้งที่ถ่ายภาพซ้ำ
        self.last_m402_value = None  # เก็บค่า M402 ครั้งล่าสุดเพื่อตรวจสอบการเปลี่ยนแปลง
        self.max_angle3_retries = 10  # จำนวนครั้งสูงสุดที่ถ่ายภาพซ้ำ
        self.last_d6007_value = None  # เก็บค่า D6007 ครั้งล่าสุดเพื่อตรวจสอบการเปลี่ยนแปลง (deprecated)
        self.last_m403_value = None  # เก็บค่า M403 ครั้งล่าสุด
        self.last_m404_value = None  # เก็บค่า M404 ครั้งล่าสุด
        self.last_m405_value = None  # เก็บค่า M405 ครั้งล่าสุด
        self.last_m406_value = None  # เก็บค่า M406 ครั้งล่าสุด
        
    def run(self):
        """Main Modbus loop"""
        try:
            # Create Modbus client
            self.modbus_client = ModbusTcpClient(self.modbus_ip, port=self.modbus_port)
            if not self.modbus_client.connect():
                self.modbus_status.emit("❌ ไม่สามารถเชื่อมต่อ Modbus ได้")
                return
            
            self.modbus_status.emit("✅ เชื่อมต่อ Modbus สำเร็จ")
            self.is_running = True
            
            while self.is_running and not self.stop_requested:
                # Check M511 (start program)
                try:
                    result = self.modbus_client.read_coils(511, 1, unit=1)
                    if result.isError():
                        m511 = False
                    else:
                        m511 = result.bits[0]
                        # ตรวจสอบการเปลี่ยนสถานะ M511
                        if m511 and not self.last_m511:
                            print("✅ M511 ON: เริ่มการทำงาน")
                            self.modbus_status.emit("✅ M511 ON: เริ่มการทำงาน")
                            self.program_enabled = True
                        elif not m511 and self.last_m511:
                            print("❌ M511 OFF: หยุดการทำงาน")
                            self.modbus_status.emit("❌ M511 OFF: หยุดการทำงาน")
                            self.program_enabled = False
                except Exception as e:
                    print(f"❌ Modbus M511 read error: {e}")
                    m511 = False
                
                self.last_m511 = m511
                
                # Check M301 (trigger) - เพิ่มการตรวจสอบที่แม่นยำขึ้น
                try:
                    result = self.modbus_client.read_coils(301, 1, unit=1)
                    if result.isError():
                        m301 = False
                    else:
                        m301 = result.bits[0]
                        # Debug: แสดงสถานะ M301 (เฉพาะเมื่อเปลี่ยน)
                        # if m301 != self.last_m301:
                        #     print(f"🔍 M301 Status: {m301} (Last: {self.last_m301})")  # Reduced spam
                except Exception as e:
                    print(f"❌ Modbus read error: {e}")
                    m301 = False
                    
                # ตรวจสอบการเปลี่ยนสถานะจาก OFF เป็น ON เท่านั้น
                if m301 and not self.last_m301:
                    # print("✅ M301 TRIGGER DETECTED: OFF → ON")  # Reduced spam
                    self.modbus_status.emit("🔔 M301 ON: ตรวจพบสัญญาณถ่ายภาพ")
                    
                    # ตรวจสอบว่าโปรแกรมเปิดใช้งานอยู่หรือไม่
                    if not self.program_enabled:
                        # print("⚠️ M301 detected but program is stopped (M511 OFF)")  # Reduced spam
                        self.modbus_status.emit("⚠️ M301 ตรวจพบแต่โปรแกรมหยุดทำงาน (M511 OFF)")
                        continue
                    
                    # ตรวจสอบว่า M130 กำลังทำงานอยู่หรือไม่
                    if self.d6006_monitoring:
                        # print("⚠️ M301 detected but M130 is active - ignoring trigger")  # Reduced spam
                        self.modbus_status.emit("⚠️ M301 ตรวจพบแต่ M130 กำลังทำงาน - ไม่สนใจสัญญาณ")
                        continue
                    
                    # Check if capture only mode (ID 5)
                    if self.capture_only_mode:
                        # ตรวจสอบว่าถ่ายครบจำนวนแล้วหรือยัง
                        if self.capture_limit_reached:
                            print(f"⏹️ ID 5 CAPTURE ONLY: ถ่ายครบแล้ว ({self.capture_limit_count} ครั้ง) - ไม่รับ M301 ต่อ")
                            self.modbus_status.emit(f"⏹️ ID 5: ถ่ายครบแล้ว ({self.capture_limit_count} ครั้ง) - ไม่รับ M301 ต่อ")
                            continue  # หยุดการถ่ายภาพต่อ
                        
                        # ID 5: Capture only mode - just capture images, no processing
                        if not self.m600_reset_pending:
                            # M600 พร้อม - ถ่ายภาพทันที → ON M600 → รอ M401 ON
                            print("📸 ID 5 CAPTURE ONLY: M301 ON - ถ่ายภาพทันที (M600 พร้อม)")
                            self.modbus_status.emit("📸 ID 5: ถ่ายภาพทันที (M600 พร้อม)")
                            self.capture_only_trigger.emit()  # Emit signal for capture only
                        else:
                            # M600 ยังไม่ reset - เก็บคิวไว้ → รอ M401=ON → ถ่ายภาพจากคิว
                            self.pending_m301_count += 1
                            print(f"📸 ID 5 CAPTURE ONLY: M301 ON - เพิ่มคิวถ่ายภาพ (คิวปัจจุบัน: {self.pending_m301_count}, รอ M401=ON)")
                            self.modbus_status.emit(f"📸 ID 5: เพิ่มคิวถ่ายภาพ (คิว: {self.pending_m301_count}, รอ M401=ON)")
                    elif not self.m600_reset_pending:
                        # M600 พร้อม - เริ่มกระบวนการทันที
                        print("🚀 M600 พร้อม - เริ่มถ่ายภาพทันที")
                        self.trigger_detected.emit()
                    else:
                        # M600 ยังไม่ reset - ถ่ายภาพและประมวลผลทันที (แบบเงียบ) แล้วเก็บผลลัพธ์ไว้
                        print(f"📋 M600 ยังไม่ reset - ถ่ายภาพและประมวลผลทันที (เก็บผลลัพธ์ไว้)")
                        self.modbus_status.emit(f"📋 ถ่ายภาพและประมวลผลทันที (รอ M401 ON เพื่อแสดงผล)")
                        self.silent_trigger_detected.emit()  # ส่งสัญญาณให้ถ่ายภาพและประมวลผลแบบเงียบ
                        self.pending_m301_count += 1
                        print(f"📋 เพิ่มคิวผลลัพธ์ (คิวปัจจุบัน: {self.pending_m301_count})")
                        self.modbus_status.emit(f"📋 เพิ่มคิวผลลัพธ์ (คิวปัจจุบัน: {self.pending_m301_count})")
                        
                elif not m301 and self.last_m301:
                    print("📉 M301 RESET: ON → OFF")
                    self.modbus_status.emit("📉 M301 OFF: รอสัญญาณถัดไป")
                
                self.last_m301 = m301
                
                # ตรวจสอบ M402 สำหรับ M130 และ M850 (ใช้ M402 แทน D6006)
                if self.d6006_monitoring or self.angle3_retry_mode:
                    try:
                        result = self.modbus_client.read_coils(402, 1, unit=1)
                        if result.isError():
                            m402 = False
                        else:
                            m402 = result.bits[0]
                        # ส่งสถานะ M402 ไปยัง GUI เฉพาะเมื่อค่าเปลี่ยน
                        if m402 != self.last_m402_value:
                            self.m402_status_updated.emit(m402)
                            self.last_m402_value = m402
                    except:
                        m402 = False
                        print("❌ M402 Read Error")
                        
                    # เมื่อ M402 ON (แทน D6006 = 300)
                    if m402 and (self.last_m402_value is None or not self.last_m402_value):
                        try:
                            # RESET M130, M850 และ M600
                            self.modbus_client.write_coil(130, False, unit=1)
                            self.modbus_status.emit("✅ RESET M130 = 0 (M402 ON)")
                            
                            self.modbus_client.write_coil(850, False, unit=1)
                            self.modbus_status.emit("✅ RESET M850 = 0 (M402 ON)")
                            
                            self.modbus_client.write_coil(600, False, unit=1)
                            self.modbus_status.emit("✅ RESET M600 = 0 (M402 ON)")
                            
                            # หยุด monitoring M402 และ reset pending
                            self.d6006_monitoring = False
                            self.m600_reset_pending = False
                            
                            # เริ่มโหมดถ่ายภาพซ้ำสำหรับ angle3
                            self.angle3_retry_mode = True
                            self.angle3_retry_count = 0
                            
                            # Reset การประมวลผลเพื่อให้ใช้ภาพใหม่
                            print("🔄 RESET การประมวลผลเพื่อใช้ภาพใหม่...")
                            self.modbus_status.emit("🔄 RESET การประมวลผลเพื่อใช้ภาพใหม่...")
                            
                            # ส่งสัญญาณให้ GUI reset การประมวลผล
                            self.reset_processing_signal.emit()
                            
                            # ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)
                            print("📸 M402 ON: ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)")
                            self.modbus_status.emit("📸 M402 ON: ถ่ายภาพ USB ใหม่ทันที (เหมือน M301)")
                            
                            # ส่งสัญญาณให้ GUI ถ่ายภาพ USB ใหม่ทันที
                            self.trigger_detected.emit()  # ส่งสัญญาณเหมือน M301
                            
                        except Exception as e:
                            print(f"❌ ไม่สามารถ RESET M130, M850 และ M600 ได้: {e}")
                    
                    # เมื่อ M402 OFF (แทน D6006 = 0)
                    if not m402 and self.last_m402_value is True:
                        # M402 OFF - Reset M850 และหยุดโหมดถ่ายภาพซ้ำ
                        if self.angle3_retry_mode:
                            try:
                                self.modbus_client.write_coil(850, False, unit=1)
                                self.modbus_status.emit("✅ RESET M850 = 0 (M402 OFF)")
                                
                                # หยุดโหมดถ่ายภาพซ้ำ
                                self.angle3_retry_mode = False
                                self.angle3_retry_count = 0
                                self.modbus_status.emit("🛑 หยุดโหมดถ่ายภาพซ้ำ (M402 OFF)")
                                
                            except Exception as e:
                                print(f"❌ ไม่สามารถ RESET M850 ได้: {e}")
                
                # ตรวจสอบ M401 เพื่อ reset M600 (ใช้ M401 ON แทน D6004 = 200)
                if self.m600_reset_pending:
                    # อ่าน M401 (coil address 401) - ใช้ทั้งโหมด auto และ capture only
                    try:
                        result = self.modbus_client.read_coils(401, 1, unit=1)
                        if result.isError():
                            m401 = False
                        else:
                            m401 = result.bits[0]
                        # ส่งสถานะ M401 ไปยัง GUI
                        self.m401_status_updated.emit(m401)
                    except:
                        m401 = False
                        print("❌ M401 Read Error")
                    
                    # ถ้า M401 = ON ให้ทำการ reset M600 และ coils อื่นๆ
                    if m401:
                        # สำหรับ capture only mode: ถ้าถ่ายครบแล้วและรอหยุด ให้หยุดหลัง M401 ON 3 วินาที
                        if self.capture_only_mode and self.capture_limit_reached:
                            print(f"⏳ CAPTURE WITH LIMIT: ถ่ายครบแล้ว - รอ 3 วินาทีหลัง M401 ON แล้วหยุด")
                            self.modbus_status.emit("⏳ ถ่ายครบแล้ว - รอ 3 วินาทีแล้วหยุด")
                            # ส่งสัญญาณให้ GUI หยุดระบบหลัง 3 วินาที
                            self.capture_limit_reached_signal.emit()
                            # Reset flag เพื่อไม่ให้ส่ง signal ซ้ำ
                            self.capture_limit_reached = False
                        
                        # ทำการ reset M600 และ coils อื่นๆ (เหมือนเดิม)
                        print(f"✅ M401 ON มาถึงแล้ว! (คิว: {self.pending_m301_count})")
                        try:
                            # RESET M600
                            self.modbus_client.write_coil(600, False, unit=1)
                            print("✅ RESET M600 = 0")
                            self.modbus_status.emit("✅ RESET M600 = 0")
                            
                            # Reset D7009 เพื่อเตรียมรับค่าใหม่
                            try:
                                reset_d7009 = self.write_register(7009, 0)
                                if reset_d7009:
                                    print("✅ RESET D7009 = 0 (พร้อมรับค่าใหม่)")
                                    self.modbus_status.emit("✅ RESET D7009 = 0 (พร้อมรับค่าใหม่)")
                                else:
                                    print("⚠️ ไม่สามารถ RESET D7009 = 0 ได้")
                                    self.modbus_status.emit("⚠️ ไม่สามารถ RESET D7009 = 0 ได้")
                            except Exception as e:
                                print(f"❌ RESET D7009 Error: {e}")
                                self.modbus_status.emit(f"❌ RESET D7009 Error: {e}")
                            
                            # RESET M100, M110, M120, M130
                            self.modbus_client.write_coil(100, False, unit=1)
                            print("✅ RESET M100 = 0")
                            self.modbus_status.emit("✅ RESET M100 = 0")
                            
                            self.modbus_client.write_coil(110, False, unit=1)
                            print("✅ RESET M110 = 0")
                            self.modbus_status.emit("✅ RESET M110 = 0")
                            
                            self.modbus_client.write_coil(120, False, unit=1)
                            print("✅ RESET M120 = 0")
                            self.modbus_status.emit("✅ RESET M120 = 0")
                            
                            self.modbus_client.write_coil(130, False, unit=1)
                            print("✅ RESET M130 = 0")
                            self.modbus_status.emit("✅ RESET M130 = 0")
                            
                            # RESET M140 (ฝาไม่ผ่าน)
                            self.modbus_client.write_coil(140, False, unit=1)
                            print("✅ RESET M140 = 0")
                            self.modbus_status.emit("✅ RESET M140 = 0")
                            
                            # RESET M720, M721, M722 (โหมดรสชาติ)
                            self.modbus_client.write_coil(720, False, unit=1)
                            print("✅ RESET M720 = 0")
                            self.modbus_status.emit("✅ RESET M720 = 0")
                            
                            self.modbus_client.write_coil(721, False, unit=1)
                            print("✅ RESET M721 = 0")
                            self.modbus_status.emit("✅ RESET M721 = 0")
                            
                            self.modbus_client.write_coil(722, False, unit=1)
                            print("✅ RESET M722 = 0")
                            self.modbus_status.emit("✅ RESET M722 = 0")
                            
                            # RESET M730-M735 (2-taste combinations)
                            self.modbus_client.write_coil(730, False, unit=1)
                            print("✅ RESET M730 = 0")
                            self.modbus_status.emit("✅ RESET M730 = 0")
                            
                            self.modbus_client.write_coil(731, False, unit=1)
                            print("✅ RESET M731 = 0")
                            self.modbus_status.emit("✅ RESET M731 = 0")
                            
                            self.modbus_client.write_coil(732, False, unit=1)
                            print("✅ RESET M732 = 0")
                            self.modbus_status.emit("✅ RESET M732 = 0")
                            
                            self.modbus_client.write_coil(733, False, unit=1)
                            print("✅ RESET M733 = 0")
                            self.modbus_status.emit("✅ RESET M733 = 0")
                            
                            self.modbus_client.write_coil(734, False, unit=1)
                            print("✅ RESET M734 = 0")
                            self.modbus_status.emit("✅ RESET M734 = 0")
                            
                            self.modbus_client.write_coil(735, False, unit=1)
                            print("✅ RESET M735 = 0")
                            self.modbus_status.emit("✅ RESET M735 = 0")
                            
                            # RESET M740-M742 (1-taste signals)
                            self.modbus_client.write_coil(740, False, unit=1)
                            print("✅ RESET M740 = 0")
                            self.modbus_status.emit("✅ RESET M740 = 0")
                            
                            self.modbus_client.write_coil(741, False, unit=1)
                            print("✅ RESET M741 = 0")
                            self.modbus_status.emit("✅ RESET M741 = 0")
                            
                            self.modbus_client.write_coil(742, False, unit=1)
                            print("✅ RESET M742 = 0")
                            self.modbus_status.emit("✅ RESET M742 = 0")
                            
                        except Exception as e:
                            print(f"❌ ไม่สามารถ RESET Modbus ได้: {e}")
                            
                        self.m600_reset_pending = False
                        
                        # Check if capture only mode (ID 5) - handle differently
                        if self.capture_only_mode:
                            # ตรวจสอบว่าถ่ายครบจำนวนแล้วหรือยัง
                            if self.capture_limit_reached:
                                print(f"⏹️ ID 5 CAPTURE ONLY: ถ่ายครบแล้ว ({self.capture_limit_count} ครั้ง) - ไม่ถ่ายภาพจากคิวต่อ")
                                self.modbus_status.emit(f"⏹️ ID 5: ถ่ายครบแล้ว ({self.capture_limit_count} ครั้ง) - ไม่ถ่ายภาพต่อ")
                                # ไม่ต้องถ่ายภาพจากคิวต่อ - รอหยุดระบบ
                            elif self.pending_m301_count > 0:
                                # มีคิว - ถ่ายภาพจากคิว
                                print(f"📸 ID 5 CAPTURE ONLY: M401=ON - ถ่ายภาพจากคิว (คิว: {self.pending_m301_count})")
                                self.modbus_status.emit(f"📸 ID 5: ถ่ายภาพจากคิว (คิว: {self.pending_m301_count})")
                                self.capture_only_trigger.emit()  # Emit signal for capture only
                                
                                # ลดคิวลง 1
                                self.pending_m301_count -= 1
                                print(f"📋 CAPTURE ONLY: ลดคิวลงเหลือ: {self.pending_m301_count}")
                                
                                # ON M600 ต่อทันที (ถ้ายังมีคิว หรือรอรอบถัดไป)
                                try:
                                    if self.modbus_client:
                                        self.modbus_client.write_coil(600, True, unit=1)
                                        # ไม่ต้อง print log นี้ทุกครั้ง (ลด spam)
                                        self.modbus_status.emit("✅ CAPTURE ONLY: ON M600 ต่อทันที")
                                        self.m600_reset_pending = True  # Set pending again for next cycle
                                    else:
                                        print("❌ CAPTURE ONLY: Modbus client ไม่พร้อมใช้งาน")
                                except Exception as e:
                                    print(f"❌ CAPTURE ONLY: ไม่สามารถ ON M600 ต่อได้: {e}")
                            else:
                                # ไม่มีคิว - ON M600 ต่อเพื่อรอ M301 ถัดไป (ถ้ายังไม่ครบ)
                                if not self.capture_limit_reached:
                                    try:
                                        if self.modbus_client:
                                            self.modbus_client.write_coil(600, True, unit=1)
                                            print("✅ CAPTURE ONLY: ON M600 ต่อทันที (รอ M301 ถัดไป)")
                                            self.modbus_status.emit("✅ CAPTURE ONLY: ON M600 ต่อทันที (รอ M301 ถัดไป)")
                                            self.m600_reset_pending = True  # Set pending again for next cycle
                                        else:
                                            print("❌ CAPTURE ONLY: Modbus client ไม่พร้อมใช้งาน")
                                    except Exception as e:
                                        print(f"❌ CAPTURE ONLY: ไม่สามารถ ON M600 ต่อได้: {e}")
                        # แสดงผลลัพธ์จาก queue (ถ้ามี) หรือถ่ายภาพใหม่ (ถ้าไม่มี)
                        elif len(self.pending_results_queue) > 0:
                            # มีผลลัพธ์ที่ประมวลผลแล้ว - แสดงผลและส่งสัญญาณ
                            bottle_result, cap_result = self.pending_results_queue.pop(0)
                            print(f"📋 แสดงผลลัพธ์จากคิว (คิวเหลือ: {len(self.pending_results_queue)})")
                            self.modbus_status.emit(f"📋 แสดงผลลัพธ์จากคิว (คิวเหลือ: {len(self.pending_results_queue)})")
                            self.result_ready_to_display.emit(bottle_result, cap_result)
                            
                            # ลดคิวลง 1
                            if self.pending_m301_count > 0:
                                self.pending_m301_count -= 1
                                print(f"📋 ลดคิวลงเหลือ: {self.pending_m301_count}")
                            
                            # ถ้ายังมีคิวอยู่ ให้แสดงสถานะ
                            if len(self.pending_results_queue) > 0:
                                print(f"📋 ยังมีคิวรออยู่: {len(self.pending_results_queue)}")
                                self.modbus_status.emit(f"📋 ยังมีคิวรออยู่: {len(self.pending_results_queue)}")
                        elif self.pending_m301_count > 0:
                            # ไม่มีผลใน queue แต่มีคิว = รอผลจาก silent อยู่ — ไม่ถ่ายภาพใหม่ (กันรอบที่ 3 หยิบอากาศ)
                            print(f"⏳ รอผลจากคิว (ยังประมวลผลไม่เสร็จ, คิว: {self.pending_m301_count}) - ไม่ถ่ายภาพใหม่")
                            self.modbus_status.emit(f"⏳ รอผลจากคิว (คิว: {self.pending_m301_count})")
                        else:
                            print("✅ ไม่มีคิวถ่ายภาพ พร้อมรับ M301 ใหม่")
                            self.modbus_status.emit("✅ ไม่มีคิวถ่ายภาพ พร้อมรับ M301 ใหม่")
                
                # Check M512 (condition for M513)
                try:
                    result = self.modbus_client.read_coils(512, 1, unit=1)
                    m512 = not result.isError() and result.bits[0]
                except:
                    m512 = False
                    
                if m512 and not self.last_m512:
                    self.modbus_status.emit("🔔 M512 ON: พร้อมรับ M513 เพื่อออกจากโปรแกรม")
                    self.m512_ready = True
                
                self.last_m512 = m512
                
                # Check M513 (stop program) - ไม่ต้องรอ M512
                try:
                    result = self.modbus_client.read_coils(513, 1, unit=1)
                    m513 = not result.isError() and result.bits[0]
                except:
                    m513 = False
                    
                if m513 and not self.last_m513:
                    self.modbus_status.emit("🔔 M513 ON: หยุดการทำงาน")
                    self.program_enabled = False
                    
                    # Reset M700 และ M701 เมื่อ M513 ON
                    try:
                        self.reset_m700()
                        print("✅ RESET M700 = 0 (จาก M513)")
                        self.modbus_status.emit("✅ RESET M700 = 0 (จาก M513)")
                        
                        self.reset_m701()
                        print("✅ RESET M701 = 0 (จาก M513)")
                        self.modbus_status.emit("✅ RESET M701 = 0 (จาก M513)")
                    except Exception as e:
                        print(f"❌ ไม่สามารถ RESET M700/M701 ได้: {e}")
                    
                    # ส่งสัญญาณให้ GUI ทำการ reset ทั้งหมดเหมือนกดปุ่ม Stop
                    self.m513_stop_signal.emit()
                    print("🔄 M513: ส่งสัญญาณให้ GUI reset ทั้งหมด")
                    
                    # ไม่ break ออกจาก loop เพื่อให้ Modbus ยังทำงานต่อ
                
                self.last_m513 = m513
                
                # ตรวจสอบ M403, M404, M405, M406 เพื่อตรวจสอบประเภทขวด (ใช้แทน D6007)
                # อ่านตลอดในโหมด auto (เหมือน D6007 เดิม)
                if not self.capture_only_mode:
                    try:
                        # อ่าน M403, M404, M405, M406 พร้อมกัน
                        result = self.modbus_client.read_coils(403, 4, unit=1)
                        if not result.isError() and result.bits:
                            m403 = result.bits[0] if len(result.bits) > 0 else False
                            m404 = result.bits[1] if len(result.bits) > 1 else False
                            m405 = result.bits[2] if len(result.bits) > 2 else False
                            m406 = result.bits[3] if len(result.bits) > 3 else False
                            
                            # ส่งค่าไปยัง GUI เฉพาะเมื่อค่าเปลี่ยน
                            if m403 != self.last_m403_value:
                                self.m403_status_updated.emit(m403)
                                self.last_m403_value = m403
                            
                            if m404 != self.last_m404_value:
                                self.m404_status_updated.emit(m404)
                                self.last_m404_value = m404
                            
                            if m405 != self.last_m405_value:
                                self.m405_status_updated.emit(m405)
                                self.last_m405_value = m405
                            
                            if m406 != self.last_m406_value:
                                self.m406_status_updated.emit(m406)
                                self.last_m406_value = m406
                    except:
                        pass  # ไม่ print log เพื่อไม่ให้รกตา
                
                # ตรวจสอบ D5002 เพื่อแสดงสถานะระบบ (0: Close, 1: Running, 2: Break Point, 3: Pause, 4: Pre-run)
                try:
                    result = self.modbus_client.read_holding_registers(5002, count=1, unit=2)
                    d5002 = result.registers[0] if not result.isError() else None
                    # ส่งค่า D5002 ไปยัง GUI เฉพาะเมื่อค่าเปลี่ยน
                    if d5002 is not None and d5002 != self.last_d5002_value:
                        self.d5002_status_updated.emit(d5002)
                        self.last_d5002_value = d5002
                except:
                    d5002 = None
                    print("❌ D5002 Read Error")
                
                # ตรวจสอบ D5001 เพื่อแสดง error code
                try:
                    result = self.modbus_client.read_holding_registers(5001, count=1, unit=2)
                    d5001 = result.registers[0] if not result.isError() else None
                    # ส่งค่า D5001 ไปยัง GUI เฉพาะเมื่อค่าเปลี่ยน
                    if d5001 is not None and d5001 != self.last_d5001_value:
                        self.d5001_status_updated.emit(d5001)
                        self.last_d5001_value = d5001
                except:
                    d5001 = None
                    print("❌ D5001 Read Error")
                
                # อ่าน M401 สำหรับแสดงสถานะใน status tab (เฉพาะโหมด capture only)
                if self.capture_only_mode:
                    try:
                        result = self.modbus_client.read_coils(401, 1, unit=1)
                        if not result.isError():
                            m401 = result.bits[0]
                            # ส่งสถานะ M401 ไปยัง GUI
                            self.m401_status_updated.emit(m401)
                    except:
                        pass  # ไม่ต้อง print error ถ้าไม่สามารถอ่านได้
                
                # Sleep to reduce CPU usage
                time.sleep(0.1)
                
        except Exception as e:
            self.modbus_status.emit(f"❌ Modbus Error: {e}")
        finally:
            if self.modbus_client:
                self.modbus_client.close()
                self.modbus_status.emit("🔌 ปิด Modbus connection")
    
    def write_coil(self, coil_address, value):
        """Write coil to Modbus"""
        # ไม่ต้องตรวจสอบเงื่อนไขใดๆ - สามารถเขียนได้ตลอด
        try:
            if self.modbus_client and self.modbus_client.is_socket_open():
                result = self.modbus_client.write_coil(coil_address, value, unit=1)
                if not result.isError():
                    print(f"✅ ON M{coil_address} = {value}")
                    return True
                else:
                    print(f"❌ ไม่สามารถ ON M{coil_address} ได้")
                    return False
            else:
                print("❌ Modbus client ไม่พร้อมใช้งาน")
                return False
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการ ON M{coil_address}: {e}")
            return False
    
    def write_register(self, register_address, value):
        """Write register to Modbus"""
        try:
            print(f"🔄 write_register: กำลังเขียน D{register_address} = {value}")
            if self.modbus_client and self.modbus_client.is_socket_open():
                print(f"✅ Modbus client พร้อมใช้งาน - กำลังส่งคำสั่ง")
                result = self.modbus_client.write_register(register_address, value, unit=1)
                if not result.isError():
                    print(f"✅ Write D{register_address} = {value} สำเร็จ")
                    return True
                else:
                    print(f"❌ ไม่สามารถเขียน D{register_address} ได้ - Error: {result}")
                    return False
            else:
                print("❌ Modbus client ไม่พร้อมใช้งาน")
                print(f"   - modbus_client exists: {self.modbus_client is not None}")
                if self.modbus_client:
                    print(f"   - socket open: {self.modbus_client.is_socket_open()}")
                return False
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการเขียน D{register_address}: {e}")
            return False
    
    def set_d4005(self, value=100):
        """Set D4005 to value without automatic reset"""
        try:
            print(f"🔄 set_d4005: เริ่มส่งค่า D4005 = {value}")
            # ส่งค่า D4005 = value
            success = self.write_register(4005, value)
            if success:
                print(f"✅ ส่งค่า D4005 = {value} สำเร็จ")
                return True
            else:
                print(f"❌ ไม่สามารถส่งค่า D4005 = {value} ได้")
                return False
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการส่งค่า D4005: {e}")
            return False
    
    # def reset_d4005(self):
    #     """Reset D4005 to 0 - Currently not used as auto-reset is disabled"""
    #     try:
    #         success = self.write_register(4005, 0)
    #         if success:
    #             print("✅ รีเซ็ท D4005 = 0 สำเร็จ")
    #         else:
    #             print("❌ ไม่สามารถรีเซ็ท D4005 = 0 ได้")
    #         return success
    #     except Exception as e:
    #         print(f"❌ ข้อผิดพลาดในการรีเซ็ท D4005: {e}")
    #         return False
    
    def on_m750(self):
        """ON M750 (angle3)"""
        return self.write_coil(750, True)
    
    def reset_m750(self):
        """RESET M750"""
        return self.write_coil(750, False)
    
    def on_m850(self):
        """ON M850 (non-angle3 detection)"""
        return self.write_coil(850, True)
    
    def on_m450(self):
        """ON M450"""
        return self.write_coil(450, True)
    
    def reset_m450(self):
        """RESET M450"""
        return self.write_coil(450, False)
    
    def reset_m850(self):
        """RESET M850"""
        return self.write_coil(850, False)
    
    def on_m90(self):
        """ON M90 (ID 7 full auto mode)"""
        return self.write_coil(90, True)
    
    def reset_m90(self):
        """RESET M90"""
        return self.write_coil(90, False)
    
    def on_m91(self):
        """ON M91 (ID 5 capture only mode)"""
        return self.write_coil(91, True)
    
    def reset_m91(self):
        """RESET M91"""
        return self.write_coil(91, False)
    
    def on_m80(self):
        """ON M80 (START button)"""
        return self.write_coil(80, True)
    
    def reset_m80(self):
        """RESET M80"""
        return self.write_coil(80, False)
    
    def on_m81(self):
        """ON M81 (STOP button)"""
        return self.write_coil(81, True)
    
    def reset_m81(self):
        """RESET M81"""
        return self.write_coil(81, False)
    
    def on_m100(self):
        """ON M100 (ดั้งเดิม) → D7009 = 10"""
        success = self.write_register(7009, 10)
        if success:
            print("✅ ON M100: D7009 = 10 (ดั้งเดิม)")
        else:
            print("❌ ON M100: ไม่สามารถเขียน D7009 = 10 ได้")
        return success
    
    def on_m110(self):
        """ON M110 (น้ำตาล 2%) → D7009 = 20"""
        success = self.write_register(7009, 20)
        if success:
            print("✅ ON M110: D7009 = 20 (น้ำตาล 2%)")
        else:
            print("❌ ON M110: ไม่สามารถเขียน D7009 = 20 ได้")
        return success
    
    def on_m120(self):
        """ON M120 (ผสมแมงลัก) → D7009 = 30"""
        success = self.write_register(7009, 30)
        if success:
            print("✅ ON M120: D7009 = 30 (ผสมแมงลัก)")
        else:
            print("❌ ON M120: ไม่สามารถเขียน D7009 = 30 ได้")
        return success
    
    def on_m140(self):
        """ส่งค่า 50 ไป D7009 (ฝาไม่ผ่าน - ไม่ตรงกับฟอร์ม)"""
        success = self.write_register(7009, 50)
        if success:
            print("✅ ON M140: D7009 = 50 (ฝาไม่ผ่าน - ไม่ตรงกับฟอร์ม)")
        else:
            print("❌ ON M140: ไม่สามารถเขียน D7009 = 50 ได้")
        return success
    
    def reset_m140(self):
        """RESET M140"""
        return self.write_coil(140, False)
    
    def on_m600(self):
        """ON M600"""
        success = self.write_coil(600, True)
        if success:
            self.m600_reset_pending = True
            print("⏳ รอ M401 ON เพื่อ RESET M600, M100, M110, M120, M130, M140...")
        return success
    
    def reset_m100(self):
        """RESET M100"""
        return self.write_coil(100, False)
    
    def reset_m110(self):
        """RESET M110"""
        return self.write_coil(110, False)
    
    def reset_m120(self):
        """RESET M120"""
        return self.write_coil(120, False)
    
    def reset_m130(self):
        """RESET M130"""
        return self.write_coil(130, False)
    
    def reset_all_bottle_types(self):
        """RESET M100, M110, M120, M130 ทั้งหมด"""
        success = True
        success &= self.reset_m100()
        success &= self.reset_m110()
        success &= self.reset_m120()
        success &= self.reset_m130()
        return success
    
    def on_m700(self):
        """ON M700 (START) and reset M701"""
        success = True
        # Reset M701 ก่อน
        success &= self.reset_m701()
        # จากนั้น ON M700
        success &= self.write_coil(700, True)
        return success
    
    def on_m701(self):
        """ON M701 (STOP) and reset M700"""
        success = True
        # Reset M700 ก่อน
        success &= self.reset_m700()
        # จากนั้น ON M701
        success &= self.write_coil(701, True)
        # Reset M701 ตัวเองหลัง 1.5 วินาที (pulse)
        QTimer.singleShot(1500, lambda: self.reset_m701_delayed())
        return success
    
    def reset_m701_delayed(self):
        """Reset M701 after delay"""
        success = self.reset_m701()
        if success:
            print("✅ RESET M701 = 0 (หลัง 1.5 วินาที)")
        else:
            print("❌ ไม่สามารถ RESET M701 ได้")
    
    def reset_m700(self):
        """RESET M700"""
        return self.write_coil(700, False)
    
    def reset_m701(self):
        """RESET M701"""
        return self.write_coil(701, False)
    
    def on_m503(self):
        """ON M503 (เปิด Gripper)"""
        return self.write_coil(503, True)
    
    def on_m505(self):
        """ON M505 (ปิด Gripper)"""
        return self.write_coil(505, True)
    
    def reset_m503(self):
        """RESET M503"""
        return self.write_coil(503, False)
    
    def reset_m505(self):
        """RESET M505"""
        return self.write_coil(505, False)
    
    def on_m507(self):
        """ON M507 (กลับทาง Motor)"""
        return self.write_coil(507, True)
    
    def reset_m507(self):
        """RESET M507 (กลับทาง Motor)"""
        return self.write_coil(507, False)
    
    def reset_m600(self):
        """RESET M600"""
        success = self.write_coil(600, False)
        if success:
            self.m600_reset_pending = False
            print("✅ RESET M600 = 0 (พร้อมรับงานใหม่)")
        return success
    
    
    def reset_m720(self):
        """RESET M720"""
        return self.write_coil(720, False)
    
    def reset_m721(self):
        """RESET M721"""
        return self.write_coil(721, False)
    
    def reset_m722(self):
        """RESET M722"""
        return self.write_coil(722, False)
    
    
    def reset_m730(self):
        """RESET M730"""
        return self.write_coil(730, False)
    
    def reset_m731(self):
        """RESET M731"""
        return self.write_coil(731, False)
    
    def reset_m732(self):
        """RESET M732"""
        return self.write_coil(732, False)
    
    def reset_m733(self):
        """RESET M733"""
        return self.write_coil(733, False)
    
    def reset_m734(self):
        """RESET M734"""
        return self.write_coil(734, False)
    
    def reset_m735(self):
        """RESET M735"""
        return self.write_coil(735, False)
    
    
    def reset_m740(self):
        """RESET M740"""
        return self.write_coil(740, False)
    
    def reset_m741(self):
        """RESET M741"""
        return self.write_coil(741, False)
    
    def reset_m742(self):
        """RESET M742"""
        return self.write_coil(742, False)
    
    def start_d6006_monitoring(self):
        """Start monitoring M402 for M130 (ใช้ M402 แทน D6006)"""
        self.d6006_monitoring = True
        print("🔍 เริ่มต้นการอ่าน M402 สำหรับ M130...")
        self.modbus_status.emit("🔍 เริ่มต้นการอ่าน M402 สำหรับ M130...")
    
    def stop(self):
        """Stop Modbus thread"""
        self.is_running = False
        self.stop_requested = True


class BottleDetectionThread(QThread):
    """Thread for processing bottle detection to avoid GUI freezing"""
    result_ready = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self, image, save_crops: bool = False, output_folder: str = "", selected_tastes: list = None):
        super().__init__()
        self.image = image
        self.save_crops = save_crops
        self.selected_tastes = selected_tastes or ["M100", "M110", "M120"]  # Default to all tastes
        self.output_folder = output_folder
        
    def run(self):
        try:
            print("🔄 BOTTLE DETECTION THREAD: Starting processing...")
            print(f"🔄 BOTTLE DETECTION THREAD: Image shape: {self.image.shape}")
            self.status_updated.emit("กำลังประมวลผลภาพ...")
            self.progress_updated.emit(10)
        
            # Save temporary image for processing
            temp_path = "temp_camera_image.jpg"
            cv2.imwrite(temp_path, self.image)
            print("🔄 BOTTLE DETECTION THREAD: Saved temp image")
            self.progress_updated.emit(20)
            
            # Process the image
            print("🔄 BOTTLE DETECTION THREAD: Calling process_bottle_image_simple...")
            self.status_updated.emit("กำลังตรวจจับขวด...")
            self.progress_updated.emit(30)
            result = process_bottle_image_simple(temp_path)
            
            # ตรวจสอบว่า result เป็น dictionary หรือไม่
            if not isinstance(result, dict):
                error_msg = f"Expected dict, got {type(result).__name__}: {result}"
                print(f"❌ BOTTLE DETECTION THREAD: {error_msg}")
                raise TypeError(error_msg)
            
            print(f"🔄 BOTTLE DETECTION THREAD: Processing result type: {type(result)}")
            print(f"🔄 BOTTLE DETECTION THREAD: Processing result keys: {list(result.keys())}")
            self.progress_updated.emit(50)
            
            # Add image info to result
            result['image'] = self.image
            result['image_shape'] = self.image.shape
            
            # Check for "angle3" in YOLO detections first (before cropping type regions)
            self.status_updated.emit("กำลังตรวจสอบ angle3...")
            self.progress_updated.emit(55)
            
            # ตรวจสอบว่า result มี summary และ detections หรือไม่
            if 'summary' not in result:
                print("⚠️ BOTTLE DETECTION THREAD: No 'summary' in result, creating default summary")
                result['summary'] = {
                    'total_detections': 0,
                    'type_detections': 0,
                    'detected_labels': []
                }
            
            if 'detections' not in result:
                print("⚠️ BOTTLE DETECTION THREAD: No 'detections' in result, creating empty list")
                result['detections'] = []
            
            print(f"🔍 BOTTLE DETECTION THREAD: YOLO detections: {result.get('detections', [])}")
            print(f"🔍 BOTTLE DETECTION THREAD: Summary: {result.get('summary', {})}")
            
            # Check if "angle3" is found in YOLO detections
            summary = result.get('summary', {})
            if not isinstance(summary, dict):
                print("⚠️ BOTTLE DETECTION THREAD: Summary is not a dict, creating new dict")
                summary = {}
                result['summary'] = summary
            
            detected_labels = summary.get('detected_labels', [])
            if not isinstance(detected_labels, list):
                print("⚠️ BOTTLE DETECTION THREAD: detected_labels is not a list, creating empty list")
                detected_labels = []
                summary['detected_labels'] = detected_labels
            # ตรวจสอบว่า YOLO ตรวจจับได้ angle3 จริงๆ หรือไม่
            angle3_detected_by_yolo = 'angle3' in detected_labels
            if angle3_detected_by_yolo:
                # ตรวจสอบว่า angle3 ถูกตรวจจับพร้อมกับ labels อื่นๆ หรือไม่
                # ถ้าเจอ angle1, type, หรือ angle2 พร้อมกับ angle3 → ไม่ใช่ angle3 จริงๆ (ส่ง 50)
                # ถ้าเจอ angle3 เพียงอย่างเดียว (หรือเจอแค่ angle3 กับ angle2) → เป็น angle3 จริงๆ (ส่ง 40)
                other_labels = [label for label in detected_labels if label not in ['angle3', 'angle2']]
                angle3_detected_alone = len(other_labels) == 0
                
                if angle3_detected_alone:
                    print(f"🎯 BOTTLE DETECTION THREAD: Found 'angle3' alone in YOLO detections - เป็น angle3 จริงๆ (ส่ง 40)")
                    result['combined_ocr_text'] = "angle3 detected by YOLO (alone)"
                    result['bottle_type'] = "M130"
                    result['angle3_detected'] = True
                    result['angle3_detected_alone'] = True  # ตรวจจับได้เพียงอย่างเดียว
                else:
                    print(f"⚠️ BOTTLE DETECTION THREAD: Found 'angle3' with other labels {other_labels} - ไม่ใช่ angle3 จริงๆ (ส่ง 50)")
                    result['combined_ocr_text'] = "angle3 detected by YOLO (with others)"
                    result['bottle_type'] = "M130"
                    result['angle3_detected'] = True
                    result['angle3_detected_alone'] = False  # ตรวจจับได้พร้อมกับ labels อื่นๆ
                
                self.progress_updated.emit(100)
                self.status_updated.emit("พบ angle3 - ประมวลผลเสร็จสิ้น")
                self.result_ready.emit(result)
                return
            else:
                print(f"❌ BOTTLE DETECTION THREAD: 'angle3' not found in YOLO detections")
            
            # Perform OCR on type crops and combine text
            combined_ocr_text = ""
            type_crops = result.get('type_crops')
            if type_crops and isinstance(type_crops, list) and len(type_crops) > 0:
                print(f"🔄 BOTTLE DETECTION THREAD: Found {len(type_crops)} type crops")
                self.status_updated.emit("กำลังประมวลผล OCR...")
                self.progress_updated.emit(60)
                
                total_crops = len(type_crops)
                for i, crop in enumerate(type_crops):
                    print(f"🔄 BOTTLE DETECTION THREAD: Processing crop {i+1}")
                    self.status_updated.emit(f"กำลังประมวลผล OCR... ({i+1}/{total_crops})")
                    
                    # ตรวจสอบว่า crop เป็น dictionary และมี original_crop หรือไม่
                    if not isinstance(crop, dict):
                        print(f"⚠️ BOTTLE DETECTION THREAD: Crop {i+1} is not a dict: {type(crop)}")
                        continue
                    
                    if 'original_crop' not in crop:
                        print(f"⚠️ BOTTLE DETECTION THREAD: Crop {i+1} has no 'original_crop'")
                        continue
                    
                    ocr_results = perform_ocr_on_image(crop['original_crop'])
                    crop['ocr_results'] = ocr_results
                    print(f"🔄 BOTTLE DETECTION THREAD: Crop {i+1} OCR results: {len(ocr_results) if ocr_results else 0}")
                    
                    # อัปเดต progress สำหรับแต่ละ crop
                    progress = 60 + (i + 1) * 20 // total_crops
                    self.progress_updated.emit(progress)
                
                    # รวมข้อความที่อ่านได้
                    if ocr_results and isinstance(ocr_results, list):
                        for ocr_result in ocr_results:
                            if isinstance(ocr_result, dict) and 'text' in ocr_result:
                                combined_ocr_text += ocr_result['text'] + " "
                
                # ลบช่องว่างที่เกิน
                combined_ocr_text = combined_ocr_text.strip()
                
                # ตรวจสอบ bottle type
                self.status_updated.emit("กำลังตรวจสอบประเภทขวด...")
                self.progress_updated.emit(85)
                bottle_type = check_bottle_type(combined_ocr_text, self.selected_tastes)
                result['combined_ocr_text'] = combined_ocr_text
                result['bottle_type'] = bottle_type
                
                print(f"📝 BOTTLE DETECTION THREAD: Combined OCR Text: '{combined_ocr_text}'")
                if bottle_type:
                    print(f"🏷️ BOTTLE DETECTION THREAD: Detected Bottle Type: {bottle_type}")
                else:
                    print("❌ BOTTLE DETECTION THREAD: ไม่พบคำว่า 'เดิม', '2%', หรือ 'ลัก'")
            else:
                print("❌ BOTTLE DETECTION THREAD: No type crops found")
                self.progress_updated.emit(85)
                result['combined_ocr_text'] = ""
                result['bottle_type'] = None
                # ถ้า YOLO ตรวจจับได้ angle3 แต่ไม่มี type crops → ไม่ใช่ angle3 จริงๆ
                if angle3_detected_by_yolo:
                    print(f"⚠️ BOTTLE DETECTION THREAD: YOLO ตรวจจับได้ angle3 แต่ไม่มี type crops - ไม่ใช่ angle3 จริงๆ")
                    result['angle3_detected'] = False
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            # Final progress update
            self.progress_updated.emit(100)
            self.status_updated.emit("ประมวลผลเสร็จสิ้น")
            
            print("🔄 BOTTLE DETECTION THREAD: Emitting result...")
            self.result_ready.emit(result)
                
        except Exception as e:
            print(f"❌ BOTTLE DETECTION THREAD: Error: {e}")
            error_result = {
                'error': str(e),
                'image': self.image,
                'image_shape': self.image.shape if self.image is not None else None
            }
            self.result_ready.emit(error_result)

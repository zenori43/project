"""
ตรวจสอบว่า defect model ใช้ GPU หรือ CPU
"""

import sys
import os

# เพิ่ม path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("ตรวจสอบ Defect Model Device")
print("="*60)

# ตรวจสอบ TensorFlow
try:
    import tensorflow as tf
    print(f"\nTensorFlow version: {tf.__version__}")
    
    # ตรวจสอบ GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\nพบ GPU: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    Details: {details}")
            except:
                pass
    else:
        print("\nไม่พบ GPU - จะใช้ CPU")
    
    # ตรวจสอบว่า GPU ถูกปิดหรือไม่ (Windows)
    if sys.platform == "win32":
        print("\nระบบ: Windows")
        print("  -> Defect model จะถูกบังคับให้ใช้ CPU mode")
        print("  -> เพื่อหลีกเลี่ยง DLL error")
    else:
        print(f"\nระบบ: {sys.platform}")
        if gpus:
            print("  -> Defect model จะใช้ GPU")
        else:
            print("  -> Defect model จะใช้ CPU")
    
except ImportError:
    print("\nไม่พบ TensorFlow")
    print("  -> Defect model ไม่สามารถใช้งานได้")
except Exception as e:
    print(f"\nเกิดข้อผิดพลาด: {e}")

# ตรวจสอบ defect model
print("\n" + "="*60)
print("ทดสอบโหลด Defect Model")
print("="*60)

try:
    from libs.detection.defect_model import _load_defect_model, _get_defect_model_path
    
    model_path = _get_defect_model_path()
    print(f"\nModel path: {model_path}")
    
    if os.path.exists(model_path):
        print("  -> ไฟล์โมเดลพบ")
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"  -> ขนาดไฟล์: {file_size:.2f} MB")
    else:
        print("  -> ไฟล์โมเดลไม่พบ!")
    
    print("\nกำลังโหลดโมเดล...")
    model = _load_defect_model()
    
    if model is not None:
        print("  -> โหลดโมเดลสำเร็จ")
        
        # ตรวจสอบว่าโมเดลใช้ device อะไร
        try:
            import numpy as np
            import cv2
            
            # สร้างภาพทดสอบ
            test_img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # รัน inference
            result = None
            try:
                import tensorflow as tf
                if tf.config.list_physical_devices("GPU"):
                    print("\n  -> ใช้ GPU สำหรับ inference")
                    with tf.device("/GPU:0"):
                        # ทดสอบ predict
                        test_array = np.expand_dims(test_img, axis=0).astype(np.float32) / 255.0
                        _ = model.predict(test_array, verbose=0)
                    result = "GPU"
                else:
                    print("\n  -> ใช้ CPU สำหรับ inference")
                    test_array = np.expand_dims(test_img, axis=0).astype(np.float32) / 255.0
                    _ = model.predict(test_array, verbose=0)
                    result = "CPU"
            except Exception as e:
                print(f"  -> Error: {e}")
                result = "Unknown"
            
            print(f"\nสรุป: Defect model ใช้ {result}")
            
        except Exception as e:
            print(f"  -> ไม่สามารถทดสอบ inference ได้: {e}")
    else:
        print("  -> โหลดโมเดลไม่สำเร็จ")
        
except Exception as e:
    print(f"\nเกิดข้อผิดพลาด: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)

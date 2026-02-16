# -*- coding: utf-8 -*-
"""
Defect inspection model (Good/NG) using fix_best_defect_model.h5.
เรียกใช้หลังครอป angle1 ของขวด - ใช้ tf_keras และ MobileNetV2 preprocess.
รองรับ GPU (TensorFlow) เมื่อมี CUDA
"""
import os
import numpy as np
import cv2
from typing import Dict, Optional

# บังคับใช้ legacy Keras ก่อน import tf_keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ตั้งค่า TensorFlow ให้ใช้ GPU ถ้ามี (ทำก่อนโหลดโมเดล)
def _setup_tf_gpu():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            print(f"✅ Defect model: TensorFlow GPU พร้อม ({len(gpus)} device(s))")
        else:
            print("ℹ️ Defect model: ไม่พบ GPU ใน TensorFlow จะใช้ CPU")
    except Exception as e:
        print(f"ℹ️ Defect model TF GPU config: {e}")

# Lazy-loaded model
_defect_model = None
_defect_model_path = None

def _get_defect_model_path() -> str:
    """Path to fix_best_defect_model.h5 (relative to project root)."""
    try:
        from config.settings import PROJECT_ROOT
        return os.path.join(PROJECT_ROOT, "models", "detection", "fix_best_defect_model.h5")
    except ImportError:
        pass
    # Fallback: same folder as this file -> go up to final
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "..", "..", "models", "detection", "fix_best_defect_model.h5")


def _load_defect_model():
    """โหลดโมเดล defect แบบ lazy (โหลดครั้งแรกเมื่อเรียกใช้). ใช้ GPU ถ้า TensorFlow เห็น CUDA"""
    global _defect_model, _defect_model_path
    if _defect_model is not None:
        return _defect_model
    _setup_tf_gpu()
    path = _get_defect_model_path()
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.exists(path):
        print(f"⚠️ Defect model not found: {path}")
        return None
    for name in ("tf_keras", "tensorflow.keras"):
        try:
            if name == "tf_keras":
                import tf_keras as keras
            else:
                import tensorflow as tf
                keras = tf.keras
            _defect_model = keras.models.load_model(path, compile=False)
            _defect_model_path = path
            # บังคับให้รันบน GPU ถ้ามี (TensorFlow 2 ใช้ GPU เองอยู่แล้ว แต่ระบุชัดเจน)
            try:
                import tensorflow as tf
                if tf.config.list_physical_devices("GPU"):
                    # โมเดลจะใช้ GPU โดยอัตโนมัติเมื่อ predict()
                    print("✅ Defect model loaded:", path, f"({name}), ใช้ GPU")
                else:
                    print("✅ Defect model loaded:", path, f"({name})")
            except Exception:
                print("✅ Defect model loaded:", path, f"({name})")
            return _defect_model
        except Exception as e:
            if name == "tf_keras":
                continue
            print(f"❌ Failed to load defect model: {e}")
            return None
    print("❌ Failed to load defect model: No module named 'tf_keras' or 'tensorflow'. Install: pip install tensorflow==2.13.1")
    return None


def run_defect_inspection(crop_bgr: np.ndarray) -> Optional[Dict]:
    """
    ตรวจสอบ defect จากภาพครอป angle1 (BGR).
    - crop_bgr: ภาพ OpenCV (BGR), ขนาดใดก็ได้ จะ resize เป็น 224x224
    Returns:
        {"score": float, "result": "NG" | "Good"} หรือ None ถ้าโมเดลไม่พร้อม
    """
    model = _load_defect_model()
    if model is None:
        return None
    try:
        from tf_keras.applications.mobilenet_v2 import preprocess_input
    except ImportError:
        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        except ImportError:
            print("⚠️ mobilenet_v2.preprocess_input not available")
            return None

    try:
        # BGR -> RGB แล้ว resize 224x224 (INTER_AREA เร็วกว่าเมื่อย่อภาพ)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        img_array = np.expand_dims(crop_resized, axis=0)
        img_array = preprocess_input(img_array.astype(np.float32))

        # รัน inference บน GPU ถ้ามี
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices("GPU"):
                with tf.device("/GPU:0"):
                    score = float(model.predict(img_array, verbose=0)[0][0])
            else:
                score = float(model.predict(img_array, verbose=0)[0][0])
        except Exception:
            score = float(model.predict(img_array, verbose=0)[0][0])
        status = "NG" if score > 0.5 else "Good"
        return {"score": round(score, 4), "result": status}
    except Exception as e:
        print(f"⚠️ Defect inspection error: {e}")
        return None

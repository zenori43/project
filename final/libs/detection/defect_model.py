# -*- coding: utf-8 -*-
"""
Defect inspection model (Good/NG) using fix_best_defect_model_v4.h5.
เรียกใช้หลังครอป angle1 ของขวด - ใช้ tf_keras และ MobileNetV2 preprocess.
รองรับ GPU (TensorFlow) เมื่อมี CUDA
"""
import os
import sys
import warnings
import numpy as np
import cv2
from contextlib import contextmanager
from typing import Dict, Optional

# ปิด DeprecationWarning จาก TensorFlow/NumPy (np.bool8 ฯลฯ) ก่อนโหลด tf
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np\\.bool.*")

# บังคับใช้ legacy Keras ก่อน import tf_keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"


def _patch_tf_load_context():
    """แก้ไข TensorFlow ที่ไม่มี tf.__internal__.load_context (ความไม่ตรงกัน tf_keras กับ TF บางเวอร์ชัน)"""
    @contextmanager
    def _noop_load_context(options=None):
        yield
    try:
        import sys
        import tensorflow as tf
        _ = getattr(tf, "__internal__", None)
        if hasattr(tf, "compat") and hasattr(tf.compat, "v2"):
            _ = getattr(tf.compat.v2, "__internal__", None)
        # แพตช์ทุกโมดูล __internal__ ของ tensorflow ใน sys.modules
        for modname, mod in list(sys.modules.items()):
            if mod is None or not modname.startswith("tensorflow.") or not modname.endswith(".__internal__"):
                continue
            if not hasattr(mod, "load_context"):
                setattr(mod, "load_context", _noop_load_context)
        internal = getattr(tf, "__internal__", None)
        if internal is not None and not hasattr(internal, "load_context"):
            setattr(internal, "load_context", _noop_load_context)
    except Exception:
        pass

# ตั้งค่า TensorFlow ให้ใช้ GPU ถ้ามี (ทำก่อนโหลดโมเดล)
def _setup_tf_gpu():
    try:
        import tensorflow as tf
        # ปิด GPU เฉพาะบน Windows (หลีกเลี่ยง DLL error) — บน Linux/Jetson ใช้ GPU ได้
        if sys.platform == "win32":
            try:
                tf.config.set_visible_devices([], 'GPU')
                print("ℹ️ Defect model: ใช้ CPU mode (ปิด GPU เพื่อหลีกเลี่ยง DLL error บน Windows)")
            except Exception:
                pass
        # ตรวจสอบ GPU
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    try:
                        # ตั้งค่า memory growth เพื่อหลีกเลี่ยง memory issues
                        tf.config.experimental.set_memory_growth(gpu, True)
                        # ตั้งค่า GPU options เพื่อลด CUDA event sync issues
                        try:
                            tf.config.experimental.set_device_policy('warn')
                        except:
                            pass
                    except RuntimeError:
                        pass
                print(f"✅ Defect model: TensorFlow GPU พร้อม ({len(gpus)} device(s))")
            else:
                print("ℹ️ Defect model: ไม่พบ GPU ใน TensorFlow จะใช้ CPU")
        except:
            print("ℹ️ Defect model: จะใช้ CPU mode")
    except ImportError as e:
        print(f"⚠️ Defect model: ไม่สามารถ import TensorFlow ได้: {e}")
        raise
    except Exception as e:
        print(f"ℹ️ Defect model TF GPU config: {e}")

# Lazy-loaded model
_defect_model = None
_defect_model_path = None

def _print_tensorflow_tips():
    """พิมพ์คำแนะนำเมื่อ import TensorFlow ไม่ได้ (โดยเฉพาะ Python 3.13 / Windows DLL)"""
    py_ver = sys.version_info
    if py_ver >= (3, 13):
        print("   💡 TensorFlow ยังไม่รองรับ Python 3.13 อย่างเป็นทางการ")
        print("   💡 แนะนำ: ใช้ Python 3.10 หรือ 3.11 แล้วติดตั้ง pip install tensorflow-cpu==2.13.1")
    elif py_ver >= (3, 12):
        print("   💡 TensorFlow 2.13 รองรับ Python 3.8–3.11 ถ้าใช้ 3.12+ แนะนำใช้ Python 3.11")
    if sys.platform == "win32":
        print("   💡 Windows: ติดตั้ง Microsoft Visual C++ 2015-2022 Redistributable (x64)")
        print("      ดาวน์โหลด: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("   💡 หรือติดตั้ง: pip install tensorflow-cpu==2.13.1 (สำหรับ Windows / Python 3.10–3.11)")


def _get_defect_model_path() -> str:
    """Path to fix_best_defect_model_v4.h5 (relative to project root)."""
    try:
        from config.settings import PROJECT_ROOT
        return os.path.join(PROJECT_ROOT, "models", "detection", "fix_best_defect_model_v4.h5")
    except ImportError:
        pass
    # Fallback: same folder as this file -> go up to final
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "..", "..", "models", "detection", "fix_best_defect_model_v4.h5")


def _load_defect_model():
    """โหลดโมเดล defect แบบ lazy (โหลดครั้งแรกเมื่อเรียกใช้). ใช้ GPU ถ้า TensorFlow เห็น CUDA"""
    global _defect_model, _defect_model_path
    if _defect_model is not None:
        return _defect_model
    
    try:
        import tensorflow as tf
        # ปิด GPU เฉพาะบน Windows (หลีกเลี่ยง DLL error)
        if sys.platform == "win32":
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
    except ImportError as e:
        print(f"❌ Defect model: ไม่สามารถ import TensorFlow ได้: {e}")
        _print_tensorflow_tips()
        return None
    except Exception as e:
        # ถ้ามี DLL error (มักเกิดบน Windows) ให้ลองใช้ CPU mode
        print(f"⚠️ Defect model: TensorFlow error - จะลองใช้ CPU mode: {e}")
        if sys.platform == "win32":
            try:
                import tensorflow as tf
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
    
    try:
        _setup_tf_gpu()
    except:
        pass  # ข้ามถ้ามีปัญหา
    
    path = _get_defect_model_path()
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.exists(path):
        print(f"⚠️ Defect model not found: {path}")
        return None

    # ลองโหลดด้วย native tf.keras ก่อน (ไม่ใช้ tf_keras เพื่อหลีกเลี่ยง load_context)
    _use_legacy = os.environ.pop("TF_USE_LEGACY_KERAS", None)
    try:
        import tensorflow as tf
        keras = tf.keras
        _defect_model = keras.models.load_model(path, compile=False)
        _defect_model_path = path
        if _use_legacy is not None:
            os.environ["TF_USE_LEGACY_KERAS"] = _use_legacy
        print(f"✅ Defect model loaded: {path} (tensorflow.keras native)")
        return _defect_model
    except Exception:
        if _use_legacy is not None:
            os.environ["TF_USE_LEGACY_KERAS"] = _use_legacy
        pass

    _patch_tf_load_context()

    for name in ("tensorflow.keras", "tf_keras"):
        try:
            if name == "tf_keras":
                import tf_keras as keras
            else:
                import tensorflow as tf
                keras = tf.keras
            
            _defect_model = keras.models.load_model(path, compile=False)
            _defect_model_path = path
            
            print(f"✅ Defect model loaded: {path} ({name})")
            return _defect_model
        except ImportError as e:
            if name == "tf_keras":
                continue
            print(f"⚠️ Defect model: Import error - {e}")
            continue
        except Exception as e:
            if name == "tf_keras":
                continue
            print(f"❌ Failed to load defect model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    print("❌ Failed to load defect model: ไม่สามารถโหลดได้")
    print("   💡 แนะนำ: pip uninstall tensorflow tensorflow-gpu")
    print("   💡 แล้ว: pip install tensorflow-cpu==2.13.1")
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

        # รัน inference บน GPU ถ้ามี (เพิ่ม error handling เพื่อหลีกเลี่ยง CUDA event sync issues)
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    # ใช้ GPU context แต่เพิ่ม error handling
                    with tf.device("/GPU:0"):
                        # ใช้ batch_size=1 เพื่อลด CUDA sync issues
                        score = float(model.predict(img_array, verbose=0, batch_size=1)[0][0])
                except Exception as gpu_err:
                    # ถ้า GPU error ให้ fallback ไป CPU และ clear session
                    print(f"⚠️ Defect GPU inference error: {gpu_err} - falling back to CPU")
                    try:
                        tf.keras.backend.clear_session()
                    except:
                        pass
                    score = float(model.predict(img_array, verbose=0, batch_size=1)[0][0])
            else:
                score = float(model.predict(img_array, verbose=0, batch_size=1)[0][0])
        except Exception as e:
            # Fallback: ลองรันอีกครั้งโดยไม่ระบุ device
            try:
                import tensorflow as tf
                try:
                    tf.keras.backend.clear_session()
                except:
                    pass
                score = float(model.predict(img_array, verbose=0, batch_size=1)[0][0])
            except Exception as e2:
                print(f"❌ Defect inference failed: {e2}")
                import traceback
                traceback.print_exc()
                return None
        status = "Good" if score > 0.6 else "NG"
        return {"score": round(score, 4), "result": status}
    except Exception as e:
        print(f"⚠️ Defect inspection error: {e}")
        return None

# -*- coding: utf-8 -*-
"""
Cap fade inspection model (ฝาจาง) using cap_fade_model.h5.
เทรนแบบเดียวกับ defect (label): score สูง = Good (ไม่จาง), score ต่ำ = NG (Fade).
ใช้ tf_keras และ MobileNetV2 preprocess เหมือน defect_model.
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

def _setup_tf_gpu():
    try:
        import tensorflow as tf
        # ปิด GPU เฉพาะบน Windows (หลีกเลี่ยง DLL error) — บน Linux/Jetson ใช้ GPU ได้
        if sys.platform == "win32":
            try:
                tf.config.set_visible_devices([], 'GPU')
                print("ℹ️ Cap fade model: ใช้ CPU mode (ปิด GPU เพื่อหลีกเลี่ยง DLL error บน Windows)")
            except Exception:
                pass
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
                print(f"✅ Cap fade model: TensorFlow GPU พร้อม ({len(gpus)} device(s))")
            else:
                print("ℹ️ Cap fade model: ไม่พบ GPU ใน TensorFlow จะใช้ CPU")
        except:
            print("ℹ️ Cap fade model: จะใช้ CPU mode")
    except ImportError as e:
        print(f"⚠️ Cap fade model: ไม่สามารถ import TensorFlow ได้: {e}")
        raise
    except Exception as e:
        print(f"ℹ️ Cap fade model TF GPU config: {e}")

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


_cap_fade_model = None
_cap_fade_model_path = None

def _get_cap_fade_model_path() -> str:
    """Path to cap_fade_model.h5 (relative to project root)."""
    try:
        from config.settings import PROJECT_ROOT
        return os.path.join(PROJECT_ROOT, "models", "detection", "cap_fade_model.h5")
    except ImportError:
        pass
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "..", "..", "models", "detection", "cap_fade_model.h5")


def _load_cap_fade_model():
    """โหลดโมเดล cap fade แบบ lazy (โหลดครั้งแรกเมื่อเรียกใช้)."""
    global _cap_fade_model, _cap_fade_model_path
    if _cap_fade_model is not None:
        return _cap_fade_model

    try:
        import tensorflow as tf
        # ปิด GPU เฉพาะบน Windows (หลีกเลี่ยง DLL error)
        if sys.platform == "win32":
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
    except ImportError as e:
        print(f"❌ Cap fade model: ไม่สามารถ import TensorFlow ได้: {e}")
        _print_tensorflow_tips()
        return None
    except Exception as e:
        if sys.platform == "win32":
            try:
                import tensorflow as tf
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass

    try:
        _setup_tf_gpu()
    except:
        pass

    path = _get_cap_fade_model_path()
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.exists(path):
        print(f"⚠️ Cap fade model not found: {path}")
        return None

    # ลองโหลดด้วย native tf.keras ก่อน (ไม่ใช้ tf_keras เพื่อหลีกเลี่ยง load_context)
    _use_legacy = os.environ.pop("TF_USE_LEGACY_KERAS", None)
    try:
        import tensorflow as tf
        keras = tf.keras
        _cap_fade_model = keras.models.load_model(path, compile=False)
        _cap_fade_model_path = path
        if _use_legacy is not None:
            os.environ["TF_USE_LEGACY_KERAS"] = _use_legacy
        print(f"✅ Cap fade model loaded: {path} (tensorflow.keras native)")
        return _cap_fade_model
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

            _cap_fade_model = keras.models.load_model(path, compile=False)
            _cap_fade_model_path = path
            print(f"✅ Cap fade model loaded: {path} ({name})")
            return _cap_fade_model
        except ImportError:
            if name == "tf_keras":
                continue
            continue
        except Exception as e:
            if name == "tf_keras":
                continue
            print(f"❌ Failed to load cap fade model: {e}")
            import traceback
            traceback.print_exc()
            return None

    print("❌ Failed to load cap fade model: ไม่สามารถโหลดได้")
    return None


def run_cap_fade_inspection(cap_bgr: np.ndarray) -> Optional[Dict]:
    """
    ตรวจสอบฝาจางจากภาพครอปฝา (BGR).
    - cap_bgr: ภาพ OpenCV (BGR), ขนาดใดก็ได้ จะ resize เป็น 224x224
    - เทรนแบบเดียวกับ defect: score > 0.5 = Good (ไม่จาง), score <= 0.5 = NG (Fade)
    Returns:
        {"score": float, "result": "Good" | "NG (Fade)"} หรือ None ถ้าโมเดลไม่พร้อม
    """
    model = _load_cap_fade_model()
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
        cap_rgb = cv2.cvtColor(cap_bgr, cv2.COLOR_BGR2RGB)
        cap_resized = cv2.resize(cap_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        img_array = np.expand_dims(cap_resized, axis=0)
        img_array = preprocess_input(img_array.astype(np.float32))

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
                    print(f"⚠️ Cap fade GPU inference error: {gpu_err} - falling back to CPU")
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
                print(f"❌ Cap fade inference failed: {e2}")
                import traceback
                traceback.print_exc()
                return None

        # Good ถ้า score > 0.5 (ไม่จาง), NG (Fade) ถ้า score <= 0.5
        status = "Good" if score > 0.5 else "NG (Fade)"
        return {"score": round(score, 4), "result": status}
    except Exception as e:
        print(f"⚠️ Cap fade inspection error: {e}")
        return None

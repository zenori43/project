# -*- coding: utf-8 -*-
"""
Main Entry Point - จุดเริ่มต้นของแอปพลิเคชัน
"""

import sys
import os
import platform

# ตรวจสอบ OS
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

# หมายเหตุ: Defect model (TensorFlow) ใช้ GPU ได้เมื่อรันผ่าน run.sh
# เพราะ run.sh ตั้ง CUDA_HOME / LD_LIBRARY_PATH ใน shell ก่อนเริ่ม Python
# ถ้ารันจาก IDE หรือคำสั่ง python3 main.py โดยตรง TensorFlow อาจไม่เห็น GPU

# IMPORTANT: Set CUDA environment variables BEFORE importing anything
# This must be done before any other imports that use CUDA
# Set CUDA environment variables (like run.sh does) - สำหรับ Linux เท่านั้น
if IS_LINUX:
    # บังคับให้เห็น GPU ตัวแรก (กัน PyTorch/TensorFlow ไม่เห็น CUDA)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'CUDA_HOME' not in os.environ:
        os.environ['CUDA_HOME'] = '/usr/local/cuda-11.4'
    if 'CUDA_PATH' not in os.environ:
        os.environ['CUDA_PATH'] = '/usr/local/cuda-11.4'

    # Add CUDA paths to LD_LIBRARY_PATH (Linux only)
    cuda_lib_paths = [
        '/usr/local/cuda-11.4/lib64',
        '/usr/local/cuda-11.4/targets/aarch64-linux/lib',
        '/usr/lib/aarch64-linux-gnu/tegra',
        '/usr/lib/aarch64-linux-gnu',
        '/usr/local/lib'
    ]
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    paths_to_add = [p for p in cuda_lib_paths if p not in current_ld_path]
    if paths_to_add:
        new_ld_path = ':'.join(paths_to_add)
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = new_ld_path + ':' + current_ld_path
        else:
            os.environ['LD_LIBRARY_PATH'] = new_ld_path

    # Add OpenCV Python path to PYTHONPATH (Linux only)
    opencv_python_path = '/usr/local/lib/python3.8/site-packages/'
    current_python_path = os.environ.get('PYTHONPATH', '')
    if opencv_python_path not in current_python_path:
        if current_python_path:
            os.environ['PYTHONPATH'] = opencv_python_path + ':' + current_python_path
        else:
            os.environ['PYTHONPATH'] = opencv_python_path

    # Add CUDA bin to PATH (Linux only)
    cuda_bin = '/usr/local/cuda-11.4/bin'
    current_path = os.environ.get('PATH', '')
    if cuda_bin not in current_path:
        os.environ['PATH'] = cuda_bin + ':' + current_path

    # IMPORTANT: Setup CUDA paths BEFORE importing OpenCV or PyTorch
    # This must be done before any other imports that use CUDA
    try:
        from core.cuda_setup import setup_cuda_paths
        setup_cuda_paths()
    except Exception as e:
        print(f"⚠️ Warning: Could not setup CUDA paths: {e}")

    # ตรวจสอบ PyTorch เห็น CUDA หรือไม่ (ต้อง import torch หลัง set env)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA: ใช้ GPU ({torch.cuda.get_device_name(0)})")
        else:
            print("⚠️ PyTorch CUDA: ไม่เห็น GPU — โมเดลจะใช้ CPU (ช้ากว่า)")
            print("   💡 แนะนำ: รันผ่าน ./run.sh ในโฟลเดอร์ final เพื่อให้ LD_LIBRARY_PATH ถูกตั้งใน shell ก่อน")
    except Exception as e:
        print(f"⚠️ PyTorch CUDA check: {e}")

# Import PyQt5 with error handling
try:
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QFont, QIcon
except ImportError as e:
    print("=" * 50)
    print("❌ ERROR: PyQt5 ไม่ได้ติดตั้ง")
    print("=" * 50)
    print(f"Error: {e}")
    print("\n💡 วิธีแก้ไข:")
    print("   pip install PyQt5")
    print("=" * 50)
    input("กด Enter เพื่อปิด...")
    sys.exit(1)


def _excepthook(etype, value, tb):
    """จับ exception ที่ไม่ถูก handle (เช่น ใน Qt slot) ให้พิมพ์ traceback ชัดเจน"""
    import traceback
    print("=" * 60)
    print("❌ Unhandled Python exception")
    print("=" * 60)
    traceback.print_exception(etype, value, tb)
    print("=" * 60)
    # เรียก default behavior ด้วย (อาจแสดง dialog ของ Qt)
    sys.__excepthook__(etype, value, tb)


def main():
    """Main function with loading screen"""
    # ให้ exception ที่หลุดจาก event/slot แสดง traceback ในคอนโซล
    sys.excepthook = _excepthook

    try:
        # สร้าง QApplication ก่อนเพื่อใช้ splash screen
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle('Fusion')
        _logo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "gui", "gui icon", "Logo1.png"
        )
        if os.path.isfile(_logo_path):
            app.setWindowIcon(QIcon(_logo_path))
        
        # ฟอนต์: อังกฤษใช้ Poppins, ภาษาไทยใช้ Mitr (Qt เลือกตาม glyph)
        # setFamilies มีตั้งแต่ Qt 5.13 — PyQt5 เก่าใช้ setFamily อย่างเดียว
        _font = QFont()
        if hasattr(_font, "setFamilies"):
            _font.setFamilies(["Poppins", "Mitr"])
        else:
            _font.setFamily("Poppins")
        _font.setPointSize(10)
        app.setFont(_font)
        
        # Import เบาๆ เพื่อแสดง splash screen
        from PyQt5.QtWidgets import QApplication
        
        # แสดง loading screen ก่อน import modules หนักๆ
        try:
            from gui.splash_screen import show_splash_screen
            splash = show_splash_screen(app)
            use_splash = True
        except Exception as e:
            print(f"⚠️ Warning: Could not load splash screen: {e}")
            splash = None
            use_splash = False
        
        # อัพเดท loading progress
        if use_splash:
            splash.set_progress(1, 10, "Starting system...")
            app.processEvents()  # ให้ GUI อัพเดท
        
        # Import modules ตามขั้นตอน
        try:
            if use_splash:
                splash.set_progress(2, 10, "Loading configuration...")
                app.processEvents()
            print("📦 Loading configuration...")
            from config import settings
            
            if use_splash:
                splash.set_progress(3, 10, "กำลังโหลด Core Modules...")
                app.processEvents()
            print("📦 กำลังโหลด Core Modules...")
            from core import camera_manager
            from core import image_processor
            from core import business_logic
            
            if use_splash:
                splash.set_progress(4, 10, "Loading GUI modules...")
                app.processEvents()
            print("📦 Loading GUI modules...")
            from gui.main_window import BottleDetectionGUI
            
            if use_splash:
                splash.set_progress(5, 10, "Preparing main window...")
                app.processEvents()
            print("📦 Preparing main window...")
            window = BottleDetectionGUI()
            
            # ยังไม่แสดง GUI — รอโหลดกล้อง/โมเดล/Defect/Modbus ในพื้นหลัง แล้วค่อยปิด loading และแสดงหน้าต่างหลัก
            init_progress_step = [6]  # 6=กล้อง 7=Sentech 8=โมเดล 9=Defect 10=Modbus/พร้อม
            def on_init_progress(msg):
                if use_splash and splash:
                    init_progress_step[0] = min(init_progress_step[0] + 1, 10)
                    splash.set_progress(init_progress_step[0], 10, msg)
                    app.processEvents()

            def on_init_complete():
                if use_splash and splash:
                    splash.set_progress(10, 10, "Ready!")
                    app.processEvents()
                    splash.close()
                print("✅ Showing main window...")
                window.show()
            
            window.init_progress.connect(on_init_progress)
            window.init_complete.connect(on_init_complete)
            
            print("✅ โปรแกรมพร้อมใช้งาน!")
            sys.exit(app.exec_())
            
        except Exception as e:
            if use_splash and splash:
                try:
                    splash.show_message(f"Error: {str(e)}")
                    QTimer.singleShot(3000, splash.close)
                except:
                    pass
            
            print("=" * 50)
            print("❌ ERROR: เกิดข้อผิดพลาดในการโหลดโปรแกรม")
            print("=" * 50)
            print(f"Error: {e}")
            print("\n📋 Stack Trace:")
            import traceback
            traceback.print_exc()
            print("=" * 50)
            if IS_WINDOWS:
                input("กด Enter เพื่อปิด...")
            sys.exit(1)
            
    except Exception as e:
        print("=" * 50)
        print("❌ ERROR: เกิดข้อผิดพลาดร้ายแรง")
        print("=" * 50)
        print(f"Error: {e}")
        print("\n📋 Stack Trace:")
        import traceback
        traceback.print_exc()
        print("=" * 50)
        if IS_WINDOWS:
            input("กด Enter เพื่อปิด...")
        sys.exit(1)


if __name__ == '__main__':
    main()


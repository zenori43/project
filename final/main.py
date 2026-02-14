# -*- coding: utf-8 -*-
"""
Main Entry Point - จุดเริ่มต้นของแอปพลิเคชัน
"""

import sys
import os

# ถ้ารันจาก IDE/โดยตรง โดยที่ LD_LIBRARY_PATH ยังไม่มี path Jetson/CUDA
# PyTorch/OpenCV จะโหลดแล้วไม่เห็น GPU — ต้อง set env ก่อนเริ่ม process จึง re-exec ด้วย env จาก set_cuda_env.sh
if sys.platform == "linux":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_script = os.path.join(script_dir, "set_cuda_env.sh")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if os.path.isfile(env_script) and "tegra" not in ld_path:
        import subprocess
        result = subprocess.run(
            ["bash", "-c", "source " + repr(env_script) + " && env"],
            capture_output=True,
            text=True,
            cwd=script_dir,
        )
        if result.returncode == 0:
            new_env = os.environ.copy()
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    new_env[k] = v
            os.execve(sys.executable, [sys.executable, __file__] + sys.argv[1:], new_env)

import platform

# ตรวจสอบ OS
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

# IMPORTANT: Set CUDA environment variables BEFORE importing anything
# This must be done before any other imports that use CUDA
# Set CUDA environment variables (like run.sh does) - สำหรับ Linux เท่านั้น
if IS_LINUX:
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

# Import PyQt5 with error handling
try:
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import QTimer
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


def main():
    """Main function with loading screen"""
    try:
        # สร้าง QApplication ก่อนเพื่อใช้ splash screen
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle('Fusion')
        
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
            splash.set_progress(1, 10, "กำลังเริ่มระบบ...")
            app.processEvents()  # ให้ GUI อัพเดท
        
        # Import modules ตามขั้นตอน
        try:
            if use_splash:
                splash.set_progress(2, 10, "กำลังโหลด Configuration...")
                app.processEvents()
            print("📦 กำลังโหลด Configuration...")
            from config import settings
            
            if use_splash:
                splash.set_progress(3, 10, "กำลังโหลด Core Modules...")
                app.processEvents()
            print("📦 กำลังโหลด Core Modules...")
            from core import camera_manager
            from core import image_processor
            from core import business_logic
            
            if use_splash:
                splash.set_progress(4, 10, "กำลังโหลด GUI Modules...")
                app.processEvents()
            print("📦 กำลังโหลด GUI Modules...")
            from gui.main_window import BottleDetectionGUI
            
            if use_splash:
                splash.set_progress(5, 10, "กำลังเตรียมหน้าต่างหลัก...")
                app.processEvents()
            print("📦 กำลังเตรียมหน้าต่างหลัก...")
            window = BottleDetectionGUI()
            
            if use_splash:
                splash.set_progress(7, 10, "กำลังเริ่มต้นระบบ...")
                app.processEvents()
            
            if use_splash:
                splash.set_progress(8, 10, "กำลังแสดงหน้าต่างหลัก...")
                app.processEvents()
            print("✅ แสดงหน้าต่างหลัก...")
            window.show()
            
            if use_splash:
                splash.set_progress(9, 10, "กำลังเตรียมพร้อม...")
                app.processEvents()
                
                splash.set_progress(10, 10, "พร้อมใช้งาน!")
                app.processEvents()
                
                # ปิด splash screen หลังจาก GUI แสดงแล้ว
                QTimer.singleShot(300, splash.close)
            
            print("✅ โปรแกรมพร้อมใช้งาน!")
            sys.exit(app.exec_())
            
        except Exception as e:
            if use_splash and splash:
                try:
                    splash.show_message(f"เกิดข้อผิดพลาด: {str(e)}")
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


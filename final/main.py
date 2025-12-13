# -*- coding: utf-8 -*-
"""
Main Entry Point - จุดเริ่มต้นของแอปพลิเคชัน
"""

import sys
import os

# IMPORTANT: Set CUDA environment variables BEFORE importing anything
# This must be done before any other imports that use CUDA
# Set CUDA environment variables (like run.sh does)
if 'CUDA_HOME' not in os.environ:
    os.environ['CUDA_HOME'] = '/usr/local/cuda-11.4'
if 'CUDA_PATH' not in os.environ:
    os.environ['CUDA_PATH'] = '/usr/local/cuda-11.4'

# Add CUDA paths to LD_LIBRARY_PATH
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

# Add OpenCV Python path to PYTHONPATH
opencv_python_path = '/usr/local/lib/python3.8/site-packages/'
current_python_path = os.environ.get('PYTHONPATH', '')
if opencv_python_path not in current_python_path:
    if current_python_path:
        os.environ['PYTHONPATH'] = opencv_python_path + ':' + current_python_path
    else:
        os.environ['PYTHONPATH'] = opencv_python_path

# Add CUDA bin to PATH
cuda_bin = '/usr/local/cuda-11.4/bin'
current_path = os.environ.get('PATH', '')
if cuda_bin not in current_path:
    os.environ['PATH'] = cuda_bin + ':' + current_path

# IMPORTANT: Setup CUDA paths BEFORE importing OpenCV or PyTorch
# This must be done before any other imports that use CUDA
from core.cuda_setup import setup_cuda_paths
setup_cuda_paths()

from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer


def main():
    """Main function with loading screen"""
    # สร้าง QApplication ก่อนเพื่อใช้ splash screen
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Import เบาๆ เพื่อแสดง splash screen
    from PyQt5.QtWidgets import QApplication
    
    # แสดง loading screen ก่อน import modules หนักๆ
    from gui.splash_screen import show_splash_screen
    splash = show_splash_screen(app)
    
    # อัพเดท loading progress
    splash.set_progress(1, 10, "กำลังเริ่มระบบ...")
    app.processEvents()  # ให้ GUI อัพเดท
    
    # Import modules ตามขั้นตอน
    try:
        splash.set_progress(2, 10, "กำลังโหลด Configuration...")
        app.processEvents()
        from config import settings
        
        splash.set_progress(3, 10, "กำลังโหลด Core Modules...")
        app.processEvents()
        from core import camera_manager
        from core import image_processor
        from core import business_logic
        
        splash.set_progress(4, 10, "กำลังโหลด GUI Modules...")
        app.processEvents()
        from gui.main_window import BottleDetectionGUI
        
        splash.set_progress(5, 10, "กำลังเตรียมหน้าต่างหลัก...")
        app.processEvents()
        window = BottleDetectionGUI()
        
        splash.set_progress(7, 10, "กำลังเริ่มต้นระบบ...")
        app.processEvents()
        
        splash.set_progress(8, 10, "กำลังแสดงหน้าต่างหลัก...")
        app.processEvents()
        window.show()
        
        splash.set_progress(9, 10, "กำลังเตรียมพร้อม...")
        app.processEvents()
        
        splash.set_progress(10, 10, "พร้อมใช้งาน!")
        app.processEvents()
        
        # ปิด splash screen หลังจาก GUI แสดงแล้ว
        QTimer.singleShot(300, splash.close)
        
        sys.exit(app.exec_())
        
    except Exception as e:
        splash.show_message(f"เกิดข้อผิดพลาด: {str(e)}")
        QTimer.singleShot(3000, splash.close)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


# -*- coding: utf-8 -*-
"""
Main Window - BottleDetectionGUI class
จัดการองค์ประกอบ (widgets), Layout และ Event Handlers สำหรับ UI
"""

# CRITICAL: Setup CUDA paths BEFORE importing any libs modules
try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    pass

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QWidget, QGroupBox, QTextEdit, QGridLayout, QListWidget, QProgressBar,
    QSplitter, QMessageBox, QCheckBox, QScrollArea, QTabWidget, QTabBar,
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QFrame, QApplication,
    QDateTimeEdit, QDialog, QToolBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime, QPropertyAnimation, QEasingCurve, QPoint, QSize
from PyQt5.QtGui import QImage, QPixmap, QWheelEvent, QIcon
import cv2
import numpy as np
import os
import time
from typing import List, Dict
import sys
import types

# Import CUDA image utilities
from core.cuda_image_utils import cuda_resize, cuda_cvtColor, cuda_gaussianBlur

# Import from our modules
from core.camera_manager import USBCamera, SentechCamera
from core.business_logic import (
    CapDetectionThread,
    BottleDetectionThread,
    ModbusThread
)
from core.image_processor import (
    perform_ocr_on_image,
    check_bottle_type,
    safe_show_img,
    fit_to_same_size,
    make_montage
)
from config.settings import (
    CAP_MODEL_PATH,
    BOTTLE_MODEL_PATH,
    ROTATION_MODEL_PATH,
    BEST_ACCURACY_MODEL_PATH,
    CRAFT_MODEL_PATH,
    CRAFT_REFINER_PATH,
    OCR_MODEL_PATH,
    FADED_TEXT_CONFIG,
    MODBUS_IP,
    MODBUS_PORT,
    CAP_DETECTION_AVAILABLE,
    YOLO_AVAILABLE
)

# Import YOLO for faded text detection
if YOLO_AVAILABLE:
    from ultralytics import YOLO

# Import bottle detection module
from libs.detection.bottledetect import (
    detect_bottle_and_crop_type, 
    process_bottle_image_simple,
    get_type_crops_only,
    get_detection_summary,
    draw_detections_on_image,
    save_cropped_type
)

# Import cap detection modules
if CAP_DETECTION_AVAILABLE:
    from libs.detection.capmodel import initialize_detector, detect_caps_from_path
    from libs.processing.rotationCRAFT import initialize_detector as initialize_craft_detector, detect_text_and_rotate_from_array
    from libs.processing.rotationmodel import (
        initialize_model, 
        get_rotated_image_from_array,
        process_craft_rotated_from_array,
        get_cropped_from_craft_rotated_array,
        process_craft_rotated_with_verification_from_array,
        get_cropped_with_verification_from_craft_rotated_array
    )
    from libs.processing.craft_line_detection import (
        initialize_detector as initialize_line_detector,
        detect_lines_from_rotation_result,
        get_cropped_lines_from_rotation_result
    )
    from libs.processing.deep_ocr import DeepOCRModel, initialize_ocr_model, recognize_text_from_craft_lines

# Import separated GUI components
from gui.login_dialog import LoginDialog
from gui.debug_stream import DebugStream
from gui.components.home_tab import create_home_tab
from gui.components.bottle_tab import create_bottle_tab
from gui.components.cap_tab import create_cap_tab
from gui.components.status_tab import create_status_tab
from gui.components.ocr_test_tab import create_ocr_test_tab
from gui.components.settings_tab import create_settings_tab
from gui.components.arean_tab import create_arean_tab
from gui.components.history_tab import (
    create_history_tab, create_history_item, create_history_compact_item,
    is_history_entry_good, is_history_entry_ng
)
from gui.components.robot_test_tab import create_robot_test_tab
from gui.components.control_panel import create_control_panel

# Import event handlers
from gui.handlers.bottle_handlers import BottleDetectionHandlers
from gui.handlers.cap_handlers import CapDetectionHandlers
from gui.handlers.robot_test_handlers import RobotTestHandlers
from gui.handlers.status_handlers import StatusHandlers
from gui.handlers.modbus_handlers import ModbusHandlers


class CollapsibleTabWidget(QWidget):
    """Custom collapsible tab widget that works like panel output - can collapse/expand each tab with vertical text"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.tabs = {}  # Store tab widgets and their states
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Left side: Tab buttons (vertical)
        tabs_panel = QWidget()
        tabs_panel.setFixedWidth(180)
        tabs_panel.setStyleSheet("background-color: #f5f5f5; border-right: 1px solid #bdc3c7;")
        tabs_layout = QVBoxLayout(tabs_panel)
        tabs_layout.setContentsMargins(5, 5, 5, 5)
        tabs_layout.setSpacing(3)
        
        # Container for tabs with scroll
        self.tabs_container = QWidget()
        self.tabs_layout = QVBoxLayout(self.tabs_container)
        self.tabs_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs_layout.setSpacing(3)
        self.tabs_layout.addStretch()
        
        # Scroll area for tabs
        scroll = QScrollArea()
        scroll.setWidget(self.tabs_container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f5f5f5;
            }
            QScrollBar:vertical {
                background-color: #e0e0e0;
                width: 10px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #bdc3c7;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #95a5a6;
            }
        """)
        
        tabs_layout.addWidget(scroll)
        self.main_layout.addWidget(tabs_panel)
        
        # Right side: Content area
        self.content_widget = QWidget()
        self.content_widget.setMinimumWidth(0)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.current_tab_index = None
        
        self.main_layout.addWidget(self.content_widget, 1)
    
    def addTab(self, widget, label):
        """Add a collapsible tab with vertical text"""
        index = len(self.tabs)
        
        # Create collapsible header button with vertical text
        header = QPushButton()
        header.setCheckable(True)
        header.setChecked(False)
        header.setMinimumWidth(130)
        header.setMinimumHeight(60)
        header.setMaximumHeight(60)
        
        # Store label for painting
        header._label_text = label
        
        # Custom paint event for vertical text
        def paint_button(self, event):
            from PyQt5.QtGui import QPainter, QFontMetrics, QPen, QBrush, QColor
            
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            rect = self.rect()
            
            # Background color
            if self.isChecked():
                bg_color = QColor("#3498db")
                text_color = QColor("white")
            else:
                bg_color = QColor("#ecf0f1")
                text_color = QColor("#2c3e50")
            
            # Draw background
            painter.fillRect(rect, QBrush(bg_color))
            
            # Draw border
            painter.setPen(QPen(QColor("#bdc3c7"), 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
            
            # Draw vertical text (rotated 90 degrees)
            painter.setPen(QPen(text_color))
            font_metrics = QFontMetrics(self.font())
            text = self._label_text
            
            # Rotate painter for vertical text
            painter.save()
            painter.translate(rect.width() / 2, rect.height() / 2)
            painter.rotate(0)
            
            # Center text
            text_width = font_metrics.width(text)
            text_height = font_metrics.height()
            x = -text_width / 2
            y = text_height / 2 - font_metrics.descent()
            
            painter.drawText(int(x), int(y), text)
            painter.restore()
        
        # Bind paint event to header button
        header.paintEvent = types.MethodType(paint_button, header)
        
        # Store tab info
        self.tabs[index] = {
            'widget': widget,
            'header': header,
            'label': label,
            'expanded': False
        }
        
        # Insert before stretch
        self.tabs_layout.insertWidget(self.tabs_layout.count() - 1, header)
        
        # Connect header click to toggle (use lambda with default parameter to capture index)
        def on_header_clicked(checked):
            self._toggleTab(index, checked)
        
        header.clicked.connect(on_header_clicked)
        
        return index
    
    def removeTab(self, index):
        """Remove a tab header and its content from the widget."""
        try:
            tab_info = self.tabs.get(index)
            if not tab_info:
                return
            
            header = tab_info.get('header')
            widget = tab_info.get('widget')
            
            # Remove header button from the tabs layout
            if header is not None:
                self.tabs_layout.removeWidget(header)
                header.setParent(None)
            
            # If this tab is currently expanded, clear the content area
            if self.current_tab_index == index:
                while self.content_layout.count():
                    item = self.content_layout.takeAt(0)
                    if item.widget():
                        w = item.widget()
                        self.content_layout.removeWidget(w)
                        w.setParent(None)
                        w.hide()
                self.current_tab_index = None
            
            # Detach tab widget from content (but don't delete it – caller may reuse)
            if widget is not None and widget.parent() is self.content_widget:
                widget.setParent(None)
                widget.hide()
            
            # Mark tab as not expanded
            tab_info['expanded'] = False
        except Exception as e:
            print(f"❌ Error in CollapsibleTabWidget.removeTab: {e}")
    
    def _toggleTab(self, index, expanded):
        """Toggle tab expansion"""
        try:
            tab_info = self.tabs[index]
            
            # Prevent infinite loop by checking if state is already correct
            if tab_info['expanded'] == expanded:
                return
            
            tab_info['expanded'] = expanded
            
            if expanded:
                # Collapse other tabs (block signals to prevent infinite loop)
                for idx, info in self.tabs.items():
                    if idx != index and info['expanded']:
                        info['header'].blockSignals(True)
                        info['header'].setChecked(False)
                        info['header'].blockSignals(False)
                        info['expanded'] = False
                        # Hide other tab widgets
                        if info['widget'].parent() == self.content_widget:
                            info['widget'].setParent(None)
                            info['widget'].hide()
                
                # Show this tab's content
                self.current_tab_index = index
                
                # Remove current widget from layout if exists
                while self.content_layout.count():
                    item = self.content_layout.takeAt(0)
                    if item.widget():
                        widget = item.widget()
                        self.content_layout.removeWidget(widget)
                        widget.setParent(None)
                        widget.hide()
                
                # Check if widget still exists and is valid before adding
                widget = tab_info['widget']
                if widget is None:
                    print(f"❌ Widget for tab {index} is None")
                    return
                
                # Check if widget is still valid (not deleted)
                try:
                    # Try to access widget properties to check if it's still valid
                    _ = widget.parent()
                except RuntimeError:
                    print(f"❌ Widget for tab {index} has been deleted, cannot add to layout")
                    return
                
                # Remove widget from its current parent if it has one
                if widget.parent():
                    old_parent = widget.parent()
                    if old_parent == self.content_widget:
                        # Already in content widget, just show it
                        widget.show()
                        tab_info['header'].update()
                        return
                    else:
                        # Remove from old parent
                        old_parent_layout = old_parent.layout()
                        if old_parent_layout:
                            old_parent_layout.removeWidget(widget)
                        widget.setParent(None)
                
                # Add tab widget to content (don't delete, just show/hide)
                widget.setParent(self.content_widget)
                self.content_layout.addWidget(widget)
                widget.show()
                tab_info['header'].update()  # Refresh button appearance
            else:
                # Hide content if this tab was expanded
                if self.current_tab_index == index:
                    while self.content_layout.count():
                        item = self.content_layout.takeAt(0)
                        if item.widget():
                            widget = item.widget()
                            self.content_layout.removeWidget(widget)
                            widget.setParent(None)
                            widget.hide()
                    self.current_tab_index = None
                
                tab_info['header'].update()  # Refresh button appearance
        except KeyError as e:
            print(f"❌ KeyError in _toggleTab: Tab index {index} not found - {e}")
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e):
                print(f"❌ RuntimeError in _toggleTab: Widget has been deleted - {e}")
                # Try to recover by removing the invalid widget reference
                if index in self.tabs:
                    print(f"⚠️ Removing invalid widget reference for tab {index}")
                    # Don't delete the tab, just mark widget as None
                    self.tabs[index]['widget'] = None
            else:
                raise
        except Exception as e:
            print(f"❌ Error in _toggleTab: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"❌ Error in _toggleTab: {e}")
            import traceback
            traceback.print_exc()
    
    def tabBar(self):
        """Return a dummy tab bar for compatibility"""
        return self
    
    def setTabPosition(self, position):
        """Dummy method for compatibility"""
        pass


class ImageZoomDialog(QDialog):
    """Popup window for zooming images"""
    def __init__(self, image, title="Image Viewer", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(800, 600)
        
        # Store original image
        if isinstance(image, np.ndarray):
            # Convert numpy array to QPixmap
            if len(image.shape) == 2:  # Grayscale
                rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:  # BGR
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_img.shape
            bytes_per_line = c * w
            qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qimg)
        else:
            self.original_pixmap = image
        
        # Zoom factor
        self.zoom_factor = 1.0
        
        # Setup UI
        layout = QVBoxLayout(self)
        
        # Image label with scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.update_image()
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
        
        # Info label
        info_label = QLabel("Double-click to reset zoom | Mouse wheel to zoom")
        info_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # Enable mouse tracking for zoom
        self.image_label.setMouseTracking(True)
        self.scroll_area.wheelEvent = self.wheel_event
    
    def wheel_event(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.1  # Zoom in
        else:
            self.zoom_factor *= 0.9  # Zoom out
        
        # Limit zoom range
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        self.update_image()
        event.accept()
    
    def mouseDoubleClickEvent(self, event):
        """Reset zoom on double-click"""
        self.zoom_factor = 1.0
        self.update_image()
        event.accept()
    
    def update_image(self):
        """Update displayed image with current zoom factor"""
        if self.zoom_factor == 1.0:
            scaled_pixmap = self.original_pixmap
        else:
            new_width = int(self.original_pixmap.width() * self.zoom_factor)
            new_height = int(self.original_pixmap.height() * self.zoom_factor)
            scaled_pixmap = self.original_pixmap.scaled(
                new_width, new_height, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())


class InitWorker(QThread):
    """โหลดกล้องและโมเดลในพื้นหลัง ไม่บล็อก GUI"""
    camera_ready = pyqtSignal(object, bool)
    sentech_ready = pyqtSignal(object, bool)
    models_ready = pyqtSignal(object)  # dict or None
    progress_msg = pyqtSignal(str)

    def run(self):
        # USB camera
        try:
            self.progress_msg.emit("กำลังโหลดกล้อง USB...")
            from core.camera_manager import USBCamera
            cam = USBCamera()
            ok = cam.open_camera()
            self.camera_ready.emit(cam, ok)
        except Exception as e:
            print(f"❌ InitWorker USB camera: {e}")
            self.camera_ready.emit(None, False)
        # Sentech camera
        try:
            self.progress_msg.emit("กำลังโหลดกล้อง Sentech...")
            from core.camera_manager import SentechCamera
            sc = SentechCamera()
            ok = sc.initialize()
            self.sentech_ready.emit(sc, ok)
        except Exception as e:
            print(f"❌ InitWorker Sentech: {e}")
            self.sentech_ready.emit(None, False)
        # Cap detection models - โหลดทีละโมเดล และถ้าโหลดไม่ได้ก็ใส่ None แทน
        d = {}
        try:
            self.progress_msg.emit("กำลังโหลดโมเดลฝา/OCR...")
            from config.settings import CAP_DETECTION_AVAILABLE, CAP_MODEL_PATH, CRAFT_MODEL_PATH, CRAFT_REFINER_PATH, ROTATION_MODEL_PATH, OCR_MODEL_PATH, YOLO_AVAILABLE
            
            if not CAP_DETECTION_AVAILABLE:
                print("⚠️ Cap detection modules not available")
                self.models_ready.emit({})
                return
            
            # โหลด cap_detector (จำเป็น)
            try:
                self.progress_msg.emit("กำลังโหลดโมเดลตรวจจับฝา...")
                from libs.detection.capmodel import initialize_detector
                d['cap_detector'] = initialize_detector(CAP_MODEL_PATH)
                print("✅ Cap detector loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load cap_detector: {e}")
                d['cap_detector'] = None
            
            # โหลด craft_detector (ไม่จำเป็น - ถ้าไม่มีก็ข้าม)
            try:
                self.progress_msg.emit("กำลังโหลดโมเดล CRAFT...")
                from libs.processing.rotationCRAFT import initialize_detector as initialize_craft_detector
                d['craft_detector'] = initialize_craft_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
                print("✅ CRAFT detector loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load craft_detector: {e}")
                d['craft_detector'] = None
            
            # โหลด rotation_model (ไม่จำเป็น - ถ้าไม่มีก็ข้าม)
            try:
                self.progress_msg.emit("กำลังโหลดโมเดลหมุนภาพ...")
                from libs.processing.rotationmodel import initialize_model
                d['rotation_model'] = initialize_model(ROTATION_MODEL_PATH)
                # พยายาม init craft detector ใน rotation model (ถ้ามี)
                if d.get('rotation_model') and d.get('craft_detector'):
                    try:
                        d['rotation_model'].init_craft_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
                    except Exception as e:
                        print(f"⚠️ Failed to init CRAFT in rotation_model: {e}")
                print("✅ Rotation model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load rotation_model: {e}")
                d['rotation_model'] = None
            
            # โหลด line_detector (ไม่จำเป็น - ถ้าไม่มีก็ข้าม)
            try:
                self.progress_msg.emit("กำลังโหลดโมเดลตรวจจับบรรทัด...")
                from libs.processing.craft_line_detection import initialize_detector as initialize_line_detector
                d['line_detector'] = initialize_line_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
                print("✅ Line detector loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load line_detector: {e}")
                d['line_detector'] = None
            
            # โหลด ocr_model - ต้องมีไฟล์ ocr.pth ใน models/ocr/
            d['ocr_model'] = None
            if os.path.exists(OCR_MODEL_PATH):
                try:
                    self.progress_msg.emit("กำลังโหลดโมเดล OCR...")
                    from libs.processing.deep_ocr import initialize_ocr_model
                    d['ocr_model'] = initialize_ocr_model(OCR_MODEL_PATH)
                    print("✅ OCR model loaded successfully")
                except Exception as e:
                    print(f"⚠️ Failed to load ocr_model: {e}")
                    d['ocr_model'] = None
            else:
                print(f"⚠️ OCR model ไม่โหลด: ไม่พบไฟล์")
                print(f"   วางไฟล์ ocr.pth ที่: {os.path.abspath(OCR_MODEL_PATH)}")
            
            # ฝาจางใช้ cap_fade_model.h5 แทน (ไม่ใช้ YOLO สำหรับ fade แล้ว)
            d['faded_text_yolo_model'] = None
            
            # ส่งผลลัพธ์ (แม้บางโมเดลจะโหลดไม่ได้ก็ยังส่ง dict ที่มีโมเดลที่โหลดได้)
            print(f"✅ Model loading complete - cap_detector: {'✅' if d.get('cap_detector') else '❌'}, "
                  f"craft_detector: {'✅' if d.get('craft_detector') else '❌'}, "
                  f"rotation_model: {'✅' if d.get('rotation_model') else '❌'}, "
                  f"line_detector: {'✅' if d.get('line_detector') else '❌'}, "
                  f"ocr_model: {'✅' if d.get('ocr_model') else '❌'}")
            self.models_ready.emit(d)
            
        except Exception as e:
            print(f"❌ InitWorker models error: {e}")
            import traceback
            traceback.print_exc()
            # ส่ง dict ที่มีโมเดลที่โหลดได้ไป (แม้จะโหลดไม่ได้เลยก็ส่ง dict ว่าง)
            self.models_ready.emit(d if d else {})
        # Defect model (Good/NG ขวด) — โหลดในหน้า loading เพื่อพร้อมก่อนแสดง GUI
        try:
            self.progress_msg.emit("กำลังโหลด Defect model (ขวด Good/NG)...")
            from libs.detection.defect_model import _load_defect_model
            if _load_defect_model() is not None:
                print("✅ Defect model โหลดในหน้า loading แล้ว (พร้อมก่อนประมวลผล)")
            else:
                print("⚠️ Defect model โหลดไม่ได้ (จะข้ามการตรวจ defect)")
        except Exception as e:
            print(f"⚠️ InitWorker Defect model ข้าม: {e}")

        # Cap fade model (ฝาจาง) — โหลดในหน้า loading เพื่อพร้อมก่อนประมวลผลฝา (ไม่ต้องโหลดตอนรัน)
        try:
            self.progress_msg.emit("กำลังโหลด Cap fade model (ฝาจาง)...")
            from libs.detection.cap_fade_model import _load_cap_fade_model
            if _load_cap_fade_model() is not None:
                print("✅ Cap fade model โหลดในหน้า loading แล้ว (พร้อมก่อนประมวลผล)")
            else:
                print("⚠️ Cap fade model โหลดไม่ได้ (จะข้ามการตรวจฝาจาง)")
        except Exception as e:
            print(f"⚠️ InitWorker Cap fade model ข้าม: {e}")


class BottleDetectionGUI(QWidget):
    """สัญญาณเมื่อโหลดระบบในพื้นหลังเสร็จ (สำหรับแสดง loading ก่อนแล้วค่อยแสดง GUI)"""
    init_complete = pyqtSignal()
    init_progress = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Bottle Detection & OCR - USB Camera + Modbus + Sentech Camera')
        
        # ตรวจสอบว่าเป็น Jetson หรือไม่ (ก่อนตั้งขนาดหน้าต่าง)
        self.is_jetson = self.check_if_jetson()
        # ปรับขนาดให้ไม่เกินหน้าจอ (ลดโอกาสเกิดแถบเลื่อนแนวนอนที่ระดับหน้าต่าง)
        self.resize(1800, 1200)
        if not self.is_jetson:
            screen = QApplication.primaryScreen()
            if screen is not None:
                g = screen.availableGeometry()
                self.resize(min(1800, max(960, g.width() - 40)), min(1200, max(700, g.height() - 80)))
        if self.is_jetson:
            print("🚀 Detected Jetson - Enabling fullscreen support")
            self.setWindowState(Qt.WindowFullScreen)
        
        self.current_image = None
        self.current_result = None
        self.batch_results = []
        self.usb_camera = None
        self.sentech_camera = None
        self.modbus_thread = None
        
        # Cap detection models
        self.cap_detector = None
        self.craft_detector = None
        self.rotation_model = None
        self.line_detector = None
        self.ocr_model = None
        
        # Cap detection results
        self.current_sentech_image = None
        self.current_cap_result = None
        self.current_bottle_type = None  # เก็บประเภทขวดที่ตรวจพบ
        self.last_bottle_result = None  # เก็บผลลัพธ์การตรวจจับขวดล่าสุด
        
        # Initialize taste mode variables
        self.current_taste_mode = 3  # Default to 3-taste mode
        self.selected_tastes = ["M100", "M110", "M120"]  # Internal codes; UI shows flavor names
        
        # Initialize expiry filtering mode variables
        self.current_expiry_mode = "normal"  # "normal" or "filter"
        self.expiry_filter_type = "range"  # "range" or "specific"
        self.expiry_date_type = "BBF"  # "BBF" or "MFG" - ใช้วันที่จากฝาตามที่ user เลือก
        self._last_cap_fail_reason = None  # "ไม่ตรงกับวันที่เลือก" เมื่อไม่ผ่านคัดกรองวัน, None เมื่อฝาไม่ผ่านเหตุอื่น
        self.expiry_start_date = None
        self.expiry_end_date = None
        self.expiry_specific_date = None
        
        # Initialize OCR test variables
        self.ocr_test_image = None
        self.ocr_test_image_path = None
        self.current_rotation_angle = 0  # Track current rotation angle
        self.original_cap_image = None  # Store original image for rotation reference
        # Manual rotation removed
        
        # Statistics for status tab
        self.total_images_processed_count = 0
        self.successful_detections_count = 0
        self.error_count_value = 0
        self.current_processing_mode = "Manual"
        
        # Debug panel state
        self.debug_panel_visible = False
        self.debug_logs = []  # Store debug logs
        
        # Login state
        self.is_admin_logged_in = False
        self.settings_tab_index = -1  # Store Settings tab index
        self.settings_tab_widget = None  # Store Settings tab widget
        self.settings_tab_label = "⚙️ Settings"  # Store Settings tab label
        # Admin-only tabs (hidden until login)
        self.modbus_status_tab_index = -1
        self.modbus_status_tab_widget = None
        self.modbus_status_tab_label = "🔌 Modbus Status"
        self.ocr_test_tab_index = -1
        self.ocr_test_tab_widget = None
        self.ocr_test_tab_label = "🔤 OCR Test"
        self.robot_test_tab_index = -1
        self.robot_test_tab_widget = None
        self.robot_test_tab_label = "🔧 Register D Test"
        
        # History storage (จำกัดจำนวน ป้องกัน memory เติบโตเมื่อรัน full auto นาน)
        self.processing_history = []  # List to store processing history (เก็บ thumbnail + ข้อมูลสรุป)
        # จำกัดสูงสุด 200 รายการในหน่วยความจำ
        self.processing_history_max = 200
        # แต่ใน UI จะแสดงแบบเต็ม (มีรูป/รายละเอียดครบ) สำหรับรายการล่าสุดเท่านั้น
        # รายการเก่ากว่านี้จะเป็นโหมดย่อ (timestamp + สถานะ + ปุ่มดูรายละเอียด)
        self.max_full_history_widgets = 60
        
        # Capture only mode folders (ID 5)
        self.capture_only_usb_folder = None
        self.capture_only_sentech_folder = None
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Initialize event handlers first (before setting up timers and connections)
        self.bottle_handlers = BottleDetectionHandlers(self)
        self.cap_handlers = CapDetectionHandlers(self)
        self.status_handlers = StatusHandlers(self)
        self.modbus_handlers = ModbusHandlers(self)
        self.robot_test_handlers = RobotTestHandlers(self)
        
        # Status update timer (after handlers are created)
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.status_handlers.update_all_status)
        self.status_timer.start(1000)  # อัพเดททุก 1 วินาที
        
        self.setup_ui()
        self.setup_print_capture()  # Setup print capture after UI is ready
        
        # โหลดกล้อง/โมเดล/Modbus ในพื้นหลังหลังหน้าต่างแสดง — ไม่ให้เปิดแล้วค้าง
        if hasattr(self, 'status_label'):
            self.status_label.setText('⏳ Loading... (you can click)')
        self._init_worker = None
        QTimer.singleShot(80, self._start_background_init)
        
        # เพิ่ม keyboard shortcuts สำหรับ Jetson
        if self.is_jetson:
            self.setup_jetson_shortcuts()
        
    def setup_ui(self):
        # สร้าง main scroll area สำหรับหน้าต่างหลัก
        main_scroll_area = QScrollArea()
        main_scroll_area.setWidgetResizable(True)
        main_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f0f0f0;
            }
            QScrollBar:vertical {
                background: #e0e0e0;
                width: 16px;
                border-radius: 8px;
            }
            QScrollBar::handle:vertical {
                background: #b0b0b0;
                border-radius: 8px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #95a5a6;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #e0e0e0;
                height: 16px;
                border-radius: 8px;
            }
            QScrollBar::handle:horizontal {
                background: #b0b0b0;
                border-radius: 8px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #95a5a6;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        # สร้าง main widget ที่จะใส่ใน scroll area
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #f0f0f0;")
        main_widget.setMinimumWidth(0)
        main_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout = QVBoxLayout(main_widget)
        
        # Title - อยู่บนสุดสุด
        title_label = QLabel("Bottle Detection & OCR - USB Camera + Modbus + Sentech Camera")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; color: #333333; background-color: #e5e5e5; border-bottom: 2px solid #cccccc;")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        
        # Top Panel - Fixed at top, visible on all tabs
        top_panel = QWidget()
        top_panel.setFixedHeight(80)  # Fixed height to prevent movement
        top_panel.setMinimumWidth(0)
        top_panel.setStyleSheet("background-color: #e0e0e0; border-bottom: 0px solid #cccccc;")
        top_panel_layout = QHBoxLayout(top_panel)
        top_panel_layout.setContentsMargins(10, 5, 10, 5)
        top_panel_layout.setSpacing(10)
        top_panel_layout.setAlignment(Qt.AlignVCenter)  # Align items to vertical center
        
        # Status Robot - แสดงสถานะ Running / Close / Break Point / Pause / Pre-run
        robot_status_label = QLabel('🔍 Waiting for D5002')
        robot_status_label.setStyleSheet("""
            QLabel {
                padding: 6px 12px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 5px;
                background-color: #ecf0f1;
                border: 2px solid #cccccc;
                color: #7f8c8d;
            }
        """)
        robot_status_label.setMinimumWidth(120)
        top_panel_layout.addWidget(robot_status_label)
        self.top_panel_robot_status = robot_status_label
        
        top_panel_layout.addSpacing(15)
        
        # Start button
        self.btn_start = QPushButton('▶️ START (M700)')
        self.btn_start.setStyleSheet("QPushButton { padding: 8px 15px; font-size: 12px; background-color: #27ae60; color: white; border-radius: 5px; font-weight: bold; } QPushButton:hover { background-color: #229954; } QPushButton:pressed { background-color: #1e8449; }")
        self.btn_start.pressed.connect(self.on_start_pressed)
        self.btn_start.released.connect(self.on_start_released)
        top_panel_layout.addWidget(self.btn_start)
        
        # Stop button — momentary เหมือน Start: กดค้าง = ON ปล่อย = OFF
        self.btn_stop = QPushButton('⏹️ STOP (M701)')
        self.btn_stop.setStyleSheet("QPushButton { padding: 8px 15px; font-size: 12px; background-color: #e74c3c; color: white; border-radius: 5px; font-weight: bold; } QPushButton:hover { background-color: #c0392b; } QPushButton:pressed { background-color: #a93226; }")
        self.btn_stop.pressed.connect(self.on_stop_pressed)
        self.btn_stop.released.connect(self.on_stop_released)
        top_panel_layout.addWidget(self.btn_stop)
        
        # Reset button
        self.btn_reset = QPushButton('🔄 RESET (D5012)')
        self.btn_reset.pressed.connect(self.on_reset_pressed)
        self.btn_reset.released.connect(self.on_reset_released)
        self.btn_reset.setStyleSheet("QPushButton { padding: 8px 15px; font-size: 12px; background-color: #f39c12; color: white; border-radius: 5px; font-weight: bold; } QPushButton:hover { background-color: #e67e22; }")
        top_panel_layout.addWidget(self.btn_reset)
        
        # Toggle ไฟ (M76) - เปิด/ปิด ไฟในหน้าหลัก
        self.btn_light_m76 = QPushButton('💡 Light (M76) Off')
        self.btn_light_m76.setCheckable(True)
        self.btn_light_m76.setChecked(False)
        self.btn_light_m76.clicked.connect(self.on_toggle_light_m76)
        self.btn_light_m76.setStyleSheet("""
            QPushButton {
                padding: 8px 15px; font-size: 12px; background-color: #7f8c8d; color: white;
                border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #95a5a6; }
            QPushButton:checked {
                background-color: #f1c40f; color: #2c3e50;
            }
            QPushButton:checked:hover { background-color: #f39c12; }
        """)
        top_panel_layout.addWidget(self.btn_light_m76)
        
        top_panel_layout.addSpacing(20)
        
        # Mode selection dropdown
        mode_label = QLabel('Program:')
        mode_label.setStyleSheet("color: #333333; padding: 5px; font-weight: bold; font-size: 12px;")
        top_panel_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("ID 7 full auto")
        self.mode_combo.addItem("ID 5 capture only")
        self.mode_combo.addItem("ID 8 reset modbus")
        self.mode_combo.setMinimumWidth(110)
        self.mode_combo.setMaximumWidth(200)
        self.mode_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                font-size: 11px;
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                color: #2c3e50;
            }
            QComboBox:hover {
                background-color: #d5dbdb;
                border: 2px solid #95a5a6;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #34495e;
                margin-right: 5px;
            }
        """)
        self.mode_combo.currentIndexChanged.connect(self.modbus_handlers.on_mode_changed)
        
        # Capture limit input (for ID 5 mode)
        self.capture_limit_label = QLabel("Count:")
        self.capture_limit_label.setStyleSheet("color: #333333; font-size: 11px;")
        self.capture_limit_spin = QSpinBox()
        self.capture_limit_spin.setMinimum(1)
        self.capture_limit_spin.setMaximum(1000)
        self.capture_limit_spin.setValue(10)
        self.capture_limit_spin.setFixedWidth(80)
        self.capture_limit_spin.setStyleSheet("""
            QSpinBox {
                padding: 5px;
                font-size: 11px;
                background-color: #ecf0f1;
                color: #2c3e50;
                border: 1px solid #7f8c8d;
                border-radius: 3px;
            }
        """)
        self.capture_limit_spin.valueChanged.connect(self.on_capture_limit_changed)
        self.capture_limit_label.setVisible(False)
        self.capture_limit_spin.setVisible(False)
        
        # Set default to "ID 7 full auto"
        self.mode_combo.setCurrentIndex(0)
        top_panel_layout.addWidget(self.mode_combo)
        top_panel_layout.addWidget(self.capture_limit_label)
        top_panel_layout.addWidget(self.capture_limit_spin)
        
        top_panel_layout.addSpacing(15)
        
        # Status indicators (Camera USB, Sentech, Modbus)
        # USB Camera Status
        usb_camera_status_widget = QWidget()
        usb_camera_status_layout = QHBoxLayout(usb_camera_status_widget)
        usb_camera_status_layout.setContentsMargins(5, 0, 5, 0)
        usb_camera_status_layout.setSpacing(5)
        
        self.usb_camera_status_indicator = QLabel('●')
        self.usb_camera_status_indicator.setFixedWidth(15)
        self.usb_camera_status_indicator.setStyleSheet("color: #f39c12; font-size: 16px; font-weight: bold;")
        usb_camera_status_layout.addWidget(self.usb_camera_status_indicator)
        
        usb_camera_status_text = QLabel('USB Camera')
        usb_camera_status_text.setStyleSheet("color: #333333; font-size: 11px;")
        usb_camera_status_layout.addWidget(usb_camera_status_text)
        
        top_panel_layout.addWidget(usb_camera_status_widget)
        
        # Sentech Camera Status
        sentech_camera_status_widget = QWidget()
        sentech_camera_status_layout = QHBoxLayout(sentech_camera_status_widget)
        sentech_camera_status_layout.setContentsMargins(5, 0, 5, 0)
        sentech_camera_status_layout.setSpacing(5)
        
        self.sentech_camera_status_indicator = QLabel('●')
        self.sentech_camera_status_indicator.setFixedWidth(15)
        self.sentech_camera_status_indicator.setStyleSheet("color: #f39c12; font-size: 16px; font-weight: bold;")
        sentech_camera_status_layout.addWidget(self.sentech_camera_status_indicator)
        
        sentech_camera_status_text = QLabel('Sentech')
        sentech_camera_status_text.setStyleSheet("color: #333333; font-size: 11px;")
        sentech_camera_status_layout.addWidget(sentech_camera_status_text)
        
        top_panel_layout.addWidget(sentech_camera_status_widget)
        
        # Modbus Status
        modbus_status_widget = QWidget()
        modbus_status_layout = QHBoxLayout(modbus_status_widget)
        modbus_status_layout.setContentsMargins(5, 0, 5, 0)
        modbus_status_layout.setSpacing(5)
        
        self.modbus_status_indicator = QLabel('●')
        self.modbus_status_indicator.setFixedWidth(15)
        self.modbus_status_indicator.setStyleSheet("color: #f39c12; font-size: 16px; font-weight: bold;")
        modbus_status_layout.addWidget(self.modbus_status_indicator)
        
        modbus_status_text = QLabel('Modbus')
        modbus_status_text.setStyleSheet("color: #333333; font-size: 11px;")
        modbus_status_layout.addWidget(modbus_status_text)
        
        top_panel_layout.addWidget(modbus_status_widget)
        
        top_panel_layout.addSpacing(15)
        
        # Taste Mode selection dropdown (simple, no selection checkboxes)
        taste_mode_label = QLabel('Taste:')
        taste_mode_label.setStyleSheet("color: #333333; padding: 5px; font-weight: bold; font-size: 12px;")
        top_panel_layout.addWidget(taste_mode_label)
        
        self.taste_mode_combo = QComboBox()
        self.taste_mode_combo.addItem("1 Taste")
        self.taste_mode_combo.addItem("2 Tastes")
        self.taste_mode_combo.addItem("3 Tastes")
        self.taste_mode_combo.setCurrentIndex(2)  # Default to 3-taste mode
        self.taste_mode_combo.setMinimumWidth(95)
        self.taste_mode_combo.setMaximumWidth(160)
        self.taste_mode_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                font-size: 11px;
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                color: #2c3e50;
            }
            QComboBox:hover {
                background-color: #d5dbdb;
                border: 2px solid #95a5a6;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #34495e;
                margin-right: 5px;
            }
        """)
        self.taste_mode_combo.currentIndexChanged.connect(self.on_taste_mode_combo_changed)
        top_panel_layout.addWidget(self.taste_mode_combo)
        
        top_panel_layout.addSpacing(15)
        
        # Expiry Mode selection dropdown (simple, no filter type selection)
        expiry_mode_label = QLabel('Expiry:')
        expiry_mode_label.setStyleSheet("color: #333333; padding: 5px; font-weight: bold; font-size: 12px;")
        top_panel_layout.addWidget(expiry_mode_label)
        
        self.expiry_mode_combo = QComboBox()
        self.expiry_mode_combo.addItem("Normal mode")
        self.expiry_mode_combo.addItem("Filter by date")
        self.expiry_mode_combo.setCurrentIndex(0)  # Default to normal mode
        self.expiry_mode_combo.setMinimumWidth(100)
        self.expiry_mode_combo.setMaximumWidth(170)
        self.expiry_mode_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                font-size: 11px;
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                color: #2c3e50;
            }
            QComboBox:hover {
                background-color: #d5dbdb;
                border: 2px solid #95a5a6;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #34495e;
                margin-right: 5px;
            }
        """)
        self.expiry_mode_combo.currentIndexChanged.connect(self.on_expiry_mode_combo_changed)
        top_panel_layout.addWidget(self.expiry_mode_combo)
        
        # Add stretch before buttons to center them
        top_panel_layout.addStretch()
        
        # Login button - ปุ่มวงกลมใช้รูปจาก login.png
        self.login_button = QPushButton()
        _login_icon_path = os.path.join(os.path.dirname(__file__), "gui icon", "login.png")
        if os.path.exists(_login_icon_path):
            self.login_button.setIcon(QIcon(_login_icon_path))
            self.login_button.setIconSize(QSize(28, 28))
        else:
            self.login_button.setText('🔐')
        self.login_button.setFixedSize(48, 48)
        self.login_button.setToolTip('Login')
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                border: none;
                border-radius: 24px;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.login_button.clicked.connect(self.on_login_button_clicked)
        top_panel_layout.addWidget(self.login_button)
        
        # Debug button
        self.debug_toggle_header_btn = QPushButton()
        _debug_icon_path = os.path.join(os.path.dirname(__file__), "gui icon", "debug.png")
        if os.path.exists(_debug_icon_path):
            self.debug_toggle_header_btn.setIcon(QIcon(_debug_icon_path))
            self.debug_toggle_header_btn.setIconSize(QSize(28, 28))
        else:
            self.debug_toggle_header_btn.setText('🐛')
        self.debug_toggle_header_btn.setFixedSize(48, 48)
        self.debug_toggle_header_btn.setToolTip('Debug')
        self.debug_toggle_header_btn.clicked.connect(self.toggle_debug_panel)
        self.debug_toggle_header_btn.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                border: none;
                border-radius: 24px;
                padding: 0;
            }
            QPushButton:hover { background-color: #95a5a6; }
            QPushButton:pressed { background-color: #6c7a7d; }
        """)
        top_panel_layout.addWidget(self.debug_toggle_header_btn)
        
        # Add stretch after buttons to center them
        top_panel_layout.addStretch()
        
        layout.addWidget(top_panel)
        
        # Robot alarm banner (มองเห็นทุกแท็บ - แสดงเมื่อ D5001 มีค่า error จากหุ่นยนต์)
        self.robot_alarm_banner = QWidget()
        self.robot_alarm_banner.setStyleSheet("""
            QWidget { background-color: #c0392b; border-bottom: 2px solid #922b21; padding: 6px; }
            QLabel { color: white; font-size: 14px; font-weight: bold; }
        """)
        robot_alarm_layout = QHBoxLayout(self.robot_alarm_banner)
        robot_alarm_layout.setContentsMargins(15, 4, 15, 4)
        self.robot_alarm_label = QLabel("Robot Alarm")
        self.robot_alarm_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        # ให้ label ขยายเต็มความกว้างแถบ และจัดตัวอักษรชิดซ้าย
        self.robot_alarm_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.robot_alarm_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        robot_alarm_layout.addWidget(self.robot_alarm_label)
        self.robot_alarm_banner.setVisible(False)
        layout.addWidget(self.robot_alarm_banner)
        
        # ตัวอักษรวิ่งสำหรับ Robot Alarm
        # ทำข้อความยาวขึ้นเพื่อให้วิ่งเต็มแถบ
        self._robot_alarm_base_text = "   Robot Alarm   " * 10
        self._robot_alarm_marquee_index = 0
        self.robot_alarm_timer = QTimer(self)
        self.robot_alarm_timer.timeout.connect(self._update_robot_alarm_marquee)
        
        # Global loading banner (มองเห็นทุกแท็บ - แสดงเมื่อประมวลผลขวดหรือฝา)
        self._global_loading_count = 0
        self._stopping_processing = False
        self.global_loading_banner = QWidget()
        self.global_loading_banner.setStyleSheet("""
            QWidget { background-color: #c0c0c0; border-bottom: 0px solid #a8a8a8; padding: 8px; }
            QLabel { color: #333333; font-size: 14px; font-weight: bold; }
        """)
        global_loading_layout = QHBoxLayout(self.global_loading_banner)
        global_loading_layout.setContentsMargins(15, 8, 15, 8)
        self.global_loading_label = QLabel("")
        self.global_loading_label.setStyleSheet("color: #333333; font-size: 14px; font-weight: bold;")
        global_loading_layout.addWidget(self.global_loading_label)
        self.global_loading_spinner = QProgressBar()
        self.global_loading_spinner.setMinimum(0)
        self.global_loading_spinner.setMaximum(0)  # indeterminate
        self.global_loading_spinner.setFixedHeight(12)
        self.global_loading_spinner.setStyleSheet("QProgressBar { border: 1px solid #a0a0a0; border-radius: 6px; background: #d8d8d8; } QProgressBar::chunk { background: #888888; border-radius: 5px; }")
        global_loading_layout.addWidget(self.global_loading_spinner, 1)
        self.global_loading_banner.setVisible(False)
        layout.addWidget(self.global_loading_banner)
        
        # Create collapsible tab widget (like panel output) - moved up before status controls
        self.tab_widget = CollapsibleTabWidget()
        
        # Style the collapsible tab widget
        self.tab_widget.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
        """)
        
        # Create Control Panel tab first (for status and controls) - using new modern design
        control_panel_tab, control_panel_widgets = create_control_panel()
        
        # Map widgets from control panel to self for event handlers
        # Status indicators
        self.camera_status_indicator = control_panel_widgets.get('camera_status_indicator')
        self.sentech_status_indicator = control_panel_widgets.get('sentech_status_indicator')
        self.modbus_status_indicator = control_panel_widgets.get('modbus_status_indicator')
        self.queue_status_indicator = control_panel_widgets.get('queue_status_indicator')
        self.d5002_status_indicator = control_panel_widgets.get('d5002_status_indicator')
        self.d5001_status_indicator = control_panel_widgets.get('d5001_status_indicator')
        
        # Control buttons
        self.btn_capture = control_panel_widgets.get('btn_capture')
        self.btn_capture.clicked.connect(self.bottle_handlers.capture_both_cameras)
        self.btn_capture_both = control_panel_widgets.get('btn_capture_both')
        if self.btn_capture_both:
            self.btn_capture_both.clicked.connect(self.bottle_handlers.capture_both_cameras)
        
        self.btn_save_image = control_panel_widgets.get('btn_save_image')
        self.btn_save_image.clicked.connect(self.bottle_handlers.save_captured_image)
        
        self.btn_select_image = control_panel_widgets.get('btn_select_image')
        self.btn_select_image.clicked.connect(self.bottle_handlers.select_image_file)
        
        self.btn_process = control_panel_widgets.get('btn_process')
        self.btn_process.clicked.connect(self.bottle_handlers.process_current_image)
        
        self.btn_stop_processing = control_panel_widgets.get('btn_stop_processing')
        self.btn_stop_processing.clicked.connect(self.stop_processing)
        
        # ถ่ายและบันทึกทั้งคู่
        self.btn_capture_and_save = control_panel_widgets.get('btn_capture_and_save')
        if self.btn_capture_and_save:
            self.btn_capture_and_save.clicked.connect(self.bottle_handlers.capture_both_and_save)
        
        # Paired processing buttons
        self.btn_select_bottle_folder = control_panel_widgets.get('btn_select_bottle_folder')
        if self.btn_select_bottle_folder:
            self.btn_select_bottle_folder.clicked.connect(self.bottle_handlers.select_bottle_folder)
        self.btn_select_cap_folder = control_panel_widgets.get('btn_select_cap_folder')
        if self.btn_select_cap_folder:
            self.btn_select_cap_folder.clicked.connect(self.bottle_handlers.select_cap_folder)
        self.btn_process_paired = control_panel_widgets.get('btn_process_paired')
        if self.btn_process_paired:
            self.btn_process_paired.clicked.connect(self.bottle_handlers.process_paired_images)
        
        # Taste mode widgets
        self.taste_mode_1 = control_panel_widgets.get('taste_mode_1')
        self.taste_mode_1.toggled.connect(self.on_taste_mode_checkbox_changed)
        
        self.taste_mode_2 = control_panel_widgets.get('taste_mode_2')
        self.taste_mode_2.toggled.connect(self.on_taste_mode_checkbox_changed)
        
        self.taste_mode_3 = control_panel_widgets.get('taste_mode_3')
        self.taste_mode_3.toggled.connect(self.on_taste_mode_checkbox_changed)
        
        self.taste_m100 = control_panel_widgets.get('taste_m100')
        self.taste_m100.toggled.connect(self.update_selected_tastes)
        
        self.taste_m110 = control_panel_widgets.get('taste_m110')
        self.taste_m110.toggled.connect(self.update_selected_tastes)
        
        self.taste_m120 = control_panel_widgets.get('taste_m120')
        self.taste_m120.toggled.connect(self.update_selected_tastes)
        
        self.taste_selection_layout = control_panel_widgets.get('taste_selection_layout')
        self.taste_mode_status = control_panel_widgets.get('taste_mode_status')
        
        # Initially hide taste selection (since 3-taste mode is default)
        self.modbus_handlers.hide_taste_selection()
        
        # Expiry mode widgets
        self.expiry_mode_normal = control_panel_widgets.get('expiry_mode_normal')
        self.expiry_mode_normal.toggled.connect(self.on_expiry_mode_changed)
        
        self.expiry_mode_filter = control_panel_widgets.get('expiry_mode_filter')
        self.expiry_mode_filter.toggled.connect(self.on_expiry_mode_changed)
        
        self.expiry_filter_range = control_panel_widgets.get('expiry_filter_range')
        self.expiry_filter_range.toggled.connect(self.on_expiry_filter_type_changed)
        
        self.expiry_filter_specific = control_panel_widgets.get('expiry_filter_specific')
        self.expiry_filter_specific.toggled.connect(self.on_expiry_filter_type_changed)
        
        self.expiry_filter_type_layout = control_panel_widgets.get('expiry_filter_type_layout')
        self.date_selection_layout = control_panel_widgets.get('date_selection_layout')
        self.range_date_layout = control_panel_widgets.get('range_date_layout')
        self.specific_date_layout = control_panel_widgets.get('specific_date_layout')
        
        self.expiry_start_date_edit = control_panel_widgets.get('expiry_start_date_edit')
        self.expiry_start_date_edit.dateChanged.connect(self.update_expiry_mode_status)
        
        self.expiry_end_date_edit = control_panel_widgets.get('expiry_end_date_edit')
        self.expiry_end_date_edit.dateChanged.connect(self.update_expiry_mode_status)
        
        self.expiry_specific_date_edit = control_panel_widgets.get('expiry_specific_date_edit')
        self.expiry_specific_date_edit.dateChanged.connect(self.update_expiry_mode_status)
        
        self.expiry_date_type_combo = control_panel_widgets.get('expiry_date_type_combo')
        self.expiry_date_type_layout = control_panel_widgets.get('expiry_date_type_layout')
        if self.expiry_date_type_combo is not None:
            self.expiry_date_type_combo.currentIndexChanged.connect(self.on_expiry_date_type_changed)
            self.expiry_date_type = self.expiry_date_type_combo.currentData() or "BBF"
        
        self.expiry_mode_status = control_panel_widgets.get('expiry_mode_status')
        
        # Initially hide expiry filter controls (since normal mode is default)
        self.hide_expiry_filter_controls()
        
        # Gripper widgets
        self.btn_gripper_open = control_panel_widgets.get('btn_gripper_open')
        self.btn_gripper_open.clicked.connect(self.modbus_handlers.open_gripper)
        
        self.btn_gripper_close = control_panel_widgets.get('btn_gripper_close')
        self.btn_gripper_close.clicked.connect(self.modbus_handlers.close_gripper)
        
        self.btn_motor_reverse = control_panel_widgets.get('btn_motor_reverse')
        self.btn_motor_reverse.mousePressEvent = self.modbus_handlers.on_motor_reverse_press
        self.btn_motor_reverse.mouseReleaseEvent = self.modbus_handlers.on_motor_reverse_release
        
        self.gripper_status_label = control_panel_widgets.get('gripper_status_label')
        
        # Progress and status widgets
        self.progress_bar = control_panel_widgets.get('progress_bar')
        self.cap_progress_bar = control_panel_widgets.get('cap_progress_bar')
        self.status_label = control_panel_widgets.get('status_label')
        self.combined_status_label = control_panel_widgets.get('combined_status_label')
        self.queue_info_label = control_panel_widgets.get('queue_info_label')
        self.d6004_status_label = control_panel_widgets.get('d6004_status_label')
        self.d6007_status_label = control_panel_widgets.get('d6007_status_label')
        self.d5002_status_label = control_panel_widgets.get('d5002_status_label')
        self.system_time_label = control_panel_widgets.get('system_time_label')
        self.processing_status_label = control_panel_widgets.get('processing_status_label')
        
        # Legacy status labels for backward compatibility
        self.camera_status_label = self.camera_status_indicator.label if self.camera_status_indicator else None
        self.sentech_camera_status_label = self.sentech_status_indicator.label if self.sentech_status_indicator else None
        self.modbus_status_label = self.modbus_status_indicator.label if self.modbus_status_indicator else None
        self.queue_status_label = self.queue_status_indicator.label if self.queue_status_indicator else None
        self.d5002_status_header = self.d5002_status_indicator.label if self.d5002_status_indicator else None
        self.d5001_status_header = self.d5001_status_indicator.label if self.d5001_status_indicator else None
        
        # Add Control Panel tab to tab widget
        self.tab_widget.addTab(control_panel_tab, "⚙️ Control Panel")
        
        # Tab 0: Home - Monitor both bottle and cap detection (will be added first, so it becomes tab 0)
        home_tab, home_widgets = create_home_tab()
        # Map widgets to self for event handlers
        self.home_main_splitter = home_widgets['main_splitter']
        self.home_bottle_splitter = home_widgets['bottle_splitter']
        self.home_bottle_image_label = home_widgets['bottle_image_label']
        self.home_bottle_image_info_label = home_widgets['bottle_image_info_label']
        self.home_bottle_crops_scroll = home_widgets['bottle_crops_scroll']
        self.home_bottle_crops_container = home_widgets['bottle_crops_container']
        self.home_bottle_crops_layout = home_widgets['bottle_crops_layout']
        self.home_bottle_verdict_label = home_widgets['bottle_verdict_label']
        self.home_bottle_results_text = home_widgets['bottle_results_text']
        self.home_cap_splitter = home_widgets['cap_splitter']
        self.home_cap_image_label = home_widgets['cap_image_label']
        self.home_cap_image_info_label = home_widgets['cap_image_info_label']
        self.home_cap_results_scroll = home_widgets['cap_results_scroll']
        self.home_cap_results_container = home_widgets['cap_results_container']
        self.home_cap_results_layout = home_widgets['cap_results_layout']
        self.home_cap_verdict_label = home_widgets['cap_verdict_label']
        self.home_cap_detection_text = home_widgets['cap_detection_text']
        # Add Home tab first (so it becomes tab 0 and can be expanded by default)
        home_tab_index = self.tab_widget.addTab(home_tab, "🏠 Home")
        
        # Expand Home tab by default (after all tabs are added)
        # We'll do this after all tabs are added, so we need to call it later
        # Store reference for later expansion
        self.home_tab_index = home_tab_index
        
        # Tab 1: Bottle Detection (USB Camera) - Use component
        bottle_tab, bottle_widgets = create_bottle_tab()
        # Map widgets to self for event handlers
        self.bottle_splitter = bottle_widgets['bottle_splitter']
        self.image_label = bottle_widgets['image_label']
        self.image_info_label = bottle_widgets['image_info_label']
        self.crops_scroll = bottle_widgets['crops_scroll']
        self.crops_container = bottle_widgets['crops_container']
        self.crops_layout = bottle_widgets['crops_layout']
        self.results_text = bottle_widgets['results_text']
        # Map bottle tab buttons
        self.btn_capture_usb = bottle_widgets['btn_capture_usb']
        self.btn_process_bottle_tab = bottle_widgets['btn_process_bottle']
        self.btn_process_all_bottle = bottle_widgets['btn_process_all_bottle']
        self.btn_save_bottle_image_tab = bottle_widgets['btn_save_bottle_image']
        self.btn_select_bottle_image_tab = bottle_widgets['btn_select_bottle_image']
        self.btn_select_multiple_bottle_images_tab = bottle_widgets['btn_select_multiple_bottle_images']
        self.btn_prev_bottle_image = bottle_widgets['btn_prev_bottle_image']
        self.btn_next_bottle_image = bottle_widgets['btn_next_bottle_image']
        # Connect event handlers for bottle tab buttons
        self.btn_capture_usb.clicked.connect(self.bottle_handlers.capture_image_manual)
        self.btn_process_bottle_tab.clicked.connect(self.bottle_handlers.process_current_image)
        # Note: process_all_bottle_images not implemented yet - using process_current_image as fallback
        if hasattr(self.bottle_handlers, 'process_all_bottle_images'):
            self.btn_process_all_bottle.clicked.connect(self.bottle_handlers.process_all_bottle_images)
        else:
            self.btn_process_all_bottle.clicked.connect(self.bottle_handlers.process_current_image)
        self.btn_save_bottle_image_tab.clicked.connect(self.bottle_handlers.save_captured_image)
        self.btn_select_bottle_image_tab.clicked.connect(self.bottle_handlers.select_image_file)
        # Connect multiple images selection button
        if hasattr(self.bottle_handlers, 'select_multiple_bottle_images'):
            self.btn_select_multiple_bottle_images_tab.clicked.connect(self.bottle_handlers.select_multiple_bottle_images)
        else:
            # Fallback to single image selection if not implemented
            self.btn_select_multiple_bottle_images_tab.clicked.connect(self.bottle_handlers.select_image_file)
        # Note: go_prev_bottle_image and go_next_bottle_image not implemented yet
        if hasattr(self.bottle_handlers, 'go_prev_bottle_image'):
            self.btn_prev_bottle_image.clicked.connect(self.bottle_handlers.go_prev_bottle_image)
        if hasattr(self.bottle_handlers, 'go_next_bottle_image'):
            self.btn_next_bottle_image.clicked.connect(self.bottle_handlers.go_next_bottle_image)
        self.tab_widget.addTab(bottle_tab, "🔍 Bottle")
        
        # Tab 2: Cap Detection (Sentech Camera) - Use component
        cap_tab, cap_widgets = create_cap_tab()
        # Map widgets to self for event handlers
        self.cap_splitter = cap_widgets['cap_splitter']
        self.sentech_image_label = cap_widgets['sentech_image_label']
        self.sentech_image_info_label = cap_widgets['sentech_image_info_label']
        self.cap_results_scroll = cap_widgets['cap_results_scroll']
        self.cap_results_container = cap_widgets['cap_results_container']
        self.cap_results_layout = cap_widgets['cap_results_layout']
        self.cap_detection_text = cap_widgets['cap_detection_text']
        self.btn_capture_sentech = cap_widgets['btn_capture_sentech']
        self.btn_save_sentech_image = cap_widgets['btn_save_sentech_image']
        self.btn_select_cap_image = cap_widgets['btn_select_cap_image']
        self.btn_select_multiple_cap_images = cap_widgets['btn_select_multiple_cap_images']
        self.btn_process_cap = cap_widgets['btn_process_cap']
        # Connect event handlers
        self.btn_capture_sentech.clicked.connect(self.cap_handlers.capture_sentech_image)
        self.btn_save_sentech_image.clicked.connect(self.cap_handlers.save_sentech_image)
        self.btn_select_cap_image.clicked.connect(self.cap_handlers.select_cap_image_file)
        self.btn_select_multiple_cap_images.clicked.connect(self.cap_handlers.select_multiple_cap_images)
        self.btn_process_cap.clicked.connect(self.cap_handlers.process_cap_detection)
        self.tab_widget.addTab(cap_tab, "🔍 Cap")
        
        # Tab 3: Area Navigation - Use component
        arean_tab, arean_widgets = create_arean_tab()
        # Map buttons to self for event handlers
        self.arean_buttons = {}
        for key, btn in arean_widgets.items():
            self.arean_buttons[key] = btn
            # Extract M code from button text (e.g., "M900" -> 900)
            m_code = int(btn.text().replace('M', ''))
            # Connect button to handler
            btn.clicked.connect(lambda checked, code=m_code: self.on_arean_button_clicked(code))
        self.tab_widget.addTab(arean_tab, "📍 Area Navigation")
        
        # Tab 4: Modbus Status - Use component
        status_tab, status_widgets = create_status_tab()
        # Map widgets to self for event handlers
        self.modbus_connection_lamp = status_widgets['modbus_connection_lamp']
        self.modbus_connection_text = status_widgets['modbus_connection_text']
        self.m100_lamp = status_widgets['m100_lamp']
        self.m110_lamp = status_widgets['m110_lamp']
        self.m120_lamp = status_widgets['m120_lamp']
        self.m130_lamp = status_widgets['m130_lamp']
        self.m140_lamp = status_widgets['m140_lamp']
        self.m600_lamp = status_widgets['m600_lamp']
        self.m700_lamp = status_widgets['m700_lamp']
        self.m701_lamp = status_widgets['m701_lamp']
        self.m503_lamp = status_widgets['m503_lamp']
        self.m505_lamp = status_widgets['m505_lamp']
        self.m507_lamp = status_widgets['m507_lamp']
        self.m90_lamp = status_widgets['m90_lamp']
        self.m720_lamp = status_widgets['m720_lamp']
        self.m721_lamp = status_widgets['m721_lamp']
        self.m722_lamp = status_widgets['m722_lamp']
        self.m750_lamp = status_widgets['m750_lamp']
        self.m850_lamp = status_widgets['m850_lamp']
        self.total_images_processed = status_widgets['total_images_processed']
        self.successful_detections = status_widgets['successful_detections']
        self.error_count = status_widgets['error_count']
        self.success_rate = status_widgets['success_rate']
        self.usb_camera_status = status_widgets['usb_camera_status']
        self.sentech_camera_status = status_widgets['sentech_camera_status']
        self.bottle_detection_status = status_widgets['bottle_detection_status']
        self.cap_detection_status = status_widgets['cap_detection_status']
        self.current_bottle_type_status = status_widgets['current_bottle_type_status']
        self.processing_mode_status = status_widgets['processing_mode_status']
        self.queue_count_status = status_widgets['queue_count_status']
        self.d6004_lamp = status_widgets['d6004_lamp']
        self.d6004_text = status_widgets['d6004_text']
        self.d6007_lamp = status_widgets['d6007_lamp']
        self.d6007_text = status_widgets['d6007_text']
        self.m402_lamp = status_widgets['m402_lamp']
        self.m402_text = status_widgets['m402_text']
        self.m403_lamp = status_widgets['m403_lamp']
        self.m403_text = status_widgets['m403_text']
        self.m404_lamp = status_widgets['m404_lamp']
        self.m404_text = status_widgets['m404_text']
        self.m405_lamp = status_widgets['m405_lamp']
        self.m405_text = status_widgets['m405_text']
        self.m406_lamp = status_widgets['m406_lamp']
        self.m406_text = status_widgets['m406_text']
        self.d5002_lamp = status_widgets['d5002_lamp']
        self.d5002_text = status_widgets['d5002_text']
        self.m401_lamp = status_widgets['m401_lamp']
        self.m401_text = status_widgets['m401_text']
        # Store widget for admin-only Modbus Status tab
        self.modbus_status_tab_widget = status_tab
        
        # Tab 5: OCR Test - Use component (admin only, tab added on login)
        ocr_test_tab, ocr_test_widgets = create_ocr_test_tab()
        # Map widgets to self for event handlers
        self.ocr_test_splitter = ocr_test_widgets['ocr_test_splitter']
        self.btn_select_ocr_image = ocr_test_widgets['btn_select_ocr_image']
        self.btn_process_ocr = ocr_test_widgets['btn_process_ocr']
        self.btn_process_ocr_enhanced = ocr_test_widgets['btn_process_ocr_enhanced']
        self.btn_process_ocr_easyocr = ocr_test_widgets['btn_process_ocr_easyocr']
        self.btn_process_ocr_tesseract = ocr_test_widgets['btn_process_ocr_tesseract']
        self.ocr_status_label = ocr_test_widgets['ocr_status_label']
        self.ocr_image_label = ocr_test_widgets['ocr_image_label']
        self.ocr_results_text = ocr_test_widgets['ocr_results_text']
        self.ocr_progress_bar = ocr_test_widgets['ocr_progress_bar']
        # Connect event handlers
        self.btn_select_ocr_image.clicked.connect(self.select_ocr_test_image)
        self.btn_process_ocr.clicked.connect(self.process_ocr_test)
        self.btn_process_ocr_enhanced.clicked.connect(self.process_ocr_test_enhanced)
        self.btn_process_ocr_easyocr.clicked.connect(self.process_ocr_easyocr)
        # Tesseract OCR handler
        self.btn_process_ocr_tesseract.clicked.connect(self.process_ocr_tesseract)
        # Store widget for admin-only OCR Test tab
        self.ocr_test_tab_widget = ocr_test_tab
        
        # Tab 6: History - Use component
        history_tab, history_widgets = create_history_tab()
        # Map widgets to self for event handlers
        self.history_scroll = history_widgets['history_scroll']
        self.history_content = history_widgets['history_content']
        self.history_content_layout = history_widgets['history_content_layout']
        self.history_content_layout_good = history_widgets['history_content_layout_good']
        self.history_content_layout_ng = history_widgets['history_content_layout_ng']
        self.history_tab_widget = history_widgets['history_tab_widget']
        self.history_count_label = history_widgets['history_count_label']
        self.btn_clear_history = history_widgets['btn_clear_history']
        self.history_empty_label = history_widgets['empty_label']
        self.history_empty_label_good = history_widgets['empty_label_good']
        self.history_empty_label_ng = history_widgets['empty_label_ng']
        # Connect event handlers
        self.btn_clear_history.clicked.connect(self.clear_history)
        self.history_tab_widget.currentChanged.connect(self._on_history_subtab_changed)
        self.tab_widget.addTab(history_tab, "📋 History")
        
        # Tab 6: Modbus Register Test - Use component (admin only, tab added on login)
        robot_test_tab, robot_test_widgets = create_robot_test_tab()
        # Map widgets to self for event handlers
        self.robot_register_combo = robot_test_widgets['robot_register_combo']
        self.robot_address_input = robot_test_widgets['robot_address_input']
        self.robot_value_input = robot_test_widgets['robot_value_input']
        self.robot_value_desc_label = robot_test_widgets['robot_value_desc_label']
        self.robot_btn_write = robot_test_widgets['robot_btn_write']
        self.robot_btn_read = robot_test_widgets['robot_btn_read']
        self.robot_btn_write_read = robot_test_widgets['robot_btn_write_read']
        self.robot_results_text = robot_test_widgets['robot_results_text']
        self.robot_btn_clear = robot_test_widgets['robot_btn_clear']
        # Connect event handlers
        self.robot_register_combo.currentIndexChanged.connect(self.robot_test_handlers.on_register_changed)
        self.robot_btn_write.clicked.connect(self.robot_test_handlers.write_register)
        self.robot_btn_read.clicked.connect(self.robot_test_handlers.read_register)
        self.robot_btn_write_read.clicked.connect(self.robot_test_handlers.write_and_read)
        self.robot_btn_clear.clicked.connect(self.robot_test_handlers.clear_results)
        # Initialize value description
        self.robot_test_handlers.update_value_description()
        # Store widget for admin-only Register D Test tab
        self.robot_test_tab_widget = robot_test_tab
        
        # Tab 8: Settings - Use component (will be added when admin logs in)
        settings_tab, settings_widgets = create_settings_tab()
        self.settings_tab_widget = settings_tab
        # Map widgets to self for event handlers
        self.padding_spinbox = settings_widgets['padding_spinbox']
        self.shrink_x_spinbox = settings_widgets['shrink_x_spinbox']
        self.shrink_y_spinbox = settings_widgets['shrink_y_spinbox']
        self.max_height_ratio_spinbox = settings_widgets['max_height_ratio_spinbox']
        # Image enhancement settings
        self.brightness_spinbox = settings_widgets['brightness_spinbox']
        self.contrast_spinbox = settings_widgets['contrast_spinbox']
        self.gamma_spinbox = settings_widgets['gamma_spinbox']
        # Histogram matching settings
        self.use_histogram_matching_check = settings_widgets['use_histogram_matching_check']
        self.reference_cap_path_label = settings_widgets['reference_cap_path_label']
        self.select_reference_btn = settings_widgets['select_reference_btn']
        self.matching_method_combo = settings_widgets['matching_method_combo']
        self.prevent_saturation_check = settings_widgets['prevent_saturation_check']
        # Connect histogram matching handlers
        self.select_reference_btn.clicked.connect(self.on_select_reference_cap)
        self.use_histogram_matching_check.stateChanged.connect(self.on_histogram_matching_settings_changed)
        self.matching_method_combo.currentIndexChanged.connect(self.on_histogram_matching_settings_changed)
        self.prevent_saturation_check.stateChanged.connect(self.on_histogram_matching_settings_changed)
        # Load existing reference cap path if available
        self.load_reference_cap_path()
        # Connect event handlers
        self.padding_spinbox.valueChanged.connect(self.on_crop_settings_changed)
        self.shrink_x_spinbox.valueChanged.connect(self.on_crop_settings_changed)
        self.shrink_y_spinbox.valueChanged.connect(self.on_crop_settings_changed)
        self.max_height_ratio_spinbox.valueChanged.connect(self.on_crop_settings_changed)
        # Connect image enhancement handlers
        self.brightness_spinbox.valueChanged.connect(self.on_image_enhancement_changed)
        self.contrast_spinbox.valueChanged.connect(self.on_image_enhancement_changed)
        self.gamma_spinbox.valueChanged.connect(self.on_image_enhancement_changed)
        # Settings tab will be added/removed dynamically based on admin login
        
        layout.addWidget(self.tab_widget)
        
        # Expand Home tab by default (after all tabs are added)
        if hasattr(self, 'home_tab_index'):
            self.tab_widget._toggleTab(self.home_tab_index, True)
        
        # เพิ่ม main scroll area เข้าไปใน layout หลัก
        main_scroll_area.setWidget(main_widget)
        
        # สร้าง slide debug panel
        self.setup_debug_panel()
        
        # สร้าง main layout สำหรับหน้าต่างหลัก (HBoxLayout เพื่อรองรับ slide panel)
        main_container = QWidget()
        main_container_layout = QHBoxLayout(main_container)
        main_container_layout.setContentsMargins(0, 0, 0, 0)
        main_container_layout.setSpacing(0)
        
        # Main content area — stretch ให้เต็มความกว้างที่เหลือ (กันเนื้อหาดันให้เกิด H-scroll)
        main_container_layout.addWidget(main_scroll_area, 1)
        
        # Slide debug panel
        main_container_layout.addWidget(self.debug_panel)
        
        # Main layout สำหรับหน้าต่างหลัก
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(main_container)
    
    def setup_debug_panel(self):
        """Setup slide debug panel"""
        # สร้าง debug panel widget
        self.debug_panel = QWidget()
        # Set valid minimum and maximum sizes (avoid negative sizes)
        self.debug_panel.setMinimumSize(0, 0)  # Minimum width and height
        self.debug_panel.setMaximumWidth(400)
        self.debug_panel.setFixedWidth(0)  # เริ่มต้นให้ซ่อน (กว้าง 0)
        self.debug_panel.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-left: 2px solid #cccccc;
            }
            QTextEdit {
                background-color: #ffffff;
                color: #2c3e50;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #e0e0e0;
                padding: 5px;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #333333;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QLabel {
                color: #333333;
                font-weight: bold;
            }
        """)
        
        # Layout สำหรับ debug panel
        debug_layout = QVBoxLayout(self.debug_panel)
        debug_layout.setContentsMargins(5, 5, 5, 5)
        debug_layout.setSpacing(5)
        
        # Header with toggle button
        header_layout = QHBoxLayout()
        
        debug_title = QLabel("🐛 Debug Panel")
        debug_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333;")
        header_layout.addWidget(debug_title)
        
        header_layout.addStretch()
        
        # Toggle button (แสดง/ซ่อน)
        self.debug_toggle_btn = QPushButton("◀ Hide")
        self.debug_toggle_btn.setFixedWidth(60)
        self.debug_toggle_btn.clicked.connect(self.toggle_debug_panel)
        header_layout.addWidget(self.debug_toggle_btn)
        
        debug_layout.addLayout(header_layout)
        
        # Debug log area
        self.debug_log_text = QTextEdit()
        self.debug_log_text.setReadOnly(True)
        self.debug_log_text.setPlaceholderText("Debug logs will appear here...")
        debug_layout.addWidget(self.debug_log_text)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        # Clear button
        self.debug_clear_btn = QPushButton("🗑️ Clear")
        self.debug_clear_btn.clicked.connect(self.clear_debug_logs)
        control_layout.addWidget(self.debug_clear_btn)
        
        # Save button
        self.debug_save_btn = QPushButton("💾 Save")
        self.debug_save_btn.clicked.connect(self.save_debug_logs)
        control_layout.addWidget(self.debug_save_btn)
        
        debug_layout.addLayout(control_layout)
        
        # Log count label
        self.debug_log_count_label = QLabel("Logs: 0")
        self.debug_log_count_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        debug_layout.addWidget(self.debug_log_count_label)
        
        # Animation สำหรับ slide (ใช้ minimumWidth)
        self.debug_animation = QPropertyAnimation(self.debug_panel, b"minimumWidth")
        self.debug_animation.setDuration(300)  # 300ms animation
        self.debug_animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.debug_animation.finished.connect(self.on_debug_animation_finished)
    
    def setup_print_capture(self):
        """Setup stdout/stderr capture for debug panel"""
        # Create debug streams
        self.debug_stdout = DebugStream(self.original_stdout, self)
        self.debug_stderr = DebugStream(self.original_stderr, self)
        
        # Connect signals to update debug panel
        self.debug_stdout.message_written.connect(self.on_print_message)
        self.debug_stderr.message_written.connect(self.on_print_error)
        
        # Redirect stdout and stderr
        sys.stdout = self.debug_stdout
        sys.stderr = self.debug_stderr
    
    def on_print_message(self, message):
        """Handle print messages from stdout"""
        # Detect log level from message content
        level = "INFO"
        if "❌" in message or "ERROR" in message.upper() or "Error" in message:
            level = "ERROR"
        elif "⚠️" in message or "WARNING" in message.upper() or "Warning" in message:
            level = "WARNING"
        elif "✅" in message or "SUCCESS" in message.upper() or "Success" in message:
            level = "SUCCESS"
        
        # Add to debug panel
        self.add_debug_log(message, level)
    
    def on_print_error(self, message):
        """Handle error messages from stderr"""
        self.add_debug_log(message, "ERROR")
        
    def on_debug_animation_finished(self):
        """Called when debug panel animation finishes"""
        if not self.debug_panel_visible:
            # ซ่อนเสร็จแล้ว - ตั้ง fixedWidth เป็น 0
            self.debug_panel.setFixedWidth(0)
        else:
            # แสดงเสร็จแล้ว - ตั้ง fixedWidth เป็น 400
            self.debug_panel.setFixedWidth(400)
        
    def toggle_debug_panel(self):
        """Toggle debug panel visibility"""
        if self.debug_panel_visible:
            # ซ่อน panel - ลบ fixedWidth ก่อน animation
            self.debug_panel.setFixedWidth(-1)  # ลบ fixedWidth
            self.debug_panel.setMinimumWidth(400)
            self.debug_animation.setStartValue(400)
            self.debug_animation.setEndValue(0)
            self.debug_toggle_btn.setText("▶ Show")
            self.debug_toggle_header_btn.setToolTip("Debug")
            self.debug_panel_visible = False
            self.add_debug_log("Debug panel hidden", "INFO")
        else:
            # แสดง panel - ลบ fixedWidth ก่อน animation
            self.debug_panel.setFixedWidth(-1)  # ลบ fixedWidth
            self.debug_panel.setMinimumWidth(0)
            self.debug_animation.setStartValue(0)
            self.debug_animation.setEndValue(400)
            self.debug_toggle_btn.setText("◀ Hide")
            self.debug_toggle_header_btn.setToolTip("Hide Debug")
            self.debug_panel_visible = True
            # โหลด log ที่เก็บไว้ตอนปิด panel (แสดง 300 บรรทัดล่าสุด)
            if self.debug_logs:
                self.debug_log_text.setPlainText("\n".join(self.debug_logs[-300:]))
                cursor = self.debug_log_text.textCursor()
                cursor.movePosition(cursor.End)
                self.debug_log_text.setTextCursor(cursor)
            if hasattr(self, 'debug_log_count_label'):
                self.debug_log_count_label.setText(f"Logs: {len(self.debug_logs)}")
            self.add_debug_log("Debug panel shown", "INFO")
        
        self.debug_animation.start()
    
    def add_debug_log(self, message, level="INFO"):
        """Add debug message to log area. เมื่อปิด Debug panel จะไม่อัปเดต GUI (ลดการใช้ CPU/ความจำ)."""
        if not hasattr(self, 'debug_log_text'):
            return
        
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        line_with_time = f"[{timestamp}] {message}"
        
        # เก็บใน list เสมอ (จำกัด 500) — ใช้ตอนเปิด panel กับ save
        self.debug_logs.append(line_with_time)
        if len(self.debug_logs) > 500:
            self.debug_logs = self.debug_logs[-500:]
        
        # เมื่อปิด Debug panel — ไม่แตะ QTextEdit/label (ลดการกินทรัพยากรมาก)
        if not getattr(self, 'debug_panel_visible', False):
            return
        
        # Color code based on level
        if level == "ERROR":
            color = "#ff4444"
            icon = "❌"
        elif level == "WARNING":
            color = "#ffaa00"
            icon = "⚠️"
        elif level == "SUCCESS":
            color = "#44ff44"
            icon = "✅"
        elif level == "INFO":
            color = "#44aaff"
            icon = "ℹ️"
        else:
            color = "#ffffff"
            icon = "📝"
        formatted_message = f'<span style="color: {color};">{icon} {line_with_time}</span>'
        
        self.debug_log_text.append(formatted_message)
        cursor = self.debug_log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.debug_log_text.setTextCursor(cursor)
        if hasattr(self, 'debug_log_count_label'):
            self.debug_log_count_label.setText(f"Logs: {len(self.debug_logs)}")
        
        # จำกัดจำนวนบรรทัดใน widget (300) — ตัดแล้ว set ใหม่ครั้งเดียว ลดการกิน memory/CPU
        if self.debug_log_text.document().blockCount() > 300:
            keep = self.debug_logs[-300:]
            self.debug_log_text.setPlainText("\n".join(keep))
            cursor = self.debug_log_text.textCursor()
            cursor.movePosition(cursor.End)
            self.debug_log_text.setTextCursor(cursor)
    
    def debug_print(self, message, level="INFO"):
        """Print and log to debug panel"""
        # Print to console (with emoji/color indicators)
        if level == "ERROR":
            print(f"❌ {message}")
        elif level == "WARNING":
            print(f"⚠️ {message}")
        elif level == "SUCCESS":
            print(f"✅ {message}")
        elif level == "INFO":
            print(f"ℹ️ {message}")
        else:
            print(message)
        
        # Also log to debug panel
        self.add_debug_log(message, level)
    
    def _on_history_subtab_changed(self, index):
        """Rebuild Good or NG tab when user switches to it (index 1=Good, 2=NG)."""
        try:
            if index == 0:
                # All tab: show total count
                total = len(self.processing_history)
                if hasattr(self, "history_count_label"):
                    self.history_count_label.setText(f"History count: {total}")
            elif index == 1:
                self._rebuild_history_good_tab()
            elif index == 2:
                self._rebuild_history_ng_tab()
        except Exception as e:
            print(f"❌ ERROR in _on_history_subtab_changed: {e}")

    def _rebuild_history_good_tab(self):
        """Fill Good sub-tab from processing_history (Good entries only)."""
        layout = self.history_content_layout_good
        empty_label = self.history_empty_label_good
        while layout.count():
            w = layout.itemAt(0).widget()
            if w:
                w.setParent(None)
        good_entries = [e for e in self.processing_history if is_history_entry_good(e)]
        for entry in reversed(good_entries):  # newest first
            try:
                item = create_history_compact_item(entry, gui_instance=self)
                layout.insertWidget(0, item)
            except Exception as e:
                print(f"❌ Error building Good history item: {e}")
        layout.addWidget(empty_label)
        empty_label.setVisible(len(good_entries) == 0)
        # Update count label to Good count
        if hasattr(self, "history_count_label"):
            self.history_count_label.setText(f"History count: {len(good_entries)} (Good)")

    def _rebuild_history_ng_tab(self):
        """Fill NG sub-tab from processing_history (NG entries only)."""
        layout = self.history_content_layout_ng
        empty_label = self.history_empty_label_ng
        while layout.count():
            w = layout.itemAt(0).widget()
            if w:
                w.setParent(None)
        ng_entries = [e for e in self.processing_history if is_history_entry_ng(e)]
        for entry in reversed(ng_entries):
            try:
                item = create_history_compact_item(entry, gui_instance=self)
                layout.insertWidget(0, item)
            except Exception as e:
                print(f"❌ Error building NG history item: {e}")
        layout.addWidget(empty_label)
        empty_label.setVisible(len(ng_entries) == 0)
        # Update count label to NG count
        if hasattr(self, "history_count_label"):
            self.history_count_label.setText(f"History count: {len(ng_entries)} (NG)")

    def clear_history(self):
        """Clear all processing history"""
        try:
            reply = QMessageBox.question(
                self, 
                "Confirm clear history", 
                "Clear all processing history?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Clear history list
                self.processing_history.clear()
                
                # Clear All tab UI
                for i in reversed(range(self.history_content_layout.count())):
                    widget = self.history_content_layout.itemAt(i).widget()
                    if widget and widget is not self.history_empty_label:
                        widget.setParent(None)
                self.history_empty_label.setVisible(True)
                
                # Clear Good tab UI
                for i in reversed(range(self.history_content_layout_good.count())):
                    widget = self.history_content_layout_good.itemAt(i).widget()
                    if widget and widget is not self.history_empty_label_good:
                        widget.setParent(None)
                self.history_empty_label_good.setVisible(True)
                
                # Clear NG tab UI
                for i in reversed(range(self.history_content_layout_ng.count())):
                    widget = self.history_content_layout_ng.itemAt(i).widget()
                    if widget and widget is not self.history_empty_label_ng:
                        widget.setParent(None)
                self.history_empty_label_ng.setVisible(True)
                
                # Update count
                self.history_count_label.setText("History count: 0")
                
                print("✅ HISTORY: Cleared all processing history")
                QMessageBox.information(self, "Done", "Processing history cleared.")
        except Exception as e:
            print(f"❌ ERROR clearing history: {e}")
            QMessageBox.warning(self, "Error", f"Failed to clear history: {str(e)}")
    
    def add_to_history(self, bottle_result, cap_result=None):
        """
        Add processing result to history
        
        Args:
            bottle_result: dict - Bottle detection result
            cap_result: dict - Cap detection result (optional)
        """
        try:
            import datetime
            import re
            
            # Extract information
            bottle_image = bottle_result.get('image')
            bottle_type = bottle_result.get('bottle_type', 'Unknown')
            ocr_text = bottle_result.get('combined_ocr_text', '')
            
            # Extract expiry date and faded status from cap result
            expiry_date = None
            cap_image = None
            line_image = None  # ภาพรวมบรรทัด (สูงสุด 3 บรรทัด) สำหรับช่องภาพบรรทัดในประวัติ
            faded_status = None
            total_area = None  # Initialize total_area
            num_chars = None   # Initialize num_chars
            
            if cap_result:
                # Try to get OCR results from different possible locations
                ocr_results = None
                line_results = []
                
                # Check if ocr_results is directly in cap_result
                if 'ocr_results' in cap_result:
                    ocr_results = cap_result['ocr_results']
                # Check if it's in cap_processing_results
                elif 'cap_processing_results' in cap_result and len(cap_result['cap_processing_results']) > 0:
                    first_cap = cap_result['cap_processing_results'][0]
                    if 'ocr_results' in first_cap:
                        ocr_results = first_cap['ocr_results']
                
                # Extract line_results from ocr_results (อย่าใช้ ocr_results ใน boolean — อาจเป็น numpy)
                if ocr_results is not None:
                    if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                        line_results = ocr_results['line_results']
                    elif isinstance(ocr_results, list):
                        line_results = ocr_results
                
                # Get lines 1-3 for expiry date display (แสดงบรรทัดที่ 1-3 ที่อ่านได้)
                expiry_lines = []
                for i, line in enumerate(line_results[:3]):  # Get first 3 lines only
                    if isinstance(line, dict):
                        line_text = line.get('recognized_text', '')
                    else:
                        line_text = str(line)
                    
                    if line_text:
                        expiry_lines.append(line_text)
                        print(f"✅ HISTORY: Line {i+1}: {line_text}")
                
                # Join lines 1-3 with newline
                if expiry_lines:
                    expiry_date = "\n".join(expiry_lines)
                    print(f"✅ HISTORY: Expiry date (lines 1-3): {expiry_date}")
                    # ถ้าขวดไม่มีข้อความ OCR ให้ใช้ข้อความจากฝาเป็น ocr_text ในประวัติ
                    if not (ocr_text or "").strip():
                        ocr_text = expiry_date
                else:
                    # Fallback: Try to extract date pattern if no lines found
                    for line in line_results:
                        if isinstance(line, dict):
                            line_text = line.get('recognized_text', '')
                        else:
                            line_text = str(line)
                        
                        # Look for date patterns
                        date_patterns = [
                            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YY or DD/MM/YYYY
                            r'BBF\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # BBF DD/MM/YY
                            r'MFG\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MFG DD/MM/YY
                        ]
                        
                        for pattern in date_patterns:
                            match = re.search(pattern, line_text, re.IGNORECASE)
                            if match:
                                expiry_date = match.group()
                                print(f"✅ HISTORY: Found expiry date: {expiry_date} from text: {line_text}")
                                break
                        
                        if expiry_date:
                            break
                
                # ใช้ข้อความ OCR จากฝาเป็น ocr_text ในประวัติ ถ้าขวดไม่มี (โหมด Batch ฝา)
                if (not (ocr_text or "").strip()) and ocr_results is not None and isinstance(ocr_results, dict):
                    total_rec = ocr_results.get('total_recognized_text')
                    if total_rec and isinstance(total_rec, str):
                        ocr_text = total_rec.strip()
                    elif line_results:
                        parts = []
                        for line in line_results[:5]:
                            if isinstance(line, dict):
                                t = line.get('recognized_text', '')
                            else:
                                t = str(line)
                            if t:
                                parts.append(t)
                        if parts:
                            ocr_text = "\n".join(parts)
                
                # Get cap image สำหรับช่อง "ภาพฝา" = ภาพครอปฝาที่ใช้ตรวจสอบ (ไม่ใช้ภาพรวมบรรทัด)
                cap_image = None
                line_detection_result = None
                if 'cap_processing_results' in cap_result and len(cap_result['cap_processing_results']) > 0:
                    first_cap = cap_result['cap_processing_results'][0]
                    if 'line_detection_result' in first_cap:
                        line_detection_result = first_cap['line_detection_result']
                elif 'line_detection_result' in cap_result:
                    line_detection_result = cap_result['line_detection_result']
                
                def _safe_copy(arr):
                    if arr is None or not hasattr(arr, 'shape'):
                        return None
                    return arr.copy() if hasattr(arr, 'copy') else np.asarray(arr).copy()
                
                # ลำดับที่ 1: ภาพครอปฝาที่ใช้ตรวจสอบ (original_crop / rotated ฯลฯ) — ใช้โชว์ในช่องภาพฝา
                if 'cap_processing_results' in cap_result and len(cap_result['cap_processing_results']) > 0:
                    first_cap = cap_result['cap_processing_results'][0]
                    for key in ('original_crop', 'unified_region', 'rotated_image', 'craft_rotated_image'):
                        if first_cap.get(key) is not None and hasattr(first_cap[key], 'shape'):
                            cap_image = _safe_copy(first_cap[key])
                            if cap_image is not None:
                                print(f"✅ HISTORY: Cap image (ภาพครอปที่ใช้ตรวจสอบ) from cap_processing_results[0]['{key}']")
                                break
                if cap_image is None and cap_result.get('image') is not None and hasattr(cap_result['image'], 'shape'):
                    cap_image = _safe_copy(cap_result['image'])
                    if cap_image is not None:
                        print(f"✅ HISTORY: Cap image from cap_result['image']")
                if cap_image is None and cap_result.get('rotated_image') is not None and hasattr(cap_result['rotated_image'], 'shape'):
                    cap_image = _safe_copy(cap_result['rotated_image'])
                    if cap_image is not None:
                        print(f"✅ HISTORY: Cap image from cap_result['rotated_image']")
                if cap_image is None and cap_result.get('craft_rotated_image') is not None and hasattr(cap_result['craft_rotated_image'], 'shape'):
                    cap_image = _safe_copy(cap_result['craft_rotated_image'])
                    if cap_image is not None:
                        print(f"✅ HISTORY: Cap image from cap_result['craft_rotated_image']")
                
                # Fallback: ไม่มีภาพครอปฝา จึงใช้ภาพบรรทัดหรือภาพเต็ม
                if cap_image is None and line_detection_result and line_detection_result.get('cropped_lines'):
                    cropped_lines = line_detection_result['cropped_lines']
                    for line_data in cropped_lines:
                        if isinstance(line_data, dict) and line_data.get('image') is not None and hasattr(line_data['image'], 'shape'):
                            img = line_data['image']
                            cap_image = img.copy() if hasattr(img, 'copy') else np.asarray(img).copy()
                            print(f"✅ HISTORY: Cap image fallback from first cropped line")
                            break
                    if cap_image is None and cropped_lines and isinstance(cropped_lines[0], dict) and cropped_lines[0].get('image') is not None:
                        cap_image = _safe_copy(cropped_lines[0]['image'])
                        if cap_image is not None:
                            print(f"✅ HISTORY: Cap image fallback from first line")
                if cap_image is None and getattr(self, 'current_sentech_image', None) is not None:
                    cap_image = _safe_copy(self.current_sentech_image)
                    if cap_image is not None:
                        print(f"✅ HISTORY: Using current_sentech_image as fallback")
                
                print(f"✅ HISTORY: Cap image found: {cap_image is not None}, type: {type(cap_image)}")
                if cap_image is not None:
                    print(f"✅ HISTORY: Cap image shape: {cap_image.shape if hasattr(cap_image, 'shape') else 'N/A'}")
                
                # สร้างภาพรวมบรรทัด (สูงสุด 3 บรรทัด) สำหรับช่อง "ภาพบรรทัด" ในประวัติ
                line_image = None
                if line_detection_result and line_detection_result.get('cropped_lines'):
                    cropped_lines = line_detection_result['cropped_lines']
                    line_images = []
                    for ld in cropped_lines[:3]:
                        if isinstance(ld, dict) and ld.get('image') is not None and hasattr(ld['image'], 'shape'):
                            line_images.append(np.asarray(ld['image'], dtype=np.uint8))
                    if line_images:
                        try:
                            max_width = max(img.shape[1] for img in line_images)
                            total_height = sum(img.shape[0] for img in line_images) + (len(line_images) - 1) * 5
                            combined = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255
                            y_offset = 0
                            for line_img in line_images:
                                h, w = line_img.shape[:2]
                                if len(line_img.shape) == 2:
                                    line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
                                if w > max_width:
                                    line_img = cv2.resize(line_img, (max_width, int(h * max_width / w)))
                                    h, w = line_img.shape[:2]
                                x_offset = (max_width - w) // 2
                                combined[y_offset:y_offset+h, x_offset:x_offset+w] = line_img[:, :, :3]
                                y_offset += h + 5
                            line_image = combined
                        except Exception as e:
                            print(f"⚠️ HISTORY: Build line image failed: {e}")
                
                # Extract faded status and area from cap result
                if 'cap_processing_results' in cap_result and len(cap_result['cap_processing_results']) > 0:
                    first_cap = cap_result['cap_processing_results'][0]
                    if 'faded_text_result' in first_cap:
                        faded_text_result = first_cap['faded_text_result']
                        faded_status = faded_text_result.get('status', None)
                        total_area = faded_text_result.get('total_area', None)
                        num_chars = faded_text_result.get('num_chars', None)
                        normalized_area = faded_text_result.get('normalized_area', None)
                        print(f"✅ HISTORY: Faded status: {faded_status}, total_area: {total_area}, num_chars: {num_chars}, normalized_area: {normalized_area}")
                elif 'faded_text_result' in cap_result:
                    faded_text_result = cap_result['faded_text_result']
                    faded_status = faded_text_result.get('status', None)
                    total_area = faded_text_result.get('total_area', None)
                    num_chars = faded_text_result.get('num_chars', None)
                    normalized_area = faded_text_result.get('normalized_area', None)
                    print(f"✅ HISTORY: Faded status: {faded_status}, total_area: {total_area}, num_chars: {num_chars}, normalized_area: {normalized_area}")
            
            # Get bottle image if available
            if bottle_image is None and self.current_image is not None:
                bottle_image = self.current_image
            
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create history entry — เก็บเฉพาะ thumbnail (max 320px) เพื่อลด memory เมื่อรัน full auto นาน
            def _thumb(img, max_side=320):
                if img is None or not hasattr(img, 'shape'):
                    return None
                try:
                    h, w = img.shape[:2]
                    if max(h, w) <= max_side:
                        return np.ascontiguousarray(img.copy() if hasattr(img, 'copy') else img)
                    scale = max_side / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    out = cuda_resize(img, (new_w, new_h))
                    if out is not None and hasattr(out, 'shape'):
                        return out
                    return cv2.resize(img, (new_w, new_h))
                except Exception:
                    try:
                        h, w = img.shape[:2]
                        if max(h, w) > max_side:
                            scale = max_side / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            return cv2.resize(img, (new_w, new_h))
                        return np.ascontiguousarray(img.copy() if hasattr(img, 'copy') else img)
                    except Exception:
                        return None
            
            bottle_img_copy = _thumb(bottle_image)
            cap_img_copy = _thumb(cap_image)
            line_img_copy = _thumb(line_image)
            if line_img_copy is None and line_image is not None and hasattr(line_image, 'shape'):
                try:
                    h, w = line_image.shape[:2]
                    if h > 0 and w > 0:
                        max_side = 320
                        if max(h, w) > max_side:
                            scale = max_side / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            line_img_copy = cv2.resize(np.asarray(line_image, dtype=np.uint8), (new_w, new_h))
                        else:
                            line_img_copy = np.ascontiguousarray(np.asarray(line_image, dtype=np.uint8))
                except Exception:
                    pass
            # ถ้ามีภาพฝาแต่ _thumb คืน None (เช่น grayscale/ขนาดพิเศษ) สร้าง thumbnail เอง
            if cap_img_copy is None and cap_image is not None and hasattr(cap_image, 'shape'):
                try:
                    h, w = cap_image.shape[:2]
                    if h > 0 and w > 0:
                        max_side = 320
                        if max(h, w) > max_side:
                            scale = max_side / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            cap_img_copy = cv2.resize(np.asarray(cap_image, dtype=np.uint8), (new_w, new_h))
                        else:
                            cap_img_copy = np.ascontiguousarray(np.asarray(cap_image, dtype=np.uint8))
                        print(f"✅ HISTORY: Cap thumbnail fallback (shape {cap_img_copy.shape})")
                except Exception as e:
                    print(f"⚠️ HISTORY: Cap thumbnail fallback failed: {e}")
            
            # ขวด: Good/NG และ score จาก defect_inspection (angle1)
            bottle_defect = None
            bottle_defect_score = None
            if bottle_result.get('defect_inspection'):
                di = bottle_result['defect_inspection']
                bottle_defect = di.get('result')  # "Good" or "NG"
                if 'score' in di:
                    bottle_defect_score = di['score']
            # ฝา: ผ่าน/ไม่ผ่าน จาก faded_status (normal=ผ่าน, faded=ไม่ผ่าน)
            cap_status = None
            if cap_result and faded_status is not None:
                cap_status = "ผ่าน" if faded_status == 'normal' else "ไม่ผ่าน"
            elif cap_result:
                cap_status = "ผ่าน"  # มีผลฝาแต่ไม่มี faded_status ให้ถือว่าผ่าน
            
            # Initialize normalized_area if not already set
            if 'normalized_area' not in locals():
                normalized_area = None
            
            history_entry = {
                'timestamp': timestamp,
                'bottle_image': bottle_img_copy,
                'cap_image': cap_img_copy,
                'line_image': line_img_copy,
                'bottle_type': bottle_type,
                'expiry_date': expiry_date,
                'ocr_text': ocr_text,
                'faded_status': faded_status,
                'total_area': total_area,
                'num_chars': num_chars,
                'normalized_area': normalized_area,
                'bottle_defect': bottle_defect,
                'bottle_defect_score': bottle_defect_score,
                'cap_status': cap_status
            }
            
            # Add to history list (จำกัดจำนวน — ดึงของเก่าออกเมื่อเกิน)
            self.processing_history.append(history_entry)
            while len(self.processing_history) > getattr(self, 'processing_history_max', 200):
                self.processing_history.pop(0)
                if hasattr(self, 'history_content_layout') and self.history_content_layout.count() > 0:
                    last_idx = self.history_content_layout.count() - 1
                    w = self.history_content_layout.itemAt(last_idx).widget()
                    # Layout คือ [ใหม่สุด..เก่าสุด, empty_label] — อย่าลบ empty_label ต้องลบรายการเก่าสุด (index ก่อน empty_label)
                    if w is self.history_empty_label:
                        last_idx -= 1
                    w = self.history_content_layout.itemAt(last_idx).widget() if last_idx >= 0 else None
                    if w:
                        w.setParent(None)
            
            # Create history item widget
            try:
                # โหมดย่อทั้งหมด: ให้ข้อมูลสรุป + ปุ่ม/ข้อความ "ดูรายละเอียด" เพื่อเปิด dialog แยก
                history_item = create_history_compact_item(history_entry, gui_instance=self)
                
                # Hide empty label if visible
                if self.history_empty_label.isVisible():
                    self.history_empty_label.setVisible(False)
                
                # Add to layout (at the top - newest first) in All tab
                self.history_content_layout.insertWidget(0, history_item)
                
                # Update count according to active sub-tab
                try:
                    active_index = self.history_tab_widget.currentIndex() if hasattr(self, "history_tab_widget") else 0
                except Exception:
                    active_index = 0
                total_count = len(self.processing_history)
                if active_index == 1:
                    good_count = sum(1 for e in self.processing_history if is_history_entry_good(e))
                    self.history_count_label.setText(f"History count: {good_count} (Good)")
                elif active_index == 2:
                    ng_count = sum(1 for e in self.processing_history if is_history_entry_ng(e))
                    self.history_count_label.setText(f"History count: {ng_count} (NG)")
                else:
                    self.history_count_label.setText(f"History count: {total_count}")
                
                print(f"✅ HISTORY: Added entry - {bottle_type} at {timestamp}")
            except Exception as e:
                print(f"❌ ERROR creating history item: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"❌ ERROR adding to history: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_debug_logs(self):
        """Clear debug log area"""
        self.debug_log_text.clear()
        self.debug_logs = []
        if hasattr(self, 'debug_log_count_label'):
            self.debug_log_count_label.setText("Logs: 0")
        self.add_debug_log("Debug logs cleared", "INFO")

    def show_history_detail(self, history_entry):
        """
        แสดง dialog รายละเอียดประวัติแบบเต็ม จากข้อมูลใน processing_history
        ใช้ร่วมกับ compact history item
        """
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout
            dlg = QDialog(self)
            dlg.setWindowTitle("รายละเอียดประวัติ")
            dlg.resize(1000, 700)
            layout = QVBoxLayout(dlg)
            
            item_widget = create_history_item(
                bottle_image=history_entry.get('bottle_image'),
                cap_image=history_entry.get('cap_image'),
                line_image=history_entry.get('line_image'),
                bottle_type=history_entry.get('bottle_type', 'Unknown'),
                expiry_date=history_entry.get('expiry_date'),
                timestamp=history_entry.get('timestamp'),
                ocr_text=history_entry.get('ocr_text', ''),
                faded_status=history_entry.get('faded_status'),
                total_area=history_entry.get('total_area'),
                num_chars=history_entry.get('num_chars'),
                normalized_area=history_entry.get('normalized_area'),
                bottle_defect=history_entry.get('bottle_defect'),
                bottle_defect_score=history_entry.get('bottle_defect_score'),
                cap_status=history_entry.get('cap_status'),
                gui_instance=self
            )
            layout.addWidget(item_widget)
            dlg.exec_()
        except Exception as e:
            print(f"❌ ERROR showing history detail dialog: {e}")
    
    def save_debug_logs(self):
        """Save debug logs to file"""
        try:
            from datetime import datetime
            filename = f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(os.getcwd(), filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for log in self.debug_logs:
                    f.write(log + '\n')
            
            self.add_debug_log(f"Debug logs saved to {filename}", "SUCCESS")
            QMessageBox.information(self, "บันทึกสำเร็จ", f"บันทึก debug logs ไว้ที่:\n{filepath}")
        except Exception as e:
            self.add_debug_log(f"Error saving logs: {e}", "ERROR")
            QMessageBox.warning(self, "ข้อผิดพลาด", f"ไม่สามารถบันทึก debug logs ได้:\n{str(e)}")
        
    def setup_camera(self):
        """Setup USB camera"""
        try:
            self.usb_camera = USBCamera()
            if self.usb_camera.open_camera():
                # Update Control Panel status indicator
                if self.camera_status_indicator:
                    self.camera_status_indicator.set_status('online')
                    self.camera_status_indicator.set_text('USB Camera: Ready')
                # Update legacy label if exists
                if self.camera_status_label:
                    self.camera_status_label.setText('📷 Camera: Ready')
                    self.camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                # Update Top Panel indicator
                if hasattr(self, 'usb_camera_status_indicator'):
                    self.usb_camera_status_indicator.setStyleSheet("color: #27ae60; font-size: 16px; font-weight: bold;")
                self.status_handlers.update_usb_camera_status("พร้อมใช้งาน", True)
                print("✅ USB Camera initialized successfully")
            else:
                # Update Control Panel status indicator
                if self.camera_status_indicator:
                    self.camera_status_indicator.set_status('offline')
                    self.camera_status_indicator.set_text('USB Camera: Cannot open')
                # Update legacy label if exists
                if self.camera_status_label:
                    self.camera_status_label.setText('📷 Camera: Cannot open')
                    self.camera_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                # Update Top Panel indicator
                if hasattr(self, 'usb_camera_status_indicator'):
                    self.usb_camera_status_indicator.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
                self.status_handlers.update_usb_camera_status("ไม่สามารถเปิดได้", False)
                print("❌ Failed to initialize USB Camera")
        except Exception as e:
            # Update Control Panel status indicator
            if self.camera_status_indicator:
                self.camera_status_indicator.set_status('error')
                self.camera_status_indicator.set_text(f'USB Camera: Error - {str(e)}')
            # Update legacy label if exists
            if self.camera_status_label:
                self.camera_status_label.setText(f'📷 Camera: Error - {str(e)}')
                self.camera_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            # Update Top Panel indicator
            if hasattr(self, 'usb_camera_status_indicator'):
                self.usb_camera_status_indicator.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
            self.update_usb_camera_status(f"ข้อผิดพลาด - {str(e)}", False)
            print(f"❌ Camera setup error: {e}")

    def _start_background_init(self):
        """เริ่มโหลดกล้อง/โมเดลในพื้นหลัง"""
        if getattr(self, '_init_worker', None) is not None:
            return
        self._init_worker = InitWorker(self)
        self._init_worker.camera_ready.connect(self._on_init_camera_ready)
        self._init_worker.sentech_ready.connect(self._on_init_sentech_ready)
        self._init_worker.models_ready.connect(self._on_init_models_ready)
        self._init_worker.progress_msg.connect(self._on_init_progress)
        self._init_worker.finished.connect(self._on_init_worker_finished)
        self._init_worker.start()

    def _on_init_progress(self, msg):
        if hasattr(self, 'status_label'):
            self.status_label.setText(f'⏳ {msg} (you can click)')
        self.init_progress.emit(msg)

    def _on_init_camera_ready(self, cam, ok):
        self.usb_camera = cam
        if cam is not None and ok:
            self.camera_status_label.setText('📷 Camera: Ready')
            self.camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            if hasattr(self, 'usb_camera_status_indicator'):
                self.usb_camera_status_indicator.setStyleSheet("color: #27ae60; font-size: 16px; font-weight: bold;")
            self.status_handlers.update_usb_camera_status("พร้อมใช้งาน", True)
            print("✅ USB Camera initialized successfully")
        else:
            self.camera_status_label.setText('📷 Camera: Cannot open' if cam else '📷 Camera: Error')
            self.camera_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            if hasattr(self, 'usb_camera_status_indicator'):
                self.usb_camera_status_indicator.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
            self.status_handlers.update_usb_camera_status("ไม่สามารถเปิดได้", False)

    def _on_init_sentech_ready(self, sc, ok):
        self.sentech_camera = sc
        if sc is not None and ok:
            self.sentech_camera_status_label.setText('📷 Sentech: Ready (Harvesters)')
            self.sentech_camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            if hasattr(self, 'sentech_camera_status_indicator'):
                self.sentech_camera_status_indicator.setStyleSheet("color: #27ae60; font-size: 16px; font-weight: bold;")
            self.status_handlers.update_sentech_camera_status("พร้อมใช้งาน", True)
            print("✅ SENTECH camera initialized successfully with Harvesters")
        else:
            self.sentech_camera_status_label.setText('📷 Sentech: Cannot connect' if sc else '📷 Sentech: Error')
            self.sentech_camera_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            if hasattr(self, 'sentech_camera_status_indicator'):
                self.sentech_camera_status_indicator.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
            self.status_handlers.update_sentech_camera_status("ไม่สามารถเชื่อมต่อได้", False)

    def _on_init_models_ready(self, d):
        if d is None or not d:
            print("⚠️ Cap detection models not loaded in background")
            # Initialize with None values
            self.cap_detector = None
            self.craft_detector = None
            self.rotation_model = None
            self.line_detector = None
            self.ocr_model = None
            self.faded_text_yolo_model = None
            return
        self.cap_detector = d.get('cap_detector')
        self.craft_detector = d.get('craft_detector')
        self.rotation_model = d.get('rotation_model')
        self.line_detector = d.get('line_detector')
        self.ocr_model = d.get('ocr_model')
        self.faded_text_yolo_model = d.get('faded_text_yolo_model')
        
        # ตรวจสอบว่า cap_detector โหลดสำเร็จหรือไม่
        if self.cap_detector is not None:
            if hasattr(self.cap_detector, 'model') and self.cap_detector.model is None:
                print(f"⚠️ WARNING: cap_detector instance exists but model is None!")
                print(f"   - cap_detector type: {type(self.cap_detector)}")
                print(f"   - cap_detector.model: {self.cap_detector.model}")
            else:
                print(f"✅ cap_detector loaded successfully - model type: {type(self.cap_detector.model) if hasattr(self.cap_detector, 'model') else 'N/A'}")
        else:
            print(f"❌ ERROR: cap_detector is None after initialization!")
        if self.line_detector and hasattr(self, 'padding_spinbox'):
            self.padding_spinbox.setValue(self.line_detector.settings.get('padding', 0))
            shrink_x = self.line_detector.settings.get('shrink_x_percent', 0.02) * 100.0
            shrink_y = self.line_detector.settings.get('shrink_y_percent', 0.10) * 100.0
            max_hr = self.line_detector.settings.get('max_height_ratio', 0.1) * 100.0
            self.shrink_x_spinbox.setValue(shrink_x)
            self.shrink_y_spinbox.setValue(shrink_y)
            self.max_height_ratio_spinbox.setValue(max_hr)
        self.optimize_ocr_for_numbers()
        print("✅ All cap detection models applied from background init")

    def _on_init_worker_finished(self):
        """โหลด Modbus บน main thread หลังกล้อง/โมเดลพร้อม"""
        self._init_worker = None
        self.init_progress.emit("กำลังเชื่อมต่อ Modbus...")
        if hasattr(self, 'status_label'):
            self.status_label.setText('⏳ Connecting Modbus...')
        QtWidgets.QApplication.processEvents()
        self.setup_modbus()
        if hasattr(self, 'status_label'):
            self.status_label.setText('⏸️ Ready (wait for M511 to start)')
        print("✅ Background init finished - GUI ready")
        self.init_complete.emit()

    def setup_sentech_camera(self):
        """Setup Sentech camera for cap detection using Harvesters only"""
        try:
            print("🔧 Setting up SENTECH Camera with Harvesters...")
            
            # Create SentechCamera instance with Harvesters
            self.sentech_camera = SentechCamera()
            
            # Initialize camera
            print("🔧 Initializing SENTECH camera...")
            success = self.sentech_camera.initialize()
            
            if success:
                self.sentech_camera_status_label.setText('📷 Sentech: Ready (Harvesters)')
                self.sentech_camera_status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                # Update Top Panel indicator
                if hasattr(self, 'sentech_camera_status_indicator'):
                    self.sentech_camera_status_indicator.setStyleSheet("color: #27ae60; font-size: 16px; font-weight: bold;")
                self.status_handlers.update_sentech_camera_status("พร้อมใช้งาน", True)
                print("✅ SENTECH camera initialized successfully with Harvesters")
            else:
                self.sentech_camera_status_label.setText('📷 Sentech: Cannot connect')
                self.sentech_camera_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                # Update Top Panel indicator
                if hasattr(self, 'sentech_camera_status_indicator'):
                    self.sentech_camera_status_indicator.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
                self.status_handlers.update_sentech_camera_status("ไม่สามารถเชื่อมต่อได้", False)
                print("❌ Failed to initialize SENTECH camera")
            
        except Exception as e:
            self.sentech_camera_status_label.setText(f'📷 Sentech: Error - {str(e)}')
            self.sentech_camera_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            # Update Top Panel indicator
            if hasattr(self, 'sentech_camera_status_indicator'):
                self.sentech_camera_status_indicator.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
            self.update_sentech_camera_status(f"ข้อผิดพลาด - {str(e)}", False)
            print(f"❌ SENTECH Camera setup error: {e}")
    
    def setup_cap_detection_models(self):
        """Setup cap detection models - ENABLED (cap processing turned on)"""
        print("🔧 Setting up cap detection models...")
        
        # ENABLED CODE - cap processing turned on
        if not CAP_DETECTION_AVAILABLE:
            print("⚠️ Cap detection modules not available")
            return
            
        try:
            print("🔧 Setting up cap detection models...")
            
            # Initialize cap detector
            self.cap_detector = initialize_detector(CAP_MODEL_PATH)
            print("✅ Cap detector initialized")
            
            # Initialize CRAFT detector
            self.craft_detector = initialize_craft_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
            print("✅ CRAFT detector initialized")
            
            # Initialize rotation model
            self.rotation_model = initialize_model(ROTATION_MODEL_PATH)
            print("✅ Rotation model initialized")
            
            # Initialize CRAFT detector in rotation model
            self.rotation_model.init_craft_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
            print("✅ CRAFT detector initialized in rotation model")
            
            # Initialize line detector
            self.line_detector = initialize_line_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
            
            # Update settings UI with current values
            if self.line_detector and hasattr(self, 'padding_spinbox'):
                self.padding_spinbox.setValue(self.line_detector.settings.get('padding', 0))
                shrink_x_percent = self.line_detector.settings.get('shrink_x_percent', 0.02) * 100.0
                shrink_y_percent = self.line_detector.settings.get('shrink_y_percent', 0.10) * 100.0
                max_height_ratio = self.line_detector.settings.get('max_height_ratio', 0.1) * 100.0
                self.shrink_x_spinbox.setValue(shrink_x_percent)
                self.shrink_y_spinbox.setValue(shrink_y_percent)
                self.max_height_ratio_spinbox.setValue(max_height_ratio)
                print(f"✅ Settings UI updated: padding={self.line_detector.settings.get('padding', 0)}, shrink_x={shrink_x_percent:.1f}%, shrink_y={shrink_y_percent:.1f}%, max_height_ratio={max_height_ratio:.1f}%")
            print("✅ Line detector initialized")
            
            # Initialize OCR model (ต้องมี fix10.pth ใน models/ocr/)
            if os.path.exists(OCR_MODEL_PATH):
                self.ocr_model = initialize_ocr_model(OCR_MODEL_PATH)
                print("✅ OCR model initialized")
            else:
                self.ocr_model = None
                print(f"⚠️ OCR model ไม่โหลด: วาง ocr.pth ที่ {os.path.abspath(OCR_MODEL_PATH)}")
            
            # Optimize OCR settings for better number recognition
            self.optimize_ocr_for_numbers()
            
            # ฝาจางใช้ cap_fade_model.h5 (libs.detection.cap_fade_model) แทน YOLO
            self.faded_text_yolo_model = None
            
            print("✅ All cap detection models initialized successfully")
            
        except Exception as e:
            print(f"❌ Error setting up cap detection models: {e}")
    
    def optimize_ocr_for_numbers(self):
        """
        ปรับปรุงการตั้งค่า DeepOCR เพื่อให้อ่านเลขได้ดีขึ้น
        """
        try:
            if self.ocr_model is None:
                print("⚠️ OCR model not available for optimization")
                return
            
            print("🔧 Optimizing OCR settings for better number recognition...")
            
            # ปรับปรุงการตั้งค่าสำหรับการอ่านเลข
            optimized_settings = {
                'imgH': 64,  # เพิ่มความสูงจาก 32 เป็น 64 เพื่อให้อ่านเลขได้ชัดขึ้น
                'imgW': 200,  # เพิ่มความกว้างจาก 100 เป็น 200 เพื่อให้อ่านเลขหลายตัวได้ดีขึ้น
                'batch_max_length': 10,  # ลดจาก 25 เป็น 10 เพราะเราต้องการอ่านเลขสั้นๆ
                'character': '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:/',  # เน้นตัวเลข
                'sensitive': True,
                'PAD': True,  # เปิดใช้ padding เพื่อให้ภาพมีขนาดสม่ำเสมอ
            }
            
            # อัปเดตการตั้งค่า
            self.ocr_model.update_settings(**optimized_settings)
            
            print("✅ OCR settings optimized for number recognition:")
            print(f"   - Image height: {optimized_settings['imgH']}")
            print(f"   - Image width: {optimized_settings['imgW']}")
            print(f"   - Max length: {optimized_settings['batch_max_length']}")
            print(f"   - Padding enabled: {optimized_settings['PAD']}")
            
        except Exception as e:
            print(f"❌ Error optimizing OCR settings: {e}")
    
    def preprocess_image_for_ocr(self, image):
        """
        ปรับปรุงภาพก่อนส่งให้ OCR เพื่อให้อ่านเลขได้ดีขึ้น
        """
        try:
            if image is None:
                return image
            
            print("🔧 Preprocessing image for better OCR...")
            
            # แปลงเป็น grayscale
            if len(image.shape) == 3:
                gray = cuda_cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # เพิ่มขนาดภาพเพื่อให้อ่านได้ชัดขึ้น
            height, width = gray.shape
            scale_factor = 2.0  # เพิ่มขนาด 2 เท่า
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize ด้วย INTER_CUBIC เพื่อให้ภาพชัดขึ้น
            resized = cuda_resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # ปรับปรุงความคมชัดด้วย Gaussian blur + unsharp mask
            blurred = cuda_gaussianBlur(resized, (0, 0), 2.0)
            sharpened = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
            
            # ปรับปรุงความคมชัดด้วย CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened)
            
            # ปรับปรุงความคมชัดด้วย morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # แปลงกลับเป็น BGR format สำหรับ DeepOCR
            if len(image.shape) == 3:
                result = cuda_cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                result = enhanced
            
            print(f"✅ Image preprocessed: {width}x{height} -> {new_width}x{new_height}")
            return result
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return image
    
    def setup_modbus(self):
        """Setup Modbus connection"""
        try:
            print("🔧 Setting up Modbus connection...")
            self.modbus_thread = ModbusThread(MODBUS_IP, MODBUS_PORT)
            self.modbus_thread.modbus_status.connect(self.status_handlers.update_modbus_status)
            self.modbus_thread.trigger_detected.connect(self.modbus_handlers.on_modbus_trigger)
            self.modbus_thread.queue_trigger_detected.connect(self.modbus_handlers.on_queue_trigger)
            self.modbus_thread.silent_trigger_detected.connect(self.modbus_handlers.on_silent_trigger)
            self.modbus_thread.result_ready_to_display.connect(self.modbus_handlers.on_result_ready_to_display)
            self.modbus_thread.pending_display_requested.connect(self.modbus_handlers.on_pending_display_requested)
            self.modbus_thread.bottle_type_detected.connect(self.on_bottle_type_detected)
            self.modbus_thread.capture_only_trigger.connect(self.modbus_handlers.on_capture_only_trigger)
            self.modbus_thread.d6004_status_updated.connect(self.status_handlers.update_d6004_status)
            self.modbus_thread.d6007_status_updated.connect(self.status_handlers.update_d6007_status)  # Deprecated
            self.modbus_thread.m403_status_updated.connect(self.status_handlers.update_m403_status)
            self.modbus_thread.m404_status_updated.connect(self.status_handlers.update_m404_status)
            self.modbus_thread.m405_status_updated.connect(self.status_handlers.update_m405_status)
            self.modbus_thread.m406_status_updated.connect(self.status_handlers.update_m406_status)
            self.modbus_thread.m76_status_updated.connect(self.update_light_m76_status)
            self.modbus_thread.d5002_status_updated.connect(self.status_handlers.update_d5002_status)
            self.modbus_thread.d5001_status_updated.connect(self.status_handlers.update_d5001_status)
            self.modbus_thread.m401_status_updated.connect(self.status_handlers.update_m401_status)
            self.modbus_thread.m402_status_updated.connect(self.status_handlers.update_m402_status)
            self.modbus_thread.reset_processing_signal.connect(self.bottle_handlers.reset_processing_for_new_image)
            self.modbus_thread.m513_stop_signal.connect(self.modbus_handlers.on_m513_stop)
            self.modbus_thread.capture_limit_reached_signal.connect(self.modbus_handlers.on_capture_limit_reached)
            print("🔧 Starting Modbus thread...")
            self.modbus_thread.start()
            print("✅ Modbus thread started successfully")
            
            # Update status tab
            self.status_handlers.update_modbus_connection_status_ui(True)
            
            # รอให้ Modbus เชื่อมต่อสำเร็จก่อน
            QTimer.singleShot(2000, self.modbus_handlers.initialize_modbus_state)
            
            # ตั้งค่าโหมดเริ่มต้น - เขียนค่า ID ที่เลือกอยู่ไปที่ D5500
            def write_initial_mode():
                """Write initial mode ID to D5500"""
                try:
                    if self.modbus_thread and self.modbus_thread.modbus_client and self.modbus_thread.modbus_client.is_socket_open():
                        mode_text = self.mode_combo.currentText()
                        if mode_text == "ID 7 full auto":
                            success = self.modbus_thread.write_register(5500, 7)
                            if success:
                                print("✅ INITIAL MODE: D5500 = 7 (ID 7 full auto mode)")
                                self.modbus_thread.capture_only_mode = False
                            else:
                                print("❌ INITIAL MODE: Failed to write D5500 = 7")
                        elif mode_text == "ID 5 capture only":
                            success = self.modbus_thread.write_register(5500, 5)
                            if success:
                                print("✅ INITIAL MODE: D5500 = 5 (ID 5 capture only mode)")
                                self.modbus_thread.capture_only_mode = True
                                self.modbus_handlers.create_capture_folders()
                            else:
                                print("❌ INITIAL MODE: Failed to write D5500 = 5")
                        elif mode_text == "ID 8 reset modbus":
                            success = self.modbus_thread.write_register(5500, 8)
                            if success:
                                print("✅ INITIAL MODE: D5500 = 8 (ID 8 reset modbus mode)")
                                # Reset modbus to initial state
                                print("🔄 INITIAL MODE: Resetting Modbus to initial state...")
                                self.modbus_handlers.reset_to_initial_state()
                                print("✅ INITIAL MODE: Modbus reset completed")
                            else:
                                print("❌ INITIAL MODE: Failed to write D5500 = 8")
                            self.modbus_thread.capture_only_mode = False
                    else:
                        # ถ้ายังไม่พร้อม ให้ลองอีกครั้งใน 500ms
                        QTimer.singleShot(500, write_initial_mode)
                except Exception as e:
                    print(f"❌ INITIAL MODE ERROR: {e}")
            
            QTimer.singleShot(2500, write_initial_mode)
            
            print("⏸️ โปรแกรมพร้อมทำงาน (รอ M511 เพื่อเริ่มการทำงาน)")
        except Exception as e:
            self.modbus_status_label.setText(f'📡 Modbus: Error - {str(e)}')
            self.modbus_status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            # Update Top Panel indicator
            if hasattr(self, 'modbus_status_indicator'):
                self.modbus_status_indicator.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
            self.status_handlers.update_modbus_connection_status_ui(False)
            print(f"❌ Modbus setup error: {e}")
    
    # Modbus operation methods delegated to modbus_handlers
    # Use self.modbus_handlers.initialize_modbus_state() instead
    # Use self.modbus_handlers.off_m503() instead
    
    # Status update methods delegated to status_handlers
    # Use self.status_handlers.update_all_status() instead
    # Use self.status_handlers.update_camera_status() instead
    # Use self.status_handlers.update_modbus_connection_status() instead
    # Use self.status_handlers.update_system_status() instead
    # Use self.status_handlers.update_modbus_status() instead
    # Use self.status_handlers.update_queue_status() instead
    # Use self.status_handlers.update_d6004_status() instead
    # Use self.status_handlers.update_d6007_status() instead
    # Use self.status_handlers.show_d6007_dialog() instead
    # Use self.status_handlers.update_usb_camera_status() instead
    # Use self.status_handlers.update_sentech_camera_status() instead
    # Use self.status_handlers.update_modbus_connection_status_ui() instead
    # Use self.status_handlers.update_coil_lamp() instead
    # Use self.status_handlers.update_register_lamp() instead
    # Use self.status_handlers.update_bottle_detection_status() instead
    # Use self.status_handlers.update_cap_detection_status() instead
    # Use self.status_handlers.update_bottle_type_status() instead
    # Use self.status_handlers.update_processing_mode_status() instead
    # Use self.status_handlers.update_performance_stats() instead
    
    # Modbus operation methods delegated to modbus_handlers
    # Use self.modbus_handlers.on_modbus_trigger() instead
    # Use self.modbus_handlers.on_mode_changed() instead
    
    def on_login_button_clicked(self):
        """Handle login button click"""
        if self.is_admin_logged_in:
            # Logout
            reply = QMessageBox.question(self, "Logout", "คุณต้องการออกจากระบบหรือไม่?",
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.logout()
        else:
            # Show login dialog
            self.show_login_dialog()
    
    def show_login_dialog(self):
        """Show login dialog"""
        dialog = LoginDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            username, password = dialog.get_credentials()
            if username == "admin" and password == "admin":
                self.is_admin_logged_in = True
                # Change login button to red circular logout state but keep icon
                self.login_button.setToolTip("Logout")
                self.login_button.setStyleSheet("""
                    QPushButton {
                        background-color: #e74c3c;
                        border: none;
                        border-radius: 24px;
                        padding: 0;
                    }
                    QPushButton:hover {
                        background-color: #c0392b;
                    }
                    QPushButton:pressed {
                        background-color: #a93226;
                    }
                """)
                # Show Settings tab by adding it
                if self.settings_tab_widget is not None and self.settings_tab_index < 0:
                    self.settings_tab_index = self.tab_widget.addTab(self.settings_tab_widget, self.settings_tab_label)
                # Show admin-only tabs: Modbus Status, OCR Test, Register D Test
                if self.modbus_status_tab_widget is not None and self.modbus_status_tab_index < 0:
                    self.modbus_status_tab_index = self.tab_widget.addTab(self.modbus_status_tab_widget, self.modbus_status_tab_label)
                if self.ocr_test_tab_widget is not None and self.ocr_test_tab_index < 0:
                    self.ocr_test_tab_index = self.tab_widget.addTab(self.ocr_test_tab_widget, self.ocr_test_tab_label)
                if self.robot_test_tab_widget is not None and self.robot_test_tab_index < 0:
                    self.robot_test_tab_index = self.tab_widget.addTab(self.robot_test_tab_widget, self.robot_test_tab_label)
                print("✅ LOGIN SUCCESS: Admin logged in - Settings tab enabled")
                QMessageBox.information(self, "Login Success", "เข้าสู่ระบบสำเร็จ!\nAdmin-only tabs are now enabled.")
            else:
                QMessageBox.warning(self, "Login Failed", "Username หรือ Password ไม่ถูกต้อง!\n\nUsername: admin\nPassword: admin")
                print("❌ LOGIN FAILED: Invalid credentials")
    
    def logout(self):
        """Logout admin"""
        self.is_admin_logged_in = False
        self.login_button.setToolTip("Login")
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                border: none;
                border-radius: 24px;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        # Hide Settings tab by removing it
        if self.settings_tab_index >= 0:
            self.tab_widget.removeTab(self.settings_tab_index)
            self.settings_tab_index = -1
        # Hide admin-only tabs by removing them
        if self.modbus_status_tab_index >= 0:
            self.tab_widget.removeTab(self.modbus_status_tab_index)
            self.modbus_status_tab_index = -1
        if self.ocr_test_tab_index >= 0:
            self.tab_widget.removeTab(self.ocr_test_tab_index)
            self.ocr_test_tab_index = -1
        if self.robot_test_tab_index >= 0:
            self.tab_widget.removeTab(self.robot_test_tab_index)
            self.robot_test_tab_index = -1
        print("✅ LOGOUT: Admin logged out - admin-only tabs disabled")
    
    def on_arean_button_clicked(self, m_code):
        """Handle area navigation button click"""
        try:
            print(f"📍 AREAN: กดปุ่ม M{m_code}")
            
            if not hasattr(self, 'modbus_thread') or self.modbus_thread is None:
                QMessageBox.warning(self, "ข้อผิดพลาด", "Modbus thread ยังไม่ได้เริ่มต้น")
                return
            
            # ส่งคำสั่ง Modbus ตาม M code (ใช้ write_coil)
            success = self.modbus_thread.write_coil(m_code, True)
            
            if success:
                self.status_label.setText(f'✅ M{m_code} sent OK')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                print(f"✅ AREAN: ส่งคำสั่ง M{m_code} สำเร็จ")
            else:
                self.status_label.setText(f'❌ Failed to send M{m_code}')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                print(f"❌ AREAN: ไม่สามารถส่งคำสั่ง M{m_code} ได้")
                QMessageBox.warning(self, "ข้อผิดพลาด", f"ไม่สามารถส่งคำสั่ง M{m_code} ได้")
                
        except Exception as e:
            error_msg = f"เกิดข้อผิดพลาดในการส่งคำสั่ง M{m_code}: {str(e)}"
            self.status_label.setText(f'❌ {error_msg}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            QMessageBox.critical(self, "ข้อผิดพลาด", error_msg)
            print(f"❌ AREAN ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_image_enhancement_changed(self):
        """Update image enhancement config when settings change"""
        try:
            from config import settings as config_settings
            
            # Get current values from spinboxes
            brightness = self.brightness_spinbox.value()
            contrast = self.contrast_spinbox.value()
            gamma = self.gamma_spinbox.value()
            
            # Update config
            config_settings.FADED_TEXT_CONFIG['IMAGE_ENHANCEMENT']['brightness'] = brightness
            config_settings.FADED_TEXT_CONFIG['IMAGE_ENHANCEMENT']['contrast'] = contrast
            config_settings.FADED_TEXT_CONFIG['IMAGE_ENHANCEMENT']['gamma'] = gamma
            
            print(f"✅ อัปเดต Image Enhancement: Brightness={brightness}, Contrast={contrast:.2f}, Gamma={gamma:.2f}")
            
        except Exception as e:
            print(f"❌ Error updating image enhancement: {str(e)}")
    
    def on_select_reference_cap(self):
        """Handle reference cap image selection"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "เลือกรูปฝาอ้างอิง (Reference Cap Image)",
                "",
                "Image files (*.png *.jpg *.jpeg *.bmp *.tiff);;All files (*.*)"
            )
            
            if file_path:
                # Update config
                from config.settings import FADED_TEXT_CONFIG
                FADED_TEXT_CONFIG['REFERENCE_CAP_IMAGE_PATH'] = file_path
                
                # Update label
                self.reference_cap_path_label.setText(os.path.basename(file_path))
                self.reference_cap_path_label.setToolTip(file_path)
                self.reference_cap_path_label.setStyleSheet("color: #27ae60; font-size: 11px; padding: 5px; border: 1px solid #27ae60; border-radius: 3px; background-color: #d5f4e6;")
                
                print(f"✅ เลือกรูปฝาอ้างอิง: {file_path}")
                
        except Exception as e:
            print(f"❌ Error selecting reference cap: {e}")
            QMessageBox.warning(self, "ข้อผิดพลาด", f"ไม่สามารถเลือกรูปอ้างอิงได้: {str(e)}")
    
    def load_reference_cap_path(self):
        """Load reference cap path from config and update UI"""
        try:
            from config.settings import FADED_TEXT_CONFIG
            ref_path = FADED_TEXT_CONFIG.get('REFERENCE_CAP_IMAGE_PATH')
            
            if ref_path and os.path.exists(ref_path):
                self.reference_cap_path_label.setText(os.path.basename(ref_path))
                self.reference_cap_path_label.setToolTip(ref_path)
                self.reference_cap_path_label.setStyleSheet("color: #27ae60; font-size: 11px; padding: 5px; border: 1px solid #27ae60; border-radius: 3px; background-color: #d5f4e6;")
            else:
                self.reference_cap_path_label.setText("ยังไม่ได้เลือก")
                self.reference_cap_path_label.setToolTip("")
                self.reference_cap_path_label.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 5px; border: 1px solid #bdc3c7; border-radius: 3px; background-color: #ecf0f1;")
        except Exception as e:
            print(f"❌ Error loading reference cap path: {e}")
    
    def on_histogram_matching_settings_changed(self):
        """Handle histogram matching settings changes"""
        try:
            from config.settings import FADED_TEXT_CONFIG
            
            # Update config
            FADED_TEXT_CONFIG['USE_HISTOGRAM_MATCHING'] = self.use_histogram_matching_check.isChecked()
            
            method_map = {0: 'clahe', 1: 'histogram_matching', 2: 'brightness', 3: 'mixed'}
            FADED_TEXT_CONFIG['HISTOGRAM_MATCHING_METHOD'] = method_map.get(
                self.matching_method_combo.currentIndex(), 'clahe'
            )
            
            FADED_TEXT_CONFIG['PREVENT_SATURATION'] = self.prevent_saturation_check.isChecked()
            
            print(f"✅ อัปเดต Histogram Matching Settings:")
            print(f"   ใช้ Histogram Matching: {FADED_TEXT_CONFIG['USE_HISTOGRAM_MATCHING']}")
            print(f"   วิธี: {FADED_TEXT_CONFIG['HISTOGRAM_MATCHING_METHOD']}")
            print(f"   ป้องกัน Saturation: {FADED_TEXT_CONFIG['PREVENT_SATURATION']}")
            
        except Exception as e:
            print(f"❌ Error updating histogram matching settings: {e}")
    
    def on_crop_settings_changed(self):
        """Handle crop settings change"""
        try:
            if self.line_detector is None:
                print("⚠️ CROP SETTINGS: Line detector not initialized yet")
                return
            
            # Get current values from spinboxes
            padding = self.padding_spinbox.value()
            shrink_x_percent = self.shrink_x_spinbox.value() / 100.0  # Convert % to decimal
            shrink_y_percent = self.shrink_y_spinbox.value() / 100.0  # Convert % to decimal
            max_height_ratio = self.max_height_ratio_spinbox.value() / 100.0  # Convert % to decimal
            
            # Update line detector settings
            self.line_detector.settings['padding'] = padding
            self.line_detector.settings['shrink_x_percent'] = shrink_x_percent
            self.line_detector.settings['shrink_y_percent'] = shrink_y_percent
            self.line_detector.settings['max_height_ratio'] = max_height_ratio
            
            print(f"✅ CROP SETTINGS UPDATED: padding={padding}, shrink_x={shrink_x_percent*100:.1f}%, shrink_y={shrink_y_percent*100:.1f}%, max_height_ratio={max_height_ratio*100:.1f}%")
            
        except Exception as e:
            print(f"❌ CROP SETTINGS ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Modbus operation methods delegated to modbus_handlers
    # Use self.modbus_handlers.on_queue_trigger() instead
    
    def toggle_auto_mode(self, state):
        """Toggle auto mode - Function kept for compatibility but auto mode is always enabled"""
        # Auto mode is always enabled, this function is kept for compatibility
        self.status_handlers.update_processing_mode_status("Auto")
    
    # Bottle detection methods delegated to bottle_handlers
    # Use self.bottle_handlers.capture_image_manual() instead
    # Use self.bottle_handlers.select_image_file() instead
    
    # Cap detection methods delegated to cap_handlers
    # Use self.cap_handlers.select_cap_image_file() instead
    
    # Bottle detection methods delegated to bottle_handlers
    # Use self.bottle_handlers.capture_image_auto() instead
    # Use self.bottle_handlers.capture_image_from_queue() instead
    # Use self.bottle_handlers.display_image() instead
    # Use self.bottle_handlers.display_cropped_images() instead
    # Use self.bottle_handlers.process_current_image() instead
    # Use self.bottle_handlers.on_processing_complete() instead
    # Use self.bottle_handlers.handle_bottle_type_detection() instead
    
    def on_bottle_type_detected(self, bottle_type, ocr_text):
        """Handle bottle type detection signal"""
        print(f"🎯 BOTTLE TYPE SIGNAL: {bottle_type} (OCR: '{ocr_text}')")
        self.bottle_handlers.handle_bottle_type_detection(
            bottle_type,
            ocr_text,
            bottle_result=getattr(self, 'last_bottle_result', None),
        )
    
    # Modbus operation methods delegated to modbus_handlers
    # Use self.modbus_handlers.on_m513_stop() instead
    
    # Bottle detection methods delegated to bottle_handlers
    # Use self.bottle_handlers.reset_processing_for_new_image() instead
    
    # Cap detection methods delegated to cap_handlers
    # Use self.cap_handlers.select_cap_image_file() instead
    # Use self.cap_handlers.capture_sentech_image() instead
    # Use self.cap_handlers.capture_sentech_image_auto() instead
    # Use self.cap_handlers.capture_sentech_image_from_queue() instead
    # Use self.cap_handlers.display_sentech_image() instead
    # Use self.cap_handlers.process_cap_detection() instead
    # Use self.cap_handlers.update_cap_progress() instead
    # Use self.cap_handlers.update_cap_progress_bar() instead
    # Use self.cap_handlers.on_cap_processing_complete() instead
    # Use self.cap_handlers.display_cap_processing_ui() instead
    # Use self.cap_handlers.display_cap_processing_complete_ui() instead
    
    def show_rotation_attempt(self, attempt_num, total_attempts, rotated_image, ocr_results=None, format_valid=None):
        """Display current rotation attempt image and OCR results"""
        try:
            # Convert numpy array to QPixmap
            height, width, channel = rotated_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rotated_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale the image to fit the display area
            scaled_pixmap = pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Update the sentech image display
            self.sentech_image_label.setPixmap(scaled_pixmap)
            self.sentech_image_label.setAlignment(Qt.AlignCenter)
            
            # Update status to show current rotation attempt
            angle = [0, 90, 180, 270, -90][attempt_num - 1]
            
            # Build status message with OCR results if available
            if ocr_results is not None and format_valid is not None:
                # Extract text from OCR results
                ocr_text = self.extract_ocr_text(ocr_results)
                status_text = f"📋 Step {attempt_num}: Rotation {angle}° | OCR: {ocr_text[:25]}{'...' if len(ocr_text) > 25 else ''} | {'✅ Valid' if format_valid else '❌ Invalid'}"
            else:
                status_text = f"📋 Step {attempt_num}: Rotating to {angle}°..."
            
            self.status_label.setText(status_text)
            
            # Force GUI update
            QApplication.processEvents()
            
            if ocr_results is not None:
                print(f"📋 STEP {attempt_num}: Rotation {angle}° | OCR: {self.extract_ocr_text(ocr_results)[:50]}... | {'✅ Valid' if format_valid else '❌ Invalid'}")
            else:
                print(f"📋 STEP {attempt_num}: Rotating to {angle}°...")
            
        except Exception as e:
            print(f"❌ Error displaying rotation attempt: {e}")
    
    def extract_ocr_text(self, ocr_results):
        """Extract text from OCR results for display"""
        try:
            if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                # Extract text from line results
                text_parts = []
                for line_result in ocr_results['line_results']:
                    if isinstance(line_result, dict) and 'text' in line_result:
                        text_parts.append(line_result['text'])
                return ' | '.join(text_parts) if text_parts else 'No text found'
            elif isinstance(ocr_results, str):
                return ocr_results
            else:
                return 'Unknown OCR format'
        except Exception as e:
            return f'Error extracting text: {e}'
    
    # Manual rotation function removed
    
    # Manual re-OCR function removed
    
    # Manual OCR display functions removed
    
    def update_bottle_progress_bar(self, value):
        """Update bottle detection progress bar with specific value"""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Processing... ({value}%)")
    
    def show_global_loading(self, message="Processing..."):
        """แสดงแถบ loading ระดับทั้งแอป (มองเห็นทุกแท็บ)"""
        self._global_loading_count = getattr(self, '_global_loading_count', 0) + 1
        self.global_loading_label.setText(f"🔄 {message}")
        self.global_loading_banner.setVisible(True)
        if hasattr(self, 'processing_status_label'):
            self.processing_status_label.setText(f"🔄 {message}")
            self.processing_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px; font-weight: bold;")
    
    def _update_robot_alarm_marquee(self):
        """อัปเดตตัวอักษรวิ่งสำหรับ Robot Alarm"""
        try:
            if not hasattr(self, 'robot_alarm_label'):
                return
            text = getattr(self, '_robot_alarm_base_text', "   Robot Alarm   ")
            if not text:
                text = "   Robot Alarm   "
            length = len(text)
            if length == 0:
                return
            self._robot_alarm_marquee_index = (getattr(self, '_robot_alarm_marquee_index', 0) + 1) % length
            idx = self._robot_alarm_marquee_index
            rotated = text[idx:] + text[:idx]
            self.robot_alarm_label.setText(rotated)
        except Exception as e:
            print(f"❌ Error updating robot alarm marquee: {e}")
    
    def set_robot_alarm_active(self, active=False):
        """แสดง/ซ่อนแถบ Robot Alarm ด้านบนเมื่อ D5001 มี error"""
        try:
            if not hasattr(self, 'robot_alarm_banner'):
                return
            if active:
                self.robot_alarm_banner.setVisible(True)
                if hasattr(self, 'robot_alarm_timer') and not self.robot_alarm_timer.isActive():
                    self.robot_alarm_timer.start(200)  # อัปเดตข้อความทุก 200ms
            else:
                self.robot_alarm_banner.setVisible(False)
                if hasattr(self, 'robot_alarm_timer') and self.robot_alarm_timer.isActive():
                    self.robot_alarm_timer.stop()
                # รีเซ็ตข้อความกลับเป็นคงที่
                if hasattr(self, 'robot_alarm_label'):
                    self.robot_alarm_label.setText("Robot Alarm")
        except Exception as e:
            print(f"❌ Error in set_robot_alarm_active: {e}")
    
    def hide_global_loading(self):
        """ซ่อนแถบ loading เมื่อการประมวลผลจบ (ใช้ ref count ถ้ามีทั้งขวดและฝารันพร้อมกัน)"""
        self._global_loading_count = max(0, getattr(self, '_global_loading_count', 0) - 1)
        if self._global_loading_count == 0:
            self.global_loading_banner.setVisible(False)
            if hasattr(self, 'processing_status_label'):
                self.processing_status_label.setText("⏸️ No processing")
                self.processing_status_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 11px;")
    
    def _clear_display_fallback(self):
        """เคลียร์ผลบนหน้าจอเมื่อกด Stop (ใช้เมื่อ modbus_handlers ไม่มีหรือ clear_current_data ล้มเหลว)"""
        try:
            self.current_result = None
            self.current_cap_result = None
            if hasattr(self, 'results_text') and self.results_text:
                self.results_text.clear()
            if hasattr(self, 'image_label') and self.image_label:
                self.image_label.clear()
                self.image_label.setText("📷 ไม่มีภาพ")
                self.image_label.setStyleSheet("color: #7f8c8d; padding: 20px; border: 2px dashed #7f8c8d;")
            if hasattr(self, 'bottle_handlers') and self.bottle_handlers:
                self.bottle_handlers.clear_cropped_images_display()
            if hasattr(self, 'sentech_image_label') and self.sentech_image_label:
                self.sentech_image_label.clear()
                self.sentech_image_label.setText("ยังไม่มีภาพจากกล้อง Sentech")
                self.sentech_image_label.setStyleSheet("color: #7f8c8d; padding: 20px; border: 2px dashed #7f8c8d;")
            if hasattr(self, 'cap_detection_text') and self.cap_detection_text:
                self.cap_detection_text.clear()
            if hasattr(self, 'cap_results_layout') and self.cap_results_layout:
                for i in reversed(range(self.cap_results_layout.count())):
                    w = self.cap_results_layout.itemAt(i).widget()
                    if w:
                        w.setParent(None)
            if hasattr(self, 'home_bottle_results_text') and self.home_bottle_results_text:
                self.home_bottle_results_text.clear()
            if hasattr(self, 'home_bottle_image_label') and self.home_bottle_image_label:
                self.home_bottle_image_label.clear()
                self.home_bottle_image_label.setText("ยังไม่มีภาพจากกล้อง USB")
            if hasattr(self, 'home_cap_image_label') and self.home_cap_image_label:
                self.home_cap_image_label.clear()
                self.home_cap_image_label.setText("ยังไม่มีภาพจากกล้อง Sentech")
        except Exception as e:
            print(f"⚠️ _clear_display_fallback: {e}")
    
    def _clear_display_on_stop(self):
        """เมื่อกด STOP (M701) หรือ stop_system — เคลียร์ผลบนหน้าจอทันที และกันผลมาสาย"""
        self._stopping_processing = True
        try:
            if hasattr(self, 'modbus_handlers') and self.modbus_handlers:
                self.modbus_handlers.clear_current_data()
            else:
                self._clear_display_fallback()
        except Exception as e:
            print(f"⚠️ _clear_display_on_stop: {e}")
            self._clear_display_fallback()
    
    def stop_processing(self):
        """Stop current processing — เรียก request_stop ก่อน รอให้ thread หยุดจริง แล้วค่อย terminate ถ้ายังไม่หยุด"""
        self._stopping_processing = True
        try:
            # ซ่อน loading และรีเซ็ตปุ่มทันทีเมื่อกด Stop (ไม่รอให้ thread หยุดก่อน)
            self._global_loading_count = 0
            if hasattr(self, 'global_loading_banner'):
                self.global_loading_banner.setVisible(False)
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(False)
            if hasattr(self, 'cap_progress_bar'):
                self.cap_progress_bar.setVisible(False)
            if hasattr(self, 'btn_stop_processing'):
                self.btn_stop_processing.setEnabled(False)
            if hasattr(self, 'btn_process'):
                self.btn_process.setEnabled(True)
            if hasattr(self, 'btn_process_bottle_tab'):
                self.btn_process_bottle_tab.setEnabled(True)
            if hasattr(self, 'btn_process_all_bottle'):
                fn = getattr(self, '_batch_all_button_should_enable', None)
                self.btn_process_all_bottle.setEnabled(fn() if callable(fn) else True)
            if hasattr(self, 'status_label'):
                self.status_label.setText('⏹️ Stopping...')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            # ขอหยุดแบบ cooperative ก่อน (ทั้งขวดและฝา)
            if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
                print("🛑 Stopping bottle detection thread...")
                if getattr(self.processing_thread, 'request_stop', None):
                    self.processing_thread.request_stop()
            if hasattr(self, 'cap_processing_thread') and self.cap_processing_thread and self.cap_processing_thread.isRunning():
                print("🛑 Stopping cap detection thread...")
                if getattr(self.cap_processing_thread, 'request_stop', None):
                    self.cap_processing_thread.request_stop()
            # Stop paired processing thread (for Process both)
            if hasattr(self, 'bottle_handlers') and hasattr(self.bottle_handlers, 'paired_thread'):
                paired = getattr(self.bottle_handlers, 'paired_thread', None)
                if paired is not None and paired.isRunning():
                    print("🛑 Stopping paired processing thread...")
                    try:
                        paired.should_stop = True
                    except Exception:
                        pass
            # รอให้หยุดจริง (สูงสุด ~2.5 วินาที) โดยให้ GUI ยังอัปเดตได้
            for _ in range(25):
                QApplication.processEvents()
                if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
                    self.processing_thread.wait(100)
                if hasattr(self, 'cap_processing_thread') and self.cap_processing_thread and self.cap_processing_thread.isRunning():
                    self.cap_processing_thread.wait(100)
                if (not (getattr(self, 'processing_thread', None) and self.processing_thread.isRunning()) and
                    not (getattr(self, 'cap_processing_thread', None) and self.cap_processing_thread.isRunning())):
                    break
            # ถ้ายังรันอยู่ ให้ terminate
            if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
                print("🛑 Bottle thread still running — terminating...")
                self.processing_thread.terminate()
                self.processing_thread.wait(1000)
            self.processing_thread = None
            if hasattr(self, 'cap_processing_thread') and self.cap_processing_thread and self.cap_processing_thread.isRunning():
                print("🛑 Cap thread still running — terminating...")
                self.cap_processing_thread.terminate()
                self.cap_processing_thread.wait(1000)
            self.cap_processing_thread = None
            
            # เคลียร์ภาพและผลลัพธ์บนหน้าจอเสมอ (กันผลมาสายจาก thread ทับหลัง clear)
            try:
                if hasattr(self, 'modbus_handlers') and self.modbus_handlers:
                    self.modbus_handlers.clear_current_data()
                else:
                    self._clear_display_fallback()
            except Exception as e:
                print(f"⚠️ clear_current_data: {e}")
                self._clear_display_fallback()
            
            # อัปเดตสถานะหลังหยุด thread เสร็จ
            if hasattr(self, 'processing_status_label'):
                self.processing_status_label.setText("⏸️ ไม่มีการประมวลผล")
                self.processing_status_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 11px;")
            if hasattr(self, 'status_label'):
                self.status_label.setText('⏹️ Stopped')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            
            print("✅ Processing stopped successfully")
            
        except Exception as e:
            print(f"❌ Error stopping processing: {e}")
            self.status_label.setText(f'❌ Stop error: {str(e)}')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        finally:
            self._stopping_processing = False
    
    # Cap detection methods delegated to cap_handlers
    # Use self.cap_handlers.on_cap_processing_complete() instead
    
    def validate_cap_result(self, cap_result):
        """
        ตรวจสอบความถูกต้องของผลลัพธ์ฝาสำหรับ M100/M110/M120
        
        Args:
            cap_result: ผลลัพธ์จากการประมวลผลฝา
            
        Returns:
            bool: True ถ้าผลลัพธ์ถูกต้อง, False ถ้าไม่ถูกต้อง
        """
        try:
            print(f"🔍 CAP VALIDATION: Validating cap result for {self.current_bottle_type}")
            
            # ตรวจสอบว่ามีผลลัพธ์ OCR หรือไม่
            # ตรวจสอบใน cap_processing_results ก่อน
            ocr_results = None
            if cap_result.get('cap_processing_results'):
                # ใช้ผลลัพธ์จาก cap_processing_results
                cap_processing_results = cap_result['cap_processing_results']
                if cap_processing_results and len(cap_processing_results) > 0:
                    # ใช้ผลลัพธ์จากฝาแรก
                    first_cap = cap_processing_results[0]
                    if first_cap.get('ocr_results'):
                        ocr_results = first_cap['ocr_results']
                        print(f"🔍 CAP VALIDATION: Using cap_processing_results")
            elif cap_result.get('final_ocr_results'):
                # ใช้ผลลัพธ์จาก final_ocr_results (fallback)
                ocr_results = cap_result['final_ocr_results']
                print(f"🔍 CAP VALIDATION: Using final_ocr_results")
            elif cap_result.get('ocr_results'):
                # ใช้ผลลัพธ์จาก ocr_results (สำหรับกรณีไม่มีฝา)
                ocr_results = cap_result['ocr_results']
                print(f"🔍 CAP VALIDATION: Using ocr_results (no cap detected)")
            
            if not ocr_results:
                print("❌ CAP VALIDATION: No OCR results found")
                return False
            
            # ตรวจสอบ faded text detection status
            faded_text_status = None
            if cap_result.get('cap_processing_results'):
                cap_processing_results = cap_result['cap_processing_results']
                if cap_processing_results and len(cap_processing_results) > 0:
                    first_cap = cap_processing_results[0]
                    if first_cap.get('faded_text_result'):
                        faded_text_status = first_cap['faded_text_result'].get('status', 'unknown')
                        print(f"🔍 CAP VALIDATION: Faded text status: {faded_text_status}")
                        
                        # หยุดประมวลผลและส่ง Modbus เหมือนขวด NG (M140, M600)
                        if faded_text_status == 'faded':
                            print("❌ CAP VALIDATION: ตรวจพบข้อความจาง - ส่ง Modbus เหมือนขวด NG")
                            if hasattr(self, 'modbus_thread') and self.modbus_thread:
                                m140_ok = self.modbus_thread.on_m140()
                                if m140_ok:
                                    if hasattr(self, 'status_handlers') and self.status_handlers:
                                        self.status_handlers.update_coil_lamp("m140", True)
                                    m600_ok = self.modbus_thread.on_m600()
                                    if m600_ok and hasattr(self, 'status_handlers') and self.status_handlers:
                                        self.status_handlers.update_coil_lamp("m600", True)
                                    print("✅ ฝาจาง → M140, M600 ส่งแล้ว")
                                else:
                                    print("❌ ไม่สามารถON M140 ได้")
                            return False
            
            # ตรวจสอบว่ามีข้อความที่อ่านได้หรือไม่
            ocr_text = ""
            
            # ตรวจสอบว่า ocr_results เป็น dictionary หรือ list
            if isinstance(ocr_results, dict):
                # ถ้าเป็น dictionary ให้หา total_recognized_text หรือ text
                if 'total_recognized_text' in ocr_results and ocr_results['total_recognized_text']:
                    ocr_text = ocr_results['total_recognized_text']
                    print(f"🔍 CAP VALIDATION: Using total_recognized_text")
                elif 'text' in ocr_results and ocr_results['text']:
                    ocr_text = ocr_results['text']
                    print(f"🔍 CAP VALIDATION: Using text field")
                elif 'recognized_text' in ocr_results and ocr_results['recognized_text']:
                    ocr_text = ocr_results['recognized_text']
                    print(f"🔍 CAP VALIDATION: Using recognized_text field")
                else:
                    # ดึงเฉพาะค่าที่เป็น string สั้น (ไม่รวม array/dict/list)
                    parts = []
                    for k, v in ocr_results.items():
                        if v is None:
                            continue
                        if isinstance(v, str) and v.strip():
                            parts.append(v.strip())
                        elif isinstance(v, (int, float)) and k in ('total_lines', 'num_characters'):
                            continue  # ไม่เอาเลขพวกนี้มาเป็นข้อความ
                        elif isinstance(v, (list, dict)):
                            # ถ้าเป็น list ของ dict ให้ดึง recognized_text จากแต่ละตัว
                            if isinstance(v, list):
                                for item in v:
                                    if isinstance(item, dict):
                                        t = item.get('recognized_text') or item.get('text')
                                        if t and isinstance(t, str):
                                            parts.append(t.strip())
                    ocr_text = " ".join(parts) if parts else ""
                    if ocr_text:
                        print(f"🔍 CAP VALIDATION: Using all values (text only)")
            elif isinstance(ocr_results, list):
                # ถ้าเป็น list ให้ดึงเฉพาะข้อความจากแต่ละบรรทัด (ไม่ใช้ str() จะได้ไม่ดึง array/bbox)
                for ocr_result in ocr_results:
                    if isinstance(ocr_result, dict):
                        t = ocr_result.get('recognized_text') or ocr_result.get('text') or ocr_result.get('total_recognized_text') or ''
                        if t and isinstance(t, str):
                            ocr_text += t.strip() + " "
                    elif isinstance(ocr_result, str):
                        ocr_text += ocr_result.strip() + " "
                ocr_text = ocr_text.strip()
            else:
                ocr_text = ""
            
            ocr_text = ocr_text.strip()
            
            if not ocr_text:
                print("❌ CAP VALIDATION: No text found in OCR results")
                return False
            
            # แสดงเฉพาะข้อความสั้น (ไม่ dump structure) และจำกัดความยาวใน log
            _preview = ocr_text[:80] + ("..." if len(ocr_text) > 80 else "")
            print(f"🔍 CAP VALIDATION: OCR text: '{_preview}'")
            
            # ตรวจสอบรูปแบบวันที่/เวลาตาม bottle type
            if self.current_bottle_type == "M100":
                # M100 (ดั้งเดิม) - ตรวจสอบรูปแบบวันที่/เวลา
                return self.validate_date_time_format(ocr_text)
            elif self.current_bottle_type == "M110":
                # M110 (น้ำตาล 2%) - ตรวจสอบรูปแบบวันที่/เวลา
                return self.validate_date_time_format(ocr_text)
            elif self.current_bottle_type == "M120":
                # M120 (ผสมแมงลัก) - ตรวจสอบรูปแบบวันที่/เวลา
                return self.validate_date_time_format(ocr_text)
            
            return True
            
        except Exception as e:
            print(f"❌ CAP VALIDATION ERROR: {e}")
            return False
    
    def _compute_cap_verdict_for_display(self, cap_result):
        """คำนวณผลฝาเพื่อแสดงป้าย PASS/NG บนหน้าหลัก (ไม่ส่ง Modbus)"""
        try:
            if not cap_result or not isinstance(cap_result, dict):
                return None
            # ฝาจาง (faded) = NG (Fade)
            caps = cap_result.get('cap_processing_results') or []
            for cap in caps:
                if not isinstance(cap, dict):
                    continue
                fr = cap.get('faded_text_result')
                if fr and isinstance(fr, dict) and fr.get('status') == 'faded':
                    return 'NG_FADE'
            # กรณี single cap (ไม่มี cap_processing_results)
            fr = cap_result.get('faded_text_result')
            if fr and isinstance(fr, dict) and fr.get('status') == 'faded':
                return 'NG_FADE'
            # M100/M110/M120 ต้องผ่าน validate_cap_result
            if getattr(self, 'current_bottle_type', None) in ["M100", "M110", "M120"]:
                if not self.validate_cap_result(cap_result):
                    return 'NG'
            return 'PASS'
        except Exception:
            return None
    
    def validate_date_time_format(self, ocr_text):
        """
        ตรวจสอบรูปแบบวันที่/เวลาจากข้อความ OCR และคัดกรองวันหมดอายุตามโหมดที่เลือก
        
        Args:
            ocr_text: ข้อความที่อ่านได้จาก OCR
            
        Returns:
            bool: True ถ้ารูปแบบถูกต้องและผ่านการคัดกรอง, False ถ้าไม่ถูกต้องหรือไม่ผ่านการคัดกรอง
        """
        try:
            print(f"🔍 DATE TIME VALIDATION: Checking format for text: '{ocr_text[:50]}...'")  # แสดงแค่ 50 ตัวอักษรแรก
            
            # รูปแบบที่ยอมรับ: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',  # DD/MM/YYYY หรือ DD-MM-YYYY
                r'\d{1,2}\.\d{1,2}\.\d{4}',      # DD.MM.YYYY
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD หรือ YYYY-MM-DD
                r'\d{4}\.\d{1,2}\.\d{1,2}',      # YYYY.MM.DD
            ]
            
            # รูปแบบเวลาที่ยอมรับ: HH:MM, HH:MM:SS
            time_patterns = [
                r'\d{1,2}:\d{2}(:\d{2})?',       # HH:MM หรือ HH:MM:SS
            ]
            
            import re
            from datetime import datetime
            
            # ตรวจสอบว่ามีรูปแบบวันที่หรือไม่
            has_date = any(re.search(pattern, ocr_text) for pattern in date_patterns)
            
            # ตรวจสอบว่ามีรูปแบบเวลาหรือไม่
            has_time = any(re.search(pattern, ocr_text) for pattern in time_patterns)
            
            # ต้องมีอย่างน้อยรูปแบบวันที่หรือเวลา
            format_valid = has_date or has_time
            
            print(f"🔍 DATE TIME VALIDATION: has_date={has_date}, has_time={has_time}, format_valid={format_valid}")
            
            # ถ้ารูปแบบไม่ถูกต้อง ให้ return False
            if not format_valid:
                return False
            
            # ล้างเหตุผลฝาไม่ผ่านจากรอบก่อน (จะตั้งใหม่ถ้าไม่ผ่านเพราะวันที่)
            self._last_cap_fail_reason = None
            
            # ถ้าเป็นโหมดปกติ (ไม่คัดกรองวัน) ให้ return True
            if self.current_expiry_mode == "normal":
                print("🔍 DATE TIME VALIDATION: Normal mode - no expiry filtering")
                return True
            
            # ถ้าเป็นโหมดคัดกรองวัน ให้ตรวจสอบวันที่จากฝา (MFG/BBF ตามที่เลือก)
            if self.current_expiry_mode == "filter":
                print("🔍 DATE TIME VALIDATION: Filter mode - checking date from cap OCR")
                
                # หาวันที่จากข้อความ OCR ตามประเภทที่ user เลือก (MFG หรือ BBF)
                expiry_date = self.extract_expiry_date_from_text(ocr_text)
                if expiry_date is None:
                    print("🔍 DATE TIME VALIDATION: No date found in text")
                    return False
                
                # ตรวจสอบการคัดกรอง: ตรงกับช่วงหรือวันที่เฉพาะที่เลือกหรือไม่
                filter_result = self.check_expiry_date_filter(expiry_date)
                print(f"🔍 DATE TIME VALIDATION: Date {expiry_date} filter result: {filter_result}")
                
                if not filter_result:
                    # ไม่ตรงกับวันที่ที่เลือก → ถือว่า NG แต่ให้แสดงสถานะ "ไม่ตรงกับวันที่เลือก"
                    self._last_cap_fail_reason = "ไม่ตรงกับวันที่เลือก"
                
                return filter_result
            
            return format_valid
            
        except Exception as e:
            print(f"❌ DATE TIME VALIDATION ERROR: {e}")
            return False
    
    def _normalize_ocr_date_text(self, ocr_text):
        """
        ตัดส่วนที่ OCR อ่านเบิ้ลซ้ำ เช่น /25/25 → /25, MFG22/06/25/25 → MFG22/06/25
        """
        import re
        if not ocr_text or not isinstance(ocr_text, str):
            return ocr_text
        # ลบส่วนซ้ำแบบ /XX/XX หรือ /X/X ที่ต่อท้าย (เช่น /25/25 → /25)
        normalized = re.sub(r'(/\d{1,2})(/\1)+', r'\1', ocr_text)
        if normalized != ocr_text:
            print(f"🔍 EXPIRY NORMALIZE: '{ocr_text[:60]}...' → '{normalized[:60]}...'")
        return normalized
    
    def extract_expiry_date_from_text(self, ocr_text):
        """
        สกัดวันที่จากข้อความ OCR ตามประเภทที่ user เลือก (MFG หรือ BBF)
        จะ normalize ข้อความก่อน (ตัด /25/25 เป็น /25 ถ้า OCR อ่านเบิ้ล)
        
        Args:
            ocr_text: ข้อความที่อ่านได้จาก OCR
            
        Returns:
            datetime.date: วันที่ที่สกัดได้ หรือ None ถ้าไม่พบ
        """
        try:
            import re
            from datetime import datetime
            
            # ตัดส่วนที่อ่านซ้ำออกก่อน (เช่น MFG22/06/25/25 → MFG22/06/25)
            ocr_text = self._normalize_ocr_date_text(ocr_text or "")
            
            date_type = getattr(self, 'expiry_date_type', 'BBF').upper()
            # รูปแบบ BBF/MFG ที่ยอมรับ: BBF DD/MM/YY, MFG DD/MM/YYYY ฯลฯ
            patterns = [
                rf'{date_type}\s*(\d{{1,2}})[/-](\d{{1,2}})[/-](\d{{2,4}})',  # BBF/MFG DD/MM/YY
                rf'{date_type}\s*(\d{{1,2}})\.(\d{{1,2}})\.(\d{{2,4}})',      # BBF/MFG DD.MM.YY
            ]
            
            for pattern in patterns:
                match = re.search(pattern, ocr_text, re.IGNORECASE)
                if match:
                    day = int(match.group(1))
                    month = int(match.group(2))
                    year = int(match.group(3))
                    
                    # แปลงปี 2 หลักเป็น 4 หลัก (ถ้าปี < 50 ให้เป็น 20xx, ถ้า >= 50 ให้เป็น 19xx)
                    if year < 100:
                        if year < 50:
                            year += 2000
                        else:
                            year += 1900
                    
                    try:
                        expiry_date = datetime(year, month, day).date()
                        print(f"🔍 EXPIRY EXTRACTION: Found {date_type} date: {expiry_date}")
                        return expiry_date
                    except ValueError:
                        print(f"🔍 EXPIRY EXTRACTION: Invalid date: {day}/{month}/{year}")
                        continue
            
            print(f"🔍 EXPIRY EXTRACTION: No {date_type} date found in text")
            return None
            
        except Exception as e:
            print(f"❌ EXPIRY EXTRACTION ERROR: {e}")
            return None
    
    def check_expiry_date_filter(self, expiry_date):
        """
        ตรวจสอบวันหมดอายุตามการคัดกรองที่ตั้งค่าไว้
        
        Args:
            expiry_date: วันหมดอายุที่สกัดได้
            
        Returns:
            bool: True ถ้าผ่านการคัดกรอง, False ถ้าไม่ผ่าน
        """
        try:
            start_date, end_date, specific_date = self.get_expiry_filter_dates()
            
            if self.expiry_filter_type == "range":
                if start_date is None or end_date is None:
                    print("🔍 EXPIRY FILTER: Range dates not set")
                    return False
                
                # ตรวจสอบว่าวันหมดอายุอยู่ในช่วงที่กำหนดหรือไม่
                is_in_range = start_date <= expiry_date <= end_date
                print(f"🔍 EXPIRY FILTER: {expiry_date} in range {start_date} - {end_date}: {is_in_range}")
                return is_in_range
                
            elif self.expiry_filter_type == "specific":
                if specific_date is None:
                    print("🔍 EXPIRY FILTER: Specific date not set")
                    return False
                
                # ตรวจสอบว่าวันหมดอายุตรงกับวันที่เฉพาะหรือไม่
                is_specific = expiry_date == specific_date
                print(f"🔍 EXPIRY FILTER: {expiry_date} matches specific date {specific_date}: {is_specific}")
                return is_specific
            
            return False
            
        except Exception as e:
            print(f"❌ EXPIRY FILTER ERROR: {e}")
            return False
    
    # Modbus operation methods delegated to modbus_handlers
    # Use self.modbus_handlers.on_bottle_type_after_cap_validation() instead
    
    def clear_cap_display_for_bottle_ng(self):
        """ล้างส่วนแสดงผลฝาเมื่อขวด NG (ไม่ได้ประมวลฝาของภาพนี้)"""
        try:
            # ล้าง layout ผลฝา (แท็บฝา + หน้าหลัก)
            if hasattr(self, 'cap_results_layout') and self.cap_results_layout:
                for i in reversed(range(self.cap_results_layout.count())):
                    w = self.cap_results_layout.itemAt(i).widget()
                    if w:
                        w.setParent(None)
            if hasattr(self, 'home_cap_results_layout') and self.home_cap_results_layout:
                for i in reversed(range(self.home_cap_results_layout.count())):
                    w = self.home_cap_results_layout.itemAt(i).widget()
                    if w:
                        w.setParent(None)
            # ข้อความผลฝา
            msg = "ขวด NG — ไม่ประมวลฝา"
            if hasattr(self, 'cap_detection_text') and self.cap_detection_text:
                self.cap_detection_text.setText(msg)
            if hasattr(self, 'home_cap_detection_text') and self.home_cap_detection_text:
                self.home_cap_detection_text.setText(msg)
            # ป้ายผลฝา
            if hasattr(self, 'home_cap_verdict_label') and self.home_cap_verdict_label:
                self.home_cap_verdict_label.setText("—")
                self.home_cap_verdict_label.setStyleSheet(
                    "font-size: 22px; font-weight: bold; padding: 10px; border-radius: 8px; "
                    "background-color: #ecf0f1; color: #7f8c8d;"
                )
            # ไม่ล้างภาพฝา — ให้ผู้ใช้ยังเห็นภาพฝาปัจจุบันแม้จะไม่ได้ประมวลผล
            # แค่รีเซ็ต current_cap_result เพื่อบอกว่าไม่มีผลการตรวจฝาสำหรับภาพนี้
            self.current_cap_result = None
        except Exception as e:
            print(f"❌ clear_cap_display_for_bottle_ng: {e}")
    
    def show_image_zoom_popup(self, image, title="Image Viewer"):
        """Show image in zoomable popup window"""
        try:
            dialog = ImageZoomDialog(image, title, self)
            dialog.exec_()
        except Exception as e:
            print(f"❌ Error showing image zoom popup: {e}")
            import traceback
            traceback.print_exc()
    
    def display_cap_detection_results(self, result):
        """Display cap detection results with step-by-step images"""
        try:
            # Debug: Check result type
            print(f"🔍 DEBUG: display_cap_detection_results called with result type: {type(result)}")
            if isinstance(result, str):
                print(f"🔍 DEBUG: Result is string: {result}")
                self.cap_detection_text.setText(f"Error: Received string instead of dictionary: {result}")
                if hasattr(self, 'home_cap_detection_text'):
                    self.home_cap_detection_text.setText(f"Error: Received string instead of dictionary: {result}")
                return
            elif not isinstance(result, dict):
                print(f"🔍 DEBUG: Result is not dict: {type(result)}")
                self.cap_detection_text.setText(f"Error: Expected dictionary, got {type(result)}")
                if hasattr(self, 'home_cap_detection_text'):
                    self.home_cap_detection_text.setText(f"Error: Expected dictionary, got {type(result)}")
                return
            
            # print("🔍 DEBUG: Starting to display cap detection results")  # Reduced spam
            # print(f"🔍 DEBUG: Result keys: {list(result.keys())}")  # Reduced spam
            
            # Clear previous results
            for i in reversed(range(self.cap_results_layout.count())):
                self.cap_results_layout.itemAt(i).widget().setParent(None)
            
            # Clear Home tab cap results too
            if hasattr(self, 'home_cap_results_layout'):
                for i in reversed(range(self.home_cap_results_layout.count())):
                    widget = self.home_cap_results_layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)
            
            # print("🔍 DEBUG: Cleared previous results")  # Reduced spam
            
            # Step 1: Original Image
            # print("🔍 DEBUG: Creating Step 1 - Original Image")  # Reduced spam
            original_container = QWidget()
            original_container.setStyleSheet("border: 2px solid #2980b9; margin: 5px; padding: 5px; background-color: white;")
            original_layout = QVBoxLayout(original_container)
            
            original_title = QLabel("📸 Step 1: ภาพต้นฉบับ (Original Image)")
            original_title.setStyleSheet("font-weight: bold; color: #2980b9; font-size: 12px;")
            original_title.setAlignment(Qt.AlignCenter)
            original_layout.addWidget(original_title)
            
            # Display original image
            if result.get('image') is not None:
                # print("🔍 DEBUG: Found original image in result")  # Reduced spam
                original_image = result['image']
                if hasattr(original_image, 'shape'):
                    # print(f"🔍 DEBUG: Original image shape: {original_image.shape}")  # Reduced spam
                    # Resize for display
                    h, w = original_image.shape[:2]
                    max_size = 200
                    if h > max_size or w > max_size:
                        scale = max_size / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        original_image = cuda_resize(original_image, (new_w, new_h))
                    
                    # Convert to RGB for Qt
                    rgb_original = cuda_cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    h, w, c = rgb_original.shape
                    bytes_per_line = c * w
                    
                    qimg = QtGui.QImage(rgb_original.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qimg)
                    
                    original_label = QLabel()
                    original_label.setPixmap(pixmap)
                    original_label.setAlignment(Qt.AlignCenter)
                    original_label.setCursor(Qt.PointingHandCursor)
                    # Store original image for zoom popup
                    orig_img = result.get('image')
                    original_label.mouseDoubleClickEvent = lambda e, img=orig_img: self.show_image_zoom_popup(img, "Original Image") if img is not None else None
                    original_layout.addWidget(original_label)
                    # print("🔍 DEBUG: Added original image to layout")  # Reduced spam
                else:
                    # print("🔍 DEBUG: Original image has no shape attribute")  # Reduced spam
                    pass
            else:
                # print("🔍 DEBUG: No original image found in result")  # Reduced spam
                pass
            
            self.cap_results_layout.addWidget(original_container)
            # print("🔍 DEBUG: Added original container to cap_results_layout")  # Reduced spam
            
            # Step 2: Processed Image (with cap detection)
            if result.get('processed_image') is not None:
                # print("🔍 DEBUG: Creating Step 2 - Processed Image")  # Reduced spam
                processed_container = QWidget()
                processed_container.setStyleSheet("border: 2px solid #e67e22; margin: 5px; padding: 5px; background-color: white;")
                processed_layout = QVBoxLayout(processed_container)
                
                processed_title = QLabel("🔍 Step 2: ภาพที่ประมวลผลแล้ว (Processed Image)")
                processed_title.setStyleSheet("font-weight: bold; color: #e67e22; font-size: 12px;")
                processed_title.setAlignment(Qt.AlignCenter)
                processed_layout.addWidget(processed_title)
                
                # Display processed image
                processed_image = result['processed_image']
                if hasattr(processed_image, 'shape'):
                    # print(f"🔍 DEBUG: Processed image shape: {processed_image.shape}")  # Reduced spam
                    # Resize for display
                    h, w = processed_image.shape[:2]
                    max_size = 200
                    if h > max_size or w > max_size:
                        scale = max_size / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        processed_image = cuda_resize(processed_image, (new_w, new_h))
                    
                    # Convert to RGB for Qt
                    rgb_processed = cuda_cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    h, w, c = rgb_processed.shape
                    bytes_per_line = c * w
                    
                    qimg = QtGui.QImage(rgb_processed.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qimg)
                    
                    processed_label = QLabel()
                    processed_label.setPixmap(pixmap)
                    processed_label.setAlignment(Qt.AlignCenter)
                    processed_label.setCursor(Qt.PointingHandCursor)
                    # Store original processed image for zoom popup
                    proc_img = result.get('processed_image')
                    processed_label.mouseDoubleClickEvent = lambda e, img=proc_img: self.show_image_zoom_popup(img, "Processed Image") if img is not None else None
                    processed_layout.addWidget(processed_label)
                    # print("🔍 DEBUG: Added processed image to layout")  # Reduced spam
                else:
                    # print("🔍 DEBUG: Processed image has no shape attribute")  # Reduced spam
                    pass
                
                self.cap_results_layout.addWidget(processed_container)
                # print("🔍 DEBUG: Added processed container to cap_results_layout")  # Reduced spam
            else:
                # print("🔍 DEBUG: No processed image found in result")  # Reduced spam
                pass
            
            # Display cap detection results
            # Display cap_processing_results (แสดงผลแม้ว่าจะมี error: 'faded_text_detected')
            if result.get('cap_processing_results'):
                print(f"🔍 DEBUG: Found {len(result['cap_processing_results'])} cap processing results")
                # Track which caps we've already rendered OCR/Step3 for (บาง flow อาจมี cap_index ซ้ำใน cap_processing_results)
                rendered_ocr_caps = set()
                rendered_line_caps = set()
                # Display full pipeline results for each cap
                for cap_index, cap_result in enumerate(result['cap_processing_results']):
                        print(f"🔍 DEBUG: Cap {cap_index+1} result keys: {list(cap_result.keys())}")
                        print(f"🔍 DEBUG: Cap {cap_index+1} has line_detection_result: {'line_detection_result' in cap_result}")
                        print(f"🔍 DEBUG: Cap {cap_index+1} has ocr_results: {'ocr_results' in cap_result}")
                        if 'ocr_results' in cap_result:
                            print(f"🔍 DEBUG: Cap {cap_index+1} OCR results count: {len(cap_result['ocr_results'])}")
                        if 'line_detection_result' in cap_result:
                            line_result = cap_result['line_detection_result']
                            print(f"🔍 DEBUG: Cap {cap_index+1} line detection result type: {type(line_result)}")
                            if isinstance(line_result, dict):
                                print(f"🔍 DEBUG: Cap {cap_index+1} line detection keys: {list(line_result.keys())}")
                                _has_err = 'error' in line_result
                                print(f"🔍 DEBUG: Cap {cap_index+1} has error: {_has_err}" + (" (OK)" if not _has_err else ""))
                                if 'cropped_lines' in line_result:
                                    print(f"🔍 DEBUG: Cap {cap_index+1} cropped_lines count: {len(line_result['cropped_lines'])}")
                                    if line_result['cropped_lines']:
                                        print(f"🔍 DEBUG: Cap {cap_index+1} first cropped_line type: {type(line_result['cropped_lines'][0])}")
                                        if isinstance(line_result['cropped_lines'][0], dict):
                                            print(f"🔍 DEBUG: Cap {cap_index+1} first cropped_line keys: {list(line_result['cropped_lines'][0].keys())}")
                        
                        cap_index = cap_result['cap_index']
                        
                        # ตรวจฝาจาง (cap_fade_model.h5): Good / NG (Fade)
                        if cap_result.get('faded_text_result'):
                            faded_result = cap_result['faded_text_result']
                            status = faded_result.get('status', 'unknown')
                            result_label = faded_result.get('result', '')  # "Good" or "NG (Fade)"
                            score = faded_result.get('score')
                            num_chars = faded_result.get('num_chars', 0)
                            total_area = faded_result.get('total_area', 0)
                            normalized_area = faded_result.get('normalized_area', None)
                            
                            if status == 'faded':
                                color = "#e74c3c"
                                emoji = "⚠️"
                            elif status == 'normal':
                                color = "#27ae60"
                                emoji = "✅"
                            else:
                                color = "#95a5a6"
                                emoji = "❓"
                            
                            faded_container = QWidget()
                            # Detailed faded-cap info is already shown in OCR section above;
                            # here we only add a compact summary to the Home tab (if available).
                            if hasattr(self, 'home_cap_results_layout'):
                                home_faded_container = QWidget()
                                home_faded_container.setStyleSheet("border: 2px solid #f39c12; margin: 5px; padding: 5px; background-color: white;")
                                home_faded_layout = QVBoxLayout(home_faded_container)
                                home_faded_title = QLabel(f"{emoji} Step 2a.5: ตรวจฝาจางที่ {cap_index+1}")
                                home_faded_title.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 11px;")
                                home_faded_title.setAlignment(Qt.AlignCenter)
                                home_faded_layout.addWidget(home_faded_title)
                                home_faded_info = QLabel(f"สถานะ: {result_label or status.upper()}" + (f"\nScore: {score}" if score is not None else ""))
                                home_faded_info.setStyleSheet(f"color: {color}; font-size: 10px; padding: 5px;")
                                home_faded_info.setAlignment(Qt.AlignCenter)
                                home_faded_layout.addWidget(home_faded_info)
                                self.home_cap_results_layout.addWidget(home_faded_container)
                            
                            if 'debug_images' in faded_result and faded_result.get('debug_images'):
                                debug_images = faded_result['debug_images']
                                
                                # Create debug images container
                                debug_container = QWidget()
                                debug_container.setStyleSheet("border: 1px solid #95a5a6; margin: 5px; padding: 5px; background-color: #f8f9fa;")
                                debug_layout = QVBoxLayout(debug_container)
                                
                                debug_title = QLabel("🔍 Debug Images - Faded Text Detection")
                                debug_title.setStyleSheet("font-weight: bold; color: #34495e; font-size: 11px;")
                                debug_title.setAlignment(Qt.AlignCenter)
                                debug_layout.addWidget(debug_title)
                                
                                # Create horizontal layout for debug images
                                debug_images_layout = QHBoxLayout()
                                
                                # Display key debug images
                                debug_image_names = ['roi_show', 'gray', 'mask_circle', 'roi_masked', 'edges', 'text_edges']
                                for img_name in debug_image_names:
                                    if img_name in debug_images and debug_images[img_name] is not None:
                                        img = debug_images[img_name]
                                        
                                        # Resize for display
                                        h, w = img.shape[:2]
                                        max_size = 120
                                        if h > max_size or w > max_size:
                                            scale = max_size / max(h, w)
                                            new_w, new_h = int(w * scale), int(h * scale)
                                            img = cuda_resize(img, (new_w, new_h))
                                        
                                        # Convert to RGB for Qt
                                        if len(img.shape) == 2:  # Grayscale
                                            rgb_img = cuda_cvtColor(img, cv2.COLOR_GRAY2RGB)
                                        else:  # BGR
                                            rgb_img = cuda_cvtColor(img, cv2.COLOR_BGR2RGB)
                                        
                                        h, w, c = rgb_img.shape
                                        bytes_per_line = c * w
                                        
                                        qimg = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                                        pixmap = QtGui.QPixmap.fromImage(qimg)
                                        
                                        # Create image container
                                        img_container = QWidget()
                                        img_container.setStyleSheet("border: 1px solid #bdc3c7; margin: 2px; padding: 2px; background-color: white;")
                                        img_layout = QVBoxLayout(img_container)
                                        
                                        # Image title
                                        img_title = QLabel(img_name.replace('_', ' ').title())
                                        img_title.setStyleSheet("font-size: 9px; color: #2c3e50; font-weight: bold;")
                                        img_title.setAlignment(Qt.AlignCenter)
                                        img_layout.addWidget(img_title)
                                        
                                        # Image with double-click to zoom
                                        img_label = QLabel()
                                        img_label.setPixmap(pixmap)
                                        img_label.setAlignment(Qt.AlignCenter)
                                        img_label.setCursor(Qt.PointingHandCursor)  # Show hand cursor
                                        # Store original image for zoom popup
                                        original_img = debug_images[img_name]
                                        # Add double-click event
                                        img_label.mouseDoubleClickEvent = lambda e, img=original_img, name=img_name: self.show_image_zoom_popup(img, name)
                                        img_layout.addWidget(img_label)
                                        
                                        debug_images_layout.addWidget(img_container)
                                
                                debug_layout.addLayout(debug_images_layout)
                                self.cap_results_layout.addWidget(debug_container)
                        
                        # CRAFT rotated image ใช้ภายใน pipeline เท่านั้น ไม่ต้องแสดงซ้ำใน UI
                        if cap_result.get('craft_rotated_image') is not None:
                            pass
                        
                        # ภาพที่หมุนสำหรับ OCR (rotated_image) ถูกแสดงในส่วนรวมด้านล่างแล้ว
                        # ไม่ต้องแสดงซ้ำในบล็อก per-cap นี้
                        if cap_result.get('rotated_image') is not None:
                            pass
                        
                        # AI rotated image (ถ้ามี - จาก flow ที่ใช้ AI model ทำนายมุม)
                        # ใช้สำหรับประมวลผลภายในเท่านั้น ไม่ต้องแสดงซ้ำใน UI
                        if cap_result.get('ai_rotated_image') is not None:
                            pass
                        
                        # Build Step 3 (line detection) and Step 4 (OCR) containers first, then add in order:
                        # Step 3 must appear above Step 4 (ตรวจจับบรรทัดก่อน แล้วค่อย OCR)
                        line_container = None
                        ocr_container = None

                        # Line detection results (Step 3) - build container (แสดงครั้งเดียวต่อ cap_index เหมือน Step 4)
                        if cap_result.get('line_detection_result') and 'error' not in cap_result.get('line_detection_result', {}) and cap_index not in rendered_line_caps:
                            line_container = QWidget()
                            line_container.setStyleSheet("border: 2px solid #9b59b6; margin: 5px; padding: 5px; background-color: white;")
                            line_layout = QVBoxLayout(line_container)

                            line_title = QLabel(f"📏 Step 3: ตรวจจับบรรทัดข้อความฝาที่ {cap_index+1}")
                            line_title.setStyleSheet("font-weight: bold; color: #9b59b6; font-size: 12px;")
                            line_title.setAlignment(Qt.AlignCenter)
                            line_layout.addWidget(line_title)

                            line_result = cap_result['line_detection_result']
                            if isinstance(line_result, dict):
                                num_lines = len(line_result.get('cropped_lines', []))
                                line_info = QLabel(f"📊 จำนวนบรรทัดที่ตรวจจับได้: {num_lines}")
                                line_info.setStyleSheet("color: #2c3e50; font-size: 11px; background-color: #ecf0f1; padding: 5px;")
                                line_info.setAlignment(Qt.AlignCenter)
                                line_layout.addWidget(line_info)

                                if 'cropped_lines' in line_result and line_result['cropped_lines']:
                                    lines_title = QLabel("🖼️ ภาพบรรทัดที่ตรวจจับได้:")
                                    lines_title.setStyleSheet("font-weight: bold; color: #8e44ad; font-size: 10px;")
                                    line_layout.addWidget(lines_title)

                                    for line_idx, line_data in enumerate(line_result['cropped_lines'][:3]):  # Show max 3 lines
                                        if isinstance(line_data, dict) and 'image' in line_data:
                                            line_image = line_data['image']
                                        elif hasattr(line_data, 'shape'):
                                            line_image = line_data
                                        else:
                                            continue

                                        if line_image is not None and hasattr(line_image, 'shape'):
                                            h, w = line_image.shape[:2]
                                            max_size = 150
                                            if h > max_size or w > max_size:
                                                scale = max_size / max(h, w)
                                                new_w, new_h = int(w * scale), int(h * scale)
                                                line_image = cuda_resize(line_image, (new_w, new_h))

                                            rgb_line = cuda_cvtColor(line_image, cv2.COLOR_BGR2RGB)
                                            h, w, c = rgb_line.shape
                                            bytes_per_line = c * w

                                            qimg = QtGui.QImage(rgb_line.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                                            pixmap = QtGui.QPixmap.fromImage(qimg)

                                            line_label = QLabel()
                                            line_label.setPixmap(pixmap)
                                            line_label.setAlignment(Qt.AlignCenter)
                                            line_label.setStyleSheet("border: 1px solid #9b59b6; margin: 2px;")
                                            line_layout.addWidget(line_label)

                                    if len(line_result.get('cropped_lines', [])) > 3:
                                        more_lines = QLabel(f"... และอีก {len(line_result['cropped_lines']) - 3} บรรทัด")
                                        more_lines.setStyleSheet("color: #7f8c8d; font-size: 10px; font-style: italic;")
                                        more_lines.setAlignment(Qt.AlignCenter)
                                        line_layout.addWidget(more_lines)

                        # OCR results (Step 4) - build container (แสดงครั้งเดียวต่อ cap_index)
                        if cap_result.get('ocr_results') and cap_index not in rendered_ocr_caps:
                            ocr_container = QWidget()
                            ocr_container.setStyleSheet("border: 2px solid #f39c12; margin: 5px; padding: 5px; background-color: white;")
                            ocr_layout = QVBoxLayout(ocr_container)

                            ocr_title = QLabel(f"📝 Step 4: OCR ผลลัพธ์ฝาที่ {cap_index+1}")
                            ocr_title.setStyleSheet("font-weight: bold; color: #f39c12; font-size: 12px;")
                            ocr_title.setAlignment(Qt.AlignCenter)
                            ocr_layout.addWidget(ocr_title)

                            ocr_results = cap_result['ocr_results']

                            if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                                line_results = ocr_results['line_results']
                                total_lines = ocr_results.get('total_lines', 0)

                                summary_label = QLabel(f"📊 จำนวนบรรทัดที่อ่านได้: {total_lines}")
                                summary_label.setStyleSheet("color: #2c3e50; font-size: 11px; background-color: #ecf0f1; padding: 5px; font-weight: bold;")
                                summary_label.setAlignment(Qt.AlignCenter)
                                ocr_layout.addWidget(summary_label)

                                for j, line_result in enumerate(line_results):
                                    if isinstance(line_result, dict):
                                        recognized_text = line_result.get('recognized_text', 'N/A')
                                        confidence = line_result.get('confidence_score', 0.0)
                                        line_index = line_result.get('line_index', j)

                                        ocr_text = f"  บรรทัด {line_index+1}: '{recognized_text}' (ความเชื่อมั่น: {confidence:.3f})"

                                        ocr_label = QLabel(ocr_text)
                                        ocr_label.setStyleSheet("color: #2c3e50; font-size: 10px; background-color: #ecf0f1; padding: 2px;")
                                        ocr_label.setWordWrap(True)
                                        ocr_layout.addWidget(ocr_label)

                            if isinstance(ocr_results, dict) and 'total_recognized_text' in ocr_results:
                                total_text = ocr_results['total_recognized_text']
                                if total_text:
                                    total_label = QLabel(f"📝 ข้อความรวม: '{total_text}'")
                                    total_label.setStyleSheet("color: #27ae60; font-size: 11px; background-color: #d5f4e6; padding: 5px; font-weight: bold;")
                                    total_label.setWordWrap(True)
                                    ocr_layout.addWidget(total_label)

                            rendered_ocr_caps.add(cap_index)

                        # Add to layout in correct order: Step 3 (line detection) first, then Step 4 (OCR)
                        if line_container is not None:
                            self.cap_results_layout.addWidget(line_container)
                            rendered_line_caps.add(cap_index)
                        if ocr_container is not None:
                            self.cap_results_layout.addWidget(ocr_container)
                else:
                    # Fallback display of cropped images is no longer needed
                    pass
            else:
                # No caps detected, show text detection results
                if result.get('craft_rotated_image') is not None:
                    # Step 2a: CRAFT Rotated Image
                    craft_container = QWidget()
                    craft_container.setStyleSheet("border: 2px solid #8e44ad; margin: 5px; padding: 5px; background-color: white;")
                    craft_layout = QVBoxLayout(craft_container)
                    
                    craft_title = QLabel("🔄 Step 2a: ภาพที่ CRAFT หมุนแล้ว (CRAFT Rotated)")
                    craft_title.setStyleSheet("font-weight: bold; color: #8e44ad; font-size: 12px;")
                    craft_title.setAlignment(Qt.AlignCenter)
                    craft_layout.addWidget(craft_title)
                    
                    # Display CRAFT rotated image
                    craft_image = result['craft_rotated_image']
                    if hasattr(craft_image, 'shape'):
                        # Resize for display
                        h, w = craft_image.shape[:2]
                        max_size = 200
                        if h > max_size or w > max_size:
                            scale = max_size / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            craft_image = cuda_resize(craft_image, (new_w, new_h))
                        
                        # Convert to RGB for Qt
                        rgb_craft = cuda_cvtColor(craft_image, cv2.COLOR_BGR2RGB)
                        h, w, c = rgb_craft.shape
                        bytes_per_line = c * w
                        
                        qimg = QtGui.QImage(rgb_craft.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(qimg)
                        
                        craft_label = QLabel()
                        craft_label.setPixmap(pixmap)
                        craft_label.setAlignment(Qt.AlignCenter)
                        craft_layout.addWidget(craft_label)
                    
                    self.cap_results_layout.addWidget(craft_container)
                
                # Step 3: Cropped Image from AI Rotation
                if result.get('cropped_image') is not None:
                    cropped_container = QWidget()
                    cropped_container.setStyleSheet("border: 2px solid #f39c12; margin: 5px; padding: 5px; background-color: white;")
                    cropped_layout = QVBoxLayout(cropped_container)
                    
                    cropped_title = QLabel("✂️ Step 3: ภาพที่ครอปแล้ว (Cropped Image)")
                    cropped_title.setStyleSheet("font-weight: bold; color: #f39c12; font-size: 12px;")
                    cropped_title.setAlignment(Qt.AlignCenter)
                    cropped_layout.addWidget(cropped_title)
                    
                    # Display cropped image
                    cropped_image = result['cropped_image']
                    if hasattr(cropped_image, 'shape'):
                        # Resize for display
                        h, w = cropped_image.shape[:2]
                        max_size = 200
                        if h > max_size or w > max_size:
                            scale = max_size / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            cropped_image = cuda_resize(cropped_image, (new_w, new_h))
                        
                        # Convert to RGB for Qt
                        rgb_cropped = cuda_cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                        h, w, c = rgb_cropped.shape
                        bytes_per_line = c * w
                        
                        qimg = QtGui.QImage(rgb_cropped.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(qimg)
                        
                        cropped_label = QLabel()
                        cropped_label.setPixmap(pixmap)
                        cropped_label.setAlignment(Qt.AlignCenter)
                        cropped_layout.addWidget(cropped_label)
                    
                    self.cap_results_layout.addWidget(cropped_container)
            
            # Update cap detection text with step-by-step process
            cap_text = f"""
=== ผลการตรวจจับฝาและข้อความ ===
📊 ภาพต้นฉบับ: {result.get('image_shape', 'N/A')}

🔄 ขั้นตอนการประมวลผล:
✅ Step 1: ตรวจจับฝา - เสร็จสิ้น
✅ Step 2: ตัดภาพ - เสร็จสิ้น
✅ Step 2a.5: ตรวจสอบรอยจาง - เสร็จสิ้น
            """
            
            if result.get('cropped_images'):
                cap_text += f"\n🔍 ฝาที่ตรวจจับได้: {len(result['cropped_images'])} ฝา"
                cap_text += f"\n✅ พบฝา {len(result['cropped_images'])} ฝา"
                
                # Check if we have full pipeline results for caps
                if result.get('cap_processing_results'):
                    cap_text += f"\n\n🔄 ประมวลผลฝาแต่ละใบผ่าน Pipeline เต็ม:"
                    failed_cap_names = []  # รายการฝาที่หมุน/ประมวลผลไม่สำเร็จ สำหรับแสดงท้าย
                    for cap_result in result['cap_processing_results']:
                        cap_index = cap_result['cap_index'] + 1
                        cap_text += f"\n\n📋 ฝาที่ {cap_index}:"
                        cap_ok = True  # สมมติสำเร็จจนกว่าจะพบขั้นตอนที่ล้มเหลว
                        
                        # ตรวจฝาจาง (cap_fade_model)
                        if cap_result.get('faded_text_result'):
                            faded_result = cap_result['faded_text_result']
                            status = faded_result.get('status', 'unknown')
                            result_label = faded_result.get('result', '')
                            score = faded_result.get('score')
                            score_str = f", score: {score}" if score is not None else ""
                            if status == 'faded':
                                cap_text += f"\n  ⚠️ Step 2a.5: ตรวจฝาจาง - NG (Fade){score_str}"
                            elif status == 'normal':
                                cap_text += f"\n  ✅ Step 2a.5: ตรวจฝาจาง - Good{score_str}"
                            else:
                                cap_text += f"\n  ❓ Step 2a.5: ตรวจฝาจาง - ไม่ทราบสถานะ{score_str}"
                        
                        # CRAFT rotation
                        if cap_result.get('craft_rotated_image') is not None:
                            cap_text += f"\n  ✅ Step 2a: CRAFT rotation - สำเร็จ"
                            
                            # AI rotation
                            if cap_result.get('ai_rotated_image') is not None:
                                cap_text += f"\n  ✅ Step 2b: AI rotation - สำเร็จ"
                                
                                # Line detection
                                if cap_result.get('line_detection_result') and 'error' not in cap_result.get('line_detection_result', {}):
                                    cap_text += f"\n  ✅ Step 3: Line detection - สำเร็จ"
                                    
                                    # OCR
                                    if cap_result.get('ocr_results'):
                                        cap_text += f"\n  ✅ Step 4: OCR - สำเร็จ"
                                        
                                        ocr_results = cap_result['ocr_results']
                                        if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                                            # Handle dictionary format from recognize_text_from_craft_lines
                                            line_results = ocr_results['line_results']
                                            total_lines = ocr_results.get('total_lines', 0)
                                            cap_text += f"\n  📝 ข้อความที่อ่านได้: {total_lines} บรรทัด"
                                            
                                            for j, line_result in enumerate(line_results):
                                                if isinstance(line_result, dict):
                                                    recognized_text = line_result.get('recognized_text', 'N/A')
                                                    confidence = line_result.get('confidence_score', 0.0)
                                                    line_index = line_result.get('line_index', j)
                                                    cap_text += f"\n    บรรทัด {line_index+1}: '{recognized_text}' (ความเชื่อมั่น: {confidence:.3f})"
                                            
                                            # Show total recognized text if available
                                            if 'total_recognized_text' in ocr_results:
                                                total_text = ocr_results['total_recognized_text']
                                                if total_text:
                                                    cap_text += f"\n  📝 ข้อความรวม: '{total_text}'"
                                        else:
                                            # Handle list format (legacy)
                                            cap_text += f"\n  📝 ข้อความที่อ่านได้: {len(ocr_results)} รายการ"
                                            for j, ocr_result in enumerate(ocr_results):
                                                # Check if ocr_result is a dictionary
                                                if isinstance(ocr_result, dict):
                                                    cap_text += f"\n    {j+1}. '{ocr_result.get('text', 'N/A')}' (ความเชื่อมั่น: {ocr_result.get('confidence', 0):.3f})"
                                                else:
                                                    # If it's a string or other type, display as is
                                                    cap_text += f"\n    {j+1}. '{str(ocr_result)}'"
                                        
                                        # ตรวจว่า format_valid หรือไม่ (ฟอร์มถูกต้อง = หมุนสำเร็จสำหรับการตรวจสอบ)
                                        format_valid = cap_result.get('format_valid', False)
                                        if not format_valid:
                                            cap_ok = False
                                        cap_text += f"\n  🎯 ฝาที่ {cap_index}: ประมวลผลเสร็จสมบูรณ์"
                                    else:
                                        cap_ok = False
                                        cap_text += f"\n  ❌ Step 4: OCR - ไม่สำเร็จ"
                                        cap_text += f"\n  ⚠️ ฝาที่ {cap_index}: ประมวลผลไม่สมบูรณ์"
                                else:
                                    cap_ok = False
                                    cap_text += f"\n  ❌ Step 3: Line detection - ไม่สำเร็จ"
                                    cap_text += f"\n  ⚠️ ฝาที่ {cap_index}: ประมวลผลไม่สมบูรณ์"
                            else:
                                cap_ok = False
                                cap_text += f"\n  ❌ Step 2b: AI rotation - ไม่สำเร็จ"
                                cap_text += f"\n  ⚠️ ฝาที่ {cap_index}: ประมวลผลไม่สมบูรณ์"
                        else:
                            cap_ok = False
                            cap_text += f"\n  ❌ Step 2a: CRAFT rotation - ไม่สำเร็จ"
                            cap_text += f"\n  ⚠️ ฝาที่ {cap_index}: ประมวลผลไม่สมบูรณ์"
                        
                        if not cap_ok:
                            failed_cap_names.append(f"ฝาที่ {cap_index}")
                    
                    # แสดงรายการฝาที่หมุนไม่สำเร็จท้ายผลลัพธ์ (สำหรับการตรวจสอบ)
                    cap_text += f"\n\n📋 รายการฝาที่หมุน/ประมวลผลไม่สำเร็จ: "
                    if failed_cap_names:
                        cap_text += ", ".join(failed_cap_names)
                    else:
                        cap_text += "ไม่มี (ทุกฝาสำเร็จ)"
                    cap_text += f"\n\n🎯 ผลลัพธ์: ระบบประมวลผลฝาเสร็จสิ้น"
                else:
                    cap_text += f"\n\n🎯 ผลลัพธ์: ระบบตรวจจับฝาเสร็จสิ้น (ไม่มีการประมวลผลเพิ่มเติม)"
            else:
                cap_text += f"\n⚠️ ไม่พบฝา เริ่มประมวลผลข้อความ..."
                
                # Step 2a: CRAFT
                if result.get('craft_rotated_image') is not None:
                    cap_text += f"\n✅ Step 2a: ตรวจจับข้อความด้วย CRAFT - เสร็จสิ้น"
                    
                    # Step 2b: AI Rotation
                    if result.get('rotated_image') is not None:
                        cap_text += f"\n✅ Step 2b: ประมวลผลด้วย AI rotation - เสร็จสิ้น"
                        
                        # Step 3: Line Detection
                        if result.get('line_detection_result') and 'error' not in result.get('line_detection_result', {}):
                            cap_text += f"\n✅ Step 3: ตรวจจับบรรทัดข้อความ - เสร็จสิ้น"
                            
                            # Step 4: OCR
                            ocr_results = None  # Initialize variable
                            if result.get('final_ocr_results'):
                                cap_text += f"\n✅ Step 4: อ่านข้อความด้วย OCR - เสร็จสิ้น"
                                
                                ocr_results = result['final_ocr_results']
                                if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                                    # Handle dictionary format from recognize_text_from_craft_lines
                                    line_results = ocr_results['line_results']
                                    total_lines = ocr_results.get('total_lines', 0)
                                    cap_text += f"\n📝 ข้อความที่อ่านได้: {total_lines} บรรทัด"
                                    
                                    for i, line_result in enumerate(line_results):
                                        if isinstance(line_result, dict):
                                            recognized_text = line_result.get('recognized_text', 'N/A')
                                            confidence = line_result.get('confidence_score', 0.0)
                                            line_index = line_result.get('line_index', i)
                                            cap_text += f"\n  บรรทัด {line_index+1}: '{recognized_text}' (ความเชื่อมั่น: {confidence:.3f})"
                                    
                                    # Show total recognized text if available
                                    if 'total_recognized_text' in ocr_results:
                                        total_text = ocr_results['total_recognized_text']
                                        if total_text:
                                            cap_text += f"\n📝 ข้อความรวม: '{total_text}'"
                                elif ocr_results:
                                    # Handle list format (legacy)
                                    cap_text += f"\n📝 ข้อความที่อ่านได้: {len(ocr_results)} รายการ"
                                    for i, ocr_result in enumerate(ocr_results):
                                        # Check if ocr_result is a dictionary
                                        if isinstance(ocr_result, dict):
                                            cap_text += f"\n  {i+1}. '{ocr_result.get('text', 'N/A')}' (ความเชื่อมั่น: {ocr_result.get('confidence', 0):.3f})"
                                        else:
                                            # If it's a string or other type, display as is
                                            cap_text += f"\n  {i+1}. '{str(ocr_result)}'"
                                
                                # แสดงผลลัพธ์ OCR ที่อ่านได้สำหรับการตรวจสอบ
                                if ocr_results:
                                    cap_text += f"\n\n🔍 ผลลัพธ์ OCR สำหรับการตรวจสอบ:"
                                    
                                    # ใช้ข้อมูลจาก combined_result (retry mechanism)
                                    combined_result = result.get('combined_result', {})
                                    
                                    # แสดงมุมการหมุนที่ใช้
                                    if 'ai_rotation_angle' in combined_result:
                                        rotation_angle = combined_result['ai_rotation_angle']
                                        cap_text += f"\n🔄 มุมการหมุนที่ใช้: {rotation_angle}°"
                                    
                                    # แสดงคะแนนการหมุน
                                    if 'score' in combined_result:
                                        score = combined_result['score']
                                        cap_text += f"\n📊 คะแนนการหมุน: {score:.1f}"
                                    
                                    # แสดงจำนวนครั้งที่หมุน
                                    if 'rotation_attempt' in combined_result:
                                        attempt = combined_result['rotation_attempt']
                                        cap_text += f"\n🔄 ครั้งที่หมุน: {attempt}/5"
                                    
                                    # แสดงข้อมูลการหมุนเพิ่มเติม
                                    if 'format_valid' in combined_result:
                                        format_valid = combined_result['format_valid']
                                        cap_text += f"\n✅ ฟอร์มถูกต้อง: {'ใช่' if format_valid else 'ไม่ใช่'}"
                                    
                                    # แสดงข้อมูลการใช้ภาพจาก CRAFT
                                    if 'used_craft_image' in combined_result:
                                        cap_text += f"\n🔄 ใช้ภาพจาก: CRAFT (หมุน {combined_result.get('ai_rotation_angle', 0)}°)"
                                    else:
                                        cap_text += f"\n🔄 ใช้ภาพจาก: AI Rotation"
                                    
                                    # ตรวจสอบรูปแบบ MFG บรรทัดแรก
                                    if isinstance(ocr_results, dict) and 'line_results' in ocr_results:
                                        line_results = ocr_results['line_results']
                                        if line_results:
                                            first_line = line_results[0]
                                            first_text = first_line.get('recognized_text', '').strip()
                                            
                                            # ตรวจสอบว่าบรรทัดแรกมี MFG หรือไม่
                                            if any(prefix in first_text.upper() for prefix in ['MFG', 'BBF']):
                                                cap_text += f"\n✅ บรรทัดแรกมี MFG/BBF: '{first_text}'"
                                            else:
                                                cap_text += f"\n❌ บรรทัดแรกไม่มี MFG/BBF: '{first_text}'"
                                                cap_text += f"\n🔄 ระบบจะหมุนภาพ 5 มุมเพื่อหาฟอร์ม MFG ที่ถูกต้อง"
                                
                                if self.current_bottle_type in ["M100", "M110", "M120"]:
                                    is_cap_valid = self.validate_cap_result(result)
                                    if is_cap_valid:
                                        cap_text += f"\n✅ ฝาผ่าน - ตรงกับฟอร์มที่คาดหวัง"
                                    else:
                                        cap_text += f"\n❌ ฝาไม่ผ่าน - ไม่ตรงกับฟอร์มที่คาดหวัง"
                                else:
                                    cap_text += f"\nℹ️ ไม่ต้องตรวจสอบฟอร์ม (ไม่ใช่ M100/M110/M120)"
                                
                                cap_text += f"\n\n🎯 ผลลัพธ์: ระบบประมวลผลข้อความเสร็จสิ้น"
                            else:
                                cap_text += f"\n❌ Step 4: OCR - ไม่สำเร็จ"
                                cap_text += f"\n\n⚠️ ผลลัพธ์: ระบบประมวลผลข้อความไม่สมบูรณ์"
                        else:
                            cap_text += f"\n❌ Step 3: ตรวจจับบรรทัดข้อความ - ไม่สำเร็จ"
                            cap_text += f"\n\n⚠️ ผลลัพธ์: ระบบประมวลผลข้อความไม่สมบูรณ์"
                    else:
                        cap_text += f"\n❌ Step 2b: AI rotation - ไม่สำเร็จ"
                        cap_text += f"\n\n⚠️ ผลลัพธ์: ระบบประมวลผลข้อความไม่สมบูรณ์"
                else:
                    cap_text += f"\n❌ Step 2a: CRAFT - ไม่สำเร็จ"
                    cap_text += f"\n\n⚠️ ผลลัพธ์: ระบบประมวลผลข้อความไม่สมบูรณ์"
            
            self.cap_detection_text.setText(cap_text)
            
            # Update Home tab cap detection text + ป้ายผลฝา (PASS/NG) ให้เห็นชัด
            if hasattr(self, 'home_cap_detection_text'):
                self.home_cap_detection_text.setText(cap_text)
            if hasattr(self, 'home_cap_verdict_label'):
                cap_verdict = self._compute_cap_verdict_for_display(result)
                v = self.home_cap_verdict_label
                if cap_verdict == 'NG_FADE':
                    v.setText("⚠️  RESULT: NG (Fade)  ⚠️")
                    v.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px; border-radius: 8px; background-color: #fdebd0; color: #e67e22; border: 2px solid #e67e22;")
                elif cap_verdict == 'NG':
                    v.setText("⚠️  RESULT: NG  ⚠️")
                    v.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px; border-radius: 8px; background-color: #fadbd8; color: #c0392b; border: 2px solid #c0392b;")
                elif cap_verdict == 'PASS':
                    v.setText("✅  RESULT: PASS  ✅")
                    v.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px; border-radius: 8px; background-color: #d5f4e6; color: #27ae60; border: 2px solid #27ae60;")
                else:
                    v.setText("—")
                    v.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px; border-radius: 8px; background-color: #ecf0f1; color: #7f8c8d;")
            
        except Exception as e:
            print(f"❌ Error displaying cap detection results: {e}")
            self.cap_detection_text.setText(f"ข้อผิดพลาดในการแสดงผล: {str(e)}")
            if hasattr(self, 'home_cap_detection_text'):
                self.home_cap_detection_text.setText(f"ข้อผิดพลาดในการแสดงผล: {str(e)}")
            if hasattr(self, 'home_cap_verdict_label'):
                self.home_cap_verdict_label.setText("—")
                self.home_cap_verdict_label.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px; border-radius: 8px; background-color: #ecf0f1; color: #7f8c8d;")
    
    # Cap detection methods delegated to cap_handlers
    # Use self.cap_handlers.display_cap_processing_ui() instead
    
    def display_cap_processing_complete_ui(self):
        """Display cap processing complete UI when cap detection is finished"""
        try:
            # Don't clear previous results - just add completion status
            # for i in reversed(range(self.cap_results_layout.count())):
            #     self.cap_results_layout.itemAt(i).widget().setParent(None)
            
            # Create complete container
            complete_container = QWidget()
            complete_container.setStyleSheet("border: 2px solid #27ae60; margin: 5px; padding: 10px; background-color: #d5f4e6; border-radius: 5px;")
            complete_layout = QVBoxLayout(complete_container)
            
            # Complete title
            complete_title = QLabel("✅ การประมวลผลฝาเสร็จสิ้น")
            complete_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60; margin: 10px;")
            complete_title.setAlignment(Qt.AlignCenter)
            complete_layout.addWidget(complete_title)
            
            # Status info
            if self.current_bottle_type in ["M100", "M110", "M120"]:
                status_info = QLabel(f"📋 ฝาผ่านการตรวจสอบ - กำลังส่งสัญญาณ {self.current_bottle_type}")
                status_info.setStyleSheet("font-size: 14px; color: #27ae60; margin: 5px; font-weight: bold;")
            else:
                status_info = QLabel("📋 การประมวลผลฝาเสร็จสิ้น")
                status_info.setStyleSheet("font-size: 14px; color: #27ae60; margin: 5px; font-weight: bold;")
            status_info.setAlignment(Qt.AlignCenter)
            complete_layout.addWidget(status_info)
            
            # System status
            system_status = QLabel("✅ ระบบพร้อมส่งสัญญาณ Modbus\n✅ การประมวลผลเสร็จสิ้นแล้ว\n✅ GUI แสดงผลครบถ้วน")
            system_status.setStyleSheet("font-size: 12px; color: #27ae60; margin: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 3px;")
            system_status.setAlignment(Qt.AlignCenter)
            complete_layout.addWidget(system_status)
            
            self.cap_results_layout.addWidget(complete_container)
            
            # Update cap detection text
            if hasattr(self, 'cap_detection_text') and self.cap_detection_text:
                if self.current_bottle_type in ["M100", "M110", "M120"]:
                    self.cap_detection_text.setText(f"✅ การประมวลผลฝาเสร็จสิ้น - กำลังส่งสัญญาณ {self.current_bottle_type}")
                    if hasattr(self, 'home_cap_detection_text'):
                        self.home_cap_detection_text.setText(f"✅ การประมวลผลฝาเสร็จสิ้น - กำลังส่งสัญญาณ {self.current_bottle_type}")
                else:
                    self.cap_detection_text.setText("✅ การประมวลผลฝาเสร็จสิ้น")
                self.cap_detection_text.setStyleSheet("""
                    QTextEdit {
                        background-color: #d5f4e6;
                        border: 2px solid #27ae60;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 12px;
                        color: #2c3e50;
                    }
                """)
            
            print("✅ CAP PROCESSING COMPLETE UI: Displayed complete state for cap detection")
            
        except Exception as e:
            print(f"❌ Error displaying cap processing complete UI: {e}")
    
    # Modbus operation methods delegated to modbus_handlers
    # Use self.modbus_handlers.on_taste_mode_changed() instead
    # Use self.modbus_handlers.show_taste_selection() instead
    # Use self.modbus_handlers.hide_taste_selection() instead
    
    def update_selected_tastes(self):
        """Update selected tastes based on checkboxes"""
        try:
            self.selected_tastes = []
            
            # Check control panel checkboxes only
            if hasattr(self, 'taste_m100') and self.taste_m100.isChecked():
                self.selected_tastes.append("M100")
            if hasattr(self, 'taste_m110') and self.taste_m110.isChecked():
                self.selected_tastes.append("M110")
            if hasattr(self, 'taste_m120') and self.taste_m120.isChecked():
                self.selected_tastes.append("M120")
            
            # Validate selection count
            if self.current_taste_mode == 1 and len(self.selected_tastes) != 1:
                if len(self.selected_tastes) > 1:
                    QMessageBox.warning(self, "คำเตือน", "กรุณาเลือกรสชาติเพียง 1 รสชาติ")
                    # Uncheck the last checked checkbox
                    if hasattr(self, 'taste_m100') and self.taste_m100.isChecked():
                        self.taste_m100.setChecked(False)
                    elif hasattr(self, 'taste_m110') and self.taste_m110.isChecked():
                        self.taste_m110.setChecked(False)
                    elif hasattr(self, 'taste_m120') and self.taste_m120.isChecked():
                        self.taste_m120.setChecked(False)
                    self.update_selected_tastes()  # Recursive call
                    return
            elif self.current_taste_mode == 2 and len(self.selected_tastes) != 2:
                if len(self.selected_tastes) > 2:
                    QMessageBox.warning(self, "คำเตือน", "กรุณาเลือกรสชาติเพียง 2 รสชาติ")
                    # Uncheck the last checked checkbox
                    if hasattr(self, 'taste_m100') and self.taste_m100.isChecked():
                        self.taste_m100.setChecked(False)
                    elif hasattr(self, 'taste_m110') and self.taste_m110.isChecked():
                        self.taste_m110.setChecked(False)
                    elif hasattr(self, 'taste_m120') and self.taste_m120.isChecked():
                        self.taste_m120.setChecked(False)
                    self.update_selected_tastes()  # Recursive call
                    return
            
            # Helper to map internal taste codes to display names
            def _format_taste_names(codes):
                name_map = {
                    "M100": "Original",
                    "M110": "Less sugar 2%",
                    "M120": "Basil seed mix",
                }
                return [name_map.get(c, c) for c in codes]
            
            # Update status and send signals
            if self.current_taste_mode == 1:
                display_names = _format_taste_names(self.selected_tastes)
                self.taste_mode_status.setText(f"โหมดปัจจุบัน: 1 รสชาติ ({', '.join(display_names)})")
                
                # Send appropriate signal for 1-taste mode
                if len(self.selected_tastes) == 1:
                    self.send_one_taste_signal()
            elif self.current_taste_mode == 2:
                display_names = _format_taste_names(self.selected_tastes)
                self.taste_mode_status.setText(f"โหมดปัจจุบัน: 2 รสชาติ ({', '.join(display_names)})")
                
                # Send appropriate signal for 2-taste mode
                if len(self.selected_tastes) == 2:
                    self.send_two_taste_signal()
            
            print(f"🔄 TASTE SELECTION: Updated selected tastes: {self.selected_tastes}")
            
        except Exception as e:
            print(f"❌ Error in update_selected_tastes: {e}")
    
    def send_two_taste_signal(self):
        """Send appropriate signal for 2-taste mode selection"""
        try:
            if not hasattr(self, 'modbus_thread') or not self.modbus_thread:
                print("❌ TASTE SIGNAL: Modbus thread not available")
                return
            
            print("🔄 TASTE SIGNAL: Sending 2-taste combination to D9002")
            
            # Determine which value to write to D9002 based on selected tastes
            if "M100" in self.selected_tastes and "M110" in self.selected_tastes:
                # ขวดดั้งเดิม + ขวดน้ำตาล 2% → D9002 = 1
                success = self.modbus_thread.write_register(9002, 1)
                if success:
                    print("🏷️ D9002 = 1 (ขวดดั้งเดิม + ขวดน้ำตาล 2%)")
                else:
                    print("❌ ไม่สามารถเขียน D9002 = 1 ได้")
                    
            elif "M100" in self.selected_tastes and "M120" in self.selected_tastes:
                # ขวดดั้งเดิม + ขวดผสมแมงลัก → D9002 = 2
                success = self.modbus_thread.write_register(9002, 2)
                if success:
                    print("🏷️ D9002 = 2 (ขวดดั้งเดิม + ขวดผสมแมงลัก)")
                else:
                    print("❌ ไม่สามารถเขียน D9002 = 2 ได้")
                    
            elif "M110" in self.selected_tastes and "M120" in self.selected_tastes:
                # ขวดน้ำตาล 2% + ขวดผสมแมงลัก → D9002 = 3
                success = self.modbus_thread.write_register(9002, 3)
                if success:
                    print("🏷️ D9002 = 3 (ขวดน้ำตาล 2% + ขวดผสมแมงลัก)")
                else:
                    print("❌ ไม่สามารถเขียน D9002 = 3 ได้")
                    
            elif "M110" in self.selected_tastes and "M100" in self.selected_tastes:
                # ขวดน้ำตาล 2% + ขวดดั้งเดิม → D9002 = 4
                success = self.modbus_thread.write_register(9002, 4)
                if success:
                    print("🏷️ D9002 = 4 (ขวดน้ำตาล 2% + ขวดดั้งเดิม)")
                else:
                    print("❌ ไม่สามารถเขียน D9002 = 4 ได้")
                    
            elif "M120" in self.selected_tastes and "M110" in self.selected_tastes:
                # ขวดผสมแมงลัก + ขวดน้ำตาล 2% → D9002 = 5
                success = self.modbus_thread.write_register(9002, 5)
                if success:
                    print("🏷️ D9002 = 5 (ขวดผสมแมงลัก + ขวดน้ำตาล 2%)")
                else:
                    print("❌ ไม่สามารถเขียน D9002 = 5 ได้")
                    
            elif "M120" in self.selected_tastes and "M100" in self.selected_tastes:
                # ขวดผสมแมงลัก + ขวดดั้งเดิม → D9002 = 6
                success = self.modbus_thread.write_register(9002, 6)
                if success:
                    print("🏷️ D9002 = 6 (ขวดผสมแมงลัก + ขวดดั้งเดิม)")
                else:
                    print("❌ ไม่สามารถเขียน D9002 = 6 ได้")
            
        except Exception as e:
            print(f"❌ Error in send_two_taste_signal: {e}")
    
    def send_one_taste_signal(self):
        """Send appropriate signal for 1-taste mode selection"""
        try:
            if not hasattr(self, 'modbus_thread') or not self.modbus_thread:
                print("❌ TASTE SIGNAL: Modbus thread not available")
                return
            
            print("🔄 TASTE SIGNAL: Sending 1-taste selection to D7007")
            
            # Determine which value to write to D7007 based on selected taste
            if "M100" in self.selected_tastes:
                # ขวดดั้งเดิม → D7007 = 1
                success = self.modbus_thread.write_register(7007, 1)
                if success:
                    print("🏷️ D7007 = 1 (ขวดดั้งเดิม)")
                else:
                    print("❌ ไม่สามารถเขียน D7007 = 1 ได้")
                    
            elif "M110" in self.selected_tastes:
                # ขวดน้ำตาล 2% → D7007 = 2
                success = self.modbus_thread.write_register(7007, 2)
                if success:
                    print("🏷️ D7007 = 2 (ขวดน้ำตาล 2%)")
                else:
                    print("❌ ไม่สามารถเขียน D7007 = 2 ได้")
                    
            elif "M120" in self.selected_tastes:
                # ขวดผสมแมงลัก → D7007 = 3
                success = self.modbus_thread.write_register(7007, 3)
                if success:
                    print("🏷️ D7007 = 3 (ขวดผสมแมงลัก)")
                else:
                    print("❌ ไม่สามารถเขียน D7007 = 3 ได้")
            
        except Exception as e:
            print(f"❌ Error in send_one_taste_signal: {e}")
    
    def on_taste_mode_checkbox_changed(self):
        """Handle taste mode checkbox change - update combo box"""
        try:
            # Update combo box to match checkbox selection
            if hasattr(self, 'taste_mode_combo'):
                self.taste_mode_combo.blockSignals(True)
                if self.taste_mode_1.isChecked():
                    self.taste_mode_combo.setCurrentIndex(0)
                elif self.taste_mode_2.isChecked():
                    self.taste_mode_combo.setCurrentIndex(1)
                elif self.taste_mode_3.isChecked():
                    self.taste_mode_combo.setCurrentIndex(2)
                self.taste_mode_combo.blockSignals(False)
            
            # Call the original modbus handler
            self.modbus_handlers.on_taste_mode_changed()
            
        except Exception as e:
            print(f"❌ Error in on_taste_mode_checkbox_changed: {e}")
    
    def on_taste_mode_combo_changed(self, index):
        """Handle taste mode combo box change"""
        try:
            # index: 0 = 1 taste, 1 = 2 taste, 2 = 3 taste
            new_mode = index + 1
            
            # Update checkboxes to match combo selection
            if hasattr(self, 'taste_mode_1') and hasattr(self, 'taste_mode_2') and hasattr(self, 'taste_mode_3'):
                self.taste_mode_1.blockSignals(True)
                self.taste_mode_2.blockSignals(True)
                self.taste_mode_3.blockSignals(True)
                
                self.taste_mode_1.setChecked(new_mode == 1)
                self.taste_mode_2.setChecked(new_mode == 2)
                self.taste_mode_3.setChecked(new_mode == 3)
                
                self.taste_mode_1.blockSignals(False)
                self.taste_mode_2.blockSignals(False)
                self.taste_mode_3.blockSignals(False)
            
            # Update current taste mode and process the change
            if new_mode != self.current_taste_mode:
                self.current_taste_mode = new_mode
                
                # Reset all taste mode signals first
                if hasattr(self, 'modbus_thread') and self.modbus_thread:
                    self.modbus_thread.reset_m720()
                    self.modbus_thread.reset_m721()
                    self.modbus_thread.reset_m722()
                    self.modbus_thread.reset_m730()
                    self.modbus_thread.reset_m731()
                    self.modbus_thread.reset_m732()
                    self.modbus_thread.reset_m733()
                    self.modbus_thread.reset_m734()
                    self.modbus_thread.reset_m735()
                    self.modbus_thread.reset_m740()
                    self.modbus_thread.reset_m741()
                    self.modbus_thread.reset_m742()
                    
                    # Reset lamp status
                    self.status_handlers.update_coil_lamp("m720", False)
                    self.status_handlers.update_coil_lamp("m721", False)
                    self.status_handlers.update_coil_lamp("m722", False)
                    self.status_handlers.update_coil_lamp("m730", False)
                    self.status_handlers.update_coil_lamp("m731", False)
                    self.status_handlers.update_coil_lamp("m732", False)
                    self.status_handlers.update_coil_lamp("m733", False)
                    self.status_handlers.update_coil_lamp("m734", False)
                    self.status_handlers.update_coil_lamp("m735", False)
                    self.status_handlers.update_coil_lamp("m740", False)
                    self.status_handlers.update_coil_lamp("m741", False)
                    self.status_handlers.update_coil_lamp("m742", False)
                
                # Process mode change
                if new_mode == 1:
                    self.modbus_handlers.show_taste_selection()
                    if hasattr(self, 'taste_mode_status'):
                        self.taste_mode_status.setText("โหมดปัจจุบัน: 1 รสชาติ - กรุณาเลือกรสชาติ")
                        self.taste_mode_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px; font-weight: bold;")
                    if hasattr(self, 'modbus_thread') and self.modbus_thread:
                        self.modbus_thread.write_register(9006, 30)
                elif new_mode == 2:
                    self.modbus_handlers.show_taste_selection()
                    if hasattr(self, 'taste_mode_status'):
                        self.taste_mode_status.setText("โหมดปัจจุบัน: 2 รสชาติ - กรุณาเลือกรสชาติ")
                        self.taste_mode_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px; font-weight: bold;")
                    if hasattr(self, 'modbus_thread') and self.modbus_thread:
                        self.modbus_thread.write_register(9006, 20)
                elif new_mode == 3:
                    self.modbus_handlers.hide_taste_selection()
                    self.selected_tastes = ["M100", "M110", "M120"]
                    if hasattr(self, 'taste_mode_status'):
                        self.taste_mode_status.setText("โหมดปัจจุบัน: 3 รสชาติ (Original, Less sugar 2%, Basil seed mix)")
                        self.taste_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px; font-weight: bold;")
                    if hasattr(self, 'modbus_thread') and self.modbus_thread:
                        self.modbus_thread.write_register(9006, 10)
            
            print(f"🔄 TASTE MODE: Changed to {new_mode} taste mode via combo box")
            
        except Exception as e:
            print(f"❌ Error in on_taste_mode_combo_changed: {e}")
    
    def on_expiry_mode_combo_changed(self, index):
        """Handle expiry mode combo box change"""
        try:
            # index: 0 = normal, 1 = filter
            if index == 0:
                # Normal mode
                if hasattr(self, 'expiry_mode_normal') and hasattr(self, 'expiry_mode_filter'):
                    self.expiry_mode_normal.blockSignals(True)
                    self.expiry_mode_filter.blockSignals(True)
                    self.expiry_mode_normal.setChecked(True)
                    self.expiry_mode_filter.setChecked(False)
                    self.expiry_mode_normal.blockSignals(False)
                    self.expiry_mode_filter.blockSignals(False)
                
                self.current_expiry_mode = "normal"
                self.hide_expiry_filter_controls()
                if hasattr(self, 'expiry_mode_status'):
                    self.expiry_mode_status.setText("โหมดปัจจุบัน: ปกติ (ไม่คัดกรองวันหมดอายุ)")
                    self.expiry_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px; font-weight: bold;")
            else:
                # Filter mode
                if hasattr(self, 'expiry_mode_normal') and hasattr(self, 'expiry_mode_filter'):
                    self.expiry_mode_normal.blockSignals(True)
                    self.expiry_mode_filter.blockSignals(True)
                    self.expiry_mode_normal.setChecked(False)
                    self.expiry_mode_filter.setChecked(True)
                    self.expiry_mode_normal.blockSignals(False)
                    self.expiry_mode_filter.blockSignals(False)
                
                self.current_expiry_mode = "filter"
                self.show_expiry_filter_controls()
                self.update_expiry_mode_status()
            
            print(f"🔄 EXPIRY MODE: Changed to {self.current_expiry_mode} mode via combo box")
            
        except Exception as e:
            print(f"❌ Error in on_expiry_mode_combo_changed: {e}")
    
    def on_expiry_mode_changed(self):
        """Handle expiry mode selection change - ทำงานแบบ radio button (ถ้ากดช่องใหม่ ช่องเก่าจะถูกยกเลิกอัตโนมัติ)"""
        try:
            # ตรวจสอบว่า checkbox ไหนถูก check
            sender = self.sender()
            
            # ถ้า checkbox ถูก uncheck (ไม่ใช่ check) ให้ข้าม
            if sender and not sender.isChecked():
                return
            
            # Update combo box to match checkbox selection
            if hasattr(self, 'expiry_mode_combo'):
                self.expiry_mode_combo.blockSignals(True)
                if sender == self.expiry_mode_normal:
                    self.expiry_mode_combo.setCurrentIndex(0)
                elif sender == self.expiry_mode_filter:
                    self.expiry_mode_combo.setCurrentIndex(1)
                self.expiry_mode_combo.blockSignals(False)
            
            # ตรวจสอบว่า checkbox ไหนถูก check และยกเลิกช่องอื่นๆ
            if sender == self.expiry_mode_normal:
                # ยกเลิกช่องอื่นๆ
                self.expiry_mode_filter.blockSignals(True)
                self.expiry_mode_filter.setChecked(False)
                self.expiry_mode_filter.blockSignals(False)
                
                # ตรวจสอบว่า checkbox ถูก check หรือไม่
                if not self.expiry_mode_normal.isChecked():
                    self.expiry_mode_normal.blockSignals(True)
                    self.expiry_mode_normal.setChecked(True)
                    self.expiry_mode_normal.blockSignals(False)
                
                self.current_expiry_mode = "normal"
                self.hide_expiry_filter_controls()
                self.expiry_mode_status.setText("โหมดปัจจุบัน: ปกติ (ไม่คัดกรองวันหมดอายุ)")
                self.expiry_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px; font-weight: bold;")
                
            elif sender == self.expiry_mode_filter:
                # ยกเลิกช่องอื่นๆ
                self.expiry_mode_normal.blockSignals(True)
                self.expiry_mode_normal.setChecked(False)
                self.expiry_mode_normal.blockSignals(False)
                
                # ตรวจสอบว่า checkbox ถูก check หรือไม่
                if not self.expiry_mode_filter.isChecked():
                    self.expiry_mode_filter.blockSignals(True)
                    self.expiry_mode_filter.setChecked(True)
                    self.expiry_mode_filter.blockSignals(False)
                
                self.current_expiry_mode = "filter"
                self.show_expiry_filter_controls()
                self.update_expiry_mode_status()
            else:
                return  # ไม่ใช่ checkbox ที่เราต้องการ
            
            print(f"🔄 EXPIRY MODE: Changed to {self.current_expiry_mode} mode")
            
        except Exception as e:
            print(f"❌ Error in on_expiry_mode_changed: {e}")
    
    def on_expiry_filter_type_changed(self):
        """Handle expiry filter type selection change - ทำงานแบบ radio button (ถ้ากดช่องใหม่ ช่องเก่าจะถูกยกเลิกอัตโนมัติ)"""
        try:
            # ตรวจสอบว่า checkbox ไหนถูก check
            sender = self.sender()
            
            # ถ้า checkbox ถูก uncheck (ไม่ใช่ check) ให้ข้าม
            if sender and not sender.isChecked():
                return
            
            # ตรวจสอบว่า checkbox ไหนถูก check และยกเลิกช่องอื่นๆ
            if sender == self.expiry_filter_range:
                # ยกเลิกช่องอื่นๆ
                self.expiry_filter_specific.blockSignals(True)
                self.expiry_filter_specific.setChecked(False)
                self.expiry_filter_specific.blockSignals(False)
                
                # ตรวจสอบว่า checkbox ถูก check หรือไม่
                if not self.expiry_filter_range.isChecked():
                    self.expiry_filter_range.blockSignals(True)
                    self.expiry_filter_range.setChecked(True)
                    self.expiry_filter_range.blockSignals(False)
                
                self.expiry_filter_type = "range"
                self.show_range_date_selection()
                self.hide_specific_date_selection()
                
            elif sender == self.expiry_filter_specific:
                # ยกเลิกช่องอื่นๆ
                self.expiry_filter_range.blockSignals(True)
                self.expiry_filter_range.setChecked(False)
                self.expiry_filter_range.blockSignals(False)
                
                # ตรวจสอบว่า checkbox ถูก check หรือไม่
                if not self.expiry_filter_specific.isChecked():
                    self.expiry_filter_specific.blockSignals(True)
                    self.expiry_filter_specific.setChecked(True)
                    self.expiry_filter_specific.blockSignals(False)
                
                self.expiry_filter_type = "specific"
                self.hide_range_date_selection()
                self.show_specific_date_selection()
            else:
                return  # ไม่ใช่ checkbox ที่เราต้องการ
            
            self.update_expiry_mode_status()
            print(f"🔄 EXPIRY FILTER TYPE: Changed to {self.expiry_filter_type}")
            
        except Exception as e:
            print(f"❌ Error in on_expiry_filter_type_changed: {e}")
    
    def on_expiry_date_type_changed(self, index):
        """Handle ใช้วันที่ (MFG/BBF) combo change"""
        try:
            if hasattr(self, 'expiry_date_type_combo') and self.expiry_date_type_combo is not None:
                self.expiry_date_type = self.expiry_date_type_combo.currentData() or "BBF"
                self.update_expiry_mode_status()
                print(f"🔄 EXPIRY DATE TYPE: ใช้วันที่ {self.expiry_date_type}")
        except Exception as e:
            print(f"❌ Error in on_expiry_date_type_changed: {e}")
    
    def show_expiry_filter_controls(self):
        """Show expiry filter controls"""
        try:
            # Show filter type selection
            for i in range(self.expiry_filter_type_layout.count()):
                widget = self.expiry_filter_type_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
            
            # Show date type (MFG/BBF) และ date selection ตาม filter type
            expiry_date_type_layout = getattr(self, 'expiry_date_type_layout', None)
            if expiry_date_type_layout is not None:
                for i in range(expiry_date_type_layout.count()):
                    w = expiry_date_type_layout.itemAt(i).widget()
                    if w:
                        w.setVisible(True)
            if self.expiry_filter_type == "range":
                self.show_range_date_selection()
                self.hide_specific_date_selection()
            else:
                self.hide_range_date_selection()
                self.show_specific_date_selection()
            
            print("🔄 EXPIRY FILTER: Showing expiry filter controls")
            
        except Exception as e:
            print(f"❌ Error in show_expiry_filter_controls: {e}")
    
    def hide_expiry_filter_controls(self):
        """Hide expiry filter controls"""
        try:
            # Hide filter type selection
            for i in range(self.expiry_filter_type_layout.count()):
                widget = self.expiry_filter_type_layout.itemAt(i).widget()
                if widget and isinstance(widget, QCheckBox):
                    widget.setVisible(False)
                elif widget and isinstance(widget, QLabel) and "ประเภทการกรอง" in widget.text():
                    widget.setVisible(False)
            
            # Hide date type (MFG/BBF) และ date selections
            expiry_date_type_layout = getattr(self, 'expiry_date_type_layout', None)
            if expiry_date_type_layout is not None:
                for i in range(expiry_date_type_layout.count()):
                    w = expiry_date_type_layout.itemAt(i).widget()
                    if w:
                        w.setVisible(False)
            self.hide_range_date_selection()
            self.hide_specific_date_selection()
            
            print("🔄 EXPIRY FILTER: Hiding expiry filter controls")
            
        except Exception as e:
            print(f"❌ Error in hide_expiry_filter_controls: {e}")
    
    def show_range_date_selection(self):
        """Show range date selection controls"""
        try:
            for i in range(self.range_date_layout.count()):
                widget = self.range_date_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
            
            print("🔄 EXPIRY FILTER: Showing range date selection")
            
        except Exception as e:
            print(f"❌ Error in show_range_date_selection: {e}")
    
    def hide_range_date_selection(self):
        """Hide range date selection controls"""
        try:
            for i in range(self.range_date_layout.count()):
                widget = self.range_date_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)
            
            print("🔄 EXPIRY FILTER: Hiding range date selection")
            
        except Exception as e:
            print(f"❌ Error in hide_range_date_selection: {e}")
    
    def show_specific_date_selection(self):
        """Show specific date selection controls"""
        try:
            for i in range(self.specific_date_layout.count()):
                widget = self.specific_date_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
            
            print("🔄 EXPIRY FILTER: Showing specific date selection")
            
        except Exception as e:
            print(f"❌ Error in show_specific_date_selection: {e}")
    
    def hide_specific_date_selection(self):
        """Hide specific date selection controls"""
        try:
            for i in range(self.specific_date_layout.count()):
                widget = self.specific_date_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)
            
            print("🔄 EXPIRY FILTER: Hiding specific date selection")
            
        except Exception as e:
            print(f"❌ Error in hide_specific_date_selection: {e}")
    
    def update_expiry_mode_status(self):
        """Update expiry mode status label"""
        try:
            if self.current_expiry_mode == "normal":
                self.expiry_mode_status.setText("โหมดปัจจุบัน: ปกติ (ไม่คัดกรองวันหมดอายุ)")
                self.expiry_mode_status.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px; font-weight: bold;")
            elif self.current_expiry_mode == "filter":
                date_type = getattr(self, 'expiry_date_type', 'BBF')
                if self.expiry_filter_type == "range":
                    start_date = self.expiry_start_date_edit.date().toString("dd/MM/yyyy")
                    end_date = self.expiry_end_date_edit.date().toString("dd/MM/yyyy")
                    self.expiry_mode_status.setText(f"โหมดปัจจุบัน: คัดกรองช่วงวันที่ ({start_date} - {end_date}) ใช้วันที่ {date_type}")
                else:  # specific
                    specific_date = self.expiry_specific_date_edit.date().toString("dd/MM/yyyy")
                    self.expiry_mode_status.setText(f"โหมดปัจจุบัน: คัดกรองวันที่เฉพาะ ({specific_date}) ใช้วันที่ {date_type}")
                self.expiry_mode_status.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px; font-weight: bold;")
            
        except Exception as e:
            print(f"❌ Error in update_expiry_mode_status: {e}")
    
    def get_expiry_filter_dates(self):
        """Get expiry filter dates for validation"""
        try:
            if self.current_expiry_mode == "normal":
                return None, None, None
            
            if self.expiry_filter_type == "range":
                start_date = self.expiry_start_date_edit.date().toPyDate()
                end_date = self.expiry_end_date_edit.date().toPyDate()
                return start_date, end_date, None
            else:  # specific
                specific_date = self.expiry_specific_date_edit.date().toPyDate()
                return None, None, specific_date
                
        except Exception as e:
            print(f"❌ Error in get_expiry_filter_dates: {e}")
            return None, None, None
    
    def select_ocr_test_image(self):
        """Select image for OCR testing"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "เลือกภาพสำหรับทดสอบ OCR", 
                "", 
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
            )
            
            if file_path:
                print(f"🔤 OCR TEST: Selected image: {file_path}")
                
                # Load and display image
                image = cv2.imread(file_path)
                if image is not None:
                    self.ocr_test_image = image
                    self.ocr_test_image_path = file_path
                    
                    # Display image
                    self.display_ocr_test_image(image)
                    
                    # Enable process buttons
                    self.btn_process_ocr.setEnabled(True)
                    self.btn_process_ocr_enhanced.setEnabled(True)
                    self.btn_process_ocr_easyocr.setEnabled(True)
                    self.btn_process_ocr_tesseract.setEnabled(True)
                    
                    # Update status
                    self.ocr_status_label.setText(f'✅ เลือกภาพแล้ว: {os.path.basename(file_path)}')
                    self.ocr_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
                    
                    # Clear previous results
                    self.ocr_results_text.clear()
                    
                    print(f"🔤 OCR TEST: Image loaded successfully, shape: {image.shape}")
                else:
                    QMessageBox.warning(self, "ข้อผิดพลาด", "ไม่สามารถโหลดภาพได้")
                    print(f"❌ OCR TEST: Failed to load image: {file_path}")
            else:
                print("🔤 OCR TEST: No image selected")
                
        except Exception as e:
            print(f"❌ Error in select_ocr_test_image: {e}")
            QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเลือกภาพ: {str(e)}")
    
    def display_ocr_test_image(self, image):
        """Display image in OCR test tab"""
        try:
            # Resize image for display
            h, w = image.shape[:2]
            max_size = 400
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                display_image = cuda_resize(image, (new_w, new_h))
            else:
                display_image = image
            
            # Convert to RGB for Qt
            rgb_image = cuda_cvtColor(display_image, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_image.shape
            bytes_per_line = c * w
            
            qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            
            self.ocr_image_label.setPixmap(pixmap)
            self.ocr_image_label.setAlignment(Qt.AlignCenter)
            
        except Exception as e:
            print(f"❌ Error in display_ocr_test_image: {e}")
    
    def process_ocr_test(self):
        """Process OCR on selected image"""
        if self.ocr_test_image is None:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาเลือกภาพก่อน")
            return
        
        try:
            print("🔤 OCR TEST: Starting OCR processing...")
            
            # Update UI
            self.btn_process_ocr.setEnabled(False)
            self.ocr_progress_bar.setVisible(True)
            self.ocr_progress_bar.setValue(0)
            self.ocr_status_label.setText('🔄 Processing OCR...')
            self.ocr_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            
            # Clear previous results
            self.ocr_results_text.clear()
            
            # Process OCR
            self.ocr_progress_bar.setValue(20)
            QtWidgets.QApplication.processEvents()
            
            # Use DeepOCR for text recognition
            if self.ocr_model is not None:
                print("🔤 OCR TEST: Using DeepOCR for text recognition...")
                
                self.ocr_progress_bar.setValue(40)
                QtWidgets.QApplication.processEvents()
                
                # Preprocess image for better number recognition
                processed_image = self.preprocess_image_for_ocr(self.ocr_test_image)
                
                # Perform OCR using DeepOCR
                ocr_result = self.ocr_model.recognize_text_from_image_array(
                    processed_image, 
                    self.ocr_test_image_path
                )
                
                self.ocr_progress_bar.setValue(80)
                QtWidgets.QApplication.processEvents()
                
                # Process results
                if 'error' not in ocr_result and ocr_result.get('recognized_text'):
                    recognized_text = ocr_result['recognized_text']
                    confidence_score = ocr_result.get('confidence_score', 0.0)
                    
                    print(f"🔤 OCR TEST: OCR completed successfully")
                    print(f"🔤 OCR TEST: Recognized text: '{recognized_text}'")
                    print(f"🔤 OCR TEST: Confidence: {confidence_score:.3f}")
                    
                    # Format results
                    result_text = f"🔤 ผลลัพธ์ OCR (DeepOCR)\n"
                    result_text += f"📊 สถานะ: สำเร็จ\n"
                    result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                    result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n\n"
                    
                    result_text += "📝 ข้อความที่อ่านได้:\n"
                    result_text += "=" * 50 + "\n"
                    result_text += f"'{recognized_text}'\n\n"
                    
                    result_text += "📊 สถิติ:\n"
                    result_text += "=" * 50 + "\n"
                    result_text += f"ความเชื่อมั่น: {confidence_score:.3f}\n"
                    result_text += f"จำนวนตัวอักษร: {len(recognized_text)}\n"
                    result_text += f"จำนวนคำ: {len(recognized_text.split())}\n"
                    
                    # Show additional info if available
                    if 'processing_time' in ocr_result:
                        result_text += f"เวลาประมวลผล: {ocr_result['processing_time']:.2f} วินาที\n"
                    
                    if 'model_info' in ocr_result:
                        result_text += f"โมเดล: {ocr_result['model_info']}\n"
                    
                else:
                    error_msg = ocr_result.get('error', 'ไม่สามารถอ่านข้อความได้')
                    result_text = f"❌ ไม่พบข้อความในภาพ\n\n"
                    result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                    result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n"
                    result_text += f"❌ ข้อผิดพลาด: {error_msg}\n"
                    print(f"🔤 OCR TEST: OCR failed - {error_msg}")
                
                # Display results
                self.ocr_results_text.setText(result_text)
                
            else:
                result_text = "❌ ข้อผิดพลาด: DeepOCR ไม่พร้อมใช้งาน\n\n"
                result_text += "กรุณาตรวจสอบการติดตั้ง DeepOCR model"
                self.ocr_results_text.setText(result_text)
                print("❌ OCR TEST: DeepOCR not available")
            
            self.ocr_progress_bar.setValue(100)
            QtWidgets.QApplication.processEvents()
            
            # Update UI
            self.btn_process_ocr.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('✅ ประมวลผล OCR เสร็จสิ้น')
            self.ocr_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            
            print("🔤 OCR TEST: OCR processing completed")
            
        except Exception as e:
            print(f"❌ Error in process_ocr_test: {e}")
            
            # Reset UI
            self.btn_process_ocr.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('❌ เกิดข้อผิดพลาดในการประมวลผล OCR')
            self.ocr_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            
            # Show error in results
            error_text = f"❌ ข้อผิดพลาดในการประมวลผล OCR\n\n"
            error_text += f"รายละเอียด: {str(e)}\n\n"
            error_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path) if self.ocr_test_image_path else 'ไม่ระบุ'}\n"
            self.ocr_results_text.setText(error_text)
    
    def process_ocr_test_enhanced(self):
        """
        ประมวลผล OCR แบบปรับปรุงสำหรับการอ่านเลข
        """
        if self.ocr_test_image is None:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาเลือกภาพก่อน")
            return
        
        try:
            print("🔤 OCR TEST ENHANCED: Starting enhanced OCR processing...")
            
            # Update UI
            self.btn_process_ocr.setEnabled(False)
            self.btn_process_ocr_enhanced.setEnabled(False)
            self.ocr_progress_bar.setVisible(True)
            self.ocr_progress_bar.setValue(0)
            self.ocr_status_label.setText('🔄 Processing OCR (enhanced)...')
            self.ocr_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            
            # Clear previous results
            self.ocr_results_text.clear()
            
            # Process OCR
            self.ocr_progress_bar.setValue(20)
            QtWidgets.QApplication.processEvents()
            
            # Use DeepOCR with enhanced preprocessing
            if self.ocr_model is not None:
                print("🔤 OCR TEST ENHANCED: Using DeepOCR with enhanced preprocessing...")
                
                self.ocr_progress_bar.setValue(40)
                QtWidgets.QApplication.processEvents()
                
                # Enhanced preprocessing for better number recognition
                processed_image = self.preprocess_image_for_ocr(self.ocr_test_image)
                
                # Additional preprocessing specifically for numbers
                enhanced_image = self.enhance_image_for_numbers(processed_image)
                
                self.ocr_progress_bar.setValue(60)
                QtWidgets.QApplication.processEvents()
                
                # Perform OCR using DeepOCR
                ocr_result = self.ocr_model.recognize_text_from_image_array(
                    enhanced_image, 
                    self.ocr_test_image_path
                )
                
                self.ocr_progress_bar.setValue(80)
                QtWidgets.QApplication.processEvents()
                
                # Process results
                if 'error' not in ocr_result and ocr_result.get('recognized_text'):
                    recognized_text = ocr_result['recognized_text']
                    confidence_score = ocr_result.get('confidence_score', 0.0)
                    
                    print(f"🔤 OCR TEST ENHANCED: OCR completed successfully")
                    print(f"🔤 OCR TEST ENHANCED: Recognized text: '{recognized_text}'")
                    print(f"🔤 OCR TEST ENHANCED: Confidence: {confidence_score:.3f}")
                    
                    # Format results
                    result_text = f"🔤 ผลลัพธ์ OCR แบบปรับปรุง (DeepOCR)\n"
                    result_text += f"📊 สถานะ: สำเร็จ\n"
                    result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                    result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n"
                    result_text += f"🔧 การปรับปรุง: เพิ่มขนาดภาพ, ปรับความคมชัด, CLAHE\n\n"
                    
                    result_text += "📝 ข้อความที่อ่านได้:\n"
                    result_text += "=" * 50 + "\n"
                    result_text += f"'{recognized_text}'\n\n"
                    
                    result_text += "📊 สถิติ:\n"
                    result_text += "=" * 50 + "\n"
                    result_text += f"ความเชื่อมั่น: {confidence_score:.3f}\n"
                    result_text += f"จำนวนตัวอักษร: {len(recognized_text)}\n"
                    result_text += f"จำนวนคำ: {len(recognized_text.split())}\n"
                    
                    # Show additional info if available
                    if 'processing_time' in ocr_result:
                        result_text += f"เวลาประมวลผล: {ocr_result['processing_time']:.2f} วินาที\n"
                    
                    if 'model_info' in ocr_result:
                        result_text += f"โมเดล: {ocr_result['model_info']}\n"
                    
                else:
                    error_msg = ocr_result.get('error', 'ไม่สามารถอ่านข้อความได้')
                    result_text = f"❌ ไม่พบข้อความในภาพ\n\n"
                    result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                    result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n"
                    result_text += f"❌ ข้อผิดพลาด: {error_msg}\n"
                    print(f"🔤 OCR TEST ENHANCED: OCR failed - {error_msg}")
                
                # Display results
                self.ocr_results_text.setText(result_text)
                
            else:
                result_text = "❌ ข้อผิดพลาด: DeepOCR ไม่พร้อมใช้งาน\n\n"
                result_text += "กรุณาตรวจสอบการติดตั้ง DeepOCR model"
                self.ocr_results_text.setText(result_text)
                print("❌ OCR TEST ENHANCED: DeepOCR not available")
            
            self.ocr_progress_bar.setValue(100)
            QtWidgets.QApplication.processEvents()
            
            # Update UI
            self.btn_process_ocr.setEnabled(True)
            self.btn_process_ocr_enhanced.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('✅ ประมวลผล OCR แบบปรับปรุงเสร็จสิ้น')
            self.ocr_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            
            print("🔤 OCR TEST ENHANCED: Enhanced OCR processing completed")
            
        except Exception as e:
            print(f"❌ Error in process_ocr_test_enhanced: {e}")
            
            # Reset UI
            self.btn_process_ocr.setEnabled(True)
            self.btn_process_ocr_enhanced.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('❌ เกิดข้อผิดพลาดในการประมวลผล OCR แบบปรับปรุง')
            self.ocr_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            
            # Show error in results
            error_text = f"❌ ข้อผิดพลาดในการประมวลผล OCR แบบปรับปรุง\n\n"
            error_text += f"รายละเอียด: {str(e)}\n\n"
            error_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path) if self.ocr_test_image_path else 'ไม่ระบุ'}\n"
            self.ocr_results_text.setText(error_text)
    
    def enhance_image_for_numbers(self, image):
        """
        ปรับปรุงภาพเพิ่มเติมสำหรับการอ่านเลข
        """
        try:
            if image is None:
                return image
            
            print("🔧 Enhancing image specifically for number recognition...")
            
            # แปลงเป็น grayscale
            if len(image.shape) == 3:
                gray = cuda_cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # เพิ่มขนาดภาพอีกครั้งเพื่อให้เลขชัดขึ้น
            height, width = gray.shape
            scale_factor = 1.5  # เพิ่มขนาดอีก 1.5 เท่า
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize ด้วย INTER_LANCZOS4 เพื่อให้ภาพชัดที่สุด
            resized = cuda_resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # ปรับปรุงความคมชัดด้วย unsharp mask
            blurred = cuda_gaussianBlur(resized, (0, 0), 1.5)
            sharpened = cv2.addWeighted(resized, 2.0, blurred, -1.0, 0)
            
            # ปรับปรุงความคมชัดด้วย CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened)
            
            # ปรับปรุงความคมชัดด้วย morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # ปรับปรุงความคมชัดด้วย bilateral filter
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # แปลงกลับเป็น BGR format สำหรับ DeepOCR
            if len(image.shape) == 3:
                result = cuda_cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                result = enhanced
            
            print(f"✅ Image enhanced for numbers: {width}x{height} -> {new_width}x{new_height}")
            return result
            
        except Exception as e:
            print(f"❌ Error enhancing image for numbers: {e}")
            return image
            
    def process_ocr_easyocr(self):
        """Process OCR using EasyOCR (English only)"""
        if self.ocr_test_image is None:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาเลือกภาพก่อน")
            return
        
        try:
            print("📖 OCR TEST EASYOCR: Starting EasyOCR processing (English)...")
            
            # Update UI
            self.btn_process_ocr_easyocr.setEnabled(False)
            self.ocr_progress_bar.setVisible(True)
            self.ocr_progress_bar.setValue(0)
            self.ocr_status_label.setText('🔄 Processing EasyOCR (EN)...')
            self.ocr_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            
            # Clear previous results
            self.ocr_results_text.clear()
            
            # Import EasyOCR
            try:
                import easyocr
                import torch
                CUDA_AVAILABLE = torch.cuda.is_available()
            except ImportError:
                result_text = "❌ ข้อผิดพลาด: EasyOCR ไม่ได้ติดตั้ง\n\n"
                result_text += "กรุณาติดตั้งด้วยคำสั่ง: pip install easyocr"
                self.ocr_results_text.setText(result_text)
                self.btn_process_ocr_easyocr.setEnabled(True)
                self.ocr_progress_bar.setVisible(False)
                return
            
            self.ocr_progress_bar.setValue(20)
            QtWidgets.QApplication.processEvents()
            
            # Initialize EasyOCR reader (English only)
            print("📖 Initializing EasyOCR reader (English)...")
            reader = easyocr.Reader(['en'], gpu=CUDA_AVAILABLE)
            
            self.ocr_progress_bar.setValue(40)
            QtWidgets.QApplication.processEvents()
            
            # Perform OCR
            print("📖 Performing OCR with EasyOCR...")
            start_time = time.time()
            results = reader.readtext(self.ocr_test_image)
            processing_time = time.time() - start_time
            
            self.ocr_progress_bar.setValue(80)
            QtWidgets.QApplication.processEvents()
            
            # Process results
            if results:
                recognized_texts = []
                confidences = []
                
                for (bbox, text, confidence) in results:
                    recognized_texts.append(text)
                    confidences.append(confidence)
                
                full_text = ' '.join(recognized_texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                print(f"📖 EasyOCR completed successfully")
                print(f"📖 Recognized text: '{full_text}'")
                print(f"📖 Average confidence: {avg_confidence:.3f}")
                
                # Format results
                result_text = f"📖 ผลลัพธ์ EasyOCR (English)\n"
                result_text += f"📊 สถานะ: สำเร็จ\n"
                result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n\n"
                
                result_text += "📝 ข้อความที่อ่านได้:\n"
                result_text += "=" * 50 + "\n"
                result_text += f"'{full_text}'\n\n"
                
                result_text += "📊 สถิติ:\n"
                result_text += "=" * 50 + "\n"
                result_text += f"ความเชื่อมั่นเฉลี่ย: {avg_confidence:.3f}\n"
                result_text += f"จำนวนข้อความที่พบ: {len(results)}\n"
                result_text += f"เวลาประมวลผล: {processing_time:.2f} วินาที\n"
                result_text += f"ใช้ GPU: {'ใช่' if CUDA_AVAILABLE else 'ไม่'}\n\n"
                
                result_text += "📋 รายละเอียดแต่ละข้อความ:\n"
                result_text += "=" * 50 + "\n"
                for i, (bbox, text, confidence) in enumerate(results, 1):
                    result_text += f"{i}. '{text}' (ความเชื่อมั่น: {confidence:.3f})\n"
                
            else:
                result_text = f"❌ ไม่พบข้อความในภาพ\n\n"
                result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n"
                result_text += f"เวลาประมวลผล: {processing_time:.2f} วินาที\n"
                print("📖 EasyOCR: No text detected")
            
            # Display results
            self.ocr_results_text.setText(result_text)
            
            self.ocr_progress_bar.setValue(100)
            QtWidgets.QApplication.processEvents()
            
            # Update UI
            self.btn_process_ocr_easyocr.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('✅ ประมวลผล EasyOCR เสร็จสิ้น')
            self.ocr_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            
            print("📖 OCR TEST EASYOCR: Processing completed")
            
        except Exception as e:
            print(f"❌ Error in process_ocr_easyocr: {e}")
            import traceback
            traceback.print_exc()
            
            # Reset UI
            self.btn_process_ocr_easyocr.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('❌ เกิดข้อผิดพลาดในการประมวลผล EasyOCR')
            self.ocr_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            
            # Show error in results
            error_text = f"❌ ข้อผิดพลาดในการประมวลผล EasyOCR\n\n"
            error_text += f"รายละเอียด: {str(e)}\n\n"
            error_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path) if self.ocr_test_image_path else 'ไม่ระบุ'}\n"
            self.ocr_results_text.setText(error_text)
    
    def process_ocr_tesseract(self):
        """Process OCR using Tesseract OCR (English only)"""
        if self.ocr_test_image is None:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาเลือกภาพก่อน")
            return
        
        try:
            print("🔍 OCR TEST TESSERACT: Starting Tesseract OCR processing (English)...")
            
            # Update UI
            self.btn_process_ocr_tesseract.setEnabled(False)
            self.ocr_progress_bar.setVisible(True)
            self.ocr_progress_bar.setValue(0)
            self.ocr_status_label.setText('🔄 Processing Tesseract OCR (EN)...')
            self.ocr_status_label.setStyleSheet("color: #f39c12; padding: 5px; font-size: 11px;")
            
            # Clear previous results
            self.ocr_results_text.clear()
            
            # Import pytesseract
            try:
                import pytesseract
            except ImportError:
                result_text = "❌ ข้อผิดพลาด: pytesseract ไม่ได้ติดตั้ง\n\n"
                result_text += "กรุณาติดตั้งด้วยคำสั่ง: pip install pytesseract\n"
                result_text += "และติดตั้ง Tesseract OCR:\n"
                result_text += "- Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                result_text += "- Windows: ดาวน์โหลดจาก https://github.com/UB-Mannheim/tesseract/wiki"
                self.ocr_results_text.setText(result_text)
                self.btn_process_ocr_tesseract.setEnabled(True)
                self.ocr_progress_bar.setVisible(False)
                return
            
            self.ocr_progress_bar.setValue(20)
            QtWidgets.QApplication.processEvents()
            
            # Preprocess image for better OCR results
            print("🔍 Preprocessing image for Tesseract OCR...")
            if len(self.ocr_test_image.shape) == 3:
                gray = cuda_cvtColor(self.ocr_test_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.ocr_test_image.copy()
            
            # Apply thresholding for better text recognition
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            self.ocr_progress_bar.setValue(40)
            QtWidgets.QApplication.processEvents()
            
            # Perform OCR with English language
            print("🔍 Performing OCR with Tesseract (English)...")
            start_time = time.time()
            
            # Get detailed data including confidence scores
            ocr_data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
            full_text = pytesseract.image_to_string(thresh, lang='eng')
            processing_time = time.time() - start_time
            
            self.ocr_progress_bar.setValue(80)
            QtWidgets.QApplication.processEvents()
            
            # Process results
            if full_text.strip():
                # Extract confidence scores
                confidences = []
                texts = []
                for i in range(len(ocr_data['text'])):
                    if int(ocr_data['conf'][i]) > 0:
                        texts.append(ocr_data['text'][i])
                        confidences.append(int(ocr_data['conf'][i]))
                
                avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
                word_count = len([t for t in texts if t.strip()])
                
                print(f"🔍 Tesseract OCR completed successfully")
                print(f"🔍 Recognized text: '{full_text.strip()}'")
                print(f"🔍 Average confidence: {avg_confidence:.3f}")
                
                # Format results
                result_text = f"🔍 ผลลัพธ์ Tesseract OCR (English)\n"
                result_text += f"📊 สถานะ: สำเร็จ\n"
                result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n\n"
                
                result_text += "📝 ข้อความที่อ่านได้:\n"
                result_text += "=" * 50 + "\n"
                result_text += f"'{full_text.strip()}'\n\n"
                
                result_text += "📊 สถิติ:\n"
                result_text += "=" * 50 + "\n"
                result_text += f"ความเชื่อมั่นเฉลี่ย: {avg_confidence:.3f}\n"
                result_text += f"จำนวนคำที่พบ: {word_count}\n"
                result_text += f"จำนวนตัวอักษร: {len(full_text.strip())}\n"
                result_text += f"เวลาประมวลผล: {processing_time:.2f} วินาที\n\n"
                
                # Show word-level details if available
                if texts:
                    result_text += "📋 รายละเอียดคำที่พบ:\n"
                    result_text += "=" * 50 + "\n"
                    word_idx = 0
                    for i in range(len(ocr_data['text'])):
                        if int(ocr_data['conf'][i]) > 0 and ocr_data['text'][i].strip():
                            word_idx += 1
                            conf = int(ocr_data['conf'][i]) / 100.0
                            result_text += f"{word_idx}. '{ocr_data['text'][i]}' (ความเชื่อมั่น: {conf:.3f})\n"
                            if word_idx >= 20:  # Limit to first 20 words
                                result_text += "... (แสดงเฉพาะ 20 คำแรก)\n"
                                break
                
            else:
                result_text = f"❌ ไม่พบข้อความในภาพ\n\n"
                result_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path)}\n"
                result_text += f"📏 ขนาดภาพ: {self.ocr_test_image.shape[1]}x{self.ocr_test_image.shape[0]}\n"
                result_text += f"เวลาประมวลผล: {processing_time:.2f} วินาที\n"
                print("🔍 Tesseract OCR: No text detected")
            
            # Display results
            self.ocr_results_text.setText(result_text)
            
            self.ocr_progress_bar.setValue(100)
            QtWidgets.QApplication.processEvents()
            
            # Update UI
            self.btn_process_ocr_tesseract.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('✅ ประมวลผล Tesseract OCR เสร็จสิ้น')
            self.ocr_status_label.setStyleSheet("color: #27ae60; padding: 5px; font-size: 11px;")
            
            print("🔍 OCR TEST TESSERACT: Processing completed")
            
        except Exception as e:
            print(f"❌ Error in process_ocr_tesseract: {e}")
            import traceback
            traceback.print_exc()
            
            # Reset UI
            self.btn_process_ocr_tesseract.setEnabled(True)
            self.ocr_progress_bar.setVisible(False)
            self.ocr_status_label.setText('❌ เกิดข้อผิดพลาดในการประมวลผล Tesseract OCR')
            self.ocr_status_label.setStyleSheet("color: #e74c3c; padding: 5px; font-size: 11px;")
            
            # Show error in results
            error_text = f"❌ ข้อผิดพลาดในการประมวลผล Tesseract OCR\n\n"
            error_text += f"รายละเอียด: {str(e)}\n\n"
            error_text += f"📁 ไฟล์: {os.path.basename(self.ocr_test_image_path) if self.ocr_test_image_path else 'ไม่ระบุ'}\n"
            if "tesseract" in str(e).lower() or "not found" in str(e).lower():
                error_text += "\n💡 หมายเหตุ: ตรวจสอบว่าได้ติดตั้ง Tesseract OCR แล้ว\n"
                error_text += "   Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                error_text += "   Windows: ดาวน์โหลดจาก https://github.com/UB-Mannheim/tesseract/wiki"
            self.ocr_results_text.setText(error_text)
    
    def on_capture_limit_changed(self, value):
        """อัปเดตจำนวนรอบเมื่อเปลี่ยนค่าในช่อง"""
        if self.modbus_thread and self.modbus_thread.capture_with_limit_mode:
            self.modbus_thread.capture_limit_count = value
            print(f"🔄 เปลี่ยนจำนวนรอบการถ่ายเป็น: {value} ครั้ง")
    
    def on_start_pressed(self):
        """ปุ่ม START แบบ momentary: กดค้าง = M700 ON"""
        self._stopping_processing = False
        if self.modbus_thread and self.modbus_thread.write_coil(700, True):
            self.status_handlers.update_coil_lamp("m700", True)
            self.status_label.setText('▶️ Hold START (M700 ON)')
            self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
            print("✅ START (กด): M700 ON")
        elif self.modbus_thread is None:
            self.status_label.setText('❌ Modbus not ready')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")

    def on_start_released(self):
        """ปุ่ม START แบบ momentary: ปล่อย = M700 OFF"""
        if self.modbus_thread and self.modbus_thread.write_coil(700, False):
            self.status_handlers.update_coil_lamp("m700", False)
            self.status_label.setText('▶️ Release START (M700 OFF)')
            print("✅ START (ปล่อย): M700 OFF")

    def on_stop_pressed(self):
        """ปุ่ม STOP แบบ momentary: กดค้าง = M701 ON + เคลียร์ผลบนหน้าจอ"""
        self._clear_display_on_stop()
        if self.modbus_thread and getattr(self.modbus_thread, 'capture_only_mode', False):
            if hasattr(self.modbus_thread, 'reset_id5_capture_session'):
                self.modbus_thread.reset_id5_capture_session()
        if self.modbus_thread and self.modbus_thread.write_coil(701, True):
            self.status_handlers.update_coil_lamp("m701", True)
            self.status_label.setText('⏹️ Hold STOP (M701 ON)')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            print("✅ STOP (กด): M701 ON")
        elif self.modbus_thread is None:
            self.status_label.setText('❌ Modbus not ready')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")

    def on_stop_released(self):
        """ปุ่ม STOP แบบ momentary: ปล่อย = M701 OFF"""
        if self.modbus_thread and self.modbus_thread.write_coil(701, False):
            self.status_handlers.update_coil_lamp("m701", False)
            self.status_label.setText('⏹️ Release STOP (M701 OFF)')
            print("✅ STOP (ปล่อย): M701 OFF")

    def start_system(self):
        """Start system by turning ON M700 (ใช้จากที่อื่น ถ้าต้องการ toggle; ปุ่มบน GUI ใช้ momentary)"""
        self._stopping_processing = False
        if self.modbus_thread:
            success_m700 = self.modbus_thread.on_m700()
            if success_m700:
                self.status_label.setText('▶️ System started (M700 ON)')
                self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
                self.status_handlers.update_coil_lamp("m700", True)
                print("✅ START: M700 ON - ระบบเริ่มทำงาน")
            else:
                self.status_label.setText('❌ Failed to start (M700 not ON)')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        else:
            self.status_label.setText('❌ Modbus not ready')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")

    def stop_system(self):
        """Stop system (ใช้จากที่อื่น เช่น M513; ปุ่มบน GUI ใช้ momentary)"""
        self._clear_display_on_stop()
        if self.modbus_thread:
            success_m701 = self.modbus_thread.on_m701()
            if success_m701:
                self.status_label.setText('⏹️ System stopped (M701 ON - wait 1.5s)')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                self.status_handlers.update_coil_lamp("m700", False)
                self.status_handlers.update_coil_lamp("m701", True)
                print("✅ STOP: M701 ON - ระบบหยุดทำงาน (รอ 1.5 วินาที)")
                ok = self.modbus_thread.reset_all_bottle_types() and self.modbus_thread.reset_m140()
                if ok:
                    print("✅ STOP: OFF M100, M110, M120, M130, M140 (ล้างผลตรวจ)")
                QTimer.singleShot(1500, self.modbus_handlers.reset_to_initial_state)
            else:
                self.status_label.setText('❌ Failed to stop (M701 not ON)')
                self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
                print("❌ STOP: ไม่สามารถ ON M701 ได้")
        else:
            self.status_label.setText('❌ Modbus not ready')
            self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
            print("❌ STOP: Modbus thread ไม่พร้อมใช้งาน")
    
    def update_stop_status(self):
        """Update status after M701 reset"""
        self.status_label.setText('⏹️ System stopped (M701 Reset)')
        self.status_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        print("✅ STOP: M701 Reset - ระบบหยุดทำงานเสร็จสิ้น")
    
    def on_reset_pressed(self):
        """Handle Reset button pressed - write 1 to D5012"""
        if self.modbus_thread:
            success = self.modbus_thread.write_register(5012, 1)
            if success:
                print("✅ RESET: D5012 = 1 (กดปุ่ม)")
            else:
                print("❌ RESET: ไม่สามารถเขียน D5012 = 1 ได้")
        else:
            print("❌ RESET: Modbus thread ไม่พร้อมใช้งาน")
    
    def on_reset_released(self):
        """Handle Reset button released - write 0 to D5012"""
        if self.modbus_thread:
            success = self.modbus_thread.write_register(5012, 0)
            if success:
                print("✅ RESET: D5012 = 0 (ปล่อยปุ่ม)")
            else:
                print("❌ RESET: ไม่สามารถเขียน D5012 = 0 ได้")
        else:
            print("❌ RESET: Modbus thread ไม่พร้อมใช้งาน")
    
    def update_light_m76_status(self, is_on):
        """อัปเดตปุ่มไฟ (M76) ตามค่าที่อ่านจาก Modbus — รู้ว่าไฟ ON หรือ OFF"""
        if not hasattr(self, 'btn_light_m76') or self.btn_light_m76 is None:
            return
        self.btn_light_m76.blockSignals(True)
        self.btn_light_m76.setChecked(bool(is_on))
        self.btn_light_m76.setText('💡 Light (M76) On' if is_on else '💡 Light (M76) Off')
        self.btn_light_m76.blockSignals(False)

    def on_toggle_light_m76(self):
        """Toggle ไฟ (M76) เปิด/ปิด - ส่ง write_coil(76, on/off)"""
        if not hasattr(self, 'modbus_thread') or self.modbus_thread is None:
            QMessageBox.warning(self, "ข้อผิดพลาด", "Modbus ยังไม่ได้เชื่อมต่อ")
            self.btn_light_m76.setChecked(not self.btn_light_m76.isChecked())  # revert toggle
            return
        is_on = self.btn_light_m76.isChecked()
        success = self.modbus_thread.write_coil(76, is_on)
        if success:
            if is_on:
                self.btn_light_m76.setText('💡 Light (M76) On')
                print("✅ ไฟ (M76): เปิด")
            else:
                self.btn_light_m76.setText('💡 Light (M76) Off')
                print("✅ ไฟ (M76): ปิด")
        else:
            self.btn_light_m76.setChecked(not is_on)  # revert on failure
            self.btn_light_m76.setText('💡 Light (M76) Off' if not is_on else '💡 Light (M76) On')
            QMessageBox.warning(self, "ข้อผิดพลาด", "ส่งคำสั่ง M76 ไม่ได้")
    
    # Modbus operation methods delegated to modbus_handlers
    # Use self.modbus_handlers.reset_to_initial_state() instead
    
    # Modbus operation helper methods delegated to modbus_handlers
    # Use self.modbus_handlers.reset_modbus_to_initial_state() instead
    # Use self.modbus_handlers.reset_ui_to_initial_state() instead
    # Use self.modbus_handlers.clear_current_data() instead
    # Use self.modbus_handlers.reset_progress_bars() instead
    # Use self.modbus_handlers.reset_camera_states() instead
    # Use self.modbus_handlers.reset_lamp_status() instead
    # Use self.modbus_handlers.reset_queue_count() instead
            
    # Bottle detection methods delegated to bottle_handlers
    # Use self.bottle_handlers.display_single_results() instead
    # Use self.bottle_handlers.display_result_image() instead
    
    # Gripper methods delegated to modbus_handlers
    # Use self.modbus_handlers.open_gripper() instead
    # Use self.modbus_handlers.close_gripper() instead
    # Use self.modbus_handlers.reset_gripper_open() instead
    # Use self.modbus_handlers.reset_gripper_close() instead
    
    # Motor reverse methods delegated to modbus_handlers
    # Use self.modbus_handlers.on_motor_reverse_press() instead
    # Use self.modbus_handlers.on_motor_reverse_release() instead
    
    def check_if_jetson(self):
        """Check if running on Jetson device"""
        try:
            # ตรวจสอบ Jetson โดยดูจาก system info
            import platform
            system_info = platform.platform().lower()
            
            # ตรวจสอบ Jetson-specific indicators
            if 'jetson' in system_info or 'tegra' in system_info:
                return True
            
            # ตรวจสอบ ARM architecture (Jetson ใช้ ARM)
            if platform.machine().startswith('aarch64') or platform.machine().startswith('arm'):
                return True
            
            # ตรวจสอบ OS (Jetson ใช้ Linux)
            if platform.system().lower() == 'linux':
                # ตรวจสอบ GPU info
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                    if 'Jetson' in result.stdout or 'Tegra' in result.stdout:
                        return True
                except:
                    pass
            
            return False
        except:
            return False
    
    def setup_jetson_shortcuts(self):
        """Setup keyboard shortcuts for Jetson fullscreen operation"""
        try:
            # Fullscreen toggle (F11)
            fullscreen_shortcut = QtWidgets.QShortcut(Qt.Key_F11, self)
            fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
            
            # Exit fullscreen (Escape)
            exit_fullscreen_shortcut = QtWidgets.QShortcut(Qt.Key_Escape, self)
            exit_fullscreen_shortcut.activated.connect(self.exit_fullscreen)
            
            # Exit application (Ctrl+Q)
            exit_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self)
            exit_shortcut.activated.connect(self.close)
            
            print("✅ Jetson keyboard shortcuts configured")
            print("   F11: Toggle fullscreen")
            print("   Escape: Exit fullscreen")
            print("   Ctrl+Q: Exit application")
            
        except Exception as e:
            print(f"⚠️ Could not setup Jetson shortcuts: {e}")
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and normal window mode"""
        if self.windowState() == Qt.WindowFullScreen:
            self.showNormal()
            print("🖥️ Switched to normal window mode")
        else:
            self.showFullScreen()
            print("🖥️ Switched to fullscreen mode")
    
    def exit_fullscreen(self):
        """Exit fullscreen mode"""
        if self.windowState() == Qt.WindowFullScreen:
            self.showNormal()
            print("🖥️ Exited fullscreen mode")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Restore original stdout/stderr before closing
        if hasattr(self, 'original_stdout') and self.original_stdout:
            sys.stdout = self.original_stdout
        if hasattr(self, 'original_stderr') and self.original_stderr:
            sys.stderr = self.original_stderr
        
        # Stop status timer
        if hasattr(self, 'status_timer') and self.status_timer.isActive():
            print("🔄 CLOSE EVENT: Stopping status timer...")
            self.status_timer.stop()
        
        # Stop Modbus thread
        if self.modbus_thread:
            self.modbus_thread.stop()
            self.modbus_thread.wait(1000)
        
        # Close USB camera
        if self.usb_camera:
            self.usb_camera.close_camera()
        
        # Close Sentech camera
        if self.sentech_camera:
            self.sentech_camera.cleanup()
        
        # Stop Sentech service
        if hasattr(self, 'sentech_service_process') and self.sentech_service_process:
            print("🔄 CLOSE EVENT: Stopping Sentech service...")
            try:
                self.sentech_service_process.terminate()
                self.sentech_service_process.wait(timeout=5)
            except:
                try:
                    self.sentech_service_process.kill()
                except:
                    pass
        

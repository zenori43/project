# -*- coding: utf-8 -*-
"""
UI ทดสอบระบบหมุนฝา: เลือกภาพหลายไฟล์, ประมวลผลทีละภาพ, ดูแต่ละ Step, แท็บประวัติ.
โปรแกรมเลือกกรอบ CRAFT อัตโนมัติ → หาขอบที่ยาวที่สุด → วัดมุมเทียบแนวนอน → หมุนให้สี่เหลี่ยมตรงแนวนอน (ไม่ต้องเลือกจุดเอง).

รันจากโฟลเดอร์ final:
  python test_cap_rotation_ui.py
"""

import sys
import os
from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))
os.chdir(FINAL_DIR)

try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    pass

import cv2
import numpy as np
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QListWidget, QListWidgetItem, QLabel, QPushButton,
    QFileDialog, QGroupBox, QScrollArea, QFrame, QProgressBar,
    QMessageBox, QSplitter, QComboBox, QGridLayout, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def numpy_bgr_to_qpixmap(bgr, max_width=800, max_height=600):
    """Convert OpenCV BGR numpy to QPixmap, optionally scaled to fit."""
    if bgr is None or not hasattr(bgr, 'shape'):
        return QPixmap()
    h, w = bgr.shape[:2]
    if len(bgr.shape) == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max_width and max_height and (w > max_width or h > max_height):
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = new_h, new_w
    c = rgb.shape[2] if len(rgb.shape) == 3 else 1
    bytes_per_line = c * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


from core.cap_rotation import compute_longest_edge_angle_and_points, get_craft_box_from_result, apply_craft_rotation


def rotate_image_by_angle(bgr_image, angle_deg):
    """หมุนภาพด้วยมุม angle_deg (องศา) รอบจุดกลาง ใช้กับ 180/-180."""
    if bgr_image is None or not hasattr(bgr_image, "shape"):
        return None
    try:
        h, w = bgr_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(bgr_image, M, (w, h))
    except Exception:
        return None


def run_pipeline_for_image(image_path, cap_detector, craft_detector):
    """
    Run cap detection -> crop -> sharpen -> CRAFT for one image.
    Returns dict: path, name, step0, step1, step2_crops, step2_selected, step3, step4, craft_angle, num_caps, info_text.
    All images are BGR numpy (or None). craft_angle in degrees.
    """
    result = {
        "path": image_path,
        "name": os.path.basename(image_path),
        "step0": None,
        "step1": None,
        "step2_crops": [],
        "step2_selected": None,
        "step3": None,
        "step4": None,
        "step5_rot180": None,
        "step5_rot_minus180": None,
        "craft_angle": None,
        "num_caps": 0,
        "info_text": "",
    }
    image = cv2.imread(image_path)
    if image is None:
        result["info_text"] = "โหลดภาพไม่สำเร็จ"
        return result

    result["step0"] = image.copy()

    # Step 1: Cap detection
    if cap_detector is None:
        result["step1"] = image.copy()
        result["step2_selected"] = image.copy()
        cap_to_use = image.copy()
        cropped_list = []
    else:
        cap_result = cap_detector.detect_caps(image_path=image_path)
        detections = cap_result.get("detections", [])
        result["num_caps"] = len(detections)

        img_det = image.copy()
        for i, d in enumerate(detections):
            bbox = d.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(x) for x in bbox]
                cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf = d.get("confidence", 0)
                cv2.putText(img_det, f"cap {i+1}: {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        result["step1"] = img_det

        cropped_list = cap_detector.crop_detections(margin=20)
        result["step2_crops"] = list(cropped_list)

        if cropped_list and detections:
            bottommost_index = 0
            max_y = 0
            for i, d in enumerate(detections):
                bbox = d.get("bbox", [])
                if len(bbox) >= 4 and bbox[3] > max_y:
                    max_y = bbox[3]
                    bottommost_index = i
            cap_to_use = cropped_list[bottommost_index]
        else:
            cap_to_use = image.copy()
        result["step2_selected"] = cap_to_use

    # Step 3: Enhance sharpness
    try:
        from core.image_processor import enhance_cap_sharpness
        enhanced = enhance_cap_sharpness(cap_to_use)
        cap_for_craft = enhanced if enhanced is not None else cap_to_use
        result["step3"] = cap_for_craft
    except Exception:
        cap_for_craft = cap_to_use
        result["step3"] = cap_to_use

    if len(cap_for_craft.shape) == 2:
        cap_for_craft_bgr = cv2.cvtColor(cap_for_craft, cv2.COLOR_GRAY2BGR)
    else:
        cap_for_craft_bgr = cap_for_craft

    # Step 4: CRAFT + สร้างภาพกรอบและเส้นวัดมุม
    result["step_craft_boxes"] = None
    result["craft_bottom_points"] = None  # (p1, p2) สำหรับแสดงว่าวัดจากจุดไหนไปจุดไหน
    craft_result = None

    if craft_detector is None or (hasattr(craft_detector, "model") and craft_detector.model is None):
        result["step4"] = cap_for_craft_bgr
        result["craft_angle"] = 0.0
        result["info_text"] = f"ฝ้าที่พบ: {result['num_caps']} | CRAFT: ไม่ได้โหลด"
    else:
            craft_result = craft_detector.detect_text_and_rotate(image_array=cap_for_craft_bgr)
            if craft_result is None:
                result["step4"] = cap_for_craft_bgr
                result["craft_angle"] = 0.0
                result["info_text"] = f"ฝาที่พบ: {result['num_caps']} | CRAFT: ไม่มีผล"
            else:
                # ใช้ logic การหมุนร่วมกับโปรแกรมหลัก (core.cap_rotation)
                rotated_img, angle_edge, rotate_deg = apply_craft_rotation(cap_for_craft_bgr, craft_result)
                box = get_craft_box_from_result(craft_result)

                if box is None or rotated_img is None:
                    result["step4"] = cap_for_craft_bgr
                    result["craft_angle"] = 0.0
                    result["info_text"] = f"ฝาที่พบ: {result['num_caps']} | CRAFT: ไม่พบกรอบ"
                else:
                    result["craft_angle"] = angle_edge
                    result["craft_bottom_points"] = compute_longest_edge_angle_and_points(box)[1:]  # P1, P2
                    result["step4"] = rotated_img
                    result["info_text"] = (
                        f"ฝาที่พบ: {result['num_caps']} | "
                        f"มุมขอบยาวสุด: {angle_edge:.2f}° | หมุนอัตโนมัติ: {rotate_deg:.2f}° (ให้สี่เหลี่ยมตรงแนวนอน)"
                    )

                # วาดกรอบ CRAFT + เส้นที่ใช้วัดมุม (P1→P2)
                vis = cap_for_craft_bgr.copy()
                try:
                    _raw = craft_result.get("text_polys")
                    if _raw is None:
                        _raw = craft_result.get("text_boxes")
                    if _raw is None:
                        polys = []
                    elif isinstance(_raw, np.ndarray):
                        polys = [_raw[i] for i in range(len(_raw))] if _raw.size > 0 else []
                    elif isinstance(_raw, (list, tuple)):
                        polys = list(_raw)
                    else:
                        polys = []

                    for poly in polys:
                        try:
                            if poly is None:
                                continue
                            arr = np.asarray(poly, dtype=np.int32)
                            if arr.size < 6:
                                continue
                            arr = arr.reshape(-1, 2)
                            cv2.polylines(vis, [arr], True, (0, 255, 0), 2)
                        except Exception:
                            pass

                    fp = craft_result.get("final_polys")
                    has_final = False
                    if fp is not None:
                        if isinstance(fp, np.ndarray):
                            has_final = fp.size > 0
                        else:
                            has_final = len(fp) > 0
                    if has_final:
                        first = fp[0] if not isinstance(fp, np.ndarray) else fp[0]
                        box_vis = np.array(first, dtype=np.int32).reshape(-1, 2)
                        cv2.polylines(vis, [box_vis], True, (255, 200, 0), 2)

                        for i_edge in range(4):
                            q1 = box_vis[i_edge]
                            q2 = box_vis[(i_edge + 1) % 4]
                            x1, y1 = int(q1[0]), int(q1[1])
                            x2, y2 = int(q2[0]), int(q2[1])
                            dx = x2 - x1
                            dy = y2 - y1
                            length = math.sqrt(dx * dx + dy * dy)
                            mx = int((x1 + x2) / 2)
                            my = int((y1 + y2) / 2) - 8
                            cv2.putText(
                                vis,
                                f"{length:.1f}",
                                (mx, my),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )

                        pts = result.get("craft_bottom_points")
                        ang = result.get("craft_angle")
                        if pts is not None and len(pts) == 2 and ang is not None:
                            p1_vis, p2_vis = pts
                            cv2.line(vis, p1_vis, p2_vis, (0, 0, 255), 4)
                            cv2.circle(vis, p1_vis, 8, (0, 0, 255), -1)
                            cv2.circle(vis, p2_vis, 8, (255, 0, 255), -1)
                            cv2.putText(vis, "P1", (p1_vis[0] - 10, p1_vis[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(vis, "P2", (p2_vis[0] + 5, p2_vis[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    else:
                        # กรอบจาก text_polys: ยังวาดเส้น P1→P2 ถ้ามีมุมแล้ว
                        pts = result.get("craft_bottom_points")
                        if pts is not None and len(pts) == 2:
                            p1_vis, p2_vis = pts
                            cv2.line(vis, p1_vis, p2_vis, (0, 0, 255), 4)
                            cv2.circle(vis, p1_vis, 8, (0, 0, 255), -1)
                            cv2.circle(vis, p2_vis, 8, (255, 0, 255), -1)
                            cv2.putText(vis, "P1", (p1_vis[0] - 10, p1_vis[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(vis, "P2", (p2_vis[0] + 5, p2_vis[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                except Exception:
                    pass
                result["step_craft_boxes"] = vis

    # หลัง CRAFT: สร้างภาพหมุน 180° / -180° จาก step4 เสมอ (ถ้ามี step4 และยังไม่มี)
    if result.get("step4") is not None and result.get("step5_rot180") is None:
        result["step5_rot180"] = rotate_image_by_angle(result["step4"], 180)
        result["step5_rot_minus180"] = rotate_image_by_angle(result["step4"], -180)

    return result


class ProcessWorker(QThread):
    one_done = pyqtSignal(object)   # result dict
    progress = pyqtSignal(str)
    finished_all = pyqtSignal()

    def __init__(self, image_paths, cap_detector, craft_detector, parent=None):
        super().__init__(parent)
        self.image_paths = list(image_paths)
        self.cap_detector = cap_detector
        self.craft_detector = craft_detector
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        for i, path in enumerate(self.image_paths):
            if self._cancel:
                break
            self.progress.emit(f"กำลังประมวลผล {i+1}/{len(self.image_paths)}: {os.path.basename(path)}")
            try:
                res = run_pipeline_for_image(path, self.cap_detector, self.craft_detector)
                self.one_done.emit(res)
            except Exception as e:
                res = {
                    "path": path,
                    "name": os.path.basename(path),
                    "step0": None, "step1": None, "step2_crops": [], "step2_selected": None,
                    "step3": None, "step4": None, "step5_rot180": None, "step5_rot_minus180": None,
                    "step_craft_boxes": None, "craft_bottom_points": None,
                    "craft_angle": None, "num_caps": 0,
                    "info_text": f"ข้อผิดพลาด: {e}",
                }
                self.one_done.emit(res)
        self.finished_all.emit()


class StepViewer(QWidget):
    """แสดงภาพของ Step ที่เลือก + ข้อความ (รวมองศา CRAFT)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.title_label = QLabel("เลือก Step ด้านล่าง")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self.title_label)

        self.angle_label = QLabel("")
        self.angle_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #8e44ad; margin: 6px 0;")
        self.angle_label.setWordWrap(True)
        layout.addWidget(self.angle_label)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #333; margin: 4px 0;")
        layout.addWidget(self.info_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, 1)

    def show_result_step(self, result, step_index):
        """result = dict from run_pipeline_for_image; step_index 0..7 (รวมหมุน 180°/-180°)."""
        step_names = [
            "Step 0: ภาพต้นฉบับ",
            "Step 1: ตรวจจับตำแหน่งฝา (YOLO)",
            "Step 2: ครอปฝาที่เลือก (ฝาล่างสุด)",
            "Step 3: ปรับความคมชัด",
            "Step 4: กรอบ CRAFT + เส้นวัดมุม (P1→P2)",
            "Step 5: หมุนด้วย CRAFT",
            "Step 6: หมุน 180° (หลัง CRAFT)",
            "Step 7: หมุน -180° (หลัง CRAFT)",
        ]
        self.title_label.setText(step_names[step_index] if 0 <= step_index <= 7 else "")

        img = None
        if result:
            if step_index == 0:
                img = result.get("step0")
            elif step_index == 1:
                img = result.get("step1")
            elif step_index == 2:
                img = result.get("step2_selected")
            elif step_index == 3:
                img = result.get("step3")
            elif step_index == 4:
                img = result.get("step_craft_boxes")
                if img is None:
                    img = result.get("step3")
            elif step_index == 5:
                img = result.get("step4")
            elif step_index == 6:
                img = result.get("step5_rot180")
            elif step_index == 7:
                img = result.get("step5_rot_minus180")
            else:
                img = result.get("step4")

        if img is not None:
            pix = numpy_bgr_to_qpixmap(img, 800, 600)
            self.image_label.setPixmap(pix)
        else:
            self.image_label.clear()
            self.image_label.setText("(ไม่มีภาพ)")

        # กำกับองศา: Step 4 = วัดจาก P1→P2, Step 5 = มุมหลังหมุน (ไม่ใช้ if pts เพื่อหลีกเลี่ยง truth value of array)
        if result and step_index == 4:
            pts = result.get("craft_bottom_points")
            angle = result.get("craft_angle")
            if pts is not None and len(pts) == 2 and angle is not None:
                p1, p2 = pts
                self.angle_label.setText(
                    f"มุมวัดจากเส้นล่าง (P1 → P2): {angle:.2f}°\n"
                    f"P1 = {p1}, P2 = {p2}"
                )
                self.angle_label.setVisible(True)
            else:
                self.angle_label.setText("ไม่มีกรอบ CRAFT หรือมุม")
                self.angle_label.setVisible(True)
        elif result and step_index == 5 and result.get("craft_angle") is not None:
            angle = result["craft_angle"]
            self.angle_label.setText(f"มุมหมุนของตัวอักษรจากกรอบ CRAFT: {angle:.2f}°")
            self.angle_label.setVisible(True)
        elif result and step_index == 6:
            self.angle_label.setText("ภาพหลังหมุน 180° (ใช้ในขั้น OCR ในโปรแกรมหลัก)")
            self.angle_label.setVisible(True)
        elif result and step_index == 7:
            self.angle_label.setText("ภาพหลังหมุน -180° (ใช้ในขั้น OCR ในโปรแกรมหลัก)")
            self.angle_label.setVisible(True)
        else:
            self.angle_label.setText("")
            self.angle_label.setVisible(False)

        info = result.get("info_text", "") if result else ""
        self.info_label.setText(info or "(ไม่มีข้อมูล)")


class CapRotationTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ทดสอบระบบหมุนฝา — เลือกภาพ / ประวัติ / ดูแต่ละ Step")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 750)

        self.image_paths = []
        self.history_results = []  # list of result dicts
        self.current_result = None
        self.cap_detector = None
        self.craft_detector = None
        self.worker = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # โหลดโมเดลตอนเริ่ม
        self.statusBar().showMessage("กำลังโหลดโมเดล Cap และ CRAFT...")
        QApplication.processEvents()
        self._load_models()

        # แถบปุ่ม
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QPushButton("เลือกภาพ...", clicked=self.open_files))
        btn_layout.addWidget(QPushButton("เลือกโฟลเดอร์...", clicked=self.open_folder))
        btn_layout.addWidget(QPushButton("ประมวลผลที่เลือก", clicked=self.process_selected))
        btn_layout.addWidget(QPushButton("ประมวลผลทั้งหมด", clicked=self.process_all))
        btn_layout.addWidget(QPushButton("ล้างรายการภาพ", clicked=self.clear_list))
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # indeterminate when not running
        self.progress_bar.setVisible(False)
        btn_layout.addWidget(self.progress_bar)
        layout.addLayout(btn_layout)

        splitter = QSplitter(Qt.Horizontal)

        # ซ้าย: แท็บ รายการภาพ + ประวัติ
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.tabs = QTabWidget()
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_widget.setMinimumWidth(220)
        self.tabs.addTab(self.list_widget, "รายการภาพ")
        self.history_list = QListWidget()
        self.history_list.setMinimumWidth(220)
        self.history_list.currentRowChanged.connect(self.on_history_selected)
        self.tabs.addTab(self.history_list, "ประวัติ")
        left_layout.addWidget(self.tabs)
        splitter.addWidget(left)

        # ขวา: เลือก Step + แสดงภาพ
        right = QWidget()
        right_layout = QVBoxLayout(right)
        step_group = QGroupBox("ขั้นตอน (Step)")
        step_layout = QHBoxLayout(step_group)
        self.step_combo = QComboBox()
        for label in ["ภาพต้นฉบับ", "ตรวจจับฝา", "ครอปฝา", "ความคมชัด", "กรอบ CRAFT + เส้นวัดมุม", "หมุน CRAFT", "หมุน 180° (หลัง CRAFT)", "หมุน -180° (หลัง CRAFT)"]:
            self.step_combo.addItem(label)
        self.step_combo.currentIndexChanged.connect(self.on_step_changed)
        step_layout.addWidget(QLabel("Step:"))
        step_layout.addWidget(self.step_combo, 1)
        right_layout.addWidget(step_group)

        self.step_viewer = StepViewer()
        right_layout.addWidget(self.step_viewer, 1)
        splitter.addWidget(right)

        splitter.setSizes([300, 700])
        layout.addWidget(splitter, 1)

        self.statusBar().showMessage("พร้อม — เลือกภาพหรือโฟลเดอร์ แล้วกดประมวลผล")
        self.step_combo.setCurrentIndex(0)

    def _load_models(self):
        try:
            from config.settings import CAP_MODEL_PATH, CRAFT_MODEL_PATH, CRAFT_REFINER_PATH
            from libs.detection.capmodel import initialize_detector as init_cap_detector
            from libs.processing.rotationCRAFT import initialize_detector as init_craft_detector

            self.cap_detector = init_cap_detector(CAP_MODEL_PATH)
            if self.cap_detector is not None and hasattr(self.cap_detector, "model") and self.cap_detector.model is None:
                self.cap_detector = None
            self.craft_detector = init_craft_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
            if self.craft_detector is not None and hasattr(self.craft_detector, "model") and self.craft_detector.model is None:
                self.craft_detector = None
        except Exception as e:
            self.statusBar().showMessage(f"โหลดโมเดลผิดพลาด: {e}")
            self.cap_detector = None
            self.craft_detector = None

    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "เลือกภาพ", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;All (*.*)"
        )
        for p in paths:
            if p and p not in self.image_paths:
                self.image_paths.append(p)
                self.list_widget.addItem(os.path.basename(p))
        self.statusBar().showMessage(f"มีภาพในรายการ {len(self.image_paths)} ไฟล์")

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์")
        if not folder:
            return
        added = 0
        for name in sorted(os.listdir(folder)):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                p = os.path.join(folder, name)
                if p not in self.image_paths:
                    self.image_paths.append(p)
                    self.list_widget.addItem(name)
                    added += 1
        self.statusBar().showMessage(f"เพิ่มจากโฟลเดอร์ {added} ไฟล์ (รวม {len(self.image_paths)} ไฟล์)")

    def clear_list(self):
        self.image_paths.clear()
        self.list_widget.clear()
        self.statusBar().showMessage("ล้างรายการแล้ว")

    def process_selected(self):
        rows = set(i.row() for i in self.list_widget.selectedIndexes())
        paths = [self.image_paths[r] for r in sorted(rows) if 0 <= r < len(self.image_paths)]
        if not paths:
            QMessageBox.information(self, "แจ้ง", "กรุณาเลือกภาพในรายการก่อน")
            return
        self._run_process(paths)

    def process_all(self):
        if not self.image_paths:
            QMessageBox.information(self, "แจ้ง", "ยังไม่มีภาพในรายการ — เลือกภาพหรือโฟลเดอร์ก่อน")
            return
        self._run_process(self.image_paths)

    def _run_process(self, paths):
        if self.worker is not None and self.worker.isRunning():
            self.statusBar().showMessage("กำลังประมวลผลอยู่ กรุณารอ")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)
        self.worker = ProcessWorker(paths, self.cap_detector, self.craft_detector, self)
        self.worker.one_done.connect(self.on_one_done)
        self.worker.progress.connect(lambda t: self.statusBar().showMessage(t))
        self.worker.finished_all.connect(self.on_finished_all)
        self.worker.start()

    def on_one_done(self, result):
        self.history_results.append(result)
        name = result["name"]
        angle = result.get("craft_angle")
        angle_str = f" {angle:.2f}°" if angle is not None else ""
        item = QListWidgetItem(f"{name}{angle_str}")
        item.setData(Qt.UserRole, len(self.history_results) - 1)
        self.history_list.addItem(item)
        # แสดงผลลัพธ์ล่าสุด
        self.current_result = result
        self.step_viewer.show_result_step(result, self.step_combo.currentIndex())

    def on_finished_all(self):
        self.progress_bar.setVisible(False)
        self.worker = None
        self.statusBar().showMessage("ประมวลผลครบทั้งหมดแล้ว — ดูได้ที่แท็บประวัติ")

    def on_history_selected(self, row):
        if row < 0 or row >= len(self.history_results):
            return
        self.current_result = self.history_results[row]
        self.step_viewer.show_result_step(self.current_result, self.step_combo.currentIndex())

    def on_step_changed(self, index):
        if self.current_result is not None:
            self.step_viewer.show_result_step(self.current_result, index)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = CapRotationTestWindow()
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

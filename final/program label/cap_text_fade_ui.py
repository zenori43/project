# -*- coding: utf-8 -*-
"""
Cap Text Fade Generator
-----------------------
1. ตรวจจับฝา (YOLO cap.pt) → ครอปฝาล่างสุด
2. ตรวจจับตัวอักษรบนฝาด้วย CRAFT → สร้างกรอบทั้งหมด → สุ่มเลือก 1 กรอบ
3. สร้างภาพ fade จาก 4 วิธี:
   - Gradient Horizontal  (fade ซ้าย→ขวา)
   - Gradient Vertical    (fade บน→ล่าง)
   - Alpha Blending       (ผสมแบบ uniform)
   - Inpainting           (ลบตัวอักษรด้วย cv2.inpaint)
4. UI เลือกได้หลายไฟล์หรือภาพเดียว บันทึกผลได้
"""

import sys
import os

# ---- path setup ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR  = os.path.dirname(SCRIPT_DIR)
if FINAL_DIR not in sys.path:
    sys.path.insert(0, FINAL_DIR)

try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    pass

# ---- standard imports ----------------------------------------------------------
import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QProgressBar,
    QGroupBox, QDoubleSpinBox, QSpinBox, QScrollArea, QGridLayout,
    QSplitter, QLineEdit, QMessageBox, QFrame, QSizePolicy,
    QAbstractItemView, QCheckBox
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui   import QImage, QPixmap, QFont, QColor, QPalette


# ================================================================================
#  Helper: numpy image → QPixmap (scaled to fit max_w × max_h)
# ================================================================================
def ndarray_to_pixmap(img: np.ndarray, max_w: int = 300, max_h: int = 240) -> QPixmap:
    if img is None or img.size == 0:
        return QPixmap()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h2, w2, c = rgb.shape
    qimg = QImage(rgb.data, w2, h2, w2 * c, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ================================================================================
#  Fade methods  (all work on BGR numpy images)
# ================================================================================

def _box_rect(box_pts: np.ndarray, img_shape: Tuple) -> Tuple[int, int, int, int]:
    """Return clamped (x1,y1,x2,y2) bounding rect from a set of 2D points."""
    pts = box_pts.astype(np.float32)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    H, W = img_shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    return x1, y1, x2, y2


def _background_for_region(image: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    """Return a plausible 'no-text' background using blur of the region."""
    region = image[y1:y2, x1:x2]
    ksize = max(3, min(region.shape[0], region.shape[1]) // 2 | 1)
    blurred = cv2.GaussianBlur(region, (ksize, ksize), 0)
    return blurred


def apply_gradient_horizontal(image: np.ndarray, box_pts: np.ndarray) -> np.ndarray:
    """Horizontal gradient fade: left=original, right=blurred background."""
    result = image.copy()
    x1, y1, x2, y2 = _box_rect(box_pts, image.shape)
    w = x2 - x1;  h = y2 - y1
    if w <= 0 or h <= 0:
        return result
    bg    = _background_for_region(image, x1, y1, x2, y2).astype(np.float32)
    fg    = image[y1:y2, x1:x2].astype(np.float32)
    alpha = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :, np.newaxis]  # (1, w, 1)
    blended = fg * (1.0 - alpha) + bg * alpha
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def apply_gradient_vertical(image: np.ndarray, box_pts: np.ndarray) -> np.ndarray:
    """Vertical gradient fade: top=original, bottom=blurred background."""
    result = image.copy()
    x1, y1, x2, y2 = _box_rect(box_pts, image.shape)
    w = x2 - x1;  h = y2 - y1
    if w <= 0 or h <= 0:
        return result
    bg    = _background_for_region(image, x1, y1, x2, y2).astype(np.float32)
    fg    = image[y1:y2, x1:x2].astype(np.float32)
    alpha = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis, np.newaxis]  # (h,1,1)
    blended = fg * (1.0 - alpha) + bg * alpha
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def apply_alpha_blending(image: np.ndarray, box_pts: np.ndarray,
                         alpha: float = 0.80) -> np.ndarray:
    """Uniform alpha blend: blend text region with blurred background."""
    result = image.copy()
    x1, y1, x2, y2 = _box_rect(box_pts, image.shape)
    w = x2 - x1;  h = y2 - y1
    if w <= 0 or h <= 0:
        return result
    bg = _background_for_region(image, x1, y1, x2, y2).astype(np.float32)
    fg = image[y1:y2, x1:x2].astype(np.float32)
    blended = fg * (1.0 - alpha) + bg * alpha
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def apply_inpainting(image: np.ndarray, box_pts: np.ndarray,
                     radius: int = 5) -> np.ndarray:
    """Use cv2.inpaint to fill in / remove the text inside the selected box."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts  = box_pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    # Dilate mask slightly so edges are also painted
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


# ================================================================================
#  Worker thread
# ================================================================================

class ProcessWorker(QThread):
    """Runs cap detection + CRAFT + fade generation for a list of image files."""

    sig_progress    = pyqtSignal(int, int, str)   # (current, total, message)
    sig_result      = pyqtSignal(dict)             # result dict for one image
    sig_finished    = pyqtSignal(str)              # summary
    sig_error       = pyqtSignal(str)

    def __init__(self, files: List[str],
                 cap_model_path: str,
                 craft_model_path: str,
                 craft_refiner_path: Optional[str],
                 cap_conf: float,
                 cap_margin: int,
                 text_threshold: float,
                 link_threshold: float,
                 low_text: float,
                 alpha_value: float = 0.80,
                 inpaint_radius: int = 5,
                 parent=None):
        super().__init__(parent)
        self.files              = files
        self.cap_model_path     = cap_model_path
        self.craft_model_path   = craft_model_path
        self.craft_refiner_path = craft_refiner_path
        self.cap_conf           = cap_conf
        self.cap_margin         = cap_margin
        self.text_threshold     = text_threshold
        self.link_threshold     = link_threshold
        self.low_text           = low_text
        self.alpha_value        = alpha_value
        self.inpaint_radius     = inpaint_radius
        self._stop_flag         = False

    def stop(self):
        self._stop_flag = True

    # ------------------------------------------------------------------ run
    def run(self):
        try:
            # --- Load cap detector ---
            self.sig_progress.emit(0, len(self.files), "กำลังโหลด Cap Model...")
            try:
                from libs.detection.capmodel import CapDetector
                cap_det = CapDetector(model_path=self.cap_model_path,
                                      conf_threshold=self.cap_conf)
            except Exception as e:
                self.sig_error.emit(f"ไม่สามารถโหลด Cap Model:\n{e}")
                return

            # --- Load CRAFT detector ---
            self.sig_progress.emit(0, len(self.files), "กำลังโหลด CRAFT Model...")
            try:
                from libs.processing.rotationCRAFT import CRAFTTextDetector
                refiner = (self.craft_refiner_path
                           if self.craft_refiner_path and os.path.exists(self.craft_refiner_path)
                           else None)
                craft_det = CRAFTTextDetector(model_path=self.craft_model_path,
                                              refiner_path=refiner)
                craft_det.set_detection_parameters(
                    text_threshold=self.text_threshold,
                    link_threshold=self.link_threshold,
                    low_text=self.low_text,
                )
            except Exception as e:
                self.sig_error.emit(f"ไม่สามารถโหลด CRAFT Model:\n{e}")
                return

            ok = skip = err = 0
            total = len(self.files)

            for idx, fpath in enumerate(self.files):
                if self._stop_flag:
                    break
                fname = os.path.basename(fpath)
                self.sig_progress.emit(idx, total, f"ประมวลผล: {fname}  ({idx+1}/{total})")
                try:
                    res = self._process_one(fpath, cap_det, craft_det)
                    res["index"]      = idx
                    res["image_path"] = fpath
                    self.sig_result.emit(res)
                    ok += 1 if res.get("success") else 0
                    skip += 1 if not res.get("success") else 0
                except Exception as e:
                    print(f"[ProcessWorker] Error on {fpath}: {e}")
                    err += 1
                    self.sig_result.emit({
                        "index": idx, "image_path": fpath,
                        "success": False, "error": str(e)
                    })

            self.sig_progress.emit(total, total, "เสร็จสิ้น")
            self.sig_finished.emit(
                f"เสร็จสิ้น  ✅ สำเร็จ: {ok}  ⚠️ ข้าม: {skip}  ❌ ผิดพลาด: {err}")

        except Exception as e:
            self.sig_error.emit(str(e))

    # -------------------------------------------------------- process one image
    def _process_one(self, fpath: str, cap_det, craft_det) -> dict:
        # Read image (BGR)
        img = cv2.imread(fpath)
        if img is None:
            return {"success": False, "error": "ไม่สามารถอ่านภาพได้"}

        # ---- Step 1: detect caps ------------------------------------------------
        det_res    = cap_det.detect_caps(image_array=img)
        detections = det_res.get("detections", [])

        if not detections:
            return {"success": False, "error": "ไม่พบฝาในภาพ", "original": img}

        # Visualise cap detections on original
        orig_vis = img.copy()
        for d in detections:
            bb = d["bbox"]
            cv2.rectangle(orig_vis, (bb[0], bb[1]), (bb[2], bb[3]), (0, 200, 0), 2)
            lbl = f"{d['class_name']} {d['confidence']:.2f}"
            cv2.putText(orig_vis, lbl, (bb[0], max(bb[1]-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

        # Select bottommost cap (highest y-center)
        bm_idx = max(range(len(detections)),
                     key=lambda i: sum(detections[i]["bbox"][1::2]) / 2)
        cap_bbox = detections[bm_idx]["bbox"]
        # Highlight selected cap in red
        cv2.rectangle(orig_vis,
                      (cap_bbox[0], cap_bbox[1]), (cap_bbox[2], cap_bbox[3]),
                      (0, 0, 220), 3)

        # ---- Step 2: crop cap ---------------------------------------------------
        cap_bgr_list = cap_det.crop_detections(margin=self.cap_margin)
        cap_bgr = cap_bgr_list[bm_idx].copy()   # BGR

        # ---- Step 3: CRAFT text detection on crop --------------------------------
        # CRAFTTextDetector expects BGR (converts internally to RGB)
        craft_res  = craft_det.detect_text_and_rotate(image_array=cap_bgr)
        text_boxes = craft_res.get("text_boxes", [])   # list of (4,2) or (N,2) arrays

        if len(text_boxes) == 0:
            return {
                "success": False,
                "error": "CRAFT ไม่พบตัวอักษรในภาพฝา",
                "original": img,
                "orig_vis": orig_vis,
                "cap_crop": cap_bgr,
            }

        # ---- Step 4: draw all boxes & randomly pick one --------------------------
        selected_idx = random.randint(0, len(text_boxes) - 1)
        sel_pts      = np.array(text_boxes[selected_idx], dtype=np.float32)

        craft_vis = cap_bgr.copy()
        for i, box in enumerate(text_boxes):
            pts      = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
            is_sel   = (i == selected_idx)
            color    = (0, 60, 255) if is_sel else (0, 200, 0)
            thickness = 3 if is_sel else 1
            cv2.polylines(craft_vis, [pts], True, color, thickness)

        # Label on selected box
        x0, y0, w0, h0 = cv2.boundingRect(sel_pts.astype(np.int32))
        cv2.putText(craft_vis,
                    f"Selected #{selected_idx+1}",
                    (max(0, x0), max(0, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 1)

        # ---- Step 5: fade methods -----------------------------------------------
        fade_h   = apply_gradient_horizontal(cap_bgr, sel_pts)
        fade_v   = apply_gradient_vertical  (cap_bgr, sel_pts)
        fade_ab  = apply_alpha_blending     (cap_bgr, sel_pts, alpha=self.alpha_value)
        fade_inp = apply_inpainting         (cap_bgr, sel_pts, radius=self.inpaint_radius)

        return {
            "success":      True,
            "original":     img,
            "orig_vis":     orig_vis,
            "cap_crop":     cap_bgr,
            "craft_vis":    craft_vis,
            "total_boxes":  len(text_boxes),
            "selected_idx": selected_idx,
            "fade_h":       fade_h,
            "fade_v":       fade_v,
            "fade_ab":      fade_ab,
            "fade_inp":     fade_inp,
        }


# ================================================================================
#  Result card  (one per processed image)
# ================================================================================

class ResultCard(QFrame):
    sig_save = pyqtSignal(dict)

    # Thumbnail settings
    IMG_W = 290
    IMG_H = 230

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.result = result
        self.setFrameShape(QFrame.Box)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.setStyleSheet("ResultCard { background: #fff; border-radius: 6px; }")
        self._build()

    # ------------------------------------------------------------------
    def _img_label(self, img: Optional[np.ndarray], title: str) -> QWidget:
        """Create a small widget: image thumbnail + caption."""
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(2, 2, 2, 2)
        vl.setSpacing(2)

        lbl = QLabel()
        if img is not None:
            lbl.setPixmap(ndarray_to_pixmap(img, self.IMG_W, self.IMG_H))
        else:
            lbl.setText("N/A")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("border: 1px solid #ddd; background: #f9f9f9;")
        lbl.setFixedSize(self.IMG_W + 2, self.IMG_H + 2)

        cap = QLabel(title)
        cap.setAlignment(Qt.AlignCenter)
        cap.setWordWrap(True)
        cap.setStyleSheet("font-size: 10px; color: #444;")

        vl.addWidget(lbl)
        vl.addWidget(cap)
        return w

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Header
        fname = os.path.basename(self.result.get("image_path", ""))
        hdr = QLabel(f"<b>{fname}</b>")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet("font-size: 12px; padding: 4px; background: #ecf0f1;")
        layout.addWidget(hdr)

        if not self.result.get("success"):
            err = QLabel(f"⚠️  {self.result.get('error', 'ผิดพลาดไม่ทราบสาเหตุ')}")
            err.setStyleSheet("color: #c0392b; font-size: 11px; padding: 4px;")
            err.setWordWrap(True)
            layout.addWidget(err)
            # Still show whatever images we have (e.g. original / cap crop)
            row = QHBoxLayout()
            for key, title in [("orig_vis", "ภาพต้นฉบับ"),
                                ("original", "ภาพต้นฉบับ"),
                                ("cap_crop", "ภาพฝา")]:
                img = self.result.get(key)
                if img is not None:
                    row.addWidget(self._img_label(img, title))
            if row.count():
                layout.addLayout(row)
            return

        # ----- Row 1: original / cap / craft -----
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(self._img_label(self.result.get("orig_vis"),
                                       "ต้นฉบับ (ตรวจจับฝา)"))
        row1.addWidget(self._img_label(self.result.get("cap_crop"),
                                       "ฝาที่ครอปได้"))
        n_boxes = self.result.get("total_boxes", 0)
        sel_idx = self.result.get("selected_idx", 0)
        row1.addWidget(self._img_label(self.result.get("craft_vis"),
                                       f"CRAFT ({n_boxes} boxes)\nเลือก #{sel_idx+1} (สีแดง)"))
        row1.addStretch()
        layout.addLayout(row1)

        # ----- Row 2: fade methods -----
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(self._img_label(self.result.get("fade_h"),   "Gradient Horizontal"))
        row2.addWidget(self._img_label(self.result.get("fade_v"),   "Gradient Vertical"))
        row2.addWidget(self._img_label(self.result.get("fade_ab"),  "Alpha Blending"))
        row2.addWidget(self._img_label(self.result.get("fade_inp"), "Inpainting"))
        layout.addLayout(row2)

        # ----- Save button -----
        btn = QPushButton("💾  บันทึกผลภาพนี้")
        btn.setStyleSheet("""
            QPushButton {
                background: #2980b9; color: white;
                border-radius: 4px; padding: 4px 12px;
            }
            QPushButton:hover { background: #3498db; }
        """)
        btn.clicked.connect(lambda: self.sig_save.emit(self.result))
        layout.addWidget(btn, alignment=Qt.AlignRight)


# ================================================================================
#  Main window
# ================================================================================

PANEL_WIDTH = 370

class CapTextFadeUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cap Text Fade Generator")
        self.setMinimumSize(1500, 820)
        self.selected_files: List[str] = []
        self.worker: Optional[ProcessWorker] = None
        self._results: List[dict] = []
        self._build_ui()
        self._load_default_paths()

    # --------------------------------------------------------- default paths
    def _load_default_paths(self):
        try:
            from config.settings import CAP_MODEL_PATH, CRAFT_MODEL_PATH, CRAFT_REFINER_PATH
            self.cap_model_edit.setText(CAP_MODEL_PATH)
            self.craft_model_edit.setText(CRAFT_MODEL_PATH)
            self.craft_refiner_edit.setText(CRAFT_REFINER_PATH)
        except Exception:
            pass

    # --------------------------------------------------------- build UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ================ LEFT PANEL ================
        left = QWidget()
        left.setFixedWidth(PANEL_WIDTH)
        ll = QVBoxLayout(left)
        ll.setSpacing(8)
        ll.setContentsMargins(4, 4, 4, 4)

        # Title
        title = QLabel("Cap Text Fade Generator")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color: #2c3e50; background: #d5e8f8; "
            "border-radius: 6px; padding: 8px;"
        )
        ll.addWidget(title)

        # ---- Model paths ----
        mg = QGroupBox("โมเดล")
        mgl = QGridLayout(mg)

        def _model_row(label_text, row_base):
            mgl.addWidget(QLabel(label_text), row_base, 0, 1, 2)
            edit = QLineEdit()
            mgl.addWidget(edit, row_base + 1, 0)
            btn = QPushButton("…")
            btn.setFixedWidth(28)
            mgl.addWidget(btn, row_base + 1, 1)
            return edit, btn

        self.cap_model_edit, b1 = _model_row("Cap Model (.pt):", 0)
        b1.clicked.connect(
            lambda: self._browse_file(self.cap_model_edit, "PyTorch Model (*.pt)"))

        self.craft_model_edit, b2 = _model_row("CRAFT Model (.pth):", 2)
        b2.clicked.connect(
            lambda: self._browse_file(self.craft_model_edit, "CRAFT Model (*.pth)"))

        self.craft_refiner_edit, b3 = _model_row("CRAFT Refiner (optional):", 4)
        self.craft_refiner_edit.setPlaceholderText("ข้ามได้ถ้าไม่มีไฟล์ refiner")
        b3.clicked.connect(
            lambda: self._browse_file(self.craft_refiner_edit, "CRAFT Refiner (*.pth)"))

        ll.addWidget(mg)

        # ---- Parameters ----
        pg = QGroupBox("พารามิเตอร์")
        pgl = QGridLayout(pg)

        def _spin_row(label, row, widget):
            pgl.addWidget(QLabel(label), row, 0)
            pgl.addWidget(widget, row, 1)

        self.cap_conf_spin = QDoubleSpinBox()
        self.cap_conf_spin.setRange(0.05, 1.0)
        self.cap_conf_spin.setSingleStep(0.05)
        self.cap_conf_spin.setValue(0.50)
        self.cap_conf_spin.setDecimals(2)
        _spin_row("Cap Confidence:", 0, self.cap_conf_spin)

        self.cap_margin_spin = QSpinBox()
        self.cap_margin_spin.setRange(0, 200)
        self.cap_margin_spin.setValue(20)
        _spin_row("Cap Margin (px):", 1, self.cap_margin_spin)

        self.text_thresh_spin = QDoubleSpinBox()
        self.text_thresh_spin.setRange(0.1, 1.0)
        self.text_thresh_spin.setSingleStep(0.05)
        self.text_thresh_spin.setValue(0.70)
        self.text_thresh_spin.setDecimals(2)
        _spin_row("CRAFT Text Threshold:", 2, self.text_thresh_spin)

        self.link_thresh_spin = QDoubleSpinBox()
        self.link_thresh_spin.setRange(0.1, 1.0)
        self.link_thresh_spin.setSingleStep(0.05)
        self.link_thresh_spin.setValue(0.40)
        self.link_thresh_spin.setDecimals(2)
        _spin_row("CRAFT Link Threshold:", 3, self.link_thresh_spin)

        self.low_text_spin = QDoubleSpinBox()
        self.low_text_spin.setRange(0.05, 1.0)
        self.low_text_spin.setSingleStep(0.05)
        self.low_text_spin.setValue(0.40)
        self.low_text_spin.setDecimals(2)
        _spin_row("CRAFT Low Text:", 4, self.low_text_spin)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.80)
        self.alpha_spin.setDecimals(2)
        _spin_row("Alpha Blend (0=original, 1=fade):", 5, self.alpha_spin)

        self.inpaint_radius_spin = QSpinBox()
        self.inpaint_radius_spin.setRange(1, 30)
        self.inpaint_radius_spin.setValue(5)
        _spin_row("Inpaint Radius (px):", 6, self.inpaint_radius_spin)

        ll.addWidget(pg)

        # ---- File selection ----
        fg = QGroupBox("ภาพที่เลือก")
        fgl = QVBoxLayout(fg)

        fb_row = QHBoxLayout()
        self.select_btn = QPushButton("📂  เลือกภาพ")
        self.select_btn.clicked.connect(self._select_files)
        self.clear_files_btn = QPushButton("🗑  ล้าง")
        self.clear_files_btn.clicked.connect(self._clear_files)
        fb_row.addWidget(self.select_btn)
        fb_row.addWidget(self.clear_files_btn)
        fgl.addLayout(fb_row)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(100)
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        fgl.addWidget(self.file_list)

        self.file_count_lbl = QLabel("ยังไม่ได้เลือกไฟล์")
        self.file_count_lbl.setStyleSheet("color:#666; font-size:10px;")
        fgl.addWidget(self.file_count_lbl)
        ll.addWidget(fg)

        # ---- Auto-save folder ----
        og = QGroupBox("โฟลเดอร์บันทึกอัตโนมัติ (ไม่บังคับ)")
        ogl = QHBoxLayout(og)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("เลือกโฟลเดอร์ เพื่อบันทึกอัตโนมัติ...")
        ogl.addWidget(self.output_dir_edit)
        obtn = QPushButton("…")
        obtn.setFixedWidth(28)
        obtn.clicked.connect(self._browse_output_dir)
        ogl.addWidget(obtn)
        ll.addWidget(og)

        # ---- Action buttons ----
        self.process_btn = QPushButton("▶  ประมวลผล")
        self.process_btn.setMinimumHeight(42)
        self.process_btn.setStyleSheet(self._green_btn_style())
        self.process_btn.clicked.connect(self._start_processing)
        ll.addWidget(self.process_btn)

        self.stop_btn = QPushButton("⏹  หยุด")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(self._red_btn_style())
        self.stop_btn.clicked.connect(self._stop_processing)
        ll.addWidget(self.stop_btn)

        # ---- Progress ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        ll.addWidget(self.progress_bar)

        self.status_lbl = QLabel("พร้อมใช้งาน")
        self.status_lbl.setStyleSheet("color:#27ae60; font-size:11px;")
        self.status_lbl.setWordWrap(True)
        ll.addWidget(self.status_lbl)

        ll.addStretch()

        # ================ RIGHT PANEL (results scroll area) ================
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        rh = QHBoxLayout()
        results_title = QLabel("ผลลัพธ์")
        results_title.setFont(QFont("Arial", 12, QFont.Bold))
        rh.addWidget(results_title)
        rh.addStretch()
        save_all_btn = QPushButton("💾  บันทึกทั้งหมด")
        save_all_btn.clicked.connect(self._save_all_results)
        rh.addWidget(save_all_btn)
        clear_res_btn = QPushButton("🗑  ล้างผล")
        clear_res_btn.clicked.connect(self._clear_results)
        rh.addWidget(clear_res_btn)
        rl.addLayout(rh)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.results_container = QWidget()
        self.results_vbox = QVBoxLayout(self.results_container)
        self.results_vbox.setSpacing(12)
        self.results_vbox.addStretch()
        self.scroll.setWidget(self.results_container)
        rl.addWidget(self.scroll)

        root.addWidget(left)
        root.addWidget(right, 1)

        # Global style
        self.setStyleSheet("""
            QMainWindow  { background: #ecf0f1; }
            QGroupBox    {
                font-weight: bold; border: 1px solid #bdc3c7;
                border-radius: 6px; margin-top: 10px; padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 4px;
            }
            QScrollArea  { border: none; }
            QListWidget  { font-size: 11px; }
            QLabel       { font-size: 11px; }
            QPushButton  { font-size: 11px; }
        """)

    # --------------------------------------------------------- styles
    @staticmethod
    def _green_btn_style():
        return """
            QPushButton {
                background:#27ae60; color:white; font-size:14px;
                font-weight:bold; border-radius:6px;
            }
            QPushButton:hover    { background:#2ecc71; }
            QPushButton:disabled { background:#95a5a6; }
        """

    @staticmethod
    def _red_btn_style():
        return """
            QPushButton {
                background:#e74c3c; color:white; font-size:13px;
                border-radius:6px;
            }
            QPushButton:hover    { background:#c0392b; }
            QPushButton:disabled { background:#95a5a6; }
        """

    # --------------------------------------------------------- helpers
    def _browse_file(self, edit: QLineEdit, ffilter: str):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์", "", ffilter)
        if path:
            edit.setText(path)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์บันทึก")
        if path:
            self.output_dir_edit.setText(path)

    def _select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "เลือกภาพ", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;All Files (*.*)"
        )
        if files:
            seen = set(self.selected_files)
            for f in files:
                if f not in seen:
                    seen.add(f)
                    self.selected_files.append(f)
            self._refresh_file_list()

    def _clear_files(self):
        self.selected_files.clear()
        self.file_list.clear()
        self.file_count_lbl.setText("ยังไม่ได้เลือกไฟล์")

    def _refresh_file_list(self):
        self.file_list.clear()
        for f in self.selected_files:
            self.file_list.addItem(os.path.basename(f))
        n = len(self.selected_files)
        self.file_count_lbl.setText(f"เลือกแล้ว {n} ไฟล์")

    def _clear_results(self):
        while self.results_vbox.count() > 1:
            item = self.results_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._results.clear()

    # --------------------------------------------------------- processing
    def _start_processing(self):
        if not self.selected_files:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกภาพก่อน")
            return

        cap_model = self.cap_model_edit.text().strip()
        if not cap_model or not os.path.exists(cap_model):
            QMessageBox.warning(self, "แจ้งเตือน",
                                f"ไม่พบ Cap Model:\n{cap_model}")
            return

        craft_model = self.craft_model_edit.text().strip()
        if not craft_model or not os.path.exists(craft_model):
            QMessageBox.warning(self, "แจ้งเตือน",
                                f"ไม่พบ CRAFT Model:\n{craft_model}")
            return

        craft_refiner = self.craft_refiner_edit.text().strip()
        if not craft_refiner or not os.path.exists(craft_refiner):
            craft_refiner = None

        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.selected_files))
        self.status_lbl.setText("กำลังเริ่มต้น...")
        self.status_lbl.setStyleSheet("color:#2980b9; font-size:11px;")

        self.worker = ProcessWorker(
            files             = list(self.selected_files),
            cap_model_path    = cap_model,
            craft_model_path  = craft_model,
            craft_refiner_path= craft_refiner,
            cap_conf          = self.cap_conf_spin.value(),
            cap_margin        = self.cap_margin_spin.value(),
            text_threshold    = self.text_thresh_spin.value(),
            link_threshold    = self.link_thresh_spin.value(),
            low_text          = self.low_text_spin.value(),
            alpha_value       = self.alpha_spin.value(),
            inpaint_radius    = self.inpaint_radius_spin.value(),
        )
        self.worker.sig_progress.connect(self._on_progress)
        self.worker.sig_result  .connect(self._on_result)
        self.worker.sig_finished.connect(self._on_finished)
        self.worker.sig_error   .connect(self._on_error)
        self.worker.start()

    def _stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.status_lbl.setText("กำลังหยุดการทำงาน...")

    # --------------------------------------------------------- worker callbacks
    def _on_progress(self, current: int, total: int, msg: str):
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(current)
        self.status_lbl.setText(msg)
        self.status_lbl.setStyleSheet("color:#2980b9; font-size:11px;")

    def _on_result(self, result: dict):
        card = ResultCard(result)
        card.sig_save.connect(self._save_one_result_dialog)
        # Insert before the trailing stretch
        self.results_vbox.insertWidget(self.results_vbox.count() - 1, card)
        self._results.append(result)

        # Auto-save if folder is set
        out_dir = self.output_dir_edit.text().strip()
        if out_dir and result.get("success"):
            self._do_save(result, out_dir)

        # Scroll to newest card
        QApplication.processEvents()
        self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        )

    def _on_finished(self, msg: str):
        self.status_lbl.setText(f"✅  {msg}")
        self.status_lbl.setStyleSheet("color:#27ae60; font-size:11px;")
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_error(self, msg: str):
        self.status_lbl.setText(f"❌  {msg}")
        self.status_lbl.setStyleSheet("color:#c0392b; font-size:11px;")
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "เกิดข้อผิดพลาด", msg)

    # --------------------------------------------------------- save helpers
    def _save_one_result_dialog(self, result: dict):
        out_dir = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์บันทึก")
        if out_dir:
            n = self._do_save(result, out_dir)
            QMessageBox.information(self, "บันทึกสำเร็จ",
                                    f"บันทึก {n} ไฟล์ไปที่:\n{out_dir}")

    def _save_all_results(self):
        if not self._results:
            QMessageBox.information(self, "แจ้งเตือน", "ยังไม่มีผลลัพธ์")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์บันทึกทั้งหมด")
        if out_dir:
            total = 0
            for r in self._results:
                if r.get("success"):
                    total += self._do_save(r, out_dir)
            QMessageBox.information(self, "บันทึกสำเร็จ",
                                    f"บันทึก {total} ไฟล์ไปที่:\n{out_dir}")

    @staticmethod
    def _do_save(result: dict, out_dir: str) -> int:
        os.makedirs(out_dir, exist_ok=True)
        stem = Path(result.get("image_path", "unknown")).stem
        mapping = [
            ("fade_h",    f"{stem}_fade_01_gradient_h.jpg"),
            ("fade_v",    f"{stem}_fade_02_gradient_v.jpg"),
            ("fade_ab",   f"{stem}_fade_03_alpha_blend.jpg"),
            ("fade_inp",  f"{stem}_fade_04_inpaint.jpg"),
        ]
        saved = 0
        for key, fname in mapping:
            img = result.get(key)
            if img is not None:
                cv2.imwrite(os.path.join(out_dir, fname), img)
                saved += 1
        return saved


# ================================================================================
#  Entry point
# ================================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Cap Text Fade Generator")
    win = CapTextFadeUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Rotate then crop text lines (CRAFT rotation + CRAFT line detection).

Pipeline (aligned with test_cap_rotation_ui / business_logic):
1) Optional enhance_cap_sharpness (BGR) — same as test UI
2) rotationCRAFT.detect_text_and_rotate(image_array=...)
3) core.cap_rotation.apply_craft_rotation — longest-edge box → horizontal (replaces CRAFT's default rotate)
4) craft_line_detection: detect lines on rotated_image (RGB) -> cropped_lines
5) Save cropped line PNGs ลงโฟลเดอร์ output โดยตรง (ไม่สร้างโฟลเดอร์ย่อย) ชื่อ `{stem}_line_N.png`
   Optional --debug: ภาพหมุน + annotate (`{stem}_rotated.png` …)
"""

from __future__ import annotations

import argparse
import os
import sys
import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np


# --- Ensure we can import from project root (final/) ---
SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DIR = SCRIPT_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

# --- Fix Windows stdout/stderr encoding before importing modules ---
# Some modules print unicode symbols (e.g. ⚠️) inside core/cuda_setup.py.
# If stdout uses a legacy code page, the import may crash with UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Python 3.7+
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from config.settings import CRAFT_MODEL_PATH, CRAFT_REFINER_PATH
from core.cap_rotation import apply_craft_rotation
from core.image_processor import enhance_cap_sharpness
from libs.processing.rotationCRAFT import CRAFTTextDetector
from libs.processing.craft_line_detection import CRAFTLineDetector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(input_path: Path, recursive: bool = False) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTS:
            yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if recursive:
        it = input_path.rglob("*")
    else:
        it = input_path.glob("*")

    for p in it:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def bool_from_arg(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def flat_file_prefix(img_path: Path, stem_counts: Dict[str, int]) -> str:
    """
    ชื่อ prefix สำหรับไฟล์ในโฟลเดอร์เดียว: ใช้ stem ครั้งแรก, ถ้าซ้ำใช้ stem_2, stem_3 …
    """
    stem = img_path.stem
    stem_counts[stem] = stem_counts.get(stem, 0) + 1
    n = stem_counts[stem]
    if n == 1:
        return stem
    return f"{stem}_{n}"


def apply_rotation_like_test_ui(image_bgr: np.ndarray, rotation_result: dict) -> None:
    """
    หมุนแบบเดียวกับ test_cap_rotation_ui / business_logic: ใช้ apply_craft_rotation บนภาพเดียวกับที่ส่งเข้า CRAFT
    อัปเดต rotation_result['rotated_image'] เป็น RGB (สำหรับ craft_line_detection)
    """
    rotated_bgr, angle_edge, rotate_deg = apply_craft_rotation(image_bgr, rotation_result)
    if rotated_bgr is not None:
        rotation_result["rotated_image"] = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
        rotation_result["rotation_angle"] = float(rotate_deg)
        rotation_result["craft_angle_edge_deg"] = float(angle_edge) if angle_edge is not None else None
    # ถ้าไม่มีกรอบ: คง rotated_image เดิมจาก CRAFT (RGB)


def save_cropped_line_images_only(out_dir: Path, line_result: dict, file_prefix: str) -> int:
    """บันทึกเฉพาะ PNG แต่ละบรรทัด — ไม่มี JSON / ภาพหมุน"""
    cropped = line_result.get("cropped_lines") or []
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line_data in cropped:
        idx = int(line_data.get("line_index", n))
        line_img = line_data.get("image")
        if line_img is None:
            continue
        path = out_dir / f"{file_prefix}_line_{idx + 1}.png"
        if len(line_img.shape) == 3 and line_img.shape[2] == 3:
            bgr = cv2.cvtColor(line_img, cv2.COLOR_RGB2BGR)
        else:
            bgr = line_img
        cv2.imwrite(str(path), bgr)
        n += 1
    return n


def process_one_image_pipeline(
    img_path: Path,
    rotation_det: CRAFTTextDetector,
    line_det: CRAFTLineDetector,
    out_dir: Path,
    file_prefix: str,
    *,
    sharpen: bool,
    save_debug: bool = False,
) -> int:
    """โหลดภาพ → (optional) sharpen → CRAFT → apply_craft_rotation → crop lines → บันทึกแค่ PNG บรรทัด
    คืนจำนวนบรรทัดที่บันทึก"""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    if sharpen:
        enhanced = enhance_cap_sharpness(img_bgr)
        img_for_craft = enhanced if enhanced is not None else img_bgr
    else:
        img_for_craft = img_bgr

    rotation_result = rotation_det.detect_text_and_rotate(image_array=img_for_craft)
    if not rotation_result or "error" in rotation_result:
        raise RuntimeError(
            rotation_result.get("error", "Unknown error") if isinstance(rotation_result, dict) else "Unknown error"
        )
    rotation_result["image_path"] = str(img_path)
    apply_rotation_like_test_ui(img_for_craft, rotation_result)

    if save_debug:
        rotated_out = out_dir / f"{file_prefix}_rotated.png"
        rotation_det.save_rotated_image(str(rotated_out), result=rotation_result)
        annotated_out = out_dir / f"{file_prefix}_rotation_annotated.png"
        annotated_bgr = rotation_det.draw_detections(result=rotation_result)
        cv2.imwrite(str(annotated_out), annotated_bgr)

    line_result = line_det.detect_lines_from_rotation_result(rotation_result)
    if not line_result or "error" in line_result:
        raise RuntimeError(
            line_result.get("error", "Unknown error") if isinstance(line_result, dict) else "Unknown error"
        )
    return save_cropped_line_images_only(out_dir, line_result, file_prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate image using rotationCRAFT, then crop text lines using craft_line_detection."
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch UI to select multiple images and process them."
    )
    parser.add_argument(
        "--input",
        required=False,
        default=None,
        help="Input image file or folder.",
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="Output folder — line PNGs saved flat as {name}_line_N.png",
    )
    parser.add_argument(
        "--recursive",
        default="false",
        help="If input is a folder, search recursively. (true/false)",
    )
    parser.add_argument(
        "--craft-model",
        default=str(CRAFT_MODEL_PATH),
        help="CRAFT main model .pth path",
    )
    parser.add_argument(
        "--craft-refiner",
        default=str(CRAFT_REFINER_PATH),
        help="CRAFT refiner model .pth path (optional; may be missing).",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU (by not using CUDA if available).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="บันทึกเพิ่ม: ภาพหมุน + ภาพ CRAFT annotate (ค่าเริ่มต้นบันทึกแค่ PNG แต่ละบรรทัด)",
    )
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Skip enhance_cap_sharpness before CRAFT (default: sharpen on, same as test_cap_rotation_ui).",
    )
    return parser.parse_args()


def run_ui() -> int:
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    from queue import Queue, Empty
    import threading

    class App:
        def __init__(self, root: tk.Tk):
            self.root = root
            self.root.title("ครอปบรรทัด — CRAFT")
            self.root.geometry("520x300")
            self.root.minsize(480, 260)

            self.queue: "Queue[tuple[str, object]]" = Queue()
            self.stop_event = threading.Event()
            self.worker_thread: Optional[threading.Thread] = None
            self.images: List[Path] = []

            self._build_ui()
            self._poll_queue()

        def _build_ui(self) -> None:
            main = ttk.Frame(self.root, padding=12)
            main.pack(fill=tk.BOTH, expand=True)

            row1 = ttk.Frame(main)
            row1.pack(fill=tk.X)
            ttk.Button(row1, text="เลือกภาพ", command=self.add_images).pack(side=tk.LEFT)
            ttk.Button(row1, text="โฟลเดอร์ภาพ", command=self.add_folder).pack(side=tk.LEFT, padx=(8, 0))
            ttk.Button(row1, text="ล้าง", command=self.clear_images).pack(side=tk.LEFT, padx=(8, 0))
            self.count_var = tk.StringVar(value="0 ไฟล์")
            ttk.Label(row1, textvariable=self.count_var).pack(side=tk.RIGHT)

            row2 = ttk.Frame(main)
            row2.pack(fill=tk.X, pady=(10, 0))
            ttk.Label(row2, text="บันทึกที่:").pack(side=tk.LEFT)
            self.output_var = tk.StringVar(value="")
            ttk.Entry(row2, textvariable=self.output_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
            ttk.Button(row2, text="เลือก…", command=self.choose_output_folder).pack(side=tk.LEFT)

            opt = ttk.Frame(main)
            opt.pack(fill=tk.X, pady=(10, 0))
            self.debug_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                opt,
                text="บันทึก debug (ภาพหมุน + CRAFT annotate)",
                variable=self.debug_var,
            ).pack(side=tk.LEFT)
            self.cpu_only_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(opt, text="ใช้ CPU เท่านั้น", variable=self.cpu_only_var).pack(side=tk.LEFT, padx=(16, 0))

            row3 = ttk.Frame(main)
            row3.pack(fill=tk.X, pady=(12, 0))
            self.start_btn = ttk.Button(row3, text="เริ่ม", command=self.start)
            self.start_btn.pack(side=tk.LEFT)
            self.stop_btn = ttk.Button(row3, text="หยุด", command=self.stop, state=tk.DISABLED)
            self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
            self.progress = ttk.Progressbar(row3, mode="determinate")
            self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))

            self.status_var = tk.StringVar(value="พร้อม — บันทึก PNG แต่ละบรรทัดลงโฟลเดอร์ที่เลือก (ไม่แยกโฟลเดอร์ย่อย)")
            ttk.Label(main, textvariable=self.status_var, wraplength=480).pack(anchor="w", pady=(12, 0))

        def _sync_count(self) -> None:
            n = len(self.images)
            self.count_var.set(f"{n} ไฟล์")

        def add_images(self) -> None:
            file_paths = filedialog.askopenfilenames(
                title="เลือกภาพ",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
            )
            for fp in file_paths:
                p = Path(fp)
                if p not in self.images:
                    self.images.append(p)
            self._sync_count()

        def add_folder(self) -> None:
            folder = filedialog.askdirectory(title="โฟลเดอร์ภาพ")
            if not folder:
                return
            folder_path = Path(folder)
            for p in folder_path.glob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p not in self.images:
                    self.images.append(p)
            self._sync_count()

        def clear_images(self) -> None:
            self.images = []
            self._sync_count()

        def choose_output_folder(self) -> None:
            folder = filedialog.askdirectory(title="โฟลเดอร์บันทึก")
            if folder:
                self.output_var.set(folder)

        def _validate_before_start(self) -> Optional[str]:
            if not self.images:
                return "เลือกภาพก่อน"
            out_dir = self.output_var.get().strip()
            if not out_dir:
                return "เลือกโฟลเดอร์บันทึก"
            if not Path(CRAFT_MODEL_PATH).exists():
                return f"ไม่พบโมเดล CRAFT: {CRAFT_MODEL_PATH}"
            return None

        def start(self) -> None:
            err = self._validate_before_start()
            if err:
                messagebox.showerror("ไม่สามารถเริ่ม", err)
                return

            self.stop_event.clear()
            self.stop_btn.configure(state=tk.NORMAL)
            self.start_btn.configure(state=tk.DISABLED)

            output_root = Path(self.output_var.get().strip())
            output_root.mkdir(parents=True, exist_ok=True)

            craft_model_path = Path(CRAFT_MODEL_PATH)
            ref_path = Path(CRAFT_REFINER_PATH)
            craft_refiner: Optional[str] = str(ref_path) if ref_path.is_file() else None

            use_cuda = not self.cpu_only_var.get()
            save_debug = self.debug_var.get()
            sharpen = True

            images = list(self.images)
            total = len(images)
            self.progress.configure(maximum=total, value=0)
            self.status_var.set("กำลังโหลดโมเดล…")

            def worker():
                try:
                    rotation_det = CRAFTTextDetector(
                        model_path=str(craft_model_path),
                        refiner_path=craft_refiner,
                        cuda=use_cuda,
                    )
                    line_det = CRAFTLineDetector(
                        model_path=str(craft_model_path),
                        refiner_path=craft_refiner,
                        cuda=use_cuda,
                    )
                    stem_counts: Dict[str, int] = {}

                    for i, img_path in enumerate(images, start=1):
                        if self.stop_event.is_set():
                            self.queue.put(("done",))
                            return
                        prefix = flat_file_prefix(img_path, stem_counts)

                        self.queue.put(("progress", i, total, img_path.name))
                        n_lines = process_one_image_pipeline(
                            img_path,
                            rotation_det,
                            line_det,
                            output_root,
                            prefix,
                            sharpen=sharpen,
                            save_debug=save_debug,
                        )
                        self.queue.put(("status", f"{img_path.name}: บันทึก {n_lines} บรรทัด"))

                    self.queue.put(("done",))
                except Exception as e:
                    self.queue.put(("error", str(e)))

            threading.Thread(target=worker, daemon=True).start()

        def stop(self) -> None:
            self.stop_event.set()
            self.status_var.set("กำลังหยุด…")

        def _poll_queue(self) -> None:
            try:
                while True:
                    kind, *payload = self.queue.get_nowait()
                    if kind == "status":
                        self.status_var.set(str(payload[0]))
                    elif kind == "progress":
                        i, total, name = payload
                        self.progress.configure(value=i, maximum=total)
                        self.status_var.set(f"({i}/{total}) {name}")
                    elif kind == "done":
                        self.start_btn.configure(state=tk.NORMAL)
                        self.stop_btn.configure(state=tk.DISABLED)
                        self.status_var.set("เสร็จแล้ว")
                    elif kind == "error":
                        err = payload[0]
                        self.status_var.set(f"ผิดพลาด: {err}")
                        messagebox.showerror("Error", err)
                        self.start_btn.configure(state=tk.NORMAL)
                        self.stop_btn.configure(state=tk.DISABLED)
            except Empty:
                pass
            self.root.after(200, self._poll_queue)

    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


def main() -> int:
    args = parse_args()

    # Use UI as the primary mode:
    # - if user explicitly passes --ui => UI
    # - if user runs without --input/--output => UI
    if args.ui or (args.input is None and args.output is None):
        return run_ui()

    if not args.input or not args.output:
        raise SystemExit("CLI mode requires --input and --output (or run without args to start UI).")

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    recursive = bool_from_arg(args.recursive)
    sharpen = not args.no_sharpen
    save_debug = args.debug

    craft_model_path = Path(args.craft_model)
    craft_refiner_path = Path(args.craft_refiner)

    if not craft_model_path.exists():
        raise FileNotFoundError(f"CRAFT model not found: {craft_model_path}")

    refiner_ok = craft_refiner_path.exists() and craft_refiner_path.is_file()
    refiner_path: Optional[str] = str(craft_refiner_path) if refiner_ok else None
    if not refiner_ok:
        print(f"⚠️ CRAFT refiner not found (optional): {craft_refiner_path}")

    use_cuda = not args.no_cuda

    print("Loading detectors...")
    rotation_det = CRAFTTextDetector(
        model_path=str(craft_model_path),
        refiner_path=refiner_path,
        cuda=use_cuda,
    )
    line_det = CRAFTLineDetector(
        model_path=str(craft_model_path),
        refiner_path=refiner_path,
        cuda=use_cuda,
    )

    images: List[Path] = list(iter_images(input_path, recursive=recursive))
    if not images:
        print(f"No images found under: {input_path}")
        return 2

    ok = 0
    failed = 0
    stem_counts: Dict[str, int] = {}

    for idx, img_path in enumerate(images, start=1):
        prefix = flat_file_prefix(img_path, stem_counts)

        print(f"[{idx}/{len(images)}] Processing: {img_path}")

        try:
            n_lines = process_one_image_pipeline(
                img_path,
                rotation_det,
                line_det,
                output_root,
                prefix,
                sharpen=sharpen,
                save_debug=save_debug,
            )
            print(f"   → บันทึก {n_lines} บรรทัด → {output_root}")
            ok += 1
        except Exception as e:
            failed += 1
            err_path = output_root / f"{prefix}_error.txt"
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(str(e))
            print(f"❌ Failed: {img_path}\n   Error: {e}")

    print(f"Done. Success: {ok}, Failed: {failed}, Total: {len(images)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


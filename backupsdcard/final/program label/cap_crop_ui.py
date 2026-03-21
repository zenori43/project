"""
โปรแกรมครอปภาพฝาล่างสุด (bottommost cap) จากภาพ
ใช้โมเดล cap.pt ตรวจจับฝา แล้วเลือกเฉพาะฝาที่อยู่ล่างสุด (y สูงสุด) ครอปแล้วบันทึก
Logic เหมือนโปรแกรมหลัก: detect -> crop_detections(margin) -> เลือกฝาที่ y_center สูงสุด
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
import threading
import sys

# เพิ่ม path สำหรับ import modules จากโฟลเดอร์ final
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

try:
    from libs.detection.capmodel import CapDetector
except ImportError as e:
    print(f"Error importing capmodel: {e}")
    try:
        messagebox.showerror("Error", f"ไม่สามารถโหลด capmodel ได้: {e}\n\nรันจากโฟลเดอร์ final หรือให้ path ชี้ไปที่ final")
    except Exception:
        pass
    sys.exit(1)


def get_default_cap_model_path():
    """ path เริ่มต้นของ cap.pt (โฟลเดอร์ final/models/detection/cap.pt) """
    final_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(final_dir, "models", "detection", "cap.pt")


class CapCropUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cap Crop Tool - ครอปฝาล่างสุด")
        self.root.geometry("800x580")

        self.model_path = get_default_cap_model_path()
        self.detector = None
        self.model_loaded = False

        self.selected_files = []
        self.output_folder = ""
        self.processing = False

        self.confidence_threshold = 0.5
        self.crop_margin = 20

        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Model path
        ttk.Label(main_frame, text="Model (cap.pt):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_path_var = tk.StringVar(value=self.model_path)
        model_entry = ttk.Entry(main_frame, textvariable=self.model_path_var, width=55)
        model_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, pady=5)

        self.model_status_var = tk.StringVar(value="กำลังโหลดโมเดล...")
        ttk.Label(main_frame, textvariable=self.model_status_var, foreground="blue").grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=2
        )

        # Confidence
        ttk.Label(main_frame, text="Confidence:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.confidence_var = tk.DoubleVar(value=self.confidence_threshold)
        confidence_scale = ttk.Scale(
            main_frame, from_=0.1, to=1.0, variable=self.confidence_var, orient=tk.HORIZONTAL
        )
        confidence_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        self.confidence_label = ttk.Label(main_frame, text=f"{self.confidence_threshold:.2f}")
        self.confidence_label.grid(row=2, column=2, pady=5)
        confidence_scale.configure(command=self._on_confidence_change)

        # Margin
        ttk.Label(main_frame, text="Margin (px):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.margin_var = tk.IntVar(value=self.crop_margin)
        margin_spin = ttk.Spinbox(main_frame, from_=0, to=100, width=6, textvariable=self.margin_var)
        margin_spin.grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)

        # Files
        ttk.Label(main_frame, text="ภาพที่เลือก:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Button(main_frame, text="เลือกภาพ", command=self.select_files).grid(row=4, column=1, sticky=tk.W, pady=5, padx=5)

        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=8)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)

        # Output folder
        ttk.Label(main_frame, text="โฟลเดอร์บันทึก:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.output_folder_var = tk.StringVar()
        output_entry = ttk.Entry(main_frame, textvariable=self.output_folder_var, width=55)
        output_entry.grid(row=6, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_folder).grid(row=6, column=2, pady=5)

        # Progress
        self.progress_var = tk.StringVar(value="พร้อมใช้งาน")
        ttk.Label(main_frame, textvariable=self.progress_var, foreground="green").grid(
            row=7, column=0, columnspan=3, sticky=tk.W, pady=5
        )
        self.progress_bar = ttk.Progressbar(main_frame, mode="determinate")
        self.progress_bar.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        self.process_button = ttk.Button(
            main_frame, text="ครอปฝาล่างสุดและบันทึก", command=self.process_images, state=tk.DISABLED
        )
        self.process_button.grid(row=9, column=0, columnspan=3, pady=10)

    def _on_confidence_change(self, value):
        self.confidence_threshold = float(value)
        self.confidence_label.config(text=f"{self.confidence_threshold:.2f}")

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="เลือกไฟล์โมเดล cap.pt",
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")]
        )
        if path:
            self.model_path_var.set(path)
            self.model_path = path
            self.load_model()

    def load_model(self):
        def do_load():
            try:
                path = self.model_path_var.get().strip()
                if not path or not os.path.exists(path):
                    self.model_status_var.set(f"❌ ไม่พบโมเดล: {path}")
                    self.model_loaded = False
                    self._check_ready()
                    return
                self.model_status_var.set("กำลังโหลด...")
                conf = float(self.confidence_var.get())
                self.detector = CapDetector(model_path=path, conf_threshold=conf)
                self.model_loaded = True
                self.model_status_var.set("✅ โมเดลโหลดสำเร็จ (ครอปฝาล่างสุดได้)")
                self._check_ready()
            except Exception as e:
                self.model_status_var.set(f"❌ โหลดไม่สำเร็จ: {e}")
                self.model_loaded = False
                self._check_ready()

        threading.Thread(target=do_load, daemon=True).start()

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="เลือกภาพ",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All", "*.*")
            ]
        )
        if files:
            self.selected_files = list(files)
            self.file_listbox.delete(0, tk.END)
            for f in self.selected_files:
                self.file_listbox.insert(tk.END, os.path.basename(f))
            self._check_ready()

    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="โฟลเดอร์บันทึกผล")
        if folder:
            self.output_folder_var.set(folder)
            self.output_folder = folder
            self._check_ready()

    def _check_ready(self):
        if self.model_loaded and self.selected_files and self.output_folder_var.get():
            self.process_button.config(state=tk.NORMAL)
        else:
            self.process_button.config(state=tk.DISABLED)

    def process_images(self):
        if self.processing:
            messagebox.showwarning("แจ้ง", "กำลังประมวลผลอยู่")
            return
        if not self.model_loaded:
            messagebox.showerror("Error", "โมเดลยังไม่โหลด")
            return
        if not self.selected_files:
            messagebox.showerror("Error", "กรุณาเลือกภาพ")
            return
        out = self.output_folder_var.get()
        if not out:
            messagebox.showerror("Error", "กรุณาเลือกโฟลเดอร์บันทึก")
            return

        os.makedirs(out, exist_ok=True)
        self.processing = True
        self.process_button.config(state=tk.DISABLED)
        self.progress_bar["maximum"] = len(self.selected_files)
        self.progress_bar["value"] = 0
        margin = int(self.margin_var.get()) if self.margin_var.get() else 20
        # อัปเดต confidence ของ detector ถ้า user เปลี่ยน
        try:
            self.detector.conf_threshold = float(self.confidence_var.get())
        except Exception:
            pass
        threading.Thread(target=self._process_thread, args=(out, margin), daemon=True).start()

    def _process_thread(self, output_folder, margin):
        try:
            total = len(self.selected_files)
            ok = 0
            skip = 0
            err = 0

            for idx, image_path in enumerate(self.selected_files):
                try:
                    self.progress_var.set(f"กำลังทำ: {os.path.basename(image_path)} ({idx+1}/{total})")
                    self.root.update_idletasks()

                    img = cv2.imread(image_path)
                    if img is None:
                        err += 1
                        continue

                    # ตรวจจับฝา (ใช้ image_array เพื่อให้ detector เก็บ last_processed_image / last_detection_result)
                    result = self.detector.detect_caps(image_array=img)
                    detections = result.get("detections", [])
                    if not detections:
                        skip += 1
                        continue

                    # ครอปทุกฝา
                    cropped_list = self.detector.crop_detections(margin=margin)

                    # เลือกฝาล่างสุด = y สูงสุด (เหมือนโปรแกรมหลัก)
                    bottommost_index = 0
                    max_y = -1
                    for i, det in enumerate(detections):
                        bbox = det.get("bbox", [0, 0, 0, 0])
                        if len(bbox) >= 4:
                            y1, y2 = bbox[1], bbox[3]
                            y_center = (y1 + y2) / 2
                            if y_center > max_y:
                                max_y = y_center
                                bottommost_index = i

                    cap_crop = cropped_list[bottommost_index]
                    base = Path(image_path).stem
                    out_path = os.path.join(output_folder, f"{base}_cap.jpg")
                    cv2.imwrite(out_path, cap_crop)
                    ok += 1

                except Exception as e:
                    print(f"Error {image_path}: {e}")
                    err += 1

                self.progress_bar["value"] = idx + 1
                self.root.update_idletasks()

            msg = f"เสร็จสิ้น | บันทึกได้: {ok}, ไม่พบฝา: {skip}, ผิดพลาด: {err}"
            self.progress_var.set(msg)
            messagebox.showinfo("เสร็จสิ้น", f"{msg}\n\nโฟลเดอร์: {output_folder}")

        except Exception as e:
            self.progress_var.set(f"เกิดข้อผิดพลาด: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.processing = False
            self.process_button.config(state=tk.NORMAL)
            self.progress_bar["value"] = 0


def main():
    root = tk.Tk()
    app = CapCropUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

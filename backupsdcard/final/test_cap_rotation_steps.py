# -*- coding: utf-8 -*-
"""
ทดสอบระบบหมุนฝา (จากโปรแกรมหลัก) แบบทีละขั้น
ลำดับ: ตรวจจับตำแหน่งฝา (YOLO) -> ครอปฝา -> ปรับความคมชัด -> หมุนด้วย CRAFT (ไม่รวม OCR)

รันจากโฟลเดอร์ final:
  python test_cap_rotation_steps.py [path_to_image]
  python test_cap_rotation_steps.py
  (ถ้าไม่ระบุ path จะใช้ temp_sentech_image.jpg ถ้ามี หรือให้ใส่ path)
"""

import sys
import os
import argparse
from pathlib import Path

# ให้ import config และ libs ได้ (รันจากที่ไหนก็ได้)
FINAL_DIR = Path(__file__).resolve().parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))
os.chdir(FINAL_DIR)

# CUDA setup ก่อน import torch/cv2
try:
    from core.cuda_setup import setup_cuda_paths
    setup_cuda_paths()
except ImportError:
    pass

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="ทดสอบขั้นตอนหมุนฝา: ตรวจจับฝา -> ครอป -> ความคมชัด -> CRAFT")
    parser.add_argument("image", nargs="?", default=None, help="path to cap/bottle image (default: temp_sentech_image.jpg)")
    parser.add_argument("-o", "--output-dir", default="test_cap_rotation_steps", help="โฟลเดอร์เก็บภาพแต่ละขั้น")
    parser.add_argument("--no-show", action="store_true", help="ไม่เปิด cv2.imshow แค่บันทึกไฟล์")
    args = parser.parse_args()

    image_path = args.image or "temp_sentech_image.jpg"
    if not os.path.isabs(image_path):
        image_path = os.path.join(FINAL_DIR, image_path)
    if not os.path.exists(image_path):
        print(f"ไม่พบไฟล์ภาพ: {image_path}")
        print("ใช้: python test_cap_rotation_steps.py <path_to_image>")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"โฟลเดอร์ผลลัพธ์: {out_dir.resolve()}")

    # โหลดภาพ
    image = cv2.imread(image_path)
    if image is None:
        print(f"โหลดภาพไม่สำเร็จ: {image_path}")
        return 1
    print(f"โหลดภาพแล้ว: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # --- Step 0: ภาพต้นฉบับ ---
    step0_path = out_dir / "step0_original.jpg"
    cv2.imwrite(str(step0_path), image)
    print(f"[Step 0] บันทึกภาพต้นฉบับ: {step0_path}")

    # --- โหลดโมเดล Cap (YOLO) ---
    from config.settings import CAP_MODEL_PATH, CRAFT_MODEL_PATH, CRAFT_REFINER_PATH
    from libs.detection.capmodel import initialize_detector as init_cap_detector

    print("กำลังโหลด Cap detector (YOLO)...")
    cap_detector = init_cap_detector(CAP_MODEL_PATH)
    if cap_detector is None or (hasattr(cap_detector, "model") and cap_detector.model is None):
        print("โหลด Cap detector ไม่ได้ — ข้ามขั้นตอนตรวจจับฝา")
        cap_detector = None
    else:
        print("โหลด Cap detector สำเร็จ")

    # --- Step 1: ตรวจจับตำแหน่งฝา ---
    if cap_detector is not None:
        cap_result = cap_detector.detect_caps(image_path=image_path)
        detections = cap_result.get("detections", [])
        print(f"[Step 1] ตรวจจับฝา: พบ {len(detections)} ฝา")

        # วาด bbox บนภาพ
        img_det = image.copy()
        for i, d in enumerate(detections):
            bbox = d.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(x) for x in bbox]
                cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf = d.get("confidence", 0)
                cv2.putText(img_det, f"cap {i+1}: {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        step1_path = out_dir / "step1_cap_detections.jpg"
        cv2.imwrite(str(step1_path), img_det)
        print(f"[Step 1] บันทึกภาพที่มี bbox ฝา: {step1_path}")

        # --- Step 2: ครอปตามตำแหน่งฝา ---
        cropped_list = cap_detector.crop_detections(margin=20)
        print(f"[Step 2] ครอปฝา: ได้ {len(cropped_list)} ภาพ")

        for i, crop in enumerate(cropped_list):
            p = out_dir / f"step2_cropped_cap_{i+1}.jpg"
            cv2.imwrite(str(p), crop)
            print(f"  -> ฝาที่ {i+1}: {p}")

        # เลือกฝาล่างสุด (y สูงสุด)
        if cropped_list and detections:
            bottommost_index = 0
            max_y = 0
            for i, d in enumerate(detections):
                bbox = d.get("bbox", [])
                if len(bbox) >= 4:
                    y2 = bbox[3]
                    if y2 > max_y:
                        max_y = y2
                        bottommost_index = i
            cap_to_use = cropped_list[bottommost_index]
            print(f"[Step 2] เลือกฝาล่างสุด: ฝาที่ {bottommost_index + 1}")
        else:
            cap_to_use = image.copy()
            print("[Step 2] ไม่มีฝาที่ตรวจจับได้ — ใช้ทั้งภาพ")
    else:
        cap_to_use = image.copy()
        cropped_list = []
        print("[Step 1–2] ข้ามการตรวจจับ/ครอปฝา — ใช้ทั้งภาพ")

    step2_one_path = out_dir / "step2_bottommost_cap.jpg"
    cv2.imwrite(str(step2_one_path), cap_to_use)
    print(f"[Step 2] บันทึกฝาที่เลือก (หรือทั้งภาพ): {step2_one_path}")

    # --- Step 3: ปรับความคมชัด (optional) ---
    try:
        from core.image_processor import enhance_cap_sharpness
        enhanced = enhance_cap_sharpness(cap_to_use)
        if enhanced is not None:
            cap_for_craft = enhanced
            step3_path = out_dir / "step3_enhanced.jpg"
            cv2.imwrite(str(step3_path), enhanced)
            if os.path.exists(step3_path):
                print(f"[Step 3] ปรับความคมชัดแล้ว: {step3_path}")
        else:
            cap_for_craft = cap_to_use
            print("[Step 3] ปรับความคมชัดไม่ได้ — ใช้ภาพเดิม")
    except Exception as e:
        cap_for_craft = cap_to_use
        print(f"[Step 3] ข้ามความคมชัด: {e}")

    # BGR สำหรับ CRAFT (รับ BGR ได้)
    if hasattr(cap_for_craft, 'shape') and len(cap_for_craft.shape) == 3 and cap_for_craft.shape[2] == 3:
        pass  # CRAFT รับ BGR แล้วแปลงในตัว
    cap_for_craft_bgr = cap_for_craft if cap_for_craft is not None else cap_to_use
    if len(cap_for_craft_bgr.shape) == 3 and cap_for_craft_bgr.shape[2] == 3:
        pass
    else:
        cap_for_craft_bgr = cv2.cvtColor(cap_for_craft_bgr, cv2.COLOR_GRAY2BGR)

    # --- Step 4: หมุนด้วย CRAFT (ไม่รวม OCR) ---
    from libs.processing.rotationCRAFT import initialize_detector as init_craft_detector

    print("กำลังโหลด CRAFT detector...")
    craft_detector = init_craft_detector(CRAFT_MODEL_PATH, CRAFT_REFINER_PATH)
    if craft_detector is None or (hasattr(craft_detector, "model") and craft_detector.model is None):
        print("โหลด CRAFT ไม่ได้ — ข้ามขั้นตอนหมุนด้วย CRAFT")
        craft_result = None
    else:
        print("โหลด CRAFT สำเร็จ กำลังรัน detect_text_and_rotate...")
        craft_result = craft_detector.detect_text_and_rotate(image_array=cap_for_craft_bgr)

    if craft_result is not None:
        angle = craft_result.get("rotation_angle", 0)
        rotated = craft_result.get("rotated_image")
        print(f"[Step 4] CRAFT: มุมหมุน = {angle:.2f}°")
        if rotated is not None:
            # CRAFT คืน RGB
            if len(rotated.shape) == 3:
                rot_bgr = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
            else:
                rot_bgr = rotated
            step4_path = out_dir / "step4_craft_rotated.jpg"
            cv2.imwrite(str(step4_path), rot_bgr)
            print(f"[Step 4] บันทึกภาพหลังหมุน CRAFT: {step4_path}")
        else:
            print("[Step 4] CRAFT ไม่ได้คืน rotated_image")
    else:
        print("[Step 4] ไม่มีผลจาก CRAFT")

    # แสดงผล (optional)
    if not args.no_show:
        try:
            cv2.imshow("Step 0: Original", image)
            cv2.imshow("Step 2: Cap (selected)", cap_to_use)
            if craft_result and craft_result.get("rotated_image") is not None:
                rot = craft_result["rotated_image"]
                if len(rot.shape) == 3:
                    rot = cv2.cvtColor(rot, cv2.COLOR_RGB2BGR)
                cv2.imshow("Step 4: CRAFT rotated", rot)
            print("กดปุ่มใดก็ตามในหน้าต่างภาพเพื่อปิด")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"ไม่สามารถแสดงภาพ: {e}")

    print("เสร็จสิ้น — ดูผลแต่ละขั้นได้ที่โฟลเดอร์:", out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

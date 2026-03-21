import os
import random
import shutil

"""
อ่านไฟล์ line_gt.txt แล้วแบ่งเป็น train/val ตามสัดส่วนที่กำหนด
โดยสร้างไฟล์:
    - line_gt_train.txt
    - line_gt_val.txt
และคัดลอกไฟล์ภาพไปไว้ในโฟลเดอร์:
    - train_images
    - val_images
"""

GT_FILE = "line_gt.txt"
TRAIN_GT_FILE = "line_gt_train.txt"
VAL_GT_FILE = "line_gt_val.txt"
TRAIN_IMG_DIR = "train_images"
VAL_IMG_DIR = "val_images"

# สัดส่วน train/val (เช่น 0.8 = 80% train, 20% val)
TRAIN_RATIO = 0.8


def main():
    if not os.path.exists(GT_FILE):
        raise FileNotFoundError(f"{GT_FILE} not found")

    with open(GT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        raise ValueError("GT file is empty.")

    # สุ่มลำดับก่อนแบ่ง
    random.shuffle(lines)

    train_count = int(len(lines) * TRAIN_RATIO)
    train_lines = lines[:train_count]
    val_lines = lines[train_count:]

    # เตรียมโฟลเดอร์เก็บภาพ train/val
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)

    # เขียนไฟล์ GT และคัดลอกภาพสำหรับ train
    with open(TRAIN_GT_FILE, "w", encoding="utf-8") as f_train:
        for line in train_lines:
            f_train.write(line + "\n")

            img_rel_path, _ = line.split("\t", 1)
            src_img_path = os.path.join(os.path.dirname(GT_FILE), img_rel_path)
            img_name = os.path.basename(img_rel_path)
            dst_img_path = os.path.join(TRAIN_IMG_DIR, img_name)

            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)

    # เขียนไฟล์ GT และคัดลอกภาพสำหรับ val
    with open(VAL_GT_FILE, "w", encoding="utf-8") as f_val:
        for line in val_lines:
            f_val.write(line + "\n")

            img_rel_path, _ = line.split("\t", 1)
            src_img_path = os.path.join(os.path.dirname(GT_FILE), img_rel_path)
            img_name = os.path.basename(img_rel_path)
            dst_img_path = os.path.join(VAL_IMG_DIR, img_name)

            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)

    print(f"Total samples: {len(lines)}")
    print(f"Train samples ({TRAIN_RATIO*100:.0f}%): {len(train_lines)} -> {TRAIN_GT_FILE}, images in '{TRAIN_IMG_DIR}'")
    print(f"Val samples ({(1-TRAIN_RATIO)*100:.0f}%): {len(val_lines)} -> {VAL_GT_FILE}, images in '{VAL_IMG_DIR}'")


if __name__ == "__main__":
    main()


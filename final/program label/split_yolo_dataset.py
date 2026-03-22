import random
import shutil
from pathlib import Path


# Base YOLO dataset directory (มี images/ และ labels/ ข้างใน)
BASE_DIR = Path(r"C:\Users\Win 10 Home\Desktop\data")

# Train/Val/Test อัตราส่วน (รวมกันควรได้ 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def collect_pairs(images_dir: Path, labels_dir: Path):
    """เก็บคู่ (image, label) ที่ชื่อ base เดียวกัน"""
    image_paths = sorted(images_dir.glob("*.png"))

    pairs = []
    for img in image_paths:
        label = labels_dir / (img.stem + ".txt")
        if label.exists():
            pairs.append((img, label))
        else:
            print(f"⚠️ ไม่มี label สำหรับรูป: {img.name} (ข้าม)")

    return pairs


def make_output_dirs(base: Path):
    """สร้างโครงสร้างโฟลเดอร์ train/ val/ test/ พร้อม images/ labels/"""
    splits = {}
    for split in ["train", "val", "test"]:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        splits[split] = (img_dir, lbl_dir)
    return splits


def split_dataset():
    images_dir = BASE_DIR / "images"
    labels_dir = BASE_DIR / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ ไม่พบโฟลเดอร์ images หรือ labels ใต้ {BASE_DIR}")
        return

    pairs = collect_pairs(images_dir, labels_dir)
    total = len(pairs)
    if total == 0:
        print("❌ ไม่พบคู่ image+label ที่ match กันเลย")
        return

    print(f"พบรูปที่มี label ครบทั้งหมด: {total} ภาพ")

    # สุ่มลำดับแบบ reproducible
    random.seed(42)
    random.shuffle(pairs)

    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    n_test = total - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # สร้างโฟลเดอร์ปลายทาง
    splits = make_output_dirs(BASE_DIR)

    def copy_pairs(pairs_list, split_name: str):
        img_dir, lbl_dir = splits[split_name]
        for img_path, lbl_path in pairs_list:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_dir / lbl_path.name)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(test_pairs, "test")

    # ใช้ข้อความธรรมดา (ไม่ใช้ emoji) กันปัญหา encoding บน Windows console
    print("แบ่งข้อมูลเรียบร้อยแล้วที่:")
    print(f"  {BASE_DIR / 'train'}")
    print(f"  {BASE_DIR / 'val'}")
    print(f"  {BASE_DIR / 'test'}")


if __name__ == "__main__":
    split_dataset()


import os
from pathlib import Path


LABELS_DIR = Path(r"C:\Users\Win 10 Home\Desktop\project_final\final\captured_images\usb_camera\labels")
TARGET_CLASS_ID = "6"  # angle3


def process_label_file(path: Path):
    """ลบเฉพาะบรรทัด class 6 (angle3) ถ้าในไฟล์มี label อื่นปนด้วย
    ถ้าไฟล์มีแต่ class 6 อย่างเดียว จะไม่แตะต้องไฟล์นั้น"""
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except UnicodeDecodeError:
        # ลองอ่านแบบ default encoding เผื่อเป็น ANSI
        with path.open("r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        return 0, 0, False

    # ดึง class id ของแต่ละบรรทัด
    class_ids = [ln.split()[0] for ln in lines if ln.split()]

    if not class_ids:
        return 0, 0, False

    # ถ้ามีแต่ angle3 ล้วน ๆ -> ไม่แก้ไฟล์นี้
    if all(cid == TARGET_CLASS_ID for cid in class_ids):
        return len(lines), 0, False

    # มี label อื่นปน -> ลบเฉพาะ angle3
    kept = [ln for ln in lines if ln.split()[0] != TARGET_CLASS_ID]
    removed_count = len(lines) - len(kept)

    if removed_count > 0:
        with path.open("w", encoding="utf-8") as f:
            for ln in kept:
                f.write(ln + "\n")

    return len(lines), removed_count, True


def main():
    if not LABELS_DIR.exists():
        print(f"Labels folder not found: {LABELS_DIR}")
        return

    total_files = 0
    touched_files = 0
    total_removed = 0

    for txt_path in LABELS_DIR.glob("*.txt"):
        total_files += 1
        orig, removed, touched = process_label_file(txt_path)
        if touched:
            touched_files += 1
            total_removed += removed
            print(f"Updated: {txt_path.name} (removed {removed} angle3 lines)")

    print("\n==== SUMMARY ====")
    print(f"Total label files: {total_files}")
    print(f"Files modified   : {touched_files}")
    print(f"Total angle3 lines removed: {total_removed}")


if __name__ == "__main__":
    main()


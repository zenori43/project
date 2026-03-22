"""
สคริปต์เทียบชื่อไฟล์ระหว่าง crop folder และ usb_camera folder
เพื่อดูว่าไฟล์ไหนที่ยังไม่ได้ครอป
"""

import os
from pathlib import Path

# Paths
crop_folder = r"C:\Users\Win 10 Home\Desktop\crop"
usb_camera_folder = r"C:\Users\Win 10 Home\Desktop\project_final\final\captured_images\usb_camera"

def get_base_name(filename):
    """ดึงชื่อไฟล์โดยตัดนามสกุลและ _angle1 ออก"""
    # ตัดนามสกุลออก
    name = Path(filename).stem
    # ตัด _angle1 ออกถ้ามี
    if name.endswith('_angle1'):
        name = name[:-7]
    return name

# อ่านไฟล์จาก crop folder
crop_files = set()
if os.path.exists(crop_folder):
    for file in os.listdir(crop_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = get_base_name(file)
            crop_files.add(base_name)
    print(f"พบไฟล์ใน crop folder: {len(crop_files)} ไฟล์")
else:
    print(f"ไม่พบ folder: {crop_folder}")

# อ่านไฟล์จาก usb_camera folder
usb_camera_files = set()
if os.path.exists(usb_camera_folder):
    for file in os.listdir(usb_camera_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = get_base_name(file)
            usb_camera_files.add(base_name)
    print(f"พบไฟล์ใน usb_camera folder: {len(usb_camera_files)} ไฟล์")
else:
    print(f"ไม่พบ folder: {usb_camera_folder}")

# หาไฟล์ที่ยังไม่ได้ครอป (อยู่ใน usb_camera แต่ไม่อยู่ใน crop)
not_cropped = usb_camera_files - crop_files
not_cropped = sorted(not_cropped)

print("\n" + "="*60)
print(f"สรุปผลการเทียบ:")
print(f"   - ครอปแล้ว: {len(crop_files)} ไฟล์")
print(f"   - ยังไม่ได้ครอป: {len(not_cropped)} ไฟล์")
print("="*60)

if not_cropped:
    print("\nไฟล์ที่ยังไม่ได้ครอป:")
    print("-" * 60)
    for i, filename in enumerate(not_cropped, 1):
        # หาไฟล์จริงใน usb_camera folder
        actual_file = None
        for file in os.listdir(usb_camera_folder):
            if get_base_name(file) == filename:
                actual_file = file
                break
        print(f"{i:3d}. {actual_file or filename}")
    
    # บันทึกลงไฟล์
    output_file = os.path.join(os.path.dirname(__file__), "not_cropped_files.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ไฟล์ที่ยังไม่ได้ครอป:\n")
        f.write("="*60 + "\n")
        for filename in not_cropped:
            actual_file = None
            for file in os.listdir(usb_camera_folder):
                if get_base_name(file) == filename:
                    actual_file = file
                    break
            f.write(f"{actual_file or filename}\n")
    print(f"\nบันทึกลงไฟล์: {output_file}")
else:
    print("\nครอปครบทุกไฟล์แล้ว!")

# แสดงไฟล์ที่อยู่ใน crop แต่ไม่อยู่ใน usb_camera (อาจเป็นไฟล์เก่า)
extra_cropped = crop_files - usb_camera_files
if extra_cropped:
    print(f"\nไฟล์ใน crop ที่ไม่อยู่ใน usb_camera ({len(extra_cropped)} ไฟล์):")
    for filename in sorted(extra_cropped)[:10]:  # แสดงแค่ 10 ไฟล์แรก
        print(f"   - {filename}")
    if len(extra_cropped) > 10:
        print(f"   ... และอีก {len(extra_cropped) - 10} ไฟล์")

print("\n" + "="*60)

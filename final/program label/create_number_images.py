from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import random
import numpy as np

# --- ตั้งค่าพารามิเตอร์ (เพิ่มฟอนต์และปรับจำนวน) ---
font_paths = [
    'arial.ttf',
    'calibri.ttf',
    'times.ttf',
    'verdana.ttf',
    'tahoma.ttf',
    'comic.ttf',
    'impact.ttf',
    r"C:\Users\Win 10 Home\AppData\Local\Microsoft\Windows\Fonts\DOTMATRI.TTF",
    r"C:\Users\Win 10 Home\AppData\Local\Microsoft\Windows\Fonts\Minecart LCD.ttf",
    r"C:\Users\Win 10 Home\AppData\Local\Microsoft\Windows\Fonts\hydrogen.ttf",
    r"C:\Users\Win 10 Home\AppData\Local\Microsoft\Windows\Fonts\alarm clock.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\comic.ttf",
    r"C:\Windows\Fonts\impact.ttf",
    r"C:\Windows\Fonts\georgia.ttf"
]

font_size = 30
image_width = 300
image_height = 120
text_color = "black"
background_color = "white"
output_dir = "number_images"
gt_file = "number_gt.txt"
num_augmentations_per_font = 3  # augmentation ต่อฟอนต์

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- สร้าง list ของตัวเลขที่ต้องการ ---
def generate_date_numbers():
    """สร้างตัวเลข 1-31 สำหรับวัน และ 1-12 สำหรับเดือน"""
    numbers = []
    
    # ตัวเลข 1-31 (วัน)
    for day in range(1, 32):
        numbers.append(str(day))
    
    # ตัวเลข 1-12 (เดือน)
    for month in range(1, 13):
        numbers.append(str(month))
    
    return numbers

# --- ฟังก์ชันสำหรับสร้างภาพจากตัวเลข (ใช้ฟอนต์ที่กำหนด) ---
def create_number_image(number, font_path):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Warning: Font '{font_path}' not found. Using default font.")
        font = ImageFont.load_default()

    # สร้างภาพ
    img = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(img)

    # คำนวณตำแหน่งข้อความให้อยู่กลางภาพ
    bbox = draw.textbbox((0, 0), number, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (image_width - text_width) / 2
    y = (image_height - text_height) / 2
    
    draw.text((x, y), number, font=font, fill=text_color)
    
    return img

# --- ฟังก์ชัน Data Augmentation (ใช้จาก creatimg.py) ---
def augment_image(img):
    augmented_img = img.copy()

    if random.random() < 0.5:
        augmented_img = augmented_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    if random.random() < 0.6:
        enhancer = ImageEnhance.Brightness(augmented_img)
        augmented_img = enhancer.enhance(random.uniform(0.7, 1.3))

    if random.random() < 0.6:
        enhancer = ImageEnhance.Contrast(augmented_img)
        augmented_img = enhancer.enhance(random.uniform(0.7, 1.3))

    if random.random() < 0.4:
        img_np = np.array(augmented_img)
        mean = 0
        var = random.uniform(50, 200)
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, img_np.shape)
        noisy_img_np = img_np + gauss
        noisy_img_np = np.clip(noisy_img_np, 0, 255)
        augmented_img = Image.fromarray(noisy_img_np.astype('uint8'))

    if random.random() < 0.3:
        angle = random.uniform(-2, 2)
        augmented_img = augmented_img.rotate(angle, resample=Image.BICUBIC, expand=False)

    return augmented_img

# --- เริ่มสร้างภาพและ label ---
numbers_to_generate = generate_date_numbers()
total_images = len(numbers_to_generate) * len(font_paths) * (num_augmentations_per_font + 1)

print(f"Generating {total_images} number images and labels...")
print(f"Numbers to generate: {numbers_to_generate}")
print(f"Total fonts available: {len(font_paths)}")
print(f"Augmentations per font: {num_augmentations_per_font}")
print(f"Total images per number: {len(font_paths) * (num_augmentations_per_font + 1)}")

image_counter = 0
with open(gt_file, "w") as f:
    for number in numbers_to_generate:
        for font_path in font_paths:
            base_img = create_number_image(number, font_path)
            
            # บันทึกภาพต้นฉบับ
            filename = f"number_{image_counter:05d}.png"
            image_path = os.path.join(output_dir, filename)
            base_img.save(image_path)
            
            label_line = f"{os.path.join(output_dir, filename)}\t{number}\n"
            f.write(label_line)
            image_counter += 1
            
            # สร้าง augmented versions
            for _ in range(num_augmentations_per_font):
                augmented_img = augment_image(base_img)
                
                filename = f"number_{image_counter:05d}.png"
                image_path = os.path.join(output_dir, filename)
                augmented_img.save(image_path)
                
                label_line = f"{os.path.join(output_dir, filename)}\t{number}\n"
                f.write(label_line)
                image_counter += 1

print(f"Number image and label generation complete. Total images: {image_counter}")
print(f"Check the '{output_dir}' folder and '{gt_file}' file.")
print(f"Generated numbers: Days 1-31, Months 1-12")

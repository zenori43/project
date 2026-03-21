from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import random
import numpy as np

# --- ตั้งค่าพารามิเตอร์ ---
font_paths = [
    'arial.ttf',
    r"C:\Users\Win 10 Home\AppData\Local\Microsoft\Windows\Fonts\DOTMATRI.TTF",
    r"C:\Users\Win 10 Home\AppData\Local\Microsoft\Windows\Fonts\Minecart LCD.ttf",
    r"C:\Users\Win 10 Home\AppData\Local\Microsoft\Windows\Fonts\hydrogen.ttf",
    r"C:\Users\Win 10 Home\AppData\Local\Microsoft\Windows\Fonts\alarm clock.ttf"
]

font_size = 30
# ขนาดภาพพื้นฐาน (จะถูกปรับอัตโนมัติให้พอดีกับข้อความใน create_base_image)
image_width = 300
image_height = 120  # ใช้เป็นค่าเริ่มต้นเท่านั้น
text_color = "black"
background_color = "white"
# สำหรับภาพแบบรวมบรรทัด (ไม่ใช้แล้ว แต่คงตัวแปรไว้เผื่ออนาคต)
output_dir = "augmented_synthetic_data"
gt_file = "gt.txt"
# สำหรับภาพที่แยกเป็นบรรทัด (ที่เราต้องการใช้จริง)
line_output_dir = "line_images"
line_gt_file = "line_gt.txt"
num_augmentations_per_text = 5
num_specific_patterns = 100 # จำนวนรูปแบบ 10:18S02 ที่ต้องการสร้าง

if not os.path.exists(line_output_dir):
    os.makedirs(line_output_dir)

# --- สร้าง list ของข้อความที่ต้องการ ---
def generate_dates():
    dates = []
    # สร้างวันที่ตั้งแต่ปี 25 ถึงปี 28
    for year in [25, 26, 27, 28]:
        for month in range(1, 13):
            days_in_month = 31
            if month == 2:
                days_in_month = 29 if year % 4 == 0 else 28
            elif month in [4, 6, 9, 11]:
                days_in_month = 30
            for day in range(1, days_in_month + 1):
                date_str = f"{day:02d}/{month:02d}/{year:02d}"
                dates.append(date_str)
    return dates

def generate_specific_pattern_texts(count):
    patterns = []
    for _ in range(count):
        # สุ่มตัวเลขสำหรับแต่ละส่วนของรูปแบบ xx:xxSxx
        part1 = random.randint(0, 23)
        part2 = random.randint(0, 59)
        part3 = random.randint(0, 99)
        pattern = f"{part1:02d}:{part2:02d}S{part3:02d}"
        patterns.append(pattern)
    return patterns

dates_combined = generate_dates()
specific_patterns = generate_specific_pattern_texts(num_specific_patterns)

# สร้างข้อความแบบบรรทัด (line-level)
def generate_line_texts():
    line_texts = []
    
    # สร้างข้อความแบบ 3 บรรทัด: MFG, BBF, และ pattern
    for i in range(len(dates_combined)):
        if i < len(specific_patterns):
            # รวม 3 บรรทัดในภาพเดียว
            mfg_date = dates_combined[i]
            bbf_date = dates_combined[(i + 1) % len(dates_combined)]  # ใช้วันที่ถัดไป
            pattern = specific_patterns[i]
            
            line_text = f"MFG {mfg_date}\nBBF {bbf_date}\n{pattern}"
            line_texts.append(line_text)
    
    # เพิ่มข้อความพิเศษ
    line_texts.append("MFG 01/01/25\nBBF 31/12/25\n00:00S00")
    line_texts.append("MFG 15/06/26\nBBF 20/08/26\n12:30S45")
    
    return line_texts

base_texts_to_generate = generate_line_texts()

# --- ฟังก์ชันสำหรับสร้างภาพจากข้อความพื้นฐาน (สุ่ม font) ---
def create_base_image(text):
    selected_font_path = random.choice(font_paths)
    try:
        font = ImageFont.truetype(selected_font_path, font_size)
    except IOError:
        print(f"Warning: Font '{selected_font_path}' not found. Using default font.")
        font = ImageFont.load_default()

    # แยกข้อความเป็นบรรทัด
    lines = text.split('\n')

    # คำนวณขนาดข้อความจริงของแต่ละบรรทัด
    temp_img = Image.new("RGB", (image_width, image_height), background_color)
    temp_draw = ImageDraw.Draw(temp_img)

    line_heights = []
    line_widths = []
    for line in lines:
        bbox = temp_draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        line_widths.append(text_width)
        line_heights.append(text_height)

    # ระยะห่างระหว่างบรรทัดและระยะขอบรอบข้อความ
    line_spacing = 5
    padding_x = 10
    padding_y = 10

    total_height = sum(line_heights) + line_spacing * (len(lines) - 1 if len(lines) > 1 else 0)
    max_width = max(line_widths) if line_widths else 0

    img_width_dynamic = max_width + padding_x * 2
    img_height_dynamic = total_height + padding_y * 2

    # สร้างภาพใหม่ที่มีขนาดพอดีกับข้อความ
    img = Image.new("RGB", (int(img_width_dynamic), int(img_height_dynamic)), background_color)
    draw = ImageDraw.Draw(img)

    # วาดข้อความแต่ละบรรทัดให้อยู่กลางแนวนอน และพอดีแนวตั้ง
    y = padding_y
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (img.width - text_width) / 2
        draw.text((x, y), line, font=font, fill=text_color)
        y += text_height + line_spacing
    
    return img

# --- ฟังก์ชัน Data Augmentation ---
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

# --- เริ่มสร้างภาพและ label (เฉพาะแบบแยกบรรทัด) ---
print(f"Generating line-level images and labels...")
line_image_counter = 0

with open(line_gt_file, "w") as f_line:
    for text in base_texts_to_generate:
        lines = text.split("\n")
        for line_text in lines:
            # ข้ามบรรทัดว่าง ถ้ามี
            if not line_text.strip():
                continue

            # ภาพบรรทัดเดียว (base)
            line_base_img = create_base_image(line_text)

            line_filename = f"line_{line_image_counter:05d}.png"
            line_image_path = os.path.join(line_output_dir, line_filename)
            line_base_img.save(line_image_path)

            line_label = f"{os.path.join(line_output_dir, line_filename)}\t{line_text}\n"
            f_line.write(line_label)
            line_image_counter += 1

            # Augmentation ของภาพบรรทัดเดียว
            for _ in range(num_augmentations_per_text):
                line_aug_img = augment_image(line_base_img)

                line_filename = f"line_{line_image_counter:05d}.png"
                line_image_path = os.path.join(line_output_dir, line_filename)
                line_aug_img.save(line_image_path)

                line_label = f"{os.path.join(line_output_dir, line_filename)}\t{line_text}\n"
                f_line.write(line_label)
                line_image_counter += 1

print(f"Line image generation complete. Total line images: {line_image_counter}")
print(f"Check the '{line_output_dir}' folder and '{line_gt_file}' file.")
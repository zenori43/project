# -*- coding: utf-8 -*-
"""
Update Brightness/Contrast Config
อัปเดตค่า brightness/contrast ใน config/settings.py ตามค่าที่วิเคราะห์ได้
"""

import os
import sys
import re

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "settings.py")

def update_config(brightness=33, contrast=1.8, update_all_tastes=True):
    """
    อัปเดตค่า brightness/contrast ใน config/settings.py
    
    Args:
        brightness: int, ค่า brightness ที่ต้องการ (default: 33)
        contrast: float, ค่า contrast ที่ต้องการ (default: 1.8)
        update_all_tastes: bool, อัปเดตทุกรส (M100, M110, M120) หรือไม่
    """
    if not os.path.exists(CONFIG_FILE):
        print(f"[ERROR] Config file not found: {CONFIG_FILE}")
        return False
    
    try:
        # Read config file
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update IMAGE_ENHANCEMENT
        pattern_brightness = r"('brightness':\s*)\d+"
        pattern_contrast = r"('contrast':\s*)[\d.]+"
        
        # Update default IMAGE_ENHANCEMENT
        content = re.sub(
            pattern_brightness,
            f"\\g<1>{brightness}",
            content,
            count=1  # Only first occurrence (IMAGE_ENHANCEMENT)
        )
        
        # Find and update IMAGE_ENHANCEMENT contrast
        img_enhance_start = content.find("'IMAGE_ENHANCEMENT': {")
        if img_enhance_start != -1:
            img_enhance_end = content.find("}", img_enhance_start)
            img_enhance_section = content[img_enhance_start:img_enhance_end]
            
            # Update contrast in IMAGE_ENHANCEMENT
            img_enhance_section = re.sub(
                pattern_contrast,
                f"\\g<1>{contrast}",
                img_enhance_section,
                count=1
            )
            
            content = content[:img_enhance_start] + img_enhance_section + content[img_enhance_end:]
        
        # Update TASTE_ENHANCEMENT_MAP if requested
        if update_all_tastes:
            # Update brightness for all tastes (M100, M110, M120)
            # Find all brightness values in TASTE_ENHANCEMENT_MAP
            taste_pattern = r"('M\d+':\s*\{[^}]*'brightness':\s*)\d+"
            content = re.sub(
                taste_pattern,
                f"\\g<1>{brightness}",
                content
            )
            
            # Update contrast for all tastes
            taste_contrast_pattern = r"('M\d+':\s*\{[^}]*'contrast':\s*)[\d.]+"
            content = re.sub(
                taste_contrast_pattern,
                f"\\g<1>{contrast}",
                content
            )
        
        # Write back
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"[OK] Updated config successfully!")
        print(f"   Brightness: {brightness}")
        print(f"   Contrast: {contrast}")
        if update_all_tastes:
            print(f"   Updated all tastes (M100, M110, M120)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to update config: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("Update Brightness/Contrast Config")
    print("=" * 60)
    print()
    
    # Get values from user
    print("Enter values from brightness analysis:")
    print("  (Press Enter to use default values)")
    print()
    
    try:
        brightness_input = input(f"Brightness (default: 33): ").strip()
        brightness = int(brightness_input) if brightness_input else 33
        
        contrast_input = input(f"Contrast (default: 1.8): ").strip()
        contrast = float(contrast_input) if contrast_input else 1.8
        
        update_all = input("Update all tastes (M100, M110, M120)? (y/n, default: y): ").strip().lower()
        update_all_tastes = update_all != 'n'
        
        print()
        print("=" * 60)
        print(f"Updating config with:")
        print(f"  Brightness: {brightness}")
        print(f"  Contrast: {contrast}")
        print(f"  Update all tastes: {update_all_tastes}")
        print("=" * 60)
        print()
        
        # Confirm
        confirm = input("Confirm update? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
        
        # Update
        success = update_config(brightness, contrast, update_all_tastes)
        
        if success:
            print()
            print("[OK] Config updated successfully!")
            print("   Please restart the main program to apply changes.")
        else:
            print()
            print("[ERROR] Failed to update config.")
            
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

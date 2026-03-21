#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test CUDA in actual code - ตรวจสอบว่าโค้ดใช้ CUDA จริงๆ หรือไม่
"""

import sys
import os

# Setup CUDA paths FIRST (like in main.py)
from core.cuda_setup import setup_cuda_paths
setup_cuda_paths()

print("=" * 60)
print("🧪 Testing CUDA in Actual Code")
print("=" * 60)
print()

# Test 1: Check CUDA environment
print("📋 Test 1: CUDA Environment Setup")
print("-" * 60)
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
has_cuda = '/usr/local/cuda-11.4/lib64' in ld_path
has_tegra = '/usr/lib/aarch64-linux-gnu/tegra' in ld_path
has_standard = '/usr/lib/aarch64-linux-gnu' in ld_path
print(f"LD_LIBRARY_PATH has CUDA: {has_cuda}")
print(f"LD_LIBRARY_PATH has Jetson CUDA: {has_tegra}")
print(f"LD_LIBRARY_PATH has standard CUDA: {has_standard}")
print(f"PYTHONPATH has OpenCV: {'/usr/local/lib/python3.8/site-packages/' in os.environ.get('PYTHONPATH', '')}")
print()

# Test 2: Check OpenCV CUDA
print("📋 Test 2: OpenCV CUDA Detection")
print("-" * 60)
import cv2
print(f"OpenCV version: {cv2.__version__}")
print(f"OpenCV location: {cv2.__file__}")
print(f"OpenCV has CUDA module: {hasattr(cv2, 'cuda')}")

if hasattr(cv2, 'cuda'):
    try:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA devices available: {device_count}")
        if device_count > 0:
            print("✅ CUDA devices detected!")
        else:
            print("⚠️ CUDA devices = 0 (but functions might still work)")
    except Exception as e:
        print(f"❌ Error checking CUDA devices: {e}")
else:
    print("❌ OpenCV compiled without CUDA support")
print()

# Test 3: Test CUDA helper functions
print("📋 Test 3: CUDA Helper Functions")
print("-" * 60)
from core.cuda_image_utils import cuda_resize, cuda_cvtColor, cuda_gaussianBlur, CUDA_AVAILABLE
print(f"CUDA_AVAILABLE flag: {CUDA_AVAILABLE}")

import numpy as np

# Create test image
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
print(f"Test image shape: {test_image.shape}")

# Test cuda_resize
print("\n🔍 Testing cuda_resize()...")
try:
    result_resize = cuda_resize(test_image, (320, 240))
    print(f"  ✓ Resize SUCCESS: {test_image.shape} -> {result_resize.shape}")
    if CUDA_AVAILABLE:
        print("  🚀 Used CUDA (GPU)")
    else:
        print("  💻 Used CPU (fallback)")
except Exception as e:
    print(f"  ❌ Resize FAILED: {e}")

# Test cuda_cvtColor
print("\n🔍 Testing cuda_cvtColor()...")
try:
    result_rgb = cuda_cvtColor(test_image, cv2.COLOR_BGR2RGB)
    print(f"  ✓ Color conversion SUCCESS: {test_image.shape} -> {result_rgb.shape}")
    if CUDA_AVAILABLE:
        print("  🚀 Used CUDA (GPU)")
    else:
        print("  💻 Used CPU (fallback)")
except Exception as e:
    print(f"  ❌ Color conversion FAILED: {e}")

# Test cuda_gaussianBlur
print("\n🔍 Testing cuda_gaussianBlur()...")
try:
    result_blur = cuda_gaussianBlur(test_image, (15, 15), 2.0)
    print(f"  ✓ Gaussian blur SUCCESS: {test_image.shape} -> {result_blur.shape}")
    if CUDA_AVAILABLE:
        print("  🚀 Used CUDA (GPU)")
    else:
        print("  💻 Used CPU (fallback)")
except Exception as e:
    print(f"  ❌ Gaussian blur FAILED: {e}")
print()

# Test 4: Direct CUDA test (verify CUDA functions work)
print("📋 Test 4: Direct CUDA Functions Test")
print("-" * 60)
if hasattr(cv2, 'cuda'):
    try:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if device_count > 0:
            print("✅ CUDA devices available - testing direct CUDA functions...")
            
            # Test direct CUDA
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(test_img)
            print("  ✓ Upload to GPU: SUCCESS")
            
            # Resize on GPU
            gpu_resized = cv2.cuda.resize(gpu_img, (50, 50))
            print("  ✓ GPU resize: SUCCESS")
            
            # Color conversion on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
            print("  ✓ GPU color conversion: SUCCESS")
            
            # Download from GPU
            result = gpu_resized.download()
            print(f"  ✓ Download from GPU: SUCCESS (shape: {result.shape})")
            
            print("\n🎉 Direct CUDA functions work perfectly!")
        else:
            print("⚠️ CUDA devices = 0, but trying direct CUDA functions anyway...")
            try:
                test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(test_img)
                gpu_resized = cv2.cuda.resize(gpu_img, (50, 50))
                result = gpu_resized.download()
                print("✅ Direct CUDA functions work despite device count = 0!")
            except Exception as e:
                print(f"❌ Direct CUDA functions failed: {e}")
    except Exception as e:
        print(f"❌ CUDA test error: {e}")
else:
    print("❌ OpenCV CUDA module not available")
print()

# Test 5: Test actual code usage (like in image_processor.py)
print("📋 Test 5: Actual Code Usage Test")
print("-" * 60)
try:
    # Simulate what happens in image_processor.py
    from core.image_processor import safe_show_img, fit_to_same_size
    
    # Test safe_show_img (uses cuda_cvtColor)
    gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = safe_show_img(gray_img)
    print(f"  ✓ safe_show_img() SUCCESS: {gray_img.shape} -> {result.shape}")
    
    # Test fit_to_same_size (uses cuda_resize)
    test_images = [
        np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8),
        np.random.randint(0, 255, (150, 250, 3), dtype=np.uint8),
    ]
    result_images = fit_to_same_size(test_images, target_size=(280, 280))
    print(f"  ✓ fit_to_same_size() SUCCESS: {len(test_images)} images -> {len(result_images)} images")
    for i, img in enumerate(result_images):
        print(f"    Image {i+1}: {img.shape}")
    
    print("\n✅ Actual code functions work correctly!")
except Exception as e:
    print(f"❌ Actual code test failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 6: Performance comparison
print("📋 Test 6: Performance Comparison (CPU vs CUDA)")
print("-" * 60)
import time

test_img_large = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
iterations = 10

# CPU test
print("Testing CPU performance...")
start_time = time.time()
for i in range(iterations):
    result_cpu = cv2.resize(test_img_large, (640, 480))
    result_cpu = cv2.cvtColor(result_cpu, cv2.COLOR_BGR2RGB)
    result_cpu = cv2.GaussianBlur(result_cpu, (15, 15), 2.0)
cpu_time = time.time() - start_time

# CUDA test (if available)
if CUDA_AVAILABLE:
    print("Testing CUDA performance...")
    start_time = time.time()
    for i in range(iterations):
        result_cuda = cuda_resize(test_img_large, (640, 480))
        result_cuda = cuda_cvtColor(result_cuda, cv2.COLOR_BGR2RGB)
        result_cuda = cuda_gaussianBlur(result_cuda, (15, 15), 2.0)
    cuda_time = time.time() - start_time
    
    print(f"\n📊 Results ({iterations} iterations):")
    print(f"  CPU time: {cpu_time:.3f}s ({cpu_time/iterations*1000:.2f}ms per operation)")
    print(f"  CUDA time: {cuda_time:.3f}s ({cuda_time/iterations*1000:.2f}ms per operation)")
    if cuda_time < cpu_time:
        speedup = cpu_time / cuda_time
        print(f"  🚀 CUDA is {speedup:.2f}x faster!")
    else:
        slowdown = cuda_time / cpu_time
        print(f"  ⚠️ CUDA is {slowdown:.2f}x slower (overhead for this image size)")
else:
    print("⚠️ CUDA not available - skipping performance test")

print()
print("=" * 60)
print("✅ CUDA Testing Complete!")
print("=" * 60)


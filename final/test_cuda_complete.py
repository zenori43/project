#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete CUDA Test - ทดสอบ CUDA ทั้ง OpenCV และ PyTorch
"""

import sys
import os

# ถ้ารันจาก IDE/โดยตรง โดยที่ LD_LIBRARY_PATH ยังไม่มี path Jetson/CUDA
# จะทำให้ PyTorch/OpenCV โหลดแล้วไม่เห็น GPU — ต้อง set env ก่อนเริ่ม process
# จึง re-exec ตัวเองด้วย env จาก set_cuda_env.sh (เฉพาะ Linux)
if sys.platform == "linux":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_script = os.path.join(script_dir, "set_cuda_env.sh")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if os.path.isfile(env_script) and "tegra" not in ld_path:
        import subprocess
        result = subprocess.run(
            ["bash", "-c", "source " + repr(env_script) + " && env"],
            capture_output=True,
            text=True,
            cwd=script_dir,
        )
        if result.returncode == 0:
            new_env = os.environ.copy()
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    new_env[k] = v
            os.execve(sys.executable, [sys.executable, __file__] + sys.argv[1:], new_env)

import time

# Setup CUDA paths FIRST (like in main.py)
from core.cuda_setup import setup_cuda_paths
setup_cuda_paths()

print("=" * 70)
print("🧪 Complete CUDA Test - OpenCV & PyTorch")
print("=" * 70)
print()

# ============================================================================
# TEST 1: OpenCV CUDA
# ============================================================================
print("📋 TEST 1: OpenCV CUDA")
print("-" * 70)

import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"OpenCV location: {cv2.__file__}")
print(f"OpenCV has CUDA module: {hasattr(cv2, 'cuda')}")

if hasattr(cv2, 'cuda'):
    try:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA devices available: {device_count}")
        
        if device_count > 0:
            print("✅ OpenCV CUDA devices detected!")
        else:
            print("⚠️ OpenCV CUDA devices = 0")
    except Exception as e:
        print(f"❌ Error checking CUDA devices: {e}")
        device_count = 0
else:
    print("❌ OpenCV compiled without CUDA support")
    device_count = 0

print()

# Test OpenCV CUDA helper functions
print("🔍 Testing OpenCV CUDA Helper Functions:")
print("-" * 70)

from core.cuda_image_utils import cuda_resize, cuda_cvtColor, cuda_gaussianBlur, CUDA_AVAILABLE

print(f"CUDA_AVAILABLE flag: {CUDA_AVAILABLE}")

# Create test image
test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
print(f"Test image shape: {test_img.shape}")
print()

# Test 1: Resize
print("1. Testing cuda_resize()...")
try:
    start_time = time.time()
    result_resize = cuda_resize(test_img, (320, 240))
    resize_time = time.time() - start_time
    print(f"   ✓ Resize SUCCESS: {test_img.shape} -> {result_resize.shape}")
    print(f"   Time: {resize_time*1000:.2f}ms")
    if CUDA_AVAILABLE:
        print("   🚀 Used CUDA (GPU)")
    else:
        print("   💻 Used CPU (fallback)")
except Exception as e:
    print(f"   ❌ Resize FAILED: {e}")

# Test 2: Color conversion
print("\n2. Testing cuda_cvtColor()...")
try:
    start_time = time.time()
    result_rgb = cuda_cvtColor(test_img, cv2.COLOR_BGR2RGB)
    cvt_time = time.time() - start_time
    print(f"   ✓ Color conversion SUCCESS: {test_img.shape} -> {result_rgb.shape}")
    print(f"   Time: {cvt_time*1000:.2f}ms")
    if CUDA_AVAILABLE:
        print("   🚀 Used CUDA (GPU)")
    else:
        print("   💻 Used CPU (fallback)")
except Exception as e:
    print(f"   ❌ Color conversion FAILED: {e}")

# Test 3: Gaussian blur
print("\n3. Testing cuda_gaussianBlur()...")
try:
    start_time = time.time()
    result_blur = cuda_gaussianBlur(test_img, (15, 15), 2.0)
    blur_time = time.time() - start_time
    print(f"   ✓ Gaussian blur SUCCESS: {test_img.shape} -> {result_blur.shape}")
    print(f"   Time: {blur_time*1000:.2f}ms")
    if CUDA_AVAILABLE:
        print("   🚀 Used CUDA (GPU)")
    else:
        print("   💻 Used CPU (fallback)")
except Exception as e:
    print(f"   ❌ Gaussian blur FAILED: {e}")

# Test 4: Direct CUDA functions
print("\n4. Testing Direct CUDA Functions...")
if hasattr(cv2, 'cuda') and device_count > 0:
    try:
        test_img_small = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Upload to GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img_small)
        print("   ✓ Upload to GPU: SUCCESS")
        
        # Resize on GPU
        gpu_resized = cv2.cuda.resize(gpu_img, (50, 50))
        print("   ✓ GPU resize: SUCCESS")
        
        # Color conversion on GPU
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        print("   ✓ GPU color conversion: SUCCESS")
        
        # Download from GPU
        result = gpu_resized.download()
        print(f"   ✓ Download from GPU: SUCCESS (shape: {result.shape})")
        
        print("   🎉 Direct CUDA functions work perfectly!")
    except Exception as e:
        print(f"   ❌ Direct CUDA functions failed: {e}")
else:
    print("   ⚠️ Skipped (CUDA devices = 0)")

print()

# ============================================================================
# TEST 2: PyTorch CUDA
# ============================================================================
print("📋 TEST 2: PyTorch CUDA")
print("-" * 70)

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print("✅ PyTorch CUDA available!")
else:
    print("❌ PyTorch CUDA not available")

print()

# Test PyTorch CUDA operations
print("🔍 Testing PyTorch CUDA Operations:")
print("-" * 70)

if torch.cuda.is_available():
    # Test 1: Tensor creation and movement
    print("1. Testing Tensor Operations...")
    try:
        # Create tensor on CPU
        cpu_tensor = torch.randn(1000, 1000)
        print(f"   CPU tensor device: {cpu_tensor.device}")
        
        # Move to CUDA
        start_time = time.time()
        cuda_tensor = cpu_tensor.cuda()
        move_time = time.time() - start_time
        print(f"   CUDA tensor device: {cuda_tensor.device}")
        print(f"   Time to move to CUDA: {move_time*1000:.2f}ms")
        
        # Test operation on CUDA
        start_time = time.time()
        result = torch.mm(cuda_tensor, cuda_tensor.t())
        torch.cuda.synchronize()  # Wait for GPU to finish
        op_time = time.time() - start_time
        print(f"   Matrix multiplication on CUDA: SUCCESS")
        print(f"   Result device: {result.device}")
        print(f"   Operation time: {op_time*1000:.2f}ms")
        print("   🚀 Used CUDA (GPU)")
    except Exception as e:
        print(f"   ❌ Tensor operations failed: {e}")
    
    # Test 2: Model initialization
    print("\n2. Testing Model Initialization...")
    try:
        # Test rotationmodel
        from libs.processing.rotationmodel import RotationModel
        rot_model = RotationModel()
        print(f"   RotationModel device: {rot_model.device}")
        print(f"   RotationModel cuda: {rot_model.cuda}")
        if rot_model.device.type == 'cuda':
            print("   🚀 Using CUDA (GPU)")
        else:
            print("   💻 Using CPU")
    except Exception as e:
        print(f"   RotationModel test failed: {e}")
    
    try:
        # Test capmodel
        from libs.detection.capmodel import CapDetector
        cap_detector = CapDetector()
        print(f"   CapDetector device: {cap_detector.device}")
        if cap_detector.device.type == 'cuda':
            print("   🚀 Using CUDA (GPU)")
        else:
            print("   💻 Using CPU")
    except Exception as e:
        print(f"   CapDetector test failed: {e}")
    
    try:
        # Test deep_ocr
        from libs.processing.deep_ocr import DeepOCRModel
        ocr_model = DeepOCRModel()
        print(f"   DeepOCRModel device: {ocr_model.device}")
        if ocr_model.device.type == 'cuda':
            print("   🚀 Using CUDA (GPU)")
        else:
            print("   💻 Using CPU")
    except Exception as e:
        print(f"   DeepOCRModel test failed: {e}")
    
    try:
        # Test bottledetect
        from libs.detection.bottledetect import CUDA_AVAILABLE
        print(f"   bottledetect CUDA_AVAILABLE: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE:
            print("   🚀 Using CUDA (GPU)")
        else:
            print("   💻 Using CPU")
    except Exception as e:
        print(f"   bottledetect test failed: {e}")
else:
    print("⚠️ PyTorch CUDA not available - skipping tests")

print()

# ============================================================================
# TEST 3: Performance Comparison
# ============================================================================
print("📋 TEST 3: Performance Comparison")
print("-" * 70)

if torch.cuda.is_available() and CUDA_AVAILABLE:
    print("Comparing CPU vs CUDA performance...")
    
    # Large test image
    test_img_large = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    iterations = 10
    
    # CPU test (OpenCV)
    print("\n🖥️ CPU Performance (OpenCV):")
    start_time = time.time()
    for i in range(iterations):
        result_cpu = cv2.resize(test_img_large, (640, 480))
        result_cpu = cv2.cvtColor(result_cpu, cv2.COLOR_BGR2RGB)
        result_cpu = cv2.GaussianBlur(result_cpu, (15, 15), 2.0)
    cpu_time = time.time() - start_time
    
    # CUDA test (OpenCV)
    print("🚀 CUDA Performance (OpenCV):")
    start_time = time.time()
    for i in range(iterations):
        result_cuda = cuda_resize(test_img_large, (640, 480))
        result_cuda = cuda_cvtColor(result_cuda, cv2.COLOR_BGR2RGB)
        result_cuda = cuda_gaussianBlur(result_cuda, (15, 15), 2.0)
    cuda_time = time.time() - start_time
    
    print(f"\n📊 Results ({iterations} iterations):")
    print(f"   CPU time: {cpu_time:.3f}s ({cpu_time/iterations*1000:.2f}ms per operation)")
    print(f"   CUDA time: {cuda_time:.3f}s ({cuda_time/iterations*1000:.2f}ms per operation)")
    if cuda_time < cpu_time:
        speedup = cpu_time / cuda_time
        print(f"   🚀 CUDA is {speedup:.2f}x faster!")
    else:
        slowdown = cuda_time / cpu_time
        print(f"   ⚠️ CUDA is {slowdown:.2f}x slower (overhead for this image size)")
    
    # PyTorch performance
    print("\n🖥️ CPU Performance (PyTorch):")
    cpu_tensor = torch.randn(1000, 1000)
    start_time = time.time()
    for i in range(iterations):
        result_cpu = torch.mm(cpu_tensor, cpu_tensor.t())
    cpu_pytorch_time = time.time() - start_time
    
    print("🚀 CUDA Performance (PyTorch):")
    cuda_tensor = torch.randn(1000, 1000).cuda()
    start_time = time.time()
    for i in range(iterations):
        result_cuda = torch.mm(cuda_tensor, cuda_tensor.t())
    torch.cuda.synchronize()
    cuda_pytorch_time = time.time() - start_time
    
    print(f"\n📊 Results ({iterations} iterations):")
    print(f"   CPU time: {cpu_pytorch_time:.3f}s ({cpu_pytorch_time/iterations*1000:.2f}ms per operation)")
    print(f"   CUDA time: {cuda_pytorch_time:.3f}s ({cuda_pytorch_time/iterations*1000:.2f}ms per operation)")
    if cuda_pytorch_time < cpu_pytorch_time:
        speedup = cpu_pytorch_time / cuda_pytorch_time
        print(f"   🚀 CUDA is {speedup:.2f}x faster!")
    else:
        slowdown = cuda_pytorch_time / cpu_pytorch_time
        print(f"   ⚠️ CUDA is {slowdown:.2f}x slower")
else:
    print("⚠️ CUDA not available - skipping performance comparison")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("📊 SUMMARY")
print("=" * 70)

print("\n✅ OpenCV CUDA:")
if hasattr(cv2, 'cuda') and device_count > 0:
    print(f"   - Status: Available ({device_count} devices)")
    print(f"   - CUDA_AVAILABLE: {CUDA_AVAILABLE}")
    print("   - Functions: resize, cvtColor, gaussianBlur")
else:
    print("   - Status: Not available or devices = 0")

print("\n✅ PyTorch CUDA:")
if torch.cuda.is_available():
    print(f"   - Status: Available ({torch.cuda.device_count()} GPUs)")
    print(f"   - Device: {torch.cuda.get_device_name(0)}")
    print("   - Models: All models use CUDA")
else:
    print("   - Status: Not available")

print()
print("=" * 70)
if (hasattr(cv2, 'cuda') and device_count > 0) and torch.cuda.is_available():
    print("🎉 ทั้ง OpenCV และ PyTorch ใช้ CUDA แล้ว! (GPU Acceleration)")
else:
    print("⚠️ CUDA อาจไม่พร้อมใช้งานสำหรับบางส่วน")
print("=" * 70)


#!/usr/bin/env python3
"""
CUDA Setup - Setup CUDA paths for Jetson AGX Xavier
This must be imported BEFORE any OpenCV or PyTorch imports
"""

import os
import sys

def setup_cuda_paths():
    """
    Setup CUDA paths for Jetson AGX Xavier
    This must be called before importing OpenCV or PyTorch
    """
    # Find CUDA installation (check multiple locations)
    # Note: cuda and cuda-11 are symlinks pointing to cuda-11.4
    cuda_paths = [
        "/usr/local/cuda-11.4",
        "/usr/local/cuda",
        "/usr/local/cuda-11",
    ]
    
    cuda_found = None
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            # Resolve symlink to get real path (important!)
            real_path = os.path.realpath(cuda_path)
            # Verify that real path has CUDA libraries
            targets_lib = os.path.join(real_path, 'targets', 'aarch64-linux', 'lib')
            if os.path.exists(targets_lib):
                cuda_found = real_path
                break
    
    if cuda_found:
        # Set CUDA_HOME and CUDA_PATH (always set, even if already exists)
        os.environ['CUDA_HOME'] = cuda_found
        os.environ['CUDA_PATH'] = cuda_found
        
        # Add CUDA bin to PATH
        cuda_bin = os.path.join(cuda_found, 'bin')
        current_path = os.environ.get('PATH', '')
        if cuda_bin not in current_path:
            os.environ['PATH'] = cuda_bin + ':' + current_path
        
        # Add CUDA lib to LD_LIBRARY_PATH
        # Jetson needs both lib64 and targets/aarch64-linux/lib
        # Also need Jetson CUDA driver library path
        cuda_lib = os.path.join(cuda_found, 'lib64')
        cuda_targets_lib = os.path.join(cuda_found, 'targets', 'aarch64-linux', 'lib')
        jetson_cuda_lib = '/usr/lib/aarch64-linux-gnu/tegra'  # Jetson CUDA driver library
        standard_cuda_lib = '/usr/lib/aarch64-linux-gnu'  # Standard CUDA driver library (important!)
        
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        paths_to_add = []
        
        # Add standard CUDA driver library first (most important for detecting CUDA devices)
        if os.path.exists(standard_cuda_lib) and standard_cuda_lib not in current_ld_path:
            paths_to_add.append(standard_cuda_lib)
        
        # Add Jetson CUDA driver library
        if os.path.exists(jetson_cuda_lib) and jetson_cuda_lib not in current_ld_path:
            paths_to_add.append(jetson_cuda_lib)
        
        # Add targets/aarch64-linux/lib (Jetson-specific, most important)
        if os.path.exists(cuda_targets_lib) and cuda_targets_lib not in current_ld_path:
            paths_to_add.append(cuda_targets_lib)
        
        # Add lib64
        if os.path.exists(cuda_lib) and cuda_lib not in current_ld_path:
            paths_to_add.append(cuda_lib)
        
        # Prepend new paths to LD_LIBRARY_PATH (important for library resolution)
        if paths_to_add:
            new_paths = ':'.join(paths_to_add)
            if current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = new_paths + ':' + current_ld_path
            else:
                os.environ['LD_LIBRARY_PATH'] = new_paths
        
        # Add OpenCV 4.6.0 Python path (installed in /usr/local/lib/python3.8/site-packages/)
        # Also add /usr/local/lib for OpenCV libraries
        opencv_python_path = '/usr/local/lib/python3.8/site-packages/'
        opencv_lib_path = '/usr/local/lib'
        
        # Add to LD_LIBRARY_PATH if not already there
        if opencv_lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = opencv_lib_path + ':' + current_ld_path
            else:
                os.environ['LD_LIBRARY_PATH'] = opencv_lib_path
        
        # Add to PYTHONPATH if not already there
        current_python_path = os.environ.get('PYTHONPATH', '')
        if opencv_python_path not in current_python_path:
            if current_python_path:
                os.environ['PYTHONPATH'] = opencv_python_path + ':' + current_python_path
            else:
                os.environ['PYTHONPATH'] = opencv_python_path
        
        print(f"✅ CUDA paths set: CUDA_HOME={cuda_found}")
        print(f"   LD_LIBRARY_PATH includes: {', '.join(paths_to_add + [opencv_lib_path])}")
        print(f"   PYTHONPATH includes: {opencv_python_path}")
    else:
        print("⚠️ CUDA not found in standard locations")

# Auto-setup on import
setup_cuda_paths()



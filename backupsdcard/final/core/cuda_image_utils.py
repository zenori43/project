# -*- coding: utf-8 -*-
"""
CUDA Image Utilities - Helper functions for CUDA-accelerated image processing
"""

import cv2
import numpy as np

# Check if CUDA is available
CUDA_AVAILABLE = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0

def cuda_resize(image, size, interpolation=cv2.INTER_LINEAR):
    """
    Resize image using CUDA if available, otherwise use CPU
    
    Args:
        image: Input image (numpy array)
        size: Target size (width, height)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    if CUDA_AVAILABLE and image is not None:
        try:
            # Ensure image is contiguous and properly aligned
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            # Resize on GPU
            gpu_resized = cv2.cuda.resize(gpu_img, size, interpolation=interpolation)
            
            # Download from GPU
            result = gpu_resized.download()
            return result
        except Exception as e:
            # Fallback to CPU silently (don't print warning for every call)
            # Only print if it's not a misaligned address error (those are common)
            if "misaligned" not in str(e).lower():
                print(f"⚠️ CUDA resize failed, using CPU: {e}")
            return cv2.resize(image, size, interpolation=interpolation)
    else:
        # Use CPU
        return cv2.resize(image, size, interpolation=interpolation)


def cuda_cvtColor(image, code):
    """
    Convert color space using CUDA if available, otherwise use CPU
    
    Args:
        image: Input image (numpy array)
        code: Color conversion code (e.g., cv2.COLOR_BGR2RGB)
        
    Returns:
        Converted image
    """
    if CUDA_AVAILABLE and image is not None:
        try:
            # Ensure image is contiguous and properly aligned
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            
            # Check image dimensions and type
            if len(image.shape) != 3 or image.shape[2] != 3:
                # Not a valid color image, use CPU
                return cv2.cvtColor(image, code)
            
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            # Convert color on GPU
            gpu_converted = cv2.cuda.cvtColor(gpu_img, code)
            
            # Download from GPU
            result = gpu_converted.download()
            return result
        except Exception as e:
            # Fallback to CPU silently (don't print warning for every call)
            # Only print if it's not a misaligned address error (those are common)
            if "misaligned" not in str(e).lower():
                print(f"⚠️ CUDA color conversion failed, using CPU: {e}")
            return cv2.cvtColor(image, code)
    else:
        # Use CPU
        return cv2.cvtColor(image, code)


def cuda_gaussianBlur(image, ksize, sigmaX):
    """
    Apply Gaussian blur using CUDA if available, otherwise use CPU
    
    Args:
        image: Input image (numpy array)
        ksize: Kernel size (width, height) or (0, 0) for auto
        sigmaX: Gaussian kernel standard deviation in X direction
        
    Returns:
        Blurred image
    """
    if CUDA_AVAILABLE and image is not None:
        try:
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            # Calculate kernel size if (0, 0) is provided
            if ksize == (0, 0) or ksize[0] == 0 or ksize[1] == 0:
                # Calculate kernel size from sigma
                ksize = (int(6 * sigmaX + 1) | 1, int(6 * sigmaX + 1) | 1)  # Make odd
                if ksize[0] < 1:
                    ksize = (1, 1)
            
            # Create Gaussian filter for CUDA (OpenCV 4.6.0 uses createGaussianFilter)
            gaussian_filter = cv2.cuda.createGaussianFilter(
                gpu_img.type(), -1, ksize, sigmaX, sigmaX
            )
            
            # Apply filter on GPU
            gpu_blurred = gaussian_filter.apply(gpu_img)
            
            # Download from GPU
            result = gpu_blurred.download()
            return result
        except Exception as e:
            # Fallback to CPU silently (don't print warning for every call)
            return cv2.GaussianBlur(image, ksize, sigmaX)
    else:
        # Use CPU
        return cv2.GaussianBlur(image, ksize, sigmaX)


def cuda_batch_operations(images, operations):
    """
    Apply multiple operations on a batch of images using CUDA
    
    Args:
        images: List of images
        operations: List of operations to apply (e.g., [('resize', (320, 240)), ('cvtColor', cv2.COLOR_BGR2RGB)])
        
    Returns:
        List of processed images
    """
    if CUDA_AVAILABLE and images:
        results = []
        for image in images:
            if image is None:
                results.append(None)
                continue
            
            try:
                # Upload to GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                # Apply operations
                current = gpu_img
                for op_name, op_args in operations:
                    if op_name == 'resize':
                        current = cv2.cuda.resize(current, op_args)
                    elif op_name == 'cvtColor':
                        current = cv2.cuda.cvtColor(current, op_args)
                    elif op_name == 'gaussianBlur':
                        current = cv2.cuda.GaussianBlur(current, op_args[0], op_args[1])
                
                # Download from GPU
                result = current.download()
                results.append(result)
            except Exception as e:
                print(f"⚠️ CUDA batch operation failed, using CPU: {e}")
                # Fallback to CPU
                result = image
                for op_name, op_args in operations:
                    if op_name == 'resize':
                        result = cv2.resize(result, op_args)
                    elif op_name == 'cvtColor':
                        result = cv2.cvtColor(result, op_args)
                    elif op_name == 'gaussianBlur':
                        result = cv2.GaussianBlur(result, op_args[0], op_args[1])
                results.append(result)
        return results
    else:
        # Use CPU
        results = []
        for image in images:
            result = image
            for op_name, op_args in operations:
                if op_name == 'resize':
                    result = cv2.resize(result, op_args)
                elif op_name == 'cvtColor':
                    result = cv2.cvtColor(result, op_args)
                elif op_name == 'gaussianBlur':
                    result = cv2.GaussianBlur(result, op_args[0], op_args[1])
            results.append(result)
        return results


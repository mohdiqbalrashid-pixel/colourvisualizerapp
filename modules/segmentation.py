from __future__ import annotations

import cv2
import numpy as np

from modules.config import (
    MASK_EXPAND_PIXELS,
    MASK_SHRINK_PIXELS,
    MASK_SMOOTH_KERNEL
)

def _generate_local_grabcut_mask(image: np.ndarray, seed_point: tuple[int, int]) -> np.ndarray:
    """
    Uses OpenCV's GrabCut algorithm locally. 
    100% Free, runs entirely on Streamlit Cloud without external APIs.
    """
    height, width = image.shape[:2]
    x, y = seed_point
    
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Set the whole image as 'Probable Background'
    mask[:] = cv2.GC_PR_BGD
    
    # Mark the clicked area as 'Definite Foreground'
    # We use an ellipse to capture a good sample of the wall's base color
    cv2.ellipse(mask, (x, y), (60, 40), 0, 0, 360, cv2.GC_FGD, -1)
    
    # Give the algorithm a bounding box to work within
    rect = (5, 5, width - 10, height - 10)
    
    try:
        # Run GrabCut for 3 iterations (balance between speed and accuracy)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        
        # Extract the foreground (1) and probable foreground (3)
        binary_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")
    except Exception:
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        
    return binary_mask

def _refine_edges(image: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    """
    Snaps the mask cleanly to doorframes and ceilings using the image's physical edges.
    """
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    try:
        refined_mask = cv2.ximgproc.guidedFilter(
            guide=guide, src=raw_mask, radius=MASK_SMOOTH_KERNEL, eps=1e-4
        )
    except AttributeError:
        refined_mask = cv2.GaussianBlur(raw_mask, (MASK_SMOOTH_KERNEL, MASK_SMOOTH_KERNEL), 0)
    
    # Morphological adjustments to fill micro-holes
    if MASK_EXPAND_PIXELS > 0:
        kernel = np.ones((MASK_EXPAND_PIXELS, MASK_EXPAND_PIXELS), np.uint8)
        refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)
        
    if MASK_SHRINK_PIXELS > 0:
        kernel = np.ones((MASK_SHRINK_PIXELS, MASK_SHRINK_PIXELS), np.uint8)
        refined_mask = cv2.erode(refined_mask, kernel, iterations=1)
        
    return np.clip(refined_mask, 0, 255).astype(np.uint8)

def create_wall_mask(image: np.ndarray, seed_point: tuple[int, int]) -> np.ndarray:
    """Main entry point for preview.py"""
    height, width = image.shape[:2]
    x, y = seed_point
    
    if not (0 <= x < width and 0 <= y < height):
        return np.zeros((height, width), dtype=np.uint8)

    raw_mask = _generate_local_grabcut_mask(image, (x, y))
    final_mask = _refine_edges(image, raw_mask)
    
    return final_mask

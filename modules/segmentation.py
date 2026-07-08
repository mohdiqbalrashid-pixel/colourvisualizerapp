from __future__ import annotations

import cv2
import numpy as np

from modules.config import (
    MASK_EXPAND_PIXELS,
    MASK_SHRINK_PIXELS,
    MASK_SMOOTH_KERNEL
)

def _generate_grabcut_mask(
    image: np.ndarray, 
    seed_point: tuple[int, int], 
    radius: int = 40
) -> np.ndarray:
    """
    Runs GrabCut segmentation surrounding a targeted seed coordinates location.
    """
    height, width = image.shape[:2]
    
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Mark overall matrix as probable background
    mask[:] = cv2.GC_PR_BGD
    
    # Assert clicked coordinate perimeter as definite foreground
    x, y = seed_point
    cv2.circle(mask, (x, y), radius, cv2.GC_FGD, -1)
    
    # Boundary container initialization
    rect = (5, 5, width - 10, height - 10)
    
    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    except Exception:
        # Emergency robust fallback to clean array if process interrupts
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        
    return binary_mask

def _refine_mask(image: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    """
    Snaps raw masks directly to hard geometric structural components.
    """
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    try:
        refined_mask = cv2.ximgproc.guidedFilter(
            guide=guide, src=raw_mask, radius=MASK_SMOOTH_KERNEL, eps=1e-4
        )
    except AttributeError:
        # Graceful degradation fallback if binary bindings mismatch
        refined_mask = cv2.GaussianBlur(raw_mask, (MASK_SMOOTH_KERNEL, MASK_SMOOTH_KERNEL), 0)
    
    # Structural morph transformations to fill micro-voids
    if MASK_EXPAND_PIXELS > 0:
        kernel = np.ones((MASK_EXPAND_PIXELS, MASK_EXPAND_PIXELS), np.uint8)
        refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)
        
    if MASK_SHRINK_PIXELS > 0:
        kernel = np.ones((MASK_SHRINK_PIXELS, MASK_SHRINK_PIXELS), np.uint8)
        refined_mask = cv2.erode(refined_mask, kernel, iterations=1)
        
    return np.clip(refined_mask, 0, 255).astype(np.uint8)

def create_wall_mask(image_np: np.ndarray, seed_point: tuple[int, int]) -> np.ndarray:
    """
    Primary interface gateway executing surface discovery calculations.
    """
    height, width = image_np.shape[:2]
    x, y = seed_point
    
    if not (0 <= x < width and 0 <= y < height):
        return np.zeros((height, width), dtype=np.uint8)

    raw_mask = _generate_grabcut_mask(image_np, (x, y))
    final_mask = _refine_mask(image_np, raw_mask)
    
    return final_mask

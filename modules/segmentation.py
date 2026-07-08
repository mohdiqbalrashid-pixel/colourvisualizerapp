from __future__ import annotations

import cv2
import numpy as np
from modules.config import MASK_SMOOTH_KERNEL

def create_wall_mask(image: np.ndarray, seed_point: tuple[int, int], tolerance: int = 20) -> np.ndarray:
    """
    Generates a wall mask using LAB color space and a tolerance threshold.
    """
    height, width = image.shape[:2]
    x, y = seed_point
    
    if not (0 <= x < width and 0 <= y < height):
        return np.zeros((height, width), dtype=np.uint8)

    # 1. Convert to LAB color space for better lighting/shadow perception
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # 2. Prepare the flood-fill algorithm
    # FloodFill requires a mask exactly 2 pixels larger than the image
    mask = np.zeros((height + 2, width + 2), np.uint8)
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    
    # 3. Execute the fill starting from the user's click
    cv2.floodFill(
        lab_image, mask, (x, y), (255, 255, 255),
        loDiff=lo_diff, upDiff=up_diff, flags=flags
    )
    
    # Extract the actual mask size
    raw_mask = mask[1:height+1, 1:width+1]
    
    # 4. Close small holes (like outlets, picture hooks, or scuffs)
    kernel = np.ones((5, 5), np.uint8)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    
    # 5. Snap cleanly to doorframes and ceilings using Guided Filter
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    try:
        final_mask = cv2.ximgproc.guidedFilter(
            guide=guide, src=raw_mask, radius=MASK_SMOOTH_KERNEL, eps=1e-4
        )
    except AttributeError:
        final_mask = cv2.GaussianBlur(raw_mask, (MASK_SMOOTH_KERNEL, MASK_SMOOTH_KERNEL), 0)
        
    # Ensure it remains a hard binary mask before returning
    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
    
    return final_mask

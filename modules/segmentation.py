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

    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    mask = np.zeros((height + 2, width + 2), np.uint8)
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    
    cv2.floodFill(
        lab_image, mask, (x, y), (255, 255, 255),
        loDiff=lo_diff, upDiff=up_diff, flags=flags
    )
    
    raw_mask = mask[1:height+1, 1:width+1]
    
    kernel = np.ones((5, 5), np.uint8)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    try:
        final_mask = cv2.ximgproc.guidedFilter(
            guide=guide, src=raw_mask, radius=MASK_SMOOTH_KERNEL, eps=1e-4
        )
    except AttributeError:
        final_mask = cv2.GaussianBlur(raw_mask, (MASK_SMOOTH_KERNEL, MASK_SMOOTH_KERNEL), 0)
        
    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
    
    return final_mask

def generate_auto_regions(image: np.ndarray, num_regions: int = 4) -> list[np.ndarray]:
    """
    Uses K-Means clustering to auto-detect the largest surfaces in the room instantly.
    """
    # 1. Shrink image for lightning-fast processing
    height, width = image.shape[:2]
    small_img = cv2.resize(image, (320, 240))
    lab_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2LAB)
    pixel_values = lab_img.reshape((-1, 3)).astype(np.float32)

    # 2. Run K-Means to find the dominant color/lighting clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(pixel_values, num_regions, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.reshape((small_img.shape[0], small_img.shape[1]))
    masks = []

    # 3. Extract the clean regions
    for i in range(num_regions):
        mask_small = np.where(labels == i, 255, 0).astype(np.uint8)
        
        # Keep only the largest connected component in this cluster to avoid random speckles
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_small, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean_mask_small = np.where(labels_im == largest_label, 255, 0).astype(np.uint8)
        else:
            clean_mask_small = mask_small

        # Scale back up to original image resolution
        mask_large = cv2.resize(clean_mask_small, (width, height), interpolation=cv2.INTER_NEAREST)

        # Only keep significant structural areas (ignoring tiny objects)
        area = np.count_nonzero(mask_large)
        total_area = height * width
        if area / total_area > 0.10: # Must be at least 10% of the photo
            masks.append(mask_large)

    # 4. Sort by size (largest first, usually the main walls/ceilings)
    masks.sort(key=np.count_nonzero, reverse=True)
    return masks[:3] # Return the top 3 dominant surfaces

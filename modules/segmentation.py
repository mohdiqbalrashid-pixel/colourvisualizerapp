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


def generate_auto_regions(image: np.ndarray, num_regions: int = 5) -> list[np.ndarray]:
    """
    Uses 5D Spatial K-Means clustering and geometric logic to auto-detect walls.
    """
    height, width = image.shape[:2]
    
    # 1. Shrink image slightly for lightning-fast processing
    scale = 320.0 / max(width, height)
    new_w, new_h = int(width * scale), int(height * scale)
    small_img = cv2.resize(image, (new_w, new_h))

    # 2. Edge-Preserving Blur (Eradicates shadows/textures, keeps doorframes sharp)
    smooth_img = cv2.bilateralFilter(small_img, d=9, sigmaColor=75, sigmaSpace=75)
    lab_img = cv2.cvtColor(smooth_img, cv2.COLOR_RGB2LAB)

    # 3. Construct 5D Features: [L, a, b, X, Y]
    # This prevents the floor and ceiling from merging if they are the same color
    y_coords, x_coords = np.mgrid[0:new_h, 0:new_w]
    spatial_weight = 0.65  # Controls how strictly we enforce physical boundaries
    
    x_scaled = (x_coords / new_w) * 255.0 * spatial_weight
    y_scaled = (y_coords / new_h) * 255.0 * spatial_weight

    features = np.dstack((lab_img, x_scaled, y_scaled))
    pixel_values = features.reshape((-1, 5)).astype(np.float32)

    # 4. Execute Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(pixel_values, num_regions, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((new_h, new_w))

    masks = []
    for i in range(num_regions):
        mask_small = np.where(labels == i, 255, 0).astype(np.uint8)

        # Keep only the largest solid block in this cluster (deletes random scattered pixels)
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_small, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean_mask_small = np.where(labels_im == largest_label, 255, 0).astype(np.uint8)
        else:
            clean_mask_small = mask_small

        # 5. Geometric Filtering
        area = np.count_nonzero(clean_mask_small)
        total_area = new_w * new_h
        
        # Rule A: Surface must take up at least 12% of the room
        if area / total_area < 0.12:
            continue 

        # Rule B: Gravity Check. 
        # Check where the mask touches the edges of the photo.
        touches_top = np.any(clean_mask_small[0:5, :] > 0)
        touches_left = np.any(clean_mask_small[:, 0:5] > 0)
        touches_right = np.any(clean_mask_small[:, -5:] > 0)
        touches_bottom = np.any(clean_mask_small[-5:, :] > 0)

        # If it ONLY touches the bottom of the photo, it is the floor/rug. Discard it.
        if touches_bottom and not (touches_top or touches_left or touches_right):
            continue

        # 6. Scale back up and smooth edges
        mask_large = cv2.resize(clean_mask_small, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Soften the jagged mathematical edges for realistic painting
        mask_large = cv2.GaussianBlur(mask_large, (11, 11), 0)
        _, mask_large = cv2.threshold(mask_large, 127, 255, cv2.THRESH_BINARY)

        masks.append(mask_large)

    # Sort by size so Surface 1 is always the biggest wall
    masks.sort(key=np.count_nonzero, reverse=True)
    
    # Return the top 3 legitimate surfaces
    return masks[:3]

from __future__ import annotations

import cv2
import numpy as np
from modules.config import MASK_SMOOTH_KERNEL

def create_wall_mask(image: np.ndarray, seed_point: tuple[int, int], tolerance: int = 20) -> np.ndarray:
    """
    Generates a wall mask using LAB color space and a tolerance threshold.
    Used for the manual 'Magic Wand' click tool in the interactive workspace.
    """
    height, width = image.shape[:2]
    x, y = seed_point
    
    if not (0 <= x < width and 0 <= y < height):
        return np.zeros((height, width), dtype=np.uint8)

    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # FloodFill requires a mask exactly 2 pixels larger than the image
    mask = np.zeros((height + 2, width + 2), np.uint8)
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    
    cv2.floodFill(
        lab_image, mask, (x, y), (255, 255, 255),
        loDiff=lo_diff, upDiff=up_diff, flags=flags
    )
    
    raw_mask = mask[1:height+1, 1:width+1]
    
    # Close small holes
    kernel = np.ones((5, 5), np.uint8)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    
    # Snap cleanly to edges
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    try:
        final_mask = cv2.ximgproc.guidedFilter(
            guide=guide, src=raw_mask, radius=MASK_SMOOTH_KERNEL, eps=1e-4
        )
    except AttributeError:
        final_mask = cv2.GaussianBlur(raw_mask, (MASK_SMOOTH_KERNEL, MASK_SMOOTH_KERNEL), 0)
        
    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
    
    return final_mask


def generate_auto_regions(image: np.ndarray, num_regions: int = 3) -> list[np.ndarray]:
    """
    Uses Graph-Based Segmentation with CLAHE Illumination Flattening and Canny Fencing.
    Used for the instant '0-Click' auto-detection buttons.
    """
    height, width = image.shape[:2]
    
    # Downscale for lightning-fast processing
    scale = 400.0 / max(width, height)
    new_w, new_h = int(width * scale), int(height * scale)
    small_img = cv2.resize(image, (new_w, new_h))

    # --- ILLUMINATION FLATTENING (CLAHE) ---
    lab = cv2.cvtColor(small_img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    limg = cv2.merge((cl, a, b))
    flat_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # --- HARD STRUCTURAL EDGES (FENCE) ---
    gray = cv2.cvtColor(flat_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    # --- GRAPH-BASED SEGMENTATION ---
    blurred = cv2.bilateralFilter(flat_img, d=9, sigmaColor=75, sigmaSpace=75)
    
    try:
        segmenter = cv2.ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5, k=400, min_size=int(new_w * new_h * 0.08)
        )
        labels = segmenter.processImage(blurred)
    except AttributeError:
        # Fallback to K-Means if Graph Segmentation fails in the environment
        pixel_values = blurred.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(pixel_values, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape((new_h, new_w))

    unique_labels, counts = np.

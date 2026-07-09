from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
from modules.config import MASK_SMOOTH_KERNEL

# Initialize the tiny ONNX AI session in memory
# We use 'u2netp' (the lightweight version) so it is lightning fast on free cloud servers
ai_session = new_session("u2netp")

def _get_ai_clutter_mask(image: np.ndarray) -> np.ndarray:
    """
    Uses a local ONNX Neural Network to detect furniture, people, and plants.
    Returns a binary mask of all the clutter in the room.
    """
    img_pil = Image.fromarray(image)
    
    # Run the AI background remover (returns an RGBA image)
    result = remove(img_pil, session=ai_session)
    
    # Extract the Alpha channel (where the solid furniture is)
    result_array = np.array(result)
    alpha_channel = result_array[:, :, 3]
    
    # Create a clean binary mask of the furniture
    _, clutter_mask = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
    
    # Expand the clutter mask slightly to ensure paint doesn't sneak around the edges of couches
    kernel = np.ones((5, 5), np.uint8)
    clutter_mask = cv2.dilate(clutter_mask, kernel, iterations=2)
    
    return clutter_mask

def create_wall_mask(image: np.ndarray, seed_point: tuple[int, int], tolerance: int = 20) -> np.ndarray:
    """
    Hybrid Engine: Generates the crisp wall mask using math, 
    then uses AI to erase any furniture that got in the way.
    """
    height, width = image.shape[:2]
    x, y = seed_point
    
    if not (0 <= x < width and 0 <= y < height):
        return np.zeros((height, width), dtype=np.uint8)

    # 1. Get the crisp architectural wall mask (Math)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    raw_wall_mask = np.zeros((height + 2, width + 2), np.uint8)
    lo_diff, up_diff = (tolerance, tolerance, tolerance), (tolerance, tolerance, tolerance)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    
    cv2.floodFill(
        lab_image, raw_wall_mask, (x, y), (255, 255, 255),
        loDiff=lo_diff, upDiff=up_diff, flags=flags
    )
    wall_mask = raw_wall_mask[1:height+1, 1:width+1]
    
    # Clean the wall edges
    kernel = np.ones((5, 5), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    try:
        wall_mask = cv2.ximgproc.guidedFilter(guide=guide, src=wall_mask, radius=MASK_SMOOTH_KERNEL, eps=1e-4)
    except AttributeError:
        wall_mask = cv2.GaussianBlur(wall_mask, (MASK_SMOOTH_KERNEL, MASK_SMOOTH_KERNEL), 0)
        
    _, final_wall = cv2.threshold(wall_mask, 127, 255, cv2.THRESH_BINARY)

    # 2. Get the semantic furniture mask (AI)
    clutter_mask = _get_ai_clutter_mask(image)

    # 3. Semantic Subtraction (Remove the couch from the wall!)
    # Bitwise AND NOT: Keep the wall, but delete anything that the AI says is furniture
    smart_mask = cv2.bitwise_and(final_wall, cv2.bitwise_not(clutter_mask))
    
    return smart_mask

def generate_auto_regions(image: np.ndarray, num_regions: int = 3) -> list[np.ndarray]:
    """
    The 0-Click Auto-Detector (Also upgraded with Semantic Subtraction)
    """
    height, width = image.shape[:2]
    scale = 400.0 / max(width, height)
    new_w, new_h = int(width * scale), int(height * scale)
    small_img = cv2.resize(image, (new_w, new_h))

    # Illumination Flattening
    lab = cv2.cvtColor(small_img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    limg = cv2.merge((clahe.apply(l_channel), a, b))
    flat_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(flat_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 30, 100), np.ones((3, 3), np.uint8), iterations=2)
    blurred = cv2.bilateralFilter(flat_img, d=9, sigmaColor=75, sigmaSpace=75)
    
    try:
        segmenter = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=400, min_size=int(new_w * new_h * 0.08))
        labels = segmenter.processImage(blurred)
    except AttributeError:
        pixel_values = blurred.reshape((-1, 3)).astype(np.float32)
        _, labels, _ = cv2.kmeans(pixel_values, 6, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape((new_h, new_w))

    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_labels = [l for _, l in sorted(zip(counts, unique_labels), reverse=True)]
    
    # Get the AI Clutter mask (only compute once per image!)
    clutter_mask = _get_ai_clutter_mask(image)
    small_clutter = cv2.resize(clutter_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    masks = []
    for label in sorted_labels:
        mask_small = np.where(labels == label, 255, 0).astype(np.uint8)
        
        # Enforce Canny Edges AND AI Clutter Edges
        mask_small = cv2.bitwise_and(mask_small, cv2.bitwise_not(edges))
        mask_small = cv2.bitwise_and(mask_small, cv2.bitwise_not(small_clutter))
        
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_small, connectivity=8)
        if num_labels > 1:
            mask_small = np.where(labels_im == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)

        if np.count_nonzero(mask_small) / (new_w * new_h) < 0.08: continue
        
        if np.any(mask_small[-5:, :] > 0) and not (np.any(mask_small[0:5, :] > 0) or np.any(mask_small[:, 0:5] > 0) or np.any(mask_small[:, -5:] > 0)):
            continue

        mask_large = cv2.threshold(cv2.GaussianBlur(cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST), (15, 15), 0), 127, 255, cv2.THRESH_BINARY)[1]
        masks.append(mask_large)
        if len(masks) == num_regions: break

    return masks

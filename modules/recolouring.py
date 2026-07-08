from __future__ import annotations

import cv2
import numpy as np
from modules.config import DEFAULT_PAINT_STRENGTH

def apply_paint(
    image: np.ndarray, 
    mask: np.ndarray, 
    target_rgb: tuple[int, int, int], 
    strength: float = DEFAULT_PAINT_STRENGTH
) -> np.ndarray:
    """
    Applies solid paint by replacing the color channels entirely 
    while shifting the luminance to preserve texture and shadows.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # 1. Convert to LAB space (L = Light/Shadows, A/B = Color)
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    
    # 2. Get target LAB values for the Jotun Paint
    target_img = np.uint8([[target_rgb]])
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)
    target_l, target_a, target_b = target_lab
    
    bool_mask = mask > 0
    if not np.any(bool_mask):
        return image.copy()
        
    # 3. Luminance Shifting
    # We find the average shadow/light of the wall, and shift it to match the paint's brightness.
    # This preserves the variance (the texture of the wall) but changes the base brightness.
    masked_l = l_channel[bool_mask]
    avg_l = np.mean(masked_l)
    l_shift = target_l - avg_l
    
    new_l = np.clip(l_channel + l_shift, 0, 255)
    
    # 4. OVERWRITE the color completely (This fixes the "transparent" look)
    l_channel[bool_mask] = new_l[bool_mask]
    a_channel[bool_mask] = target_a
    b_channel[bool_mask] = target_b
    
    # 5. Convert back to standard RGB image
    merged_lab = cv2.merge([l_channel, a_channel, b_channel]).astype(np.uint8)
    painted_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    
    # 6. Feather the edges so it blends smoothly into corners and doorframes
    soft_mask = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0
    soft_mask_3d = np.dstack([soft_mask, soft_mask, soft_mask]) * strength
    
    # Combine the 100% solid painted area with the original image
    final_output = (painted_rgb * soft_mask_3d + image * (1.0 - soft_mask_3d)).astype(np.uint8)
    
    return final_output

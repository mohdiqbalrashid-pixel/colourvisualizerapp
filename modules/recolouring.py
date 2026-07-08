from __future__ import annotations

import cv2
import numpy as np

from modules.config import DEFAULT_PAINT_STRENGTH

def _create_boundary_alpha(mask: np.ndarray, strength: float) -> np.ndarray:
    """
    Generates feathered transitions for seamless paint edges.
    """
    base = mask.astype(np.float32) / 255.0
    base = np.clip(base, 0.0, 1.0)
    
    binary = (base > 0.05).astype(np.uint8)
    if not np.any(binary > 0):
        return np.zeros_like(base, dtype=np.float32)
        
    soft = cv2.GaussianBlur(base, (11, 11), 0)
    return np.clip(soft * float(strength), 0.0, 1.0)

def apply_paint(
    image_np: np.ndarray,
    mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    strength: float = DEFAULT_PAINT_STRENGTH,
) -> np.ndarray:
    """
    High accuracy paint simulation using luminance shifts in CIELAB color space.
    """
    if image_np is None or mask is None:
        return image_np

    # Form verification
    if image_np.dtype != np.uint8:
        image_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image_np.copy()
        
    if mask.ndim == 3:
        mask_uint8 = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_uint8 = mask.astype(np.uint8)

    alpha = _create_boundary_alpha(mask_uint8, strength)
    if not np.any(alpha > 0):
        return image_uint8

    # Convert source scene and destination swatch into LAB space
    image_lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    target_rgb_array = np.uint8([[target_rgb]])
    target_lab = cv2.cvtColor(target_rgb_array, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)
    target_l, target_a, target_b = target_lab

    l_channel, a_channel, b_channel = cv2.split(image_lab)
    bool_mask = mask_uint8 > 0
    
    # Balance luminosity relative to contextual source illumination averages
    avg_original_l = np.mean(l_channel[bool_mask]) if np.any(bool_mask) else 128.0
    l_shift = target_l - avg_original_l
    
    # Factor shifting and combine
    shifted_l = np.clip(l_channel + l_shift, 0.0, 255.0)
    
    # Allocate properties
    l_channel[bool_mask] = shifted_l[bool_mask]
    a_channel[bool_mask] = target_a
    b_channel[bool_mask] = target_b
    
    # Reassemble and transition safely back to RGB space
    modified_lab = cv2.merge([l_channel, a_channel, b_channel])
    modified_rgb = cv2.cvtColor(np.clip(modified_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    # Blend target area using mask opacity profiles
    alpha_3d = np.dstack([alpha] * 3)
    final_output = (modified_rgb * alpha_3d + image_uint8 * (1.0 - alpha_3d)).astype(np.uint8)
    
    return final_output

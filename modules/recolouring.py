import cv2
import numpy as np

def apply_paint(
    image: np.ndarray, 
    mask: np.ndarray, 
    target_rgb: tuple[int, int, int], 
    strength: float = 1.0
) -> np.ndarray:
    """
    Applies paint using advanced Multiply and Screen blending modes
    for photorealistic shadow and highlight retention.
    """
    # Ensure formats
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # Create the solid color layer
    solid_color = np.full_like(image, target_rgb, dtype=np.float32)
    img_float = image.astype(np.float32)
    
    # Normalize original image for calculations (0.0 to 1.0)
    img_normalized = img_float / 255.0
    color_normalized = solid_color / 255.0
    
    # 1. MULTIPLY BLEND (Perfect for shadows)
    # This darkens the new color based on the shadows in the original wall
    multiply_blend = img_normalized * color_normalized
    
    # 2. SCREEN BLEND (Perfect for highlights)
    # This ensures glare from windows/lights reflects realistically off the new paint
    screen_blend = 1.0 - (1.0 - img_normalized) * (1.0 - color_normalized)
    
    # 3. LUMINANCE MASKING (The secret sauce)
    # We use the original wall's brightness to mix the Multiply and Screen layers.
    # Dark areas get the Multiply blend; bright areas get the Screen blend.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_3d = np.dstack([gray, gray, gray])
    
    # Combine them based on the wall's lighting
    blended_normalized = (multiply_blend * (1.0 - gray_3d)) + (screen_blend * gray_3d)
    
    # Convert back to standard image format
    blended_final = (blended_normalized * 255.0).clip(0, 255).astype(np.uint8)
    
    # Feather the mask for seamless edges
    soft_mask = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
    soft_mask_3d = np.dstack([soft_mask, soft_mask, soft_mask]) * strength
    
    # Apply only where the AI mask says the wall is
    final_output = (blended_final * soft_mask_3d + image * (1.0 - soft_mask_3d)).astype(np.uint8)
    
    return final_output

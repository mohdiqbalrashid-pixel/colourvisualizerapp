from __future__ import annotations

import os
import tempfile
import streamlit as st
import replicate
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

# Securely load the API token from Streamlit Cloud Secrets
if "REPLICATE_API_TOKEN" in st.secrets:
    os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

def _get_sam_mask_from_api(image: np.ndarray, click_point: tuple[int, int]) -> np.ndarray:
    """
    Sends the image and click coordinates to Meta's SAM via Replicate API.
    Returns a perfect, context-aware AI mask.
    """
    x, y = click_point
    
    # 1. Convert OpenCV array to a PIL Image
    img_pil = Image.fromarray(image)
    
    # 2. Save it to a temporary file (The most reliable way to send files via API)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
        img_pil.save(temp_img, format='JPEG', quality=85)
        temp_path = temp_img.name
    
    try:
        # 3. Open the file as a file object and hand it to Replicate
        with open(temp_path, "rb") as file_handle:
            output_url = replicate.run(
                "pablodawson/segment-anything-model-automatic:latest",
                input={
                    "image": file_handle,
                    "input_points": f"[{x}, {y}]",
                    "input_labels": "[1]" # 1 tells the AI: "This click is the object I want"
                }
            )
            
        # 4. Clean up the temporary file immediately after sending
        os.remove(temp_path)
        
        # 5. Download the resulting mask and convert it back to an OpenCV array
        response = requests.get(output_url)
        mask_img = Image.open(BytesIO(response.content)).convert("L")
        return np.array(mask_img)
        
    except Exception as e:
        # Failsafe: Clean up the file if an error occurs and return an empty mask
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.error(f"AI Connection Error: {e}")
        return np.zeros(image.shape[:2], dtype=np.uint8)

def create_wall_mask(image: np.ndarray, seed_point: tuple[int, int]) -> np.ndarray:
    """
    Main entry point for preview.py.
    Checks to ensure the click is within the image, then calls the AI.
    """
    height, width = image.shape[:2]
    x, y = seed_point
    
    if not (0 <= x < width and 0 <= y < height):
        return np.zeros((height, width), dtype=np.uint8)

    final_mask = _get_sam_mask_from_api(image, (x, y))
    
    return final_mask

import streamlit as st
import replicate
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

# Note: You would need to add 'replicate' to your requirements.txt

def get_sam_mask_from_api(image: np.ndarray, click_point: tuple[int, int]) -> np.ndarray:
    """
    Sends the image and click coordinates to Meta's SAM via Replicate API.
    Returns a perfect, context-aware AI mask.
    """
    x, y = click_point
    
    # 1. Convert numpy image to a format we can send over the internet
    img_pil = Image.fromarray(image)
    img_byte_arr = BytesIO()
    img_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # 2. Call the AI Model (Using a standard SAM endpoint on Replicate)
    # The API key is securely stored in Streamlit Cloud's settings
    try:
        output_url = replicate.run(
            "pablodawson/segment-anything-model-automatic:latest",
            input={
                "image": img_byte_arr,
                "input_points": f"[{x}, {y}]",
                "input_labels": "[1]" # 1 means "foreground"
            }
        )
        
        # 3. Download the resulting mask and convert it back to an OpenCV array
        response = requests.get(output_url)
        mask_img = Image.open(BytesIO(response.content)).convert("L")
        return np.array(mask_img)
        
    except Exception as e:
        st.error(f"AI API Error: {e}")
        return np.zeros(image.shape[:2], dtype=np.uint8)

def create_wall_mask(image: np.ndarray, seed_point: tuple[int, int]) -> np.ndarray:
    """Main entry point for preview.py"""
    with st.spinner("Meta SAM is analyzing room geometry..."):
        return get_sam_mask_from_api(image, seed_point)

import streamlit as st
import numpy as np
from PIL import Image

def build_sidebar() -> None:
    st.header("Controls")
    
    # 1. Image Upload
    uploaded_file = st.file_uploader("1. Upload Room Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV-friendly format and save to memory
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state["uploaded_image"] = np.array(image)
        
        # If a new image is uploaded, clear the old painted results
        if st.button("Reset Image"):
            st.session_state["wall_mask"] = None
            st.session_state["painted_image"] = None
            st.session_state["selected_surface_point"] = None
            st.rerun()

    st.divider()
    
    # 2. Colour Picker
    st.subheader("2. Choose Paint Colour")
    hex_color = st.color_picker("Pick a hex colour", "#BA0C2F") # Default Jotun Red
    
    # Convert HEX to RGB for our OpenCV recolouring engine
    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    st.session_state["selected_colour"] = rgb_color

    st.divider()
    
    # 3. Paint Settings
    st.session_state["paint_strength"] = st.slider(
        "Paint Strength / Opacity", 
        min_value=0.0, 
        max_value=1.0, 
        value=1.0, 
        step=0.05
    )

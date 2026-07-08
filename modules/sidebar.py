import streamlit as st
import numpy as np
from PIL import Image

def build_sidebar() -> None:
    st.header("Controls")
    
    # 1. Image Upload
    uploaded_file = st.file_uploader("1. Upload Room Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state["uploaded_image"] = np.array(image)
        
        if st.button("Reset Image Workspace"):
            st.session_state["wall_mask"] = None
            st.session_state["painted_image"] = None
            st.session_state["selected_surface_point"] = None
            st.rerun()

    st.divider()
    
    # 2. Masking Controls (NEW)
    st.subheader("2. Selection Tools")
    st.session_state["click_mode"] = st.radio(
        "Click Action",
        ["New Wall 🔄", "Add to Wall ➕", "Erase from Wall ➖"],
        horizontal=True
    )
    
    st.session_state["tolerance"] = st.slider(
        "Selection Tolerance (Higher = Grabs more shadows)", 
        min_value=5, max_value=80, value=20, step=1
    )

    st.divider()
    
    # 3. Colour Picker
    st.subheader("3. Choose Paint Colour")
    hex_color = st.color_picker("Pick a hex colour", "#BA0C2F") 
    
    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    st.session_state["selected_colour"] = rgb_color
    
    st.session_state["paint_strength"] = st.slider(
        "Paint Strength / Opacity", 0.0, 1.0, 1.0, 0.05
    )

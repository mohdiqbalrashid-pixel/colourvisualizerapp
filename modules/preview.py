import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas

from modules.segmentation import create_wall_mask
from modules.recolouring import apply_paint

def _apply_and_save_paint():
    """Helper function to quickly apply paint when masks change."""
    original_image = st.session_state["uploaded_image"]
    mask = st.session_state.get("wall_mask")
    color = st.session_state.get("selected_colour", (186, 12, 47))
    strength = st.session_state.get("paint_strength", 1.0)
    
    if mask is not None:
        painted = apply_paint(original_image, mask, color, strength)
        st.session_state["painted_image"] = painted

def build_preview() -> None:
    if st.session_state.get("uploaded_image") is None:
        st.info("👈 Please upload an image in the sidebar to get started.")
        return

    original_image = st.session_state["uploaded_image"]
    
    # Use the painted image if it exists. This fixes the "Image below" issue!
    display_image = st.session_state.get("painted_image", original_image)
    
    st.subheader("Interactive Workspace")
    
    # Create Professional Tabs for different tools
    tab1, tab2 = st.tabs(["🪄 AI Magic Wand", "📐 Manual Polygons"])
    
    with tab1:
        st.caption("Click any shadow or missed spot to automatically fill it.")
        
        # Render the image for clicking
        click_data = streamlit_image_coordinates(display_image, key="ai_wand")
        
        if click_data is not None:
            point = (click_data["x"], click_data["y"])
            
            # Only trigger if they clicked a new location
            if st.session_state.get("selected_surface_point") != point:
                st.session_state["selected_surface_point"] = point
                
                click_mode = st.session_state.get("click_mode", "Add to Wall ➕")
                tolerance = st.session_state.get("tolerance", 20)
                
                with st.spinner("Calculating..."):
                    new_chunk = create_wall_mask(original_image, point, tolerance)
                    current_mask = st.session_state.get("wall_mask")
                    
                    if current_mask is None or "New Wall" in click_mode:
                        st.session_state["wall_mask"] = new_chunk
                    elif "Add" in click_mode:
                        st.session_state["wall_mask"] = cv2.bitwise_or(current_mask, new_chunk)
                    elif "Erase" in click_mode:
                        st.session_state["wall_mask"] = cv2.bitwise_and(current_mask, cv2.bitwise_not(new_chunk))
                    
                _apply_and_save_paint()
                st.rerun() # Refresh the interface to show the new paint instantly
                
    with tab2:
        st.caption("Click around a shape to draw a polygon. Double-click to close the shape.")
        
        col1, col2 = st.columns(2)
        with col1:
            manual_action = st.radio("Polygon Action", ["Add Paint ➕", "Erase Paint ➖"], horizontal=True)
        with col2:
            if st.button("Apply Polygon Area", use_container_width=True):
                if st.session_state.get("last_canvas_mask") is not None:
                    drawn_mask = st.session_state["last_canvas_mask"]
                    current_mask = st.session_state.get("wall_mask")
                    
                    if current_mask is None:
                        current_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                        
                    # Merge the user's drawn shape with the AI mask
                    if "Add" in manual_action:
                        st.session_state["wall_mask"] = cv2.bitwise_or(current_mask, drawn_mask)
                    else:
                        st.session_state["wall_mask"] = cv2.bitwise_and(current_mask, cv2.bitwise_not(drawn_mask))
                        
                    _apply_and_save_paint()
                    st.rerun()
        
        # The interactive drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 180, 0.4)", # Semi-transparent pink drawing guide
            stroke_width=2,
            stroke_color="#FF00B4",
            background_image=Image.fromarray(display_image),
            update_streamlit=True,
            height=display_image.shape[0],
            width=display_image.shape[1],
            drawing_mode="polygon",
            key="polygon_canvas",
        )
        
        # Extract the shape the user drew into a Numpy array mask behind the scenes
        if canvas_result.image_data is not None:
            alpha_channel = canvas_result.image_data[:, :, 3]
            binary_drawn_mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
            st.session_state["last_canvas_mask"] = binary_drawn_mask

import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.segmentation import create_wall_mask
from modules.recolouring import apply_paint

def _apply_and_save_paint():
    """Helper function to apply paint when masks change."""
    original_image = st.session_state.get("uploaded_image")
    if original_image is None:
        return
        
    mask = st.session_state.get("wall_mask")
    color = st.session_state.get("selected_colour", (186, 12, 47))
    strength = st.session_state.get("paint_strength", 1.0)
    
    if mask is not None:
        painted = apply_paint(original_image, mask, color, strength)
        st.session_state["painted_image"] = painted

def build_preview() -> None:
    original_image = st.session_state.get("uploaded_image")
    if original_image is None:
        st.info("👈 Please upload an image in the sidebar to get started.")
        return

    display_image = st.session_state.get("painted_image")
    if display_image is None:
        display_image = original_image
    
    st.subheader("Interactive Workspace")
    
    tab1, tab2 = st.tabs(["🪄 AI Magic Wand", "📐 Manual Polygons"])
    
    with tab1:
        st.caption("Click any shadow or missed spot to automatically fill it.")
        
        click_data = streamlit_image_coordinates(display_image, key="ai_wand")
        
        if click_data is not None:
            point = (click_data["x"], click_data["y"])
            
            # Prevent infinite reloading loops
            if st.session_state.get("last_ai_click") != point:
                st.session_state["last_ai_click"] = point
                
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
                st.rerun() 
                
    with tab2:
        st.caption("Click multiple points on the image to draw a shape. Click 'Apply Shape' to fill it.")
        
        # Initialize our anchor point memory
        if "polygon_points" not in st.session_state:
            st.session_state["polygon_points"] = []
            
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            manual_action = st.radio("Polygon Action", ["Add Paint ➕", "Erase Paint ➖"], horizontal=True)
        with col2:
            # Only allow apply if they've drawn a proper shape (3 or more points)
            if st.button("Apply Shape", use_container_width=True) and len(st.session_state["polygon_points"]) > 2:
                # 1. Convert the user's clicked points into an OpenCV polygon
                pts = np.array(st.session_state["polygon_points"], np.int32).reshape((-1, 1, 2))
                poly_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(poly_mask, [pts], 255)
                
                # 2. Merge it with our existing wall mask
                current_mask = st.session_state.get("wall_mask")
                if current_mask is None:
                    current_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                    
                if "Add" in manual_action:
                    st.session_state["wall_mask"] = cv2.bitwise_or(current_mask, poly_mask)
                else:
                    st.session_state["wall_mask"] = cv2.bitwise_and(current_mask, cv2.bitwise_not(poly_mask))
                    
                # 3. Clear points and repaint!
                st.session_state["polygon_points"] = []
                _apply_and_save_paint()
                st.rerun()
                
        with col3:
            if st.button("Clear Points", use_container_width=True):
                st.session_state["polygon_points"] = []
                st.rerun()
        
        # Visually draw the polygon on a temporary canvas for the user
        poly_display = display_image.copy()
        points = st.session_state["polygon_points"]
        
        for i, pt in enumerate(points):
            cv2.circle(poly_display, pt, 5, (255, 0, 180), -1) # Draw anchor dot
            if i > 0:
                cv2.line(poly_display, points[i-1], pt, (255, 0, 180), 2) # Draw connecting line
        
        # Close the shape visually if they have enough points
        if len(points) > 2:
            cv2.line(poly_display, points[-1], points[0], (255, 0, 180), 2)
        
        # Render the clickable image
        poly_click = streamlit_image_coordinates(poly_display, key="poly_clicker")
        
        if poly_click is not None:
            new_point = (poly_click["x"], poly_click["y"])
            # Only add the point if it's new (prevents double-firing)
            if not points or points[-1] != new_point:
                st.session_state["polygon_points"].append(new_point)
                st.rerun()

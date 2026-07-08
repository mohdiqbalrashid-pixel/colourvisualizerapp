import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.segmentation import create_wall_mask
from modules.recolouring import apply_paint

# --- UNDO ENGINE ---
def _save_state_to_history():
    """Saves the current mask to memory before we change it."""
    if "mask_history" not in st.session_state:
        st.session_state["mask_history"] = []
        
    current_mask = st.session_state.get("wall_mask")
    if current_mask is not None:
        # Save a copy to the stack
        st.session_state["mask_history"].append(current_mask.copy())
        
        # Keep memory clean (limit to 15 undo steps)
        if len(st.session_state["mask_history"]) > 15:
            st.session_state["mask_history"].pop(0)

def _undo_last_action():
    """Restores the mask from the last saved state."""
    if st.session_state.get("mask_history"):
        st.session_state["wall_mask"] = st.session_state["mask_history"].pop()
        _apply_and_save_paint()
        st.session_state["polygon_points"] = [] # Clear any half-drawn polygons

# --- PAINT ENGINE ---
def _apply_and_save_paint():
    """Applies the Jotun colour to the final mask."""
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
    
    # Initialize mask if it doesn't exist yet
    if st.session_state.get("wall_mask") is None:
        st.session_state["wall_mask"] = np.zeros(original_image.shape[:2], dtype=np.uint8)
    
    st.subheader("Interactive Workspace")
    
    # The Global Undo Button
    col_title, col_undo = st.columns([3, 1])
    with col_undo:
        st.button("↩️ Undo Last Action", on_click=_undo_last_action, use_container_width=True, 
                  disabled=not bool(st.session_state.get("mask_history")))
    
    tab1, tab2 = st.tabs(["🪄 AI Magic Wand", "🖌️ Manual Touch-ups"])
    
    # ==========================================
    # TAB 1: AI MAGIC WAND
    # ==========================================
    with tab1:
        st.caption("Click any shadow or missed spot to automatically fill it.")
        
        click_data = streamlit_image_coordinates(display_image, key="ai_wand")
        
        if click_data is not None:
            point = (click_data["x"], click_data["y"])
            
            if st.session_state.get("last_ai_click") != point:
                st.session_state["last_ai_click"] = point
                
                click_mode = st.session_state.get("click_mode", "Add to Wall ➕")
                tolerance = st.session_state.get("tolerance", 20)
                
                with st.spinner("Calculating..."):
                    _save_state_to_history() # Save state before AI runs
                    
                    new_chunk = create_wall_mask(original_image, point, tolerance)
                    current_mask = st.session_state.get("wall_mask")
                    
                    if "New Wall" in click_mode:
                        st.session_state["wall_mask"] = new_chunk
                    elif "Add" in click_mode:
                        st.session_state["wall_mask"] = cv2.bitwise_or(current_mask, new_chunk)
                    elif "Erase" in click_mode:
                        st.session_state["wall_mask"] = cv2.bitwise_and(current_mask, cv2.bitwise_not(new_chunk))
                    
                _apply_and_save_paint()
                st.rerun() 
                
    # ==========================================
    # TAB 2: MANUAL BRUSH & POLYGONS
    # ==========================================
    with tab2:
        if "polygon_points" not in st.session_state:
            st.session_state["polygon_points"] = []
            
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            tool = st.radio("Tool", ["Brush 🔴", "Polygon 📐"], horizontal=True)
        with t_col2:
            manual_action = st.radio("Action", ["Add Paint ➕", "Erase Paint ➖"], horizontal=True)
        with t_col3:
            if tool == "Brush 🔴":
                brush_size = st.slider("Brush Size", min_value=5, max_value=150, value=30, step=5)
            else:
                if st.button("Apply Shape", use_container_width=True) and len(st.session_state["polygon_points"]) > 2:
                    _save_state_to_history()
                    
                    pts = np.array(st.session_state["polygon_points"], np.int32).reshape((-1, 1, 2))
                    poly_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [pts], 255)
                    
                    current_mask = st.session_state["wall_mask"]
                    if "Add" in manual_action:
                        st.session_state["wall_mask"] = cv2.bitwise_or(current_mask, poly_mask)
                    else:
                        st.session_state["wall_mask"] = cv2.bitwise_and(current_mask, cv2.bitwise_not(poly_mask))
                        
                    st.session_state["polygon_points"] = []
                    _apply_and_save_paint()
                    st.rerun()

        # Visual feedback layer
        interactive_display = display_image.copy()
        
        # Draw polygon anchor points if using polygon tool
        if tool == "Polygon 📐":
            points = st.session_state["polygon_points"]
            for i, pt in enumerate(points):
                cv2.circle(interactive_display, pt, 5, (255, 0, 180), -1)
                if i > 0:
                    cv2.line(interactive_display, points[i-1], pt, (255, 0, 180), 2)
            if len(points) > 2:
                cv2.line(interactive_display, points[-1], points[0], (255, 0, 180), 2)
                
        # Register the click
        manual_click = streamlit_image_coordinates(interactive_display, key="manual_clicker")
        
        if manual_click is not None:
            new_point = (manual_click["x"], manual_click["y"])
            
            # BRUSH LOGIC
            if tool == "Brush 🔴":
                if st.session_state.get("last_brush_click") != new_point:
                    st.session_state["last_brush_click"] = new_point
                    
                    _save_state_to_history()
                    
                    # Stamp a perfect circle of paint (or eraser) onto the mask
                    color_val = 255 if "Add" in manual_action else 0
                    cv2.circle(st.session_state["wall_mask"], new_point, brush_size, color_val, -1)
                    
                    _apply_and_save_paint()
                    st.rerun()
                    
            # POLYGON LOGIC
            elif tool == "Polygon 📐":
                points = st.session_state["polygon_points"]
                if not points or points[-1] != new_point:
                    st.session_state["polygon_points"].append(new_point)
                    st.rerun()

import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.segmentation import create_wall_mask, generate_auto_regions
from modules.recolouring import apply_paint

# --- UNDO ENGINE ---
def _save_state_to_history():
    if "mask_history" not in st.session_state:
        st.session_state["mask_history"] = []
        
    current_mask = st.session_state.get("wall_mask")
    if current_mask is not None:
        st.session_state["mask_history"].append(current_mask.copy())
        if len(st.session_state["mask_history"]) > 15:
            st.session_state["mask_history"].pop(0)

def _undo_last_action():
    if st.session_state.get("mask_history"):
        st.session_state["wall_mask"] = st.session_state["mask_history"].pop()
        _apply_and_save_paint()
        st.session_state["polygon_points"] = []

# --- PAINT ENGINE WITH EDGE SHIFTING ---
def _apply_and_save_paint():
    original_image = st.session_state.get("uploaded_image")
    if original_image is None:
        return
        
    mask = st.session_state.get("wall_mask")
    color = st.session_state.get("selected_colour", (186, 12, 47))
    strength = st.session_state.get("paint_strength", 1.0)
    edge_shift = st.session_state.get("edge_shift", 0)
    
    if mask is not None:
        working_mask = mask.copy()
        
        # Apply mathematical expansion (dilate) or shrinking (erode)
        if edge_shift > 0:
            kernel = np.ones((3, 3), np.uint8)
            working_mask = cv2.dilate(working_mask, kernel, iterations=edge_shift)
        elif edge_shift < 0:
            kernel = np.ones((3, 3), np.uint8)
            working_mask = cv2.erode(working_mask, kernel, iterations=abs(edge_shift))
            
        painted = apply_paint(original_image, working_mask, color, strength)
        st.session_state["painted_image"] = painted

def build_preview() -> None:
    original_image = st.session_state.get("uploaded_image")
    if original_image is None:
        st.info("👈 Please upload an image in the sidebar to get started.")
        return

    display_image = st.session_state.get("painted_image")
    if display_image is None:
        display_image = original_image
    
    if st.session_state.get("wall_mask") is None:
        st.session_state["wall_mask"] = np.zeros(original_image.shape[:2], dtype=np.uint8)
    
    # --- AUTO-DETECTION ENGINE ---
    image_id = id(original_image)
    if st.session_state.get("last_processed_image_id") != image_id:
        with st.spinner("🤖 AI mapping room geometry..."):
            st.session_state["auto_masks"] = generate_auto_regions(original_image)
            st.session_state["last_processed_image_id"] = image_id

    # Display the 1-Click Fast Paint buttons
    if st.session_state.get("auto_masks"):
        st.write("⚡ **Fast Paint:** Auto-Detected Surfaces")
        cols = st.columns(len(st.session_state["auto_masks"]))
        
        for i, (col, auto_mask) in enumerate(zip(cols, st.session_state["auto_masks"])):
            if col.button(f"Paint Surface {i+1}", use_container_width=True):
                _save_state_to_history()
                
                click_mode = st.session_state.get("click_mode", "New Wall 🔄")
                if "Add" in click_mode:
                    st.session_state["wall_mask"] = cv2.bitwise_or(st.session_state["wall_mask"], auto_mask)
                else:
                    st.session_state["wall_mask"] = auto_mask
                    
                _apply_and_save_paint()
                st.rerun()
                
        st.divider()

    st.subheader("Interactive Workspace")
    
    # --- GLOBAL WORKSPACE CONTROLS ---
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        # The new Edge Shifter slider
        new_edge_shift = st.slider(
            "↔️ Fine-tune Edges (Shrink ➖ | Expand ➕)", 
            min_value=-15, max_value=15, value=st.session_state.get("edge_shift", 0), step=1
        )
        # Re-calculate the paint only if the slider was actually moved
        if new_edge_shift != st.session_state.get("edge_shift", 0):
            st.session_state["edge_shift"] = new_edge_shift
            _apply_and_save_paint()
            st.rerun()
            
    with ctrl_col2:
        st.markdown("<br>", unsafe_allow_html=True) # Formatting alignment
        st.button("↩️ Undo Last Action", on_click=_undo_last_action, use_container_width=True, 
                  disabled=not bool(st.session_state.get("mask_history")))
    
    # --- INTERACTIVE TABS ---
    tab1, tab2 = st.tabs(["🪄 AI Magic Wand", "🖌️ Manual Touch-ups"])
    
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
                    _save_state_to_history()
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

        interactive_display = display_image.copy()
        
        if tool == "Polygon 📐":
            points = st.session_state["polygon_points"]
            for i, pt in enumerate(points):
                cv2.circle(interactive_display, pt, 5, (255, 0, 180), -1)
                if i > 0:
                    cv2.line(interactive_display, points[i-1], pt, (255, 0, 180), 2)
            if len(points) > 2:
                cv2.line(interactive_display, points[-1], points[0], (255, 0, 180), 2)
                
        manual_click = streamlit_image_coordinates(interactive_display, key="manual_clicker")
        
        if manual_click is not None:
            new_point = (manual_click["x"], manual_click["y"])
            
            if tool == "Brush 🔴":
                if st.session_state.get("last_brush_click") != new_point:
                    st.session_state["last_brush_click"] = new_point
                    _save_state_to_history()
                    
                    color_val = 255 if "Add" in manual_action else 0
                    cv2.circle(st.session_state["wall_mask"], new_point, brush_size, color_val, -1)
                    
                    _apply_and_save_paint()
                    st.rerun()
                    
            elif tool == "Polygon 📐":
                points = st.session_state["polygon_points"]
                if not points or points[-1] != new_point:
                    st.session_state["polygon_points"].append(new_point)
                    st.rerun()

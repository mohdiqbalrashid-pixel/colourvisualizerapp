import streamlit as st
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.segmentation import create_wall_mask
from modules.recolouring import apply_paint

def build_preview() -> None:
    if st.session_state.get("uploaded_image") is None:
        st.info("👈 Please upload an image in the sidebar to get started.")
        return

    original_image = st.session_state["uploaded_image"]
    
    st.subheader("Interactive Canvas")
    st.caption("Click the wall. If it misses a shadow, select 'Add to Wall ➕' and click the missed area!")
    
    click_data = streamlit_image_coordinates(original_image, key="image_canvas")
    
    if click_data is not None:
        point = (click_data["x"], click_data["y"])
        
        # Only process if they clicked a NEW spot
        if st.session_state.get("selected_surface_point") != point:
            st.session_state["selected_surface_point"] = point
            
            # Fetch user settings from the sidebar
            click_mode = st.session_state.get("click_mode", "New Wall 🔄")
            tolerance = st.session_state.get("tolerance", 20)
            
            with st.spinner("Calculating boundaries..."):
                # 1. Get the chunk of wall for this specific click
                new_chunk = create_wall_mask(original_image, point, tolerance)
                
                # 2. Get the existing mask (if any)
                current_mask = st.session_state.get("wall_mask")
                
                # 3. Merge the masks based on the user's selected mode!
                if current_mask is None or "New Wall" in click_mode:
                    st.session_state["wall_mask"] = new_chunk
                elif "Add" in click_mode:
                    # Bitwise OR combines the two masks together
                    st.session_state["wall_mask"] = cv2.bitwise_or(current_mask, new_chunk)
                elif "Erase" in click_mode:
                    # Bitwise AND NOT subtracts the new chunk from the existing mask
                    st.session_state["wall_mask"] = cv2.bitwise_and(current_mask, cv2.bitwise_not(new_chunk))
                
            with st.spinner("Applying realistic paint..."):
                color = st.session_state.get("selected_colour", (186, 12, 47))
                strength = st.session_state.get("paint_strength", 1.0)
                mask = st.session_state["wall_mask"]
                
                painted = apply_paint(original_image, mask, color, strength)
                st.session_state["painted_image"] = painted
                
    # Display the final painted result
    if st.session_state.get("painted_image") is not None:
        st.success("Paint applied! Adjust tolerance and click again if needed.")
        st.image(st.session_state["painted_image"], use_container_width=True)

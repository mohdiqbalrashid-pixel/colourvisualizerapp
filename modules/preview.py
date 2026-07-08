import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.segmentation import create_wall_mask
from modules.recolouring import apply_paint

def build_preview() -> None:
    if st.session_state.get("uploaded_image") is None:
        st.info("👈 Please upload an image in the sidebar to get started.")
        return

    original_image = st.session_state["uploaded_image"]
    
    st.subheader("3. Click on the wall to paint")
    st.caption("The AI will automatically detect the edges of the wall you click on.")
    
    # Display the image and capture click coordinates
    # We use a key so Streamlit doesn't constantly reload the image
    click_data = streamlit_image_coordinates(original_image, key="image_canvas")
    
    if click_data is not None:
        # Extract X and Y coordinates
        point = (click_data["x"], click_data["y"])
        
        # Only re-calculate if the user clicked a NEW spot
        if st.session_state.get("selected_surface_point") != point:
            st.session_state["selected_surface_point"] = point
            
            with st.spinner("AI is calculating wall boundaries..."):
                mask = create_wall_mask(original_image, point)
                st.session_state["wall_mask"] = mask
                
            with st.spinner("Applying realistic paint..."):
                color = st.session_state.get("selected_colour", (186, 12, 47))
                strength = st.session_state.get("paint_strength", 1.0)
                
                painted = apply_paint(original_image, mask, color, strength)
                st.session_state["painted_image"] = painted
                
    # Display the final painted result, or the original image if nothing is clicked
    if st.session_state.get("painted_image") is not None:
        st.success("Paint applied!")
        st.image(st.session_state["painted_image"], use_container_width=True)
    else:
        # If no click has happened yet, just show the standard image
        pass

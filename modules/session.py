import streamlit as st

def initialise_session() -> None:
    """Sets up the blank memory states for the application on first load."""
    defaults = {
        "uploaded_image": None,
        "selected_colour": None,
        "selected_surface_point": None,
        "wall_mask": None,
        "painted_image": None,
        "paint_strength": 1.0,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sync_legacy_to_app() -> None:
    """A placeholder to keep app.py happy if we need to sync old variables later."""
    pass

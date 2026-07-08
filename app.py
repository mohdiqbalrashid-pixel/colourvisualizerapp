import streamlit as st

from modules.config import APP_CAPTION, APP_NAME, APP_VERSION, configure_page
from modules.preview import build_preview
from modules.session import initialise_session, sync_legacy_to_app
from modules.sidebar import build_sidebar


def build_header() -> None:
    title_col, version_col = st.columns([4, 1])

    with title_col:
        st.title(f"🎨 {APP_NAME}")
        st.caption(APP_CAPTION)

    with version_col:
        st.markdown(
            f"""
            <div style="
                text-align:right;
                padding-top:18px;
                color:#666666;
                font-size:0.9rem;
            ">
                {APP_VERSION}
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_status_bar() -> None:
    image_loaded = st.session_state.get("uploaded_image") is not None
    colour_selected = st.session_state.get("selected_colour") is not None
    point_selected = st.session_state.get("selected_surface_point") is not None
    mask_available = st.session_state.get("wall_mask") is not None
    painted_available = st.session_state.get("painted_image") is not None

    status_items = []

    status_items.append("✅ Image loaded" if image_loaded else "⬜ Image pending")
    status_items.append("✅ Colour selected" if colour_selected else "⬜ Colour pending")
    status_items.append("✅ Surface clicked" if point_selected else "⬜ Surface pending")
    status_items.append("✅ Mask ready" if mask_available else "⬜ Mask pending")
    status_items.append("✅ Preview ready" if painted_available else "⬜ Preview pending")

    st.caption(" &nbsp; | &nbsp; ".join(status_items))


def main() -> None:
    configure_page()
    initialise_session()

    build_header()
    build_status_bar()

    st.divider()

    left, right = st.columns([1, 2])
    
    with left:
        build_sidebar()
        
    with right:
        build_preview()


if __name__ == "__main__":
    main()

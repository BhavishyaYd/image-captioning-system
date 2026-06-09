from __future__ import annotations

from pathlib import Path

import streamlit as st
from predict import generate_caption

st.set_page_config(
    page_title="Smart Image Captioning",
    page_icon="🖼️",
    layout="centered",
)


def init_state() -> None:
    """Initialize session variables."""
    if "current_caption" not in st.session_state:
        st.session_state.current_caption = None
    if "current_image_bytes" not in st.session_state:
        st.session_state.current_image_bytes = None
    if "current_image_name" not in st.session_state:
        st.session_state.current_image_name = None


def set_current_image(image_bytes: bytes, image_name: str) -> None:
    """Persist selected image in session state."""
    st.session_state.current_image_bytes = image_bytes
    st.session_state.current_image_name = image_name
    st.session_state.current_caption = None


def clear_current_image() -> None:
    """Remove image and caption."""
    st.session_state.current_image_bytes = None
    st.session_state.current_image_name = None
    st.session_state.current_caption = None


def generate_caption_for_image() -> str | None:
    """Generate caption for current image."""
    if not st.session_state.current_image_bytes:
        st.warning("Upload an image first.")
        return None

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    filename = st.session_state.current_image_name or "image.jpg"
    temp_path = outputs_dir / filename
    temp_path.write_bytes(st.session_state.current_image_bytes)

    with st.spinner("Generating caption..."):
        caption = generate_caption(temp_path)

    if not caption:
        st.warning("Caption generation returned empty output.")
        return None

    st.session_state.current_caption = caption
    return caption


init_state()

st.title("🖼️ Image Captioning System")
st.caption("Upload an image and generate a caption using AI.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    set_current_image(uploaded_file.getvalue(), uploaded_file.name)

if st.session_state.current_image_bytes:
    st.image(
        st.session_state.current_image_bytes,
        caption=st.session_state.current_image_name,
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ Generate Caption", use_container_width=True, type="primary"):
            try:
                caption = generate_caption_for_image()
                if caption:
                    st.success("Caption generated!")
            except Exception as exc:
                st.error(f"Failed to generate caption: {exc}")

    with col2:
        if st.button("🔄 Clear Image", use_container_width=True):
            clear_current_image()
            st.rerun()

if st.session_state.current_caption:
    st.divider()
    st.subheader("Generated Caption")
    st.text_area(
        "Caption",
        value=st.session_state.current_caption,
        height=100,
        disabled=True,
        label_visibility="collapsed",
    )
    st.download_button(
        "📥 Download Caption",
        data=st.session_state.current_caption.encode("utf-8"),
        file_name="caption.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.divider()
st.caption("Powered by TensorFlow/Keras + Streamlit")

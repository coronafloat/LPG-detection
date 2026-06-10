import streamlit as st

from config import CLAHE_DATASET_URL, ORIGINAL_DATASET_URL
from views.styles import render_page_header


def render_dataset_card(title, description, button_label, url):
    st.markdown(
        f"""
        <div class="info-card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button(button_label, url, use_container_width=True)


def render_datasets():
    render_page_header(
        "Datasets",
        "Two dataset variants are prepared to compare raw visual conditions against contrast-enhanced imagery for LPG cylinder detection.",
    )

    col1, col2 = st.columns(2)
    with col1:
        render_dataset_card(
            "Original Dataset",
            "The original dataset contains curated LPG cylinder images collected from diverse sources, preserving real lighting, background, angle, and camera variations.",
            "Open Original Dataset",
            ORIGINAL_DATASET_URL,
        )
    with col2:
        render_dataset_card(
            "CLAHE Dataset",
            "The CLAHE dataset applies localized contrast enhancement to improve visibility of cylinder boundaries, labels, and surface details in challenging lighting conditions.",
            "Open CLAHE Dataset",
            CLAHE_DATASET_URL,
        )

    st.caption(
        "Replace ORIGINAL_DATASET_URL and CLAHE_DATASET_URL in config.py with the final Google Drive links."
    )

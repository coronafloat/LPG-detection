import streamlit as st

from views.styles import render_page_header


def render_about():
    render_page_header(
        "About",
        "A computer vision project for detecting and counting LPG cylinders in real-world distribution and monitoring environments.",
    )

    st.write(
        """
        This project develops an LPG cylinder detection system using YOLO-based object detection.
        Based on the project journal, the system focuses on recognizing four LPG cylinder categories:
        3 kg, 5.5 kg, 12 kg, and 50 kg. The goal is to support faster and more consistent visual
        monitoring for warehouse inventory, distribution workflows, and safety-related inspection.
        """
    )
    st.write(
        """
        The research highlights common industrial detection challenges, including uneven illumination,
        shadows, reflective cylinder surfaces, overlapping objects, and cluttered backgrounds. To improve
        robustness, the project evaluates original images alongside CLAHE-enhanced images, where localized
        contrast enhancement helps emphasize cylinder boundaries, labels, and structural details before
        detection.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Detection Scope</h3>
                <p>Detects LPG cylinder categories commonly found in Indonesian distribution contexts: 3 kg, 5.5 kg, 12 kg, and 50 kg.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Model Approach</h3>
                <p>Uses YOLO object detection for real-time localization and classification, with support for image and video-based testing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>Enhancement Strategy</h3>
                <p>Includes CLAHE preprocessing to improve local contrast under low-light, shadowed, or reflective industrial conditions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

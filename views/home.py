import tempfile

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from utils.detection import process_frame
from views.styles import render_page_header


def render_home():
    render_page_header(
        "LPG Gas Detection & Counting System",
        "Upload a YOLO model, test LPG cylinder detection on images or videos, and count objects inside a configurable active region.",
    )

    st.sidebar.header("System Settings")

    uploaded_model_file = st.sidebar.file_uploader("Upload Model (.pt)", type=["pt"])
    model = None

    if uploaded_model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            tmp_file.write(uploaded_model_file.read())
            model_path = tmp_file.name

        try:
            model = YOLO(model_path)
            st.sidebar.success(f"Model '{uploaded_model_file.name}' loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    else:
        st.sidebar.warning("Please upload a .pt model file to get started.")

    enhancement_option = st.sidebar.selectbox(
        "Select Image Enhancement:",
        ["None (Original)", "CLAHE", "HE (Histogram Equalization)", "CS (Contrast Stretching)"],
    )

    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Region Settings")
    st.sidebar.info("Objects are counted when their center point enters this box.")

    r_x1 = st.sidebar.slider("X1 Position (Left)", 0, 640, 100)
    r_y1 = st.sidebar.slider("Y1 Position (Top)", 0, 480, 100)
    r_x2 = st.sidebar.slider("X2 Position (Right)", 0, 640, 500)
    r_y2 = st.sidebar.slider("Y2 Position (Bottom)", 0, 480, 400)

    if model is None:
        st.info("Hello! Please upload a .pt model file from the left sidebar to get started.")
        return

    tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])

    with tab1:
        st.header("Image Test")
        uploaded_img = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            temp_ids = set()

            processed_img, count = process_frame(
                img_bgr.copy(),
                model,
                enhancement_option,
                (r_x1, r_y1, r_x2, r_y2),
                conf_threshold,
                temp_ids,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Input")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader(f"Region Filter Result ({enhancement_option})")
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.metric("Total Detected", count)

    with tab2:
        st.header("Video Test")
        uploaded_video = st.file_uploader("Upload Video (MP4)", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            if st.button("Reset Count"):
                st.session_state.counted_ids = set()

            stop_button = st.button("Stop Video")

            if "counted_ids" not in st.session_state:
                st.session_state.counted_ids = set()

            current_ids = st.session_state.counted_ids

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))
                processed_frame, count = process_frame(
                    frame,
                    model,
                    enhancement_option,
                    (r_x1, r_y1, r_x2, r_y2),
                    conf_threshold,
                    current_ids,
                )

                stframe.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    caption=f"Total Counted: {count}",
                )

            cap.release()

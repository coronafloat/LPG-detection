import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from pathlib import Path

from lpg_detection.frame_processor import process_frame

BASE_DIR = Path(__file__).resolve().parent
MODEL_PRESETS = {
    "YOLO12s Original": BASE_DIR / "models" / "yolo12s_original" / "best.pt",
    "YOLO12s CLAHE": BASE_DIR / "models" / "yolo12s_clahe" / "best.pt",
}


@st.cache_resource(show_spinner=False)
def load_model_from_path(model_path):
    return YOLO(model_path)


@st.cache_resource(show_spinner=False)
def load_model_from_upload(file_name, file_bytes):
    # Save the uploaded model to a temporary file so YOLO can read its path.
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(file_bytes)
        model_path = tmp_file.name

    return YOLO(model_path)

# ==========================================
# 1. GUI & SIDEBAR CONFIGURATION
# ==========================================

st.set_page_config(page_title="LPG Gas Detection - YOLOv12", layout="wide")

st.title("LPG Gas Detection & Counting System")
st.markdown("Object Detection, Region Counting, & Enhancement")

# --- Sidebar Control Panel ---
st.sidebar.header("System Settings")

# Model selection
model = None
selected_model_label = None
model_source = st.sidebar.radio(
    "Model Source",
    ["Preset Model", "Upload Custom Model"],
)

if model_source == "Preset Model":
    selected_model_label = st.sidebar.selectbox(
        "Select Model:",
        list(MODEL_PRESETS.keys()),
    )
    selected_model_path = MODEL_PRESETS[selected_model_label]

    if selected_model_path.exists():
        try:
            model = load_model_from_path(str(selected_model_path))
            st.sidebar.success(f"Model '{selected_model_label}' loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    else:
        st.sidebar.warning(
            "Model file not found. Place the model at: "
            f"`{selected_model_path.relative_to(BASE_DIR)}`"
        )
else:
    uploaded_model_file = st.sidebar.file_uploader("Upload Model (.pt)", type=['pt'])

    if uploaded_model_file is not None:
        try:
            model = load_model_from_upload(
                uploaded_model_file.name,
                uploaded_model_file.getvalue(),
            )
            selected_model_label = uploaded_model_file.name
            st.sidebar.success(f"Model '{uploaded_model_file.name}' loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    else:
        st.sidebar.warning("Please upload a .pt model file to get started.")
    
# Enhancement selection
enhancement_option = st.sidebar.selectbox(
    "Select Image Enhancement:",
    ["None (Original)", "CLAHE", "HE (Histogram Equalization)", "CS (Contrast Stretching)"]
)

# Confidence Threshold
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# --- REGION SETTINGS ---
st.sidebar.markdown("---")
st.sidebar.subheader("Region Settings")
st.sidebar.info("Objects are counted when their center point enters this box.")

r_x1 = st.sidebar.slider("X1 Position (Left)", 0, 640, 100)
r_y1 = st.sidebar.slider("Y1 Position (Top)", 0, 480, 100)
r_x2 = st.sidebar.slider("X2 Position (Right)", 0, 640, 500)
r_y2 = st.sidebar.slider("Y2 Position (Bottom)", 0, 480, 400)

st.sidebar.markdown("---")
with st.sidebar.expander("Authors"):
    st.markdown("""
    - Christine Dewi Ph.D
    - Emmanuel Manggala Nusa
    - Cindy Cahya Juliandani
    - Ellena Putri Permana
    - Sabrina Rachman
    - Genesy Matthew Wibowo
    """)

st.sidebar.caption("© 2026 Fakultas Teknologi Informasi Universitas Kristen Satya Wacana")
# ==========================================
# 2. TAB DISPLAY (EXECUTION LOGIC)
# ==========================================

if model is not None:
    st.caption(f"Active model: {selected_model_label}")

    tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])

    # --- TAB 1: IMAGE ---
    with tab1:
        st.header("Image Test")
        uploaded_img = st.file_uploader("Upload Image (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            
            # Create a temporary set for this image, reset on each new upload.
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

    # --- TAB 2: VIDEO ---
    with tab2:
        st.header("Video Test")
        uploaded_video = st.file_uploader("Upload Video (MP4)", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            # Save the uploaded video to a temporary file.
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            # Reset counter button.
            if st.button("Reset Count"):
                st.session_state.counted_ids = set()
            
            # Stop button.
            stop_button = st.button("Stop Video")
            
            # Initialize the unique ID set using session state.
            if 'counted_ids' not in st.session_state:
                st.session_state.counted_ids = set()
                
            # Use the current session set so it can be reset with the button above.
            current_ids = st.session_state.counted_ids 

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize the frame to reduce processing load.
                frame = cv2.resize(frame, (640, 480))
                
                # Process frame.
                processed_frame, count = process_frame(
                    frame,
                    model,
                    enhancement_option,
                    (r_x1, r_y1, r_x2, r_y2),
                    conf_threshold,
                    current_ids,
                )
                
                # Display output.
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Total Counted: {count}")
                
            cap.release()
else:
    st.info("Hello! Please select an available preset model or upload a .pt model file from the left sidebar to get started.")

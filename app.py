import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# ==========================================
# 1. IMAGE ENHANCEMENT FUNCTION
# ==========================================

def apply_enhancement(image, method):
    """
    Applies the selected image enhancement technique.
    """
    if method == 'None (Original)':
        return image
    
    elif method == 'CLAHE':
        # Contrast Limited Adaptive Histogram Equalization
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    elif method == 'HE (Histogram Equalization)':
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    elif method == 'CS (Contrast Stretching)':
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return image

# ==========================================
# 2. GUI & SIDEBAR CONFIGURATION
# ==========================================

st.set_page_config(page_title="LPG Gas Detection - YOLOv11", layout="wide")

st.title("LPG Gas Detection & Counting System")
st.markdown("Object Detection, Region Counting, & Enhancement")

# --- Sidebar Control Panel ---
st.sidebar.header("System Settings")

# Upload Model
uploaded_model_file = st.sidebar.file_uploader("Upload Model (.pt)", type=['pt'])
model = None 

if uploaded_model_file is not None:
    # Save the uploaded model to a temporary file so YOLO can read its path.
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(uploaded_model_file.read())
        model_path = tmp_file.name

    try:
        # Load the model from the temporary path.
        model = YOLO(model_path)
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

# ==========================================
# 3. MAIN LOGIC (TRACKING & COUNTING)
# ==========================================

def process_frame(frame, model, enhancement_type, region_coords, conf, counted_ids):
    """
    Core function: enhancement -> tracking -> region filtering -> unique counting.
    The 'counted_ids' input is a set used to store unique object IDs.
    """
    # 1. Apply enhancement.
    enhanced_frame = apply_enhancement(frame, enhancement_type)
    
    # 2. Run tracking. persist=True keeps object IDs stable across frames.
    # Uses the default tracker, such as 'bytetrack.yaml' or 'botsort.yaml'.
    results = model.track(enhanced_frame, conf=conf, persist=True, verbose=False)
    
    # Get region coordinates.
    rx1, ry1, rx2, ry2 = region_coords
    
    # --- REGION & BACKGROUND VISUALIZATION ---
    # Darken the area outside the active region for focus.
    overlay = enhanced_frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, enhanced_frame, 0.7, 0, enhanced_frame)
    
    # Restore brightness inside the active region.
    roi = frame[ry1:ry2, rx1:rx2]
    # Ensure the coordinates are valid.
    if roi.size > 0:
        enhanced_frame[ry1:ry2, rx1:rx2] = apply_enhancement(roi, enhancement_type)

    # Draw the active region boundary in yellow.
    cv2.rectangle(enhanced_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

    # 3. Iterate through each detected object.
    # Ensure detected objects exist and boxes are not empty.
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist() # Get tracking IDs.
        clss = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[cls]
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # REGION FILTERING LOGIC
            # Objects are processed only if their center point is inside the active region.
            if (rx1 < center_x < rx2) and (ry1 < center_y < ry2):
                
                # Unique ID check: add it if it has not been counted before.
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                
                # Draw the counted bounding box in green.
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(enhanced_frame, (center_x, center_y), 5, (0, 0, 255), -1) # Red center point.
                
                # Display label and ID.
                cv2.putText(enhanced_frame, f"#{track_id} {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Objects outside the active region are not drawn.

    # Count total unique IDs that have entered the active region.
    total_count = len(counted_ids)

    # ==========================================
    # LEGEND
    # ==========================================

    # 1. Detection count text in red.
    cv2.putText(enhanced_frame, f"DETECTED: {total_count}", (10, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 2. Region legend text in yellow.
    cv2.putText(enhanced_frame, "Yellow Box: Active Region", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return enhanced_frame, total_count

# ==========================================
# 4. TAB DISPLAY (EXECUTION LOGIC)
# ==========================================

if model is not None:
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
            
            processed_img, count = process_frame(img_bgr.copy(), model, enhancement_option, (r_x1, r_y1, r_x2, r_y2), conf_threshold, temp_ids)
            
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
                processed_frame, count = process_frame(frame, model, enhancement_option, (r_x1, r_y1, r_x2, r_y2), conf_threshold, current_ids)
                
                # Display output.
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Total Counted: {count}")
                
            cap.release()
else:
    st.info("Hello! Please upload a .pt model file from the left sidebar to get started.")

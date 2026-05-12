import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# ==========================================
# 1. FUNGSI IMAGE ENHANCEMENT
# ==========================================

def apply_enhancement(image, method):
    """
    Menerapkan teknik enhancement sesuai pilihan user.
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
# 2. KONFIGURASI GUI & SIDEBAR
# ==========================================

st.set_page_config(page_title="Deteksi Gas LPG - YOLOv11", layout="wide")

st.title("🔍 Sistem Deteksi & Penghitungan Gas LPG")
st.markdown("Tugas Akhir Deep Learning: Object Detection, Region Counting, & Enhancement")

# --- Sidebar Control Panel ---
st.sidebar.header("⚙️ Pengaturan Sistem")

# Upload Model
uploaded_model_file = st.sidebar.file_uploader("Upload Model (.pt)", type=['pt'])
model = None 

if uploaded_model_file is not None:
    # Simpan file model ke temporary file agar bisa dibaca path-nya oleh YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(uploaded_model_file.read())
        model_path = tmp_file.name

    try:
        # Load model dari path temporary
        model = YOLO(model_path)
        st.sidebar.success(f"Model '{uploaded_model_file.name}' berhasil dimuat!")
    except Exception as e:
        st.sidebar.error(f"Error memuat model: {e}")
else:
    st.sidebar.warning("⚠️ Silakan upload file model .pt untuk memulai.")
    
# Pilihan Enhancement
enhancement_option = st.sidebar.selectbox(
    "Pilih Image Enhancement:",
    ["None (Original)", "CLAHE", "HE (Histogram Equalization)", "CS (Contrast Stretching)"]
)

# Confidence Threshold
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# --- PENGATURAN REGION ---
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Pengaturan Area (Region)")
st.sidebar.info("Objek akan dihitung jika titik tengahnya masuk ke dalam kotak ini.")

r_x1 = st.sidebar.slider("Posisi X1 (Kiri)", 0, 640, 100)
r_y1 = st.sidebar.slider("Posisi Y1 (Atas)", 0, 480, 100)
r_x2 = st.sidebar.slider("Posisi X2 (Kanan)", 0, 640, 500)
r_y2 = st.sidebar.slider("Posisi Y2 (Bawah)", 0, 480, 400)

# ==========================================
# 3. LOGIKA UTAMA (TRACKING & COUNTING)
# ==========================================

def process_frame(frame, model, enhancement_type, region_coords, conf, counted_ids):
    """
    Fungsi inti: Enhancement -> TRACKING -> Filter Region -> Unique Counting
    Input 'counted_ids' adalah SET untuk menyimpan ID unik.
    """
    # 1. Terapkan Enhancement
    enhanced_frame = apply_enhancement(frame, enhancement_type)
    
    # 2. Lakukan TRACKING (persist=True untuk menjaga ID objek)
    # Gunakan tracker default 'bytetrack.yaml' atau 'botsort.yaml'
    results = model.track(enhanced_frame, conf=conf, persist=True, verbose=False)
    
    # Ambil koordinat region
    rx1, ry1, rx2, ry2 = region_coords
    
    # --- VISUALISASI REGION & BACKGROUND ---
    # Gelapkan area di luar region (Efek Fokus)
    overlay = enhanced_frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, enhanced_frame, 0.7, 0, enhanced_frame)
    
    # Kembalikan kecerahan area di dalam region
    roi = frame[ry1:ry2, rx1:rx2]
    # Pastikan koordinat valid
    if roi.size > 0:
        enhanced_frame[ry1:ry2, rx1:rx2] = apply_enhancement(roi, enhancement_type)

    # Gambar Garis Batas Region (Kuning)
    cv2.rectangle(enhanced_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

    # 3. Loop setiap objek yang terdeteksi
    # Pastikan ada objek yang terdeteksi (boxes tidak kosong)
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist() # Ambil ID Tracking
        clss = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[cls]
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # LOGIKA FILTER REGION
            # Objek hanya diproses jika titik tengahnya ada di dalam kotak region
            if (rx1 < center_x < rx2) and (ry1 < center_y < ry2):
                
                # Cek Unik ID: Jika belum ada di database, tambahkan
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                
                # Visualisasi Bounding Box (Hijau - Counted)
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(enhanced_frame, (center_x, center_y), 5, (0, 0, 255), -1) # Titik tengah merah
                
                # Tampilkan Label + ID
                cv2.putText(enhanced_frame, f"#{track_id} {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Jika di luar region, objek tidak digambar kotaknya (atau bisa digambar warna merah jika mau)

    # Hitung total ID unik yang sudah masuk set
    total_count = len(counted_ids)

    # ==========================================
    # MEMBUAT LEGENDA (SESUAI REQUEST)
    # ==========================================

    # 1. Tulisan Jumlah Deteksi (Warna Merah: 0, 0, 255)
    cv2.putText(enhanced_frame, f"DETECTED: {total_count}", (10, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 2. Tulisan Legenda Region (Warna Kuning: 0, 255, 255)
    cv2.putText(enhanced_frame, "Yellow Box: Active Region", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return enhanced_frame, total_count

# ==========================================
# 4. TAMPILAN TAB (LOGIKA EKSEKUSI)
# ==========================================

if model is not None:
    tab1, tab2 = st.tabs(["🖼️ Deteksi Gambar", "🎥 Deteksi Video"])

    # --- TAB 1: GAMBAR ---
    with tab1:
        st.header("Uji Coba Gambar")
        uploaded_img = st.file_uploader("Upload Gambar (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            
            # Buat set sementara untuk gambar ini (reset setiap kali upload baru)
            temp_ids = set()
            
            processed_img, count = process_frame(img_bgr.copy(), model, enhancement_option, (r_x1, r_y1, r_x2, r_y2), conf_threshold, temp_ids)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Input")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader(f"Hasil Filter Region ({enhancement_option})")
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.metric("Total Terdeteksi", count)

    # --- TAB 2: VIDEO ---
    with tab2:
        st.header("Uji Coba Video")
        uploaded_video = st.file_uploader("Upload Video (MP4)", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            # Simpan video ke tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            # Tombol Reset Counter
            if st.button("Reset Counter Hitungan"):
                st.session_state.counted_ids = set()
            
            # Tombol Stop
            stop_button = st.button("Stop Video")
            
            # Inisialisasi Set ID Unik (Gunakan session state atau variabel lokal)
            if 'counted_ids' not in st.session_state:
                st.session_state.counted_ids = set()
                
            # Gunakan set lokal untuk run kali ini (agar bisa direset via tombol di atas)
            current_ids = st.session_state.counted_ids 

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame agar ringan
                frame = cv2.resize(frame, (640, 480))
                
                # PROSES FRAME
                processed_frame, count = process_frame(frame, model, enhancement_option, (r_x1, r_y1, r_x2, r_y2), conf_threshold, current_ids)
                
                # Tampilkan
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Total Counted: {count}")
                
            cap.release()
else:
    st.info("👋 Halo! Silakan upload file model (.pt) pada sidebar di sebelah kiri untuk memulai.")
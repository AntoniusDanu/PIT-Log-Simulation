import streamlit as st
st.set_page_config(page_title="Simulasi PIT Log", layout="wide")

import os
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from streamlit_autorefresh import st_autorefresh

# === Setup Upload Dir ===
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Load Models ===
@st.cache_resource
def load_models():
    yolo_model = YOLO("best.pt")
    ocr_model = PaddleOCR(lang='en')
    return yolo_model, ocr_model

yolo_model, ocr_model = load_models()

# === App State ===
if 'pit_log' not in st.session_state:
    st.session_state.pit_log = ["Empty"] * 5
    st.session_state.summary = []
    st.session_state.log = []
    st.session_state.image_queue = []
    st.session_state.simulation_running = False
    st.session_state.last_process_time = 0

# === Auto Refresh (10 detik) ===
st_autorefresh(interval=10 * 1000, key="refresh")

# === Utility Functions ===
def detect_plate(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Gagal Membaca"

    image = cv2.resize(image, (640, 640))
    cv2.imwrite(image_path, image)

    results = yolo_model(image_path)
    if not results or len(results[0].boxes) == 0:
        return "Tidak Terbaca"

    boxes = results[0].boxes.xyxy.numpy()
    x1, y1, x2, y2 = map(int, boxes[0])
    h, w, _ = image.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    plate_image = image[y1:y2, x1:x2]

    if plate_image.size == 0:
        return "Tidak Terbaca"

    plate_path = "plate.jpg"
    cv2.imwrite(plate_path, plate_image)
    ocr_results = ocr_model.ocr(plate_path, cls=True)
    text = ocr_results[0][0][1][0] if ocr_results else "Tidak Terbaca"
    return text

def process_image():
    if st.session_state.image_queue:
        image = st.session_state.image_queue.pop(0)
        file_path = os.path.join(UPLOAD_DIR, image)
        timestamp = datetime.now().strftime("%H:%M:%S")
        plate = detect_plate(file_path)
        for i in range(5):
            if st.session_state.pit_log[i] == "Empty":
                st.session_state.pit_log[i] = plate
                st.session_state.summary.append(f"PIT {i+1} - {timestamp} - {plate}")
                st.session_state.log.append(f"[{timestamp}] Plat terdeteksi: {plate} di PIT {i+1}")
                break

# === Layout ===
st.title("🚗 Simulasi Pembacaan Plat Nomor PIT 1–5")

col1, col2 = st.columns([2, 3])

with col1:
    st.header("📤 Upload Foto")
    uploaded_files = st.file_uploader("Upload satu atau lebih gambar plat nomor", type=["jpg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            st.session_state.image_queue.append(file.name)
            st.session_state.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Upload: {file.name}")
        st.success("Foto berhasil diupload ke antrean.")

    st.header("⚙️ Kontrol Simulasi")
    colb1, colb2, colb3 = st.columns(3)

    if colb1.button("▶️ Start"):
        st.session_state.simulation_running = True
        st.session_state.last_process_time = time.time() - 180

    if colb2.button("⏹ Stop"):
        st.session_state.simulation_running = False

    if colb3.button("🔄 Reset"):
        # Reset state
        st.session_state.pit_log = ["Empty"] * 5
        st.session_state.summary = []
        st.session_state.log = []
        st.session_state.image_queue = []

        # Hapus semua file dari folder uploads
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                st.warning(f"Gagal menghapus {filename}: {e}")
        st.success("Reset berhasil, termasuk menghapus foto yang telah diupload.")

    # Pemrosesan otomatis setiap 3 menit
    if st.session_state.simulation_running and st.session_state.image_queue:
        current_time = time.time()
        if current_time - st.session_state.last_process_time > 180:
            process_image()
            st.session_state.last_process_time = current_time

with col2:
    st.header("📊 Status PIT")
    pit_cols = st.columns(5)
    for i in range(5):
        pit_cols[i].metric(label=f"PIT {i+1}", value=st.session_state.pit_log[i])

    st.header("📜 Log Activity")
    for entry in reversed(st.session_state.log[-20:]):
        st.write(entry)

    st.header("📋 Summary")
    for item in reversed(st.session_state.summary[-10:]):
        st.markdown(f"- {item}")

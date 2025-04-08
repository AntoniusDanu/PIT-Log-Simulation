from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List
from pydantic import BaseModel
from datetime import datetime
import shutil
import os
import asyncio
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import pytz

# === Firebase Setup ===
cred = credentials.Certificate("./serviceAccountKey.json")  # Path to your Firebase credentials
firebase_admin.initialize_app(cred)
db = firestore.client()

# === FastAPI App ===
app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Directories ===
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Model Load ===
MODEL_PATH = "./best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model {MODEL_PATH} tidak ditemukan!")

yolo_model = YOLO(MODEL_PATH)
ocr = PaddleOCR(lang='en')

# === Simulation State ===
pit_log = ["Empty"] * 5
summary = []
log = []
image_queue = []
simulation_running = False

# === State Schema ===
class State(BaseModel):
    pit_log: List[str]
    summary: List[str]
    log: List[str]

# === OCR + YOLO Detection ===
def detect_plate(image_path):
    image = cv2.imread(image_path)
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
    ocr_results = ocr.ocr(plate_path, cls=True)
    text = ocr_results[0][0][1][0] if ocr_results else "Tidak Terbaca"
    timestamp = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S %Z")
    data = {
        "plate_number": text,
        "timestamp": timestamp,
        "bounding_box": boxes[0].tolist()
    }
    db.collection("detected_plates").add(data)
    return text

# === Background Process ===
async def process_images():
    global simulation_running
    while simulation_running:
        if image_queue:
            image = image_queue.pop(0)
            file_path = os.path.join(UPLOAD_DIR, image)
            timestamp = datetime.now().strftime("%H:%M:%S")
            plate = detect_plate(file_path)
            for i in range(5):
                if pit_log[i] == "Empty":
                    pit_log[i] = plate
                    summary.append(f"PIT {i+1} - {timestamp} - {plate}")
                    log.append(f"[{timestamp}] Plat terdeteksi: {plate} di PIT {i+1}")
                    break
            await asyncio.sleep(2)
        else:
            await asyncio.sleep(1)

# === Endpoints ===
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        image_queue.append(file.filename)
        log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Upload: {file.filename}")
    return {"status": "ok"}

@app.post("/start")
async def start_simulasi(background_tasks: BackgroundTasks):
    global simulation_running
    simulation_running = True
    background_tasks.add_task(process_images)
    return {"status": "started"}

@app.post("/stop")
async def stop_simulasi():
    global simulation_running
    simulation_running = False
    return {"status": "stopped"}

@app.post("/reset")
async def reset():
    global pit_log, summary, log, image_queue
    pit_log = ["Empty"] * 5
    summary.clear()
    log.clear()
    image_queue.clear()
    return {"status": "reset"}

@app.get("/state", response_model=State)
async def get_state():
    return State(
        pit_log=pit_log,
        summary=summary,
        log=log[-30:]
    )

@app.get("/")
def root():
    return FileResponse("frontend.html")

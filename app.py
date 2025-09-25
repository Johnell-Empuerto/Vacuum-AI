from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import uvicorn

app = FastAPI()

# Load your YOLO model
model = YOLO("best.pt")

# Serve static files (frontend)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
    return JSONResponse({"detections": detections})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

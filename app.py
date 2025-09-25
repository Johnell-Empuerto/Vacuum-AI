from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load your trained YOLO model
model = YOLO("best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run detection
    results = model(frame)

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                "label": model.names[int(cls)],
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2]
            })

    return JSONResponse(content={"detections": detections})

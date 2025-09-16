import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# ================================
# 1. App Setup
# ================================
app = Flask(__name__)

# Load model once (at startup)
MODEL_PATH = "best_model.h5"   # Make sure this file is in your project folder
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224
LABELS = ["Compostable", "Non-Compostable"]

# ================================
# 2. Prediction Endpoint
# ================================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = "temp.jpg"
    file.save(filepath)

    try:
        # Read and preprocess image
        img = cv2.imread(filepath)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img, verbose=0)[0][0]
        label = LABELS[int(pred > 0.5)]
        confidence = float(pred if label == "Non-Compostable" else 1 - pred)

        return jsonify({
            "label": label,
            "confidence": round(confidence * 100, 2)
        })

    finally:
        # Always remove the uploaded file after processing
        if os.path.exists(filepath):
            os.remove(filepath)

# ================================
# 3. Health Check
# ================================
@app.route("/", methods=["GET"])
def home():
    return "✅ Compostable Classifier API is running!"

# ================================
# 4. Run App
# ================================
if __name__ == "__main__":
    # Use host='0.0.0.0' for Render/Railway deployment
    app.run(host="0.0.0.0", port=5000, debug=True)

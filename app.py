import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ================================
# 1. App Setup
# ================================
app = Flask(__name__)
CORS(app)

# Load model once (at startup)
MODEL_PATH = "best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

# Match your dataset class order (from ImageDataGenerator flow_from_directory)
LABELS = [
    "apple", "banana", "cardboard", "cucumber", "food_organics",
    "leaves_flowers", "mango", "onion", "orange", "paper", "tomato", "vegetation",
    "battery", "clothes", "coins", "glass", "metal", "plastic", "shoes_slippers"
]

# Optional grouping for Compostable / Non-Compostable
GROUPS = {
    "compostable": {"apple","banana","cardboard","cucumber","food_organics",
                    "leaves_flowers","mango","onion","orange","paper","tomato","vegetation"},
    "non_compostable": {"battery","clothes","coins","glass","metal","plastic","shoes_slippers"}
}

# ================================
# 2. Prediction Endpoint
# ================================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join("/tmp", "temp.jpg")
    file.save(filepath)

    try:
        # Preprocess
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        preds = model.predict(img, verbose=0)[0]
        idx = np.argmax(preds)
        label = LABELS[idx]
        confidence = float(preds[idx])

        # Compostable / Non-Compostable group
        group = "compostable" if label in GROUPS["compostable"] else "non_compostable"

        # Confidence threshold
        THRESHOLD = 0.70
        if confidence < THRESHOLD:
            result_label = f"Not sure but {group}"
        else:
            result_label = label

        return jsonify({
            "label": result_label,
            "group": group,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to delete temp file: {e}")

# ================================
# 3. Serve index.html
# ================================
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

# ================================
# 4. Run App
# ================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

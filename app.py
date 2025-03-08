from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import json
import os
import gdown  # For downloading from Google Drive
import joblib
import sklearn


app = Flask(__name__)

# Google Drive links (IDs extracted from your links)
model_url = "https://drive.google.com/file/d/1aq2bGqD1QdU-OGv6t-lNRKDddnzqsHKn/view?usp=drive_link"
label_map_url = "https://drive.google.com/file/d/1Prb-KxeQ-LuBlWtyDSA4W0-9C-JRPO5E/view?usp=drive_link"
scaler_url = "https://drive.google.com/file/d/1jb6ay-oAgURpfh0UfWbtqo5iRtqeZhaW/view?usp=drive_link"

# File paths
model_path = "ISL_SVM_Model.pkl"
scaler_path = "scaler.pkl"
label_map_path = "label_map.json"

# Load model in the old environment
model = joblib.load("ISL_SVM_Model.pkl")

# Save it in the current Scikit-learn version
joblib.dump(model, "ISL_SVM_Model_v2.pkl")

# Function to download files if not present
def download_file(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists.")

# Ensure all required files are downloaded
download_file(model_url, model_path)
download_file(label_map_url, label_map_path)
download_file(scaler_url, scaler_path)

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load label map
with open(label_map_path, "r") as f:
    label_map = json.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        img = Image.open(file).convert("L").resize((64, 64))
        features = np.array(img).flatten().reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": label_map[str(prediction)]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")

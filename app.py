from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

model = joblib.load("ISL_SVM_Model.pkl")
scaler = joblib.load("scaler.pkl")

with open("label_map.json", "r") as f:
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
    app.run(port=5000)

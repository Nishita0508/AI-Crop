import os
import uuid
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_disease_model.h5")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "Train")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_class_names() -> list[str]:
    return sorted(
        [name for name in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, name))]
    )


def preprocess_image(image_path: str) -> np.ndarray:
    try:
        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMG_SIZE,
            color_mode="rgb",
        )
        image_array = tf.keras.utils.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array.astype("float32")
    except Exception as exc:
        raise ValueError(f"Unable to read image file: {exc}") from exc


def determine_status(disease_name: str) -> str:
    return "Healthy" if "Healthy" in disease_name else "Diseased"


def get_recommendation(disease_name: str) -> Dict[str, Any]:
    disease_key = disease_name.strip().lower()

    remedies = {
        "apple - apple scab": {
            "remedy": "Remove infected leaves promptly and apply a copper-based fungicide or sulfur spray.",
            "prevention": "Avoid overhead watering and improve air circulation around the trees."
        },
        "apple - black rot": {
            "remedy": "Prune and destroy affected fruit and branches, then apply a fungicide labeled for black rot.",
            "prevention": "Keep fruit dry, thin the canopy, and remove fallen fruit from the ground."
        },
        "apple - cedar apple rust": {
            "remedy": "Use fungicides and remove nearby juniper hosts to reduce reinfection.",
            "prevention": "Space trees well and avoid wet foliage during cool periods."
        },
        "bell pepper - bacterial spot": {
            "remedy": "Remove heavily infected leaves and apply copper bactericide treatments.",
            "prevention": "Water at the base of plants and avoid touching wet foliage."
        },
        "cherry - powdery mildew": {
            "remedy": "Apply sulfur or potassium bicarbonate fungicide and improve plant airflow.",
            "prevention": "Avoid dense planting and keep leaves dry."
        },
        "corn (maize) - cercospora leaf spot": {
            "remedy": "Use fungicides and rotate crops to reduce disease pressure.",
            "prevention": "Plant resistant hybrids and manage crop residues."
        },
        "corn (maize) - common rust": {
            "remedy": "Apply rust-targeted fungicide if disease pressure is high.",
            "prevention": "Use resistant varieties and monitor fields regularly."
        },
        "corn (maize) - northern leaf blight": {
            "remedy": "Treat with fungicide and remove infected residue after harvest.",
            "prevention": "Use resistant varieties and avoid excessive nitrogen."
        },
        "grape - black rot": {
            "remedy": "Prune infected clusters and spray a copper or mancozeb fungicide.",
            "prevention": "Remove fallen leaves and improve canopy ventilation."
        },
        "grape - esca (black measles)": {
            "remedy": "Remove affected vines and avoid using infected wood for propagation.",
            "prevention": "Prune carefully and keep vineyard sanitation high."
        },
        "grape - leaf blight": {
            "remedy": "Apply fungicide and remove severely infected foliage.",
            "prevention": "Avoid overhead irrigation and keep vines well spaced."
        },
        "peach - bacterial spot": {
            "remedy": "Use copper sprays and remove infected shoots early.",
            "prevention": "Reduce overhead irrigation and sanitize pruning tools."
        },
        "potato - early blight": {
            "remedy": "Use chlorothalonil or copper fungicides and remove infected leaves.",
            "prevention": "Mulch soil, avoid wet foliage, and rotate crops."
        },
        "potato - late blight": {
            "remedy": "Apply copper-based fungicide immediately and remove infected plant material.",
            "prevention": "Use resistant varieties and monitor weather conditions for blight risk."
        },
        "strawberry - leaf scorch": {
            "remedy": "Remove infected leaves and use fungicides appropriate for leaf scorch.",
            "prevention": "Improve airflow and minimize leaf wetness."
        },
        "tomato - bacterial spot": {
            "remedy": "Treat with copper bactericide and remove badly infected leaves.",
            "prevention": "Avoid splashing water and rotate tomato crops yearly."
        },
        "tomato - early blight": {
            "remedy": "Use fungicide and remove lower infected leaves promptly.",
            "prevention": "Mulch the soil and keep foliage dry."
        },
        "tomato - late blight": {
            "remedy": "Apply copper-based fungicide and remove infected foliage immediately.",
            "prevention": "Provide good spacing and avoid wet leaves overnight."
        },
        "tomato - septoria leaf spot": {
            "remedy": "Use fungicide and prune lower leaves to improve airflow.",
            "prevention": "Water at the base and remove fallen leaves."
        },
        "tomato - yellow leaf curl virus": {
            "remedy": "Remove infected plants and manage whiteflies to stop spread.",
            "prevention": "Use insect-proof netting and plant resistant varieties."
        },
        "tomato - healthy": {
            "remedy": "Keep the plant healthy with regular watering, balanced fertilization, and pest monitoring.",
            "prevention": "Inspect leaves weekly and maintain good airflow around the plant."
        }
    }

    for key, data in remedies.items():
        if key in disease_key:
            return data

    return {
        "remedy": "Inspect the plant closely and apply crop-specific treatment if symptoms persist.",
        "prevention": "Maintain clean tools, remove debris, and monitor the leaves regularly."
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Please upload a valid image file (JPG, JPEG, PNG, WEBP)."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        file.save(save_path)
        image_array = preprocess_image(save_path)
        prediction = model.predict(image_array, verbose=0)
        class_names = get_class_names()
        pred_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
        disease_name = class_names[pred_index]
        recommendation = get_recommendation(disease_name)

        result = {
            "disease": disease_name,
            "confidence": round(confidence, 2),
            "status": determine_status(disease_name),
            "recommended_action": recommendation["remedy"],
            "prevention": recommendation["prevention"],
            "image_url": f"/uploads/{unique_name}",
        }
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

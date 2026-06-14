import os
import uuid
from typing import Any, Dict

from flask import (
    Flask, jsonify, render_template, request,
    send_from_directory, session, redirect, url_for,
)
from werkzeug.utils import secure_filename

from routes.auth_routes      import auth_bp
from routes.dashboard_routes import dashboard_bp
from routes.crop_routes      import crop_bp
from routes.disease_routes   import disease_bp
from utils.model_loader      import predict_disease

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "agri_ai_secret_key_2026")
app.register_blueprint(auth_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(crop_bp)
app.register_blueprint(disease_bp)

UPLOAD_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _determine_status(disease_name: str) -> str:
    return "Healthy" if "healthy" in disease_name.lower() else "Diseased"


def _get_recommendation(disease_name: str) -> Dict[str, Any]:
    """Return remedy and prevention tip for the predicted disease.

    Matching is case-insensitive and tolerates both display-name format
    ('Tomato - Early Blight') and PlantVillage folder-name format
    ('Tomato___Early_blight') by normalising both sides before comparison.
    """
    def _key(s: str) -> str:
        return s.lower().replace("___", " - ").replace("_", " ").strip()

    disease_key = _key(disease_name)

    remedies: Dict[str, Dict[str, str]] = {
        "apple - apple scab": {
            "remedy":     "Remove infected leaves promptly and apply a copper-based fungicide or sulfur spray.",
            "prevention": "Avoid overhead watering and improve air circulation around the trees.",
        },
        "apple - black rot": {
            "remedy":     "Prune and destroy affected fruit and branches, then apply a fungicide labelled for black rot.",
            "prevention": "Keep fruit dry, thin the canopy, and remove fallen fruit from the ground.",
        },
        "apple - cedar apple rust": {
            "remedy":     "Use fungicides and remove nearby juniper hosts to reduce reinfection.",
            "prevention": "Space trees well and avoid wet foliage during cool periods.",
        },
        "bell pepper - bacterial spot": {
            "remedy":     "Remove heavily infected leaves and apply copper bactericide treatments.",
            "prevention": "Water at the base of plants and avoid touching wet foliage.",
        },
        "cherry - powdery mildew": {
            "remedy":     "Apply sulfur or potassium bicarbonate fungicide and improve plant airflow.",
            "prevention": "Avoid dense planting and keep leaves dry.",
        },
        "corn (maize) - cercospora leaf spot": {
            "remedy":     "Use fungicides and rotate crops to reduce disease pressure.",
            "prevention": "Plant resistant hybrids and manage crop residues.",
        },
        "corn (maize) - common rust": {
            "remedy":     "Apply rust-targeted fungicide if disease pressure is high.",
            "prevention": "Use resistant varieties and monitor fields regularly.",
        },
        "corn (maize) - northern leaf blight": {
            "remedy":     "Treat with fungicide and remove infected residue after harvest.",
            "prevention": "Use resistant varieties and avoid excessive nitrogen.",
        },
        "grape - black rot": {
            "remedy":     "Prune infected clusters and spray a copper or mancozeb fungicide.",
            "prevention": "Remove fallen leaves and improve canopy ventilation.",
        },
        "grape - esca (black measles)": {
            "remedy":     "Remove affected vines and avoid using infected wood for propagation.",
            "prevention": "Prune carefully and keep vineyard sanitation high.",
        },
        "grape - leaf blight": {
            "remedy":     "Apply fungicide and remove severely infected foliage.",
            "prevention": "Avoid overhead irrigation and keep vines well spaced.",
        },
        "peach - bacterial spot": {
            "remedy":     "Use copper sprays and remove infected shoots early.",
            "prevention": "Reduce overhead irrigation and sanitise pruning tools.",
        },
        "potato - early blight": {
            "remedy":     "Use chlorothalonil or copper fungicides and remove infected leaves.",
            "prevention": "Mulch soil, avoid wet foliage, and rotate crops.",
        },
        "potato - late blight": {
            "remedy":     "Apply copper-based fungicide immediately and remove infected plant material.",
            "prevention": "Use resistant varieties and monitor weather conditions for blight risk.",
        },
        "strawberry - leaf scorch": {
            "remedy":     "Remove infected leaves and use fungicides appropriate for leaf scorch.",
            "prevention": "Improve airflow and minimise leaf wetness.",
        },
        "tomato - bacterial spot": {
            "remedy":     "Treat with copper bactericide and remove badly infected leaves.",
            "prevention": "Avoid splashing water and rotate tomato crops yearly.",
        },
        "tomato - early blight": {
            "remedy":     "Use fungicide and remove lower infected leaves promptly.",
            "prevention": "Mulch the soil and keep foliage dry.",
        },
        "tomato - late blight": {
            "remedy":     "Apply copper-based fungicide and remove infected foliage immediately.",
            "prevention": "Provide good spacing and avoid wet leaves overnight.",
        },
        "tomato - septoria leaf spot": {
            "remedy":     "Use fungicide and prune lower leaves to improve airflow.",
            "prevention": "Water at the base and remove fallen leaves.",
        },
        "tomato - yellow leaf curl virus": {
            "remedy":     "Remove infected plants and manage whiteflies to stop spread.",
            "prevention": "Use insect-proof netting and plant resistant varieties.",
        },
        "tomato - healthy": {
            "remedy":     "Keep the plant healthy with regular watering, balanced fertilisation, and pest monitoring.",
            "prevention": "Inspect leaves weekly and maintain good airflow around the plant.",
        },
    }

    # 1. Try key-in-disease_key (display name substring match)
    for key, data in remedies.items():
        if key in disease_key:
            return data

    # 2. Try disease_key-in-key (reversed — handles longer folder names)
    for key, data in remedies.items():
        if disease_key in key:
            return data

    # 3. Word-overlap fallback (≥2 non-trivial words in common)
    _stopwords = {"the", "a", "an", "and", "or", "of", "in", "for"}
    disease_words = set(disease_key.split()) - _stopwords
    best_match, best_score = None, 0
    for key, data in remedies.items():
        key_words = set(key.split()) - _stopwords
        overlap   = len(disease_words & key_words)
        if overlap > best_score:
            best_score, best_match = overlap, data
    if best_score >= 2 and best_match:
        return best_match

    return {
        "remedy":     "Inspect the plant closely and apply crop-specific treatment if symptoms persist.",
        "prevention": "Maintain clean tools, remove debris, and monitor the leaves regularly.",
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard.dashboard"))
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """AJAX endpoint used by the frontend JavaScript."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({"error": "Please upload a valid image file (JPG, JPEG, PNG, WEBP)."}), 400

    filename    = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path   = os.path.join(UPLOAD_DIR, unique_name)

    try:
        file.save(save_path)
        disease_name, confidence = predict_disease(save_path)
        recommendation = _get_recommendation(disease_name)

        return jsonify({
            "disease":            disease_name,
            "confidence":         round(confidence, 2),
            "status":             _determine_status(disease_name),
            "recommended_action": recommendation["remedy"],
            "prevention":         recommendation["prevention"],
            "image_url":          f"/uploads/{unique_name}",
        })

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

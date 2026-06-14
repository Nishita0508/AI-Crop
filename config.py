"""
config.py
Centralized configuration for the Flask application.

Disease model selection order
------------------------------
1. Env var  DISEASE_MODEL_VARIANT  (e.g. "yolov8n", "yolov8s")
2. Fallback: models/disease_model_best.pt  (written by train_yolo.py)

Class names are loaded from  models/class_name_mapping.json  (written by
train_yolo.py) so the display names always stay in sync with whatever
dataset was used during training.  The hardcoded list below is a fallback
that covers a standard PlantVillage 29-class setup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR    = Path(__file__).resolve().parent
_MODELS_DIR  = _BASE_DIR / "models"
_MAPPING_FILE  = _MODELS_DIR / "class_name_mapping.json"
_CLASSES_FILE  = _MODELS_DIR / "disease_classes.json"

# ── Disease model path resolution ─────────────────────────────────────────────
_variant = os.environ.get("DISEASE_MODEL_VARIANT", "")
_variant_path = _MODELS_DIR / f"disease_model_{_variant}.pt" if _variant else None
_best_path    = _MODELS_DIR / "disease_model_best.pt"

# Pick the first path that actually exists on disk
_active_model: str
if _variant_path and _variant_path.exists():
    _active_model = str(_variant_path)
elif _best_path.exists():
    _active_model = str(_best_path)
else:
    # Model not yet trained — point at expected location so the error message
    # is helpful rather than a raw FileNotFoundError deep inside Keras/YOLO.
    _active_model = str(_best_path)


# ── Class name helpers ─────────────────────────────────────────────────────────

_FALLBACK_CLASSES: list[str] = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Bell Pepper - Bacterial Spot",
    "Bell Pepper - Healthy",
    "Cherry - Healthy",
    "Cherry - Powdery Mildew",
    "Corn (Maize) - Cercospora Leaf Spot",
    "Corn (Maize) - Common Rust",
    "Corn (Maize) - Healthy",
    "Corn (Maize) - Northern Leaf Blight",
    "Grape - Black Rot",
    "Grape - Esca (Black Measles)",
    "Grape - Healthy",
    "Grape - Leaf Blight",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Potato - Early Blight",
    "Potato - Healthy",
    "Potato - Late Blight",
    "Strawberry - Healthy",
    "Strawberry - Leaf Scorch",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Healthy",
    "Tomato - Late Blight",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Yellow Leaf Curl Virus",
]


def _load_display_classes() -> list[str]:
    """Return display names in class-index order.

    Prefers the mapping file written by train_yolo.py (values in index order).
    Falls back to the raw classes file, then to the hardcoded list.
    """
    # mapping file: {folder_name: display_name} — sorted keys = training order
    if _MAPPING_FILE.exists():
        with open(_MAPPING_FILE, encoding="utf-8") as fh:
            mapping: dict[str, str] = json.load(fh)
        # Keys are already sorted alphabetically (same order as YOLO training)
        return list(mapping.values())

    if _CLASSES_FILE.exists():
        with open(_CLASSES_FILE, encoding="utf-8") as fh:
            return json.load(fh)

    return _FALLBACK_CLASSES


# ── Config class ──────────────────────────────────────────────────────────────

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key-in-production")

    # ── Uploads ────────────────────────────────────────────────────────────
    UPLOAD_FOLDER      = str(_BASE_DIR / "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024   # 5 MB

    # ── Disease model (YOLOv8 .pt) ─────────────────────────────────────────
    DISEASE_MODEL_PATH = _active_model

    # ── Crop recommendation model (Scikit-learn pickle) ────────────────────
    CROP_MODEL_PATH = str(_MODELS_DIR / "crop_model.pkl")

    # ── Class labels ────────────────────────────────────────────────────────
    # Loaded dynamically so they match whatever was trained.
    DISEASE_CLASSES = _load_display_classes()

    # Image size expected by the YOLO model
    DISEASE_IMG_SIZE = (224, 224)

    # ── Session ─────────────────────────────────────────────────────────────
    SESSION_PERMANENT = False

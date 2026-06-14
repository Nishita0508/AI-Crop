"""
utils/model_loader.py
Singleton loaders and prediction helpers for:
  - Disease detection   → YOLOv8 classification model  (.pt)
  - Crop recommendation → Scikit-learn model            (.pkl)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config import Config

# ── Singleton holders ─────────────────────────────────────────────────────────
_disease_model: YOLO | None = None
_crop_model = None

# ── Class-name mapping  (folder_name → display_name) ─────────────────────────
# Populated once from models/class_name_mapping.json written by train_yolo.py.
# Falls back to an empty dict so _get_display_name() uses the normaliser below.
_mapping_file = Path(Config.DISEASE_MODEL_PATH).parent / "class_name_mapping.json"
_CLASS_MAPPING: dict[str, str] = {}
if _mapping_file.exists():
    with open(_mapping_file, encoding="utf-8") as _fh:
        _CLASS_MAPPING = json.load(_fh)


# ── Name normaliser (fallback when mapping file absent) ───────────────────────

def _normalize_class_name(folder: str) -> str:
    """Convert PlantVillage folder names to human-readable display names.

    Apple___Apple_scab         →  Apple - Apple Scab
    Tomato___Early_blight      →  Tomato - Early Blight
    Corn_(Maize)___Common_rust →  Corn (Maize) - Common Rust
    """
    parts = folder.split("___")
    if len(parts) == 2:
        plant   = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip().title()
        return f"{plant} - {disease}"
    return folder.replace("_", " ").strip().title()


def _get_display_name(raw: str) -> str:
    """Return the human-readable display name for a YOLO class label."""
    return _CLASS_MAPPING.get(raw, _normalize_class_name(raw))


# ── Loaders ───────────────────────────────────────────────────────────────────

def get_disease_model() -> YOLO:
    global _disease_model
    if _disease_model is None:
        path = Config.DISEASE_MODEL_PATH
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Disease model not found at: {path}\n"
                "Run  python train_yolo.py  to train and save the model first."
            )
        _disease_model = YOLO(path)
    return _disease_model


def get_crop_model():
    global _crop_model
    if _crop_model is None:
        path = Config.CROP_MODEL_PATH
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Crop model not found at: {path}\n"
                "Ensure  models/crop_model.pkl  is present."
            )
        with open(path, "rb") as fh:
            _crop_model = pickle.load(fh)
    return _crop_model


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_disease(image_path: str) -> tuple[str, float]:
    """Run disease classification on a single image.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    disease_name : str
        Human-readable display name of the predicted class.
    confidence : float
        Prediction confidence as a percentage (0.0 – 100.0).
    """
    model   = get_disease_model()
    results = model.predict(image_path, imgsz=224, verbose=False)

    probs      = results[0].probs          # ultralytics Probs object
    top1_idx   = int(probs.top1)           # index of highest-confidence class
    confidence = float(probs.top1conf) * 100
    raw_name   = results[0].names[top1_idx]  # class name stored in the model

    display_name = _get_display_name(raw_name)
    return display_name, round(confidence, 2)


def predict_crop(features: dict) -> str:
    """Predict the recommended crop from soil/weather features.

    Parameters
    ----------
    features : dict
        Keys: N, P, K, temperature, humidity, ph, rainfall

    Returns
    -------
    crop_name : str
    """
    model = get_crop_model()
    input_array = np.array([[
        features["N"],
        features["P"],
        features["K"],
        features["temperature"],
        features["humidity"],
        features["ph"],
        features["rainfall"],
    ]])
    prediction = model.predict(input_array)
    return str(prediction[0])

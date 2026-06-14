"""
predict.py
Standalone single-image prediction script for quick testing.
Runs the active disease model on one image and prints the result.

Usage:
    python predict.py path/to/leaf.jpg
    python predict.py path/to/leaf.jpg --model models/disease_model_yolov8s.pt
    python predict.py path/to/leaf.jpg --top 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO
from config import Config


def _normalize(folder: str) -> str:
    """Convert PlantVillage folder name to display name."""
    parts = folder.split("___")
    if len(parts) == 2:
        return f"{parts[0].replace('_', ' ')} - {parts[1].replace('_', ' ').title()}"
    return folder.replace("_", " ").title()


def predict_single(image_path: str, model_path: str, top_k: int = 5) -> None:
    img = Path(image_path)
    if not img.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    pt = Path(model_path)
    if not pt.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("  Run  python train_yolo.py  first.")
        sys.exit(1)

    print(f"\nModel : {pt.name}")
    print(f"Image : {img.name}")
    print("─" * 56)

    model   = YOLO(str(pt))
    results = model.predict(str(img), imgsz=224, verbose=False)
    probs   = results[0].probs
    names   = results[0].names

    # ── Top-K predictions ──────────────────────────────────────────────────
    k = min(top_k, len(names))
    top_vals, top_idxs = torch.topk(probs.data, k)

    print(f"  {'Rank':<6} {'Class':<38} {'Confidence':>10}")
    print("  " + "─" * 56)
    for rank, (idx, conf) in enumerate(zip(top_idxs.tolist(), top_vals.tolist()), 1):
        display = _normalize(names[idx])
        bar     = "█" * int(conf * 28) + "░" * (28 - int(conf * 28))
        print(f"  #{rank:<5} {display:<38} {conf * 100:>9.2f}%  {bar}")

    print("─" * 56)
    top1_name = _normalize(names[int(probs.top1)])
    top1_conf = float(probs.top1conf) * 100
    status    = "Healthy" if "healthy" in top1_name.lower() else "Diseased"
    print(f"  Prediction : {top1_name}")
    print(f"  Confidence : {top1_conf:.2f}%")
    print(f"  Status     : {status}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-image plant disease prediction")
    parser.add_argument("image",          type=str,
                        help="Path to the leaf image")
    parser.add_argument("--model", "-m",  type=str,
                        default=Config.DISEASE_MODEL_PATH,
                        help="Path to .pt model (default: active model from config)")
    parser.add_argument("--top",   "-k",  type=int, default=5,
                        help="Show top-K predictions (default: 5)")
    args = parser.parse_args()
    predict_single(args.image, args.model, args.top)


if __name__ == "__main__":
    main()

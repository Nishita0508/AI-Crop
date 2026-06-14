"""
check_model.py
Quick sanity-check for the active YOLOv8 disease model.
Prints architecture summary, number of classes, and class names.

Usage:
    python check_model.py
    python check_model.py --model models/disease_model_yolov8s.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

BASE_DIR    = Path(__file__).resolve().parent
MODELS_DIR  = BASE_DIR / "models"
DEFAULT_PT  = MODELS_DIR / "disease_model_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a YOLOv8 disease model")
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_PT),
        help=f"Path to .pt model file (default: {DEFAULT_PT})",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("  Run  python train_yolo.py  to train the models first.")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # ── Model info ─────────────────────────────────────────────────────────
    info = model.info(verbose=False)
    print(f"\n  Task          : {model.task}")
    print(f"  Model type    : {type(model.model).__name__}")

    names: dict = model.names
    print(f"  Num classes   : {len(names)}")
    print(f"\n  Class list:")
    for idx, name in sorted(names.items()):
        print(f"    [{idx:>3}]  {name}")

    # ── Mapping file ───────────────────────────────────────────────────────
    mapping_file = model_path.parent / "class_name_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, encoding="utf-8") as fh:
            mapping: dict = json.load(fh)
        print(f"\n  Display-name mapping loaded from: {mapping_file.name}")
        for folder, display in list(mapping.items())[:5]:
            print(f"    {folder!r:45s} → {display!r}")
        if len(mapping) > 5:
            print(f"    ... and {len(mapping) - 5} more")
    else:
        print(f"\n  No class_name_mapping.json found in {model_path.parent}")

    print("\n  Model OK.")


if __name__ == "__main__":
    main()

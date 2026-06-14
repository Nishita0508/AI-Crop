#!/usr/bin/env python3
"""
train_yolo.py
=============
Trains YOLOv8n-cls (Nano) and YOLOv8s-cls (Small) on the PlantVillage
leaf-disease dataset side-by-side.  After both finish:
  - Both models are validated and benchmarked for inference speed.
  - The winner (highest top-1 val accuracy) is saved as
      models/disease_model_best.pt   ← used by the Flask app
  - Each variant is also saved individually:
      models/disease_model_yolov8n.pt
      models/disease_model_yolov8s.pt
  - A JSON comparison report lands at
      runs/classify/comparison_report.json
  - Class names are saved at
      models/disease_classes.json          ← raw folder names
      models/class_name_mapping.json       ← folder name → display name

Usage
-----
    python train_yolo.py                        # auto-detect GPU, 50 epochs
    python train_yolo.py --epochs 30 --batch 32
    python train_yolo.py --device cpu           # force CPU
    python train_yolo.py --variant n            # train only YOLOv8n
    python train_yolo.py --variant s            # train only YOLOv8s
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR   = DATASET_DIR / "Train"
VAL_DIR     = DATASET_DIR / "Test"
OUTPUT_DIR  = BASE_DIR / "runs" / "classify"
MODELS_DIR  = BASE_DIR / "models"
YAML_PATH   = DATASET_DIR / "dataset.yaml"

# ── Model variants to train ───────────────────────────────────────────────────
ALL_VARIANTS: list[dict] = [
    {
        "key":     "yolov8n",
        "weights": "yolov8n-cls.pt",
        "label":   "YOLOv8n  (Nano   ~6 MB)",
        "run_name": "yolov8n_cls",
    },
    {
        "key":     "yolov8s",
        "weights": "yolov8s-cls.pt",
        "label":   "YOLOv8s  (Small ~22 MB)",
        "run_name": "yolov8s_cls",
    },
]

# How many inference passes to average
WARMUP_RUNS    = 10
BENCHMARK_RUNS = 100


# ── Utility functions ─────────────────────────────────────────────────────────

def detect_device(requested: str) -> str:
    """Resolve 'auto' to '0' (GPU) or 'cpu'."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU detected: {gpu_name}")
        return "0"
    print("  No GPU detected — using CPU (training will be slower).")
    return "cpu"


def safe_workers() -> int:
    """Windows multiprocessing with PyTorch DataLoader needs workers=0."""
    return 0 if platform.system() == "Windows" else 4


def discover_classes(train_dir: Path) -> list[str]:
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Make sure your dataset is at  dataset/Train/<ClassName>/images..."
        )
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class sub-directories found inside {train_dir}")
    return classes


def _normalize_folder_name(folder: str) -> str:
    """Convert PlantVillage-style folder names to human-readable display names.

    Examples
    --------
    Apple___Apple_scab          → Apple - Apple Scab
    Tomato___Early_blight       → Tomato - Early Blight
    Corn_(Maize)___Common_rust  → Corn (Maize) - Common Rust
    Tomato___healthy            → Tomato - Healthy
    """
    parts = folder.split("___")
    if len(parts) == 2:
        plant   = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip().title()
        return f"{plant} - {disease}"
    # No triple-underscore separator — just clean underscores
    return folder.replace("_", " ").strip().title()


def build_class_name_mapping(classes: list[str]) -> dict[str, str]:
    return {cls: _normalize_folder_name(cls) for cls in classes}


def write_dataset_yaml(classes: list[str]) -> Path:
    content: dict = {
        "path":  str(DATASET_DIR),
        "train": "Train",
        "val":   "Test",
        "nc":    len(classes),
        "names": classes,
    }
    with open(YAML_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(content, fh, default_flow_style=False, allow_unicode=True)
    print(f"  Dataset YAML written → {YAML_PATH}")
    return YAML_PATH


def find_sample_image(directory: Path) -> Path | None:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        hits = list(directory.rglob(ext))
        if hits:
            return hits[0]
    return None


# ── Core training logic ───────────────────────────────────────────────────────

def train_variant(
    variant:   dict,
    data_dir:  Path,
    epochs:    int,
    batch:     int,
    device:    str,
) -> dict:
    """Train one YOLO variant and return its metric dict."""
    sep = "═" * 68
    print(f"\n{sep}")
    print(f"  TRAINING  →  {variant['label']}")
    print(f"{sep}")

    model = YOLO(variant["weights"])

    t0 = time.perf_counter()
    model.train(
        data      = str(data_dir),   # classify needs a directory, not a YAML file
        epochs    = epochs,
        imgsz     = 224,
        batch     = batch,
        project   = str(OUTPUT_DIR),
        name      = variant["run_name"],
        patience  = 15,          # early stopping
        workers   = safe_workers(),
        optimizer = "AdamW",
        lr0       = 0.001,
        lrf       = 0.01,        # final LR = lr0 * lrf
        warmup_epochs = 3,
        cos_lr    = True,        # cosine LR schedule
        # ── augmentation ───────────────────────────────────────────
        augment   = True,
        hsv_h     = 0.015,       # hue jitter
        hsv_s     = 0.7,         # saturation jitter
        hsv_v     = 0.4,         # value jitter
        fliplr    = 0.5,         # horizontal flip prob
        degrees   = 15.0,        # rotation ± degrees
        translate = 0.1,         # translation fraction
        scale     = 0.5,         # scale gain
        mosaic    = 0.0,         # mosaic is detection-only — keep off
        # ───────────────────────────────────────────────────────────
        device    = device,
        verbose   = True,
        exist_ok  = True,        # overwrite same-named run (re-run safe)
    )
    train_time_min = round((time.perf_counter() - t0) / 60, 2)

    best_pt = OUTPUT_DIR / variant["run_name"] / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(
            f"Expected best.pt not found at {best_pt}. Training may have failed."
        )

    # ── Validate best weights ──────────────────────────────────────────────
    print(f"\n  Validating best weights for {variant['label']} ...")
    val_model   = YOLO(str(best_pt))
    val_metrics = val_model.val(
        data    = str(data_dir),
        imgsz   = 224,
        batch   = batch,
        device  = device,
        verbose = False,
    )

    # ClassifyMetrics attributes: top1, top5 (0-1 floats)
    top1 = float(getattr(val_metrics, "top1", 0.0)) * 100
    top5 = float(getattr(val_metrics, "top5", 0.0)) * 100

    return {
        "label":             variant["label"],
        "best_weights":      str(best_pt),
        "top1_accuracy_pct": round(top1, 2),
        "top5_accuracy_pct": round(top5, 2),
        "training_time_min": train_time_min,
        "model_size_mb":     round(best_pt.stat().st_size / (1024 ** 2), 2),
        "avg_inference_ms":  None,   # filled in by benchmark step
    }


def benchmark_inference(model_path: str, val_dir: Path, device: str) -> float:
    """Average inference latency in milliseconds over BENCHMARK_RUNS passes."""
    sample = find_sample_image(val_dir)
    if sample is None:
        print("  WARNING: No sample image found for benchmarking — skipping.")
        return -1.0

    model = YOLO(model_path)

    # Warmup
    for _ in range(WARMUP_RUNS):
        model.predict(str(sample), device=device, verbose=False)

    t0 = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        model.predict(str(sample), device=device, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) / BENCHMARK_RUNS * 1000
    return round(elapsed_ms, 2)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_comparison_table(results: dict[str, dict]) -> None:
    names  = list(results.keys())
    labels = [results[n]["label"] for n in names]
    col_w  = max(len(lb) for lb in labels) + 2

    metrics_rows = [
        ("Top-1 Val Accuracy (%)",  "top1_accuracy_pct"),
        ("Top-5 Val Accuracy (%)",  "top5_accuracy_pct"),
        ("Avg Inference (ms/img)",  "avg_inference_ms"),
        ("Model Size (MB)",         "model_size_mb"),
        ("Training Time (min)",     "training_time_min"),
    ]

    sep = "═" * (36 + col_w * 2)
    print(f"\n{sep}")
    print("  FINAL COMPARISON REPORT")
    print(sep)
    header = f"  {'Metric':<34}" + "".join(f"{lb:>{col_w}}" for lb in labels)
    print(header)
    print("  " + "─" * (34 + col_w * 2))
    for row_label, key in metrics_rows:
        row = f"  {row_label:<34}"
        for n in names:
            val = results[n].get(key)
            row += f"{str(val):>{col_w}}"
        print(row)
    print("  " + "─" * (34 + col_w * 2))

    winner = max(results, key=lambda k: results[k]["top1_accuracy_pct"])
    print(f"\n  Winner  →  {results[winner]['label']}")
    print(f"  Active model saved to:  models/disease_model_best.pt")
    print(f"{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8n-cls and YOLOv8s-cls for crop disease detection"
    )
    p.add_argument(
        "--epochs",  type=int,   default=50,
        help="Training epochs per model (default: 50)",
    )
    p.add_argument(
        "--batch",   type=int,   default=16,
        help="Batch size (default: 16; reduce to 8 if OOM on GPU)",
    )
    p.add_argument(
        "--device",  type=str,   default="auto",
        help="Device: 'auto' | '0' (GPU) | 'cpu'  (default: auto)",
    )
    p.add_argument(
        "--variant", type=str,   default="both",
        choices=["n", "s", "both"],
        help="Which variant to train: 'n'=nano, 's'=small, 'both' (default: both)",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = detect_device(args.device)

    print("\n" + "═" * 68)
    print("  AI-Crop  |  YOLOv8 Classification Training")
    print(f"  Device   : {'GPU (cuda:0)' if device == '0' else 'CPU'}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Variant  : {args.variant}")
    print("═" * 68)

    # ── 1. Discover dataset ────────────────────────────────────────────────
    print("\n[1/6] Discovering dataset ...")
    classes    = discover_classes(TRAIN_DIR)
    n_train    = sum(1 for _ in TRAIN_DIR.rglob("*") if _.is_file())
    n_val      = sum(1 for _ in VAL_DIR.rglob("*") if _.is_file())
    mapping    = build_class_name_mapping(classes)
    print(f"  Classes    : {len(classes)}")
    print(f"  Train imgs : {n_train}")
    print(f"  Val   imgs : {n_val}")

    # ── 2. Write dataset YAML ──────────────────────────────────────────────
    print("\n[2/6] Writing dataset.yaml ...")
    yaml_path = write_dataset_yaml(classes)

    # ── 3. Select variants to train ────────────────────────────────────────
    if args.variant == "n":
        variants = [ALL_VARIANTS[0]]
    elif args.variant == "s":
        variants = [ALL_VARIANTS[1]]
    else:
        variants = ALL_VARIANTS

    # ── 4. Train ───────────────────────────────────────────────────────────
    print(f"\n[3/6] Training {len(variants)} model(s) ...")
    results: dict[str, dict] = {}
    for variant in variants:
        metrics = train_variant(variant, DATASET_DIR, args.epochs, args.batch, device)
        results[variant["key"]] = metrics

    # ── 5. Benchmark inference speed ───────────────────────────────────────
    print("\n[4/6] Benchmarking inference speed ...")
    for key, data in results.items():
        print(f"  Benchmarking {data['label']} ...")
        ms = benchmark_inference(data["best_weights"], VAL_DIR, device)
        results[key]["avg_inference_ms"] = ms
        print(f"    → {ms} ms/image")

    # ── 6. Save models ────────────────────────────────────────────────────
    print("\n[5/6] Saving models ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for key, data in results.items():
        dest = MODELS_DIR / f"disease_model_{key}.pt"
        shutil.copy2(data["best_weights"], dest)
        results[key]["saved_to"] = str(dest)
        print(f"  Saved {data['label']:30s} → {dest.name}")

    # Winner = highest top-1 accuracy
    winner_key  = max(results, key=lambda k: results[k]["top1_accuracy_pct"])
    active_path = MODELS_DIR / "disease_model_best.pt"
    shutil.copy2(results[winner_key]["best_weights"], active_path)
    print(f"\n  Active model (winner) → {active_path}")

    # ── 7. Save class metadata ─────────────────────────────────────────────
    classes_json = MODELS_DIR / "disease_classes.json"
    with open(classes_json, "w", encoding="utf-8") as fh:
        json.dump(classes, fh, indent=2, ensure_ascii=False)

    mapping_json = MODELS_DIR / "class_name_mapping.json"
    with open(mapping_json, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)

    print(f"  Class names  → {classes_json.name}")
    print(f"  Name mapping → {mapping_json.name}")

    # ── 8. Write comparison report ─────────────────────────────────────────
    print("\n[6/6] Writing comparison report ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "winner":       winner_key,
        "winner_label": results[winner_key]["label"],
        "active_model": str(active_path),
        "classes_file": str(classes_json),
        "num_classes":  len(classes),
        "training_config": {
            "epochs":    args.epochs,
            "batch":     args.batch,
            "img_size":  224,
            "device":    device,
            "optimizer": "AdamW",
        },
        "results": results,
    }
    report_path = OUTPUT_DIR / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Report → {report_path}")

    # ── Summary table ──────────────────────────────────────────────────────
    print_comparison_table(results)

    print("  Run the Flask app and the winner model loads automatically.")
    print("  To force a specific variant, set env var:")
    print("      DISEASE_MODEL_VARIANT=yolov8n   or   DISEASE_MODEL_VARIANT=yolov8s\n")


if __name__ == "__main__":
    main()

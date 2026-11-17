"""
YOLO Model Fine-Tuning Module for KITTI Autonomous Driving Dataset.
Supports transfer learning, Apple Silicon MPS acceleration with CPU fallback,
configurable M4-friendly batch sizes, and automated checkpoint management.
"""

import os
import sys
import argparse
import shutil
import json
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO

from detect import get_optimal_device


def train_kitti_model(
    data_yaml: str = "dataset/data.yaml",
    base_weights: str = "models/yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 8,
    img_size: int = 640,
    device: Optional[str] = None,
    project_dir: str = "runs/detect",
    run_name: str = "kitti_yolo",
    save_best_to: str = "models/best.pt",
) -> Dict[str, Any]:
    """
    Fine-tune YOLO model on KITTI dataset via transfer learning.

    Args:
        data_yaml: Path to dataset data.yaml file.
        base_weights: Pretrained Ultralytics YOLO weights.
        epochs: Number of training epochs (default 50).
        batch_size: Training batch size tuned for M4 Unified Memory (default 8).
        img_size: Input resolution (default 640).
        device: 'mps' or 'cpu'. If None, auto-detected.
        project_dir: Directory where training runs are saved.
        run_name: Subfolder name for this run.
        save_best_to: Destination path for the best model checkpoint.

    Returns:
        Dictionary containing training summary and metric paths.
    """
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config not found at: {data_yaml}")

    if not os.path.exists(base_weights):
        # Fallback to standard weights if not in models/
        if os.path.exists("yolov8n.pt"):
            base_weights = "yolov8n.pt"
        else:
            raise FileNotFoundError(f"Base weights not found at: {base_weights}")

    selected_device = device if device is not None else get_optimal_device()
    print(f"=== Starting YOLO Training on KITTI ===")
    print(f"Base Weights: {base_weights}")
    print(f"Dataset:      {data_yaml}")
    print(f"Device:       {selected_device} (Apple Silicon M4 Optimized)")
    print(f"Epochs:       {epochs}")
    print(f"Batch Size:   {batch_size}")
    print(f"Image Size:   {img_size}x{img_size}")

    # Load pretrained model for transfer learning
    model = YOLO(base_weights)

    # Launch training
    train_results = model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=selected_device,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        pretrained=True,
        workers=2,
        verbose=True,
    )

    # Locate the best saved weights
    save_dir = getattr(model.trainer, "save_dir", os.path.join(project_dir, run_name))
    best_weights_path = os.path.join(save_dir, "weights", "best.pt")

    if os.path.exists(best_weights_path):
        os.makedirs(os.path.dirname(os.path.abspath(save_best_to)), exist_ok=True)
        shutil.copyfile(best_weights_path, save_best_to)
        print(f"Successfully exported best model to: {save_best_to}")
    else:
        # If best.pt not found, check last.pt
        last_weights_path = os.path.join(save_dir, "weights", "last.pt")
        if os.path.exists(last_weights_path):
            shutil.copyfile(last_weights_path, save_best_to)
            print(f"Exported last model checkpoint to: {save_best_to}")

    summary = {
        "device": selected_device,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "base_weights": base_weights,
        "save_dir": str(save_dir),
        "best_weights": os.path.abspath(save_best_to),
        "results_csv": os.path.join(str(save_dir), "results.csv"),
    }

    # Save summary json
    summary_path = os.path.join(str(save_dir), "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on KITTI Dataset")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="models/yolov8n.pt", help="Pretrained weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for M4 Unified Memory")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device (mps or cpu)")
    args = parser.parse_args()

    train_kitti_model(
        data_yaml=args.data,
        base_weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
    )

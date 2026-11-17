"""
Model Evaluation Module for Autonomous Driving Object Recognition.
Calculates authentic evaluation metrics on validation set:
Precision, Recall, F1-Score, mAP@0.50, and mAP@0.50:0.95.
Discovers and extracts actual training graphs and confusion matrices.
"""

import os
import glob
from typing import Dict, Any, Optional, List
import pandas as pd
from ultralytics import YOLO

from detect import get_optimal_device


class ModelEvaluator:
    """
    Evaluates fine-tuned or pretrained YOLO checkpoints on benchmark datasets.
    Extracts authentic validation metrics without fabrication.
    """

    def __init__(self, weights_path: str = "models/best.pt", data_yaml: str = "dataset/data.yaml"):
        if not os.path.exists(weights_path):
            # Fallback to yolov8n.pt if best.pt is not yet trained
            fallback = "models/yolov8n.pt"
            if os.path.exists(fallback):
                weights_path = fallback
            else:
                raise FileNotFoundError(f"Weights file not found at {weights_path}")

        self.weights_path = os.path.abspath(weights_path)
        self.data_yaml = os.path.abspath(data_yaml)
        self.device = get_optimal_device()

    def run_validation(self, batch_size: int = 8, img_size: int = 640) -> Dict[str, Any]:
        """
        Runs actual validation pass using Ultralytics validator.

        Returns:
            Dictionary with genuine precision, recall, F1, mAP50, mAP50-95,
            per-class breakdowns, and paths to validation curves.
        """
        if not os.path.exists(self.data_yaml):
            raise FileNotFoundError(f"data.yaml not found at {self.data_yaml}")

        model = YOLO(self.weights_path)

        # Run model validation on dataset
        metrics = model.val(
            data=self.data_yaml,
            batch=batch_size,
            imgsz=img_size,
            device=self.device,
            verbose=False,
        )

        box_metrics = metrics.box
        p = float(box_metrics.p.mean()) if hasattr(box_metrics.p, "mean") else float(box_metrics.p or 0.0)
        r = float(box_metrics.r.mean()) if hasattr(box_metrics.r, "mean") else float(box_metrics.r or 0.0)
        map50 = float(box_metrics.map50)
        map50_95 = float(box_metrics.map)

        # Calculate harmonic mean (F1-score)
        if (p + r) > 0:
            f1 = 2 * (p * r) / (p + r)
        else:
            f1 = 0.0

        # Per-class breakdown
        per_class: Dict[str, Dict[str, float]] = {}
        names = model.names
        for class_idx, class_name in names.items():
            if class_idx < len(box_metrics.maps):
                cls_p = float(box_metrics.p[class_idx]) if hasattr(box_metrics.p, "__getitem__") and len(box_metrics.p) > class_idx else p
                cls_r = float(box_metrics.r[class_idx]) if hasattr(box_metrics.r, "__getitem__") and len(box_metrics.r) > class_idx else r
                cls_f1 = (2 * cls_p * cls_r / (cls_p + cls_r)) if (cls_p + cls_r) > 0 else 0.0
                cls_map50 = float(box_metrics.maps[class_idx]) if hasattr(box_metrics, "maps") else 0.0

                per_class[class_name] = {
                    "precision": round(cls_p, 4),
                    "recall": round(cls_r, 4),
                    "f1_score": round(cls_f1, 4),
                    "map50": round(cls_map50, 4),
                }

        # Locate generated validation artifacts and graphs
        save_dir = str(getattr(metrics, "save_dir", ""))
        generated_plots = self.find_metric_plots(save_dir)

        return {
            "model_weights": self.weights_path,
            "device": self.device,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "per_class": per_class,
            "plots": generated_plots,
            "save_dir": save_dir,
        }

    @staticmethod
    def find_metric_plots(directory: str) -> Dict[str, str]:
        """Scans directory for standard YOLO metric plots."""
        plot_keys = {
            "results": ["results.png", "results.csv"],
            "confusion_matrix": ["confusion_matrix.png", "confusion_matrix_normalized.png"],
            "f1_curve": ["F1_curve.png", "BoxF1_curve.png"],
            "pr_curve": ["PR_curve.png", "BoxPR_curve.png"],
            "val_batch_labels": ["val_batch0_labels.jpg"],
            "val_batch_pred": ["val_batch0_pred.jpg"],
        }

        found_plots: Dict[str, str] = {}
        if not directory or not os.path.exists(directory):
            # Check runs/detect/ for any previous run plots
            candidate_dirs = glob.glob("runs/detect/*") + glob.glob("runs/detect/train*")
            if candidate_dirs:
                directory = sorted(candidate_dirs)[-1]
            else:
                return found_plots

        for key, filenames in plot_keys.items():
            for fname in filenames:
                fpath = os.path.join(directory, fname)
                if os.path.exists(fpath):
                    found_plots[key] = fpath
                    break

        return found_plots

    @staticmethod
    def read_results_csv(csv_path: str) -> Optional[pd.DataFrame]:
        """Safely reads training results.csv into a clean pandas DataFrame."""
        if not os.path.exists(csv_path):
            return None
        try:
            df = pd.read_csv(csv_path)
            # Strip column whitespace
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception:
            return None


if __name__ == "__main__":
    weights = "models/best.pt" if os.path.exists("models/best.pt") else "models/yolov8n.pt"
    print(f"Testing ModelEvaluator with {weights} on dataset/data.yaml...")
    evaluator = ModelEvaluator(weights_path=weights, data_yaml="dataset/data.yaml")
    results = evaluator.run_validation(batch_size=2)
    print("Validation Results:")
    print(f"  Precision:    {results['precision'] * 100:.2f}%")
    print(f"  Recall:       {results['recall'] * 100:.2f}%")
    print(f"  F1-Score:     {results['f1_score'] * 100:.2f}%")
    print(f"  mAP@0.50:     {results['map50'] * 100:.2f}%")
    print(f"  mAP@0.50:0.95:{results['map50_95'] * 100:.2f}%")
    print(f"Per-Class Metrics: {results['per_class']}")
    print(f"Found plots: {list(results['plots'].keys())}")

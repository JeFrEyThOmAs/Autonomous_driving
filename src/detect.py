"""
Object Detection Module using Ultralytics YOLO.
Optimized for Apple Silicon (MPS) with automatic CPU fallback.
Supports both official pretrained COCO weights and fine-tuned KITTI models.
"""

import os
import time
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def get_optimal_device() -> str:
    """
    Select the optimal execution device for Apple Silicon M4 / macOS.
    Prefers Apple Metal Performance Shaders (MPS), with graceful CPU fallback.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


class RoadObjectDetector:
    """
    Encapsulates YOLO inference, label mapping, and visualization for autonomous driving.
    """

    # Target autonomous driving classes
    TARGET_CLASSES = ["Car", "Pedestrian", "Cyclist"]

    # COCO class index mapping to road target classes when using standard weights
    COCO_ROAD_MAPPING = {
        0: "Pedestrian",  # person
        1: "Cyclist",     # bicycle
        2: "Car",         # car
        3: "Cyclist",     # motorcycle
        5: "Car",         # bus
        7: "Car",         # truck
    }

    # Consistent BGR color palette for visualization
    CLASS_COLORS = {
        "Car": (245, 130, 48),        # Amber / Orange
        "Pedestrian": (60, 20, 220),   # Crimson / Red
        "Cyclist": (250, 190, 40),     # Gold / Yellow
        "Other": (180, 180, 180),      # Grey
    }

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.35, iou_threshold: float = 0.45):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to YOLO weights (.pt). If None, defaults to models/yolov8n.pt.
            conf_threshold: Minimum confidence score to retain detection.
            iou_threshold: NMS IoU threshold.
        """
        self.device = get_optimal_device()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Determine weight file path
        if model_path is None or not os.path.exists(model_path):
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.pt"),
                os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"),
                "models/yolov8n.pt",
                "models/best.pt",
                "yolov8n.pt",
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError("No YOLO model checkpoint found. Run setup or place weights in models/.")

        self.model_path = os.path.abspath(model_path)
        self.model = YOLO(self.model_path)
        self.is_custom_kitti = self._check_if_kitti_model()

    def _check_if_kitti_model(self) -> bool:
        """
        Check if loaded model has KITTI classes (Car, Pedestrian, Cyclist)
        or default COCO 80 classes.
        """
        names = getattr(self.model, "names", {})
        if len(names) == 3:
            return True
        return False

    def map_class_name(self, raw_class_id: int, raw_name: str) -> Optional[str]:
        """
        Map model class names to standard target classes: Car, Pedestrian, Cyclist.
        Filters out non-road classes if running on standard COCO weights.
        """
        if self.is_custom_kitti:
            # Custom 3-class KITTI model: 0=Car, 1=Pedestrian, 2=Cyclist
            names = self.model.names
            return names.get(raw_class_id, raw_name)

        # Standard COCO weights mapping
        if raw_class_id in self.COCO_ROAD_MAPPING:
            return self.COCO_ROAD_MAPPING[raw_class_id]
        
        # Check by name string
        lower_name = raw_name.lower()
        if "person" in lower_name or "pedestrian" in lower_name:
            return "Pedestrian"
        if "car" in lower_name or "bus" in lower_name or "truck" in lower_name or "van" in lower_name:
            return "Car"
        if "cycle" in lower_name or "bike" in lower_name:
            return "Cyclist"

        return None

    def detect(self, image: np.ndarray, filter_target_classes: bool = True) -> Dict[str, Any]:
        """
        Perform detection on a single image.

        Args:
            image: Input image as numpy array in BGR format (standard OpenCV format).
            filter_target_classes: If True, only retains Car, Pedestrian, Cyclist.

        Returns:
            Dictionary containing:
                - detections: List of dicts (box, class_name, confidence, class_id)
                - annotated_image: Image with bounding boxes and labels drawn
                - counts: Dictionary with counts per class
                - total_count: Total road objects detected
                - inference_time_ms: Time taken for forward pass in ms
                - device: Device used ('mps' or 'cpu')
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is invalid or empty.")

        height, width = image.shape[:2]
        start_time = time.perf_counter()

        # Run inference using device configuration
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {cls_name: 0 for cls_name in self.TARGET_CLASSES}

        if results and len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                raw_name = self.model.names.get(cls_id, str(cls_id))

                mapped_name = self.map_class_name(cls_id, raw_name)

                if filter_target_classes and mapped_name not in self.TARGET_CLASSES:
                    continue

                display_name = mapped_name if mapped_name else raw_name
                if display_name not in counts:
                    counts[display_name] = 0
                counts[display_name] += 1

                # Normalize bounding box coordinates [x_center, y_center, w, h] in [0, 1]
                x1, y1, x2, y2 = xyxy
                x_center = ((x1 + x2) / 2.0) / width
                y_center = ((y1 + y2) / 2.0) / height
                bbox_w = (x2 - x1) / width
                bbox_h = (y2 - y1) / height

                detections.append({
                    "box": [round(coord, 1) for coord in xyxy],
                    "box_normalized": [round(x_center, 4), round(y_center, 4), round(bbox_w, 4), round(bbox_h, 4)],
                    "confidence": round(conf, 3),
                    "class_name": display_name,
                    "class_id": cls_id,
                })

        annotated_image = self.draw_annotations(image.copy(), detections)

        return {
            "detections": detections,
            "annotated_image": annotated_image,
            "counts": counts,
            "total_count": len(detections),
            "inference_time_ms": round(inference_time_ms, 2),
            "device": self.device,
            "image_dimensions": {"width": width, "height": height},
        }

    def draw_annotations(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw clean, high-visibility bounding boxes and labels for autonomous driving presentation.
        """
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = self.CLASS_COLORS.get(det["class_name"], (0, 255, 0))

            # Main bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label banner
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            banner_y1 = max(0, y1 - th - 8)
            banner_y2 = y1
            banner_x2 = min(image.shape[1], x1 + tw + 10)

            # Filled banner background
            cv2.rectangle(image, (x1, banner_y1), (banner_x2, banner_y2), color, -1)
            # Text on banner (white)
            cv2.putText(
                image,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return image


if __name__ == "__main__":
    import sys

    sample_img = sys.argv[1] if len(sys.argv) > 1 else "sample_data/road_scene.jpg"
    print(f"Testing RoadObjectDetector on {sample_img}...")

    detector = RoadObjectDetector()
    print(f"Initialized detector on device: {detector.device} (Model: {detector.model_path})")

    img = cv2.imread(sample_img)
    if img is None:
        print(f"Error: Could not read image at {sample_img}")
        sys.exit(1)

    result = detector.detect(img)
    print(f"Inference Time: {result['inference_time_ms']} ms")
    print(f"Total Objects Detected: {result['total_count']}")
    print(f"Counts: {result['counts']}")
    for i, d in enumerate(result["detections"]):
        print(f"  [{i+1}] {d['class_name']} ({d['confidence']*100:.1f}%) at {d['box']}")

    output_path = "sample_data/output_detected.jpg"
    cv2.imwrite(output_path, result["annotated_image"])
    print(f"Saved annotated image to {output_path}")

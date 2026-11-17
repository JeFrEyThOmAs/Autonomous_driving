"""
KITTI to YOLO Dataset Converter.
Extracts 2D bounding boxes from KITTI annotation text files, maps to target classes
(Car, Pedestrian, Cyclist), normalizes coordinates to YOLO format [x_center, y_center, w, h],
splits into 80:20 train/validation sets, and writes data.yaml configuration.
"""

import os
import glob
import shutil
import random
from typing import Dict, List, Tuple, Optional, Any
import cv2
import yaml


class KITTItoYOLOConverter:
    """
    Handles robust conversion of the KITTI Object Detection benchmark format into
    Ultralytics YOLO training format.
    """

    # Target classes for this autonomous driving project
    TARGET_CLASSES = {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2,
    }

    # Unsupported classes to deliberately ignore (not silently mapped)
    IGNORED_CLASSES = {"DontCare", "Misc", "Tram", "Truck", "Van", "Person_sitting"}

    def __init__(self, output_root: str = "dataset"):
        self.output_root = os.path.abspath(output_root)
        self.images_train_dir = os.path.join(self.output_root, "images", "train")
        self.images_val_dir = os.path.join(self.output_root, "images", "val")
        self.labels_train_dir = os.path.join(self.output_root, "labels", "train")
        self.labels_val_dir = os.path.join(self.output_root, "labels", "val")
        self.yaml_path = os.path.join(self.output_root, "data.yaml")

    def create_directory_structure(self):
        """Creates standard YOLO dataset folder hierarchy."""
        for path in [
            self.images_train_dir,
            self.images_val_dir,
            self.labels_train_dir,
            self.labels_val_dir,
        ]:
            os.makedirs(path, exist_ok=True)

    def parse_kitti_annotation(
        self,
        annotation_file: str,
        img_width: int,
        img_height: int,
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Parse single KITTI label file.
        Each row format: type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry

        Returns:
            (yolo_lines, class_counts)
        """
        yolo_lines: List[str] = []
        counts: Dict[str, int] = {k: 0 for k in self.TARGET_CLASSES.keys()}
        counts["ignored"] = 0

        if not os.path.exists(annotation_file):
            return yolo_lines, counts

        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or len(parts) < 8:
                    continue

                obj_type = parts[0]

                # Strict target class filtering
                if obj_type not in self.TARGET_CLASSES:
                    counts["ignored"] += 1
                    continue

                class_id = self.TARGET_CLASSES[obj_type]
                counts[obj_type] += 1

                # 2D Bounding Box in image pixels: left, top, right, bottom
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])

                # Clamp to image boundaries
                x1 = max(0.0, min(float(img_width), x1))
                x2 = max(0.0, min(float(img_width), x2))
                y1 = max(0.0, min(float(img_height), y1))
                y2 = max(0.0, min(float(img_height), y2))

                w = x2 - x1
                h = y2 - y1

                if w <= 1.0 or h <= 1.0:
                    continue

                # Normalize to [0, 1] for YOLO format
                x_center = (x1 + (w / 2.0)) / img_width
                y_center = (y1 + (h / 2.0)) / img_height
                norm_w = w / img_width
                norm_h = h / img_height

                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                )

        return yolo_lines, counts

    def convert_dataset(
        self,
        kitti_images_dir: str,
        kitti_labels_dir: str,
        val_split: float = 0.20,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Converts full KITTI dataset directory with an 80:20 train/validation split.

        Args:
            kitti_images_dir: Folder containing .png / .jpg images
            kitti_labels_dir: Folder containing .txt annotation files
            val_split: Fraction reserved for validation (default 0.20 for 80:20 split)
            random_seed: Seed for reproducible splitting
        """
        self.create_directory_structure()

        # Find all images
        image_extensions = ("*.png", "*.jpg", "*.jpeg")
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(kitti_images_dir, ext)))

        image_files.sort()
        if not image_files:
            raise FileNotFoundError(f"No image files found in {kitti_images_dir}")

        random.seed(random_seed)
        shuffled_images = image_files.copy()
        random.shuffle(shuffled_images)

        num_val = int(len(shuffled_images) * val_split)
        val_set = set(shuffled_images[:num_val])

        stats = {
            "total_images": len(image_files),
            "train_images": len(image_files) - num_val,
            "val_images": num_val,
            "class_counts": {k: 0 for k in self.TARGET_CLASSES.keys()},
            "ignored_objects": 0,
        }

        for img_path in shuffled_images:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(kitti_labels_dir, f"{basename}.txt")

            # Read image dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            yolo_lines, class_counts = self.parse_kitti_annotation(label_path, w, h)

            for cls_name, c in class_counts.items():
                if cls_name in stats["class_counts"]:
                    stats["class_counts"][cls_name] += c
            stats["ignored_objects"] += class_counts.get("ignored", 0)

            # Determine destination folder (train vs val)
            is_val = img_path in val_set
            dest_img_dir = self.images_val_dir if is_val else self.images_train_dir
            dest_lbl_dir = self.labels_val_dir if is_val else self.labels_train_dir

            # Copy image
            target_img_path = os.path.join(dest_img_dir, os.path.basename(img_path))
            shutil.copyfile(img_path, target_img_path)

            # Write YOLO label file
            target_lbl_path = os.path.join(dest_lbl_dir, f"{basename}.txt")
            with open(target_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")

        self.generate_yaml()
        return stats

    def generate_yaml(self) -> str:
        """Generates data.yaml for Ultralytics YOLO training."""
        data = {
            "path": self.output_root,
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.TARGET_CLASSES),
            "names": list(self.TARGET_CLASSES.keys()),
        }

        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return self.yaml_path


def create_sample_kitti_subset(sample_dir: str = "dataset/kitti_raw_sample"):
    """
    Creates a sample KITTI raw dataset subset for testing and academic demonstration.
    """
    img_dir = os.path.join(sample_dir, "images")
    lbl_dir = os.path.join(sample_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Use sample road scene to generate 10 synthetic KITTI training frames
    base_img = cv2.imread("sample_data/road_scene.jpg")
    if base_img is None:
        base_img = np.zeros((375, 1242, 3), dtype=np.uint8)

    h, w = base_img.shape[:2]

    # Sample KITTI annotation lines
    sample_annotations = [
        # Frame 0: 2 Cars, 1 Pedestrian, 1 DontCare
        [
            f"Car 0.00 0 -1.58 {int(w*0.1)} {int(h*0.4)} {int(w*0.35)} {int(h*0.8)} 1.5 1.6 3.8 4.2 1.5 14.8 -1.2",
            f"Car 0.00 0 -1.58 {int(w*0.6)} {int(h*0.45)} {int(w*0.85)} {int(h*0.85)} 1.5 1.6 3.8 4.2 1.5 14.8 -1.2",
            f"Pedestrian 0.00 0 -0.20 {int(w*0.42)} {int(h*0.45)} {int(w*0.52)} {int(h*0.85)} 1.8 0.5 1.2 1.8 1.4 8.4 0.0",
            f"DontCare -1 -1 -10 {int(w*0.0)} {int(h*0.1)} {int(w*0.2)} {int(h*0.3)} -1000 -1000 -1000 -1000 -1000 -1000 -10",
        ],
        # Frame 1: 1 Cyclist, 1 Car
        [
            f"Cyclist 0.00 0 -1.65 {int(w*0.3)} {int(h*0.48)} {int(w*0.42)} {int(h*0.82)} 1.7 0.6 1.8 3.7 1.4 12.8 -1.3",
            f"Car 0.00 0 -1.58 {int(w*0.55)} {int(h*0.42)} {int(w*0.78)} {int(h*0.8)} 1.5 1.6 3.8 4.2 1.5 14.8 -1.2",
            f"Van 0.00 0 -1.61 {int(w*0.8)} {int(h*0.4)} {int(w*0.95)} {int(h*0.75)} 2.0 1.9 4.8 5.0 1.5 18.0 -1.2", # Ignored
        ],
    ]

    for i in range(10):
        name = f"{i:06d}"
        # Slight variation for image
        img_copy = base_img.copy()
        cv2.putText(img_copy, f"KITTI FRAME {name}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(img_dir, f"{name}.png"), img_copy)

        # Write annotation
        annos = sample_annotations[i % len(sample_annotations)]
        with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(annos) + "\n")

    print(f"Created sample KITTI subset (10 frames) in {sample_dir}")


if __name__ == "__main__":
    print("Testing KITTItoYOLOConverter with sample dataset subset...")
    create_sample_kitti_subset()

    converter = KITTItoYOLOConverter(output_root="dataset")
    stats = converter.convert_dataset(
        kitti_images_dir="dataset/kitti_raw_sample/images",
        kitti_labels_dir="dataset/kitti_raw_sample/labels",
        val_split=0.20,
    )

    print("KITTI to YOLO Conversion Completed!")
    print(f"Stats: {stats}")
    print(f"Generated YAML configuration at {converter.yaml_path}")

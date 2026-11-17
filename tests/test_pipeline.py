"""
Automated Integration and Unit Test Suite for Autonomous Driving Perception System.
Verifies all modules against functional requirements.
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from detect import RoadObjectDetector, get_optimal_device
from decision_engine import DecisionEngine, DecisionAction
from video_processor import VideoProcessor
from dataset_converter import KITTItoYOLOConverter
from evaluation import ModelEvaluator


class TestAutonomousDrivingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_img_path = "sample_data/road_scene.jpg"
        cls.assertTrue(os.path.exists(cls.sample_img_path), "Sample road image missing")
        cls.image = cv2.imread(cls.sample_img_path)

    def test_01_device_selection(self):
        """Test device detection selects 'mps' or 'cpu' without CUDA crash."""
        dev = get_optimal_device()
        self.assertIn(dev, ["mps", "cpu"])
        print(f"Optimal device detected: {dev}")

    def test_02_image_detector(self):
        """Test RoadObjectDetector inference and target class output."""
        detector = RoadObjectDetector("models/yolov8n.pt")
        res = detector.detect(self.image)

        self.assertIn("detections", res)
        self.assertIn("counts", res)
        self.assertIn("annotated_image", res)
        self.assertIn("inference_time_ms", res)
        self.assertGreater(res["total_count"], 0)
        self.assertEqual(res["annotated_image"].shape, self.image.shape)

        # Verify bounding box coordinates
        for d in res["detections"]:
            x1, y1, x2, y2 = d["box"]
            self.assertLessEqual(x1, x2)
            self.assertLessEqual(y1, y2)
            self.assertIn(d["class_name"], ["Car", "Pedestrian", "Cyclist"])

    def test_03_fine_tuned_model_loading(self):
        """Test loading fine-tuned KITTI model checkpoint."""
        if os.path.exists("models/best.pt"):
            kitti_detector = RoadObjectDetector("models/best.pt")
            self.assertTrue(kitti_detector.is_custom_kitti)
            self.assertEqual(len(kitti_detector.model.names), 3)

    def test_04_decision_engine_rules(self):
        """Test DecisionEngine action logic and priority resolution."""
        engine = DecisionEngine(danger_zone_x=(0.3, 0.7), danger_zone_y=(0.4, 0.9))
        w, h = 640, 480

        # Scenario 1: Pedestrian in danger zone -> STOP
        ped_det = [{
            "box": [200, 200, 300, 400],
            "class_name": "Pedestrian",
            "confidence": 0.9,
        }]
        res1 = engine.evaluate(ped_det, w, h)
        self.assertEqual(res1["action"], DecisionAction.STOP)

        # Scenario 2: Car in danger zone -> SLOW DOWN
        car_det = [{
            "box": [200, 200, 300, 400],
            "class_name": "Car",
            "confidence": 0.9,
        }]
        res2 = engine.evaluate(car_det, w, h)
        self.assertEqual(res2["action"], DecisionAction.SLOW_DOWN)

        # Scenario 3: Cyclist in danger zone -> SLOW DOWN
        cyc_det = [{
            "box": [200, 200, 300, 400],
            "class_name": "Cyclist",
            "confidence": 0.9,
        }]
        res3 = engine.evaluate(cyc_det, w, h)
        self.assertEqual(res3["action"], DecisionAction.SLOW_DOWN)

        # Scenario 4: Pedestrian + Car together in danger zone -> STOP (Priority check)
        multi_det = [
            {"box": [200, 200, 300, 400], "class_name": "Car", "confidence": 0.9},
            {"box": [220, 210, 280, 380], "class_name": "Pedestrian", "confidence": 0.85},
        ]
        res4 = engine.evaluate(multi_det, w, h)
        self.assertEqual(res4["action"], DecisionAction.STOP)

        # Scenario 5: Object completely outside danger zone -> CONTINUE
        outside_det = [{
            "box": [10, 10, 50, 50],
            "class_name": "Pedestrian",
            "confidence": 0.9,
        }]
        res5 = engine.evaluate(outside_det, w, h)
        self.assertEqual(res5["action"], DecisionAction.CONTINUE)

    def test_05_video_processor(self):
        """Test VideoProcessor frame execution and output shape."""
        vp = VideoProcessor()
        annotated_frame, det_res, dec_res = vp.process_frame(self.image)
        self.assertEqual(annotated_frame.shape, self.image.shape)
        self.assertIn(dec_res["action"], [DecisionAction.STOP, DecisionAction.SLOW_DOWN, DecisionAction.CONTINUE])

    def test_06_kitti_converter(self):
        """Test KITTI annotation parsing and coordinate normalization."""
        converter = KITTItoYOLOConverter(output_root="dataset")
        line = "Car 0.00 0 -1.58 100.0 50.0 300.0 250.0 1.5 1.6 3.8 4.2 1.5 14.8 -1.2"
        temp_txt = "sample_data/temp_kitti.txt"
        with open(temp_txt, "w") as f:
            f.write(line + "\n")

        yolo_lines, counts = converter.parse_kitti_annotation(temp_txt, 1000, 500)
        os.remove(temp_txt)

        self.assertEqual(len(yolo_lines), 1)
        cls_id, xc, yc, nw, nh = [float(v) for v in yolo_lines[0].split()]
        self.assertEqual(int(cls_id), 0) # Car
        self.assertAlmostEqual(xc, 0.2, places=3)
        self.assertAlmostEqual(yc, 0.3, places=3)
        self.assertAlmostEqual(nw, 0.2, places=3)
        self.assertAlmostEqual(nh, 0.4, places=3)

    def test_07_model_evaluator(self):
        """Test ModelEvaluator metric plots discovery."""
        plots = ModelEvaluator.find_metric_plots("runs/detect/runs/detect/kitti_yolo")
        self.assertIsInstance(plots, dict)
        self.assertTrue(len(plots) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

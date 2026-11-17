"""
Decision Simulation Engine for Autonomous Driving.
Interprets YOLO perception outputs to generate simulated driving actions:
STOP, SLOW DOWN, or CONTINUE.

NOTE: This is an academic image-based 2D decision simulation.
Bounding box geometries represent 2D image coordinates and do NOT
represent calibrated physical radar/lidar distance measurements in metres.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class DecisionAction:
    STOP = "STOP"
    SLOW_DOWN = "SLOW DOWN"
    CONTINUE = "CONTINUE"


class DecisionEngine:
    """
    Decoupled decision simulation engine that analyzes spatial location of
    detected objects relative to a configurable central Danger Zone.
    """

    # Visual color mapping (BGR)
    COLOR_STOP = (0, 0, 220)       # Vivid Red
    COLOR_SLOW = (0, 140, 255)     # Amber / Orange
    COLOR_CONTINUE = (0, 180, 0)   # Green

    def __init__(
        self,
        danger_zone_x: Tuple[float, float] = (0.25, 0.75),
        danger_zone_y: Tuple[float, float] = (0.45, 0.95),
        min_confidence: float = 0.35,
    ):
        """
        Initialize the Decision Engine.

        Args:
            danger_zone_x: (min_ratio, max_ratio) normalized horizontal bounds of danger zone [0.0 - 1.0].
            danger_zone_y: (min_ratio, max_ratio) normalized vertical bounds of danger zone [0.0 - 1.0].
            min_confidence: Minimum detection confidence required to trigger a driving decision.
        """
        self.danger_zone_x = danger_zone_x
        self.danger_zone_y = danger_zone_y
        self.min_confidence = min_confidence

    def set_danger_zone(self, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]):
        """Update danger zone boundaries."""
        self.danger_zone_x = x_bounds
        self.danger_zone_y = y_bounds

    def compute_danger_zone_pixels(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized danger zone ratios to pixel coordinates (x1, y1, x2, y2)."""
        x1 = int(self.danger_zone_x[0] * img_width)
        x2 = int(self.danger_zone_x[1] * img_width)
        y1 = int(self.danger_zone_y[0] * img_height)
        y2 = int(self.danger_zone_y[1] * img_height)
        return x1, y1, x2, y2

    def check_overlap(
        self,
        bbox: List[float],
        dz_pixel: Tuple[int, int, int, int],
        overlap_threshold: float = 0.15,
    ) -> bool:
        """
        Check if a bounding box overlaps with the danger zone.
        Uses intersection over bounding box area or bottom-center proximity.
        """
        bx1, by1, bx2, by2 = bbox
        dx1, dy1, dx2, dy2 = dz_pixel

        # Compute intersection rectangle
        ix1 = max(bx1, dx1)
        iy1 = max(by1, dy1)
        ix2 = min(bx2, dx2)
        iy2 = min(by2, dy2)

        if ix2 <= ix1 or iy2 <= iy1:
            return False

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        box_area = max(1.0, (bx2 - bx1) * (by2 - by1))

        # Check intersection ratio
        overlap_ratio = intersection_area / box_area

        # Also check if object bottom-center (contact point on road) is inside danger zone
        bottom_cx = (bx1 + bx2) / 2.0
        bottom_cy = by2
        bottom_inside = (dx1 <= bottom_cx <= dx2) and (dy1 <= bottom_cy <= dy2)

        return (overlap_ratio >= overlap_threshold) or bottom_inside

    def evaluate(self, detections: List[Dict[str, Any]], image_width: int, image_height: int) -> Dict[str, Any]:
        """
        Evaluate detections against the danger zone to determine simulated driving action.

        Rules:
        - If a Pedestrian is inside the Danger Zone -> STOP (Highest Priority)
        - If a Vehicle (Car) or Cyclist is inside the Danger Zone -> SLOW DOWN
        - If no critical objects are inside the Danger Zone -> CONTINUE

        Returns:
            Dictionary with:
                - action: "STOP" | "SLOW DOWN" | "CONTINUE"
                - reason: Descriptive explanation of the decision
                - danger_zone_pixels: (x1, y1, x2, y2)
                - triggered_objects: List of objects inside the danger zone
                - color_bgr: Color tuple for overlay rendering
        """
        dz_pixels = self.compute_danger_zone_pixels(image_width, image_height)
        triggered_objects = []

        has_pedestrian_in_zone = False
        has_vehicle_in_zone = False
        has_cyclist_in_zone = False

        for det in detections:
            conf = det.get("confidence", 0.0)
            if conf < self.min_confidence:
                continue

            bbox = det.get("box", [0, 0, 0, 0])
            cls_name = det.get("class_name", "")

            if self.check_overlap(bbox, dz_pixels):
                triggered_objects.append(det)
                if cls_name == "Pedestrian":
                    has_pedestrian_in_zone = True
                elif cls_name in ["Car", "Vehicle", "Bus", "Truck"]:
                    has_vehicle_in_zone = True
                elif cls_name in ["Cyclist", "Bicycle", "Motorcycle"]:
                    has_cyclist_in_zone = True

        # Rule Decision Logic: Prioritize STOP over SLOW DOWN
        if has_pedestrian_in_zone:
            action = DecisionAction.STOP
            color = self.COLOR_STOP
            reason = f"Pedestrian detected inside Danger Zone (Immediate Hazard)"
        elif has_cyclist_in_zone:
            action = DecisionAction.SLOW_DOWN
            color = self.COLOR_SLOW
            reason = f"Cyclist detected inside Danger Zone (Caution Advised)"
        elif has_vehicle_in_zone:
            action = DecisionAction.SLOW_DOWN
            color = self.COLOR_SLOW
            reason = f"Vehicle detected inside Danger Zone (Maintain Distance)"
        else:
            action = DecisionAction.CONTINUE
            color = self.COLOR_CONTINUE
            reason = "Path clear: No obstacles detected within Danger Zone"

        return {
            "action": action,
            "reason": reason,
            "danger_zone_pixels": dz_pixels,
            "triggered_objects": triggered_objects,
            "color_bgr": color,
            "object_in_zone_count": len(triggered_objects),
        }

    def overlay_decision(
        self,
        image: np.ndarray,
        decision_result: Dict[str, Any],
        show_danger_zone: bool = True,
    ) -> np.ndarray:
        """
        Draw a visual HUD overlay onto the image including:
        - Danger Zone bounding corridor (with dashed or semi-transparent shading)
        - Decision action banner with high-contrast color coding
        - Explanatory reason and academic simulation notice
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        action = decision_result["action"]
        reason = decision_result["reason"]
        color = decision_result["color_bgr"]
        dz = decision_result["danger_zone_pixels"]

        # 1. Draw Danger Zone Box & Semi-transparent tint
        if show_danger_zone:
            overlay = annotated.copy()
            x1, y1, x2, y2 = dz
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            # Alpha blend for subtle corridor tint
            cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)
            # Solid danger zone border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(
                annotated,
                "CENTRAL DANGER ZONE (SIMULATED)",
                (x1 + 6, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        # 2. Top HUD Decision Banner
        banner_height = 54
        banner_bg = np.zeros((banner_height, w, 3), dtype=np.uint8)
        # Tint banner background slightly with action color
        cv2.rectangle(banner_bg, (0, 0), (w, banner_height), (30, 30, 30), -1)
        # Left status badge
        badge_width = 170
        cv2.rectangle(banner_bg, (0, 0), (badge_width, banner_height), color, -1)

        # Draw Action text inside badge
        font_scale = 0.85
        (tw, th), _ = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_x = max(10, (badge_width - tw) // 2)
        text_y = (banner_height + th) // 2 - 2
        cv2.putText(
            banner_bg,
            action,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Reason & Notice text next to badge
        cv2.putText(
            banner_bg,
            f"DECISION: {reason}",
            (badge_width + 15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            banner_bg,
            "[Simulated Driving Recommendation | 2D Bounding-Box Analysis]",
            (badge_width + 15, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

        # Blend banner at the top of the annotated image
        annotated[0:banner_height, 0:w] = banner_bg

        return annotated


if __name__ == "__main__":
    from detect import RoadObjectDetector

    print("Testing DecisionEngine integration...")
    detector = RoadObjectDetector()
    engine = DecisionEngine()

    img = cv2.imread("sample_data/road_scene.jpg")
    det_res = detector.detect(img)
    dec_res = engine.evaluate(det_res["detections"], img.shape[1], img.shape[0])

    print(f"Decision: {dec_res['action']}")
    print(f"Reason: {dec_res['reason']}")
    print(f"Objects inside danger zone: {dec_res['object_in_zone_count']}")

    hud_img = engine.overlay_decision(det_res["annotated_image"], dec_res)
    cv2.imwrite("sample_data/output_decision.jpg", hud_img)
    print("Saved decision visualization to sample_data/output_decision.jpg")

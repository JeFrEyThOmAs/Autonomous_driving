"""
Video and Webcam Processing Module for Autonomous Driving Perception.
Provides memory-efficient frame-by-frame processing, live video streaming,
file rendering, and graceful macOS camera handling.
"""

import os
import time
from typing import Generator, Dict, Any, Optional, Tuple, Callable
import cv2
import numpy as np

from detect import RoadObjectDetector
from decision_engine import DecisionEngine


class VideoProcessor:
    """
    Manages offline video detection and online webcam processing.
    Ensures memory efficiency by using generators rather than caching entire video streams.
    """

    def __init__(self, detector: Optional[RoadObjectDetector] = None, decision_engine: Optional[DecisionEngine] = None):
        self.detector = detector if detector is not None else RoadObjectDetector()
        self.decision_engine = decision_engine if decision_engine is not None else DecisionEngine()

    def process_frame(self, frame: np.ndarray, include_decision_overlay: bool = True) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """
        Process a single image/video frame through detection and decision simulation.

        Returns:
            (annotated_frame, detection_result, decision_result)
        """
        h, w = frame.shape[:2]
        det_result = self.detector.detect(frame)
        dec_result = self.decision_engine.evaluate(det_result["detections"], w, h)

        if include_decision_overlay:
            annotated_frame = self.decision_engine.overlay_decision(det_result["annotated_image"], dec_result)
        else:
            annotated_frame = det_result["annotated_image"]

        return annotated_frame, det_result, dec_result

    def process_video_stream(
        self,
        video_path: str,
        max_fps: Optional[float] = None,
        frame_skip: int = 1,
        stop_callback: Optional[Callable[[], bool]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Memory-efficient generator that yields processed frames and stats one by one.
        Allows real-time display in Streamlit without exhausting Unified Memory on M4.

        Yields:
            Dict containing:
                - frame: RGB or BGR numpy array
                - frame_idx: Current frame index
                - total_frames: Total video frames
                - det_result: Detection dictionary
                - dec_result: Decision dictionary
                - fps: Instantaneous processing FPS
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frame_idx = 0
        prev_time = time.perf_counter()

        try:
            while cap.isOpened():
                if stop_callback and stop_callback():
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                curr_time = time.perf_counter()
                fps = 1.0 / max(1e-4, (curr_time - prev_time))
                prev_time = curr_time

                annotated_frame, det_result, dec_result = self.process_frame(frame)

                yield {
                    "frame": annotated_frame,
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "det_result": det_result,
                    "dec_result": dec_result,
                    "fps": round(fps, 1),
                }

                # Optional rate limiting to avoid maxing out CPU/GPU if requested
                if max_fps and max_fps > 0:
                    sleep_time = (1.0 / max_fps) - (time.perf_counter() - curr_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        finally:
            cap.release()

    def process_video_to_file(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process an input video and render an annotated output video file.
        Uses standard mp4v codec for broad compatibility.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_inference_time = 0.0
        action_counts = {"STOP": 0, "SLOW DOWN": 0, "CONTINUE": 0}

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, det_result, dec_result = self.process_frame(frame)
                writer.write(annotated_frame)

                action_counts[dec_result["action"]] = action_counts.get(dec_result["action"], 0) + 1
                total_inference_time += det_result["inference_time_ms"]
                frame_count += 1

                if progress_callback and total_frames > 0:
                    progress_callback(frame_count / total_frames, frame_count, total_frames)

        finally:
            cap.release()
            writer.release()

        avg_inference_ms = total_inference_time / max(1, frame_count)

        return {
            "output_path": output_path,
            "total_frames_processed": frame_count,
            "avg_inference_time_ms": round(avg_inference_ms, 2),
            "action_counts": action_counts,
        }

    @staticmethod
    def get_webcam_capture(camera_index: int = 0) -> Tuple[Optional[cv2.VideoCapture], str]:
        """
        Safely open macOS webcam device using AVFoundation backend.
        Returns (capture_object, status_message).
        """
        # On macOS, cv2.CAP_AVFOUNDATION is the standard native camera framework
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            # Try default backend as fallback
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            error_msg = (
                f"Unable to access camera (Index {camera_index}). "
                "On macOS, please ensure Camera access is granted to VS Code or Terminal in: "
                "System Settings -> Privacy & Security -> Camera."
            )
            return None, error_msg

        return cap, "Camera initialized successfully."


if __name__ == "__main__":
    print("Testing VideoProcessor on synthetic video sample_data/sample_drive.mp4...")
    vp = VideoProcessor()
    
    stats = vp.process_video_to_file(
        input_path="sample_data/sample_drive.mp4",
        output_path="sample_data/output_drive_processed.mp4",
    )
    print("Video Processing Completed Successfully!")
    print(f"Stats: {stats}")

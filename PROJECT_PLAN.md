# Project Plan: Autonomous Driving and Object Recognition Using YOLO-Based Deep Learning

**Platform:** Apple MacBook Air M4 (macOS arm64)  
**Environment:** Python 3.13 / Virtual Environment (`.venv`)  
**Core Frameworks:** Ultralytics YOLO (v8/v11), PyTorch (MPS / CPU fallback), OpenCV, Streamlit, NumPy

---

## 1. Project Overview & Architecture

This academic project implements a complete perception-and-decision simulation pipeline for autonomous driving scenarios:

```
Input Image / Video / Webcam Frame
                |
                v
       Preprocessing (Resize, Normalize)
                |
                v
       YOLO Backbone (Feature Extraction)
                |
                v
       YOLO Neck (Feature Fusion / PANet / FPN)
                |
                v
       YOLO Detection Head (Anchor-free BBoxes, Classes, Confidences)
                |
                v
[ Detections: Cars, Pedestrians, Cyclists + BBoxes + Confidence ]
                |
                v
   Decision Simulation Engine (Central Danger Zone Analysis)
                |
                v
 Simulated Recommendation: [ STOP | SLOW DOWN | CONTINUE ]
```

### Architectural Principles:
1. **Separation of Concerns:** Object detection (`src/detect.py`) and decision intelligence (`src/decision_engine.py`) are strictly decoupled.
2. **Apple Silicon Acceleration:** PyTorch uses Apple's Metal Performance Shaders (`mps`) when available, with automatic CPU fallback.
3. **Academic Scope:** Explicitly clarifies that bounding box geometries in 2D image coordinates are simulated approximations and do not represent physical lidar/radar metric distances.

---

## 2. Phase-by-Phase Roadmap & Tracking

| Phase | Description | Status | Verification Target |
|---|---|---|---|
| **Phase 1** | Project setup, folder structure, requirements, Apple Silicon environment | 🟢 COMPLETED | Environment verified; dependencies and weights loaded |
| **Phase 2** | Pretrained YOLO image detection module (`detect.py`) | 🟢 COMPLETED | Runs inference on road scene, outputs detected bboxes and stats (4 detections, 876ms on CPU) |
| **Phase 3** | Video & Webcam detection modules (`video_processor.py`) | 🟢 COMPLETED | Frame-by-frame memory-efficient streaming verified on sample_drive.mp4 (90 frames at ~35 FPS) |
| **Phase 4** | Decision Simulation Engine (`decision_engine.py`) | 🟢 COMPLETED | Danger zone spatial analysis, STOP / SLOW DOWN / CONTINUE logic verified with HUD overlay |
| **Phase 5** | Streamlit Dashboard (`app.py`) | 🟢 COMPLETED | Full modern dark-themed interactive multi-page dashboard with 7 distinct pages |
| **Phase 6** | KITTI Dataset Integration (`dataset_converter.py`) | 🟢 COMPLETED | Converts KITTI annotations to YOLO format (80/20 train/val split) with data.yaml |
| **Phase 7** | YOLO Model Fine-Tuning & Evaluation (`train.py`, `evaluation.py`) | 🟢 COMPLETED | Transfer learning script with M4 batch tuning + genuine evaluation metrics and plots |
| **Phase 8** | End-to-End System Integration & Testing (`tests/test_pipeline.py`) | 🟢 COMPLETED | Full automated integration test suite (7 tests) passing with 100% success |

---

## 3. Architectural & Implementation Decisions

- **Device Selection Logic:** Automatically checks `torch.backends.mps.is_available()`. If `True`, utilizes device `'mps'`; otherwise falls back seamlessly to `'cpu'`. Never makes CUDA-specific assumptions.
- **MacBook Air M4 Memory Optimization:** Default image size set to 640x640, inference batch size = 1, training batch size = 8 or 16 depending on available Unified Memory.
- **Video Memory Management:** Generator-based frame processing in `video_processor.py` avoids reading whole video files into RAM at once; processed video uses standard `mp4v` / H.264 codecs.
- **Decision Engine Rules:**
  - Evaluates spatial overlap between object bounding boxes and the configurable central *Danger Zone*.
  - Priority: `STOP` (Pedestrian in danger zone) > `SLOW DOWN` (Vehicle or Cyclist in danger zone) > `CONTINUE` (Clear or outside danger zone).
  - Explicit UI disclaimer: Image-based 2D simulation for academic evaluation.
- **KITTI Class Filtering:**
  - Strict mapping: `Car`, `Van` -> `Car` (class 0); `Pedestrian`, `Person_sitting` -> `Pedestrian` (class 1); `Cyclist` -> `Cyclist` (class 2).
  - Unrelated or noisy classes (`DontCare`, `Misc`, `Tram`, `Truck` if outside car definition) are filtered out deliberately without silent fallback.

---

## 4. Hardware Resource & Storage Monitoring

- **Available Disk Space:** ~15 GB free on root APFS volume.
- **KITTI Dataset Note:** Full KITTI 2D detection dataset is ~12 GB compressed (~29 GB uncompressed). A sample KITTI subset structure and conversion pipeline are provided so that conversion and fine-tuning pipelines can be tested immediately without risking disk exhaustion.

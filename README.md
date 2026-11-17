# Autonomous Driving and Object Recognition Using YOLO-Based Deep Learning

An end-to-end computer vision and perception decision-support framework designed and optimized for **Apple Silicon (MacBook Air M4 / macOS)** using **Ultralytics YOLO**, **PyTorch**, **OpenCV**, and **Streamlit**.

---

## 1. System Architecture

The application adopts a decoupled architecture separating perception from tactical driving simulation:

```
[ Input: Road Image / Video File / Live Webcam ]
                       |
                       v
       Preprocessing & Normalization (640x640)
                       |
                       v
       YOLO Backbone (CSPDarknet Feature Extraction)
                       |
                       v
       YOLO Neck (PANet / FPN Multi-Scale Feature Fusion)
                       |
                       v
       YOLO Detection Head (Anchor-Free Bounding Box Regression)
                       |
                       v
[ Bounding Boxes + Classes (Car, Pedestrian, Cyclist) + Confidences ]
                       |
                       v
           Decision Simulation Engine
         (Spatial Danger Zone Analysis)
                       |
                       v
       Simulated Tactical Driving Action:
         🔴 STOP | 🟠 SLOW DOWN | 🟢 CONTINUE
```

> **Academic Notice:** This is an image-based 2D simulation for academic evaluation. Proximity is evaluated using normalized 2D image coordinates and does not represent physical metric distances from calibrated LiDAR or RADAR sensors.

---

## 2. Key Features

- **Apple Silicon M4 Native Acceleration:** Seamlessly utilizes Apple's Metal Performance Shaders (`mps`) when available, with automatic CPU fallback.
- **Three Target Autonomous Classes:** Explicit classification of **Cars**, **Pedestrians**, and **Cyclists**.
- **Decoupled Decision Simulation Engine:** Configurable central danger corridor with safety priority rules:
  - **STOP (Red):** Triggered when a pedestrian enters the danger zone corridor.
  - **SLOW DOWN (Orange):** Triggered when a vehicle or cyclist is in the corridor.
  - **CONTINUE (Green):** Triggered when the corridor is clear.
- **KITTI Benchmark Converter:** Fully converts KITTI 2D detection annotations into normalized YOLO format with an 80:20 train/val split. Unsupported classes (`DontCare`, `Van`, `Misc`) are filtered strictly without silent corruption.
- **Transfer Learning Pipeline:** Fine-tunes YOLO on KITTI with M4-friendly batch sizes and saves the best model checkpoint (`models/best.pt`).
- **Authentic Model Evaluation:** Computes and displays true Precision, Recall, F1-Score, mAP@0.50, and mAP@0.50:0.95 along with training curves and confusion matrices.
- **Modern Dark-Themed Streamlit Dashboard:** Multi-page dashboard with interactive widgets for images, videos, live webcam, decision simulation, dataset tools, and metrics.

---

## 3. Project Structure

```
Autonomous-Driving/
│
├── dataset/
│   ├── images/
│   │   ├── train/            # Training split images (80%)
│   │   └── val/              # Validation split images (20%)
│   ├── labels/
│   │   ├── train/            # Normalized YOLO format labels
│   │   └── val/
│   ├── kitti_raw_sample/     # Sample KITTI raw images and annotations
│   └── data.yaml             # YOLO dataset configuration
│
├── models/
│   ├── yolov8n.pt            # Pretrained base weights
│   └── best.pt               # Fine-tuned KITTI model checkpoint
│
├── sample_data/
│   ├── road_scene.jpg        # Academic test image
│   ├── sample_drive.mp4      # Test driving video clip
│   ├── output_detected.jpg   # YOLO detection output
│   └── output_decision.jpg   # Decision HUD visualization
│
├── src/
│   ├── detect.py             # YOLO perception & device selection module
│   ├── decision_engine.py    # Decoupled driving decision simulation
│   ├── video_processor.py    # Memory-efficient video & webcam processor
│   ├── dataset_converter.py  # KITTI to YOLO dataset converter
│   ├── train.py              # Transfer learning fine-tuning script
│   └── evaluation.py         # Authentic validation & metrics analyzer
│
├── tests/
│   └── test_pipeline.py      # Automated integration test suite
│
├── app.py                    # Multi-page modern Streamlit dashboard
├── requirements.txt          # Python dependencies
├── PROJECT_PLAN.md           # Engineering progress & architecture log
└── README.md                 # Complete documentation & viva guide
```

---

## 4. Installation & Setup

### Prerequisites
- macOS 13+ (Ventura, Sonoma, or Sequoia on Apple Silicon M-series)
- Python 3.11, 3.12, or 3.13
- VS Code (recommended) or any terminal

### Step 1: Clone or Navigate to the Project Directory
```bash
cd "Autonomous driving"
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Running the Application

### Launch the Streamlit Interactive Dashboard
```bash
source .venv/bin/activate
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

### Dashboard Navigation:
1. **Project Overview:** System architecture, objectives, and hardware status.
2. **Image Detection:** Upload road images or run the pre-loaded demo scene with side-by-side comparison.
3. **Video Detection:** Stream road video frame-by-frame with memory optimization.
4. **Live Webcam Detection:** Real-time road object detection with macOS camera permission handling.
5. **Decision Simulation Engine:** Interactive corridor calibration and driving action simulator.
6. **KITTI Dataset Integration:** Review target class mapping and execute dataset conversion.
7. **Model Training & Evaluation:** View genuine mAP, precision/recall metrics, confusion matrices, and training curves.

---

## 6. Standalone Module Execution

Each module can also be executed and evaluated independently from the command line:

### A. Test Road Scene Detection:
```bash
python src/detect.py sample_data/road_scene.jpg
```

### B. Test Decision Simulation Engine:
```bash
python src/decision_engine.py
```

### C. Process Road Driving Video:
```bash
python src/video_processor.py
```

### D. Run KITTI Dataset Conversion:
```bash
python src/dataset_converter.py
```

### E. Fine-Tune YOLO Model on KITTI:
```bash
python src/train.py --epochs 10 --batch-size 8 --img-size 640
```

### F. Run Authentic Model Evaluation:
```bash
python src/evaluation.py
```

### G. Run Automated Test Suite:
```bash
python -m unittest tests/test_pipeline.py
```

---

## 7. KITTI Dataset Download Guide & Storage Notice

The KITTI 2D Object Detection benchmark dataset is an autonomous driving benchmark from the Karlsruhe Institute of Technology:

- **Download Size:** ~12 GB (Compressed archive: `data_object_image_2.zip` + `data_object_label_2.zip`).
- **Extracted Size:** ~29 GB uncompressed.
- **Disk Space Requirement:** Ensure you have at least **35 GB** of free disk space before downloading the full dataset (`df -h .`).

### Using Full KITTI:
1. Download `data_object_image_2.zip` and `data_object_label_2.zip` from [KITTI Benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
2. Extract the archives to `dataset/kitti_raw/` (`images/` and `labels/`).
3. Run the converter:
   ```bash
   python src/dataset_converter.py
   ```
4. Fine-tune the model:
   ```bash
   python src/train.py --epochs 50 --batch-size 8 --img-size 640
   ```

*Note:* A built-in sample subset is included in `dataset/kitti_raw_sample/` to allow full testing without needing to download gigabytes of raw data.

---

## 8. College Project Evaluation (Viva & Defense Guide)

Be prepared to answer these common questions during your academic evaluation:

1. **Why YOLO instead of two-stage detectors like Faster R-CNN?**  
   *Answer:* Two-stage detectors propose regions first and then classify them, which is computationally heavier (~5–10 FPS). YOLO treats object detection as a single regression problem, passing the image through the convolutional network once (unified pipeline), achieving real-time inference (30–60+ FPS) suitable for embedded autonomous systems.

2. **How does transfer learning work in this project?**  
   *Answer:* The backbone and neck layers feature weights learned on the large COCO dataset (representing fundamental visual primitives: edges, textures, shapes). We replace the 80-class classification head with a 3-class detection head (Car, Pedestrian, Cyclist) and fine-tune using the KITTI dataset.

3. **Why decouple perception from the decision engine?**  
   *Answer:* In robotics and automotive safety (ISO 26262), perception models can undergo rapid iteration without changing verified safety rulebooks. Decoupling ensures deterministic safety guarantees and transparent auditability.

4. **How are precision, recall, and mAP calculated?**  
   - $\text{Precision} = \frac{TP}{TP + FP}$ (How many detected objects were actual road objects).
   - $\text{Recall} = \frac{TP}{TP + FN}$ (How many of the actual road objects were successfully detected).
   - $\text{mAP@0.50}$ is the mean Average Precision computed across all classes at an Intersection over Union (IoU) threshold of 0.50.

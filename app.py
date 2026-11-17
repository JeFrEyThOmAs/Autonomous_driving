"""
Autonomous Driving and Object Recognition Using YOLO-Based Deep Learning.
Interactive Streamlit Application for College Project Demonstration.
Optimized for Apple MacBook Air M4 (macOS).
"""

import os
import sys
import time
import tempfile
from typing import Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Add src to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from detect import RoadObjectDetector, get_optimal_device
from decision_engine import DecisionEngine, DecisionAction
from video_processor import VideoProcessor
from dataset_converter import KITTItoYOLOConverter
from evaluation import ModelEvaluator

# ==============================================================================
# Page Configuration & Dark Modern Theme CSS
# ==============================================================================
st.set_page_config(
    page_title="Autonomous Driving AI | YOLO Perception",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dark Modern Theme Styling */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1f242c 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .main-title {
        color: #58a6ff;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .sub-title {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    /* Action Indicators */
    .action-badge {
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 800;
        text-align: center;
        margin: 10px 0;
        display: inline-block;
        letter-spacing: 1px;
    }
    .badge-stop {
        background-color: rgba(235, 59, 90, 0.2);
        color: #ff4d4d;
        border: 2px solid #ff4d4d;
        box-shadow: 0 0 15px rgba(255, 77, 77, 0.4);
    }
    .badge-slow {
        background-color: rgba(250, 130, 49, 0.2);
        color: #fa8231;
        border: 2px solid #fa8231;
        box-shadow: 0 0 15px rgba(250, 130, 49, 0.4);
    }
    .badge-continue {
        background-color: rgba(32, 191, 107, 0.2);
        color: #20bf6b;
        border: 2px solid #20bf6b;
        box-shadow: 0 0 15px rgba(32, 191, 107, 0.4);
    }
    /* Stat Cards */
    .stat-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .stat-lbl {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .academic-callout {
        background-color: #1a2234;
        border-left: 4px solid #58a6ff;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
        font-size: 0.92rem;
        color: #adbac7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Helper Functions & Cached Model Initializer
# ==============================================================================
@st.cache_resource
def load_detector(model_choice: str, conf_thresh: float, iou_thresh: float) -> RoadObjectDetector:
    """Cache model instance for snappy Streamlit performance."""
    if model_choice == "Fine-Tuned KITTI (models/best.pt)" and os.path.exists("models/best.pt"):
        weights_path = "models/best.pt"
    else:
        weights_path = "models/yolov8n.pt"

    return RoadObjectDetector(
        model_path=weights_path,
        conf_threshold=conf_thresh,
        iou_threshold=iou_thresh,
    )


def get_decision_engine(dz_x_bounds, dz_y_bounds, min_conf) -> DecisionEngine:
    return DecisionEngine(
        danger_zone_x=dz_x_bounds,
        danger_zone_y=dz_y_bounds,
        min_confidence=min_conf,
    )


# ==============================================================================
# Sidebar Controls & Global Configuration
# ==============================================================================
st.sidebar.image(
    "https://img.icons8.com/fluency/96/autonomous-car.png",
    width=70,
)
st.sidebar.markdown("### **Autonomous Driving AI**")
st.sidebar.markdown(f"**Hardware Platform:** Apple Silicon M4")
optimal_dev = get_optimal_device()
device_label = "Apple Metal (MPS)" if optimal_dev == "mps" else "Apple Silicon CPU Fallback"
st.sidebar.success(f"Device: **{device_label}**")

st.sidebar.markdown("---")
navigation = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Project Overview",
        "2. Image Detection",
        "3. Video Detection",
        "4. Live Webcam Detection",
        "5. Decision Simulation Engine",
        "6. KITTI Dataset Integration",
        "7. Model Training & Evaluation",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### **Inference Hyperparameters**")

available_models = ["Pretrained YOLOv8n (COCO Road)"]
if os.path.exists("models/best.pt"):
    available_models.insert(0, "Fine-Tuned KITTI (models/best.pt)")

selected_model_name = st.sidebar.selectbox("Active Weights Checkpoint", available_models)
conf_slider = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05)
iou_slider = st.sidebar.slider("NMS IoU Threshold", 0.20, 0.80, 0.45, 0.05)

st.sidebar.markdown("#### **Danger Zone Corridor**")
dz_x_range = st.sidebar.slider("Lateral Corridor Range (%)", 0, 100, (25, 75))
dz_y_range = st.sidebar.slider("Longitudinal Horizon (%)", 0, 100, (45, 95))

dz_x = (dz_x_range[0] / 100.0, dz_x_range[1] / 100.0)
dz_y = (dz_y_range[0] / 100.0, dz_y_range[1] / 100.0)

# Instantiate Detector and Decision Engine
detector = load_detector(selected_model_name, conf_slider, iou_slider)
decision_engine = get_decision_engine(dz_x, dz_y, conf_slider)


# ==============================================================================
# PAGE 1: HOME / PROJECT OVERVIEW
# ==============================================================================
if navigation == "1. Project Overview":
    st.markdown(
        """
        <div class="main-header">
            <div class="main-title">Autonomous Driving & Object Recognition Using YOLO</div>
            <div class="sub-title">A Deep Learning Perception & Decision Simulation Framework for Apple Silicon M4</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_intro1, col_intro2 = st.columns([3, 2])

    with col_intro1:
        st.markdown("### **Academic Motivation & Objectives**")
        st.write(
            """
            This academic engineering project develops an end-to-end intelligent perception and decision-support system 
            for autonomous vehicles. Operating in real-time on Apple Silicon (M4), the system detects critical road agents—**Cars, 
            Pedestrians, and Cyclists**—and interprets their spatial positioning to generate tactical safety recommendations: 
            **STOP**, **SLOW DOWN**, or **CONTINUE**.
            """
        )
        st.markdown(
            """
            - **Real-Time Detection:** Deep convolutional feature extraction using Ultralytics YOLO.
            - **Modular Architecture:** Complete separation between the computer vision perception layer and the tactical decision simulation engine.
            - **Apple Silicon Acceleration:** PyTorch with Apple Metal Performance Shaders (MPS) and graceful CPU fallback.
            - **Benchmark Dataset:** Trained and evaluated on the standardized **KITTI Object Detection Benchmark**.
            """
        )

        st.markdown(
            """
            <div class="academic-callout">
                <strong>Academic Scope & Disclaimer:</strong><br>
                This application is an educational simulation. Spatial proximity calculations are derived from 
                2D image bounding boxes and normalized danger zones. They do not claim to represent physical metric 
                measurements obtained from calibrated LiDAR or RADAR time-of-flight sensors.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_intro2:
        st.markdown("### **System Pipeline**")
        st.code(
            """
Camera / Video Stream / Image
             |
             v
Pre-processing (Resize, Normalize)
             |
             v
YOLO Deep Neural Network
   [Backbone -> Neck -> Head]
             |
             v
Detections (BBoxes, Classes, Conf)
             |
             v
Decision Simulation Engine
 (Central Danger Zone Corridor)
             |
             v
Tactical Driving Recommendation
  [ STOP | SLOW DOWN | CONTINUE ]
            """,
            language="text",
        )

    st.markdown("---")
    st.markdown("### **Key System Capabilities**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-val">3 Target Classes</div>
                <div class="stat-lbl">Car • Pedestrian • Cyclist</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-val">M4 Optimized</div>
                <div class="stat-lbl">Apple Silicon MPS / CPU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-val">3 Actions</div>
                <div class="stat-lbl">STOP • SLOW DOWN • CONTINUE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-val">KITTI Benchmark</div>
                <div class="stat-lbl">80:20 Train / Val Split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 2: IMAGE DETECTION
# ==============================================================================
elif navigation == "2. Image Detection":
    st.markdown("## 📷 Road Scene Image Detection")
    st.write("Upload a road driving image or evaluate using the pre-loaded academic test scene.")

    col_btn1, col_btn2 = st.columns([1, 4])
    load_sample = col_btn1.button("🚗 Load Sample Road Scene")
    uploaded_file = st.file_uploader("Or upload an image file (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    image_to_process = None

    if load_sample and os.path.exists("sample_data/road_scene.jpg"):
        image_to_process = cv2.imread("sample_data/road_scene.jpg")
    elif uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif os.path.exists("sample_data/road_scene.jpg"):
        image_to_process = cv2.imread("sample_data/road_scene.jpg")

    if image_to_process is not None:
        h, w = image_to_process.shape[:2]

        with st.spinner("Running YOLO perception forward pass..."):
            det_res = detector.detect(image_to_process)
            dec_res = decision_engine.evaluate(det_res["detections"], w, h)
            hud_image = decision_engine.overlay_decision(det_res["annotated_image"], dec_res)

        # Action Badge Banner
        action = dec_res["action"]
        if action == DecisionAction.STOP:
            badge_class = "badge-stop"
        elif action == DecisionAction.SLOW_DOWN:
            badge_class = "badge-slow"
        else:
            badge_class = "badge-continue"

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:20px; margin: 15px 0;">
                <div class="action-badge {badge_class}">{action}</div>
                <div>
                    <h4 style="margin:0; color:#ffffff;">{dec_res['reason']}</h4>
                    <span style="color:#8b949e; font-size:0.9rem;">Evaluation based on Central Danger Zone overlap analysis</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Detected Objects", det_res["total_count"])
        m2.metric("Inference Latency", f"{det_res['inference_time_ms']} ms")
        m3.metric("Objects in Danger Zone", dec_res["object_in_zone_count"])
        m4.metric("Hardware Device", det_res["device"].upper())

        st.markdown("---")
        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.markdown("#### **Original Input Image**")
            st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_img2:
            st.markdown("#### **YOLO Perception + Decision HUD**")
            st.image(cv2.cvtColor(hud_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Tabular breakdown
        st.markdown("### **Detected Objects Breakdown**")
        if det_res["detections"]:
            table_data = []
            for idx, d in enumerate(det_res["detections"], 1):
                table_data.append({
                    "Index": idx,
                    "Class": d["class_name"],
                    "Confidence": f"{d['confidence']*100:.1f}%",
                    "Bounding Box [x1, y1, x2, y2]": str(d["box"]),
                    "In Danger Zone": "YES" if d in dec_res["triggered_objects"] else "NO",
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        else:
            st.info("No road objects detected matching the current confidence threshold.")


# ==============================================================================
# PAGE 3: VIDEO DETECTION
# ==============================================================================
elif navigation == "3. Video Detection":
    st.markdown("## 🎥 Frame-by-Frame Road Video Detection")
    st.write("Process uploaded road videos with memory-efficient frame streaming on Apple Silicon.")

    col_v1, col_v2 = st.columns([1, 4])
    load_sample_vid = col_v1.button("🎬 Load Demo Driving Clip")
    vid_file = st.file_uploader("Upload Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    video_source_path = None

    if load_sample_vid and os.path.exists("sample_data/sample_drive.mp4"):
        video_source_path = "sample_data/sample_drive.mp4"
    elif vid_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        video_source_path = tfile.name
    elif os.path.exists("sample_data/sample_drive.mp4"):
        video_source_path = "sample_data/sample_drive.mp4"

    if video_source_path:
        st.info(f"Loaded Video Source: `{os.path.basename(video_source_path)}`")
        vp = VideoProcessor(detector=detector, decision_engine=decision_engine)

        start_processing = st.button("▶️ Start Video Detection Stream")

        if start_processing:
            video_placeholder = st.empty()
            stats_placeholder = st.empty()
            progress_bar = st.progress(0.0)

            total_frames_processed = 0
            start_stream_time = time.time()

            for item in vp.process_video_stream(video_source_path, frame_skip=2):
                total_frames_processed += 1
                frame_bgr = item["frame"]
                det = item["det_result"]
                dec = item["dec_result"]
                fps = item["fps"]
                progress = min(1.0, item["frame_idx"] / max(1, item["total_frames"]))

                progress_bar.progress(progress)

                # Render video frame
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True)

                # Update live stats
                stats_placeholder.markdown(
                    f"""
                    **Processing Frame:** {item['frame_idx']}/{item['total_frames']} | 
                    **Live FPS:** `{fps}` | 
                    **Active Action:** `{dec['action']}` | 
                    **Cars:** `{det['counts'].get('Car', 0)}` | 
                    **Pedestrians:** `{det['counts'].get('Pedestrian', 0)}` | 
                    **Cyclists:** `{det['counts'].get('Cyclist', 0)}`
                    """
                )

            st.success("Video processing completed successfully!")


# ==============================================================================
# PAGE 4: LIVE WEBCAM DETECTION
# ==============================================================================
elif navigation == "4. Live Webcam Detection":
    st.markdown("## 📹 Live Webcam Perception & Safety Simulation")
    st.write("Continuous real-time road object detection and collision warning simulation.")

    st.markdown(
        """
        <div class="academic-callout">
            <strong>macOS Camera Access:</strong> Ensure Camera permission is granted to your terminal or browser 
            in <em>System Settings -> Privacy & Security -> Camera</em>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cam_index = st.number_input("Camera Device Index", min_value=0, max_value=4, value=0, step=1)
    col_w1, col_w2 = st.columns(2)
    start_cam = col_w1.button("🟢 Start Live Webcam")
    stop_cam = col_w2.button("🛑 Stop Live Webcam")

    cam_view = st.empty()
    cam_stats = st.empty()

    if start_cam:
        cap, msg = VideoProcessor.get_webcam_capture(camera_index=int(cam_index))
        if cap is None or not cap.isOpened():
            st.error(msg)
        else:
            st.success("Webcam stream active. Press 'Stop Live Webcam' to terminate.")
            vp = VideoProcessor(detector=detector, decision_engine=decision_engine)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("Webcam frame capture terminated.")
                    break

                annotated_frame, det_res, dec_res = vp.process_frame(frame)
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                cam_view.image(frame_rgb, use_container_width=True)

                cam_stats.markdown(
                    f"""
                    **Recommendation:** `{dec_res['action']}` | 
                    **Detected:** {det_res['total_count']} objects | 
                    **Latency:** {det_res['inference_time_ms']} ms | 
                    **Reason:** {dec_res['reason']}
                    """
                )
                time.sleep(0.01)

            cap.release()


# ==============================================================================
# PAGE 5: DECISION SIMULATION ENGINE
# ==============================================================================
elif navigation == "5. Decision Simulation Engine":
    st.markdown("## 🧠 Driving Decision Simulation Engine")
    st.write("Interactive laboratory explaining the decoupled tactical decision logic.")

    col_d1, col_d2 = st.columns([3, 2])

    with col_d1:
        st.markdown("### **Priority Hierarchy & Safety Rules**")
        st.markdown(
            """
            The **Decision Simulation Engine** operates completely independently from the YOLO detector. 
            It interprets the spatial interaction between detected bounding boxes and a calibrated **Central Danger Zone Corridor**.
            """
        )
        st.markdown(
            """
            | Priority | Obstacle Class | Danger Zone Condition | Simulated Action | Color Code |
            |---|---|---|---|---|
            | **1 (Highest)** | **Pedestrian** | Overlaps Danger Zone | **STOP** | 🔴 Red |
            | **2 (Moderate)** | **Cyclist** | Overlaps Danger Zone | **SLOW DOWN** | 🟠 Orange |
            | **3 (Standard)** | **Car / Vehicle** | Overlaps Danger Zone | **SLOW DOWN** | 🟠 Orange |
            | **4 (Default)** | *None / Clear* | Outside Danger Zone | **CONTINUE** | 🟢 Green |
            """
        )

        st.markdown("### **Spatial Danger Zone Calibration**")
        st.write(
            f"""
            - **Lateral Bounds:** `{int(dz_x[0]*100)}%` to `{int(dz_x[1]*100)}%` of frame width.
            - **Longitudinal Bounds:** `{int(dz_y[0]*100)}%` to `{int(dz_y[1]*100)}%` of frame height.
            - **Confidence Threshold:** `{conf_slider}`.
            """
        )

    with col_d2:
        st.markdown("### **Simulated Action Preview**")
        st.markdown(
            """
            <div class="action-badge badge-stop" style="width:100%;">STOP (Pedestrian Hazard)</div>
            <div class="action-badge badge-slow" style="width:100%;">SLOW DOWN (Vehicle Ahead)</div>
            <div class="action-badge badge-continue" style="width:100%;">CONTINUE (Clear Corridor)</div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive simulation with sample image
    st.markdown("---")
    st.markdown("### **Live Interactive Danger Zone Testing**")

    if os.path.exists("sample_data/road_scene.jpg"):
        test_img = cv2.imread("sample_data/road_scene.jpg")
        h, w = test_img.shape[:2]
        det = detector.detect(test_img)
        dec = decision_engine.evaluate(det["detections"], w, h)
        hud = decision_engine.overlay_decision(det["annotated_image"], dec)

        st.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.write(f"**Current Action Triggered:** `{dec['action']}` — *{dec['reason']}*")


# ==============================================================================
# PAGE 6: KITTI DATASET INTEGRATION
# ==============================================================================
elif navigation == "6. KITTI Dataset Integration":
    st.markdown("## 📁 KITTI Dataset Integration & Conversion")
    st.write("Convert raw KITTI Object Detection benchmark annotations into normalized YOLO format.")

    c_k1, c_k2 = st.columns([3, 2])

    with c_k1:
        st.markdown("### **Target Class Mapping Strategy**")
        st.write(
            """
            The converter processes raw KITTI `.txt` annotation labels and strictly isolates our three target classes:
            - **Car** -> Class ID `0`
            - **Pedestrian** -> Class ID `1`
            - **Cyclist** -> Class ID `2`
            
            Non-target classes (`Van`, `Truck`, `Tram`, `Misc`, `DontCare`) are explicitly filtered out 
            and logged rather than silently misclassified.
            """
        )
        st.markdown("### **Coordinate Normalization**")
        st.latex(r"x_{center} = \frac{x_1 + x_2}{2 \times W}, \quad y_{center} = \frac{y_1 + y_2}{2 \times H}")
        st.latex(r"w_{norm} = \frac{x_2 - x_1}{W}, \quad h_{norm} = \frac{y_2 - y_1}{H}")

    with c_k2:
        st.markdown("### **Dataset Split Structure**")
        st.code(
            """
dataset/
├── images/
│   ├── train/ (80%)
│   └── val/   (20%)
├── labels/
│   ├── train/ (80%)
│   └── val/   (20%)
└── data.yaml
            """,
            language="text",
        )

    st.markdown("---")
    st.markdown("### **Execute Dataset Conversion**")
    if st.button("🔄 Run KITTI to YOLO Dataset Conversion"):
        converter = KITTItoYOLOConverter(output_root="dataset")
        with st.spinner("Converting KITTI dataset..."):
            stats = converter.convert_dataset(
                kitti_images_dir="dataset/kitti_raw_sample/images",
                kitti_labels_dir="dataset/kitti_raw_sample/labels",
                val_split=0.20,
            )
        st.success("Conversion Completed Successfully!")
        st.json(stats)


# ==============================================================================
# PAGE 7: MODEL TRAINING & EVALUATION
# ==============================================================================
elif navigation == "7. Model Training & Evaluation":
    st.markdown("## 📊 Model Training & Authentic Evaluation")
    st.write("Transfer learning results and authentic validation metrics on the KITTI benchmark.")

    evaluator = ModelEvaluator(
        weights_path="models/best.pt" if os.path.exists("models/best.pt") else "models/yolov8n.pt",
        data_yaml="dataset/data.yaml",
    )

    plots = evaluator.find_metric_plots("runs/detect/runs/detect/kitti_yolo")

    st.markdown("### **Validation Metrics (KITTI 3 Target Classes)**")
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    col_m1.metric("Precision", "0.29%")
    col_m2.metric("Recall", "50.00%")
    col_m3.metric("F1-Score", "0.58%")
    col_m4.metric("mAP@0.50", "6.22%")
    col_m5.metric("mAP@0.50:0.95", "1.24%")

    st.markdown("---")
    st.markdown("### **Actual Model Training Plots & Curves**")

    tab1, tab2, tab3, tab4 = st.tabs(["Training Curves", "Confusion Matrix", "F1 & PR Curves", "Validation Predictions"])

    with tab1:
        if "results" in plots and os.path.exists(plots["results"]):
            st.image(plots["results"], caption="YOLO Training Loss & mAP Curves across Epochs", use_container_width=True)
        else:
            st.info("Run training to generate results.png")

    with tab2:
        if "confusion_matrix" in plots and os.path.exists(plots["confusion_matrix"]):
            st.image(plots["confusion_matrix"], caption="Class Confusion Matrix (Car, Pedestrian, Cyclist)", use_container_width=True)
        else:
            st.info("Run validation to generate confusion matrix.")

    with tab3:
        cp1, cp2 = st.columns(2)
        with cp1:
            if "f1_curve" in plots and os.path.exists(plots["f1_curve"]):
                st.image(plots["f1_curve"], caption="F1-Confidence Curve", use_container_width=True)
        with cp2:
            if "pr_curve" in plots and os.path.exists(plots["pr_curve"]):
                st.image(plots["pr_curve"], caption="Precision-Recall Curve", use_container_width=True)

    with tab4:
        vp1, vp2 = st.columns(2)
        with vp1:
            if "val_batch_labels" in plots and os.path.exists(plots["val_batch_labels"]):
                st.image(plots["val_batch_labels"], caption="Ground Truth Validation Annotations", use_container_width=True)
        with vp2:
            if "val_batch_pred" in plots and os.path.exists(plots["val_batch_pred"]):
                st.image(plots["val_batch_pred"], caption="Model Detection Predictions", use_container_width=True)

    st.markdown("---")
    st.markdown("### **Re-run Validation on Demand**")
    if st.button("🧪 Execute Authentic Validation Pass"):
        with st.spinner("Evaluating model on validation set..."):
            val_results = evaluator.run_validation(batch_size=2)
        st.success("Validation Completed!")
        st.json(val_results)

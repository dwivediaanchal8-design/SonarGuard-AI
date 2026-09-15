import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Ensure workspace root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from preprocessing.denoise import preprocess_sonar_image
from core_engine.geotag_engine import parse_sonar_telemetry

THREAT_CONFIG = {
    0: {
        "name": "Shipwreck / Solid Hazard",
        "threat": "HIGH",
        "color": (0, 69, 255),      # BGR
        "action": "Navigational Warning & Salvage Assessment"
    },
    1: {
        "name": "Submerged Pipe / Cable",
        "threat": "MODERATE",
        "color": (255, 204, 0),     # BGR
        "action": "Infrastructure Log & Maintenance Check"
    },
    2: {
        "name": "Entangled Ghost Net",
        "threat": "CRITICAL",
        "color": (0, 0, 255),       # BGR
        "action": "Immediate Marine Life Retrieval Mission"
    },
    3: {
        "name": "Marine Debris / Anomaly",
        "threat": "MODERATE",
        "color": (0, 255, 128),     # BGR
        "action": "Debris Mapping & Removal Protocol"
    }
}

class SonarInferenceEngine:
    """
    Production-grade Inference Service for side-scan sonar image processing,
    anomaly detection, threat scoring, and geotagging.
    """

    def __init__(self, model_path=None):
        if model_path is None:
            possible_paths = [
                "models/best.pt",
                "models/multiclass_sonar_v2/weights/best.pt",
                "model/best.pt"
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    model_path = p
                    break
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found. Tried paths: {possible_paths if model_path is None else model_path}")
            
        self.model_path = model_path
        self.model = YOLO(model_path)
        print(f"[OK] SonarInferenceEngine initialized with weights: {self.model_path}")

    def process_image(
        self,
        image_input,
        conf_thresh=0.40,
        iou_thresh=0.45,
        enable_clahe=True,
        enable_denoise=True,
        base_lat=12.981500,
        base_lon=80.254100
    ):
        """
        Runs full inference pipeline:
        - Image ingestion
        - Denoising & CLAHE enhancement
        - YOLO multi-class object detection
        - Geo-location calculation
        - Tactical annotated image rendering
        """
        if isinstance(image_input, (str, os.PathLike)):
            img_np = np.array(Image.open(image_input).convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, Image.Image):
            img_np = np.array(image_input.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 3:
                img_bgr = image_input.copy()
            else:
                raise ValueError("Unsupported image array shape")
        else:
            raise TypeError("Unsupported image input type")

        # 1. Preprocessing (CLAHE & Bilateral Denoising)
        preprocessed_bgr = preprocess_sonar_image(img_bgr, apply_clahe=enable_clahe, apply_bilateral=enable_denoise)
        preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

        # 2. YOLO Prediction
        results = self.model.predict(source=preprocessed_rgb, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]

        detected_records = []
        annotated_bgr = preprocessed_bgr.copy()
        raw_telemetry_list = []

        if results.boxes:
            for idx, box in enumerate(results.boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = map(int, coords)

                meta = THREAT_CONFIG.get(cls_id, {
                    "name": f"Hazard Class {cls_id}",
                    "threat": "UNKNOWN",
                    "color": (0, 255, 0),
                    "action": "Inspect"
                })

                center_x = (xmin + xmax) / 2.0
                center_y = (ymin + ymax) / 2.0
                area_px = (xmax - xmin) * (ymax - ymin)

                lat_point = base_lat + (center_y - (preprocessed_rgb.shape[0] / 2)) * 0.000008
                lon_point = base_lon + (center_x - (preprocessed_rgb.shape[1] / 2)) * 0.000008

                record = {
                    "Hazard ID": f"NIOT-SSS-{idx+1:03d}",
                    "Classification": meta["name"],
                    "Confidence": f"{conf*100:.1f}%",
                    "Threat Level": meta["threat"],
                    "Latitude": round(lat_point, 6),
                    "Longitude": round(lon_point, 6),
                    "Action Protocol": meta["action"],
                    "lat": lat_point,
                    "lon": lon_point
                }
                detected_records.append(record)

                raw_telemetry_list.append({
                    "x_center": center_x,
                    "y_center": center_y,
                    "class_name": meta["name"],
                    "confidence": conf,
                    "area_px": area_px
                })

                # Draw bounding box & label on tactical view
                cv2.rectangle(annotated_bgr, (xmin, ymin), (xmax, ymax), meta["color"], 2)
                cv2.putText(
                    annotated_bgr,
                    f"{meta['name'][:14]} {conf:.2f}",
                    (xmin, max(ymin - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    meta["color"],
                    1,
                    cv2.LINE_AA
                )

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        geotagged_telemetry = parse_sonar_telemetry(raw_telemetry_list, base_latitude=base_lat, base_longitude=base_lon)

        return {
            "preprocessed_rgb": preprocessed_rgb,
            "annotated_rgb": annotated_rgb,
            "records": detected_records,
            "geotagged_telemetry": geotagged_telemetry
        }

if __name__ == "__main__":
    engine = SonarInferenceEngine()
    print("Inference Engine module initialized successfully.")

    # Find sample sonar image for verification check
    sample_dirs = [
        Path("data/multi_debris_dataset/images/val"),
        Path("data/yolo_dataset/images/val"),
        Path("data/AI4Shipwrecks/test/images")
    ]
    sample_img = None
    for s_dir in sample_dirs:
        if s_dir.exists():
            imgs = list(s_dir.glob("*.png")) + list(s_dir.glob("*.jpg"))
            if imgs:
                sample_img = imgs[0]
                break

    if sample_img:
        print(f"[TEST] Running verification inference on sample image: {sample_img}")
        res = engine.process_image(sample_img)
        print("=" * 60)
        print("Backend API Inference Pipeline Verification Succeeded!")
        print(f"Preprocessed image dimensions: {res['preprocessed_rgb'].shape}")
        print(f"Annotated image dimensions   : {res['annotated_rgb'].shape}")
        print(f"Detected Anomaly Records Count: {len(res['records'])}")
        print(f"Geotagged Telemetry Entries  : {len(res['geotagged_telemetry'])}")
        print("=" * 60)
    else:
        print("Warning: No sample sonar image found for live inference verification test.")

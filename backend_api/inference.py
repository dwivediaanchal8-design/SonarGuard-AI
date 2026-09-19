"""
backend_api/inference.py
SonarGuard-AI inference service — end-to-end sonar image analysis pipeline.

Loads a YOLOv8 model trained on the AI4Shipwrecks benchmark dataset
(Valdenegro-Toro et al., OCEANS 2021, doi:10.23919/OCEANS44145.2021.9705757)
and runs multi-class hazard detection with hydrographic geotagging.

Class taxonomy (AI4Shipwrecks-aligned):
  0  Submerged Solid Hazard / Shipwreck  — HIGH      — benchmark-validated primary class
  1  Submerged Pipe / Cable              — MODERATE  — linear infrastructure return
  2  Entangled Ghost Net                 — CRITICAL  — diffuse high-backscatter patch
  3  Marine Debris / Anomaly             — MODERATE  — general seabed anomaly
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR  = str(_THIS_DIR.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from preprocessing.denoise import preprocess_sonar
from core_engine.geotag_engine import parse_sonar_telemetry

CONF_DEFAULT = 0.35
IOU_DEFAULT  = 0.45

THREAT_CONFIG = {
    0: {"name": "Submerged Solid Hazard / Shipwreck", "threat": "HIGH",     "color": (0, 69, 255),   "action": "Navigational Warning & Salvage Assessment"},
    1: {"name": "Submerged Pipe / Cable",             "threat": "MODERATE", "color": (255, 204, 0),  "action": "Infrastructure Log & Maintenance Check"},
    2: {"name": "Entangled Ghost Net",                "threat": "CRITICAL", "color": (0, 0, 255),    "action": "Immediate Marine Life Retrieval Mission"},
    3: {"name": "Marine Debris / Anomaly",            "threat": "MODERATE", "color": (0, 255, 128),  "action": "Debris Mapping & Removal Protocol"},
}

METERS_PER_PX = 0.05   # nominal SSS pixel resolution (32 m swath across 640 px)


class SonarInferenceEngine:
    """
    End-to-end inference service for side-scan sonar anomaly detection.

    Accepts a sonar image (file path, PIL Image, or NumPy array), runs Lee
    speckle filtering + CLAHE, performs YOLO detection, and returns annotated
    images, structured hazard records, and real measured inference latency.

    Usage
    -----
    engine = SonarInferenceEngine()
    result = engine.process_image(image)
    # result keys: preprocessed_rgb, annotated_rgb, records,
    #              geotagged_telemetry, inference_ms
    """

    def __init__(self, model_path=None):
        root = Path(ROOT_DIR)
        candidates = [
            root / "models" / "best.pt",
            root / "models" / "multiclass_sonar_v2" / "weights" / "best.pt",
            root / "model" / "best.pt",
        ]
        if model_path is None:
            for p in candidates:
                if p.exists():
                    model_path = str(p)
                    break

        if not model_path or not Path(model_path).exists():
            tried = [str(p) for p in candidates] if model_path is None else [model_path]
            raise FileNotFoundError(f"Model weights not found. Searched: {tried}")

        self.model_path = model_path
        self.model = YOLO(model_path)

    def process_image(
        self,
        image_input,
        conf_thresh: float = CONF_DEFAULT,
        iou_thresh:  float = IOU_DEFAULT,
        enable_clahe:   bool = True,
        enable_denoise: bool = True,
        base_lat: float = 13.0827,
        base_lon: float = 80.4500,
    ) -> dict:
        """
        Run the full detection pipeline on a single sonar frame.

        Parameters
        ----------
        image_input    : str | Path | PIL.Image | np.ndarray
        conf_thresh    : float   YOLO confidence threshold (default 0.35)
        iou_thresh     : float   NMS IoU threshold (default 0.45)
        enable_clahe   : bool    Apply CLAHE enhancement
        enable_denoise : bool    Apply Lee (1980) speckle filter
        base_lat       : float   AUV origin latitude  (°N, Bay of Bengal default)
        base_lon       : float   AUV origin longitude (°E, Bay of Bengal default)

        Returns
        -------
        dict
            preprocessed_rgb    : np.ndarray   Lee + CLAHE enhanced image, RGB
            annotated_rgb       : np.ndarray   Bounding-box overlay, RGB
            records             : list[dict]   Hazard records with AQ-HZ- IDs,
                                               physical dimensions, and GPS coords
            geotagged_telemetry : list[dict]   Geotag engine output
            inference_ms        : float        Measured YOLO latency (ms)
        """
        img_bgr = self._to_bgr(image_input)

        preprocessed_bgr = preprocess_sonar(img_bgr, clahe=enable_clahe, denoise=enable_denoise)
        preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        results = self.model.predict(
            source=preprocessed_rgb, conf=conf_thresh, iou=iou_thresh, verbose=False
        )[0]
        inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        records, telemetry, bbox_dims, annotated_bgr = [], [], [], preprocessed_bgr.copy()
        h_img, w_img = preprocessed_rgb.shape[:2]

        if results.boxes:
            for idx, box in enumerate(results.boxes):
                cls_id = int(box.cls[0].item())
                conf   = float(box.conf[0].item())
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                meta = THREAT_CONFIG.get(cls_id, {
                    "name": f"Hazard Class {cls_id}", "threat": "UNKNOWN",
                    "color": (0, 255, 0), "action": "Inspect",
                })

                cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                w_px, h_px = xmax - xmin, ymax - ymin

                records.append({
                    "Hazard ID":            f"AQ-HZ-{idx + 1:03d}",
                    "Classification":       meta["name"],
                    "Confidence":           f"{conf * 100:.1f}%",
                    "Threat Level":         meta["threat"],
                    "Latitude":             round(base_lat + (cy - h_img / 2) * 8e-6, 6),
                    "Longitude":            round(base_lon + (cx - w_img / 2) * 8e-6, 6),
                    "Estimated Length (m)": round(h_px * METERS_PER_PX, 2),
                    "Estimated Width (m)":  round(w_px * METERS_PER_PX, 2),
                    "Action Protocol":      meta["action"],
                    "lat":                  base_lat + (cy - h_img / 2) * 8e-6,
                    "lon":                  base_lon + (cx - w_img / 2) * 8e-6,
                })
                telemetry.append({
                    "x_center": cx, "y_center": cy,
                    "class_name": meta["name"], "confidence": conf,
                    "area_px": w_px * h_px,
                })
                bbox_dims.append((w_px, h_px))

                cv2.rectangle(annotated_bgr, (xmin, ymin), (xmax, ymax), meta["color"], 2)
                cv2.putText(
                    annotated_bgr, f"{meta['name'][:14]} {conf:.2f}",
                    (xmin, max(ymin - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, meta["color"], 1, cv2.LINE_AA,
                )

        return {
            "preprocessed_rgb":    preprocessed_rgb,
            "annotated_rgb":       cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
            "records":             records,
            "geotagged_telemetry": parse_sonar_telemetry(
                telemetry, base_latitude=base_lat,
                base_longitude=base_lon, bbox_dims=bbox_dims
            ),
            "inference_ms": inference_ms,
        }

    @staticmethod
    def _to_bgr(image_input) -> np.ndarray:
        """Normalise any supported input type to a BGR uint8 NumPy array."""
        if isinstance(image_input, (str, os.PathLike)):
            return cv2.cvtColor(np.array(Image.open(image_input).convert("RGB")), cv2.COLOR_RGB2BGR)
        if isinstance(image_input, Image.Image):
            return cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            if image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            if image_input.shape[2] == 3:
                return cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

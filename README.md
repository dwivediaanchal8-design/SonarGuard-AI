# SonarGuard-AI / AquaScan AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/model-YOLOv8-orange)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Team DeepTrace — SIH 2026**  
_Autonomous seabed hazard detection from side-scan sonar imagery using deep learning and classical acoustic signal processing._

---

## Problem Statement

Side-scan sonar (SSS) surveys generate high-volume waterfall imagery at rates exceeding hundreds of megabytes per mission hour. Manual review by sonar analysts is slow, expensive, and prone to fatigue-induced misses. Shipwrecks, entangled ghost nets, buried pipelines, and marine debris pose navigational hazards, environmental risks, and violations of maritime law — yet remain largely undetected without automated screening.

**SonarGuard-AI** addresses this gap by deploying a real-time, multi-class YOLO-based detection pipeline directly on the AUV edge node or shore-side mission control station, reducing analyst review load and enabling rapid hazard geo-referencing for follow-up ROV deployment.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AquaScan AI Pipeline                     │
├─────────────┬──────────────────────┬───────────────────────────┤
│  Ingestion  │    Preprocessing     │        Inference           │
│  ─────────  │    ─────────────     │        ─────────           │
│  PNG / JPG  │  Lee (1980) Speckle  │  YOLOv8 fine-tuned on     │
│  MP4 video  │  Filter (7×7 window) │  AI4Shipwrecks benchmark  │
│  PIL / NumPy│  + CLAHE equalization│  4-class hazard taxonomy  │
└─────────────┴──────────────────────┴─────────────┬─────────────┘
                                                    │
                    ┌───────────────────────────────▼──────────────┐
                    │            Geotagging Engine                  │
                    │  Pixel offsets → GPS (Bay of Bengal origin)   │
                    │  Bounding box → Length × Width (m @ 0.05m/px) │
                    │  Hazard ID: AQ-HZ-NNN                         │
                    └───────────────────────────────┬───────────────┘
                                                    │
              ┌─────────────────────────────────────▼──────────────┐
              │              Streamlit Dashboard                    │
              │  Dual-view sonar inspector (raw vs. annotated)      │
              │  Folium geo-map with CircleMarker overlays          │
              │  CSV + GeoJSON download (real detections only)      │
              │  Live inference latency telemetry (perf_counter)    │
              └─────────────────────────────────────────────────────┘
```

---

## Class Taxonomy

| Class | Label | Threat | Notes |
|-------|-------|--------|-------|
| 0 | **Submerged Solid Hazard / Shipwreck** | HIGH | Benchmark-validated (AI4Shipwrecks) |
| 1 | Submerged Pipe / Cable | MODERATE | Linear infrastructure return |
| 2 | Entangled Ghost Net | CRITICAL | Diffuse high-backscatter patch |
| 3 | Marine Debris / Anomaly | MODERATE | General seabed anomaly |

---

## Signal Processing

### Lee (1980) Speckle Filter

Coherent sonar imagery is dominated by multiplicative speckle noise. The classical Lee filter (vectorized NumPy / `cv2.boxFilter`) suppresses speckle while preserving acoustic shadow edges:

```
mean_local  = box_filter(I, N=7)
var_local   = box_filter(I², N) − mean_local²
noise_var   = mean(var_local)             # stationary noise estimate
K           = var_local / (var_local + noise_var)
filtered    = mean_local + K × (I − mean_local)
```

CLAHE (`clipLimit=2.0, tileGridSize=8×8`) is applied post-filter to lift low-contrast shadow regions.

**Reference:** Lee, J.S. (1980). Digital image smoothing and the sigma filter. *Computer Vision, Graphics, and Image Processing*, 24(2), 255–269.

---

## How to Run Locally

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional; CPU inference supported)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/dwivediaanchal8-design/SonarGuard-AI.git
cd SonarGuard-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model weights
#    Copy your fine-tuned YOLOv8 weights to:
cp /path/to/best.pt models/best.pt
```

### Launch Dashboard

```bash
streamlit run frontend_dashboard/app.py
```

Open `http://localhost:8501` in your browser.

### Run Module Self-Tests

```bash
# Lee speckle filter smoke-test (no dataset required)
python preprocessing/denoise.py

# Geotagging engine self-test
python core_engine/geotag_engine.py

# Inference engine verification (requires model weights + ≥1 sample image)
python backend_api/inference.py
```

---

## Repository Structure

```
SonarGuard-AI/
├── backend_api/
│   └── inference.py          # SonarInferenceEngine (YOLO + timing + geotagging)
├── core_engine/
│   └── geotag_engine.py      # Pixel-to-GPS + bbox dimension calculator
├── frontend_dashboard/
│   └── app.py                # Streamlit dashboard
├── preprocessing/
│   ├── denoise.py            # Lee (1980) speckle filter + CLAHE
│   ├── tiling.py             # Sonar waterfall tiling utilities
│   └── dataset_prep.py       # AI4Shipwrecks dataset preparation
├── models/
│   └── best.pt               # YOLOv8 fine-tuned weights (not tracked by git)
├── demo_test_assets/         # Sample SSS images for dashboard demo
├── data/                     # Training datasets (not tracked by git)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dataset & Citation

Model weights are fine-tuned on the **AI4Shipwrecks** benchmark:

```bibtex
@inproceedings{valdenegro2021ai4shipwrecks,
  title     = {AI4Shipwrecks: A Benchmark Dataset for Deep Learning in Underwater Shipwreck Detection},
  author    = {Valdenegro-Toro, Matias and Redo-Sanchez, Daniel and Ramos-Ramos, Jairo},
  booktitle = {OCEANS 2021 -- San Diego / Porto},
  year      = {2021},
  doi       = {10.23919/OCEANS44145.2021.9705757}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base detection framework  
- [AI4Shipwrecks](https://github.com/mvaldenegro/marine-debris-fls-datasets) for the benchmark SSS dataset  
- [Streamlit](https://streamlit.io/) for the rapid prototyping dashboard framework  
- National Institute of Ocean Technology (NIOT), Chennai, for domain guidance on SSS survey parameters

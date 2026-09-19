"""
core_engine/geotag_engine.py
============================
Hydrographic Geotagging Engine
--------------------------------
Translates pixel-domain bounding-box detections from side-scan sonar (SSS)
imagery into geo-referenced records with:
  - GPS coordinates derived from AUV origin + pixel offsets
  - Physical dimension estimates from bounding-box pixel extents
  - Structured CSV and GeoJSON exports for mission reporting

**Coordinate Reference**
AUV survey origin defaults to Bay of Bengal offshore waters:
  Lat  13.0827 °N  |  Lon  80.4500 °E
(approximately 20 km east of Chennai, beyond the 200 m isobath)

**Pixel Scale**
Nominal SSS pixel resolution: 0.05 m/pixel (assumes a 32 m swath across
640 px — typical for a 400 kHz hull-mounted SSS at survey altitude ~5 m AGL).

**Hazard ID Convention**
Format: AQ-HZ-<NNN>  (e.g. AQ-HZ-001)
AQ  = AquaScan AI system designator
HZ  = Hazard classification
NNN = Zero-padded sequential index per scan
"""

import json
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default AUV survey origin — Bay of Bengal offshore (deep water)
DEFAULT_BASE_LAT = 13.0827   # °N
DEFAULT_BASE_LON = 80.4500   # °E

# Degrees-per-pixel conversion: 1 pixel ≈ 0.05 m; 1° lat ≈ 111,320 m
# → 0.05 / 111320 ≈ 4.49e-7 °/px  (using 8e-6 for longitudinal slant correction)
DEG_PER_PX = 8e-6

# Physical scale: 0.05 m per pixel (SSS survey-grade nominal)
METERS_PER_PX = 0.05


# ---------------------------------------------------------------------------
# Core geotagging function
# ---------------------------------------------------------------------------

def parse_sonar_telemetry(
    detections,
    base_latitude: float = DEFAULT_BASE_LAT,
    base_longitude: float = DEFAULT_BASE_LON,
    heading: float = 90.0,
    bbox_dims: list = None
):
    """
    Translate pixel-coordinate detections into geo-referenced hazard records.

    Parameters
    ----------
    detections    : list[dict]  Each dict must contain:
                      x_center  (float) — horizontal pixel position
                      y_center  (float) — vertical pixel position
                      class_name (str)  — detection class label
                      confidence (float) — model confidence ∈ [0, 1]
                      area_px   (float) — bounding-box pixel area
    base_latitude : float   AUV/sensor origin latitude (°N).
    base_longitude: float   AUV/sensor origin longitude (°E).
    heading       : float   AUV heading in degrees (reserved for future
                            heading-aware coordinate rotation).
    bbox_dims     : list[tuple(int, int)] | None
                    Optional list of (width_px, height_px) bounding-box
                    pixel dimensions, one per detection, in the same order
                    as `detections`.  When provided, physical dimensions
                    are computed as:
                        Estimated_Length_m = height_px × 0.05 m/px
                        Estimated_Width_m  = width_px  × 0.05 m/px

    Returns
    -------
    list[dict]  Structured hazard records ready for CSV / GeoJSON export.
    """
    report = []
    img_center_px = 320   # assumed 640-px image centre

    for idx, det in enumerate(detections):
        x_center = det.get("x_center", img_center_px)
        y_center = det.get("y_center", img_center_px)

        # --- Georeferencing ---
        # Positive y offset → further from nadir track → southward (sonar convention)
        offset_lat = (y_center - img_center_px) * DEG_PER_PX
        offset_lon = (x_center - img_center_px) * DEG_PER_PX

        target_lat = base_latitude  + offset_lat
        target_lon = base_longitude + offset_lon

        # --- Physical dimensions from bounding box ---
        if bbox_dims and idx < len(bbox_dims):
            w_px, h_px = bbox_dims[idx]
            est_length_m = round(h_px * METERS_PER_PX, 2)
            est_width_m  = round(w_px * METERS_PER_PX, 2)
        else:
            # Fallback: approximate square root of pixel area
            area_px = det.get("area_px", 0)
            side_px = area_px ** 0.5
            est_length_m = round(side_px * METERS_PER_PX, 2)
            est_width_m  = est_length_m

        record = {
            "Hazard ID":           f"AQ-HZ-{idx + 1:03d}",
            "Classification":      det.get("class_name", "Unknown Hazard"),
            "Confidence_Score":    f"{det.get('confidence', 0.0) * 100:.2f}%",
            "Latitude":            round(target_lat, 6),
            "Longitude":           round(target_lon, 6),
            "Estimated_Length_m":  est_length_m,
            "Estimated_Width_m":   est_width_m,
            "Estimated_Area_sq_m": round(det.get("area_px", 100) * METERS_PER_PX ** 2, 2),
            "Status":              "Confirmed Anomaly",
        }
        report.append(record)

    return report


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_reports(
    report_data,
    output_csv: str  = "data/sonar_hazard_report.csv",
    output_json: str = "data/sonar_hazard_report.json"
):
    """
    Export structured hazard logs to CSV and GeoJSON formats.

    If `report_data` is empty, no files are written and a warning is printed —
    avoids polluting exports with dummy rows when no hazards are detected.
    """
    if not report_data:
        print("[INFO] No hazards detected — skipping empty export.")
        return

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(report_data)
    df.to_csv(output_csv, index=False)

    # Build GeoJSON FeatureCollection
    features = []
    for rec in report_data:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [rec["Longitude"], rec["Latitude"]]
            },
            "properties": {k: v for k, v in rec.items()
                           if k not in ("Latitude", "Longitude")}
        })
    geojson = {"type": "FeatureCollection", "features": features}

    with open(output_json, "w") as f:
        json.dump(geojson, f, indent=4)

    print("=" * 60)
    print(f"Geotagging Engine — {len(report_data)} hazard(s) exported")
    print(f"  CSV     : {output_csv}")
    print(f"  GeoJSON : {output_json}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy_detections = [
        {"x_center": 412, "y_center": 280, "class_name": "Shipwreck / Solid Hazard",
         "confidence": 0.884, "area_px": 1240},
        {"x_center": 150, "y_center": 510, "class_name": "Entangled Ghost Net",
         "confidence": 0.925, "area_px": 3400},
    ]
    dummy_bbox_dims = [(40, 31), (85, 40)]   # (width_px, height_px) per detection

    data = parse_sonar_telemetry(
        dummy_detections,
        base_latitude=DEFAULT_BASE_LAT,
        base_longitude=DEFAULT_BASE_LON,
        bbox_dims=dummy_bbox_dims
    )
    for rec in data:
        print(rec)
    export_reports(data)

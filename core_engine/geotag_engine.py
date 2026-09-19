"""
core_engine/geotag_engine.py
Translates pixel-domain bounding-box detections into geo-referenced hazard records.

Survey origin defaults to Bay of Bengal offshore waters (13.0827 °N, 80.4500 °E),
approximately 20 km east of Chennai beyond the 200 m isobath.

Pixel scale: 0.05 m/px — nominal for a 400 kHz SSS at ~5 m altitude, 32 m swath,
640 px across-track resolution.

Hazard ID format: AQ-HZ-NNN (AquaScan system designator, zero-padded per scan).
"""

import json
import pandas as pd
from pathlib import Path

DEFAULT_BASE_LAT  = 13.0827
DEFAULT_BASE_LON  = 80.4500
DEG_PER_PX        = 8e-6     # ~0.05 m/px converted to degrees at survey latitude
METERS_PER_PX     = 0.05


def parse_sonar_telemetry(
    detections,
    base_latitude:  float = DEFAULT_BASE_LAT,
    base_longitude: float = DEFAULT_BASE_LON,
    heading:        float = 90.0,
    bbox_dims:      list  = None,
) -> list:
    """
    Convert pixel-space detections to geo-referenced hazard records.

    Parameters
    ----------
    detections     : list[dict]  Each item must contain x_center, y_center,
                                 class_name, confidence, area_px.
    base_latitude  : float       AUV origin latitude (°N).
    base_longitude : float       AUV origin longitude (°E).
    heading        : float       AUV heading in degrees (reserved for future
                                 heading-aware coordinate rotation).
    bbox_dims      : list[(int, int)] | None
                                 Optional (width_px, height_px) per detection.
                                 When provided, physical dimensions are computed
                                 as length = h_px × 0.05 m, width = w_px × 0.05 m.

    Returns
    -------
    list[dict]  Structured hazard records ready for CSV / GeoJSON export.
    """
    report = []
    img_center = 320

    for idx, det in enumerate(detections):
        cx = det.get("x_center", img_center)
        cy = det.get("y_center", img_center)

        lat = base_latitude  + (cy - img_center) * DEG_PER_PX
        lon = base_longitude + (cx - img_center) * DEG_PER_PX

        if bbox_dims and idx < len(bbox_dims):
            w_px, h_px = bbox_dims[idx]
            length_m = round(h_px * METERS_PER_PX, 2)
            width_m  = round(w_px * METERS_PER_PX, 2)
        else:
            side = det.get("area_px", 0) ** 0.5
            length_m = width_m = round(side * METERS_PER_PX, 2)

        report.append({
            "Hazard ID":           f"AQ-HZ-{idx + 1:03d}",
            "Classification":      det.get("class_name", "Unknown"),
            "Confidence_Score":    f"{det.get('confidence', 0.0) * 100:.2f}%",
            "Latitude":            round(lat, 6),
            "Longitude":           round(lon, 6),
            "Estimated_Length_m":  length_m,
            "Estimated_Width_m":   width_m,
            "Estimated_Area_sq_m": round(det.get("area_px", 0) * METERS_PER_PX ** 2, 2),
            "Status":              "Confirmed Anomaly",
        })

    return report


def export_reports(
    report_data,
    output_csv:  str = "data/sonar_hazard_report.csv",
    output_json: str = "data/sonar_hazard_report.json",
) -> None:
    """
    Write hazard records to CSV and GeoJSON. Silently skips when report_data is empty
    to avoid polluting exports with zero-detection placeholder rows.

    Parameters
    ----------
    report_data  : list[dict]  Output of parse_sonar_telemetry.
    output_csv   : str         CSV output path.
    output_json  : str         GeoJSON FeatureCollection output path.
    """
    if not report_data:
        print("[INFO] No hazards detected — skipping empty export.")
        return

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_data).to_csv(output_csv, index=False)

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["Longitude"], r["Latitude"]]},
            "properties": {k: v for k, v in r.items() if k not in ("Latitude", "Longitude")},
        }
        for r in report_data
    ]
    with open(output_json, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=4)

    print(f"Exported {len(report_data)} hazard(s) → {output_csv}, {output_json}")


if __name__ == "__main__":
    sample = [
        {"x_center": 412, "y_center": 280, "class_name": "Submerged Solid Hazard / Shipwreck",
         "confidence": 0.884, "area_px": 1240},
        {"x_center": 150, "y_center": 510, "class_name": "Entangled Ghost Net",
         "confidence": 0.925, "area_px": 3400},
    ]
    records = parse_sonar_telemetry(sample, bbox_dims=[(40, 31), (85, 40)])
    for r in records:
        print(r)
    export_reports(records)

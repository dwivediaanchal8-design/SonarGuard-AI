import json
import pandas as pd
from pathlib import Path

def parse_sonar_telemetry(detections, base_latitude=13.0827, base_longitude=80.2707, heading=90.0):
    """
    Translates pixel coordinates from side-scan sonar image detections
    into geo-referenced GPS coordinates (Latitude, Longitude) and formats structured hazard logs.
    """
    report = []
    
    for idx, det in enumerate(detections):
        x_center = det.get("x_center", 320)
        y_center = det.get("y_center", 320)
        
        offset_lat = (y_center - 320) * 0.000008
        offset_lon = (x_center - 320) * 0.000008
        
        target_lat = base_latitude + offset_lat
        target_lon = base_longitude + offset_lon

        report.append({
            "Target_ID": f"SSS_HAZARD_{idx + 1:03d}",
            "Classification": det.get("class_name", "Unknown Hazard"),
            "Confidence_Score": f"{det.get('confidence', 0.0) * 100:.2f}%",
            "Latitude": round(target_lat, 6),
            "Longitude": round(target_lon, 6),
            "Estimated_Area_sq_m": round(det.get("area_px", 100) * 0.04, 2),
            "Status": "Confirmed Anomaly"
        })

    return report

def export_reports(report_data, output_csv="data/sonar_hazard_report.csv", output_json="data/sonar_hazard_report.json"):
    """Exports structured hazard logs to CSV and JSON formats."""
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(report_data)
    df.to_csv(output_csv, index=False)
    
    with open(output_json, "w") as f:
        json.dump(report_data, f, indent=4)
        
    print("=" * 60)
    print("Geotagging Engine Processed Hazards Successfully")
    print(f"Structured CSV: {output_csv}")
    print(f"Structured JSON: {output_json}")
    print("=" * 60)

if __name__ == "__main__":
    dummy_detections = [
        {"x_center": 412, "y_center": 280, "class_name": "Ghost Net", "confidence": 0.884, "area_px": 1240},
        {"x_center": 150, "y_center": 510, "class_name": "Shipwreck", "confidence": 0.925, "area_px": 3400}
    ]
    data = parse_sonar_telemetry(dummy_detections)
    export_reports(data)

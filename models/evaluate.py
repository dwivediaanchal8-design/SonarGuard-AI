import os
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

def run_model_evaluation(
    model_path=None,
    data_config="models/marine_multiclass.yaml",
    project="models",
    name="evaluation_results"
):
    """
    Evaluates trained YOLO model on marine sonar validation split and calculates mAP, Precision, and Recall metrics.
    """
    if model_path is None:
        model_paths = [
            "models/best.pt",
            "models/multiclass_sonar_v2/weights/best.pt",
            "model/best.pt"
        ]
        weights_path = None
        for p in model_paths:
            if os.path.exists(p):
                weights_path = p
                break
    else:
        weights_path = model_path if os.path.exists(model_path) else None

    if not weights_path:
        print("❌ Error: best.pt weights file not found! Please check model path.")
        return None

    config_path = data_config if os.path.exists(data_config) else "marine_multiclass.yaml"

    print(f"🚀 Loading trained model weights from: {weights_path}")
    model = YOLO(weights_path)

    print("📊 Evaluating model on marine sonar validation split...")
    metrics = model.val(
        data=config_path,
        split="val",
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.45,
        plots=True,
        project=project,
        name=name
    )

    mp = metrics.box.mp * 100         # Mean Precision
    mr = metrics.box.mr * 100         # Mean Recall
    map50 = metrics.box.map50 * 100   # mAP @ IoU 0.50
    map_all = metrics.box.map * 100   # mAP @ IoU 0.50:0.95

    print("\n" + "="*50)
    print("🎯 OFFICIAL VALIDATION METRICS")
    print("="*50)
    print(f"✅ Mean Precision (P)       : {mp:.2f}%")
    print(f"✅ Mean Recall (R)          : {mr:.2f}%")
    print(f"✅ mAP @ 0.50               : {map50:.2f}%")
    print(f"✅ mAP @ [0.50:0.95]        : {map_all:.2f}%")
    print("="*50)

    class_names = ["Shipwreck / Solid Hazard", "Ghost Net / Fishing Gear", "Submerged Pipe / Cable"]
    print("\n📋 Class-wise AP@0.50 breakdown:")
    for i, cls_name in enumerate(class_names):
        if i < len(metrics.box.ap50):
            print(f" - {cls_name:30s}: {metrics.box.ap50[i]*100:.2f}%")

    print(f"\n📁 Visual graphs saved at: {project}/{name}/")
    return metrics

if __name__ == "__main__":
    run_model_evaluation()

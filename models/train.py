import os
from pathlib import Path
from ultralytics import YOLO

def start_training(
    base_model="models/best.pt",
    data_config="models/marine_multiclass.yaml",
    epochs=15,
    imgsz=640,
    batch=16,
    workers=2,
    project="models",
    name="multiclass_sonar_v2"
):
    """
    Trains the multi-class YOLO model on side-scan sonar datasets (Shipwreck, Ghost Net, Pipe/Cable).
    """
    model_path = base_model if os.path.exists(base_model) else "yolov8n.pt"
    config_path = data_config if os.path.exists(data_config) else "marine_multiclass.yaml"
    
    print(f"🚀 Initializing YOLO training with weights: {model_path} and config: {config_path}")
    model = YOLO(model_path)
    
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project=project,
        save=True,
        workers=workers
    )
    return results

if __name__ == "__main__":
    start_training()

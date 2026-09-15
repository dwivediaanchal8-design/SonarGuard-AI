import os
import cv2
import numpy as np
from pathlib import Path

def inject_debris(img):
    """
    Injects synthetic acoustic signatures of Subsea Pipelines (Class 1) and Ghost Nets (Class 2)
    onto side-scan sonar image tiles alongside existing Shipwreck anomalies (Class 0).
    """
    h, w = img.shape[:2]
    annotations = []
    
    # Class 1: Submerged Pipe / Cable / Infrastructure
    pipe_x = np.random.randint(50, max(51, w - 180))
    pipe_y = np.random.randint(50, max(51, h - 180))
    pipe_len = np.random.randint(80, 140)
    pipe_thick = np.random.randint(6, 12)
    
    cv2.line(img, (pipe_x, pipe_y), (pipe_x + pipe_len, pipe_y + 20), (240, 240, 240), pipe_thick)
    cv2.line(img, (pipe_x, pipe_y + pipe_thick), (pipe_x + pipe_len, pipe_y + 20 + pipe_thick), (10, 10, 10), pipe_thick + 2)
    
    annotations.append(f"1 {(pipe_x + pipe_len/2)/w:.6f} {(pipe_y + 10)/h:.6f} {pipe_len/w:.6f} {(pipe_thick*2)/h:.6f}\n")

    # Class 2: Entangled Ghost Net (Web-like acoustic texture)
    net_x = np.random.randint(50, max(51, w - 150))
    net_y = np.random.randint(50, max(51, h - 150))
    net_w = np.random.randint(60, 120)
    net_h = np.random.randint(60, 120)
    
    # Draw acoustic mesh pattern
    cv2.ellipse(img, (net_x + net_w//2, net_y + net_h//2), (net_w//2, net_h//2), 0, 0, 360, (180, 180, 180), 2)
    for _ in range(5):
        pt1 = (net_x + np.random.randint(0, net_w), net_y + np.random.randint(0, net_h))
        pt2 = (net_x + np.random.randint(0, net_w), net_y + np.random.randint(0, net_h))
        cv2.line(img, pt1, pt2, (210, 210, 210), 1)
        
    # Acoustic shadow behind net
    cv2.rectangle(img, (net_x + net_w, net_y), (min(w-1, net_x + net_w + 30), net_y + net_h), (20, 20, 20), -1)
    
    annotations.append(f"2 {(net_x + net_w/2)/w:.6f} {(net_y + net_h/2)/h:.6f} {net_w/w:.6f} {net_h/h:.6f}\n")
    
    return img, annotations

def generate_multiclass_debris_dataset(
    src_img_dir="data/yolo_dataset/images/train",
    out_dir="data/multi_debris_dataset"
):
    """
    Generates multi-class sonar hazard dataset (Shipwrecks, Ghost Nets, Pipes)
    from base tile scans.
    """
    src_path = Path(src_img_dir)
    out_path = Path(out_dir)
    
    out_img_train = out_path / "images" / "train"
    out_lbl_train = out_path / "labels" / "train"
    val_img_train = out_path / "images" / "val"
    val_lbl_train = out_path / "labels" / "val"

    for p in [out_img_train, out_lbl_train, val_img_train, val_lbl_train]:
        p.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        print(f"Warning: Source image path {src_path} does not exist.")
        return

    files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg'))]
    split_idx = int(len(files) * 0.8)
    
    for idx, fname in enumerate(files):
        img_path = src_path / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        mod_img, annots = inject_debris(img)
        
        target_img = out_img_train if idx < split_idx else val_img_train
        target_lbl = out_lbl_train if idx < split_idx else val_lbl_train
        
        cv2.imwrite(str(target_img / fname), mod_img)
        lbl_name = Path(fname).stem + ".txt"
        
        orig_lbl = Path("data/yolo_dataset/labels/train") / lbl_name
        orig_content = []
        if orig_lbl.exists():
            with open(orig_lbl, "r") as f:
                orig_content = f.readlines()
                
        with open(target_lbl / lbl_name, "w") as f:
            f.writelines(orig_content + annots)
            
    print(f"Dataset generated: Multi-class debris (Shipwrecks, Ghost Nets, Pipes) in {out_dir}")

if __name__ == "__main__":
    generate_multiclass_debris_dataset()

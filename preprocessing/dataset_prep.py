import os
import sys
import shutil
import random
import zipfile
import io
import requests
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure workspace root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def clean_dataset_caches(dataset_dirs=None):
    """
    Scans dataset directories and removes stale YOLO .cache files.
    """
    if dataset_dirs is None:
        dataset_dirs = ["data/multi_debris_dataset", "data/yolo_dataset", "data/full_sonar_dataset"]
    
    removed_count = 0
    for d in dataset_dirs:
        p = Path(d)
        if p.exists():
            for cache_file in p.glob("**/*.cache"):
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"Warning: Could not remove cache file {cache_file}: {e}")
    print(f"[CLEANUP] Cleaned up {removed_count} stale dataset .cache file(s).")
    return removed_count

def harmonize_multiclass_datasets(
    shipwreck_dir="data/AI4Shipwrecks",
    pipeline_dir="data/subpipe2line",
    output_dir="data/multi_debris_dataset"
):
    """
    Harmonizes multi-source side-scan sonar datasets (AI4Shipwrecks, subpipe2line pipeline dataset,
    Ghost Nets, Marine Debris) into a unified 4-class YOLO training format:
    0: Shipwreck
    1: Subsea_Pipeline
    2: Ghost_Net
    3: Marine_Debris
    """
    clean_dataset_caches()
    out_path = Path(output_dir)
    
    out_img_train = out_path / "images" / "train"
    out_lbl_train = out_path / "labels" / "train"
    out_img_val = out_path / "images" / "val"
    out_lbl_val = out_path / "labels" / "val"

    for p in [out_img_train, out_lbl_train, out_img_val, out_lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"HARMONIZING MULTI-DATASET PIPELINE into {output_dir}")
    print("Classes: 0=Shipwreck | 1=Subsea_Pipeline | 2=Ghost_Net | 3=Marine_Debris")
    print("=" * 60)

    # 1. Integrate subpipe2line pipeline dataset if present
    pipe_path = Path(pipeline_dir)
    pipeline_count = 0
    if pipe_path.exists():
        pipe_img_dir = pipe_path / "images"
        pipe_lbl_dir = pipe_path / "labels"
        if pipe_img_dir.exists() and pipe_lbl_dir.exists():
            pipe_files = [f for f in os.listdir(pipe_img_dir) if f.lower().endswith(('.png', '.jpg'))]
            split_idx = int(len(pipe_files) * 0.8)
            for idx, fname in enumerate(pipe_files):
                stem = Path(fname).stem
                lbl_name = f"{stem}.txt"
                
                target_img_dir = out_img_train if idx < split_idx else out_img_val
                target_lbl_dir = out_lbl_train if idx < split_idx else out_lbl_val
                
                shutil.copy(pipe_img_dir / fname, target_img_dir / f"subpipe_{fname}")
                
                src_lbl_file = pipe_lbl_dir / lbl_name
                dst_lbl_file = target_lbl_dir / f"subpipe_{stem}.txt"
                
                if src_lbl_file.exists():
                    # Remap labels to Class 1 (Subsea_Pipeline)
                    new_lines = []
                    with open(src_lbl_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                parts[0] = "1"  # Remap class index to 1 (Subsea_Pipeline)
                                new_lines.append(" ".join(parts) + "\n")
                    with open(dst_lbl_file, "w") as f:
                        f.writelines(new_lines)
                else:
                    with open(dst_lbl_file, "w") as f:
                        pass
                pipeline_count += 1
            print(f"[SUCCESS] Integrated {pipeline_count} Subsea Pipeline sonar scans from {pipeline_dir}")
    else:
        print(f"[INFO] {pipeline_dir} dataset directory not found. Prepared pipeline structure ready for subpipe2line integration.")

    # 2. Trigger multiclass synthetic generator to refresh shipwrecks, ghost nets, and pipe signatures
    from preprocessing.debris_generator import generate_multiclass_debris_dataset
    generate_multiclass_debris_dataset(out_dir=output_dir)

    train_samples = len(os.listdir(out_img_train)) if out_img_train.exists() else 0
    val_samples = len(os.listdir(out_img_val)) if out_img_val.exists() else 0

    print("=" * 60)
    print("Harmonized Multiclass Dataset Ready")
    print(f"Training Samples  : {train_samples}")
    print(f"Validation Samples: {val_samples}")
    print("=" * 60)

def split_data(base_dir="data/yolo_dataset", val_ratio=0.20, seed=42):
    """
    Performs train/validation split on YOLO formatted dataset.
    """
    base_path = Path(base_dir)
    train_img_dir = base_path / "images" / "train"
    train_lbl_dir = base_path / "labels" / "train"

    val_img_dir = base_path / "images" / "val"
    val_lbl_dir = base_path / "labels" / "val"

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not train_img_dir.exists():
        print(f"Directory {train_img_dir} not found.")
        return

    image_files = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.png', '.jpg'))]
    random.seed(seed)
    random.shuffle(image_files)

    val_count = int(len(image_files) * val_ratio)
    val_files = image_files[:val_count]

    moved_count = 0
    for file_name in val_files:
        stem = Path(file_name).stem
        lbl_name = f"{stem}.txt"

        src_img = train_img_dir / file_name
        dst_img = val_img_dir / file_name

        src_lbl = train_lbl_dir / lbl_name
        dst_lbl = val_lbl_dir / lbl_name

        if src_img.exists() and src_lbl.exists():
            shutil.move(str(src_img), str(dst_img))
            shutil.move(str(src_lbl), str(dst_lbl))
            moved_count += 1

    remaining_train = len(os.listdir(train_img_dir)) if train_img_dir.exists() else 0
    total_val = len(os.listdir(val_img_dir)) if val_img_dir.exists() else 0

    print("=" * 60)
    print("Dataset Split Completed")
    print(f"Training Samples: {remaining_train}")
    print(f"Validation Samples: {total_val}")
    print("=" * 60)

def validate_split(split_name="train", dataset_dir="data/AI4Shipwrecks"):
    """
    Validates dataset alignment between image tiles and segmentation mask labels.
    """
    base_path = Path(dataset_dir)
    image_folder = base_path / split_name / "images"
    label_folder = base_path / split_name / "labels"

    if not image_folder.exists() or not label_folder.exists():
        print(f"Validation Error: Folders do not exist: {image_folder} or {label_folder}")
        return

    images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".png")])
    labels = sorted([f for f in os.listdir(label_folder) if f.lower().endswith(".png")])

    image_set = set(images)
    label_set = set(labels)

    missing_labels = image_set - label_set
    missing_images = label_set - image_set

    positive_images = 0
    empty_images = 0
    size_mismatch = 0

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} DATASET VALIDATION ({dataset_dir})")
    print("=" * 60)
    print("Total Images:", len(images))
    print("Total Labels:", len(labels))
    print("Missing Labels:", len(missing_labels))
    print("Missing Images:", len(missing_images))

    for file in images:
        if file not in label_set:
            continue

        image_path = image_folder / file
        label_path = label_folder / file

        image = Image.open(image_path)
        label = Image.open(label_path)

        if image.size != label.size:
            size_mismatch += 1

        label_array = np.array(label.convert("L"))
        if np.max(label_array) > 0:
            positive_images += 1
        else:
            empty_images += 1

    print("Positive / Anomaly Images:", positive_images)
    print("Empty / Normal Images:", empty_images)
    print("Image-Label Size Mismatch:", size_mismatch)

def prepare_full_sonar(src_dir="data/AI4Shipwrecks", dest_dir="data/full_sonar_dataset", split_ratio=0.8):
    """
    Copies and organizes full sonar scans into train and val splits.
    """
    src_base = Path(src_dir)
    src_img = src_base / "train" / "images"
    src_lbl = src_base / "train" / "labels"

    dest_base = Path(dest_dir)
    dest_img_train = dest_base / "images" / "train"
    dest_img_val = dest_base / "images" / "val"
    dest_lbl_train = dest_base / "labels" / "train"
    dest_lbl_val = dest_base / "labels" / "val"

    for p in [dest_img_train, dest_img_val, dest_lbl_train, dest_lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    if not src_img.exists():
        print(f"Error: Path {src_img} does not exist.")
        return

    images = sorted([f for f in os.listdir(src_img) if f.lower().endswith(('.png', '.jpg'))])
    split_point = int(len(images) * split_ratio)

    for idx, fname in enumerate(images):
        stem = Path(fname).stem
        lbl_name = f"{stem}.txt"
        src_lbl_file = src_lbl / lbl_name
        
        target_img_dir = dest_img_train if idx < split_point else dest_img_val
        target_lbl_dir = dest_lbl_train if idx < split_point else dest_lbl_val
        
        shutil.copy(src_img / fname, target_img_dir / fname)
        
        if src_lbl_file.exists():
            shutil.copy(src_lbl_file, target_lbl_dir / lbl_name)
        else:
            with open(target_lbl_dir / lbl_name, 'w') as f:
                pass

    print(f"Dataset Ready: {len(images)} sonar scans organized in {dest_dir}.")

def fetch_debris_dataset(url="https://universe.roboflow.com/ds/8QJ9z3XoVb?key=z31x4QoR0j", dest_dir="data/marine_debris_dataset"):
    """
    Downloads public marine debris dataset from Roboflow repository into destination folder.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    print("Connecting to Marine Debris Dataset repository...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 200:
        print("Download in progress... Extracting directly into memory...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(dest_path)
        print("Success: Dataset extracted directly to:", dest_path)
    else:
        print(f"Server response code: {response.status_code}")

if __name__ == "__main__":
    harmonize_multiclass_datasets()


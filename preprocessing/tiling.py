import os
import cv2
import numpy as np
from pathlib import Path

def generate_tiles(
    base_dir="data/AI4Shipwrecks",
    output_dir="data/yolo_dataset",
    tile_size=640,
    stride=480,
    min_area=250
):
    """
    Splits large side-scan sonar waterfall images and mask labels into 640x640 sub-tiles with YOLO annotations.
    """
    base_path = Path(base_dir)
    train_img_dir = base_path / "train" / "images"
    train_mask_dir = base_path / "train" / "labels"

    out_path = Path(output_dir)
    out_img_dir = out_path / "images" / "train"
    out_lbl_dir = out_path / "labels" / "train"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not train_mask_dir.exists():
        print(f"Warning: Mask directory {train_mask_dir} does not exist.")
        return 0

    mask_files = [f for f in os.listdir(train_mask_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))]
    total_saved_tiles = 0

    for mask_file in mask_files:
        mask_path = train_mask_dir / mask_file
        img_path = train_img_dir / mask_file
        
        if not img_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(img_path))

        if mask is None or image is None:
            continue

        h, w = mask.shape
        tile_idx = 0

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                mask_tile = mask[y:y + tile_size, x:x + tile_size]
                
                if np.count_nonzero(mask_tile) == 0:
                    continue

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_tile, connectivity=8)
                tile_yolo_lines = []

                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < min_area:
                        continue

                    bx = stats[i, cv2.CC_STAT_LEFT]
                    by = stats[i, cv2.CC_STAT_TOP]
                    bw = stats[i, cv2.CC_STAT_WIDTH]
                    bh = stats[i, cv2.CC_STAT_HEIGHT]

                    xc = (bx + bw / 2.0) / tile_size
                    yc = (by + bh / 2.0) / tile_size
                    nw = bw / tile_size
                    nh = bh / tile_size

                    tile_yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

                if len(tile_yolo_lines) > 0:
                    tile_name = f"{Path(mask_file).stem}_tile_{tile_idx}"
                    img_tile = image[y:y + tile_size, x:x + tile_size]
                    cv2.imwrite(str(out_img_dir / f"{tile_name}.png"), img_tile)

                    with open(out_lbl_dir / f"{tile_name}.txt", "w") as f:
                        f.writelines(tile_yolo_lines)

                    total_saved_tiles += 1
                    tile_idx += 1

    print("=" * 60)
    print("Tiling Process Completed")
    print(f"Total Valid Tiles Created: {total_saved_tiles}")
    print(f"Destination: {output_dir}")
    print("=" * 60)
    return total_saved_tiles

if __name__ == "__main__":
    generate_tiles()

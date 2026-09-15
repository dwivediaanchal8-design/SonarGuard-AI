import cv2
import os
import numpy as np
from pathlib import Path

def generate_extended_sonar_video(
    source_dir="data/AI4Shipwrecks/train/images",
    output_dir="data/real_test_samples",
    output_filename="real_sonar_auv_stream.mp4"
):
    """
    Stitches side-scan sonar image tiles into a continuous waterfall video stream for AUV simulation.
    """
    src_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sonar_files = sorted([f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not sonar_files:
        print(f"Warning: No sonar images found in {source_dir}")
        return None

    frame_size = 640
    all_strips = []

    for fname in sonar_files[:6]:
        img_file = src_path / fname
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w, _ = img.shape
        scale = frame_size / float(w)
        resized = cv2.resize(img, (frame_size, int(h * scale)))
        all_strips.append(resized)

    if not all_strips:
        print("Error: Could not read sonar image tiles.")
        return None

    full_waterfall = np.vstack(all_strips)
    total_h, _, _ = full_waterfall.shape
    print(f"Total Stitched Waterfall Height: {total_h}px")

    video_file = out_path / output_filename
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20.0
    out = cv2.VideoWriter(str(video_file), fourcc, fps, (frame_size, frame_size))

    step = 3
    total_frames = 0
    for y in range(0, total_h - frame_size, step):
        frame = full_waterfall[y:y + frame_size, 0:frame_size]
        out.write(frame)
        total_frames += 1

    out.release()
    duration_sec = total_frames / fps
    print("=" * 60)
    print("Extended Real Sonar Video Rendered Successfully!")
    print(f"Total Frames: {total_frames} | Duration: {duration_sec:.1f}s")
    print(f"Saved to: {video_file}")
    print("=" * 60)
    return str(video_file)

if __name__ == "__main__":
    generate_extended_sonar_video()

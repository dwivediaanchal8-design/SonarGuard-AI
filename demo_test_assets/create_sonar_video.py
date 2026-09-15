import os
import cv2
import numpy as np
from pathlib import Path

def create_demo_sonar_video(
    src_dir="data/AI4Shipwrecks/train/images",
    output_path="demo_test_assets/test_sonar_stream.mp4",
    num_frames=30,
    fps=10
):
    """
    Sequences consecutive sonar acoustic pings / waterfall tiles into a demo AUV video stream.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    src_path = Path(src_dir)
    images = sorted(list(src_path.glob("*.png")) + list(src_path.glob("*.jpg")))
    
    if not images:
        # Fallback to demo_test_assets
        images = sorted(list(Path("demo_test_assets").glob("*.png")))

    if not images:
        print("Error: No images found to create video.")
        return None

    frame_size = 640
    frames = []

    # Read base images and resize to 640x640
    base_imgs = []
    for img_p in images[:5]:
        img = cv2.imread(str(img_p))
        if img is not None:
            resized = cv2.resize(img, (frame_size, frame_size))
            base_imgs.append(resized)

    if not base_imgs:
        print("Error: Could not read base images.")
        return None

    # Stack base images vertically to simulate waterfall sonar swath
    waterfall = np.vstack(base_imgs)
    h_total, w_total, _ = waterfall.shape

    # Extract 30 sliding-window ping frames
    step_y = max(1, (h_total - frame_size) // num_frames)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_file), fourcc, float(fps), (frame_size, frame_size))

    for i in range(num_frames):
        y_start = min(i * step_y, h_total - frame_size)
        ping_frame = waterfall[y_start:y_start + frame_size, 0:frame_size].copy()
        
        # Add acoustic ping line scan overlay effect
        ping_line_y = (i * 20) % frame_size
        cv2.line(ping_frame, (0, ping_line_y), (frame_size, ping_line_y), (0, 242, 254), 1)
        
        out.write(ping_frame)

    out.release()
    print(f"[SUCCESS] Demo Sonar Stream Video created: {out_file} ({num_frames} frames @ {fps} FPS)")
    return str(out_file)

if __name__ == "__main__":
    create_demo_sonar_video()

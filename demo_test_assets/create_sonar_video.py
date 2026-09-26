"""
create_sonar_video.py
Generates demo_test_assets/test_sonar_stream.mp4 in browser-compatible H.264.

Strategy (priority order):
  1. imageio + ffmpeg-imageio  →  true H.264 output (best, always works)
  2. OpenCV avc1/H264          →  H.264 if openh264 DLL is present
  3. OpenCV mp4v               →  MPEG-4 fallback (Streamlit still serves it)
"""
import os
import cv2
import numpy as np
from pathlib import Path


# ── Image helpers ──────────────────────────────────────────────────────────────

def _collect_images(src_dir: str, limit: int = 8):
    """Return up to `limit` BGR frames from the source directory (or fallbacks)."""
    candidates = [
        src_dir,
        "data/multi_debris_dataset/images/train",
        "data/full_sonar_dataset/images/train",
        "data/yolo_dataset/images/val",
        "demo_test_assets",
    ]
    for d in candidates:
        p = Path(d)
        imgs = sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")))
        if imgs:
            frames = []
            for ip in imgs[:limit]:
                img = cv2.imread(str(ip))
                if img is not None:
                    frames.append(cv2.resize(img, (640, 640)))
            if frames:
                print(f"[INFO] Source images from: {d}  ({len(frames)} tiles)")
                return frames
    return []


def _synthetic_waterfall(n_tiles: int = 4, size: int = 640):
    """Generate realistic-looking synthetic sonar grayscale tiles."""
    rng = np.random.default_rng(42)
    tiles = []
    for seed in range(n_tiles):
        tile = np.zeros((size, size, 3), dtype=np.uint8)
        # Dark seabed gradient (bottom of water column is brighter due to return)
        for row in range(size):
            val = int(25 + 70 * (row / size))
            tile[row, :] = (val, val, val)
        # Speckle noise
        noise = rng.integers(0, 20, (size, size), dtype=np.uint8)
        for c in range(3):
            tile[:, :, c] = np.clip(tile[:, :, c].astype(np.int16) + noise, 0, 255)
        # Bright nadir stripe
        cx = size // 2
        tile[:, cx - 8:cx + 8] = np.clip(
            tile[:, cx - 8:cx + 8].astype(np.int16) + 100, 0, 255)
        # Shadow blobs (simulated debris signatures)
        for _ in range(rng.integers(3, 7)):
            bx = int(rng.integers(40, size - 100))
            by = int(rng.integers(40, size - 70))
            bw = int(rng.integers(30, 80))
            bh = int(rng.integers(20, 50))
            tile[by:by + bh, bx:bx + bw] = np.clip(
                tile[by:by + bh, bx:bx + bw].astype(np.int16) - 35, 0, 255)
        tiles.append(tile)
    print("[INFO] Using synthetic sonar waterfall (no source images found)")
    return tiles


def _build_frames(base_imgs, num_frames: int = 60, size: int = 640):
    """Slide a window down the stacked waterfall to produce ping frames."""
    waterfall = np.vstack(base_imgs)
    h_total = waterfall.shape[0]
    step_y = max(1, (h_total - size) // num_frames)

    frames = []
    for i in range(num_frames):
        y_start = min(i * step_y, h_total - size)
        frame = waterfall[y_start:y_start + size, 0:size].copy()
        # Ping-line scan overlay
        py = (i * (size // num_frames)) % size
        cv2.line(frame, (0, py), (size, py), (0, 229, 255), 1)
        # Frame counter
        cv2.putText(frame, f"AUV PING {i+1:03d}/{num_frames}", (10, 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 229, 255), 1, cv2.LINE_AA)
        frames.append(frame)
    return frames


# ── Encoder strategies ────────────────────────────────────────────────────────

def _encode_imageio(frames, out_path: Path, fps: int) -> bool:
    """Use imageio_ffmpeg bundled binary to produce true H.264 via subprocess."""
    try:
        import imageio_ffmpeg
        import subprocess

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        W, H = frames[0].shape[1], frames[0].shape[0]

        # Write intermediate mp4v file via OpenCV
        tmp_path = out_path.parent / "_tmp_raw.mp4"
        fcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fcc, float(fps), (W, H))
        if not writer.isOpened():
            return False
        for f in frames:
            writer.write(f)
        writer.release()

        # Transcode to H.264 with bundled ffmpeg
        proc = subprocess.run(
            [
                ffmpeg_exe, "-y",
                "-i", str(tmp_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out_path),
            ],
            capture_output=True,
        )
        try:
            tmp_path.unlink()
        except Exception:
            pass

        if proc.returncode == 0 and out_path.exists():
            print("[CODEC] imageio_ffmpeg + libx264 = true H.264 MP4")
            return True
        print(f"[WARN] ffmpeg transcode failed (rc={proc.returncode})")
        return False
    except Exception as e:
        print(f"[WARN] imageio_ffmpeg encode failed: {e}")
        return False


def _encode_opencv(frames, out_path: Path, fps: int) -> bool:
    """Try OpenCV with avc1 / mp4v fallback."""
    size = frames[0].shape[1], frames[0].shape[0]
    for tag in ("avc1", "mp4v", "XVID"):
        fcc = cv2.VideoWriter_fourcc(*tag)
        writer = cv2.VideoWriter(str(out_path), fcc, float(fps), size)
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
            print(f"[CODEC] OpenCV fourcc={tag}")
            return True
        writer.release()
    return False


# ── Main entry ─────────────────────────────────────────────────────────────────

def create_demo_sonar_video(
    src_dir: str = "data/AI4Shipwrecks/train/images",
    output_path: str = "demo_test_assets/test_sonar_stream.mp4",
    num_frames: int = 60,
    fps: int = 12,
):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    base_imgs = _collect_images(src_dir) or _synthetic_waterfall()
    frames = _build_frames(base_imgs, num_frames=num_frames)

    # Remove stale file so VideoWriter doesn't append
    if out_file.exists():
        out_file.unlink()

    # Try imageio first (true H.264), fall back to OpenCV
    ok = _encode_imageio(frames, out_file, fps)
    if not ok or not out_file.exists():
        ok = _encode_opencv(frames, out_file, fps)

    if not ok or not out_file.exists():
        print("[ERROR] All encoders failed.")
        return None

    # Verify
    cap = cv2.VideoCapture(str(out_file))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc_str = fcc_val.to_bytes(4, "little").decode("ascii", errors="replace")
    cap.release()

    kb = out_file.stat().st_size // 1024
    print(f"[SUCCESS] {out_file}  |  {n} frames @ {fps} FPS  |  codec={fcc_str}  |  {kb} KB")
    return str(out_file)


if __name__ == "__main__":
    create_demo_sonar_video()

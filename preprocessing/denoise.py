import cv2
import numpy as np
from pathlib import Path


def apply_clahe(img_gray, clip_limit=1.5, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)


def lee_filter(img_gray, window_size=7):
    img_float = img_gray.astype(np.float32)
    mean = cv2.boxFilter(img_float, -1, (window_size, window_size))
    mean_sq = cv2.boxFilter(img_float ** 2, -1, (window_size, window_size))
    var = np.maximum(mean_sq - mean ** 2, 0.0)

    noise_var = float(np.mean(var))
    if noise_var < 1e-6:
        return img_gray

    k = var / (var + noise_var)
    filtered = mean + k * (img_float - mean)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def preprocess_sonar(img_bgr, clahe=False, denoise=False):
    """Mild bilateral denoise only — preserves natural sonar acoustic texture.
    Lee filter and CLAHE are available via keyword args but off by default
    to prevent over-exposure of the grey seabed / shadow bands.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
    # Mild edge-preserving smoothing — does NOT alter tonal range
    processed = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)
    if denoise:
        processed = lee_filter(processed)
    if clahe:
        processed = apply_clahe(processed, clip_limit=1.5, tile_grid_size=(8, 8))
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)


def apply_clahe_to_file(image_path, output_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), apply_clahe(lee_filter(img), clip_limit=1.5, tile_grid_size=(8, 8)))
    return str(output_path)


if __name__ == "__main__":
    sample_dir = Path("data/yolo_dataset/images/val")
    imgs = list(sample_dir.glob("*.png")) if sample_dir.exists() else []

    if imgs:
        out = Path("data/denoised_output") / f"lee_{imgs[0].name}"
        apply_clahe_to_file(imgs[0], out)
        print(f"Lee + CLAHE -> {out}")
    else:
        rng = np.random.default_rng(42)
        noisy = np.clip(
            rng.random((640, 640)) * 255 * rng.gamma(1.5, 1.0, (640, 640)),
            0, 255
        ).astype(np.uint8)
        out = lee_filter(noisy)
        assert out.shape == noisy.shape and out.dtype == np.uint8
        assert np.mean(out) < 240, f"Over-exposure detected: mean={np.mean(out):.1f}"
        print(f"Smoke test passed mean={np.mean(out):.1f} shape={out.shape}")
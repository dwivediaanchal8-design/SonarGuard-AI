"""
preprocessing/denoise.py
Sonar speckle suppression using the Lee (1980) adaptive filter and CLAHE.

Lee filter algorithm (Lee, J.S., 1980. "Digital image smoothing and the
sigma filter." Computer Vision, Graphics, and Image Processing, 24(2)):

    mean_local = box_filter(I, N)
    var_local  = box_filter(I², N) - mean_local²
    noise_var  = mean(var_local)            # stationary speckle assumption
    K          = var_local / (var_local + noise_var)
    filtered   = mean_local + K * (I - mean_local)

K → 0 in uniform noise regions (smooths to local mean).
K → 1 at edges and targets (preserves original pixel).
"""

import cv2
import numpy as np
from pathlib import Path


def lee_filter(img_gray: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Apply the Lee (1980) speckle filter to a single-channel grayscale image.

    Parameters
    ----------
    img_gray    : np.ndarray  uint8 or float32, shape (H, W)
    window_size : int         sliding window side length (default 7, must be odd)

    Returns
    -------
    np.ndarray  uint8, shape (H, W), values clipped to [0, 255]
    """
    img_f = img_gray.astype(np.float64)

    mean_local = cv2.boxFilter(
        img_f, ddepth=-1, ksize=(window_size, window_size),
        normalize=True, borderType=cv2.BORDER_REFLECT_101
    )
    mean_sq = cv2.boxFilter(
        img_f ** 2, ddepth=-1, ksize=(window_size, window_size),
        normalize=True, borderType=cv2.BORDER_REFLECT_101
    )

    var_local = np.maximum(mean_sq - mean_local ** 2, 0.0)
    noise_var = np.mean(var_local)

    denom = var_local + noise_var
    K = np.where(denom > 1e-9, var_local / denom, 0.0)

    filtered = mean_local + K * (img_f - mean_local)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def preprocess_sonar(
    img_bgr: np.ndarray,
    clahe: bool = True,
    denoise: bool = True,
    window_size: int = 7,
) -> np.ndarray:
    """
    Full sonar preprocessing pipeline: BGR → grayscale → Lee filter → CLAHE → BGR.

    Parameters
    ----------
    img_bgr     : np.ndarray  Input image in BGR colour space.
    clahe       : bool        Apply CLAHE contrast enhancement (clipLimit=2.0, tile=8×8).
    denoise     : bool        Apply Lee (1980) speckle filter.
    window_size : int         Lee filter window size (default 7).

    Returns
    -------
    np.ndarray  BGR uint8, same spatial dimensions as input.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()

    if denoise:
        gray = lee_filter(gray, window_size=window_size)

    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_clahe(image_path, output_path):
    """
    Load a sonar image from disk, run Lee filter + CLAHE, write result to output_path.

    Parameters
    ----------
    image_path  : str | Path  Source image (grayscale or colour).
    output_path : str | Path  Destination path; parent directories created if needed.

    Returns
    -------
    str | None  output_path on success, None if the source image cannot be read.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lee_filter(img))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), enhanced)
    return output_path


if __name__ == "__main__":
    sample_dir = Path("data/yolo_dataset/images/val")
    imgs = list(sample_dir.glob("*.png")) if sample_dir.exists() else []

    if imgs:
        out = Path("data/denoised_output") / f"lee_{imgs[0].name}"
        apply_clahe(imgs[0], out)
        print(f"Lee + CLAHE → {out}")
    else:
        rng = np.random.default_rng(42)
        noisy = np.clip(
            rng.random((640, 640)) * 255 * rng.gamma(1.5, 1.0, (640, 640)),
            0, 255
        ).astype(np.uint8)
        out = lee_filter(noisy)
        assert out.shape == noisy.shape and out.dtype == np.uint8
        print(f"Smoke test passed  shape={out.shape}  dtype={out.dtype}")
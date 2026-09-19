"""
preprocessing/denoise.py
========================
Sonar Speckle Suppression & Contrast Enhancement Pipeline
----------------------------------------------------------
Implements the classical **Lee (1980) speckle filter** for side-scan sonar
imagery.  The Lee filter is the standard algorithm cited in the SSS literature
(Lee, J.S., 1980. "Digital image smoothing and the sigma filter". *Computer
Vision, Graphics, and Image Processing*, 24(2), pp.255-269).

Algorithm summary
~~~~~~~~~~~~~~~~~
Given a grayscale image I and a window of size N×N:

    mean_local  = box_filter(I, N)                          # local mean
    mean_sq     = box_filter(I², N)                          # local mean of squares
    var_local   = mean_sq − mean_local²                     # local variance
    noise_var   = mean(var_local)                            # global noise variance estimate
                                                              (stationary speckle assumption)
    K           = var_local / (var_local + noise_var)       # Lee weighting coefficient
    filtered    = mean_local + K × (I − mean_local)        # classical Lee output

K ∈ [0, 1]:  K→0 in flat (noise-only) regions → output ≈ local mean (smooth).
             K→1 in edge / target regions   → output ≈ original pixel (preserve).

After Lee filtering, CLAHE (Contrast Limited Adaptive Histogram Equalization)
is applied to lift low-contrast shadow features typical of SSS waterfall images.
"""

import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Core Lee Speckle Filter (vectorized NumPy / OpenCV)
# ---------------------------------------------------------------------------

def lee_speckle_filter(img_gray: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Apply the classical Lee (1980) speckle filter to a single-channel
    grayscale image.

    Parameters
    ----------
    img_gray    : np.ndarray, dtype uint8 or float32, shape (H, W)
                  Input grayscale sonar image.
    window_size : int (odd), default 7
                  Local neighbourhood size for mean/variance estimation.
                  Typical values for SSS: 5, 7, 9.

    Returns
    -------
    np.ndarray, dtype uint8, shape (H, W)
        Lee-filtered image clipped to [0, 255].
    """
    # Work in float64 for numerical stability
    img_f = img_gray.astype(np.float64)

    # Local mean:  E[x]
    mean_local = cv2.boxFilter(
        img_f, ddepth=-1, ksize=(window_size, window_size),
        normalize=True, borderType=cv2.BORDER_REFLECT_101
    )

    # Local mean of squares:  E[x²]
    mean_sq = cv2.boxFilter(
        img_f ** 2, ddepth=-1, ksize=(window_size, window_size),
        normalize=True, borderType=cv2.BORDER_REFLECT_101
    )

    # Local variance:  Var[x] = E[x²] − E[x]²
    var_local = mean_sq - mean_local ** 2
    # Clamp floating-point rounding negatives
    var_local = np.maximum(var_local, 0.0)

    # Global noise variance estimate (stationary speckle assumption)
    noise_var = np.mean(var_local)

    # Lee weighting coefficient K ∈ [0, 1]
    # Avoid division by zero in uniform-noise degenerate images
    denom = var_local + noise_var
    K = np.where(denom > 1e-9, var_local / denom, 0.0)

    # Lee filter output: filtered = mean + K * (pixel − mean)
    filtered = mean_local + K * (img_f - mean_local)

    return np.clip(filtered, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API — pipeline-compatible wrapper used by SonarInferenceEngine
# ---------------------------------------------------------------------------

def preprocess_sonar_image(
    img_bgr: np.ndarray,
    apply_clahe: bool = True,
    apply_bilateral: bool = True,   # parameter kept for API compatibility; controls Lee filter
    window_size: int = 7,
) -> np.ndarray:
    """
    Full sonar preprocessing pipeline:
    1. Convert BGR → Grayscale
    2. Apply Lee (1980) speckle filter    (when apply_bilateral=True)
    3. Apply CLAHE contrast enhancement   (when apply_clahe=True)
    4. Convert Grayscale → BGR for downstream YOLO ingestion

    The `apply_bilateral` flag controls the Lee filter (name kept for
    backwards-compatible API; the actual algorithm is Lee, not bilateral).

    Parameters
    ----------
    img_bgr       : np.ndarray  Input image in BGR colour space.
    apply_clahe   : bool        Enable CLAHE enhancement.
    apply_bilateral : bool      Enable Lee speckle filter (legacy flag name).
    window_size   : int         Lee filter neighbourhood size (default 7).

    Returns
    -------
    np.ndarray  Preprocessed image in BGR colour space, dtype uint8.
    """
    # Step 0 — ensure we have a grayscale working copy
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    processed = gray

    # Step 1 — Lee (1980) Speckle Filter
    if apply_bilateral:
        processed = lee_speckle_filter(processed, window_size=window_size)

    # Step 2 — CLAHE Contrast Enhancement
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

    # Return as BGR for YOLO / downstream consumers
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# File-based helper (used by dataset_prep & __main__ test)
# ---------------------------------------------------------------------------

def apply_clahe_filter(image_path, output_path):
    """
    Reads a sonar image file, applies the full Lee + CLAHE pipeline,
    and writes the result to output_path.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    filtered = lee_speckle_filter(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), enhanced)
    return output_path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_dir = Path("data/yolo_dataset/images/val")
    sample_images = list(sample_dir.glob("*.png")) if sample_dir.exists() else []

    if sample_images:
        test_img = sample_images[0]
        out_dir = Path("data/denoised_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"lee_filtered_{test_img.name}"

        result = apply_clahe_filter(test_img, out_file)
        print("=" * 60)
        print("Step 2 (Lee Speckle Filter + CLAHE) Completed Successfully")
        print(f"Processed : {test_img.name}")
        print(f"Output    : {result}")
        print("=" * 60)
    else:
        # Smoke-test on a synthetic noisy image even without dataset
        print("[Smoke Test] No dataset found — generating synthetic SSS image...")
        rng = np.random.default_rng(42)
        synthetic = (rng.random((640, 640)) * 255).astype(np.uint8)
        # Simulate speckle multiplicative noise
        speckle = rng.gamma(shape=1.5, scale=1.0, size=(640, 640))
        noisy = np.clip(synthetic * speckle, 0, 255).astype(np.uint8)

        out = lee_speckle_filter(noisy, window_size=7)
        assert out.shape == noisy.shape, "Shape mismatch — Lee filter broken!"
        assert out.dtype == np.uint8,    "Dtype error — Lee filter broken!"
        print("=" * 60)
        print("Lee Speckle Filter smoke-test PASSED.")
        print(f"  Input  : shape={noisy.shape}, dtype={noisy.dtype}")
        print(f"  Output : shape={out.shape},  dtype={out.dtype}")
        print("=" * 60)
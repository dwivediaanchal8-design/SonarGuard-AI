import cv2
import numpy as np
from pathlib import Path

def preprocess_sonar_image(img_bgr, apply_clahe=True, apply_bilateral=True):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and 
    Bilateral Filtering to suppress acoustic speckle noise while preserving acoustic shadow edges.
    """
    processed = img_bgr.copy()
    if apply_bilateral:
        processed = cv2.bilateralFilter(processed, d=7, sigmaColor=50, sigmaSpace=50)
    if apply_clahe:
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return processed

def apply_clahe_filter(image_path, output_path):
    """Reads a sonar image file, applies CLAHE and Bilateral filter, and writes to output_path."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    denoised_img = cv2.bilateralFilter(enhanced_img, d=7, sigmaColor=50, sigmaSpace=50)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), denoised_img)
    return output_path

if __name__ == "__main__":
    sample_dir = Path("data/yolo_dataset/images/val")
    sample_images = list(sample_dir.glob("*.png"))

    if sample_images:
        test_img = sample_images[0]
        out_dir = Path("data/denoised_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"denoised_{test_img.name}"

        result = apply_clahe_filter(test_img, out_file)
        print("=" * 60)
        print("Step 2 (Noise Filter) Completed Successfully")
        print(f"Processed: {test_img.name}")
        print(f"Denoised Image Saved: {result}")
        print("=" * 60)
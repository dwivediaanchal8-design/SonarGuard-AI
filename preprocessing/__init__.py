"""
Preprocessing Module — Inference-Time API
==========================================
Exposes only the two functions required at serving/inference time.

Training utilities (dataset_prep, debris_generator, tiling) are intentionally
NOT imported here to avoid loading `requests`, `zipfile`, and other heavy
training dependencies on every app boot.  Import those modules directly
when needed during training/dataset preparation workflows:

    # In training scripts only:
    from preprocessing.dataset_prep import prepare_full_sonar
    from preprocessing.debris_generator import inject_debris
    from preprocessing.tiling import generate_tiles
"""

from .denoise import preprocess_sonar_image, apply_clahe_filter

__all__ = [
    "preprocess_sonar_image",
    "apply_clahe_filter",
]

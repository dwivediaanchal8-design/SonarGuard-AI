"""
Preprocessing Module for Underwater Sonar AI
Contains acoustic noise reduction, sliding-window tiling, synthetic debris generation, and dataset management utilities.
"""

from .denoise import preprocess_sonar_image, apply_clahe_filter
from .tiling import generate_tiles
from .debris_generator import inject_debris, generate_multiclass_debris_dataset
from .dataset_prep import split_data, validate_split, prepare_full_sonar, fetch_debris_dataset

__all__ = [
    "preprocess_sonar_image",
    "apply_clahe_filter",
    "generate_tiles",
    "inject_debris",
    "generate_multiclass_debris_dataset",
    "split_data",
    "validate_split",
    "prepare_full_sonar",
    "fetch_debris_dataset"
]

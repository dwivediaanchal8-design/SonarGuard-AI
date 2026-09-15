"""
Models Module for Underwater Sonar AI
Provides deep learning YOLO multi-class model training, validation evaluation, and weight management.
"""

from .train import start_training
from .evaluate import run_model_evaluation

__all__ = ["start_training", "run_model_evaluation"]

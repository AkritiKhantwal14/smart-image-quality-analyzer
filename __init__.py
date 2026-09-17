"""
Smart Image Quality Analyzer Package
-----------------------------------
An academic computer vision toolkit for analyzing brightness, contrast,
sharpness, noise, and histogram distribution of digital images.
"""

from .image_loader import load_image, get_image_metadata
from .preprocessing import to_grayscale, apply_gaussian_blur, equalize_histogram
from .brightness import calculate_brightness
from .contrast import calculate_contrast
from .blur_detection import calculate_sharpness
from .noise_detection import calculate_noise
from .histogram import calculate_histogram, plot_histogram
from .quality_score import evaluate_quality
from .report import generate_full_report

__all__ = [
    "load_image",
    "get_image_metadata",
    "to_grayscale",
    "apply_gaussian_blur",
    "equalize_histogram",
    "calculate_brightness",
    "calculate_contrast",
    "calculate_sharpness",
    "calculate_noise",
    "calculate_histogram",
    "plot_histogram",
    "evaluate_quality",
    "generate_full_report",
]

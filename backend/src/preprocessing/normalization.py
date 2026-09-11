"""Intensity normalization.

Raw SSS returns vary widely in dynamic range between surveys (different
sonar gain settings, water conditions, ranges), so downstream steps
(denoising, the VAE, YOLO) need a consistent intensity scale to compare
across images at all. Three interchangeable strategies are provided since
which one is "correct" depends on how much outlier-robustness the survey
needs.
"""

from __future__ import annotations

import numpy as np


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """Stretch to the full [0, 255] range. Simple, but sensitive to outlier pixels."""
    image = image.astype(np.float64)
    lo, hi = image.min(), image.max()
    if hi - lo < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    return np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def normalize_zscore(image: np.ndarray, clip_sigma: float = 3.0) -> np.ndarray:
    """Standardize to zero mean/unit variance, clip outliers, then rescale to [0, 255].

    More robust than minmax to a handful of extreme bright/dark pixels
    (e.g. a saturated return), since those get clipped rather than
    stretching the whole histogram.
    """
    image = image.astype(np.float64)
    mean, std = image.mean(), image.std()
    if std < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    z = (image - mean) / std
    z = np.clip(z, -clip_sigma, clip_sigma)
    stretched = (z + clip_sigma) / (2 * clip_sigma) * 255.0
    return stretched.astype(np.uint8)


def normalize_percentile(image: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Stretch the [low_pct, high_pct] percentile range to [0, 255], clipping the rest.

    Good default for sonar: robust to the small number of very bright
    specular returns and very dark shadow pixels without needing to tune
    a sigma like z-score does.
    """
    image = image.astype(np.float64)
    lo, hi = np.percentile(image, [low_pct, high_pct])
    if hi - lo < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    stretched = (image - lo) / (hi - lo) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


_METHODS = {
    "minmax": normalize_minmax,
    "zscore": normalize_zscore,
    "percentile": normalize_percentile,
}


def normalize_intensity(image: np.ndarray, method: str = "percentile", **kwargs) -> np.ndarray:
    """Dispatch to one of the normalization strategies by name."""
    if method not in _METHODS:
        raise ValueError(f"Unknown normalization method '{method}'. Choose from {list(_METHODS)}.")
    return _METHODS[method](image, **kwargs)

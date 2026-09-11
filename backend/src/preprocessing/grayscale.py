"""Grayscale conversion.

Side-scan sonar intensity is inherently single-channel, but ingested files
may arrive as RGB/RGBA (e.g. screenshots of a sonar viewer, or exported
mosaics with a colormap baked in) or already grayscale. This module makes
that safe and explicit rather than assuming a fixed channel count.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an arbitrary-channel image to single-channel grayscale.

    Handles: already-grayscale (2D) arrays, BGR, BGRA, and single-channel
    arrays with a trailing dim of 1. Raises on anything else rather than
    guessing.
    """
    if image.ndim == 2:
        return image

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")

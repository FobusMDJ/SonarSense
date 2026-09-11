"""Resolution standardization + contrast enhancement -- the last step before
an image is considered "clean" and ready for the VAE / YOLO11-Seg branches.

Different sonars/tow speeds produce different pixel resolutions and
along-track/across-track aspect ratios; standardizing to one target shape
is what lets a single model batch train across a mixed dataset.
"""

from __future__ import annotations

import cv2
import numpy as np


def standardize_resolution(image: np.ndarray, target_size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Resize to `target_size` (width, height).

    Uses INTER_AREA when shrinking (avoids aliasing/moire on the fine
    speckle texture) and INTER_CUBIC when enlarging (smoother than linear
    for upsampling low-resolution far-range returns).
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    interpolation = cv2.INTER_AREA if (target_w < w or target_h < h) else cv2.INTER_CUBIC
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def enhance_contrast(image: np.ndarray, method: str = "clahe", clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Boost local contrast so faint/low-return targets stay visible.

    CLAHE (default) is applied per-tile, which matters for SSS because
    contrast needs vary a lot across a single image (near-range vs.
    far-range attenuation) -- a single global histogram equalization would
    over- or under-enhance one side of the swath.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        return clahe.apply(image)
    if method == "histeq":
        return cv2.equalizeHist(image)
    if method == "none":
        return image

    raise ValueError(f"Unknown contrast method '{method}'. Choose from: clahe, histeq, none.")

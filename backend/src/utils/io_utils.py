"""Image I/O helpers used throughout the preprocessing pipeline.

Kept deliberately thin: everything downstream works on plain numpy arrays
so that swapping the ingestion source (flat image files today, XTF/raw
sonar logs later) doesn't ripple through the rest of the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm",
    # Netpbm formats -- SubPipe ships its SSS frames as raw .pbm pixmaps;
    # cv2.imread decodes these natively, no special-casing needed below.
    ".pbm", ".ppm", ".pnm",
    # BenthiCat's SSS-Pretraining tiles ship as raw float32 .npy arrays
    # (already normalized to ~[0, 1] by BenthiCat's own pipeline), not
    # encoded images -- handled specially in load_image() below.
    ".npy",
}


def load_image(path: str | Path, as_gray: bool = False) -> np.ndarray:
    """Load an image file into a numpy array (BGR by default, matching cv2 convention).

    Raises FileNotFoundError with a clear message rather than cv2's silent
    None-return-on-failure, which is a common source of confusing crashes
    downstream.

    .npy files (BenthiCat tiles) are handled separately from cv2: they're
    float32 arrays already scaled to roughly [0, 1] by BenthiCat's own
    pipeline, not encoded images. Rescaling them to uint8 [0, 255] here --
    rather than threading a second scale convention through the rest of the
    codebase -- means dropout detection, normalization, and the denoiser all
    see one consistent domain regardless of source format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
        array = np.clip(array, 0.0, 1.0) * 255.0
        return array.round().astype(np.uint8)

    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_UNCHANGED
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"Failed to decode image (unsupported/corrupt file?): {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save a numpy array to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to write image to {path}")


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Safely rescale an arbitrary-range float/int array to uint8 [0, 255].

    Used whenever a processing step (denoising, normalization) produces
    float output that needs to go back to a displayable/writable format.
    """
    image = image.astype(np.float64)
    lo, hi = float(image.min()), float(image.max())
    if hi - lo < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def list_image_files(directory: str | Path, extensions: Optional[set] = None) -> list[Path]:
    """List supported image files in a directory, sorted for reproducible ordering."""
    directory = Path(directory)
    exts = extensions or SUPPORTED_EXTENSIONS
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)

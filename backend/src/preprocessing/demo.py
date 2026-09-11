"""End-to-end sanity check for the preprocessing pipeline.

No real SSS data exists in this repo yet, so this generates a synthetic
sonar-like waterfall image -- textured seafloor, multiplicative speckle,
a nadir gap, a couple of dropout pings, and two classic bright-highlight
/ acoustic-shadow targets (one long & thin like a pipe, one blob-like a
debris cluster) -- runs it through the full pipeline, and writes a
before/after comparison figure so the result can be checked visually.

Run from the project root:  python -m src.preprocessing.demo
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import get_logger
from src.utils.io_utils import save_image

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "preprocessing_demo.png"


def _fractal_noise(h: int, w: int, octaves: int = 5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Cheap fractal/Perlin-ish noise via summed blurred-noise octaves --
    good enough to look like sandy/rippled seafloor texture without pulling
    in a noise library dependency.
    """
    from scipy.ndimage import gaussian_filter

    rng = rng or np.random.default_rng(0)
    result = np.zeros((h, w), dtype=np.float64)
    amplitude = 1.0
    total_amp = 0.0
    for octave in range(octaves):
        sigma = max(1, (2 ** (octaves - octave)))
        layer = gaussian_filter(rng.standard_normal((h, w)), sigma=sigma)
        result += amplitude * layer
        total_amp += amplitude
        amplitude *= 0.5
    result /= total_amp
    return (result - result.min()) / (result.max() - result.min())


def generate_synthetic_sss(height: int = 400, width: int = 600, seed: int = 42) -> np.ndarray:
    """Build a synthetic SSS waterfall image with realistic-ish artifacts."""
    rng = np.random.default_rng(seed)

    seafloor = _fractal_noise(height, width, octaves=5, rng=rng)
    image = 60 + seafloor * 90  # mid-gray textured seafloor, ~60-150

    # Multiplicative speckle noise, characteristic of coherent sonar imaging.
    speckle = rng.gamma(shape=4.0, scale=0.25, size=(height, width))
    image = image * speckle

    # Nadir gap: bright/saturated column directly under the tow track.
    nadir_col = width // 2
    image[:, nadir_col - 4 : nadir_col + 4] = 255

    # A couple of dropout ping rows (motion-induced signal loss).
    for row in rng.choice(height, size=3, replace=False):
        image[row, :] = 0

    # Target 1: elongated pipe-like object -- bright highlight then acoustic shadow.
    py, px, plen, pw = 120, 150, 90, 10
    image[py : py + pw, px : px + plen] = 230  # bright return
    image[py + pw : py + pw + 18, px : px + plen] = 15  # acoustic shadow

    # Target 2: compact debris-cluster-like blob (e.g. tangled net).
    cy, cx, r = 260, 420, 22
    yy, xx = np.ogrid[:height, :width]
    blob_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
    image[blob_mask] = 210
    shadow_mask = (yy - cy) ** 2 + (xx - (cx + r + 12)) ** 2 <= (r * 0.8) ** 2
    image[shadow_mask] = 20

    return np.clip(image, 0, 255).astype(np.uint8)


def run_demo() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    synthetic = generate_synthetic_sss()
    raw_path = RAW_DIR / "synthetic_demo_001.png"
    save_image(raw_path, synthetic)
    logger.info("Wrote synthetic SSS test image to %s", raw_path)

    pipeline = PreprocessingPipeline()
    result = pipeline.run(raw_path)

    clean_path = PROCESSED_DIR / "synthetic_demo_001_clean.png"
    save_image(clean_path, result.final)
    logger.info("Wrote cleaned image to %s", clean_path)

    stages = [
        ("1. Raw ingested", result.grayscale),
        ("2. Dropout mask", result.dropout_mask.astype(np.uint8) * 255),
        ("3. Dropout repaired", result.dropout_repaired),
        ("4. Normalized", result.normalized),
        ("5. Denoised (Lee)", result.denoised),
        ("6. Final (resized + CLAHE)", result.final),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (title, img) in zip(axes.flat, stages):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"SonarSense preprocessing pipeline -- {result.metadata['dropout_pct']}% dropout repaired",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(REPORT_PATH, dpi=140)
    plt.close(fig)
    logger.info("Wrote before/after comparison figure to %s", REPORT_PATH)


if __name__ == "__main__":
    run_demo()

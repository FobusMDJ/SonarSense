"""Dropout / corruption handling.

AUV motion (heave, pitch, roll) and towfish altitude changes cause
characteristic SSS artifacts: dead rows/columns of near-constant intensity
where pings were lost or saturated, and the permanent nadir gap directly
beneath the sensor track. Left alone, these regions bias normalization
(their intensity is not real seafloor signal) and can be mistaken for
anomalies downstream. This module detects and repairs them before anything
else touches the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.config import get_logger

logger = get_logger(__name__)


@dataclass
class DropoutConfig:
    # A row/column is flagged as dropout if its intensity std is below this
    # (i.e. it's suspiciously flat/constant -- a real seafloor ping has texture).
    flat_std_threshold: float = 2.0
    # A row/column is flagged if its mean intensity is within this many
    # gray levels of pure black or pure white (saturation / total signal loss).
    saturation_margin: float = 3.0
    # A column is flagged if at least this fraction of its pixels are darker
    # than dark_pixel_threshold. Catches the real nadir gap, which is NOT
    # uniformly flat -- it's a black band with a thin bright specular ridge
    # down the middle, so std/mean-based checks alone miss it (std stays
    # high because of the ridge; mean stays above a strict saturation_margin
    # because it's "very dark", not literally zero).
    #
    # Measured directly against real SubPipe frames (see analysis on
    # 1693569222.750.pbm): the old (40, 0.6) combo caught genuine dropout
    # fine on HF (mostly literal col_mean<2 zero columns) but on LF
    # incorrectly flagged huge amounts of real, nonzero, just-low-intensity
    # backscatter as "unobserved" (weak-but-real columns with mean 2-45
    # made up ~46% of everything flagged on the sample LF frame, vs ~11%
    # on HF). Tightening to (15, 0.9) -- only a column that's *almost
    # entirely* near-black counts -- keeps HF's real dropout detection
    # intact (truly-dead fraction unchanged, 59.5%) while cutting LF's
    # false-positive "weak signal called unobserved" from ~46% to ~5% of
    # the frame, and total LF flagged area from ~65% to ~25%.
    dark_pixel_threshold: float = 15.0
    dark_fraction_threshold: float = 0.9
    inpaint_radius: int = 5
    # A contiguous flagged row/column run is treated as physically
    # unobserved (e.g. the real nadir gap, or an extended sensor failure)
    # once it reaches this length, vs a brief accidental dropout that's
    # safe to inpaint and trust. Expressed as a FRACTION of the relevant
    # axis length (row-runs measured against image height, column-runs
    # against image width), not a fixed pixel count -- a fixed count like
    # the old default of 10px means wildly different things on BenthiCat's
    # 384x384 tiles vs SubPipe's 5000x500 frames (10 columns is 2.6% of
    # a 384-wide tile but 0.2% of a 5000-wide frame), which was silently
    # flagging ~80% of every SubPipe frame as "unobserved". Used by
    # detect_validity_mask()/repair_and_mask() to decide what's safe to
    # inpaint (short runs: real signal dominates nearby, a fair
    # approximation) vs what must stay excluded from training/scoring
    # (long runs: no ping was ever received, nothing to approximate).
    large_region_min_frac: float = 0.03
    # Floor on the resulting pixel run length, so tiny images don't get an
    # unreasonably small (or zero) threshold.
    large_region_min_run_px: int = 3


def detect_dropout_mask(image: np.ndarray, config: DropoutConfig | None = None) -> np.ndarray:
    """Return a boolean mask (same H, W as `image`) marking dropout pixels.

    Operates row-wise and column-wise since SSS dropout is characteristically
    linear (a whole ping lost, or the nadir column), not scattered like
    salt-and-pepper noise -- that distinction is what lets this avoid
    flagging small dark/bright natural features as dropout.
    """
    cfg = config or DropoutConfig()
    if image.ndim != 2:
        raise ValueError("detect_dropout_mask expects a single-channel (grayscale) image")

    h, w = image.shape
    mask = np.zeros((h, w), dtype=bool)

    row_std = image.std(axis=1)
    row_mean = image.mean(axis=1)
    bad_rows = (row_std < cfg.flat_std_threshold) | (row_mean < cfg.saturation_margin) | (
        row_mean > 255 - cfg.saturation_margin
    )
    mask[bad_rows, :] = True

    col_std = image.std(axis=0)
    col_mean = image.mean(axis=0)
    col_dark_fraction = (image < cfg.dark_pixel_threshold).mean(axis=0)
    bad_cols = (
        (col_std < cfg.flat_std_threshold)
        | (col_mean < cfg.saturation_margin)
        | (col_mean > 255 - cfg.saturation_margin)
        | (col_dark_fraction > cfg.dark_fraction_threshold)
    )
    mask[:, bad_cols] = True

    n_bad = int(mask.sum())
    if n_bad:
        logger.info(
            "Dropout mask: %d/%d rows, %d/%d cols flagged (%.1f%% of pixels)",
            int(bad_rows.sum()), h, int(bad_cols.sum()), w, 100.0 * n_bad / (h * w),
        )
    return mask


def _contiguous_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(start_index, length) for every contiguous True run in a 1D boolean array."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(flags):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            runs.append((start, i - start))
            in_run = False
    if in_run:
        runs.append((start, len(flags) - start))
    return runs


def _split_small_large(flags: np.ndarray, min_run: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a 1D bad-row/column boolean array by contiguous run length:
    runs shorter than `min_run` (brief glitches) vs at/above it (physically
    unobserved). See DropoutConfig.large_region_min_run.
    """
    small = np.zeros_like(flags)
    large = np.zeros_like(flags)
    for start, length in _contiguous_runs(flags):
        target = large if length >= min_run else small
        target[start : start + length] = True
    return small, large


def detect_validity_mask(image: np.ndarray, config: DropoutConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Detect dropout, split by run length into (repairable_mask, unobserved_mask).

    Same row/column heuristics as detect_dropout_mask(), but a short flagged
    run (a handful of dead pixels from a sensor glitch) is treated
    differently from a long one (the actual nadir gap, or an extended
    sensor failure): the former is safe to inpaint and trust afterward,
    the latter never had a real observation and should stay excluded from
    anything that assumes the pixel is real signal (denoiser loss, anomaly
    scoring, etc.). A pixel flagged "unobserved" on either axis stays
    unobserved even if the other axis alone would call it repairable.

    Returns (repairable_mask, unobserved_mask), both boolean (H, W).
    """
    cfg = config or DropoutConfig()
    if image.ndim != 2:
        raise ValueError("detect_validity_mask expects a single-channel (grayscale) image")
    h, w = image.shape

    row_std = image.std(axis=1)
    row_mean = image.mean(axis=1)
    bad_rows = (row_std < cfg.flat_std_threshold) | (row_mean < cfg.saturation_margin) | (
        row_mean > 255 - cfg.saturation_margin
    )

    col_std = image.std(axis=0)
    col_mean = image.mean(axis=0)
    col_dark_fraction = (image < cfg.dark_pixel_threshold).mean(axis=0)
    bad_cols = (
        (col_std < cfg.flat_std_threshold)
        | (col_mean < cfg.saturation_margin)
        | (col_mean > 255 - cfg.saturation_margin)
        | (col_dark_fraction > cfg.dark_fraction_threshold)
    )

    min_run_rows = max(cfg.large_region_min_run_px, round(cfg.large_region_min_frac * h))
    min_run_cols = max(cfg.large_region_min_run_px, round(cfg.large_region_min_frac * w))
    small_rows, large_rows = _split_small_large(bad_rows, min_run_rows)
    small_cols, large_cols = _split_small_large(bad_cols, min_run_cols)

    unobserved_mask = np.zeros((h, w), dtype=bool)
    unobserved_mask[large_rows, :] = True
    unobserved_mask[:, large_cols] = True

    repairable_mask = np.zeros((h, w), dtype=bool)
    repairable_mask[small_rows, :] = True
    repairable_mask[:, small_cols] = True
    repairable_mask &= ~unobserved_mask  # unobserved wins on overlap

    if repairable_mask.any() or unobserved_mask.any():
        logger.info(
            "Validity mask: %d px repairable (%.1f%%), %d px unobserved/excluded (%.1f%%)",
            int(repairable_mask.sum()), 100.0 * repairable_mask.sum() / (h * w),
            int(unobserved_mask.sum()), 100.0 * unobserved_mask.sum() / (h * w),
        )
    return repairable_mask, unobserved_mask


def repair_dropout(image: np.ndarray, mask: np.ndarray, config: DropoutConfig | None = None) -> np.ndarray:
    """Fill masked-out dropout regions via inpainting.

    Uses cv2's Navier-Stokes inpainting, which propagates plausible texture
    from surrounding valid pixels -- reasonable for thin dropout rows/columns,
    though it is not a substitute for a real acoustic-shadow-aware model.
    """
    cfg = config or DropoutConfig()
    if not mask.any():
        return image.copy()

    mask_u8 = (mask.astype(np.uint8)) * 255
    image_u8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
    repaired = cv2.inpaint(image_u8, mask_u8, cfg.inpaint_radius, cv2.INPAINT_NS)
    return repaired


def handle_dropout(image: np.ndarray, config: DropoutConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: detect + repair in one call. Returns (repaired_image, mask).

    Repairs *everything* flagged, regardless of how wide the flagged region
    is -- fine for producing a single displayable "clean" image
    (PreprocessingPipeline's use case), but see repair_and_mask() when a
    caller needs to know which pixels are real vs invented (e.g. training).
    """
    mask = detect_dropout_mask(image, config)
    repaired = repair_dropout(image, mask, config)
    return repaired, mask


def repair_and_mask(image: np.ndarray, config: DropoutConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Detect + repair, but distinguish *why* a region was flagged.

    Short flagged runs (brief sensor glitches) are inpainted the same way
    handle_dropout() does, and counted as valid afterward -- real signal
    dominates nearby, so the interpolation is a fair stand-in. Long flagged
    runs (the actual nadir gap, or an extended sensor failure) are filled
    with a flat neutral value -- the per-image median of the valid pixels,
    deliberately NOT a realistic-looking reconstruction -- and reported as
    invalid, since no ping was ever received there and nothing should be
    trained or scored as if it had been.

    Returns (processed_image, validity_mask); validity_mask is True for
    pixels that are either untouched or safely repaired, False for pixels
    that were never actually observed.
    """
    cfg = config or DropoutConfig()
    repairable_mask, unobserved_mask = detect_validity_mask(image, cfg)

    processed = repair_dropout(image, repairable_mask, cfg) if repairable_mask.any() else image.copy()

    if unobserved_mask.any():
        valid_pixels = processed[~unobserved_mask]
        fill_value = float(np.median(valid_pixels)) if valid_pixels.size else 0.0
        processed = processed.copy()
        processed[unobserved_mask] = fill_value

    validity_mask = ~unobserved_mask
    return processed, validity_mask

"""Orchestrates the full BASIC PREPROCESSING -> SSS DENOISING -> CLEAN SSS
IMAGE stage of the SonarSense pipeline:

  ingest -> grayscale -> dropout repair -> intensity normalization
  -> resolution standardization -> denoising -> contrast enhancement
  -> clean SSS image

Resize happens BEFORE denoising, not after (deliberately not what the
original pipeline diagram's ordering suggests): denoising cost scales with
pixel count, and downstream (VAE/YOLO) only ever consumes target_size
regardless, so denoising at native resolution just to immediately downscale
was paying for detail no consumer used -- often 8-10x the pixels needed for
an edge/embedded target. See process_record()'s comment for the accuracy
tradeoff this introduces.

Every stage is independently importable/testable (see the sibling modules);
this file just wires them together, driven by config/preprocessing.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.preprocessing.dropout import DropoutConfig, handle_dropout
from src.preprocessing.enhancement import enhance_contrast, standardize_resolution
from src.preprocessing.denoising import denoise
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.ingestion import SSSRecord, ingest_directory, ingest_file
from src.preprocessing.normalization import normalize_intensity
from src.utils.config import get_logger, load_config
from src.utils.io_utils import save_image

logger = get_logger(__name__)


@dataclass
class StageResult:
    """The output image after every stage, kept around for before/after
    visualization and debugging rather than only returning the final image.

    Field order matches execution order: `resized` is the normalized image
    after resolution standardization but BEFORE denoising; `denoised` is
    that resized image after denoising. (Resize now runs before denoise --
    see the module docstring -- so `resized` no longer means "final size,
    still un-denoised" in the way the name alone might suggest to an old
    caller; check which stage you actually want by name, not by assuming
    the old ordering.)
    """

    record: SSSRecord
    grayscale: np.ndarray
    dropout_mask: np.ndarray
    dropout_repaired: np.ndarray
    normalized: np.ndarray
    resized: np.ndarray
    denoised: np.ndarray
    final: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreprocessingPipeline:
    """Runs a single sonar image (or a whole directory of them) through
    every preprocessing stage, in order.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str | Path] = None):
        self.config = config if config is not None else load_config(config_path)

    def process_record(self, record: SSSRecord) -> StageResult:
        cfg = self.config

        gray = to_grayscale(record.image)

        dropout_cfg = DropoutConfig(**cfg.get("dropout", {})) if cfg.get("dropout") else DropoutConfig()
        repaired, mask = handle_dropout(gray, dropout_cfg)

        norm_cfg = dict(cfg.get("normalization", {}))
        norm_method = norm_cfg.pop("method", "percentile")
        normalized = normalize_intensity(repaired, method=norm_method, **norm_cfg)

        # Resize BEFORE denoising, not after. B2U's cost scales with pixel
        # count (width**2+1 forward passes over however many pixels you hand
        # it), and downstream (VAE/YOLO) only ever consumes target_size
        # anyway -- denoising a raw 5000x500 SubPipe frame just to immediately
        # throw away everything past 512x512 was paying for resolution never
        # used. Native-resolution frames commonly ran 8-10x more pixels than
        # target_size, so this is the single biggest lever for edge/embedded
        # throughput, on top of `fast=True` below.
        #
        # Tradeoff worth watching: the checkpoint was trained on native-
        # resolution crops, so feeding it a pre-shrunk image is technically
        # out-of-distribution relative to training. Spot-check output
        # quality (e.g. via src.preprocessing.inspect_denoiser) if this
        # pipeline's target_size differs a lot from what train_denoiser.py's
        # --patch_size was set to.
        enh_cfg = dict(cfg.get("enhancement", {}))
        target_size = tuple(enh_cfg.get("target_size", (512, 512)))
        resized = standardize_resolution(normalized, target_size=target_size)

        denoise_cfg = dict(cfg.get("denoising", {}))
        denoised = denoise(resized, **denoise_cfg) if denoise_cfg else denoise(resized)

        final = enhance_contrast(
            denoised,
            method=enh_cfg.get("contrast_method", "clahe"),
            clip_limit=enh_cfg.get("clip_limit", 2.0),
            tile_grid_size=enh_cfg.get("tile_grid_size", 8),
        )

        return StageResult(
            record=record,
            grayscale=gray,
            dropout_mask=mask,
            dropout_repaired=repaired,
            normalized=normalized,
            denoised=denoised,
            resized=resized,
            final=final,
            metadata={
                "dropout_pixels": int(mask.sum()),
                "dropout_pct": round(100.0 * mask.sum() / mask.size, 2),
            },
        )

    def run(self, image_path: str | Path) -> StageResult:
        record = ingest_file(image_path)
        result = self.process_record(record)
        logger.info(
            "Processed %s: %.1f%% dropout repaired, final shape %s",
            record.record_id, result.metadata["dropout_pct"], result.final.shape,
        )
        return result

    def run_batch(self, input_dir: str | Path, output_dir: str | Path) -> list[StageResult]:
        output_dir = Path(output_dir)
        results = []
        for record in ingest_directory(input_dir):
            result = self.process_record(record)
            save_image(output_dir / f"{record.record_id}_clean.png", result.final)
            results.append(result)
            logger.info(
                "[%d] %s -> %.1f%% dropout repaired",
                len(results), record.record_id, result.metadata["dropout_pct"],
            )
        logger.info("Batch complete: %d image(s) written to %s", len(results), output_dir)
        return results

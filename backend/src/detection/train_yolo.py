"""Trains YOLO11 on a preprocessed (grayscale + normalized, native-swath
sources additionally dropout-repaired + object-centered-tiled) SSS dataset
produced by src/preprocessing/preprocess_yolo_dataset.py or its v2
(preprocess_yolo_dataset_v2.py) -- pass either dataset's data.yaml via --data.

Speed: uses cache='ram' by default. The one real training run of this
project so far was disk-I/O-bound (ultralytics' own warning: "Slow image
access detected... use local storage instead of remote/mounted storage"),
not GPU-bound -- caching every image in RAM once removes that bottleneck
entirely, for free, since these datasets are only a couple GB. This does
not change what the model sees or learns, just how fast it gets fed.

VRAM safety: --batch defaults to 16, proven safe on an RTX 5070 Ti (12GB) in
a real prior run of this project. If a CUDA OOM happens anyway, train()
below catches it, halves the batch, and restarts the run rather than losing
the whole thing to a crash -- see train()'s retry loop.

Deliberately NOT run on denoised images -- this trains on the pipeline's
preprocessed-but-pre-denoising output, per the earlier train/serve decision:
the one paper that isolated a deep-restoration denoising step's effect on
mAP measured only +0.3%, small and possibly negative once shadow erosion
risk is counted, so denoising stays a separate, independently-evaluated
pipeline stage rather than something baked into the detector's training
distribution. (Testing this trained model against denoised test images too,
to see whether that gap is real for our data, is a separate later step --
not part of training.)

Augmentation is YOLO11's built-in augmentation, but every value below is set
EXPLICITLY (even where it matches the ultralytics default) and justified for
single-channel SSS rather than left implicit, in keeping with how this
project has treated preprocessing choices so far:

  - flipud=0.0   Vertical flip is invalid: image rows are pings ordered in
                 time (along-track). Flipping that axis presents swath
                 geometry that never occurs in a real pass. (Same reasoning
                 already applied to the VAE's flip augmentation.)
  - fliplr=0.5   Horizontal (across-track) flip IS valid: side-scan sonar
                 has no inherent left-right handedness within one ping.
  - degrees=0.0  No rotation. Along-track and cross-track resolution are
                 not the same (a recurring point from the SSS-preprocessing
                 research), so an arbitrarily-rotated tile depicts a
                 resolution relationship that doesn't correspond to any
                 real acquisition geometry.
  - hsv_h=0.0, hsv_s=0.0
                 Hue/saturation jitter is a no-op-or-worse on grayscale
                 imagery (loaded as R=G=B): there is no color information
                 to jitter, so these are zeroed rather than left at
                 ultralytics' RGB-photo defaults.
  - hsv_v=0.4    Brightness/gain jitter IS kept -- varying sonar gain
                 settings is a physically real source of variation.
  - translate=0.1, scale=0.5, mosaic=1.0
                 Kept at ultralytics defaults -- standard-issue,
                 orientation-agnostic augmentation that plausibly helps
                 with this dataset's small objects and class imbalance
                 (mosaic composites 4 images per training sample, which is
                 the built-in analogue of the copy-paste/compositing
                 techniques the SSS literature recommends for rare
                 classes). close_mosaic disables it for the last N epochs
                 so training ends on undistorted images.
  - mixup=0.0, copy_paste=0.0, shear=0.0, perspective=0.0
                 Left off. Custom class-imbalance-aware compositing
                 (proper object-level copy-paste targeting the rarest
                 classes specifically) was already deferred until a
                 baseline run shows which classes actually underperform --
                 turning on ultralytics' generic copy_paste now would be
                 half of that idea without the targeting that makes it
                 worthwhile. Shear/perspective have no sonar-acquisition
                 analogue.

Requires ultralytics + torch, which are NOT installed in the cloud-side
sandbox this script was authored in -- run this where your GPU and
ultralytics/torch already live (the same place train_denoiser.py and
train_vae.py run).

Usage:
    python -m src.detection.train_yolo \\
        --data data/processed/yolo_dataset_preprocessed/data.yaml \\
        --model yolo11s.pt --epochs 150 --imgsz 512 --batch 16
"""

from __future__ import annotations

import argparse

from src.utils.config import get_logger

logger = get_logger(__name__)


def _train_once(args: argparse.Namespace, batch: int):
    from ultralytics import YOLO

    model = YOLO(args.model)
    logger.info("Base weights: %s | data: %s | imgsz=%d epochs=%d batch=%d cache=%s device=%s",
                args.model, args.data, args.imgsz, args.epochs, batch, args.cache, args.device or "auto")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        cache=args.cache,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        # -- augmentation, see module docstring for the reasoning behind each --
        flipud=0.0,
        fliplr=0.5,
        degrees=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        mosaic=1.0,
        close_mosaic=args.close_mosaic,
        mixup=0.0,
        copy_paste=0.0,
    )
    return model


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from ultralytics import YOLO  # noqa: F401  -- import-checked here, used inside _train_once
    except ImportError as exc:
        raise ImportError(
            "ultralytics/torch are not installed here. This script trains where "
            "your GPU + ultralytics/torch environment lives, not in the cloud "
            "sandbox that authored it. `pip install ultralytics` there first."
        ) from exc

    # Speed fix: this dataset is a couple GB, comfortably fits in RAM, and the
    # observed bottleneck on a real run of this project was disk I/O (workers
    # waiting on file reads), not GPU compute -- cache='ram' loads every image
    # once and keeps it there, which is a straight speed win with zero effect
    # on what the model actually learns (same images, same augmentation).
    #
    # VRAM safety net: batch is proven safe at args.batch (16 by default) from
    # a real prior run on this exact GPU tier, so this should never trigger --
    # but if a CUDA OOM happens anyway (e.g. something else grabbed VRAM in
    # the background), retry with the batch halved instead of just crashing
    # and losing the whole run. Quality is unaffected either way: ultralytics
    # auto-scales the learning rate against a nominal batch of 64 regardless
    # of what --batch actually is, and augmentation/epochs/data don't change.
    batch = args.batch
    attempts = 0
    while True:
        attempts += 1
        try:
            model = _train_once(args, batch)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 2 or attempts >= 4:
                logger.error("CUDA OOM persisted down to batch=%d -- giving up. "
                             "Close other GPU-using apps (browser, games) and retry.", batch)
                raise
            new_batch = max(2, batch // 2)
            logger.warning("CUDA OOM at batch=%d -- retrying from scratch at batch=%d instead of crashing.", batch, new_batch)
            batch = new_batch

    metrics = model.val(data=args.data, imgsz=args.imgsz, split="test")
    logger.info("Test-split validation: mAP50=%.4f mAP50-95=%.4f", metrics.box.map50, metrics.box.map)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=str, required=True, help="Path to the preprocessed dataset's data.yaml.")
    p.add_argument("--model", type=str, default="yolo11s.pt",
                   help="Pretrained base weights. 'n' or 's' recommended given the edge-deployment target -- "
                        "'m'/'l'/'x' trade accuracy for a footprint that may not fit the target device.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=512, help="Must match preprocess_yolo_dataset.py's TILE_SIZE.")
    p.add_argument("--batch", type=int, default=16, help="Proven safe at this size on an RTX 5070 Ti (12GB) in a prior run. Auto-halved on CUDA OOM instead of crashing -- see train().")
    p.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "false"],
                   help="ultralytics image cache. 'ram' removes the disk-I/O bottleneck diagnosed on this project's last real run (dataset is a couple GB, fits easily) -- biggest available speed win with no effect on model quality.")
    p.add_argument("--patience", type=int, default=30, help="Early-stop if val mAP doesn't improve for this many epochs.")
    p.add_argument("--close_mosaic", type=int, default=10, help="Disable mosaic augmentation for the final N epochs.")
    p.add_argument("--device", type=str, default=None, help="e.g. '0' for first GPU, 'cpu'. Default: ultralytics auto-selects.")
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="sonarsense_yolo11")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())

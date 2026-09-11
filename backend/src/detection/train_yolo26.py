"""Trains YOLO26 (Ultralytics' 2026 unified detection model family) on the
standalone Train_Top dataset produced by
src/preprocessing/build_train_top_dataset.py (data/processed/train_top_yolo
by default) -- pass its data.yaml via --data. This is a SEPARATE, smaller,
4-class dataset (shipwreck, human, aircraft, pipe -- no cylinder, no
ghost_net) built specifically from data/processed/Train_Top and deliberately
NOT merged into yolo_dataset_v2_preprocessed's 5-class corpus. A model
trained with this script will not know cylinder or ghost_net at all -- that
was an explicit scope decision (standalone Train_Top run), not an oversight.
Point --data at a different data.yaml (e.g. yolo_dataset_v2_preprocessed's)
to train YOLO26 on the bigger 5-class corpus instead; nothing else in this
script is dataset-specific.

Why YOLO26 and not YOLO11 (src/detection/train_yolo.py, used for every prior
run of this project): released Jan 2026, same ultralytics `YOLO(...).train()`
API, but DFL-free regression + an STAL small-object-aware label assigner +
up to 43% faster CPU ONNX inference than YOLO11n on the vendor's own
benchmark -- relevant to this project's edge-deployment target the same way
train_yolo.py's n/s-over-m/l/x guidance is. No new training arguments are
required versus YOLO11; `model.train(...)` below is called exactly the way
train_yolo.py calls it. YOLO26 also supports NMS-free end-to-end inference
(`model.train(..., nms=False)` swaps in an end-to-end head) -- left at the
default (NMS on) here since that's an inference/export-time tradeoff, not
something this training script should force silently; pass --end_to_end to
opt in.

Everything else mirrors train_yolo.py's already-established, sonar-specific
choices -- restated here rather than assumed, per how this project treats
every preprocessing/training decision:

Speed: cache='disk' by default (changed from an original 'ram' default --
see the caveat below). Removes the disk-I/O bottleneck diagnosed on the one
real YOLO11 training run of this project without ultralytics' RAM cache
crashing the dataloader on Windows (see next paragraph). Does not change
what the model sees or learns -- --cache ram/false remain available via the
CLI flag if you want to try them.

WINDOWS + cache='ram' CAVEAT (why the default changed): the original
reasoning here was "Train_Top is well under a GB, fits easily" -- true of
the dataset's on-disk PNG size, but NOT of its decoded-in-RAM cache size
(ultralytics caches images as full decoded uint8 arrays; a real run logged
"Caching images (3.0GB RAM)" for Train_Top's 4075 images). On Windows,
PyTorch's DataLoader spawns worker processes via `multiprocessing.spawn`
(no fork), which pickles the ENTIRE dataset object -- including that whole
RAM image cache -- once per worker at startup. That hit a real, reproducible
crash on a real Windows run of this exact script:
    OSError: [Errno 22] Invalid argument
    ...when serializing ultralytics.data.base.BaseDataset._ImageCache object
    ...when serializing dict item 'ims'
followed by `_pickle.UnpicklingError: pickle data was truncated` in the
spawned child. cache='disk' avoids this because the dataset object being
pickled per worker no longer carries gigabytes of decoded image arrays --
each worker reads its own small per-image .npy cache file instead. If you
deliberately want cache='ram' on Windows anyway, pair it with --workers 0
(no worker subprocesses => nothing gets pickled across a process boundary)
-- slower per-epoch than parallel workers, but won't crash.

VRAM safety: --batch defaults to 16 (the value proven safe on an RTX 5070
Ti / 12GB for YOLO11 in a prior run of this project -- YOLO26's memory
footprint per size tier is comparable). If a CUDA OOM happens anyway,
train() below catches it, halves the batch, and restarts rather than losing
the whole run -- see train()'s retry loop.

Augmentation -- identical reasoning to train_yolo.py, because the physical
justification (sonar acquisition geometry) is the same regardless of model
architecture:

  - flipud=0.0   Vertical flip is invalid: image rows are pings ordered in
                 time (along-track). Flipping that axis presents swath
                 geometry that never occurs in a real pass.
  - fliplr=0.5   Horizontal (across-track) flip IS valid: side-scan sonar
                 has no inherent left-right handedness within one ping.
  - degrees=0.0  No rotation. Along-track and cross-track resolution are
                 not the same, so an arbitrarily-rotated tile depicts a
                 resolution relationship with no real acquisition analogue.
  - hsv_h=0.0, hsv_s=0.0
                 Hue/saturation jitter is a no-op on grayscale imagery
                 (loaded as R=G=B) -- zeroed rather than left at
                 ultralytics' RGB-photo defaults.
  - hsv_v=0.4    Brightness/gain jitter IS kept -- varying sonar gain
                 settings is a physically real source of variation.
  - translate=0.1, scale=0.5, mosaic=1.0
                 Kept at ultralytics defaults -- orientation-agnostic,
                 plausibly helps with small objects and class imbalance.
                 close_mosaic disables it for the last N epochs.
  - mixup=0.0, copy_paste=0.0, shear=0.0, perspective=0.0
                 Left off -- same reasoning as train_yolo.py: no targeted
                 class-imbalance-aware compositing has been built yet, and
                 shear/perspective have no sonar-acquisition analogue.

NOTE ON THIS PARTICULAR DATASET: Train_Top's "aircraft" class carries a
labeling caveat (see build_train_top_dataset.py's docstring/SOURCES.md) --
~89 of its source images were tagged via leftover annotation-tool metadata
rather than a deliberately chosen label, and several checked visually look
like genuine shipwreck-type sonar frames. This script trains on whatever
data.yaml says regardless -- it does not second-guess the dataset -- so
that caveat is worth re-reading before trusting an "aircraft" prediction
from a model trained here.

Requires ultralytics + torch (a version of ultralytics new enough to know
about YOLO26 -- `pip install -U ultralytics` if `YOLO("yolo26n.pt")` raises
a not-found error), NOT installed in the cloud-side sandbox this script was
authored in -- run this where your GPU and ultralytics/torch already live.

Usage:
    python -m src.detection.train_yolo26 \\
        --data data/processed/train_top_yolo/data.yaml \\
        --model yolo26s.pt --epochs 150 --imgsz 512 --batch 16
"""

from __future__ import annotations

import argparse

from src.utils.config import get_logger

logger = get_logger(__name__)


def _train_once(args: argparse.Namespace, batch: int):
    from ultralytics import YOLO

    model = YOLO(args.model)
    logger.info("Base weights: %s | data: %s | imgsz=%d epochs=%d batch=%d cache=%s device=%s end_to_end=%s",
                args.model, args.data, args.imgsz, args.epochs, batch, args.cache, args.device or "auto", args.end_to_end)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        cache=args.cache,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        nms=not args.end_to_end,  # YOLO26-specific: False swaps in NMS-free end-to-end inference/export
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
            "sandbox that authored it. `pip install -U ultralytics` there first "
            "(YOLO26 needs a recent-enough ultralytics release)."
        ) from exc

    # Same OOM-safety net as train_yolo.py: batch is proven safe at
    # args.batch (16 by default) from a real YOLO11 run on this project's
    # GPU tier; if a CUDA OOM happens anyway, retry with the batch halved
    # instead of crashing and losing the whole run. Quality is unaffected:
    # ultralytics auto-scales the learning rate against a nominal batch of
    # 64 regardless of what --batch actually is.
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
    p.add_argument("--data", type=str, required=True, help="Path to train_top_yolo's (or any other) data.yaml.")
    p.add_argument("--model", type=str, default="yolo26s.pt",
                   help="Pretrained base weights. 'n' or 's' recommended given the edge-deployment target -- "
                        "'m'/'l'/'x' trade accuracy for a footprint that may not fit the target device.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=512, help="Must match build_train_top_dataset.py's TILE_SIZE.")
    p.add_argument("--batch", type=int, default=16, help="Proven safe at this size on an RTX 5070 Ti (12GB) for YOLO11 in a prior run. Auto-halved on CUDA OOM instead of crashing -- see train().")
    p.add_argument("--cache", type=str, default="disk", choices=["ram", "disk", "false"],
                   help="ultralytics image cache. 'disk' avoids a real Windows crash seen with 'ram' "
                        "(OSError [Errno 22] pickling the full RAM image cache into each spawned "
                        "dataloader worker -- see module docstring). Pass --cache ram --workers 0 if "
                        "you want RAM caching anyway.")
    p.add_argument("--workers", type=int, default=8, help="Dataloader worker processes. Set to 0 to disable "
                        "worker subprocesses entirely (avoids the Windows pickling crash even with --cache ram, "
                        "at the cost of slower per-epoch augmentation throughput).")
    p.add_argument("--patience", type=int, default=30, help="Early-stop if val mAP doesn't improve for this many epochs.")
    p.add_argument("--close_mosaic", type=int, default=10, help="Disable mosaic augmentation for the final N epochs.")
    p.add_argument("--end_to_end", action="store_true", help="YOLO26-specific: train for NMS-free end-to-end inference/export instead of the default NMS head.")
    p.add_argument("--device", type=str, default=None, help="e.g. '0' for first GPU, 'cpu'. Default: ultralytics auto-selects.")
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="sonarsense_yolo26_train_top")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())

"""Offline data-prep pass for VAE training: runs every N01/N05 patch through
the production PreprocessingPipeline (dropout repair -> normalize -> resize
-> denoise -> CLAHE contrast) exactly once, and caches the result to disk as
PNGs plus a manifest.

Why this exists as its own script rather than denoising inside the VAE's
DataLoader on every epoch: the VAE will run for many epochs over the same
images, and re-deriving a deterministic per-image transform on every sample
on every epoch would be pure waste. Pay that cost once here; train_vae.py
then just loads a plain PNG.

Denoising method: defaults to `lee` (the classical, fully-implemented Lee
filter -- no training, no GPU forward pass, runs in plain numpy/scipy) NOT
`blind2unblind`. This is the fast/light default: `models/` has no trained
Blind2Unblind checkpoint yet, and training one first (train_denoiser.py,
itself a real multi-hour job) would directly contradict wanting this VAE
ready quickly. Pass `--denoise_method blind2unblind --weights_path ...`
later, once a B2U checkpoint exists, to switch to the learned denoiser --
nothing else in this script needs to change, `denoise()` (denoising.py)
already dispatches on method name. Whichever method is used here MUST also
be what runs in production inference, or the VAE's reconstruction-error
thresholds won't transfer (train/serve skew) -- see denoising.py.

Does NOT touch train_denoiser.py, dropout.py, or pipeline.py -- reads
existing preprocessing config/pipeline as-is, adds a denoising override.

Usage (fast/light default -- Lee filter, no checkpoint needed):
    python -m src.preprocessing.prepare_vae_dataset \\
        --data_dir data/raw/N01 data/raw/N05 \\
        --save_dir data/processed/vae_cache

Usage (once a Blind2Unblind checkpoint exists, for the higher-fidelity path):
    python -m src.preprocessing.prepare_vae_dataset \\
        --data_dir data/raw/N01 data/raw/N05 \\
        --denoise_method blind2unblind --weights_path models/blind2unblind_epoch2.pth \\
        --save_dir data/processed/vae_cache
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import get_logger
from src.utils.io_utils import list_image_files, save_image

logger = get_logger(__name__)


def build_config(
    denoise_method: str, weights_path: str | None, device: str | None, fast: bool, target_size: tuple[int, int]
) -> dict:
    return {
        "dropout": {},
        "normalization": {"method": "percentile"},
        "enhancement": {
            "target_size": list(target_size),
            "contrast_method": "clahe",
            "clip_limit": 2.0,
            "tile_grid_size": 8,
        },
        "denoising": {
            "method": denoise_method,
            "weights_path": weights_path,
            "device": device,
            "fast": fast,
            # Loud failure beats silently training on Lee-filtered images
            # when a LEARNED method was explicitly requested but has no
            # checkpoint. Irrelevant for method="lee" (denoise() ignores
            # weights_path/fallback for it entirely).
            "fallback_on_missing_weights": False,
        },
    }


def prepare(args: argparse.Namespace) -> Path:
    if args.denoise_method != "lee" and not args.weights_path:
        raise ValueError(
            f"--denoise_method {args.denoise_method} needs a trained checkpoint -- pass --weights_path, "
            "or drop back to --denoise_method lee (the default) to run now without one."
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args.denoise_method, args.weights_path, args.device, not args.no_fast, tuple(args.target_size))
    pipeline = PreprocessingPipeline(config=config)

    paths = []
    for d in args.data_dir:
        found = list_image_files(d)
        logger.info("  %d source images from %s", len(found), d)
        paths.extend(found)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No supported images found in {args.data_dir}")

    manifest_path = save_dir / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else []
    done_stems = {e["stem"] for e in manifest}

    start = time.time()
    n_skipped = 0
    for i, path in enumerate(paths):
        stem = path.stem
        if stem in done_stems:
            n_skipped += 1
            continue

        try:
            result = pipeline.run(path)
        except Exception as exc:  # noqa: BLE001 -- one bad file shouldn't kill a multi-hour job
            logger.warning("skip %s: %s", path, exc)
            continue

        out_name = f"{stem}.png"
        save_image(save_dir / out_name, result.final)
        manifest.append({
            "stem": stem,
            "file": out_name,
            "source": str(path),
            "source_dir": str(path.parent),
            "dropout_pct": result.metadata["dropout_pct"],
        })

        if (i + 1) % args.log_every == 0 or (i + 1) == len(paths):
            elapsed = time.time() - start
            rate = (i + 1 - n_skipped) / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[%d/%d] processed (skipped %d already-done), %.2f img/s, elapsed %.1fs",
                i + 1, len(paths), n_skipped, rate, elapsed,
            )
            json.dump(manifest, open(manifest_path, "w"), indent=2)  # periodic save -- safe to interrupt/resume

    json.dump(manifest, open(manifest_path, "w"), indent=2)
    logger.info("Done: %d cached images -> %s", len(manifest), manifest_path)
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, nargs="+", required=True, help="Source dirs (e.g. data/raw/N01 data/raw/N05).")
    p.add_argument("--denoise_method", type=str, default="lee", choices=["lee", "blind2unblind", "dspnet"],
                   help="'lee' (default) is fast/light: classical filter, no training, no GPU needed, no --weights_path "
                        "required. Switch to 'blind2unblind' once a trained checkpoint exists for higher-fidelity "
                        "denoising -- but that method MUST also be what production inference uses (train/serve match).")
    p.add_argument("--weights_path", type=str, default=None, help="Required only if --denoise_method is a learned method (e.g. models/blind2unblind_epoch2.pth).")
    p.add_argument("--save_dir", type=str, required=True, help="Where to write cached PNGs + manifest.json. Resumable: re-running skips stems already in manifest.json.")
    p.add_argument("--target_size", type=int, nargs=2, default=[512, 512], help="width height, matching pipeline.py's enhancement.target_size.")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                   help="Only matters for --denoise_method blind2unblind/dspnet (ignored by the default 'lee', which "
                        "is plain CPU numpy/scipy and already fast). Omit to auto-pick a GPU if visible.")
    p.add_argument("--no_fast", action="store_true",
                   help="Only matters for --denoise_method blind2unblind (ignored by 'lee'). Use the full width**2-view "
                        "B2U recipe instead of the fast single-pass mode. Off by default since fast=True is what "
                        "pipeline.py actually ships -- the VAE should train on what it'll see in production.")
    p.add_argument("--limit", type=int, default=None, help="Cap the number of source images processed (for a quick smoke test).")
    p.add_argument("--log_every", type=int, default=200)
    return p


if __name__ == "__main__":
    parsed = build_arg_parser().parse_args()
    prepare(parsed)

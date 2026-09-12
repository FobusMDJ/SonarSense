"""Qualitative + quantitative check of a trained ConvVAE checkpoint
(train_vae.py's output): reconstruction quality and the anomaly-score
(reconstruction-error) distribution on held-out data.

Held-out means genuinely held out: this script rebuilds the exact same
train/val split train_vae.py used (VAEPatchDataset filtered the same way,
then torch.utils.data.random_split with the same --split_seed), so the
patches scored here are ones the model never trained on -- as long as you
pass the same --cache_dir/--max_dropout_pct/--val_fraction/--split_seed
values used for training (the defaults match train_vae.py's own defaults,
so if you trained with plain defaults, no extra flags are needed here).

Outputs:
  1. Summary stats (mean/std/min/max/percentiles) of reconstruction error
     over the whole held-out val set -- this is your baseline "what does
     normal seafloor score" reference for later picking an anomaly
     threshold.
  2. --n_samples example panels saved as PNGs: original | reconstruction |
     per-pixel error heatmap, side by side. Blurry reconstructions are
     expected (see model.py's docstring) -- what matters is whether the
     heatmap lights up on genuine structure, not sharpness.

Usage (held-out baseline from vae_cache):
    python -m src.vae.test_vae \\
        --weights_path src/vae/vae_epoch100.pth \\
        --cache_dir data/processed/vae_cache \\
        --out_dir vae_eval_out --n_samples 12

Usage (score every image in a whole folder -- e.g. the YOLO test split,
which has both debris-labeled and background images -- and, if --label_dir
is given, split stats by label + report AUROC for how well reconstruction
error alone separates the two):
    python -m src.vae.test_vae \\
        --weights_path src/vae/vae_epoch100.pth \\
        --image_dir data/processed/yolo_dataset_v2_preprocessed/test/images \\
        --label_dir data/processed/yolo_dataset_v2_preprocessed/test/labels \\
        --out_dir vae_eval_out --n_samples 12
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from src.utils.config import get_logger
from src.vae.model import ConvVAE
# VAEPatchDataset is only needed for the --cache_dir baseline path (it
# reproduces train_vae.py's held-out split) -- imported lazily inside
# build_val_set() instead of here, so --image_path/--image_dir heatmap
# generation works with just model.py + this file + the checkpoint, no
# train_vae.py required. See this file's docstring / the "portable" note
# in test_single_image()'s caller for why that matters.

logger = get_logger(__name__)


def load_model(weights_path: str, device: torch.device) -> ConvVAE:
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model = ConvVAE(
        input_size=ckpt["input_size"],
        latent_dim=ckpt["latent_dim"],
        channels=tuple(ckpt["channels"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(
        "Loaded %s (epoch=%s, input_size=%d, latent_dim=%d, channels=%s)",
        weights_path, ckpt.get("epoch", "?"), ckpt["input_size"], ckpt["latent_dim"], tuple(ckpt["channels"]),
    )
    return model


def build_val_set(args: argparse.Namespace):
    """Reproduces train_vae.py's exact train/val split so this evaluates
    only patches the model never saw during training. Only called for the
    --cache_dir path -- imports train_vae.py lazily so --image_path/
    --image_dir work without it (see the import comment near the top of
    this file)."""
    from src.vae.train_vae import VAEPatchDataset

    full_dataset = VAEPatchDataset(args.cache_dir, max_dropout_pct=args.max_dropout_pct, flip_prob=0.0)
    n_val = max(1, int(len(full_dataset) * args.val_fraction))
    n_train = len(full_dataset) - n_val
    generator = torch.Generator().manual_seed(args.split_seed)
    _, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val], generator=generator)
    logger.info("Held-out val set: %d patches (%d total, %d used for training)", n_val, len(full_dataset), n_train)
    return val_set


def summarize_errors(model: ConvVAE, val_set, device: torch.device, batch_size: int = 32) -> np.ndarray:
    loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    all_errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            all_errors.append(model.reconstruction_error(batch).cpu().numpy())
    errors = np.concatenate(all_errors)
    logger.info(
        "Reconstruction error over %d held-out patches: mean=%.6f std=%.6f min=%.6f p50=%.6f p95=%.6f p99=%.6f max=%.6f",
        len(errors), errors.mean(), errors.std(), errors.min(),
        np.percentile(errors, 50), np.percentile(errors, 95), np.percentile(errors, 99), errors.max(),
    )
    return errors


def save_example_panels(model: ConvVAE, val_set, device: torch.device, out_dir: Path, n_samples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = min(n_samples, len(val_set))
    rng = np.random.default_rng(0)
    indices = rng.choice(len(val_set), size=n_samples, replace=False)

    with torch.no_grad():
        for rank, idx in enumerate(indices):
            x = val_set[int(idx)].unsqueeze(0).to(device)  # (1, 1, H, W)
            recon, _, _ = model(x)
            error = model.reconstruction_error(x).item()
            panel = _build_panel(x[0, 0], recon[0, 0], error)
            out_path = out_dir / f"sample_{rank:02d}_err{error:.5f}.png"
            cv2.imwrite(str(out_path), panel)

    logger.info("Wrote %d example panels (original | reconstruction | error heatmap) to %s", n_samples, out_dir)


def _build_panel(x: torch.Tensor, recon: torch.Tensor, error: float) -> np.ndarray:
    """Builds one original | reconstruction | error-heatmap side-by-side
    BGR panel (as a plain numpy array, ready for cv2.imwrite) from two
    (H, W) tensors in [0, 1] plus the scalar reconstruction error."""
    orig = (x.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    rec = (recon.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    diff = np.abs(orig.astype(np.float32) - rec.astype(np.float32))
    diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    rec_bgr = cv2.cvtColor(rec, cv2.COLOR_GRAY2BGR)
    panel = np.hstack([orig_bgr, rec_bgr, heatmap])
    cv2.putText(panel, f"recon_error={error:.5f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def test_single_image(model: ConvVAE, image_path: str, device: torch.device, out_dir: Path) -> float:
    """Runs the VAE on one arbitrary image (e.g. a known-anomaly crop from
    the YOLO dataset, not from vae_cache) and saves its panel.

    Caveat worth knowing: vae_cache's PNGs went through the FULL production
    PreprocessingPipeline used by prepare_vae_dataset.py, denoising and CLAHE
    contrast enhancement included. train_yolo.py deliberately trains on
    images WITHOUT that denoising/CLAHE step (see that script's docstring),
    so a raw YOLO-dataset image fed here differs from the VAE's training
    distribution in more ways than just "contains an anomaly or not" --
    some of any elevated error could be this pipeline mismatch, not
    genuine novelty. Treat this as a quick qualitative smell test, not a
    calibrated score, unless/until this image is run through the same
    denoise+CLAHE pipeline the cache was built with first.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if image.shape != (model.input_size, model.input_size):
        logger.warning(
            "Image is %s, resizing to %dx%d to match the model -- see this function's docstring "
            "for why that alone doesn't make this a fully like-for-like test.",
            image.shape, model.input_size, model.input_size,
        )
        image = cv2.resize(image, (model.input_size, model.input_size), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon, _, _ = model(x)
        error = model.reconstruction_error(x).item()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    out_path = out_dir / f"single_{stem}_err{error:.5f}.png"
    cv2.imwrite(str(out_path), _build_panel(x[0, 0], recon[0, 0], error))
    logger.info("Image %s -> reconstruction_error=%.6f (panel saved to %s)", image_path, error, out_path)
    return error


def _is_positive_label(label_path: Path) -> bool:
    """True if this YOLO label file exists and has >=1 non-blank line (i.e.
    the image has an annotated object). YOLO's own convention for a
    "background" image (no objects) is an empty or absent label file --
    same convention denoise_test_set.py relies on."""
    if not label_path.exists():
        return False
    return any(line.strip() for line in label_path.read_text().splitlines())


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U / rank-sum identity (equivalent to
    sklearn.metrics.roc_auc_score) -- avoids adding scikit-learn as a
    dependency for one metric when scipy (a rank function) is already
    required by denoising.py."""
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))
    pos_ranks = ranks[: len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def test_image_folder(
    model: ConvVAE,
    image_dir: Path,
    device: torch.device,
    out_dir: Path,
    label_dir: Path | None = None,
    n_example_panels: int = 12,
    save_all: bool = False,
) -> list[dict]:
    """Runs the VAE over every image in image_dir (e.g. the whole YOLO test
    split). If label_dir is given, each image is additionally tagged
    positive (has >=1 annotated debris object) or background (none) using
    the same labels YOLO trains on -- letting you check whether
    reconstruction error alone actually separates debris from background,
    via each group's stats and an AUROC score.

    Same pipeline-mismatch caveat as test_single_image applies to every
    image here: this folder wasn't run through the VAE's own denoise+CLAHE
    preprocessing, so treat this as a qualitative/smell-test signal, not a
    fully calibrated benchmark.
    """
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    results = []
    with torch.no_grad():
        for i, path in enumerate(image_paths, 1):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning("Skipping unreadable image: %s", path)
                continue
            if image.shape != (model.input_size, model.input_size):
                image = cv2.resize(image, (model.input_size, model.input_size), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            error = model.reconstruction_error(x).item()
            is_positive = _is_positive_label(label_dir / f"{path.stem}.txt") if label_dir is not None else None
            results.append({"path": str(path), "stem": path.stem, "error": error, "is_positive": is_positive})
            if i % 200 == 0 or i == len(image_paths):
                logger.info("Scored %d/%d images", i, len(image_paths))

    errors = np.array([r["error"] for r in results])

    if label_dir is not None:
        pos_errors = np.array([r["error"] for r in results if r["is_positive"]])
        neg_errors = np.array([r["error"] for r in results if r["is_positive"] is False])
        logger.info(
            "Debris-labeled images: n=%d mean=%.6f p50=%.6f | Background images: n=%d mean=%.6f p50=%.6f",
            len(pos_errors), pos_errors.mean() if len(pos_errors) else float("nan"),
            np.percentile(pos_errors, 50) if len(pos_errors) else float("nan"),
            len(neg_errors), neg_errors.mean() if len(neg_errors) else float("nan"),
            np.percentile(neg_errors, 50) if len(neg_errors) else float("nan"),
        )
        if len(pos_errors) and len(neg_errors):
            auroc = _auroc(pos_errors, neg_errors)
            logger.info(
                "AUROC (does reconstruction error separate debris images from backgrounds?) = %.4f "
                "-- 0.5 means no separation (error is useless as an anomaly signal here), 1.0 is perfect separation.",
                auroc,
            )
    else:
        logger.info(
            "All images: n=%d mean=%.6f std=%.6f p50=%.6f p95=%.6f max=%.6f",
            len(errors), errors.mean(), errors.std(), np.percentile(errors, 50), np.percentile(errors, 95), errors.max(),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "folder_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "reconstruction_error", "is_positive"])
        for r in sorted(results, key=lambda r: -r["error"]):
            writer.writerow([r["path"], f"{r['error']:.6f}", r["is_positive"]])
    logger.info("Wrote per-image scores (highest error first) to %s", csv_path)

    ranked = sorted(results, key=lambda r: -r["error"])

    if save_all:
        # Every image, ranked highest-error (most anomalous per the VAE)
        # first -- browsing the folder in name order goes most-to-least
        # anomalous.
        panels_dir = out_dir / "panels"
        panels_dir.mkdir(parents=True, exist_ok=True)
        for rank, r in enumerate(ranked):
            image = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
            if image.shape != (model.input_size, model.input_size):
                image = cv2.resize(image, (model.input_size, model.input_size), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                recon, _, _ = model(x)
            tag = "pos" if r["is_positive"] else "neg" if r["is_positive"] is False else "na"
            out_path = panels_dir / f"{rank:04d}_{tag}_{r['stem']}_err{r['error']:.5f}.png"
            cv2.imwrite(str(out_path), _build_panel(x[0, 0], recon[0, 0], r["error"]))
            if (rank + 1) % 200 == 0 or rank + 1 == len(ranked):
                logger.info("Saved %d/%d panels", rank + 1, len(ranked))
        logger.info("Wrote all %d panels (original | reconstruction | error heatmap), ranked highest-error first, to %s", len(ranked), panels_dir)
    else:
        # Save panels for the highest- and lowest-error images: what the
        # VAE finds MOST and LEAST anomalous, for eyeballing whether "most
        # anomalous" actually lines up with genuine debris.
        half = max(1, n_example_panels // 2)
        for group_name, group in (("highest_error", ranked[:half]), ("lowest_error", ranked[-half:])):
            for rank, r in enumerate(group):
                image = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
                if image.shape != (model.input_size, model.input_size):
                    image = cv2.resize(image, (model.input_size, model.input_size), interpolation=cv2.INTER_AREA)
                x = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    recon, _, _ = model(x)
                tag = "pos" if r["is_positive"] else "neg" if r["is_positive"] is False else "na"
                out_path = out_dir / f"{group_name}_{rank:02d}_{tag}_{r['stem']}_err{r['error']:.5f}.png"
                cv2.imwrite(str(out_path), _build_panel(x[0, 0], recon[0, 0], r["error"]))
        logger.info("Wrote example panels for the highest- and lowest-error images to %s", out_dir)

    return results


def run(args: argparse.Namespace) -> None:
    if not args.cache_dir and not args.image_path and not args.image_dir:
        raise ValueError("Pass --cache_dir (held-out baseline), --image_path (single image), --image_dir (whole folder), or a combination.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.weights_path, device)

    baseline = None
    if args.cache_dir:
        val_set = build_val_set(args)
        baseline = summarize_errors(model, val_set, device)
        save_example_panels(model, val_set, device, Path(args.out_dir), args.n_samples)

    if args.image_path:
        error = test_single_image(model, args.image_path, device, Path(args.out_dir))
        if baseline is not None:
            pct_rank = (baseline < error).mean() * 100
            logger.info(
                "This image's error (%.6f) is higher than %.1f%% of held-out NORMAL patches "
                "(mean=%.6f, p95=%.6f) -- the higher that percentile, the more anomalous the VAE finds it.",
                error, pct_rank, baseline.mean(), np.percentile(baseline, 95),
            )

    if args.image_dir:
        test_image_folder(
            model, Path(args.image_dir), device, Path(args.out_dir),
            label_dir=Path(args.label_dir) if args.label_dir else None,
            n_example_panels=args.n_samples,
            save_all=args.all_panels,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights_path", type=str, required=True, help="Checkpoint from train_vae.py, e.g. vae_best.pth or a numbered epoch snapshot.")
    p.add_argument("--cache_dir", type=str, default=None, help="Same --cache_dir used to train this checkpoint. Omit to skip the held-out baseline.")
    p.add_argument("--image_path", type=str, default=None, help="Path to a single arbitrary image (e.g. a YOLO-dataset crop) to test individually -- see test_single_image()'s docstring for the pipeline-mismatch caveat.")
    p.add_argument("--image_dir", type=str, default=None, help="Path to a whole folder of images (e.g. the YOLO test split) to score every image in -- see test_image_folder()'s docstring.")
    p.add_argument("--label_dir", type=str, default=None, help="Matching YOLO labels folder for --image_dir -- if given, splits stats into debris-labeled vs background images and reports AUROC.")
    p.add_argument("--all_panels", action="store_true", help="With --image_dir: save an original|reconstruction|heatmap panel for EVERY image (into <out_dir>/panels/, ranked highest-error first), instead of just the top/bottom --n_samples extremes. Warning: one PNG per input image -- for a large folder (e.g. 1840 test images) this writes that many files.")
    p.add_argument("--max_dropout_pct", type=float, default=50.0, help="Must match the value used at training time to reproduce the same held-out split.")
    p.add_argument("--val_fraction", type=float, default=0.1, help="Must match the value used at training time.")
    p.add_argument("--split_seed", type=int, default=42, help="Must match the value used at training time.")
    p.add_argument("--out_dir", type=str, default="vae_eval_out")
    p.add_argument("--n_samples", type=int, default=12, help="Number of original|reconstruction|heatmap example panels to save.")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

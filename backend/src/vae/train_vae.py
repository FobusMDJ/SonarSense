"""Trains the seafloor-background VAE on cached, fully-preprocessed
(dropout-repaired, normalized, resized, Blind2Unblind-denoised, CLAHE-
enhanced) "hard negative" patches -- real seafloor, no known marine debris.

Run `prepare_vae_dataset.py` first to produce the cache this reads; that
script's docstring explains why the VAE trains on the denoised pipeline
output rather than raw/filtered images (train/serve consistency -- this is
exactly what the VAE receives in production -- plus a lower noise floor for
the model to have to treat as "normal").

Self-supervised like the denoiser, but unlike it, only a subset of the raw
data is appropriate input here: N01/N05 (no labeled objects at all) rather
than HF/LF (which contain the very pipe/debris objects this model needs to
treat as anomalous, i.e. reconstruct poorly). Mixing in HF/LF would teach
the VAE to reconstruct pipe-shaped structures well, which directly weakens
the anomaly signal for one of the target classes.

Loss: reconstruction MSE + KL divergence, weighted by `kl_weight` with a
linear warm-up over `kl_warmup_epochs` (standard beta-VAE technique to
avoid posterior collapse -- the encoder mapping every input to the same
uninformative latent because the KL term dominates before the model has
learned anything useful to reconstruct). Mirrors the same "anneal a weight
across training" idea train_denoiser.py uses for its beta schedule, applied
to a different failure mode.

Only augmentation used: random horizontal (left-right / across-track) flip.
Side-scan sonar has no inherent left-right handedness within a single
across-track cut, so this augmentation is physically valid. Vertical flip
or rotation is NOT used -- rows are pings ordered in time (along-track),
so flipping or rotating that axis would present the network with swath
geometry that doesn't correspond to how a real object ever actually
appears in a real pass.

Fast/light by default, none of it at the cost of what's actually learned:
  - --channels defaults to (16, 32, 64, 128, 128, 128) instead of model.py's
    doc-example (32, 64, 128, 256, 256, 256) -- half the width in every
    layer, which is exactly the "narrow it for edge/embedded" lever model.py
    already exposes on purpose. Fewer channels = fewer FLOPs and a smaller
    model, at some cost to reconstruction fidelity -- acceptable per
    model.py's own framing: blurry reconstructions of normal seafloor are
    fine, the metric that matters is reconstruction-error SEPARATION between
    normal and anomalous patches, not image sharpness. --latent_dim is
    likewise halved to 128. Widen both back toward model.py's example if a
    baseline run shows the anomaly signal is too weak.
  - Mixed precision (torch.autocast + GradScaler) on CUDA: same math, half
    the memory traffic per step, meaningfully faster on any recent NVIDIA
    GPU. Off automatically on CPU. Disable with --no_amp if it ever causes
    trouble (rare, but AMP + BatchNorm can occasionally need a lower LR).
  - DataLoader: pin_memory + persistent_workers when --num_workers > 0 --
    both are free wins for a many-epoch run over small fixed-size images
    (faster host->GPU copy, no worker respawn cost every epoch), zero effect
    on what's loaded.
  - CUDA OOM safety net, same pattern as train_yolo.py: catches
    torch.cuda.OutOfMemoryError, halves --batch_size (floor 2, max 4
    attempts) and restarts from scratch rather than losing the whole run.
    Cheap here specifically because this model trains from random init (no
    pretrained weights to lose) and OOM -- if it happens at all -- shows up
    on the very first batch, so "restart from scratch" costs nothing.

Usage:
    python -m src.vae.train_vae \\
        --cache_dir data/processed/vae_cache --save_dir models --n_epoch 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from src.utils.config import get_logger
from src.vae.model import ConvVAE

logger = get_logger(__name__)


class VAEPatchDataset(Dataset):
    """Reads the manifest.json produced by prepare_vae_dataset.py and loads
    each cached (already fully-preprocessed, fixed-size) PNG directly --
    no on-the-fly denoising here, that cost was already paid once offline.

    Excludes patches dominated by physically-unobserved pixels (nadir gap /
    sensor dropout): a patch that's mostly filled-in stand-in rather than
    real seafloor doesn't represent genuine background, and would corrupt
    the "normal" distribution the VAE is meant to learn. Mirrors
    SSSPatchDataset's min_valid_fraction philosophy on the denoiser side,
    at the patch-selection level instead of the loss level (there's no
    per-pixel validity mask to weight a loss by here -- N01/N05 patches are
    fixed-size, not randomly cropped, so filtering whole patches is the
    equivalent lever).
    """

    def __init__(self, cache_dir: str | Path, max_dropout_pct: float = 50.0, flip_prob: float = 0.5):
        self.cache_dir = Path(cache_dir)
        manifest = json.load(open(self.cache_dir / "manifest.json"))
        self.entries = [e for e in manifest if e["dropout_pct"] <= max_dropout_pct]
        n_excluded = len(manifest) - len(self.entries)
        if n_excluded:
            logger.info("Excluded %d/%d cached patches with >%.0f%% unobserved pixels", n_excluded, len(manifest), max_dropout_pct)
        if not self.entries:
            raise FileNotFoundError(f"No usable entries in {self.cache_dir}/manifest.json after filtering")
        self.flip_prob = flip_prob

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> torch.Tensor:
        entry = self.entries[index]
        image = cv2.imread(str(self.cache_dir / entry["file"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to read cached patch: {entry['file']}")
        if np.random.rand() < self.flip_prob:
            image = np.ascontiguousarray(image[:, ::-1])  # across-track flip only -- see module docstring
        return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)


def kl_weight_schedule(epoch: int, warmup_epochs: int, target: float) -> float:
    if warmup_epochs <= 0:
        return target
    return target * min(1.0, epoch / warmup_epochs)


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float) -> dict[str, torch.Tensor]:
    recon_loss = torch.nn.functional.mse_loss(recon, target, reduction="mean")
    # Per-sample KL to a standard normal prior, averaged over the batch --
    # NOT summed over latent dims then averaged over pixels, which would
    # make its scale depend on latent_dim in a way that fights kl_weight.
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total = recon_loss + kl_weight * kl
    return {"total": total, "recon": recon_loss, "kl": kl}


def _train_once(args: argparse.Namespace, batch_size: int, device: torch.device) -> Path:
    full_dataset = VAEPatchDataset(args.cache_dir, max_dropout_pct=args.max_dropout_pct)
    n_val = max(1, int(len(full_dataset) * args.val_fraction))
    n_train = len(full_dataset) - n_val
    generator = torch.Generator().manual_seed(args.split_seed)
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val], generator=generator)
    logger.info("VAEPatchDataset: %d train / %d val patches", n_train, n_val)

    # pin_memory + persistent_workers: free speed wins for a many-epoch run
    # over small fixed-size images -- faster host->GPU copy, no worker
    # respawn cost every epoch. Both no-ops (safely ignored) when
    # num_workers=0 or device is CPU.
    use_workers = args.num_workers > 0
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_workers,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = ConvVAE(input_size=args.input_size, latent_dim=args.latent_dim, channels=tuple(args.channels)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Mixed precision on CUDA: same math, less memory traffic per step,
    # meaningfully faster on any recent NVIDIA GPU -- no effect on what's
    # learned. Both are no-ops (disabled) on CPU or if --no_amp is passed.
    amp_enabled = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    if amp_enabled:
        logger.info("Mixed precision (AMP) enabled.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    logger.info(
        "Training on device: %s | batch=%d channels=%s latent_dim=%d amp=%s",
        device, batch_size, tuple(args.channels), args.latent_dim, amp_enabled,
    )

    for epoch in range(1, args.n_epoch + 1):
        model.train()
        kl_w = kl_weight_schedule(epoch, args.kl_warmup_epochs, args.kl_weight)
        epoch_start = time.time()
        running = {"total": 0.0, "recon": 0.0, "kl": 0.0}

        for iteration, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                recon, mu, logvar = model(batch)
                losses = vae_loss(recon, batch, mu, logvar, kl_w)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            for k in running:
                running[k] += losses[k].item()

            if iteration % args.log_every == 0:
                logger.info(
                    "epoch %d iter %d | kl_weight=%.4f total=%.6f recon=%.6f kl=%.6f",
                    epoch, iteration, kl_w, losses["total"].item(), losses["recon"].item(), losses["kl"].item(),
                )

        n_batches = max(1, len(train_loader))
        logger.info(
            "epoch %d train done in %.1fs | avg total=%.6f recon=%.6f kl=%.6f",
            epoch, time.time() - epoch_start, running["total"] / n_batches, running["recon"] / n_batches, running["kl"] / n_batches,
        )

        model.eval()
        val_recon = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    recon, mu, logvar = model(batch)
                    val_recon += torch.nn.functional.mse_loss(recon, batch, reduction="mean").item()
        val_recon /= max(1, len(val_loader))
        logger.info("epoch %d val recon_loss=%.6f", epoch, val_recon)

        if epoch % args.snapshot_every == 0 or epoch == args.n_epoch:
            _save_checkpoint(save_dir / f"vae_epoch{epoch}.pth", model, epoch, args)
            logger.info("Saved checkpoint: vae_epoch%d.pth", epoch)

        if val_recon < best_val:
            best_val = val_recon
            _save_checkpoint(save_dir / "vae_best.pth", model, epoch, args)
            logger.info("New best val recon_loss=%.6f -> vae_best.pth", val_recon)

    final_path = save_dir / "vae_sss.pth"
    _save_checkpoint(final_path, model, args.n_epoch, args)
    logger.info("Training complete. Final model: %s", final_path)
    return final_path


def train(args: argparse.Namespace) -> Path:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # VRAM safety net, same pattern as train_yolo.py: this model trains from
    # random init (nothing pretrained to lose), and a CUDA OOM -- if it
    # happens at all -- shows up on the very first batch, so restarting from
    # scratch after halving the batch costs effectively nothing. Quality is
    # unaffected by batch size here beyond the usual (small) BatchNorm/
    # gradient-noise sensitivity, which halving-as-a-last-resort accepts in
    # exchange for not crashing the whole run.
    batch = args.batch_size
    attempts = 0
    while True:
        attempts += 1
        try:
            return _train_once(args, batch, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 2 or attempts >= 4:
                logger.error("CUDA OOM persisted down to batch=%d -- giving up. "
                             "Close other GPU-using apps and retry, or pass a narrower --channels.", batch)
                raise
            new_batch = max(2, batch // 2)
            logger.warning("CUDA OOM at batch=%d -- retrying from scratch at batch=%d instead of crashing.", batch, new_batch)
            batch = new_batch


def _save_checkpoint(path: Path, model: ConvVAE, epoch: int, args: argparse.Namespace) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "input_size": args.input_size,
            "latent_dim": args.latent_dim,
            "channels": list(args.channels),
        },
        path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", type=str, required=True, help="Output of prepare_vae_dataset.py (PNGs + manifest.json).")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--input_size", type=int, default=512, help="Must match prepare_vae_dataset.py's --target_size.")
    parser.add_argument("--latent_dim", type=int, default=128, help="Halved from model.py's doc-example (256) -- fast/light default, see module docstring.")
    parser.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64, 128, 128, 128], help="Encoder/decoder channel widths. Halved from model.py's doc-example (32,64,128,256,256,256) as the fast/light default -- widen back toward that if a baseline run needs stronger reconstruction fidelity. See model.py's docstring.")
    parser.add_argument("--max_dropout_pct", type=float, default=50.0, help="Exclude cached patches with more than this %% unobserved pixels.")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16, help="Auto-halved on CUDA OOM instead of crashing -- see train().")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision on CUDA (on by default -- see module docstring). No effect on CPU.")
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--kl_weight", type=float, default=1e-3, help="Target beta-VAE KL weight after warm-up. Small by default since recon_loss is a mean over 512*512=262144 pixels while KL is per-latent-dim -- an unscaled KL term would dominate and cause posterior collapse.")
    parser.add_argument("--kl_warmup_epochs", type=int, default=10, help="Linearly ramp kl_weight from 0 to its target over this many epochs.")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--snapshot_every", type=int, default=10)
    return parser


if __name__ == "__main__":
    parsed = build_arg_parser().parse_args()
    train(parsed)

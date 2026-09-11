"""Trains the Blind2Unblind denoiser on real, unlabeled noisy SSS patches.

Self-supervised: no clean targets, no synthetic noise added, no labels
needed at all -- only real SSS images (post grayscale/dropout-repair/
normalization, matching the distribution the trained model will see at
inference in pipeline.py). Point this at a directory of such images (e.g.
BenthiCat's SSS-Pretraining tiles, or SubPipe frames after running them
through PreprocessingPipeline) and it trains directly on their real noise.

Algorithm and every hyperparameter here (Lambda1, Lambda2, increase_ratio,
the Thread1/Thread2 beta-annealing schedule, mask width=4, patch size 256)
match Wang et al., "Blind2Unblind: Self-Supervised Image Denoising with
Visible Blind Spots", CVPR 2022, cross-checked against the authors'
released training scripts (github.com/zejinwang/Blind2Unblind) for
correctness. See denoiser_model.py and denoiser_masker.py for the pieces;
this file is the training loop that wires them together.

Usage:
    python -m src.preprocessing.train_denoiser \\
        --data_dir data/raw/N01 data/raw/N05 --save_dir models --n_epoch 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.denoiser_masker import GlobalAwareMasker
from src.preprocessing.denoiser_model import DenoiserUNet
from src.preprocessing.dropout import DropoutConfig, repair_and_mask
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.normalization import normalize_intensity
from src.utils.config import get_logger
from src.utils.io_utils import list_image_files, load_image

logger = get_logger(__name__)

# Exact schedule from the official implementation: alpha (Lambda1) weights
# the reconstruction term; beta anneals from Lambda2 up to increase_ratio
# over the [Thread1, Thread2] fraction of total training, staying flat
# outside that range. This ramps up the "re-visible" regularization as
# training progresses, which is what keeps the blind-spot branch from
# collapsing into a trivial identity-like mapping.
THREAD1 = 0.4
THREAD2 = 1.0


class SSSPatchDataset(Dataset):
    """Random-crops training patches from a directory of real SSS images.

    Each source image is run through grayscale conversion, dropout
    detection/repair, and percentile normalization exactly once (cached),
    so patches are cropped from the SAME distribution the denoiser will see
    when it's slotted into PreprocessingPipeline at inference -- not from
    raw images with nadir bars and saturation artifacts still in them.

    Also tracks, per pixel, whether it's a real observation or a filled-in
    stand-in for a physically-unobserved region (the nadir gap, or an
    extended sensor failure -- see dropout.repair_and_mask()). Every
    __getitem__ call returns (image_patch, validity_patch) so the training
    loop can exclude invented pixels from the loss -- the network should
    only ever be scored against pixels a real ping actually produced.
    """

    def __init__(
        self,
        data_dir: str | Path | list[str | Path],
        patch_size: int = 256,
        dropout_config: DropoutConfig | None = None,
        min_valid_fraction: float = 0.5,
        max_crop_attempts: int = 8,
    ):
        # Accept either one directory or a list -- lets a single training run
        # mix sources (e.g. BenthiCat's N01 + N05 .npy tiles alongside
        # SubPipe's .pbm frames) without pre-merging them on disk.
        data_dirs = [data_dir] if isinstance(data_dir, (str, Path)) else list(data_dir)
        self.paths: list[Path] = []
        for d in data_dirs:
            found = list_image_files(d)
            logger.info("  %d images from %s", len(found), d)
            self.paths.extend(found)
        if not self.paths:
            raise FileNotFoundError(f"No supported images found in {data_dirs}")
        self.patch_size = patch_size
        self.dropout_config = dropout_config or DropoutConfig()
        # A crop that lands mostly/entirely inside an unobserved region
        # (e.g. squarely on the nadir gap) contributes little-to-no usable
        # signal to the loss -- retry a few times for a better crop rather
        # than training on a near-empty patch. Falls back to the best crop
        # found if nothing clears the threshold (e.g. a tile dominated by
        # the gap), so this never loops forever or raises.
        self.min_valid_fraction = min_valid_fraction
        self.max_crop_attempts = max_crop_attempts
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        logger.info("SSSPatchDataset: %d source images total from %d source dir(s)", len(self.paths), len(data_dirs))

    def __len__(self) -> int:
        return len(self.paths)

    def _preprocessed(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index not in self._cache:
            image = load_image(self.paths[index])
            gray = to_grayscale(image)
            repaired, validity = repair_and_mask(gray, self.dropout_config)
            normed = normalize_intensity(repaired, method="percentile")
            self._cache[index] = (normed, validity)
        return self._cache[index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, validity = self._preprocessed(index)
        h, w = image.shape

        # Pad up if the source is smaller than one patch (small tiles, e.g.
        # BenthiCat's 384x384 crops, are still safely handled). Reflected
        # border pixels are synthetic too, not real observations -- extend
        # validity the same way rather than defaulting padding to "valid".
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
            validity = np.pad(validity, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = image.shape

        best_patch, best_valid, best_frac = None, None, -1.0
        for _ in range(self.max_crop_attempts):
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            valid_patch = validity[top : top + self.patch_size, left : left + self.patch_size]
            frac = float(valid_patch.mean())
            if frac > best_frac:
                best_patch = image[top : top + self.patch_size, left : left + self.patch_size]
                best_valid = valid_patch
                best_frac = frac
            if frac >= self.min_valid_fraction:
                break

        image_tensor = torch.from_numpy(best_patch.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)
        valid_tensor = torch.from_numpy(best_valid.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        return image_tensor, valid_tensor


def compute_beta(epoch: int, n_epoch: int, lambda2: float, increase_ratio: float) -> float:
    """The exact beta-annealing schedule from the paper/reference code."""
    progress = epoch / n_epoch
    if progress <= THREAD1:
        return lambda2
    if progress <= THREAD2:
        return lambda2 + (progress - THREAD1) * (increase_ratio - lambda2) / (THREAD2 - THREAD1)
    return increase_ratio


def train_step(
    model: DenoiserUNet,
    masker: GlobalAwareMasker,
    noisy: torch.Tensor,
    valid: torch.Tensor,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    """One Blind2Unblind training step. Returns the loss components (as
    floats, for logging) -- caller is responsible for backward()/step().

    `valid` (n, 1, h, w), matching `noisy`, is 1 where a pixel is a real
    observation and 0 where it's a filled-in stand-in for a physically
    unobserved region (see dropout.repair_and_mask()). Every loss term is
    masked by it and averaged only over valid pixels -- the network is
    never rewarded or penalized for what it predicts where no ping was
    ever received, so it can't learn to reproduce the fill value instead
    of real speckle statistics.
    """
    n, c, h, w = noisy.shape
    valid_count = valid.sum().clamp(min=1.0)

    # Blind-spot branch: width**2 masked passes, reassembled into one image
    # where every pixel is a genuine blind-spot prediction.
    net_input, mask = masker.all_views(noisy)
    blind_output = model(net_input)
    blind_output = (blind_output * mask).view(n, -1, c, h, w).sum(dim=1)
    diff = (blind_output - noisy) * valid

    # Non-blind branch: the same network sees the full, unmasked image.
    # no_grad -- this term acts as a fixed target the blind branch is
    # pushed to compensate for, not something optimized directly.
    with torch.no_grad():
        visible_output = model(noisy)
    exp_diff = (visible_output - noisy) * valid

    revisible = diff + beta * exp_diff
    loss_reg = alpha * torch.sum(diff**2) / valid_count
    loss_rev = torch.sum(revisible**2) / valid_count
    loss_all = loss_reg + loss_rev

    return {
        "loss_all": loss_all,
        "loss_reg": loss_reg.detach().item(),
        "loss_rev": loss_rev.detach().item(),
        "diff": (torch.sum(diff**2) / valid_count).detach().item(),
        "exp_diff": (torch.sum(exp_diff**2) / valid_count).detach().item(),
        "valid_frac": valid.mean().detach().item(),
    }


def train(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    dropout_config = DropoutConfig(large_region_min_frac=args.large_region_min_frac)
    dataset = SSSPatchDataset(
        args.data_dir,
        patch_size=args.patch_size,
        dropout_config=dropout_config,
        min_valid_fraction=args.min_valid_fraction,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    model = DenoiserUNet(in_channels=1, out_channels=1, base_width=args.base_width).to(device)
    masker = GlobalAwareMasker(width=args.width)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ratio = args.n_epoch / 100
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(m * ratio)) for m in (20, 40, 60, 80)],
        gamma=args.gamma,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    final_beta = args.lambda2

    for epoch in range(1, args.n_epoch + 1):
        model.train()
        beta = compute_beta(epoch, args.n_epoch, args.lambda2, args.increase_ratio)
        final_beta = beta
        epoch_start = time.time()
        running_loss = 0.0

        for iteration, (noisy, valid) in enumerate(loader):
            noisy = noisy.to(device)
            valid = valid.to(device)
            optimizer.zero_grad()
            losses = train_step(model, masker, noisy, valid, alpha=args.lambda1, beta=beta)
            losses["loss_all"].backward()
            optimizer.step()
            running_loss += losses["loss_all"].item()

            if iteration % args.log_every == 0:
                logger.info(
                    "epoch %d iter %d | beta=%.2f loss_reg=%.6f loss_rev=%.6f diff=%.6f exp_diff=%.6f valid_frac=%.2f",
                    epoch, iteration, beta, losses["loss_reg"], losses["loss_rev"], losses["diff"], losses["exp_diff"],
                    losses["valid_frac"],
                )

        scheduler.step()
        logger.info(
            "epoch %d done in %.1fs, avg loss=%.6f", epoch, time.time() - epoch_start, running_loss / max(1, len(loader)),
        )

        if epoch % args.snapshot_every == 0 or epoch == args.n_epoch:
            ckpt_path = save_dir / f"blind2unblind_epoch{epoch}.pth"
            _save_checkpoint(ckpt_path, model, epoch, final_beta, args)
            logger.info("Saved checkpoint: %s", ckpt_path)

    final_path = save_dir / "blind2unblind_sss.pth"
    _save_checkpoint(final_path, model, args.n_epoch, final_beta, args)
    logger.info("Training complete. Final model: %s", final_path)
    return final_path


def _save_checkpoint(path: Path, model: DenoiserUNet, epoch: int, final_beta: float, args: argparse.Namespace) -> None:
    """Bundles weights with `final_beta` -- inference needs it to reproduce
    the paper's pred_mid = (blind + beta*visible) / (1 + beta) blend
    exactly, using whatever beta training actually annealed to.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "final_beta": final_beta,
            "width": args.width,
            "base_width": args.base_width,
        },
        path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Blind2Unblind on real noisy SSS images.")
    parser.add_argument(
        "--data_dir", type=str, required=True, nargs="+",
        help="One or more directories of real (unlabeled) SSS images/tiles. "
        "Accepts .png/.jpg/.tif/.bmp/.pgm, Netpbm .pbm/.ppm/.pnm (SubPipe), "
        "and BenthiCat's raw float32 .npy tiles, mixed freely across dirs.",
    )
    parser.add_argument("--save_dir", type=str, default="models", help="Where to write checkpoints.")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor at each milestone.")
    parser.add_argument("--width", type=int, default=4, help="Mask cell size; width**2 forward passes per step.")
    parser.add_argument("--base_width", type=int, default=48, help="UNet base channel width (wf). Lower for edge/faster training.")
    parser.add_argument("--lambda1", type=float, default=1.0, help="alpha: weight on the blind reconstruction loss.")
    parser.add_argument("--lambda2", type=float, default=2.0, help="beta at the start of training.")
    parser.add_argument("--increase_ratio", type=float, default=20.0, help="beta at the end of training.")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--snapshot_every", type=int, default=10, help="Save a checkpoint every N epochs.")
    parser.add_argument(
        "--large_region_min_frac", type=float, default=0.03,
        help="Contiguous bad row/col run, as a fraction of that axis's length (row-runs vs "
        "height, col-runs vs width), at/above which it's treated as physically unobserved "
        "(e.g. nadir gap) and excluded from the loss, vs a shorter run that's inpainted and "
        "trusted as a brief sensor glitch. Fraction-based so it behaves consistently across "
        "very different image sizes (BenthiCat 384x384 tiles vs SubPipe 5000x500 frames).",
    )
    parser.add_argument(
        "--min_valid_fraction", type=float, default=0.5,
        help="Retry a random crop (up to 8x) if fewer than this fraction of its pixels are "
        "real observations -- avoids training on patches that landed mostly on the nadir gap.",
    )
    return parser


if __name__ == "__main__":
    parsed = build_arg_parser().parse_args()
    train(parsed)

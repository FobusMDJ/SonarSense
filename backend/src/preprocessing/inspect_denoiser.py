"""Read-only inspection tool: save before/after Blind2Unblind samples for
manual visual review. Does NOT modify dropout.py or train_denoiser.py, and
does not touch training in progress -- it only reads images and (optionally)
a checkpoint, and writes new PNGs to --out_dir.

Reproduces exactly what SSSPatchDataset feeds the model during training
(grayscale -> repair_and_mask -> percentile normalize), so what you see in
`_input.png` is the literal network input, not an approximation of it.

Two modes:

1. --scan DIR: walks every supported image in DIR, computes its validity
   mask (detection only, no inpainting -- fast), and prints each file's
   unobserved% sorted, bucketed into <2% / 2-15% / 15-40% / 40%+ so you can
   pick representative examples (a "normal" one, a small-dropout one, a
   large-gap one, a 50-60% one) by filename instead of guessing.

2. --images PATH [PATH ...]: for each image, saves:
     <stem>_raw.png       original, grayscale, unmodified
     <stem>_input.png     repaired+normalized network input (what B2U sees)
     <stem>_mask.png      validity overlay: red=unobserved/excluded,
                          yellow=repaired(small dropout), rest=untouched
     <stem>_denoised.png  B2U output (only if --checkpoint given)
     <stem>_panel.png     all of the above side by side, labeled

Usage:
    # find candidates first
    python -m src.preprocessing.inspect_denoiser --scan data\\raw\\SubPipeMiniSSS\\DATA\\SSS_LF_images\\Image --limit 3000

    # then render the ones you picked
    python -m src.preprocessing.inspect_denoiser \\
        --checkpoint models\\blind2unblind_epoch2.pth \\
        --images data\\raw\\N01\\<normal>.npy data\\raw\\SubPipeMiniSSS\\...\\<small_dropout>.pbm \\
                 data\\raw\\SubPipeMiniSSS\\...\\<large_gap>.pbm data\\raw\\SubPipeMiniSSS\\...\\<fifty_pct>.pbm \\
        --out_dir inspect_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.dropout import DropoutConfig, detect_validity_mask, repair_and_mask
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.normalization import normalize_intensity
from src.utils.config import get_logger
from src.utils.io_utils import list_image_files, load_image

logger = get_logger(__name__)


def scan(data_dir: str, limit: int) -> None:
    paths = list_image_files(data_dir)[:limit]
    cfg = DropoutConfig()
    results = []
    for p in paths:
        try:
            gray = to_grayscale(load_image(p))
        except Exception as exc:  # noqa: BLE001 -- best-effort scan, skip unreadable files
            logger.warning("skip %s: %s", p, exc)
            continue
        _, unobserved = detect_validity_mask(gray, cfg)
        frac = 100.0 * unobserved.mean()
        results.append((frac, p))

    results.sort(key=lambda r: r[0])
    buckets = [("normal (<2%)", 0, 2), ("small dropout (2-15%)", 2, 15),
               ("large gap (15-40%)", 15, 40), ("heavy (40%+)", 40, 101)]
    for name, lo, hi in buckets:
        matches = [(f, p) for f, p in results if lo <= f < hi]
        print(f"\n--- {name}: {len(matches)} files ---")
        for f, p in matches[:5]:
            print(f"  {f:5.1f}%  {p}")
    print(f"\nScanned {len(results)} images total from {data_dir}")


def _best_crop_bounds(unobserved: np.ndarray, size: int) -> tuple[slice, slice]:
    """Pick the `size`x`size` (or smaller, if the image already fits) window
    with the highest concentration of unobserved pixels, so a crop of a huge
    SubPipe frame still lands on the actual dropout/gap artifact instead of
    an arbitrary or average patch. No-op (full extent) on axes already <= size.
    """
    h, w = unobserved.shape

    def best_1d(counts: np.ndarray, length: int, want: int) -> slice:
        if length <= want:
            return slice(0, length)
        cs = np.cumsum(counts)
        cs = np.insert(cs, 0, 0)
        window_sums = cs[want:] - cs[:-want]
        start = int(np.argmax(window_sums))
        return slice(start, start + want)

    row_slice = best_1d(unobserved.sum(axis=1), h, min(size, h))
    col_slice = best_1d(unobserved.sum(axis=0), w, min(size, w))
    return row_slice, col_slice


def _mask_overlay(gray: np.ndarray, repairable: np.ndarray, unobserved: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[unobserved] = (0, 0, 255)     # red: excluded from loss entirely
    overlay[repairable] = (0, 255, 255)   # yellow: inpainted, still counted valid
    return overlay


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def inspect(images: list[str], checkpoint: str | None, out_dir: str, crop_size: int, device: str | None = None, fast: bool = False) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DropoutConfig()

    denoiser = None
    if checkpoint:
        from src.preprocessing.denoising import Blind2UnblindDenoiser
        denoiser = Blind2UnblindDenoiser(weights_path=checkpoint, device=device)

    for path in images:
        path = Path(path)
        stem = path.stem
        gray_full = to_grayscale(load_image(path))
        # Compute detection/repair on the FULL image first, exactly like
        # SSSPatchDataset._preprocessed() does (repair_and_mask runs once on
        # the whole source image, and only THEN gets cropped) -- so what
        # you're looking at here is the same mask/repair a training crop
        # would actually see, not a mask recomputed fresh on a sub-window
        # (which would use different row/col statistics and could disagree).
        repairable_mask, unobserved_mask = detect_validity_mask(gray_full, cfg)
        processed_full, validity_full = repair_and_mask(gray_full, cfg)
        normed_full = normalize_intensity(processed_full, method="percentile")

        # Full-frame inference on a 5000-wide SubPipe image needs the model
        # to run width**2=16 masked forward passes at full resolution --
        # easily blows memory with no GPU. Crop to a manageable window
        # centered on the densest unobserved region so the artifact you
        # actually want to look at (the gap, the dropout run) stays in
        # frame instead of being cropped away arbitrarily.
        if gray_full.shape[0] > crop_size or gray_full.shape[1] > crop_size:
            row_slice, col_slice = _best_crop_bounds(unobserved_mask, crop_size)
        else:
            row_slice, col_slice = slice(0, gray_full.shape[0]), slice(0, gray_full.shape[1])
        gray = gray_full[row_slice, col_slice]
        repairable_mask = repairable_mask[row_slice, col_slice]
        unobserved_mask = unobserved_mask[row_slice, col_slice]
        normed = normed_full[row_slice, col_slice]

        cv2.imwrite(str(out / f"{stem}_raw.png"), gray)
        cv2.imwrite(str(out / f"{stem}_input.png"), normed)
        mask_img = _mask_overlay(gray, repairable_mask, unobserved_mask)
        cv2.imwrite(str(out / f"{stem}_mask.png"), mask_img)

        panels = [
            _label(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "raw"),
            _label(cv2.cvtColor(normed, cv2.COLOR_GRAY2BGR), "input (repaired+normed)"),
            _label(mask_img, f"mask (unobs={100*unobserved_mask.mean():.1f}%)"),
        ]

        if denoiser is not None:
            denoised = denoiser.denoise(normed, fast=fast)
            cv2.imwrite(str(out / f"{stem}_denoised.png"), denoised)
            panels.append(_label(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR), "B2U denoised" + (" (fast)" if fast else "")))

        # common height for hconcat
        h = min(p.shape[0] for p in panels)
        resized = []
        for p in panels:
            scale = h / p.shape[0]
            w = int(p.shape[1] * scale)
            resized.append(cv2.resize(p, (w, h)))
        panel = cv2.hconcat(resized)
        cv2.imwrite(str(out / f"{stem}_panel.png"), panel)
        print(f"saved {stem}: unobserved={100*unobserved_mask.mean():.1f}%  repairable={100*repairable_mask.mean():.1f}%  -> {out}/{stem}_panel.png")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan", type=str, default=None, help="Directory to scan and bucket by unobserved%%.")
    p.add_argument("--limit", type=int, default=3000, help="Max files to scan (scan mode only).")
    p.add_argument("--images", type=str, nargs="+", default=None, help="Specific image paths to render before/after panels for.")
    p.add_argument("--checkpoint", type=str, default=None, help="Trained B2U checkpoint (.pth). Omit to skip the denoised output.")
    p.add_argument("--out_dir", type=str, default="inspect_out", help="Where to write PNGs.")
    p.add_argument("--crop_size", type=int, default=768, help="Max width/height run through the model. Larger SubPipe frames are cropped to this, centered on the densest unobserved region, to avoid OOM from width**2 masked passes at full resolution (mainly a concern on CPU; raise this, e.g. to 6000, on a GPU with room to spare so the shown unobserved%% matches --scan's whole-image number instead of a worst-window crop).")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Where to run the model. Omit to auto-pick a GPU if one is visible to this process, else CPU.")
    p.add_argument("--fast", action="store_true", help="Skip the width**2 masked-view reconstruction and use a single forward pass instead (~21x faster on the tested checkpoint, ~1.3/255 mean pixel difference). See Blind2UnblindDenoiser.denoise's docstring.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.scan:
        scan(args.scan, args.limit)
    elif args.images:
        inspect(args.images, args.checkpoint, args.out_dir, args.crop_size, args.device, args.fast)
    else:
        raise SystemExit("Pass either --scan DIR or --images PATH [PATH ...]")

"""One-off eval helper: runs an already-preprocessed YOLO test split's
images through denoising.py's denoise() (Lee filter by default) and writes
the result to a parallel images/ + labels/ folder pair, plus a copy of the
dataset's data.yaml with `test:` pointed at the new denoised images.

Why this exists: train_yolo.py deliberately trains on preprocessed-but-not-
denoised images (see that script's module docstring) -- denoising is kept
as a separate, independently-evaluated pipeline stage rather than baked
into the detector's training distribution. This script is that evaluation:
it lets you run the already-trained best.pt against a denoised copy of the
same test images to see whether denoising actually helps or hurts mAP on
this project's own data, without retraining anything.

Labels are copied unchanged (not regenerated) -- denoising only touches
pixel values, object locations in the scene don't move.

Usage (Lee filter -- fast, no checkpoint needed):
    python -m src.preprocessing.denoise_test_set \\
        --src_images data/processed/yolo_dataset_v2_preprocessed/test/images \\
        --src_labels data/processed/yolo_dataset_v2_preprocessed/test/labels \\
        --dst data/processed/yolo_dataset_v2_preprocessed/test_denoised \\
        --data_yaml data/processed/yolo_dataset_v2_preprocessed/data.yaml

Then evaluate the already-trained model against it:
    python -c "from ultralytics import YOLO; m = YOLO('weights/best.pt'); \\
        metrics = m.val(data='data/processed/yolo_dataset_v2_preprocessed/data_denoised_test.yaml', imgsz=512, split='test'); \\
        print('mAP50:', metrics.box.map50, 'mAP50-95:', metrics.box.map)"

Pass --clahe to also apply CLAHE contrast enhancement after denoising, with the
exact same parameters prepare_vae_dataset.py uses (clip_limit=2.0,
tile_grid_size=8). Denoising alone (the default, no --clahe) only replicates
ONE of the two pipeline stages that differ between how the VAE's training
images (vae_cache) and the YOLO test images were built -- CLAHE is the other.
Add --clahe to produce images that match the VAE's actual training
preprocessing as closely as possible (this project's YOLO preprocessing
already handles dropout-repair/normalization/resize equivalently, so only
these last two stages -- denoise, then contrast-enhance -- need re-applying,
not the whole pipeline from scratch):
    python -m src.preprocessing.denoise_test_set \\
        --src_images data/processed/yolo_dataset_v2_preprocessed/test/images \\
        --src_labels data/processed/yolo_dataset_v2_preprocessed/test/labels \\
        --dst data/processed/yolo_dataset_v2_preprocessed/test_vae_matched \\
        --clahe
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from src.preprocessing.denoising import denoise
from src.preprocessing.enhancement import enhance_contrast
from src.utils.config import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def run(args: argparse.Namespace) -> None:
    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    dst = Path(args.dst)
    dst_images = dst / "images"
    dst_labels = dst / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {src_images}")

    if args.denoise_method != "lee" and not args.weights_path:
        raise ValueError(
            f"--denoise_method {args.denoise_method} needs a trained checkpoint -- pass --weights_path, "
            "or drop back to --denoise_method lee (the default) to run now without one."
        )

    for i, path in enumerate(image_paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Skipping unreadable image: %s", path)
            continue
        denoised = denoise(
            image,
            method=args.denoise_method,
            weights_path=args.weights_path,
            device=args.device,
            fast=args.fast,
            fallback_on_missing_weights=False,
        )
        output = enhance_contrast(denoised, method="clahe", clip_limit=2.0, tile_grid_size=8) if args.clahe else denoised
        cv2.imwrite(str(dst_images / path.name), output)

        label_path = src_labels / f"{path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, dst_labels / label_path.name)
        else:
            # Background / no-object image -- ultralytics treats a missing
            # label file as "no objects present". Write an empty one
            # explicitly so nothing downstream has to guess why it's absent.
            (dst_labels / f"{path.stem}.txt").touch()

        if i % 200 == 0 or i == len(image_paths):
            logger.info("Denoised %d/%d", i, len(image_paths))

    logger.info("Done: %d images -> %s", len(image_paths), dst)

    if args.data_yaml:
        write_denoised_data_yaml(Path(args.data_yaml), dst_images, out_name=args.out_yaml_name)


def write_denoised_data_yaml(orig_yaml: Path, denoised_images_dir: Path, out_name: str = "data_denoised_test.yaml") -> Path:
    """Copies the original data.yaml but points `test:` at the new denoised
    images folder. train/val stay untouched -- ultralytics checks that their
    paths exist even for a test-only val() call, but never actually loads
    them in that mode.

    The written path is relative to orig_yaml's OWN folder, not the
    caller's cwd: this data.yaml has no `path:` key (deliberately -- see
    preprocess_yolo_dataset_v2.py's data.yaml comment), and ultralytics
    resolves path-less relative entries against the yaml file's own
    directory. Writing a cwd-relative path here instead double-prefixes
    the dataset folder and breaks path resolution (seen firsthand: a val()
    run failed looking for ".../yolo_dataset_v2_preprocessed/data/processed/
    yolo_dataset_v2_preprocessed/test_denoised/images").
    """
    import os

    import yaml

    with open(orig_yaml) as f:
        cfg = yaml.safe_load(f)
    rel = os.path.relpath(denoised_images_dir.resolve(), start=orig_yaml.resolve().parent)
    cfg["test"] = rel.replace("\\", "/")
    out_path = orig_yaml.parent / out_name
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info("Wrote %s (test -> %s)", out_path, cfg["test"])
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_images", type=str, required=True, help="Existing test/images folder (already preprocessed, not yet denoised).")
    p.add_argument("--src_labels", type=str, required=True, help="Matching test/labels folder -- copied unchanged.")
    p.add_argument("--dst", type=str, required=True, help="Output folder -- images/ and labels/ subfolders are created here.")
    p.add_argument("--data_yaml", type=str, default=None, help="Original data.yaml -- if given, writes a copy alongside it (see --out_yaml_name) with test: pointing at the new denoised images.")
    p.add_argument("--out_yaml_name", type=str, default="data_denoised_test.yaml", help="Filename for the generated yaml (written next to --data_yaml). Give each denoise method its own name (e.g. data_denoised_lee.yaml / data_denoised_b2u.yaml) so re-running one method doesn't overwrite another's results.")
    p.add_argument("--denoise_method", type=str, default="lee", choices=["lee", "blind2unblind", "dspnet"])
    p.add_argument("--weights_path", type=str, default=None, help="Required for blind2unblind/dspnet -- ignored by lee.")
    p.add_argument("--device", type=str, default=None, help="Learned-denoiser device ('cuda'/'cpu'). Ignored by lee.")
    p.add_argument("--fast", action="store_true", help="blind2unblind only -- single forward pass instead of the width**2-view reconstruction. Ignored by lee.")
    p.add_argument("--clahe", action="store_true", help="Also apply CLAHE contrast enhancement after denoising (same params prepare_vae_dataset.py uses) -- produces images matching the VAE's actual training preprocessing, not just the denoising step alone.")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

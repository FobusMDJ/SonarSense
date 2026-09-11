"""Picks out, from an already-preprocessed YOLO dataset split, the images
that contain the LARGEST annotated instances of one or more target classes
(e.g. shipwreck/ghost_net/pipe) -- "large" meaning the biggest normalized
bounding-box area (width * height, both in [0, 1] per YOLO's own label
format), which for side-scan sonar tiles generally means a close-range,
dominant-in-frame object rather than a small/distant one.

Why by rank instead of a fixed area threshold: a hardcoded "area > X" cutoff
is a guess that can silently return zero images (threshold too high) or
hundreds (too low) depending on the dataset. Ranking and taking the top N
per class always returns something proportional to what you asked for, and
--min_area is still available afterward to tighten further once you've seen
what the actual area distribution looks like (this script logs it either
way -- see selection_manifest.csv).

Copies (not moves) the selected images + their original label files into a
fresh images/ + labels/ folder pair -- ready to point vae_analysis.py or
denoise_test_set.py at directly.

Usage:
    python -m src.preprocessing.select_large_objects \\
        --images_dir data/processed/yolo_dataset_v2_preprocessed/test/images \\
        --labels_dir data/processed/yolo_dataset_v2_preprocessed/test/labels \\
        --data_yaml data/processed/yolo_dataset_v2_preprocessed/data.yaml \\
        --classes shipwreck ghost_net pipe \\
        --top_n 20 \\
        --dst data/processed/yolo_dataset_v2_preprocessed/test_large_objects
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from src.utils.config import get_logger

logger = get_logger(__name__)


def load_class_names(data_yaml: Path) -> list[str]:
    import yaml

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def find_largest_boxes(labels_dir: Path, target_class_ids: set[int]) -> list[dict]:
    """Returns one row per (image, target-class box) found: {stem, class_id,
    area, line}. An image with two target-class boxes produces two rows --
    ranking/selection happens per class afterward, in run()."""
    rows = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id not in target_class_ids:
                continue
            _, _, _, w, h = parts[:5]
            area = float(w) * float(h)
            rows.append({"stem": label_path.stem, "class_id": cls_id, "area": area})
    return rows


def run(args: argparse.Namespace) -> None:
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    dst = Path(args.dst)
    dst_images = dst / "images"
    dst_labels = dst / "labels"

    class_names = load_class_names(Path(args.data_yaml))
    name_to_id = {name: i for i, name in enumerate(class_names)}
    unknown = [c for c in args.classes if c not in name_to_id]
    if unknown:
        raise ValueError(f"Unknown class name(s) {unknown} -- data.yaml only has: {class_names}")
    target_class_ids = {name_to_id[c] for c in args.classes}

    rows = find_largest_boxes(labels_dir, target_class_ids)
    if not rows:
        raise FileNotFoundError(
            f"No boxes found for classes {args.classes} anywhere in {labels_dir} -- "
            "check --classes spelling and that this is the right labels folder."
        )

    id_to_name = {v: k for k, v in name_to_id.items()}
    selected_stems: set[str] = set()
    manifest_rows = []
    for cls_id in sorted(target_class_ids):
        cls_rows = sorted((r for r in rows if r["class_id"] == cls_id), key=lambda r: -r["area"])
        if args.min_area is not None:
            cls_rows = [r for r in cls_rows if r["area"] >= args.min_area]
        top = cls_rows[: args.top_n]
        cls_name = id_to_name[cls_id]
        logger.info(
            "%s: %d boxes total, selecting top %d by area (largest area=%.4f, smallest selected=%.4f)",
            cls_name, len(cls_rows), len(top), top[0]["area"] if top else float("nan"), top[-1]["area"] if top else float("nan"),
        )
        for rank, r in enumerate(top, 1):
            selected_stems.add(r["stem"])
            manifest_rows.append({"stem": r["stem"], "class": cls_name, "area": r["area"], "rank": rank})

    if not selected_stems:
        raise ValueError(
            "Nothing selected -- --min_area is probably set higher than every box's actual area. "
            "Drop --min_area (or lower it) and re-run; selection_manifest.csv logs the true area "
            "distribution once at least one image is selected."
        )

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for stem in sorted(selected_stems):
        label_src = labels_dir / f"{stem}.txt"
        image_src = None
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = images_dir / f"{stem}{suffix}"
            if candidate.exists():
                image_src = candidate
                break
        if image_src is None:
            logger.warning("No image found for label %s -- skipping", stem)
            continue
        shutil.copy2(image_src, dst_images / image_src.name)
        shutil.copy2(label_src, dst_labels / label_src.name)
        n_copied += 1

    manifest_path = dst / "selection_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "class", "area", "rank"])
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda r: (r["class"], r["rank"])))

    logger.info(
        "Selected %d unique images (%d copied) across classes %s -> %s (manifest: %s)",
        len(selected_stems), n_copied, args.classes, dst, manifest_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--labels_dir", type=str, required=True)
    p.add_argument("--data_yaml", type=str, required=True, help="Used to map --classes names to label class ids.")
    p.add_argument("--classes", type=str, nargs="+", required=True, help="Class names (must match data.yaml's names list exactly), e.g. --classes shipwreck ghost_net pipe.")
    p.add_argument("--top_n", type=int, default=20, help="How many of the largest-area images to select PER class (an image can be selected for more than one class and is only copied once).")
    p.add_argument("--min_area", type=float, default=None, help="Optional extra filter: normalized box area (width*height, both in [0,1]) must be at least this. Omit to just take the top --top_n regardless of absolute size.")
    p.add_argument("--dst", type=str, required=True, help="Output folder -- images/ and labels/ subfolders are created here, plus selection_manifest.csv.")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

"""Preprocesses build_yolo_dataset_v2.py's output
(data/processed/yolo_dataset_v2) into data/processed/yolo_dataset_v2_preprocessed
-- same scheme as preprocess_yolo_dataset.py (v1): grayscale + normalize
everywhere, no denoising, no CLAHE (train on preprocessed-but-not-denoised
images, denoising stays a separate production-pipeline step). Object-centered
tiling + dropout-repair are applied only to genuinely native-resolution
raw-sensor sources; everything else gets a simple resize to 512x512.

Which sources get which treatment -- checked empirically, not assumed, same
as v1:

  - AI4Shipwrecks ('ai4sw' prefix) -- unchanged from v1. Native swaths up to
    1728x9404, genuinely raw single-channel SSS. Full pipeline + tiling.

  - SubPipe ('subpipe_hf'/'subpipe_lf' prefixes) -- ALSO gets the full
    pipeline + tiling this time, unlike v1 (which only had one such
    source). Its images are native 2500x500 (LF) / 5000x500 (HF) full-swath
    sonar exports, not pre-cropped tiles -- same "large native swath" profile
    as AI4Shipwrecks, just wide instead of tall. One wrinkle: SubPipe's
    height (500px) is BELOW the 512 tile size in every single image, unlike
    AI4Shipwrecks where every dimension safely exceeds it. Tiling logic here
    pads the short dimension up to TILE_SIZE with zero rows/cols appended at
    the bottom/right (never prepended, so existing pixel coordinates --
    and therefore already-computed absolute box coordinates -- don't shift)
    before computing tile placements.

  - Cylinders ('cyl_*' prefixes) and Nets ('net_*' prefixes) -- simple
    resize only, NOT the full pipeline, despite genuinely being sonar-
    adjacent data. Checked actual images: both arrived as Roboflow exports
    already cropped to a fixed 640x640, not native full-resolution swaths --
    same profile as v1's Roboflow sonar_detect source, which also got the
    simple treatment for the same reason (dropout-repair's row/column
    statistics need to see a real full swath; a pre-cropped 640x640 tile has
    already lost that context, and re-applying dropout-repair to an
    arbitrary crop risks flagging normal image content as "dropout").
    Cylinders is also forward-looking/acoustic sonar, not side-scan, and
    Nets showed at least one non-grayscale (false-color-rendered) sample in
    a manual check -- both further reasons not to treat them as raw
    side-scan sensor output.

  - AquaScan-1K ('aquascan' prefix) -- unchanged from v1: simple resize
    only (RGB screenshots, literally named "Screenshot_2025-...png").

Usage:
    python -m src.preprocessing.preprocess_yolo_dataset_v2 \\
        --in_dir data/processed/yolo_dataset_v2 --out_dir data/processed/yolo_dataset_v2_preprocessed --stage tiled
    python -m src.preprocessing.preprocess_yolo_dataset_v2 \\
        --in_dir data/processed/yolo_dataset_v2 --out_dir data/processed/yolo_dataset_v2_preprocessed --stage others
    python -m src.preprocessing.preprocess_yolo_dataset_v2 \\
        --in_dir data/processed/yolo_dataset_v2 --out_dir data/processed/yolo_dataset_v2_preprocessed --stage finalize
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.dropout import DropoutConfig, repair_and_mask
from src.preprocessing.enhancement import standardize_resolution
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.normalization import normalize_intensity
from src.utils.config import get_logger

logger = get_logger(__name__)

TILE_SIZE = 512
MIN_BOX_OVERLAP_FRACTION = 0.3
BG_TILES_PER_IMAGE = 2
BG_MAX_BOX_OVERLAP_FRACTION = 0.1
DEDUP_DIST_PX = 128
BG_SAMPLE_ATTEMPTS = 20

# Sources getting the full grayscale->dropout-repair->normalize->tile
# pipeline. Everything else in the manifest gets the simple treatment.
FULL_PIPELINE_SOURCES = {"ai4shipwrecks", "subpipe_mini_sss"}

MANIFEST_FIELDS = ["split", "source", "prefix", "out_stem", "orig_image", "orig_label", "class_ids_present"]


def _manifest_writer(out_dir: Path):
    path = out_dir / "MANIFEST.csv"
    is_new = not path.exists()
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
    if is_new:
        writer.writeheader()
    return f, writer


def _read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cid, cx, cy, w, h = line.split()
        out.append((int(cid), float(cx), float(cy), float(w), float(h)))
    return out


def _to_abs_boxes(labels: list[tuple[int, float, float, float, float]], img_w: int, img_h: int) -> list[dict]:
    boxes = []
    for cid, cx, cy, w, h in labels:
        cx_a, cy_a, w_a, h_a = cx * img_w, cy * img_h, w * img_w, h * img_h
        boxes.append({
            "cid": cid,
            "x0": cx_a - w_a / 2, "y0": cy_a - h_a / 2, "x1": cx_a + w_a / 2, "y1": cy_a + h_a / 2,
            "area": w_a * h_a,
        })
    return boxes


def _pad_to_min_size(image: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    """Zero-pads at the bottom/right only (never top/left) so existing pixel
    -- and therefore already-computed absolute box -- coordinates never
    shift. A no-op if the image already meets min_h/min_w in both dims."""
    h, w = image.shape[:2]
    pad_h, pad_w = max(0, min_h - h), max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return image
    return cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)


def _tile_rect_for_center(cx: float, cy: float, img_w: int, img_h: int) -> tuple[int, int]:
    x0 = int(round(cx - TILE_SIZE / 2))
    y0 = int(round(cy - TILE_SIZE / 2))
    x0 = max(0, min(x0, img_w - TILE_SIZE))
    y0 = max(0, min(y0, img_h - TILE_SIZE))
    return x0, y0


def _is_dedup(x0: int, y0: int, chosen: list[tuple[int, int]]) -> bool:
    return any(abs(x0 - cx0) < DEDUP_DIST_PX and abs(y0 - cy0) < DEDUP_DIST_PX for cx0, cy0 in chosen)


def _boxes_in_tile(boxes: list[dict], x0: int, y0: int) -> list[tuple[int, float, float, float, float]]:
    x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
    out = []
    for b in boxes:
        ix0, iy0 = max(b["x0"], x0), max(b["y0"], y0)
        ix1, iy1 = min(b["x1"], x1), min(b["y1"], y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter_area = (ix1 - ix0) * (iy1 - iy0)
        if b["area"] <= 0 or inter_area / b["area"] < MIN_BOX_OVERLAP_FRACTION:
            continue
        cx = ((ix0 + ix1) / 2 - x0) / TILE_SIZE
        cy = ((iy0 + iy1) / 2 - y0) / TILE_SIZE
        w = (ix1 - ix0) / TILE_SIZE
        h = (iy1 - iy0) / TILE_SIZE
        out.append((b["cid"], cx, cy, w, h))
    return out


def _max_box_overlap_fraction(x0: int, y0: int, boxes: list[dict]) -> float:
    if not boxes:
        return 0.0
    x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
    best = 0.0
    for b in boxes:
        ix0, iy0 = max(b["x0"], x0), max(b["y0"], y0)
        ix1, iy1 = min(b["x1"], x1), min(b["y1"], y1)
        if ix1 <= ix0 or iy1 <= iy0 or b["area"] <= 0:
            continue
        best = max(best, ((ix1 - ix0) * (iy1 - iy0)) / b["area"])
    return best


def stage_tiled(in_dir: Path, out_dir: Path) -> None:
    """Full pipeline + object-centered tiling, for every row whose source is
    in FULL_PIPELINE_SOURCES (currently: AI4Shipwrecks, SubPipe)."""
    f, writer = _manifest_writer(out_dir)
    try:
        cfg = DropoutConfig()
        n_tiles_total = 0
        with open(in_dir / "MANIFEST.csv") as mf:
            rows = [r for r in csv.DictReader(mf) if r["source"] in FULL_PIPELINE_SOURCES]

        for row in rows:
            split, out_stem, source = row["split"], row["out_stem"], row["source"]
            img_path = in_dir / split / "images" / f"{out_stem}.png"
            lbl_path = in_dir / split / "labels" / f"{out_stem}.txt"
            out_img_dir = out_dir / split / "images"
            out_lbl_dir = out_dir / split / "labels"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            if any(out_img_dir.glob(f"{out_stem}__t*.png")):
                continue  # resumable, same logic as v1

            gray = to_grayscale(cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED))
            repaired, _validity = repair_and_mask(gray, cfg)
            normed = normalize_intensity(repaired, method="percentile")
            orig_h, orig_w = normed.shape

            labels = _read_yolo_labels(lbl_path)
            # Boxes converted to absolute pixels using the ORIGINAL
            # (pre-padding) dimensions -- padding only appends rows/cols
            # after existing content, so these coordinates stay valid in
            # the padded frame without any shift.
            boxes = _to_abs_boxes(labels, orig_w, orig_h)

            normed = _pad_to_min_size(normed, TILE_SIZE, TILE_SIZE)
            img_h, img_w = normed.shape

            chosen: list[tuple[int, int]] = []
            tile_kinds: list[str] = []
            for b in boxes:
                cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
                x0, y0 = _tile_rect_for_center(cx, cy, img_w, img_h)
                if _is_dedup(x0, y0, chosen):
                    continue
                chosen.append((x0, y0))
                tile_kinds.append("object")

            rng = random.Random(out_stem)
            n_bg_added = 0
            attempts = 0
            while n_bg_added < BG_TILES_PER_IMAGE and attempts < BG_SAMPLE_ATTEMPTS:
                attempts += 1
                x0 = rng.randint(0, max(0, img_w - TILE_SIZE))
                y0 = rng.randint(0, max(0, img_h - TILE_SIZE))
                if _is_dedup(x0, y0, chosen):
                    continue
                if _max_box_overlap_fraction(x0, y0, boxes) > BG_MAX_BOX_OVERLAP_FRACTION:
                    continue
                chosen.append((x0, y0))
                tile_kinds.append("background")
                n_bg_added += 1

            for i, ((x0, y0), kind) in enumerate(zip(chosen, tile_kinds)):
                tile_img = normed[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]
                tile_boxes = _boxes_in_tile(boxes, x0, y0)
                tile_stem = f"{out_stem}__t{i}"
                cv2.imwrite(str(out_img_dir / f"{tile_stem}.png"), tile_img)
                lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h in tile_boxes]
                (out_lbl_dir / f"{tile_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                class_ids = sorted({str(cid) for cid, *_ in tile_boxes})
                writer.writerow({
                    "split": split, "source": source, "prefix": row["prefix"], "out_stem": tile_stem,
                    "orig_image": row["orig_image"],
                    "orig_label": f"tile {i} ({kind}) of {out_stem}, from preprocessed+dropout-repaired full image",
                    "class_ids_present": ";".join(class_ids),
                })
                n_tiles_total += 1
            f.flush()
            logger.info("[%s] %s (%dx%d orig, %dx%d padded) -> %d tiles (%d object-centered, %d background)",
                        source, out_stem, orig_w, orig_h, img_w, img_h, len(chosen), len(boxes), n_bg_added)
        logger.info("[tiled] done: %d source images -> %d tiles", len(rows), n_tiles_total)
    finally:
        f.close()


def stage_others(in_dir: Path, out_dir: Path) -> None:
    """Simple grayscale->normalize->resize-to-512, for every row whose
    source is NOT in FULL_PIPELINE_SOURCES."""
    f, writer = _manifest_writer(out_dir)
    try:
        with open(in_dir / "MANIFEST.csv") as mf:
            rows = [r for r in csv.DictReader(mf) if r["source"] not in FULL_PIPELINE_SOURCES]

        src_index: dict[str, dict[str, Path]] = {}
        n = 0
        n_skipped = 0
        for row in rows:
            split, out_stem, prefix, source = row["split"], row["out_stem"], row["prefix"], row["source"]

            out_img_dir = out_dir / split / "images"
            out_lbl_dir = out_dir / split / "labels"
            out_img_path = out_img_dir / f"{out_stem}.png"
            out_lbl_path = out_lbl_dir / f"{out_stem}.txt"
            if out_img_path.exists() and out_lbl_path.exists():
                n_skipped += 1
                continue

            if split not in src_index:
                in_img_dir = in_dir / split / "images"
                src_index[split] = {p.stem: p for p in in_img_dir.iterdir() if p.is_file()}
            img_path = src_index[split].get(out_stem)
            if img_path is None:
                logger.warning("missing source image for %s, skipping", out_stem)
                continue

            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            gray = to_grayscale(cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED))
            normed = normalize_intensity(gray, method="percentile")
            resized = standardize_resolution(normed, target_size=(TILE_SIZE, TILE_SIZE))
            cv2.imwrite(str(out_img_path), resized)

            in_lbl_path = in_dir / split / "labels" / f"{out_stem}.txt"
            label_text = in_lbl_path.read_text() if in_lbl_path.exists() else ""
            out_lbl_path.write_text(label_text)
            class_ids = sorted({line.split()[0] for line in label_text.splitlines() if line.strip()})

            writer.writerow({
                "split": split, "source": source, "prefix": prefix, "out_stem": out_stem,
                "orig_image": row["orig_image"], "orig_label": "(unchanged -- normalized coords are resolution-independent)",
                "class_ids_present": ";".join(class_ids),
            })
            n += 1
            if n % 250 == 0:
                f.flush()
                logger.info("[others] %d new + %d already-done / %d total", n, n_skipped, len(rows))
        f.flush()
        logger.info("[others] done this run: %d new, %d already-done, %d total", n, n_skipped, len(rows))
    finally:
        f.close()


def finalize(out_dir: Path, class_names: list[str]) -> None:
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "# Auto-generated by preprocess_yolo_dataset_v2.py -- grayscale +\n"
        "# normalize everywhere; dropout-repair + object-centered tiling\n"
        "# additionally for AI4Shipwrecks AND SubPipe (both native full-swath\n"
        "# raw-sensor sources); simple resize for Cylinders/Nets/AquaScan-1K\n"
        "# (pre-cropped Roboflow exports / RGB screenshots). No denoising, no\n"
        "# CLAHE -- trains on preprocessed-but-not-denoised images by design.\n"
        "#\n"
        "# No 'path:' key on purpose -- see preprocess_yolo_dataset.py's data.yaml\n"
        "# comment for why (ultralytics resolves a relative path against the\n"
        "# caller's cwd, not this file's own folder).\n"
        "train: train/images\n"
        "val: test/images\n"
        "test: test/images\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    counts: dict[str, dict[str, int]] = {"train": {c: 0 for c in class_names}, "test": {c: 0 for c in class_names}}
    image_counts = {"train": 0, "test": 0}
    with open(out_dir / "MANIFEST.csv") as fh:
        for row in csv.DictReader(fh):
            image_counts[row["split"]] += 1
            for cid in row["class_ids_present"].split(";"):
                if cid:
                    counts[row["split"]][class_names[int(cid)]] += 1

    lines = ["# yolo_dataset_v2_preprocessed -- summary\n"]
    lines.append(f"Images: train={image_counts['train']}, test={image_counts['test']}\n")
    lines.append("\n| class | train boxes | test boxes |\n|---|---|---|\n")
    for c in class_names:
        lines.append(f"| {c} | {counts['train'][c]} | {counts['test'][c]} |\n")
    (out_dir / "SOURCES.md").write_text("".join(lines))
    print("".join(lines))
    print(f"data.yaml -> {data_yaml}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", type=str, required=True, help="build_yolo_dataset_v2.py's output directory.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--stage", type=str, default="all", choices=["all", "tiled", "others", "finalize"])
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["shipwreck", "human", "cylinder", "ghost_net", "pipe"]

    if args.stage in ("all", "tiled"):
        stage_tiled(in_dir, out_dir)
    if args.stage in ("all", "others"):
        stage_others(in_dir, out_dir)
    if args.stage in ("all", "finalize"):
        finalize(out_dir, class_names)

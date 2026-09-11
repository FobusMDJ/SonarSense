"""Preprocesses the merged YOLO dataset (build_yolo_dataset.py's output) into
data/processed/yolo_dataset_preprocessed/{train,test} -- grayscale +
normalize everywhere, dropout-repair + object-centered tiling ONLY for
AI4Shipwrecks, resize-to-512 for everything else. Denoising and CLAHE are
deliberately NOT applied (per-decision: train YOLO on preprocessed-but-not-
denoised images -- see prior discussion; denoising remains a
production-pipeline step evaluated separately, not baked into this
training set).

Why AI4Shipwrecks gets different treatment than the other four sources:
checking actual file properties (not assumed) showed AI4Shipwrecks is
genuinely raw-ish single-channel SSS (mode 'L', up to 1728x9404) -- the
same kind of data src/preprocessing/dropout.py was built for. Roboflow
sonar_detect, AquaScan-1K, and the five year-folders are all RGB, fixed
small sizes (416/640/1024 square), and AquaScan's filenames are literally
"Screenshot_2025-...png" -- rendered/screen-captured images, not raw
acoustic amplitude data. Running dropout-repair (which looks for flat-std
or saturated rows/cols as evidence of sensor dropout) on a screenshot has
no real phenomenon to detect and risks inpainting over a legitimate UI
element or color-mapped region. So: AI4Shipwrecks gets the full
grayscale -> dropout-repair -> normalize pipeline; everything else gets
grayscale -> normalize only.

Why AI4Shipwrecks is TILED, not just resized: its images run up to 9404px
tall against a 512 target -- a straight resize would shrink a shipwreck
down ~18x, likely past the point of being detectable at all (the small-
object problem several sonar-YOLO papers specifically tile to avoid, see
Transformer-YOLOv5 among the sources cited in the preprocessing-technique
research turn). But naive dense-grid tiling of a 9000px-tall image at any
reasonable overlap produces tens of thousands of near-duplicate,
mostly-empty tiles from a mere 261 source images -- wasteful and would
dominate/skew the whole merged dataset by sheer volume. Instead this
tiles OBJECT-CENTERED: one 512x512 tile per labeled box (so no box is
split across a boundary the way a fixed grid risks), deduplicating tiles
whose centers land close together, plus a couple of background-only tiles
per image for negative examples. This keeps AI4Shipwrecks' contribution
proportionate to the other sources instead of a runaway multiplier.

Usage:
    python -m src.preprocessing.preprocess_yolo_dataset \\
        --in_dir data/processed/yolo_dataset --out_dir data/processed/yolo_dataset_preprocessed --stage ai4shipwrecks
    python -m src.preprocessing.preprocess_yolo_dataset \\
        --in_dir data/processed/yolo_dataset --out_dir data/processed/yolo_dataset_preprocessed --stage others
    python -m src.preprocessing.preprocess_yolo_dataset \\
        --in_dir data/processed/yolo_dataset --out_dir data/processed/yolo_dataset_preprocessed --stage finalize
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
MIN_BOX_OVERLAP_FRACTION = 0.3   # a box survives a tile only if this much of its area is inside it
BG_TILES_PER_IMAGE = 2
BG_MAX_BOX_OVERLAP_FRACTION = 0.1  # a "background" tile candidate must not cover more than this much of any box
DEDUP_DIST_PX = 128              # skip a new tile whose top-left is this close to an already-chosen one
BG_SAMPLE_ATTEMPTS = 20

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
        # clip to tile, convert to tile-local normalized coords
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


def stage_ai4shipwrecks(in_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        cfg = DropoutConfig()
        n_tiles_total = 0
        with open(in_dir / "MANIFEST.csv") as mf:
            rows = [r for r in csv.DictReader(mf) if r["source"] == "ai4shipwrecks"]

        for row in rows:
            split, out_stem = row["split"], row["out_stem"]
            img_path = in_dir / split / "images" / f"{out_stem}.png"
            lbl_path = in_dir / split / "labels" / f"{out_stem}.txt"
            out_img_dir = out_dir / split / "images"
            out_lbl_dir = out_dir / split / "labels"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            # Resume support: skip a source image only if EVERY tile we'd
            # produce for it already exists is unverifiable without redoing
            # the work, so instead skip if at least one tile file for this
            # stem is already on disk -- cheap, and re-running after a
            # partial source-image failure just regenerates that one image's
            # tiles (harmless: filenames are deterministic per box/bg index).
            if any(out_img_dir.glob(f"{out_stem}__t*.png")):
                continue

            gray = to_grayscale(cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED))
            repaired, _validity = repair_and_mask(gray, cfg)
            normed = normalize_intensity(repaired, method="percentile")
            img_h, img_w = normed.shape

            labels = _read_yolo_labels(lbl_path)
            boxes = _to_abs_boxes(labels, img_w, img_h)

            chosen: list[tuple[int, int]] = []
            tile_kinds: list[str] = []
            for b in boxes:
                cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
                x0, y0 = _tile_rect_for_center(cx, cy, img_w, img_h)
                if _is_dedup(x0, y0, chosen):
                    continue
                chosen.append((x0, y0))
                tile_kinds.append("object")

            rng = random.Random(out_stem)  # deterministic per-image, so reruns reproduce the same tiles
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
                    "split": split, "source": "ai4shipwrecks", "prefix": "ai4sw", "out_stem": tile_stem,
                    "orig_image": row["orig_image"],
                    "orig_label": f"tile {i} ({kind}) of {out_stem}, from preprocessed+dropout-repaired full image",
                    "class_ids_present": ";".join(class_ids),
                })
                n_tiles_total += 1
            f.flush()
            logger.info("[ai4shipwrecks] %s (%dx%d) -> %d tiles (%d object-centered, %d background)",
                        out_stem, img_w, img_h, len(chosen), len(boxes), n_bg_added)
        logger.info("[ai4shipwrecks] done: %d source images -> %d tiles", len(rows), n_tiles_total)
    finally:
        f.close()


def stage_others(in_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        with open(in_dir / "MANIFEST.csv") as mf:
            rows = [r for r in csv.DictReader(mf) if r["source"] != "ai4shipwrecks"]

        # Build a stem -> path lookup for each split's source image dir ONCE
        # via a single directory scan, instead of a fresh glob() per row --
        # glob() re-lists the whole (thousands-of-files) directory on every
        # call, which made this stage effectively O(n^2) and got dramatically
        # slower as it progressed. A single iterdir() per split fixes that.
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
                # Skip BEFORE touching the (expensive-to-index) source dir.
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

            # YOLO box coords are already normalized (resolution-independent)
            # -- resizing the image never requires touching the label file.
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
        "# Auto-generated by preprocess_yolo_dataset.py -- grayscale + normalize\n"
        "# everywhere; dropout-repair + object-centered tiling additionally for\n"
        "# AI4Shipwrecks (the only genuinely raw-SSS source). No denoising, no\n"
        "# CLAHE -- this trains on preprocessed-but-not-denoised images by design.\n"
        "#\n"
        "# No 'path:' key on purpose: ultralytics resolves a relative 'path' value\n"
        "# against the CALLER's current working directory (confirmed empirically --\n"
        "# 'path: .' broke when training was launched from the repo root instead of\n"
        "# from this folder), not against this yaml file's own location. Omitting\n"
        "# 'path' makes ultralytics fall back to this file's parent directory\n"
        "# instead, which is what we actually want and is invocation-CWD-independent.\n"
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

    lines = ["# Preprocessed YOLO dataset -- summary\n"]
    lines.append(f"Images: train={image_counts['train']}, test={image_counts['test']}\n")
    lines.append("\n| class | train boxes | test boxes |\n|---|---|---|\n")
    for c in class_names:
        lines.append(f"| {c} | {counts['train'][c]} | {counts['test'][c]} |\n")
    (out_dir / "SOURCES.md").write_text("".join(lines))
    print("".join(lines))
    print(f"data.yaml -> {data_yaml}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", type=str, required=True, help="build_yolo_dataset.py's output directory.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--stage", type=str, default="all", choices=["all", "ai4shipwrecks", "others", "finalize"])
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["shipwreck", "aircraft", "fish", "other", "human", "unknown_a", "unknown_b"]

    if args.stage in ("all", "ai4shipwrecks"):
        stage_ai4shipwrecks(in_dir, out_dir)
    if args.stage in ("all", "others"):
        stage_others(in_dir, out_dir)
    if args.stage in ("all", "finalize"):
        finalize(out_dir, class_names)

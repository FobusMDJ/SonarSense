"""Merges every labeled sonar dataset under data/raw/Labelled into one
consistent YOLO detection dataset: data/processed/yolo_dataset/{train,test}.

Why this exists: the collected datasets are five genuinely different things
wearing the same "images + labels" shape --

  - Roboflow "sonar_detect" (train/valid/test, 4 classes: aircraft, fish,
    other, shipwreck) -- already clean, image/label counts match 1:1.
  - AI4Shipwrecks (train/test) -- labels are PNG segmentation masks, not
    YOLO boxes. Converted here via connected-component analysis: one box
    per mask blob above a minimum-area threshold (filters annotation-noise
    specks, keeps real debris-field pieces), all mapped to class
    "shipwreck" (same real-world category as Roboflow's, deliberately
    unified rather than kept as a second distinct class).
  - AquaScan-1K (flat, single class "Human") -- clean 1:1 image/label
    pairs, no existing split.
  - Five year-organized folders (2010/2015/2017/2018/2021) -- paired
    jpg+txt, class ids 0/1 with NO classes.txt/obj.names anywhere in the
    source. Kept as "unknown_a"/"unknown_b" placeholders rather than
    guessed at -- there's a plausible resemblance to the leftover
    Training/obj.names.txt scheme (MILCO/NOMBO, a real naval mine-vs-
    non-mine sonar taxonomy), but nothing in the source data confirms
    that, so it is NOT asserted as fact. Rename these two classes once
    you've confirmed what they actually are; don't train on them as
    unlabeled placeholders longer than necessary.
  - Deliberately EXCLUDED: the four whole-image "engineering platform" /
    "pipeline or cable" / "seabed surface" / "underwater residual mound"
    folders (no boxes, per-user decision to leave out rather than
    approximate with full-frame boxes); Training/ (leftover YOLOv4
    weights/notebook, not a dataset); sonar.mines/sonar.rocks/sonar.names
    (the 1988 UCI "Sonar, Mines vs. Rocks" tabular dataset -- 1D signal
    features, not imagery, unrelated to this task despite the filenames).

Negative (no-object) images are kept, not dropped -- an empty label file
is a legitimate YOLO training example (real background, teaches the
model what NOT to fire on), true for the year-folders (most are majority-
empty) and roughly 38% of AI4Shipwrecks.

Split policy: Roboflow's own train+valid -> merged train, Roboflow's own
test -> merged test (their real untouched holdout). AI4Shipwrecks: same,
train->train, test->test. AquaScan-1K and the year-folders have no
existing split, so each gets an 85/15 deterministic random split (fixed
seed, per-source) -- keeps every source represented in both merged splits
rather than, say, one entire year landing only in test.

Filenames are prefixed by source (rf__, ai4sw__, aquascan__, y2010__, ...)
to guarantee uniqueness across sources with very different naming
conventions and keep provenance visible at a glance -- see MANIFEST.csv
for the full per-file record (source, split, original path, class ids
present) and SOURCES.md for the human-readable summary this script
prints and also writes to disk.

Usage:
    # everything in one call (only safe if your shell has no short
    # per-call time limit -- device_bash sessions should chunk instead):
    python -m src.preprocessing.build_yolo_dataset --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset

    # chunked (recommended under a ~180s-per-call shell): one source at a
    # time, each call appends to the same manifest/output dirs.
    python -m src.preprocessing.build_yolo_dataset --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset --stage roboflow
    python -m src.preprocessing.build_yolo_dataset --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset --stage ai4shipwrecks
    python -m src.preprocessing.build_yolo_dataset --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset --stage aquascan
    python -m src.preprocessing.build_yolo_dataset --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset --stage years
    python -m src.preprocessing.build_yolo_dataset --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset --stage finalize   # writes data.yaml + SOURCES.md from the accumulated manifest
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Global class taxonomy every source's original class ids get remapped into.
GLOBAL_CLASSES = ["shipwreck", "aircraft", "fish", "other", "human", "unknown_a", "unknown_b"]
CLASS_ID = {name: i for i, name in enumerate(GLOBAL_CLASSES)}

ROBOFLOW_REMAP = {0: CLASS_ID["aircraft"], 1: CLASS_ID["fish"], 2: CLASS_ID["other"], 3: CLASS_ID["shipwreck"]}
YEAR_REMAP = {0: CLASS_ID["unknown_a"], 1: CLASS_ID["unknown_b"]}

MIN_MASK_COMPONENT_AREA = 300  # px; filters AI4Shipwrecks mask annotation-noise specks, see module docstring
SPLIT_SEED = 42
SPLIT_TRAIN_FRACTION = 0.85

MANIFEST_FIELDS = ["split", "source", "prefix", "out_stem", "orig_image", "orig_label", "class_ids_present"]


def _manifest_writer(out_dir: Path):
    path = out_dir / "MANIFEST.csv"
    is_new = not path.exists()
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
    if is_new:
        writer.writeheader()
    return f, writer


def _copy_pair(
    img_src: Path, label_lines: list[str], split: str, out_dir: Path, prefix: str, source: str, writer, manifest_file,
) -> None:
    out_stem = f"{prefix}__{img_src.stem}"
    img_dst = out_dir / split / "images" / f"{out_stem}{img_src.suffix.lower()}"
    lbl_dst = out_dir / split / "labels" / f"{out_stem}.txt"
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    if img_dst.exists() and lbl_dst.exists():
        # Resuming a stage that got cut off mid-run (e.g. a shell timeout on
        # a large source): this exact output already exists from a prior
        # partial run of the SAME stage -- skip rather than re-copy or
        # error, so re-running the same command is safe and just continues
        # where it left off. A real cross-source collision can't reach this
        # branch (the per-source-split prefix makes that unreachable); if
        # img_dst exists but lbl_dst doesn't, something genuinely went
        # wrong mid-write, so fall through and let it be overwritten/fixed.
        return
    shutil.copy2(img_src, img_dst)
    lbl_dst.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
    class_ids = sorted({line.split()[0] for line in label_lines})
    writer.writerow({
        "split": split, "source": source, "prefix": prefix, "out_stem": out_stem,
        "orig_image": str(img_src), "orig_label": "(derived from mask)" if source == "ai4shipwrecks" else str(img_src.with_suffix(".txt")),
        "class_ids_present": ";".join(class_ids),
    })
    # Flush after every row (not just at stage end) so a mid-run timeout --
    # routine on the largest sources under a ~180s-per-call shell -- never
    # leaves the manifest behind what's actually on disk; the skip-if-
    # exists check above depends on the manifest staying trustworthy for a
    # resumed run to be safe, not just the image/label files themselves.
    manifest_file.flush()


def _remap_label_file(path: Path, remap: dict[int, int]) -> list[str]:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        orig_id = int(parts[0])
        new_id = remap[orig_id]
        lines.append(" ".join([str(new_id)] + parts[1:]))
    return lines


def stage_roboflow(labelled_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        # Roboflow's own train+valid -> merged train (both legitimately part
        # of "data used during training/model-selection"); their test ->
        # merged test (their real held-out set, untouched).
        for roboflow_split, out_split in [("train", "train"), ("valid", "train"), ("test", "test")]:
            img_dir = labelled_dir / roboflow_split / "images"
            lbl_dir = labelled_dir / roboflow_split / "labels"
            if not img_dir.exists():
                continue
            n = 0
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                lines = _remap_label_file(lbl_path, ROBOFLOW_REMAP)
                # prefix includes the source sub-split (not just "rf") so a
                # filename reused across Roboflow's own train/valid/test
                # (unlikely with their rf-hashed names, but not guaranteed)
                # can't silently collide and overwrite in the merged output.
                _copy_pair(img_path, lines, out_split, out_dir, f"rf_{roboflow_split}", "roboflow_sonar_detect", writer, f)
                n += 1
            print(f"[roboflow] {roboflow_split} -> {out_split}: {n} images")
    finally:
        f.close()


def _mask_to_boxes(mask_path: Path) -> list[str]:
    mask = np.array(Image.open(mask_path))
    mask_u8 = (mask > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    h, w = mask.shape[:2]
    lines = []
    for i in range(1, n):  # label 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_MASK_COMPONENT_AREA:
            continue
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        lines.append(f"{CLASS_ID['shipwreck']} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


def stage_ai4shipwrecks(labelled_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        for split in ("train", "test"):
            img_dir = labelled_dir / "AI4Shipwrecks" / split / "images"
            lbl_dir = labelled_dir / "AI4Shipwrecks" / split / "labels"
            if not img_dir.exists():
                continue
            n = 0
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() != ".png":
                    continue
                mask_path = lbl_dir / img_path.name
                lines = _mask_to_boxes(mask_path) if mask_path.exists() else []
                _copy_pair(img_path, lines, split, out_dir, "ai4sw", "ai4shipwrecks", writer, f)
                n += 1
            print(f"[ai4shipwrecks] {split}: {n} images")
    finally:
        f.close()


def _deterministic_split(stems: list[str], seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * SPLIT_TRAIN_FRACTION)
    return set(shuffled[:n_train]), set(shuffled[n_train:])


def stage_aquascan(labelled_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        img_dir = labelled_dir / "AquaScan-1K" / "images"
        lbl_dir = labelled_dir / "AquaScan-1K" / "labels"
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        train_stems, _ = _deterministic_split([p.stem for p in images], SPLIT_SEED)
        counts = {"train": 0, "test": 0}
        for img_path in images:
            split = "train" if img_path.stem in train_stems else "test"
            lines = _remap_label_file(lbl_dir / f"{img_path.stem}.txt", {0: CLASS_ID["human"]})
            _copy_pair(img_path, lines, split, out_dir, "aquascan", "aquascan_1k", writer, f)
            counts[split] += 1
        print(f"[aquascan] train: {counts['train']}, test: {counts['test']}")
    finally:
        f.close()


def stage_years(labelled_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        for year in ("2010", "2015", "2017", "2018", "2021"):
            year_dir = labelled_dir / year / year
            if not year_dir.exists():
                continue
            images = sorted(p for p in year_dir.iterdir() if p.suffix.lower() == ".jpg")
            train_stems, _ = _deterministic_split([p.stem for p in images], SPLIT_SEED)
            counts = {"train": 0, "test": 0}
            for img_path in images:
                split = "train" if img_path.stem in train_stems else "test"
                lines = _remap_label_file(img_path.with_suffix(".txt"), YEAR_REMAP)
                _copy_pair(img_path, lines, split, out_dir, f"y{year}", f"year_{year}", writer, f)
                counts[split] += 1
            print(f"[year {year}] train: {counts['train']}, test: {counts['test']}")
    finally:
        f.close()


def finalize(out_dir: Path) -> None:
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "# Auto-generated by build_yolo_dataset.py -- merges Roboflow sonar_detect,\n"
        "# AI4Shipwrecks, AquaScan-1K, and the 2010/2015/2017/2018/2021 year folders.\n"
        "# 'test' doubles as 'val' here since only a train/test split was requested --\n"
        "# if you want a true held-out test set never touched during model selection,\n"
        "# carve a separate val split out of train before running long training jobs.\n"
        # Relative, not out_dir.resolve() -- an absolute path baked in here
        # would be whatever filesystem this script happened to run under
        # (e.g. a device_bash sandbox's /sessions/.../mnt/... view), which
        # does not exist on the machine actually running YOLO training.
        # Ultralytics resolves train/val/test relative to `path`, so "."
        # (this yaml's own directory) works unchanged on any machine/OS.
        "path: .\n"
        "train: train/images\n"
        "val: test/images\n"
        "test: test/images\n"
        f"nc: {len(GLOBAL_CLASSES)}\n"
        f"names: {GLOBAL_CLASSES}\n"
    )

    # Per-split, per-class counts straight from the manifest -- the actual
    # numbers, not a re-derivation, so this can't drift from what's on disk.
    counts: dict[str, dict[str, int]] = {"train": {c: 0 for c in GLOBAL_CLASSES}, "test": {c: 0 for c in GLOBAL_CLASSES}}
    image_counts = {"train": 0, "test": 0}
    with open(out_dir / "MANIFEST.csv") as fh:
        for row in csv.DictReader(fh):
            image_counts[row["split"]] += 1
            for cid in row["class_ids_present"].split(";"):
                if cid:
                    counts[row["split"]][GLOBAL_CLASSES[int(cid)]] += 1

    lines = ["# Merged YOLO dataset -- source summary\n"]
    lines.append(f"Images: train={image_counts['train']}, test={image_counts['test']}\n")
    lines.append("\n| class | train boxes | test boxes |\n|---|---|---|\n")
    for c in GLOBAL_CLASSES:
        lines.append(f"| {c} | {counts['train'][c]} | {counts['test'][c]} |\n")
    lines.append(
        "\nunknown_a / unknown_b (from the 2010/2015/2017/2018/2021 year folders) have "
        "NO confirmed semantics -- no classes.txt/obj.names existed anywhere in that "
        "source. There's a plausible resemblance to the leftover Training/obj.names.txt "
        "scheme (MILCO / NOMBO, a real naval mine-vs-non-mine-bottom-object sonar "
        "taxonomy) but this is NOT confirmed. Verify and rename before relying on these "
        "two classes.\n\n"
        "Excluded entirely: the 4 whole-image classification folders (627 images, no "
        "boxes -- 'engineering platform' / 'pipeline or cable' / 'seabed surface' / "
        "'underwater residual mound'), Training/ (leftover YOLOv4 artifacts, not a "
        "dataset), and sonar.mines/sonar.rocks/sonar.names (the unrelated 1988 UCI "
        "tabular sonar dataset).\n"
    )
    (out_dir / "SOURCES.md").write_text("".join(lines))
    print("".join(lines))
    print(f"data.yaml -> {data_yaml}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labelled_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--stage", type=str, default="all", choices=["all", "roboflow", "ai4shipwrecks", "aquascan", "years", "finalize"])
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    labelled_dir = Path(args.labelled_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = {
        "roboflow": stage_roboflow,
        "ai4shipwrecks": stage_ai4shipwrecks,
        "aquascan": stage_aquascan,
        "years": stage_years,
    }
    if args.stage == "all":
        for fn in stages.values():
            fn(labelled_dir, out_dir)
        finalize(out_dir)
    elif args.stage == "finalize":
        finalize(out_dir)
    else:
        stages[args.stage](labelled_dir, out_dir)

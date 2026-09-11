"""Merges a SECOND, separate super-dataset -- NOT the same one as
build_yolo_dataset.py / data/processed/yolo_dataset -- built specifically
around the project's problem statement's named target classes (SIH26057:
ghost nets, pipes, cylinders, shipwrecks), rather than whatever classes
happened to come along with earlier sources. Output:
data/processed/yolo_dataset_v2/{train,test}.

Exactly 5 sources, all under data/raw/Labelled:

  - AI4Shipwrecks (train/test) -- same mask-to-bbox conversion as the first
    merge (connected-component analysis, MIN_MASK_COMPONENT_AREA filters
    annotation-noise specks). -> class "shipwreck".

  - AquaScan-1K (flat, single class, no existing split -> 85/15 seeded
    split). -> class "human". Not a PS target class, but it's genuine SSS
    sensor data with a real, unambiguous label (unlike the old merge's
    aircraft/fish/other/unknown_a/unknown_b, which came from sources
    outside this 5-source list and are deliberately NOT carried over here
    -- see "Don't use the last super dataset" in the request this script
    was built for).

  - Cylinders (UATD via Roboflow: train/valid/test, 10 classes) -> only
    the "cylinder" class (index 3 in its own data.yaml) is kept. This is
    NOT a simple remap: checking the actual label files first (before
    writing this) showed only ~7% of images (563/7582 in train alone)
    contain a cylinder at all -- the rest contain OTHER UATD classes
    (ball, cube, tire, ROV, cages, metal bucket, human body, plane).
    Carrying those images over with an empty label file would be a real
    labeling error: an "empty label" is supposed to mean genuine
    background, but these images actually contain an unlabeled object we
    chose not to track, which would teach the model a false negative
    (visually-object-shaped scene -> "nothing here"). So: an image is kept
    only if it has >=1 cylinder box (all its OTHER class lines are
    dropped, only cylinder lines survive) or if it's genuinely empty (0
    objects of ANY class -- a real background frame, rare in this
    dataset). Images containing only non-cylinder objects are skipped
    entirely, not converted to blank negatives.
    Also note: UATD is acoustic/forward-looking sonar, not side-scan --
    a real domain gap flagged when this dataset was first suggested, kept
    here anyway for lack of any side-scan cylinder alternative.

  - Nets (the real data is nested at
    Nets/sss-crab-pot-detection-ds/{train,valid,test}, NOT the stray
    top-level Nets/{train,valid,test} -- that top-level copy turned out to
    be an accidental duplicate of the Cylinders/UATD download and is
    ignored entirely by this script). Real side-scan sonar (Humminbird
    consumer SSS), labels in a HuggingFace imagefolder metadata.jsonl
    format per split (NOT plain YOLO txt) -- bbox is [x, y, width, height]
    in ABSOLUTE PIXELS (top-left corner), category is a string. Converted
    here to YOLO-normalized boxes using each image's actual dimensions.
    Category "Crab-Pot" (the only one present in this particular
    download) -> class "ghost_net". This is an honest proxy, not a true
    match -- these are derelict crab pots, not nets -- flagged clearly
    when this dataset was first suggested and again in SOURCES.md below.
    Every image in this source genuinely has 0-or-more crab-pot boxes (a
    single-purpose dataset, unlike Cylinders), so 0-box images are kept as
    real background, same as AI4Shipwrecks/AquaScan/SubPipe.

  - SubPipe (SubPipeMiniSSS/DATA/SSS_{HF,LF}_images, no existing split ->
    85/15 seeded split PER frequency band so both bands land in both
    splits). Single class ("Pipeline" per its own classes.txt) -> class
    "pipe". Images ship as .pbm (confirmed cv2-readable), but ultralytics'
    own image-format allowlist does not include .pbm -- so unlike every
    other source here, SubPipe images are re-encoded to .png on copy
    (see _copy_pair_convert_to_png) rather than byte-copied as-is.
    0-box images are kept as real background (genuine plain-seafloor
    frames, same reasoning as AI4Shipwrecks/AquaScan).

Global class list (fixed order, this dataset's own taxonomy -- does NOT
match build_yolo_dataset.py's 7-class list, intentionally, per the
instruction not to reuse or extend the old merge):
    0 shipwreck   1 human   2 cylinder   3 ghost_net   4 pipe

Usage:
    python -m src.preprocessing.build_yolo_dataset_v2 --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset_v2
    # or chunked, one source per call (recommended under a ~180s-per-call shell):
    python -m src.preprocessing.build_yolo_dataset_v2 --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset_v2 --stage shipwreck
    python -m src.preprocessing.build_yolo_dataset_v2 --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset_v2 --stage human
    python -m src.preprocessing.build_yolo_dataset_v2 --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset_v2 --stage cylinders
    python -m src.preprocessing.build_yolo_dataset_v2 --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset_v2 --stage nets
    python -m src.preprocessing.build_yolo_dataset_v2 --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset_v2 --stage subpipe
    python -m src.preprocessing.build_yolo_dataset_v2 --labelled_dir data/raw/Labelled --out_dir data/processed/yolo_dataset_v2 --stage finalize
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

GLOBAL_CLASSES = ["shipwreck", "human", "cylinder", "ghost_net", "pipe"]
CLASS_ID = {name: i for i, name in enumerate(GLOBAL_CLASSES)}

CYLINDERS_ORIG_CLASS_ID = 3  # 'cylinder' in Cylinders/data.yaml's own names list
MIN_MASK_COMPONENT_AREA = 300  # px; same as build_yolo_dataset.py -- filters AI4Shipwrecks mask noise specks
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


def _write_manifest_row(writer, manifest_file, split, source, prefix, out_stem, orig_image, orig_label, label_lines) -> None:
    class_ids = sorted({line.split()[0] for line in label_lines})
    writer.writerow({
        "split": split, "source": source, "prefix": prefix, "out_stem": out_stem,
        "orig_image": str(orig_image), "orig_label": str(orig_label),
        "class_ids_present": ";".join(class_ids),
    })
    manifest_file.flush()  # so a mid-run timeout never leaves the manifest behind what's actually on disk


def _copy_pair(img_src: Path, label_lines: list[str], split: str, out_dir: Path, prefix: str, source: str, writer, manifest_file) -> None:
    """Byte-copies img_src as-is. Use for sources whose images are already
    in an ultralytics-recognized format (jpg/png)."""
    out_stem = f"{prefix}__{img_src.stem}"
    img_dst = out_dir / split / "images" / f"{out_stem}{img_src.suffix.lower()}"
    lbl_dst = out_dir / split / "labels" / f"{out_stem}.txt"
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    if img_dst.exists() and lbl_dst.exists():
        return  # resumable: already done by a prior partial run
    shutil.copy2(img_src, img_dst)
    lbl_dst.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
    _write_manifest_row(writer, manifest_file, split, source, prefix, out_stem, img_src, img_src.with_suffix(".txt"), label_lines)


def _copy_pair_convert_to_png(img_src: Path, label_lines: list[str], split: str, out_dir: Path, prefix: str, source: str, writer, manifest_file) -> None:
    """Re-encodes img_src to .png on copy. Use for SubPipe's .pbm images --
    cv2 can read .pbm fine, but it's not in ultralytics' recognized image
    extension set, so a byte-copy would silently vanish from training."""
    out_stem = f"{prefix}__{img_src.stem}"
    img_dst = out_dir / split / "images" / f"{out_stem}.png"
    lbl_dst = out_dir / split / "labels" / f"{out_stem}.txt"
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    if img_dst.exists() and lbl_dst.exists():
        return
    img = cv2.imread(str(img_src), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"WARNING: failed to decode {img_src}, skipping")
        return
    cv2.imwrite(str(img_dst), img)
    lbl_dst.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
    _write_manifest_row(writer, manifest_file, split, source, prefix, out_stem, img_src, img_src.with_suffix(".txt"), label_lines)


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
        if orig_id not in remap:
            continue
        lines.append(" ".join([str(remap[orig_id])] + parts[1:]))
    return lines


def _deterministic_split(stems: list[str], seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * SPLIT_TRAIN_FRACTION)
    return set(shuffled[:n_train]), set(shuffled[n_train:])


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


def stage_shipwreck(labelled_dir: Path, out_dir: Path) -> None:
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
            print(f"[shipwreck] {split}: {n} images")
    finally:
        f.close()


def stage_human(labelled_dir: Path, out_dir: Path) -> None:
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
        print(f"[human/aquascan] train: {counts['train']}, test: {counts['test']}")
    finally:
        f.close()


def stage_cylinders(labelled_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        counts = {"train": 0, "test": 0}
        skipped_other_only = 0
        for src_split, out_split in [("train", "train"), ("valid", "train"), ("test", "test")]:
            img_dir = labelled_dir / "Cylinders" / src_split / "images"
            lbl_dir = labelled_dir / "Cylinders" / src_split / "labels"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                orig_lines = [l for l in (lbl_path.read_text().splitlines() if lbl_path.exists() else []) if l.strip()]
                cyl_lines = [l for l in orig_lines if int(l.split()[0]) == CYLINDERS_ORIG_CLASS_ID]
                if not cyl_lines and orig_lines:
                    # Image has objects, just none of them are a cylinder --
                    # skip entirely rather than write a misleading "empty"
                    # label (see module docstring).
                    skipped_other_only += 1
                    continue
                remapped = []
                for line in cyl_lines:
                    parts = line.split()
                    remapped.append(" ".join([str(CLASS_ID["cylinder"])] + parts[1:]))
                _copy_pair(img_path, remapped, out_split, out_dir, f"cyl_{src_split}", "uatd_cylinders", writer, f)
                counts[out_split] += 1
        print(f"[cylinders] train: {counts['train']}, test: {counts['test']} (skipped {skipped_other_only} other-class-only images)")
    finally:
        f.close()


def stage_nets(labelled_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        counts = {"train": 0, "test": 0}
        base_root = labelled_dir / "Nets" / "sss-crab-pot-detection-ds"
        for src_split, out_split in [("train", "train"), ("valid", "train"), ("test", "test")]:
            base = base_root / src_split
            jsonl_path = base / "metadata.jsonl"
            if not jsonl_path.exists():
                continue
            for line in jsonl_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                img_path = base / row["file_name"]
                if not img_path.exists():
                    print(f"WARNING: {img_path} listed in metadata.jsonl but missing on disk, skipping")
                    continue
                w, h = Image.open(img_path).size
                boxes = row["objects"]["bbox"]
                out_lines = []
                for (x, y, bw, bh) in boxes:
                    cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                    nw, nh = bw / w, bh / h
                    # Defensive clip -- source coords are trusted but not
                    # re-derived from scratch here, so guard against any
                    # stray out-of-bounds box rather than emit an invalid
                    # (outside [0,1]) YOLO label.
                    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                    nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
                    out_lines.append(f"{CLASS_ID['ghost_net']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                _copy_pair(img_path, out_lines, out_split, out_dir, f"net_{src_split}", "crabpot_ghostnet_proxy", writer, f)
                counts[out_split] += 1
        print(f"[ghost_net/crabpot] train: {counts['train']}, test: {counts['test']}")
    finally:
        f.close()


def stage_subpipe(labelled_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        counts = {"train": 0, "test": 0}
        for band in ("HF", "LF"):
            img_dir = labelled_dir / "SubPipeMiniSSS" / "DATA" / f"SSS_{band}_images" / "Image"
            lbl_dir = labelled_dir / "SubPipeMiniSSS" / "DATA" / f"SSS_{band}_images" / "YOLO_Annotation"
            if not img_dir.exists():
                continue
            images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".pbm")
            train_stems, _ = _deterministic_split([p.stem for p in images], SPLIT_SEED)
            for img_path in images:
                split = "train" if img_path.stem in train_stems else "test"
                lines = _remap_label_file(lbl_dir / f"{img_path.stem}.txt", {0: CLASS_ID["pipe"]})
                _copy_pair_convert_to_png(img_path, lines, split, out_dir, f"subpipe_{band.lower()}", "subpipe_mini_sss", writer, f)
                counts[split] += 1
        print(f"[pipe/subpipe] train: {counts['train']}, test: {counts['test']}")
    finally:
        f.close()


def finalize(out_dir: Path) -> None:
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "# Auto-generated by build_yolo_dataset_v2.py -- a SEPARATE merge from\n"
        "# data/processed/yolo_dataset, built around the PS's actual target\n"
        "# classes (ghost nets, pipes, cylinders, shipwrecks) plus AquaScan-1K's\n"
        "# genuine 'human' class. 'test' doubles as 'val' -- only a train/test\n"
        "# split was requested.\n"
        "#\n"
        "# No 'path:' key on purpose -- ultralytics resolves a relative 'path'\n"
        "# against the CALLER's cwd, not this file's own folder (confirmed the\n"
        "# hard way on the first merged dataset); omitting it falls back to this\n"
        "# yaml's own directory instead, which is what we actually want.\n"
        "train: train/images\n"
        "val: test/images\n"
        "test: test/images\n"
        f"nc: {len(GLOBAL_CLASSES)}\n"
        f"names: {GLOBAL_CLASSES}\n"
    )

    counts: dict[str, dict[str, int]] = {"train": {c: 0 for c in GLOBAL_CLASSES}, "test": {c: 0 for c in GLOBAL_CLASSES}}
    image_counts = {"train": 0, "test": 0}
    with open(out_dir / "MANIFEST.csv") as fh:
        for row in csv.DictReader(fh):
            image_counts[row["split"]] += 1
            for cid in row["class_ids_present"].split(";"):
                if cid:
                    counts[row["split"]][GLOBAL_CLASSES[int(cid)]] += 1

    lines = ["# yolo_dataset_v2 -- source summary\n"]
    lines.append(f"Images: train={image_counts['train']}, test={image_counts['test']}\n")
    lines.append("\n| class id | class | train boxes | test boxes |\n|---|---|---|---|\n")
    for i, c in enumerate(GLOBAL_CLASSES):
        lines.append(f"| {i} | {c} | {counts['train'][c]} | {counts['test'][c]} |\n")
    lines.append(
        "\nSources: AI4Shipwrecks -> shipwreck (mask-to-bbox). AquaScan-1K -> human "
        "(real SSS, real label, not a PS target class but kept since it's clean "
        "signal). Cylinders (UATD/Roboflow) -> cylinder ONLY -- images containing "
        "only the dataset's other 9 classes were dropped entirely, not kept as "
        "blank negatives (would have falsely taught 'no object' on images that do "
        "contain an unlabeled object). UATD is forward-looking/acoustic sonar, not "
        "side-scan -- a real domain gap, kept anyway for lack of a side-scan "
        "cylinder alternative. Nets/sss-crab-pot-detection-ds -> ghost_net: an "
        "HONEST PROXY, not a true match -- these are labeled derelict crab pots, "
        "not nets, on real side-scan sonar. SubPipeMiniSSS -> pipe, real "
        "side-scan sonar, re-encoded from .pbm to .png since ultralytics doesn't "
        "recognize .pbm as an image format.\n"
    )
    (out_dir / "SOURCES.md").write_text("".join(lines))
    print("".join(lines))
    print(f"data.yaml -> {data_yaml}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labelled_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--stage", type=str, default="all", choices=["all", "shipwreck", "human", "cylinders", "nets", "subpipe", "finalize"])
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    labelled_dir = Path(args.labelled_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = {
        "shipwreck": stage_shipwreck,
        "human": stage_human,
        "cylinders": stage_cylinders,
        "nets": stage_nets,
        "subpipe": stage_subpipe,
    }
    if args.stage == "all":
        for fn in stages.values():
            fn(labelled_dir, out_dir)
        finalize(out_dir)
    elif args.stage == "finalize":
        finalize(out_dir)
    else:
        stages[args.stage](labelled_dir, out_dir)

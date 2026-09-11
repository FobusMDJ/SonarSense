"""Converts data/processed/Train_Top (SCTD + SubPipe) into a standalone YOLO
dataset at data/processed/train_top_yolo -- built per explicit instruction
NOT to merge this into yolo_dataset_v2_preprocessed's 5-class corpus (see
claude/backend-pipeline-plan.md's sibling docs for that dataset). This is a
separate, smaller, 4-class dataset meant to be trained on by itself with
YOLO26 (src/detection/train_yolo26.py), not folded into the existing
shipwreck/human/cylinder/ghost_net/pipe taxonomy.

Two sources, two very different starting formats:

  - SCTD (data/processed/Train_Top/SCTD) -- Pascal VOC XML annotations
    (Annotations/*.xml) + JPEGImages/*.jpg. 357 annotated images, three
    <name> values found: "ship", "human", "aircraft".

    IMPORTANT CAVEAT ON "aircraft": ~89 of the 357 XML files carry
    <folder>UAV_data</folder> and <name>aircraft</name> -- leftover
    metadata from an unrelated "UAV autolanding" annotation-tool template
    (owner "ChaojieZhu", source "The UAV autolanding", flickrid NULL), not
    a deliberately-chosen label. Several of these images were checked
    visually before this script was written and are genuine side-scan
    sonar frames -- wreck-shaped objects with acoustic shadows, one with
    a visible nadir gap -- indistinguishable from the correctly-tagged
    "ship" files. Despite that, per explicit instruction this script keeps
    "aircraft" as ITS OWN class exactly as SCTD labels it, rather than
    folding it into "shipwreck" or dropping it. If that turns out to be
    wrong once someone reviews all ~89 images, re-run stage_sctd after
    editing SCTD_NAME_MAP below (aircraft -> shipwreck) and the split/
    manifest logic requires no other changes.

    SCTD images are pre-cropped per-target chips (mixed sizes, e.g.
    415x385, 744x497, 388x565) -- not native full-swath rasters -- so they
    get the same "simple" treatment preprocess_yolo_dataset_v2.py uses for
    Cylinders/Nets/AquaScan-1K: grayscale -> percentile-normalize ->
    resize to 512x512. No dropout-repair, no tiling (those need a real
    swath's row/column statistics, which a pre-cropped chip doesn't have).

  - SubPipe (data/processed/Train_Top/SubPipe) -- Annotations/*.txt +
    Images/*.png, already in YOLO-normalized format AND already run
    through this project's full grayscale/dropout-repair/normalize/tile
    pipeline: filenames match preprocess_yolo_dataset_v2.py's
    "{out_stem}__t{i}.png" tile-naming convention exactly, and a sample
    image confirmed single-channel 512x512 uint8. So these are copied
    through byte-for-byte, no re-processing. Labels use class id 4, which
    is "pipe" in yolo_dataset_v2_preprocessed's 5-class scheme (its own
    data.yaml: 0 shipwreck, 1 human, 2 cylinder, 3 ghost_net, 4 pipe) --
    remapped here to whatever index "pipe" gets in THIS script's smaller
    4-class list. Any other class id found in a SubPipe label (would mean
    a genuine cylinder/ghost_net box with no home in this 4-class
    taxonomy) is dropped from that line and counted/reported in
    SOURCES.md rather than silently discarded -- see stage_subpipe.

Class list (fixed order, this dataset's own taxonomy -- distinct from
yolo_dataset_v2_preprocessed's 5-class list on purpose):
    0 shipwreck   1 human   2 aircraft   3 pipe

No cylinder, no ghost_net -- Train_Top has zero examples of either. A model
trained only on this dataset will not know those two classes at all; that
was an explicit, deliberate choice (train on Train_Top standalone, not
merged into the bigger corpus) -- not an oversight.

Usage:
    python -m src.preprocessing.build_train_top_dataset \\
        --train_top_dir data/processed/Train_Top --out_dir data/processed/train_top_yolo
    # or one source at a time (recommended under a ~180s-per-call shell):
    python -m src.preprocessing.build_train_top_dataset --train_top_dir data/processed/Train_Top --out_dir data/processed/train_top_yolo --stage sctd
    python -m src.preprocessing.build_train_top_dataset --train_top_dir data/processed/Train_Top --out_dir data/processed/train_top_yolo --stage subpipe
    python -m src.preprocessing.build_train_top_dataset --train_top_dir data/processed/Train_Top --out_dir data/processed/train_top_yolo --stage finalize
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2

from src.preprocessing.enhancement import standardize_resolution
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.normalization import normalize_intensity
from src.utils.config import get_logger

logger = get_logger(__name__)

GLOBAL_CLASSES = ["shipwreck", "human", "aircraft", "pipe"]
CLASS_ID = {name: i for i, name in enumerate(GLOBAL_CLASSES)}

# VOC <name> -> our class name. "aircraft" kept literal per explicit
# instruction -- see module docstring for the mislabeling caveat.
SCTD_NAME_MAP = {"ship": "shipwreck", "human": "human", "aircraft": "aircraft"}

# yolo_dataset_v2_preprocessed's OWN class id for "pipe" (its data.yaml:
# 0 shipwreck, 1 human, 2 cylinder, 3 ghost_net, 4 pipe). Train_Top/SubPipe's
# label files were produced against that scheme, so this is the only id we
# know how to remap; anything else has no home in this 4-class taxonomy.
SUBPIPE_OLD_ID_TO_NEW_NAME = {4: "pipe"}

SPLIT_SEED = 42
SPLIT_TRAIN_FRACTION = 0.85
TILE_SIZE = 512  # matches SubPipe's already-tiled size; SCTD chips are resized to this too

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
    manifest_file.flush()


def _deterministic_split(stems: list[str], seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * SPLIT_TRAIN_FRACTION)
    return set(shuffled[:n_train]), set(shuffled[n_train:])


def _parse_voc_xml(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]], Counter]:
    """Returns (width, height, boxes, unmapped_name_counts). boxes are
    (class_name, xmin, ymin, xmax, ymax) in absolute pixels, already clipped
    to the image bounds. Objects whose <name> isn't in SCTD_NAME_MAP are
    skipped and counted in unmapped_name_counts rather than silently
    dropped without a trace. difficult=1 objects are skipped too (standard
    VOC convention) -- none were observed in a manual sample of this
    dataset, so this should be a no-op in practice."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width, height = int(size.find("width").text), int(size.find("height").text)

    boxes = []
    unmapped = Counter()
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        difficult = obj.find("difficult")
        if difficult is not None and difficult.text.strip() == "1":
            continue
        if name not in SCTD_NAME_MAP:
            unmapped[name] += 1
            continue
        bnd = obj.find("bndbox")
        xmin = max(0.0, float(bnd.find("xmin").text))
        ymin = max(0.0, float(bnd.find("ymin").text))
        xmax = min(float(width), float(bnd.find("xmax").text))
        ymax = min(float(height), float(bnd.find("ymax").text))
        if xmax <= xmin or ymax <= ymin:
            continue  # degenerate box, skip rather than emit a zero-area YOLO line
        boxes.append((SCTD_NAME_MAP[name], xmin, ymin, xmax, ymax))
    return width, height, boxes, unmapped


def stage_sctd(train_top_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        ann_dir = train_top_dir / "SCTD" / "Annotations"
        img_dir = train_top_dir / "SCTD" / "JPEGImages"
        xml_paths = sorted(ann_dir.glob("*.xml"))
        train_stems, _ = _deterministic_split([p.stem for p in xml_paths], SPLIT_SEED)

        counts = {"train": 0, "test": 0}
        total_unmapped = Counter()
        n_missing_image = 0

        for xml_path in xml_paths:
            stem = xml_path.stem
            img_path = img_dir / f"{stem}.jpg"
            if not img_path.exists():
                logger.warning("SCTD: %s has no matching image (%s), skipping", xml_path.name, img_path)
                n_missing_image += 1
                continue

            width, height, boxes, unmapped = _parse_voc_xml(xml_path)
            total_unmapped.update(unmapped)

            split = "train" if stem in train_stems else "test"
            out_stem = f"sctd__{stem}"
            out_img_path = out_dir / split / "images" / f"{out_stem}.png"
            out_lbl_path = out_dir / split / "labels" / f"{out_stem}.txt"
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

            lines = []
            for class_name, xmin, ymin, xmax, ymax in boxes:
                cx, cy = (xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height
                w, h = (xmax - xmin) / width, (ymax - ymin) / height
                lines.append(f"{CLASS_ID[class_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if not (out_img_path.exists() and out_lbl_path.exists()):
                gray = to_grayscale(cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED))
                normed = normalize_intensity(gray, method="percentile")
                resized = standardize_resolution(normed, target_size=(TILE_SIZE, TILE_SIZE))
                cv2.imwrite(str(out_img_path), resized)
                out_lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))
                _write_manifest_row(writer, f, split, "sctd", "sctd", out_stem, img_path, xml_path, lines)

            counts[split] += 1

        logger.info("[sctd] train: %d, test: %d (missing image: %d)", counts["train"], counts["test"], n_missing_image)
        if total_unmapped:
            logger.warning("[sctd] object <name> values outside SCTD_NAME_MAP, DROPPED from labels (not silently "
                            "included as anything else): %s", dict(total_unmapped))
    finally:
        f.close()


def stage_subpipe(train_top_dir: Path, out_dir: Path) -> None:
    f, writer = _manifest_writer(out_dir)
    try:
        ann_dir = train_top_dir / "SubPipe" / "Annotations"
        img_dir = train_top_dir / "SubPipe" / "Images"
        img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".png")
        train_stems, _ = _deterministic_split([p.stem for p in img_paths], SPLIT_SEED)

        counts = {"train": 0, "test": 0}
        n_dropped_lines = 0
        dropped_ids = Counter()

        for img_path in img_paths:
            stem = img_path.stem
            lbl_path = ann_dir / f"{stem}.txt"
            raw_lines = [l for l in (lbl_path.read_text().splitlines() if lbl_path.exists() else []) if l.strip()]

            lines = []
            for line in raw_lines:
                parts = line.split()
                old_id = int(parts[0])
                new_name = SUBPIPE_OLD_ID_TO_NEW_NAME.get(old_id)
                if new_name is None:
                    n_dropped_lines += 1
                    dropped_ids[old_id] += 1
                    continue
                lines.append(" ".join([str(CLASS_ID[new_name])] + parts[1:]))

            split = "train" if stem in train_stems else "test"
            out_stem = f"subpipe__{stem}"
            out_img_path = out_dir / split / "images" / f"{out_stem}.png"
            out_lbl_path = out_dir / split / "labels" / f"{out_stem}.txt"
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

            if not (out_img_path.exists() and out_lbl_path.exists()):
                shutil.copy2(img_path, out_img_path)  # already grayscale/normalized/tiled -- byte copy
                out_lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))
                _write_manifest_row(writer, f, split, "subpipe", "subpipe", out_stem, img_path, lbl_path, lines)

            counts[split] += 1

        logger.info("[subpipe] train: %d, test: %d", counts["train"], counts["test"])
        if n_dropped_lines:
            logger.warning("[subpipe] %d label lines used a class id with no home in this 4-class taxonomy, "
                            "DROPPED (not silently kept under the wrong class): id counts %s",
                            n_dropped_lines, dict(dropped_ids))
    finally:
        f.close()


def finalize(out_dir: Path) -> None:
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "# Auto-generated by build_train_top_dataset.py -- a STANDALONE dataset\n"
        "# from data/processed/Train_Top (SCTD + SubPipe), deliberately NOT merged\n"
        "# into yolo_dataset_v2_preprocessed's 5-class corpus. 'test' doubles as\n"
        "# 'val' -- only a train/test split was built.\n"
        "#\n"
        "# No 'path:' key on purpose -- ultralytics resolves a relative 'path'\n"
        "# against the CALLER's cwd, not this file's own folder.\n"
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

    lines = ["# train_top_yolo -- source summary\n"]
    lines.append(f"Images: train={image_counts['train']}, test={image_counts['test']}\n")
    lines.append("\n| class id | class | train boxes | test boxes |\n|---|---|---|---|\n")
    for i, c in enumerate(GLOBAL_CLASSES):
        lines.append(f"| {i} | {c} | {counts['train'][c]} | {counts['test'][c]} |\n")
    lines.append(
        "\nSources: SCTD (Pascal VOC XML, 357 images) -> shipwreck/human/aircraft. "
        "Grayscale + percentile-normalize + resize to 512x512 (pre-cropped chips, "
        "not native swaths -- no dropout-repair/tiling). "
        "IMPORTANT: ~89 of the 357 SCTD files carry class='aircraft' as leftover "
        "annotation-tool template metadata (folder='UAV_data', owner='ChaojieZhu', "
        "source='The UAV autolanding') rather than a deliberately-chosen label -- "
        "several were checked visually and are genuine sonar wreck-shaped frames, "
        "indistinguishable from the correctly-tagged 'ship' files. Kept as its own "
        "class 'aircraft' here per explicit instruction, NOT folded into shipwreck "
        "or dropped -- review the ~89 files under SCTD/Annotations with "
        "folder=UAV_data before trusting this class if that matters for your use. "
        "SubPipe (already grayscale/normalized/tiled 512x512 PNGs + YOLO labels) "
        "-> pipe, byte-copied as-is; any label line using a class id besides the "
        "one known to mean 'pipe' was dropped and counted in the build log rather "
        "than silently kept under the wrong class.\n"
        "\nNo cylinder, no ghost_net -- Train_Top has zero examples of either; a "
        "model trained only on this dataset will not know those classes at all.\n"
    )
    (out_dir / "SOURCES.md").write_text("".join(lines))
    print("".join(lines))
    print(f"data.yaml -> {data_yaml}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_top_dir", type=str, required=True, help="Path to data/processed/Train_Top.")
    p.add_argument("--out_dir", type=str, required=True, help="Output dataset dir, e.g. data/processed/train_top_yolo.")
    p.add_argument("--stage", type=str, default="all", choices=["all", "sctd", "subpipe", "finalize"])
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_top_dir = Path(args.train_top_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = {"sctd": stage_sctd, "subpipe": stage_subpipe}
    if args.stage == "all":
        for fn in stages.values():
            fn(train_top_dir, out_dir)
        finalize(out_dir)
    elif args.stage == "finalize":
        finalize(out_dir)
    else:
        stages[args.stage](train_top_dir, out_dir)

"""Combines two independently-trained YOLO checkpoints into one set of
detections per image, WITHOUT touching either model's weights.

Why not just merge the .pt files: your model and a differently-trained
one almost never share the same class head (different class count, order,
or even different classes entirely -- e.g. one has 'plane' and the other
doesn't). Averaging or concatenating their weights would scramble both
models' learned behavior rather than combine their strengths, and there is
no reliable way to "merge" two independently-trained detection heads at
the weight level. See the module this replaces (a from-scratch weight
merge) -- deliberately not implemented, for that reason.

What this does instead, safely: runs BOTH models on every image, but only
KEEPS each model's predictions for the classes you've told it that model is
actually good at (--model_a_classes / --model_b_classes). Each class name
is routed to exactly one model -- no overlap allowed between the two lists
-- so there's no need to reconcile two models' opinions on the same class
(no NMS-across-models, no confidence calibration issue): model A's "pipe"
detections are trusted because you already measured model A is best at
pipe (see denoise_test_set.py-style per-class mAP comparisons), and model
B's "shipwreck"/"human"/"plane" detections are trusted the same way. Each
model's accuracy on ITS classes is exactly what it was before -- nothing
is compromised, because nothing about either model changed.

Cost: every image now runs through two models instead of one (roughly 2x
inference time), and this produces detections/labels/an annotated preview,
not a single portable .pt file -- there is no way to get the combined
behavior into one file without an actual retrain on a unified dataset (see
train_yolo.py once both datasets are merged, if you want that instead).

Usage:
    python -m src.detection.ensemble_predict \\
        --model_a weights/best.pt --model_a_classes pipe cylinder ghost_net \\
        --model_b weights/friend_best.pt --model_b_classes shipwreck human plane \\
        --image_dir data/processed/yolo_dataset_v2_preprocessed/test/images \\
        --out_dir ensemble_out
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# BGR -- one color per source model, so the annotated preview also shows
# WHICH model contributed each box at a glance.
COLOR_A = (0, 200, 0)    # green
COLOR_B = (255, 120, 0)  # blue-ish


def detect_filtered(yolo_model, image_path: str, conf: float, allowed_classes: set[str], source_tag: str) -> list[dict]:
    """Runs one model, keeps only detections whose class name is in
    allowed_classes. Coordinates are in the ORIGINAL image's pixel space
    (ultralytics rescales its own internal resize back automatically)."""
    results = yolo_model.predict(source=image_path, conf=conf, verbose=False)
    r = results[0]
    names = r.names
    kept = []
    for box in r.boxes:
        cls_name = names[int(box.cls[0])]
        if cls_name not in allowed_classes:
            continue
        kept.append({
            "cls_name": cls_name,
            "conf": float(box.conf[0]),
            "xyxy": box.xyxy[0].cpu().numpy().tolist(),
            "source": source_tag,
        })
    return kept


def draw_annotated(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        color = COLOR_A if d["source"] == "A" else COLOR_B
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{d['cls_name']} {d['conf']:.2f} ({d['source']})"
        cv2.putText(canvas, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def write_yolo_labels(detections: list[dict], image_shape: tuple[int, int], class_to_id: dict[str, int], out_path: Path) -> None:
    h, w = image_shape
    lines = []
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{class_to_id[d['cls_name']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def run(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model_a_classes = set(args.model_a_classes)
    model_b_classes = set(args.model_b_classes)
    overlap = model_a_classes & model_b_classes
    if overlap:
        raise ValueError(
            f"--model_a_classes and --model_b_classes both claim {overlap} -- each class must be routed to "
            "exactly ONE model. Pick whichever model you've actually measured is better at that class "
            "(e.g. via a per-class mAP comparison) and list it under that model only."
        )

    model_a = YOLO(args.model_a)
    model_b = YOLO(args.model_b)
    logger.info("Model A (%s) trusted for: %s", args.model_a, sorted(model_a_classes))
    logger.info("Model B (%s) trusted for: %s", args.model_b, sorted(model_b_classes))

    combined_classes = sorted(model_a_classes | model_b_classes)
    class_to_id = {name: i for i, name in enumerate(combined_classes)}
    logger.info("Combined class list (unified ids for output labels): %s", class_to_id)

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "classes.txt", "w") as f:
        f.write("\n".join(combined_classes) + "\n")

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if args.limit:
        image_paths = image_paths[: args.limit]

    all_rows = []
    counts = {"A": 0, "B": 0}
    for i, path in enumerate(image_paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Skipping unreadable image: %s", path)
            continue

        dets_a = detect_filtered(model_a, str(path), args.conf_a, model_a_classes, "A")
        dets_b = detect_filtered(model_b, str(path), args.conf_b, model_b_classes, "B")
        detections = dets_a + dets_b
        counts["A"] += len(dets_a)
        counts["B"] += len(dets_b)

        annotated = draw_annotated(image, detections)
        cv2.imwrite(str(out_dir / "annotated" / f"{path.stem}.png"), annotated)
        write_yolo_labels(detections, image.shape, class_to_id, out_dir / "labels" / f"{path.stem}.txt")

        for d in detections:
            all_rows.append({"image": str(path), **d})

        if i % 100 == 0 or i == len(image_paths):
            logger.info("Processed %d/%d images", i, len(image_paths))

    with open(out_dir / "ensemble_detections.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "cls_name", "conf", "xyxy", "source"])
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(
        "Done: %d images -> %s (%d detections from model A, %d from model B)",
        len(image_paths), out_dir, counts["A"], counts["B"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_a", type=str, required=True, help="Path to your .pt checkpoint.")
    p.add_argument("--model_a_classes", type=str, nargs="+", required=True, help="Class names to trust from model A (must match model A's own class names exactly).")
    p.add_argument("--conf_a", type=float, default=0.25, help="Confidence threshold for model A.")
    p.add_argument("--model_b", type=str, required=True, help="Path to your friend's .pt checkpoint.")
    p.add_argument("--model_b_classes", type=str, nargs="+", required=True, help="Class names to trust from model B (must match model B's own class names exactly). Must not overlap with --model_a_classes.")
    p.add_argument("--conf_b", type=float, default=0.25, help="Confidence threshold for model B.")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="ensemble_out")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N images -- useful for a quick trial run.")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

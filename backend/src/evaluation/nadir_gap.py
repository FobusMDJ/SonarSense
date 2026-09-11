"""Nadir/no-data gap detector + hard-negative candidate miner, and a
validation script that tests whether this signal actually correlates with
the shipwreck Bkg-FP false positives found in Phase 1 -- BEFORE any of it is
used to generate training labels or touch YOLO.

Background (from claude/phase1-baseline-error-analysis.md): a full-swath
side-scan sonar tile has a blind strip directly beneath the tow vehicle
(no acoustic return -- the "nadir gap"), which appears as a large,
near-uniform, low-mean, low-variance vertical band. One visually-inspected
shipwreck Bkg-FP panel (ai4sw__Artificial_Reef_01) showed a false "shipwreck"
box on a bright texture cluster immediately adjacent to exactly this kind of
band. That was ONE example. This module turns "adjacent to the nadir gap"
into a measurable per-image signal and checks, across every shipwreck
Bkg-FP and a size-matched TP control group, whether Bkg-FP boxes are
actually closer to a detected gap band than TP boxes are -- rather than
assuming the one visual example generalizes.

IMPORTANT: like contrast_proxy/shadow_proxy in error_analysis.py, gap
detection here is a PIXEL-ONLY heuristic (low local mean + low local
variance over a contiguous vertical run of columns). It is not a validated
sonar-navigation nadir estimate and it will occasionally fire on other dark,
untextured regions (deep shadow, saturated dropout). Treated as weak,
correlational evidence only -- exactly like the other proxies in this
project.

Two things live here:

  1. detect_nadir_gap() / box_distance_to_gap() -- the detector + distance
     metric, used by the validation step below.
  2. find_gap_adjacent_bright_clusters() -- the hard-negative CANDIDATE
     miner (bright blobs sitting next to a detected gap, in images with no
     GT box there). This produces review candidates only. It does NOT write
     YOLO labels and is not wired into training anywhere -- per the explicit
     instruction to validate the gap signal before generating any labels or
     touching YOLO.

Usage (validation -- the step to run first):
    python -m src.evaluation.nadir_gap validate \\
        --weights_path best.pt \\
        --images_dir <val_images_dir> \\
        --labels_dir <val_labels_dir> \\
        --out_dir nadir_gap_validation_out

Usage (hard-negative candidate mining -- only after validation supports it):
    python -m src.evaluation.nadir_gap mine \\
        --images_dir <images_dir> --labels_dir <labels_dir> \\
        --out_dir nadir_gap_candidates_out
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------
# 1. Nadir-gap band detector (pixel-only heuristic)
# --------------------------------------------------------------------------

def detect_nadir_gap(
    gray: np.ndarray,
    min_width_frac: float = 0.015,
    max_width_frac: float = 0.60,
    abs_std_thresh: float = 3.0,
) -> dict | None:
    """Scan columns for a contiguous, (near-)perfectly-flat vertical band
    consistent with a full-swath nadir/no-data gap.

    REVISION NOTE: an earlier version of this function gated on BOTH
    col_mean and col_std each below a per-image PERCENTILE threshold. That
    version silently missed the one directly-confirmed example from Phase 1
    (ai4sw__Artificial_Reef_01__t1.png) -- diagnostic column-profiling on
    that image showed a real no-data band (columns 256-351, std EXACTLY
    0.0, a constant fill value) that the percentile-based mean gate rejected
    simply because other, unrelated dark regions elsewhere in the same
    image pulled the percentile threshold down. That is a distinctive,
    physically real signature -- a no-data/no-return fill is literally
    constant, i.e. std ~ 0 -- being defeated by a scene-relative statistic
    that has nothing to do with flatness. Fixed by gating on an ABSOLUTE
    std threshold (std < abs_std_thresh, in raw 0-255 pixel units) instead
    of a percentile-relative one. col_mean is no longer a hard gate at all;
    it is not used to reject candidate columns, since the gap's own
    brightness varies from tile to tile and is not diagnostic on its own.

    For every column x: col_std[x] = std over the column. A column is a gap
    candidate if col_std[x] < abs_std_thresh. The widest contiguous run of
    gap-candidate columns whose width falls within
    [min_width_frac, max_width_frac] * image_width is returned as the band.

    Returns None if no run in that width range is found (e.g. the image has
    no visible gap in-frame, or it's a pre-cropped/UI-chrome image where
    this heuristic doesn't apply at all).
    """
    h, w = gray.shape[:2]
    col_mean = gray.mean(axis=0).astype(np.float64)
    col_std = gray.std(axis=0).astype(np.float64)

    is_gap_col = col_std < abs_std_thresh

    min_w = max(1, int(w * min_width_frac))
    max_w = max(min_w, int(w * max_width_frac))

    best = None
    run_start = None
    for x in range(w + 1):
        col_ok = x < w and is_gap_col[x]
        if col_ok and run_start is None:
            run_start = x
        if (not col_ok) and run_start is not None:
            run_len = x - run_start
            if min_w <= run_len <= max_w:
                band_mean = float(col_mean[run_start:x].mean())
                band_std = float(col_std[run_start:x].mean())
                # score: widest run wins (a wider constant-fill run is
                # stronger evidence of a real gap than a narrow one), flatness
                # (lower band_std) as tiebreaker. Mean is not part of the
                # score at all -- see the flatness-vs-mean note above.
                score = run_len - band_std
                if best is None or score > best["score"]:
                    best = {
                        "x1": run_start, "x2": x, "width": run_len,
                        "band_mean": band_mean, "band_std": band_std,
                        "score": score,
                    }
            run_start = None

    return best


def box_distance_to_gap(xyxy: list[float], gap: dict | None) -> float | None:
    """Min horizontal pixel distance from a box to a detected gap band.
    0 if the box overlaps the band. None if no gap was detected in this image."""
    if gap is None:
        return None
    x1, _, x2, _ = xyxy
    if x2 < gap["x1"]:
        return gap["x1"] - x2
    if x1 > gap["x2"]:
        return x1 - gap["x2"]
    return 0.0


# --------------------------------------------------------------------------
# 2. Hard-negative CANDIDATE miner (bright clusters adjacent to a gap band)
#    -- produces review candidates only, does not write YOLO labels.
# --------------------------------------------------------------------------

def find_gap_adjacent_bright_clusters(
    gray: np.ndarray,
    gap: dict,
    existing_boxes: list[list[float]],
    adjacency_px: int = 40,
    bright_percentile: float = 90.0,
    min_cluster_area: int = 30,
) -> list[dict]:
    """Within adjacency_px of either edge of the gap band, find bright
    connected components (candidate 'bright cluster near the gap' regions --
    the visual pattern behind the one confirmed shipwreck Bkg-FP example).
    Skips clusters that overlap an existing GT or predicted box (those are
    already accounted for). Returns candidate boxes with a crude score --
    NOT confidence-calibrated, NOT a detector, for human review only."""
    h, w = gray.shape[:2]
    x1, x2 = gap["x1"], gap["x2"]
    zone_x1 = max(0, x1 - adjacency_px)
    zone_x2 = min(w, x2 + adjacency_px)
    zone = gray[:, zone_x1:zone_x2]
    if zone.size == 0:
        return []

    thresh_val = np.percentile(zone, bright_percentile)
    mask = (zone >= thresh_val).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    def overlaps_existing(bx: list[float]) -> bool:
        for eb in existing_boxes:
            ix1, iy1 = max(bx[0], eb[0]), max(bx[1], eb[1])
            ix2, iy2 = min(bx[2], eb[2]), min(bx[3], eb[3])
            if ix2 > ix1 and iy2 > iy1:
                return True
        return False

    candidates = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_cluster_area:
            continue
        lx, ly, lw, lh = (int(stats[lbl, cv2.CC_STAT_LEFT]), int(stats[lbl, cv2.CC_STAT_TOP]),
                           int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT]))
        box = [zone_x1 + lx, ly, zone_x1 + lx + lw, ly + lh]
        if overlaps_existing(box):
            continue
        candidates.append({
            "xyxy": box,
            "area_px": area,
            "dist_to_gap": box_distance_to_gap(box, gap),
        })
    candidates.sort(key=lambda c: -c["area_px"])
    return candidates


# --------------------------------------------------------------------------
# Validation: does gap-adjacency actually correlate with real shipwreck FPs?
# --------------------------------------------------------------------------

SHIPWRECK_CLASS_ID = 0  # data.yaml order: [shipwreck, human, cylinder, ghost_net, pipe]


def run_validate(args: argparse.Namespace) -> None:
    from scipy import stats as scipy_stats
    from ultralytics import YOLO

    from src.evaluation.error_analysis import load_yolo_labels, match_detections

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.weights_path)
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    logger.info("Validating nadir-gap distance signal over %d images from %s", len(image_paths), images_dir)

    rows = []
    n_gap_found = 0
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Unreadable image, skipping: %s", path)
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(labels_dir / f"{path.stem}.txt", w, h)

        results = yolo.predict(source=str(path), conf=args.conf, verbose=False)
        preds = [
            {"class_id": int(box.cls[0]), "conf": float(box.conf[0]),
             "xyxy": box.xyxy[0].cpu().numpy().tolist()}
            for box in results[0].boxes
        ]
        matches = match_detections(gts, preds, iou_fg=args.iou_fg, iou_bg=args.iou_bg)

        gap = detect_nadir_gap(img)
        if gap is not None:
            n_gap_found += 1

        for m in matches:
            gt_cls = m["gt"]["class_id"] if m["gt"] else None
            pred_cls = m["pred"]["class_id"] if m["pred"] else None
            is_shipwreck = (gt_cls == SHIPWRECK_CLASS_ID) or (pred_cls == SHIPWRECK_CLASS_ID)
            if not is_shipwreck:
                continue
            if m["type"] not in ("Bkg-FP", "TP"):
                continue
            ref_box = (m["pred"] or m["gt"])["xyxy"]
            dist = box_distance_to_gap(ref_box, gap)
            rows.append({
                "image": path.name,
                "error_type": m["type"],
                "gap_detected": gap is not None,
                "gap_width_px": gap["width"] if gap else "",
                "img_width_px": w,
                "dist_to_gap_px": dist if dist is not None else "",
                "dist_to_gap_frac_width": (dist / w) if dist is not None else "",
            })

    with open(out_dir / "gap_distance_instances.csv", "w", newline="") as f:
        fieldnames = ["image", "error_type", "gap_detected", "gap_width_px", "img_width_px",
                      "dist_to_gap_px", "dist_to_gap_frac_width"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fp_dists = [r["dist_to_gap_frac_width"] for r in rows if r["error_type"] == "Bkg-FP" and r["dist_to_gap_frac_width"] != ""]
    tp_dists = [r["dist_to_gap_frac_width"] for r in rows if r["error_type"] == "TP" and r["dist_to_gap_frac_width"] != ""]

    n_fp_total = sum(1 for r in rows if r["error_type"] == "Bkg-FP")
    n_tp_total = sum(1 for r in rows if r["error_type"] == "TP")

    report = {
        "n_images": len(image_paths),
        "n_images_with_gap_detected": n_gap_found,
        "frac_images_with_gap_detected": round(n_gap_found / len(image_paths), 4) if image_paths else 0,
        "n_shipwreck_bkgfp_instances": n_fp_total,
        "n_shipwreck_bkgfp_with_gap_in_image": len(fp_dists),
        "n_shipwreck_tp_instances": n_tp_total,
        "n_shipwreck_tp_with_gap_in_image": len(tp_dists),
    }

    if fp_dists and tp_dists:
        report["mean_dist_to_gap_frac_width_BkgFP"] = round(float(np.mean(fp_dists)), 4)
        report["median_dist_to_gap_frac_width_BkgFP"] = round(float(np.median(fp_dists)), 4)
        report["mean_dist_to_gap_frac_width_TP"] = round(float(np.mean(tp_dists)), 4)
        report["median_dist_to_gap_frac_width_TP"] = round(float(np.median(tp_dists)), 4)
        u_stat, p_value = scipy_stats.mannwhitneyu(fp_dists, tp_dists, alternative="less")
        report["mannwhitney_u"] = float(u_stat)
        report["mannwhitney_p_value_BkgFP_closer_than_TP"] = float(p_value)
        # fraction "close to gap" using a fixed practical threshold too, not just the mean
        for frac_thresh in (0.02, 0.05, 0.10):
            fp_close = sum(1 for d in fp_dists if d <= frac_thresh) / len(fp_dists)
            tp_close = sum(1 for d in tp_dists if d <= frac_thresh) / len(tp_dists)
            report[f"frac_BkgFP_within_{frac_thresh}_width_of_gap"] = round(fp_close, 4)
            report[f"frac_TP_within_{frac_thresh}_width_of_gap"] = round(tp_close, 4)
    else:
        report["note"] = "Insufficient instances with a detected gap in-image to compare distributions."

    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation report: %s", json.dumps(report, indent=2))
    logger.info("Wrote %s and %s", out_dir / "gap_distance_instances.csv", out_dir / "validation_report.json")


def run_mine(args: argparse.Namespace) -> None:
    from src.evaluation.error_analysis import load_yolo_labels

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(exist_ok=True)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    all_candidates = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(labels_dir / f"{path.stem}.txt", w, h)
        gap = detect_nadir_gap(img)
        if gap is None:
            continue
        existing_boxes = [g["xyxy"] for g in gts]
        cands = find_gap_adjacent_bright_clusters(img, gap, existing_boxes)
        for c in cands:
            c["image"] = path.name
            all_candidates.append(c)
            canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(canvas, (gap["x1"], 0), (gap["x2"], h), (255, 0, 0), 1)
            x1, y1, x2, y2 = (int(v) for v in c["xyxy"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.imwrite(str(panels_dir / f"{path.stem}_{len(all_candidates):04d}.png"), canvas)

    with open(out_dir / "candidates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "xyxy", "area_px", "dist_to_gap"])
        writer.writeheader()
        for c in all_candidates:
            writer.writerow({"image": c["image"], "xyxy": c["xyxy"], "area_px": c["area_px"], "dist_to_gap": c["dist_to_gap"]})

    logger.info("Mined %d hard-negative CANDIDATES (review only, no labels written) -> %s", len(all_candidates), out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    pv = sub.add_parser("validate", help="Test whether gap-distance correlates with real shipwreck Bkg-FP vs TP.")
    pv.add_argument("--weights_path", type=str, required=True)
    pv.add_argument("--images_dir", type=str, required=True)
    pv.add_argument("--labels_dir", type=str, required=True)
    pv.add_argument("--conf", type=float, default=0.25)
    pv.add_argument("--iou_fg", type=float, default=0.5)
    pv.add_argument("--iou_bg", type=float, default=0.1)
    pv.add_argument("--out_dir", type=str, default="nadir_gap_validation_out")

    pm = sub.add_parser("mine", help="Mine gap-adjacent bright-cluster hard-negative CANDIDATES for review (no labels written).")
    pm.add_argument("--images_dir", type=str, required=True)
    pm.add_argument("--labels_dir", type=str, required=True)
    pm.add_argument("--out_dir", type=str, default="nadir_gap_candidates_out")

    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.mode == "validate":
        run_validate(args)
    else:
        run_mine(args)

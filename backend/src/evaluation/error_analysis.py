"""Empirical error analysis for the trained YOLO baseline (best.pt) -- Phase 1
of the physics-aware investigation: establish what the model is ACTUALLY
getting wrong before proposing any dataset or architecture changes.

This does three things, in order:

  1. Runs ultralytics' own model.val() for the official mAP50/mAP50-95/
     precision/recall/per-class AP numbers, plus its own confusion matrix
     (saved automatically by ultralytics under runs/detect/val*/).

  2. Runs a custom per-instance error categorization, inspired by TIDE
     (Bolya et al., "TIDE: A General Toolbox for Identifying Object
     Detection Errors", ECCV 2020) -- matches every prediction against
     ground truth by IoU and buckets each into one of:
       TP        -- IoU >= iou_fg, correct class
       Cls       -- IoU >= iou_fg, wrong class (localization fine, classification wrong)
       Loc       -- iou_bg <= IoU < iou_fg, correct class (found it, box is off)
       ClsLoc    -- iou_bg <= IoU < iou_fg, wrong class (both wrong)
       Dupe-FP   -- would-be match, but a higher-confidence prediction already claimed that GT box
       Bkg-FP    -- IoU < iou_bg against every GT box (model hallucinated a detection)
       Miss-FN   -- a GT box with no prediction overlapping it above iou_bg at all
     This is a simplified, documented variant of TIDE's scheme (not a
     reimplementation of the original toolbox), chosen because it separates
     "wrong class" from "wrong box" from "missed entirely" from "invented" --
     exactly the taxonomy the investigation calls for -- rather than
     ultralytics' own confusion matrix, which only reports class confusion at
     a single fixed IoU and folds Loc/ClsLoc/Dupe into blunter buckets.

  3. For every non-TP instance (and a matched sample of TPs for comparison),
     computes PIXEL-ONLY heuristic proxies for physical properties that
     SSS-domain literature associates with detectability, and (optionally)
     the existing VAE's reconstruction error on that region:
       - target/background contrast proxy (bright-target-on-dark-background
         backscatter contrast, or the reverse)
       - a crude shadow-adjacency proxy (is one side of the box darker than
         the surrounding ring, consistent with an acoustic shadow -- NOT a
         validated shadow detector, see the module docstring warning below)
       - box aspect ratio (geometry proxy: elongated vs compact)
       - box area, normalized to image area, bucketed into small/medium/
         large via terciles computed from this dataset's own GT distribution
       - VAE reconstruction error on the (padded, resized) region, using the
         EXISTING trained vae_epoch100.pth -- no new VAE is trained here.

IMPORTANT ON THE PROXIES: "contrast_proxy" and "shadow_proxy" are computed
from pixel intensity alone, in an image that has already been through this
project's own preprocessing pipeline (dropout repair, normalization, and
-- depending on which test folder you point this at -- denoising/CLAHE).
They are NOT calibrated acoustic backscatter or validated shadow-geometry
measurements, and they say nothing about material. Treat them as weak,
image-only evidence to correlate against error categories empirically --
exactly the caution the investigation plan requires ("do not force a
physical explanation when the evidence is insufficient"). Whether a
correlation shows up is an empirical question this script answers per run,
not an assumption baked into it.

Outputs (under --out_dir):
  official_val_metrics.json   -- ultralytics' own mAP50/mAP50-95/P/R/per-class AP
  instances.csv                -- ONE ROW PER MATCHED/UNMATCHED INSTANCE, full dataset,
                                   with error type, classes, IoU, size bucket, physics
                                   proxies, VAE score -- this is the ground truth for the
                                   frequency table in the error taxonomy.
  summary_by_error_type.csv    -- counts + mean physics proxies per error type (the
                                   quick "what's actually going wrong, how often" table.
  summary_by_class.csv         -- per-class recall/precision-relevant counts from the
                                   custom matcher (cross-check against ultralytics' own).
  summary_by_size.csv          -- TP/Cls/Loc/Miss counts per size bucket ("performance
                                   by object size").
  panels/<error_type>/*.png    -- up to --panels_per_type representative example images
                                   per category, GT box in green, prediction in red/
                                   orange, labelled with class/conf/IoU -- for the actual
                                   visual inspection the investigation calls for (do not
                                   conclude from the CSVs alone).

Usage (quick trial on a handful of images first):
    python -m src.evaluation.error_analysis \\
        --weights_path best.pt \\
        --data_yaml data/processed/yolo_dataset_v2_preprocessed/data.yaml \\
        --vae_weights_path src/vae/vae_epoch100.pth \\
        --out_dir error_analysis_out --limit 50

Usage (full test split):
    python -m src.evaluation.error_analysis \\
        --weights_path best.pt \\
        --data_yaml data/processed/yolo_dataset_v2_preprocessed/data.yaml \\
        --vae_weights_path src/vae/vae_epoch100.pth \\
        --out_dir error_analysis_out
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from src.utils.config import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------
# Ground-truth loading
# --------------------------------------------------------------------------

def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """YOLO label txt (class cx cy w h, all normalized) -> [{class_id, xyxy}]
    in absolute pixel coords for this specific image."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append({"class_id": cls_id, "xyxy": [x1, y1, x2, y2], "area_norm": w * h})
    return boxes


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# TIDE-inspired instance matching
# --------------------------------------------------------------------------

def match_detections(
    gts: list[dict], preds: list[dict], iou_fg: float = 0.5, iou_bg: float = 0.1
) -> list[dict]:
    """Returns one dict per prediction, plus one per unmatched GT (Miss-FN).
    See module docstring for the category definitions."""
    gt_matched = [False] * len(gts)
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["conf"])
    results = []
    for pi in order:
        p = preds[pi]
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gts):
            v = iou(p["xyxy"], g["xyxy"])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi == -1 or best_iou < iou_bg:
            results.append({"type": "Bkg-FP", "pred": p, "gt": None, "iou": best_iou})
            continue
        g = gts[best_gi]
        if gt_matched[best_gi]:
            results.append({"type": "Dupe-FP", "pred": p, "gt": g, "iou": best_iou})
            continue
        if best_iou >= iou_fg:
            err_type = "TP" if p["class_id"] == g["class_id"] else "Cls"
        else:
            err_type = "Loc" if p["class_id"] == g["class_id"] else "ClsLoc"
        results.append({"type": err_type, "pred": p, "gt": g, "iou": best_iou})
        gt_matched[best_gi] = True

    for gi, g in enumerate(gts):
        if not gt_matched[gi]:
            results.append({"type": "Miss-FN", "pred": None, "gt": g, "iou": 0.0})
    return results


# --------------------------------------------------------------------------
# Pixel-only physics PROXIES (explicitly heuristic -- see module docstring)
# --------------------------------------------------------------------------

def _clip_box(xyxy: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


def contrast_and_shadow_proxy(gray: np.ndarray, xyxy: list[float], expand: float = 0.6) -> dict:
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = _clip_box(xyxy, w, h)
    if x2 <= x1 or y2 <= y1:
        return {"contrast_proxy": float("nan"), "shadow_proxy": float("nan"), "shadow_side": None}

    bw, bh = x2 - x1, y2 - y1
    ex, ey = max(1, int(bw * expand)), max(1, int(bh * expand))
    ox1, oy1 = max(0, x1 - ex), max(0, y1 - ey)
    ox2, oy2 = min(w, x2 + ex), min(h, y2 + ey)

    outer = gray[oy1:oy2, ox1:ox2].astype(np.float32)
    ring_mask = np.ones(outer.shape, dtype=bool)
    iy1, iy2 = y1 - oy1, y2 - oy1
    ix1, ix2 = x1 - ox1, x2 - ox1
    ring_mask[max(0, iy1):max(0, iy2), max(0, ix1):max(0, ix2)] = False
    ring_vals = outer[ring_mask]
    target_vals = gray[y1:y2, x1:x2].astype(np.float32)

    ring_mean = float(ring_vals.mean()) if ring_vals.size else float("nan")
    target_mean = float(target_vals.mean()) if target_vals.size else float("nan")
    contrast_proxy = target_mean - ring_mean

    # Darkest of the 4 adjacent strips (same size as the box, just outside
    # it) vs the ring mean -- a crude "is there a dark region right next to
    # this target" check, NOT a validated acoustic-shadow detector.
    strips = {
        "left": gray[y1:y2, max(0, x1 - bw):x1],
        "right": gray[y1:y2, x2:min(w, x2 + bw)],
        "up": gray[max(0, y1 - bh):y1, x1:x2],
        "down": gray[y2:min(h, y2 + bh), x1:x2],
    }
    best_side, best_strength = None, -1e9
    for side, strip in strips.items():
        if strip.size == 0:
            continue
        strip_mean = float(strip.astype(np.float32).mean())
        strength = ring_mean - strip_mean  # positive => strip darker than surrounding ring
        if strength > best_strength:
            best_strength, best_side = strength, side

    return {
        "contrast_proxy": contrast_proxy,
        "shadow_proxy": best_strength if best_side else float("nan"),
        "shadow_side": best_side,
    }


def geometry_proxy(xyxy: list[float]) -> dict:
    x1, y1, x2, y2 = xyxy
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return {"aspect_ratio": bw / bh, "box_w": bw, "box_h": bh}


# --------------------------------------------------------------------------
# Size buckets, computed from this dataset's own GT area distribution
# --------------------------------------------------------------------------

def compute_size_terciles(areas: list[float]) -> tuple[float, float]:
    if not areas:
        return 0.01, 0.04
    arr = np.array(areas)
    return float(np.percentile(arr, 33.3)), float(np.percentile(arr, 66.6))


def size_bucket(area_norm: float, t1: float, t2: float) -> str:
    if area_norm <= t1:
        return "small"
    if area_norm <= t2:
        return "medium"
    return "large"


# --------------------------------------------------------------------------
# Visualization panels
# --------------------------------------------------------------------------

def draw_panel(image: np.ndarray, class_names: list[str], record: dict) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    gt, pred = record.get("gt"), record.get("pred")
    if gt is not None:
        x1, y1, x2, y2 = (int(v) for v in gt["xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(canvas, f"GT:{class_names[gt['class_id']]}", (x1, max(0, y1 - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    if pred is not None:
        x1, y1, x2, y2 = (int(v) for v in pred["xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 100, 255), 2)
        cv2.putText(canvas, f"P:{class_names[pred['class_id']]} {pred['conf']:.2f}",
                    (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{record['type']} IoU={record['iou']:.2f}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    import yaml
    from ultralytics import YOLO

    from src.vae.test_vae import load_model as load_vae_model
    from src.vae.vae_analysis import crop_box, run_vae

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    panels_dir = out_dir / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data_yaml)
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    class_names = list(cfg["names"].values()) if isinstance(cfg["names"], dict) else list(cfg["names"])
    logger.info("Classes: %s", class_names)

    # --- Step 1: official ultralytics val() metrics -----------------------
    yolo = YOLO(args.weights_path)
    logger.info("Running official model.val() for baseline mAP/P/R/per-class AP ...")
    val_metrics = yolo.val(data=str(data_yaml), split="test", verbose=False)
    official = {
        "mAP50": float(val_metrics.box.map50),
        "mAP50-95": float(val_metrics.box.map),
        "precision_mean": float(val_metrics.box.mp),
        "recall_mean": float(val_metrics.box.mr),
        "per_class_AP50": {},
        "per_class_AP50-95": {},
    }
    for i, cls_idx in enumerate(val_metrics.box.ap_class_index):
        name = class_names[int(cls_idx)]
        official["per_class_AP50"][name] = float(val_metrics.box.ap50[i])
        official["per_class_AP50-95"][name] = float(val_metrics.box.ap[i])
    with open(out_dir / "official_val_metrics.json", "w") as f:
        json.dump(official, f, indent=2)
    logger.info("Official metrics: mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f",
                official["mAP50"], official["mAP50-95"], official["precision_mean"], official["recall_mean"])
    logger.info("(ultralytics also wrote its own confusion matrix + PR curves under runs/detect/val*/)")

    # --- Step 2: figure out image/label dirs -------------------------------
    images_dir = data_yaml.parent / cfg["test"]
    labels_dir = images_dir.parent / "labels"
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        image_paths = image_paths[: args.limit]
    logger.info("Custom per-instance error analysis over %d images from %s", len(image_paths), images_dir)

    # First pass: gather all GT areas to set data-driven size terciles.
    all_areas = []
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        for g in load_yolo_labels(labels_dir / f"{p.stem}.txt", w, h):
            all_areas.append(g["area_norm"])
    t1, t2 = compute_size_terciles(all_areas)
    logger.info("Size-bucket thresholds (data-driven terciles of normalized GT area): small<=%.5f, medium<=%.5f, large>%.5f",
                t1, t2, t2)

    vae_model = None
    if not args.skip_vae:
        vae_model = load_vae_model(args.vae_weights_path, device)

    panel_counts = defaultdict(int)
    for et in ["TP", "Cls", "Loc", "ClsLoc", "Dupe-FP", "Bkg-FP", "Miss-FN"]:
        (panels_dir / et).mkdir(parents=True, exist_ok=True)

    rows = []
    n_done = 0
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Unreadable image, skipping: %s", path)
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(labels_dir / f"{path.stem}.txt", w, h)

        results = yolo.predict(source=str(path), conf=args.conf, verbose=False)
        r = results[0]
        preds = [
            {"class_id": int(box.cls[0]), "conf": float(box.conf[0]),
             "xyxy": box.xyxy[0].cpu().numpy().tolist()}
            for box in r.boxes
        ]

        matches = match_detections(gts, preds, iou_fg=args.iou_fg, iou_bg=args.iou_bg)
        for m in matches:
            ref_box = (m["gt"] or m["pred"])["xyxy"]
            geo = geometry_proxy(ref_box)
            phys = contrast_and_shadow_proxy(img, ref_box)
            area_norm = m["gt"]["area_norm"] if m["gt"] else (geo["box_w"] * geo["box_h"]) / (w * h)
            bucket = size_bucket(area_norm, t1, t2)

            vae_err = None
            if vae_model is not None:
                crop = crop_box(img, [int(v) for v in ref_box], pad=8)
                if crop.size > 0:
                    vae_err = run_vae(vae_model, crop, device)["scalar_error"]

            row = {
                "image": path.name,
                "error_type": m["type"],
                "iou": round(m["iou"], 4),
                "gt_class": class_names[m["gt"]["class_id"]] if m["gt"] else "",
                "pred_class": class_names[m["pred"]["class_id"]] if m["pred"] else "",
                "pred_conf": round(m["pred"]["conf"], 4) if m["pred"] else "",
                "size_bucket": bucket,
                "area_norm": round(area_norm, 6),
                "aspect_ratio": round(geo["aspect_ratio"], 3),
                "contrast_proxy": round(phys["contrast_proxy"], 3) if phys["contrast_proxy"] == phys["contrast_proxy"] else "",
                "shadow_proxy": round(phys["shadow_proxy"], 3) if phys["shadow_proxy"] == phys["shadow_proxy"] else "",
                "shadow_side": phys["shadow_side"] or "",
                "vae_recon_error": round(vae_err, 6) if vae_err is not None else "",
            }
            rows.append(row)

            if panel_counts[m["type"]] < args.panels_per_type:
                panel = draw_panel(img, class_names, m)
                cv2.imwrite(str(panels_dir / m["type"] / f"{path.stem}_{panel_counts[m['type']]:02d}.png"), panel)
                panel_counts[m["type"]] += 1

        n_done += 1
        if n_done % 200 == 0 or n_done == len(image_paths):
            logger.info("Processed %d/%d images", n_done, len(image_paths))

    with open(out_dir / "instances.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                 ["image", "error_type", "iou", "gt_class", "pred_class", "pred_conf",
                                  "size_bucket", "area_norm", "aspect_ratio", "contrast_proxy",
                                  "shadow_proxy", "shadow_side", "vae_recon_error"])
        writer.writeheader()
        writer.writerows(rows)

    _write_aggregates(rows, out_dir)
    logger.info("Done. %d instances -> %s (panels: %s)", len(rows), out_dir, panels_dir)


def _mean(vals: list[float]) -> float | str:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else ""


def _write_aggregates(rows: list[dict], out_dir: Path) -> None:
    by_type = defaultdict(list)
    by_class = defaultdict(lambda: defaultdict(int))
    by_size = defaultdict(lambda: defaultdict(int))

    for row in rows:
        by_type[row["error_type"]].append(row)
        key_class = row["gt_class"] or row["pred_class"]
        by_class[key_class][row["error_type"]] += 1
        by_size[row["size_bucket"]][row["error_type"]] += 1

    with open(out_dir / "summary_by_error_type.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_type", "count", "mean_iou", "mean_contrast_proxy", "mean_shadow_proxy",
                          "mean_aspect_ratio", "mean_vae_recon_error"])
        for et, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            writer.writerow([
                et, len(items),
                _mean([i["iou"] for i in items]),
                _mean([i["contrast_proxy"] for i in items]),
                _mean([i["shadow_proxy"] for i in items]),
                _mean([i["aspect_ratio"] for i in items]),
                _mean([i["vae_recon_error"] for i in items]),
            ])

    error_types = ["TP", "Cls", "Loc", "ClsLoc", "Dupe-FP", "Bkg-FP", "Miss-FN"]
    with open(out_dir / "summary_by_class.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class"] + error_types)
        for cls, counts in sorted(by_class.items()):
            writer.writerow([cls] + [counts.get(et, 0) for et in error_types])

    with open(out_dir / "summary_by_size.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size_bucket"] + error_types)
        for bucket in ["small", "medium", "large"]:
            counts = by_size.get(bucket, {})
            writer.writerow([bucket] + [counts.get(et, 0) for et in error_types])


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights_path", type=str, required=True, help="Trained YOLO checkpoint, e.g. best.pt.")
    p.add_argument("--data_yaml", type=str, required=True)
    p.add_argument("--vae_weights_path", type=str, default=None, help="Existing VAE checkpoint (e.g. src/vae/vae_epoch100.pth). Required unless --skip_vae.")
    p.add_argument("--skip_vae", action="store_true", help="Skip the VAE cross-check (faster, e.g. for a first quick pass).")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions fed into the matcher.")
    p.add_argument("--iou_fg", type=float, default=0.5, help="IoU >= this + correct class = TP.")
    p.add_argument("--iou_bg", type=float, default=0.1, help="IoU < this against every GT = background FP.")
    p.add_argument("--panels_per_type", type=int, default=20, help="Max example panel images saved per error category.")
    p.add_argument("--out_dir", type=str, default="error_analysis_out")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N images -- use for a quick trial run first.")
    p.add_argument("--device", type=str, default=None)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if not args.skip_vae and not args.vae_weights_path:
        raise SystemExit("--vae_weights_path is required unless --skip_vae is passed.")
    run(args)

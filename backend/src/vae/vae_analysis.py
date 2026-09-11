"""Full VAE anomaly-analysis suite: runs a trained ConvVAE checkpoint over a
folder of images and, per image, saves every visualization module (original,
reconstruction, anomaly overlay, plain difference heatmap, edge/contour map,
difference map with a legend, 3D anomaly-score surface) into its own
subfolder -- plus, optionally, the same analysis run again on each bounding
box a trained YOLO checkpoint (best.pt) detects in that image, so a
detection can be cross-checked against how anomalous the VAE finds that
specific region. This is the "visual cue" half of the project's
evidence-driven confidence engine (see the project description): YOLO says
WHAT it thinks an object is, the VAE independently scores HOW UNUSUAL that
patch of seafloor looks relative to normal background -- two votes for the
same detection instead of one.

Per-image output layout (mirrors the reference mockup's numbered modules):
    <out_dir>/<image_stem>/
        01_original.png
        02_reconstruction.png
        03_anomaly_overlay.png       -- heatmap blended over the original
        04_difference_heatmap.png    -- plain colorized |orig - recon|
        05_edge_contour_map.png      -- contours of the thresholded error map
        06_difference_map_legend.png -- grayscale diff with a High/Low colorbar
        07_3d_anomaly_surface.png    -- 3D surface plot of the error map
        summary.json                 -- whole-image error + per-box results
        boxes/box00_<class>_conf<c>_vaeerr<e>/
            03_anomaly_overlay.png
            04_difference_heatmap.png
            (one such folder per YOLO detection, only if --yolo_weights given)

Two master CSVs land directly in --out_dir:
    analysis_summary.csv  -- one row per image (whole-image error, box count)
    box_level_scores.csv  -- one row per YOLO detection (class, yolo conf,
                              vae error, bbox) -- only written if
                              --yolo_weights was given

Usage (VAE-only, no YOLO):
    python -m src.vae.vae_analysis \\
        --weights_path src/vae/vae_epoch100.pth \\
        --image_dir data/processed/yolo_dataset_v2_preprocessed/test/images \\
        --out_dir vae_analysis_out

Usage (VAE + YOLO box-level analysis):
    python -m src.vae.vae_analysis \\
        --weights_path src/vae/vae_epoch100.pth \\
        --image_dir data/processed/yolo_dataset_v2_preprocessed/test/images \\
        --yolo_weights weights/best.pt --yolo_conf 0.25 \\
        --out_dir vae_analysis_out
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless -- this project has no display server
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.config import get_logger
from src.vae.model import ConvVAE
from src.vae.test_vae import load_model

logger = get_logger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------
# Core VAE pass -- one reconstruction + error map, everything else below is
# a different way of visualizing this same result.
# --------------------------------------------------------------------------

def run_vae(model: ConvVAE, image: np.ndarray, device: torch.device) -> dict:
    """image: (H, W) uint8 grayscale, any size (resized to model.input_size).
    Returns original/reconstruction as uint8 (H, W) at model.input_size, the
    per-pixel error map normalized to [0, 1], and the scalar reconstruction
    error (same metric ConvVAE.reconstruction_error reports)."""
    if image.shape != (model.input_size, model.input_size):
        image = cv2.resize(image, (model.input_size, model.input_size), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon, _, _ = model(x)
        error = model.reconstruction_error(x).item()

    orig_u8 = (x[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    rec_u8 = (recon[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    diff = np.abs(orig_u8.astype(np.float32) - rec_u8.astype(np.float32))
    error_map = diff / diff.max() if diff.max() > 0 else diff  # normalized [0, 1], same shape as image

    return {"original": orig_u8, "reconstruction": rec_u8, "error_map": error_map, "scalar_error": error}


# --------------------------------------------------------------------------
# Visualization modules -- each takes the run_vae() result and writes one
# PNG. Kept as separate functions (not methods) so any one of them can be
# reused/extended independently -- e.g. for a future module 4, 5, 6, 9...
# matching more of the reference mockup.
# --------------------------------------------------------------------------

def save_anomaly_overlay(result: dict, path: Path, alpha: float = 0.55) -> None:
    """Module 3: heatmap blended directly over the original image -- where
    the anomaly is, in the context of what it actually looks like."""
    orig_bgr = cv2.cvtColor(result["original"], cv2.COLOR_GRAY2BGR)
    heatmap = cv2.applyColorMap((result["error_map"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, orig_bgr, 1 - alpha, 0)
    cv2.imwrite(str(path), overlay)


def save_difference_heatmap(result: dict, path: Path) -> None:
    """Module 4: the plain colorized error map alone, no blending."""
    heatmap = cv2.applyColorMap((result["error_map"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(path), heatmap)


def save_edge_contour_map(result: dict, path: Path, threshold: float = 0.3) -> None:
    """Module 8: outlines of the anomalous region(s) -- threshold the error
    map, trace contours, draw them in white on black. Highlights object
    BOUNDARIES rather than a filled blob, useful for reading off shape."""
    mask = (result["error_map"] >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(mask)
    cv2.drawContours(canvas, contours, -1, 255, 2)  # thickness=2 -- bolder, cleaner outline
    cv2.imwrite(str(path), canvas)


# Dark theme shared by the two matplotlib-based modules, to match the
# reference mockup's dark cards rather than matplotlib's default white
# figure background.
_DARK_BG = "#0d1117"
_DARK_FG = "#e6edf3"


def save_difference_map_legend(result: dict, path: Path) -> None:
    """Module 11: grayscale difference map with an explicit High/Low
    difference colorbar -- same information as module 4, but calibrated
    (a bar you can read a value off of) rather than just colorful."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    im = ax.imshow(result["error_map"], cmap="gray", vmin=0, vmax=1)
    ax.set_title("Difference Map (Original - Reconstruction)", color=_DARK_FG, fontsize=11)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=_DARK_FG, labelcolor=_DARK_FG)
    cbar.outline.set_edgecolor(_DARK_FG)
    cbar.ax.text(0.5, 1.03, "High", transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=9, color=_DARK_FG)
    cbar.ax.text(0.5, -0.03, "Low", transform=cbar.ax.transAxes, ha="center", va="top", fontsize=9, color=_DARK_FG)
    cbar.set_label("Difference", color=_DARK_FG)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=_DARK_BG)
    plt.close(fig)


def save_3d_anomaly_surface(result: dict, path: Path, grid_size: int = 64) -> None:
    """Module 10: 3D surface plot of the error map, where height = anomaly
    score. Downsampled to grid_size x grid_size first (default 64) --
    plotting all 512x512 points is slow and visually indistinguishable
    from a coarser grid at this figure size; raise grid_size for a finer
    (slower) surface."""
    small = cv2.resize(result["error_map"], (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)  # smooths per-pixel jaggies into a readable surface, cosmetic only
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=(5.5, 5), facecolor=_DARK_BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=_DARK_BG)
    ax.plot_surface(xx, yy, small, cmap="jet", vmin=0, vmax=1, rcount=grid_size, ccount=grid_size, antialiased=True)
    ax.set_xlabel("X (pixels)", color=_DARK_FG)
    ax.set_ylabel("Y (pixels)", color=_DARK_FG)
    ax.set_zlabel("Anomaly Score", color=_DARK_FG)
    ax.set_zlim(0, 1)
    ax.set_title("3D Anomaly Surface", color=_DARK_FG, fontsize=11)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0.04))
        axis.line.set_color(_DARK_FG)
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.15)
    ax.tick_params(colors=_DARK_FG)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=_DARK_BG)
    plt.close(fig)


def save_all_modules(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "01_original.png"), result["original"])
    cv2.imwrite(str(out_dir / "02_reconstruction.png"), result["reconstruction"])
    save_anomaly_overlay(result, out_dir / "03_anomaly_overlay.png")
    save_difference_heatmap(result, out_dir / "04_difference_heatmap.png")
    save_edge_contour_map(result, out_dir / "05_edge_contour_map.png")
    save_difference_map_legend(result, out_dir / "06_difference_map_legend.png")
    save_3d_anomaly_surface(result, out_dir / "07_3d_anomaly_surface.png")


# --------------------------------------------------------------------------
# YOLO box-level analysis -- crop each detection, run the same VAE analysis
# on just that region.
# --------------------------------------------------------------------------

def detect_boxes(yolo_model, image_path: str, conf: float) -> list[dict]:
    """Runs YOLO on one image, returns [{cls_name, conf, xyxy}, ...] in the
    ORIGINAL image's pixel coordinates (not the VAE's 512x512 space)."""
    results = yolo_model.predict(source=image_path, conf=conf, verbose=False)
    boxes = []
    r = results[0]
    names = r.names
    for box in r.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
        boxes.append({
            "cls_name": names[int(box.cls[0])],
            "conf": float(box.conf[0]),
            "xyxy": xyxy,
        })
    return boxes


def crop_box(image: np.ndarray, xyxy: list[int], pad: int = 8) -> np.ndarray:
    """Crops xyxy out of image with a small pixel margin (context around
    the detection, not just the tight box), clamped to image bounds."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return image[y1:y2, x1:x2]


# --------------------------------------------------------------------------
# Per-image + whole-folder orchestration
# --------------------------------------------------------------------------

def analyze_image(
    model: ConvVAE,
    image_path: Path,
    device: torch.device,
    out_dir: Path,
    yolo_model=None,
    yolo_conf: float = 0.25,
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_out_dir = out_dir / image_path.stem
    result = run_vae(model, image, device)
    save_all_modules(result, image_out_dir)

    summary = {"image": str(image_path), "whole_image_error": result["scalar_error"], "boxes": []}

    if yolo_model is not None:
        boxes = detect_boxes(yolo_model, str(image_path), yolo_conf)
        for i, box in enumerate(boxes):
            crop = crop_box(image, box["xyxy"])
            if crop.size == 0:
                logger.warning("Empty crop for box %d in %s -- skipping", i, image_path)
                continue
            box_result = run_vae(model, crop, device)
            tag = f"box{i:02d}_{box['cls_name']}_conf{box['conf']:.2f}_vaeerr{box_result['scalar_error']:.5f}"
            box_out_dir = image_out_dir / "boxes" / tag
            box_out_dir.mkdir(parents=True, exist_ok=True)
            save_anomaly_overlay(box_result, box_out_dir / "03_anomaly_overlay.png")
            save_difference_heatmap(box_result, box_out_dir / "04_difference_heatmap.png")
            summary["boxes"].append({
                "cls_name": box["cls_name"], "yolo_conf": box["conf"], "xyxy": box["xyxy"],
                "vae_error": box_result["scalar_error"],
            })

    with open(image_out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.weights_path, device)

    yolo_model = None
    if args.yolo_weights:
        from ultralytics import YOLO

        yolo_model = YOLO(args.yolo_weights)
        logger.info("Loaded YOLO weights from %s (conf threshold=%.2f)", args.yolo_weights, args.yolo_conf)

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if args.limit:
        image_paths = image_paths[: args.limit]

    image_rows = []
    box_rows = []
    for i, path in enumerate(image_paths, 1):
        summary = analyze_image(model, path, device, out_dir, yolo_model=yolo_model, yolo_conf=args.yolo_conf)
        image_rows.append({
            "image": summary["image"], "whole_image_error": summary["whole_image_error"],
            "num_boxes": len(summary["boxes"]),
        })
        for b in summary["boxes"]:
            box_rows.append({"image": summary["image"], **b})
        if i % 50 == 0 or i == len(image_paths):
            logger.info("Analyzed %d/%d images", i, len(image_paths))

    with open(out_dir / "analysis_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "whole_image_error", "num_boxes"])
        writer.writeheader()
        writer.writerows(image_rows)
    logger.info("Wrote %s", out_dir / "analysis_summary.csv")

    if yolo_model is not None:
        with open(out_dir / "box_level_scores.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "cls_name", "yolo_conf", "xyxy", "vae_error"])
            writer.writeheader()
            writer.writerows(box_rows)
        logger.info("Wrote %s (%d detections)", out_dir / "box_level_scores.csv", len(box_rows))

    logger.info("Done: %d images analyzed -> %s", len(image_paths), out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights_path", type=str, required=True, help="VAE checkpoint from train_vae.py.")
    p.add_argument("--image_dir", type=str, required=True, help="Folder of images to analyze (e.g. the YOLO test split).")
    p.add_argument("--out_dir", type=str, default="vae_analysis_out")
    p.add_argument("--yolo_weights", type=str, default=None, help="Trained YOLO checkpoint (e.g. weights/best.pt) -- if given, also runs the VAE on every detected bounding box, cropped from the original image.")
    p.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO detection confidence threshold. Ignored if --yolo_weights not given.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N images -- useful for a quick trial run before committing to the whole folder.")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

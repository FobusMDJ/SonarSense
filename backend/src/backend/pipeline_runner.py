"""Orchestrates one end-to-end run of the pipeline for a single uploaded
sonar log: ingest -> preprocess -> YOLO detect -> VAE analyze -> confidence
score -> geolocate -> persist to SQLite -> write JSON/CSV report.

This is the one place all the pipeline stages get wired together; the
FastAPI layer (main.py) just calls `process_log()` in a background task and
relays the progress callback over the log's WebSocket connection.

DENOISING / CONTRAST DEFAULTS -- read before changing them: best.pt and the
VAE checkpoint were both trained on images from
preprocess_yolo_dataset_v2.py, which deliberately runs NEITHER denoise() NOR
CLAHE ("trains on preprocessed-but-not-denoised images by design" -- that
script's own docstring). Before src/preprocessing/denoising.py grew a
method="none" option, PreprocessingPipeline.process_record() had no way to
reproduce that at inference: it always ran the Lee filter (config/
preprocessing.yaml's unconditional default) + CLAHE, so every live
detection was already running on a distribution the model never saw
training. `denoise_method`/`contrast_method` below default to "none"
specifically to close that gap and match what the model actually learned.
Passing "lee" or "blind2unblind" (B2U -- trained checkpoints exist at
models/B2U.pth and models/B2Ueph2.pth, see config/backend.yaml's
b2u_weights_path) is a legitimate thing to want to try in production even
though it's technically out-of-distribution for the current checkpoints --
just go in with that tradeoff in mind, and re-train/fine-tune on
denoised images before trusting it as the new default.

KNOWN V1 LIMITATIONS (stated plainly, not hidden):
  - Ingestion is fully materialized into memory before processing starts
    (`list(ingest_source(...))`), so frame-count-based progress reporting is
    possible. Fine for demo-scale logs; a very large raw XTF log could use a
    lot of memory. Streaming both directions (progress reporting AND low
    memory) needs a two-pass design (count first, then stream) -- not done
    here to keep v1 straightforward.
  - `pixels_to_meters` must be given at the FINAL (post-resize) image
    resolution, not the source sonar's native resolution -- i.e. it already
    accounts for whatever `target_size` the preprocessing config resizes
    to. Simpler than threading a resize-scale correction through
    geolocation, but it means this value needs to be re-derived if
    target_size changes. Documented here rather than silently assumed.
  - For XTF-sourced frames, a detection's row is mapped back to its
    original ping's NavFix by PROPORTION (row_in_final / final_height *
    tile_height_at_ingest), because preprocessing resizes the tile before
    YOLO/VAE ever see it. This is an approximation, not an exact ping
    lookup -- reasonable at typical resize ratios, but worth re-checking
    once a real XTF file is available to validate against.
"""

from __future__ import annotations

import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from src.backend import db_backend as db
from src.backend.model_release import INFERENCE_LOCK, JOB_LOCK, get_vae_model, get_yolo_model
from src.confidence.scoring import score_detection
from src.geolocation.geojson_export import write_geojson
from src.geolocation.georeference import geolocate_detection
from src.geolocation.nav import NavFix
from src.preprocessing.ingestion import SSSRecord, ingest_source
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import get_logger, load_config
from src.vae.vae_analysis import crop_box, run_vae, save_all_modules

logger = get_logger(__name__)

ProgressCallback = Callable[[dict], None]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(progress_cb: Optional[ProgressCallback], **kwargs) -> None:
    if progress_cb is not None:
        try:
            progress_cb(kwargs)
        except Exception:
            logger.exception("progress_cb raised -- continuing pipeline regardless")


def _source_format_of(records: list[SSSRecord]) -> str:
    if not records:
        return "unknown"
    return records[0].metadata.get("source_format", "image")


def _nav_fix_for_detection(record: SSSRecord, bbox: list[float], final_shape: tuple[int, int]) -> Optional[NavFix]:
    """Resolve the NavFix relevant to one detection, format-dependent."""
    meta = record.metadata
    if "nav_fix" in meta:  # image+sidecar path: one fix per whole frame
        return meta["nav_fix"]

    nav_fixes_per_row = meta.get("nav_fixes_per_row")
    if nav_fixes_per_row:
        final_h = final_shape[0]
        y_center = (bbox[1] + bbox[3]) / 2.0
        tile_h = len(nav_fixes_per_row)
        row = int((y_center / max(1, final_h)) * tile_h)
        row = max(0, min(tile_h - 1, row))
        return nav_fixes_per_row[row]

    return None


def _build_preprocessing_config(denoise_method: str, denoise_weights_path: Optional[Path],
                                 contrast_method: str) -> dict:
    """Loads config/preprocessing.yaml (dropout/normalization/target_size
    etc. stay whatever that file says) and overrides just the denoising and
    contrast-enhancement sections with what THIS request asked for -- see
    process_log's docstring / the module docstring for why "none"/"none" is
    the default rather than config/preprocessing.yaml's own "lee"/"clahe"
    (that file's defaults are for offline batch preprocessing, not this
    train/serve-consistency-sensitive live path)."""
    cfg = dict(load_config())  # config/preprocessing.yaml, the shared default
    cfg["denoising"] = {
        "method": denoise_method,
        "window_size": 7,
        "weights_path": str(denoise_weights_path) if denoise_weights_path else None,
        "fallback_on_missing_weights": True,
    }
    enh = dict(cfg.get("enhancement", {}))
    enh["contrast_method"] = contrast_method
    cfg["enhancement"] = enh
    return cfg


def _process_log_unlocked(
    log_id: str,
    source_path: Path,
    yolo_weights: Path,
    vae_weights: Path,
    db_path: Path,
    output_dir: Path,
    pixels_to_meters: float,
    nav_sidecar_path: Optional[Path] = None,
    yolo_conf: float = 0.25,
    port_is_left: bool = True,
    device: Optional[str] = None,
    denoise_method: str = "none",
    denoise_weights_path: Optional[Path] = None,
    contrast_method: str = "none",
    progress_cb: Optional[ProgressCallback] = None,
) -> None:
    out_dir = output_dir / log_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with db.get_connection(db_path) as conn:
        db.update_log_status(conn, log_id, "processing")

    try:
        _emit(progress_cb, log_id=log_id, stage="ingest", message=f"Ingesting {source_path.name}")
        records = list(ingest_source(source_path, nav_sidecar_path))
        n_frames = len(records)
        if n_frames == 0:
            raise ValueError(f"No frames could be ingested from {source_path}")
        source_format = _source_format_of(records)

        preproc_cfg = _build_preprocessing_config(denoise_method, denoise_weights_path, contrast_method)
        pipeline = PreprocessingPipeline(config=preproc_cfg)
        yolo = get_yolo_model(str(yolo_weights))
        vae_model = get_vae_model(str(vae_weights), str(torch_device))
        detector_inference_ms = 0.0

        frame_records = []  # per-frame working state, built up across passes
        for i, record in enumerate(records):
            _emit(progress_cb, log_id=log_id, stage="preprocess", frame_index=i, n_frames_total=n_frames,
                  message=f"Preprocessing frame {i + 1}/{n_frames}")
            stage_result = pipeline.process_record(record)
            final = stage_result.final

            _emit(progress_cb, log_id=log_id, stage="detect", frame_index=i, n_frames_total=n_frames,
                  message=f"Running YOLO on frame {i + 1}/{n_frames}")
            started = time.perf_counter()
            with INFERENCE_LOCK:
                results = yolo.predict(source=final, conf=yolo_conf, verbose=False)
            detector_inference_ms += (time.perf_counter() - started) * 1000.0
            r = results[0]
            names = r.names
            boxes = [
                {"class_name": names[int(b.cls[0])], "conf": float(b.conf[0]),
                 "xyxy": b.xyxy[0].cpu().numpy().tolist()}
                for b in r.boxes
            ]

            _emit(progress_cb, log_id=log_id, stage="vae", frame_index=i, n_frames_total=n_frames,
                  message=f"Running VAE analysis on frame {i + 1}/{n_frames}")
            with INFERENCE_LOCK:
                vae_result = run_vae(vae_model, final, torch_device)
            frame_out_dir = out_dir / "vae" / record.record_id
            save_all_modules(vae_result, frame_out_dir)

            box_vae_errors = []
            for box in boxes:
                crop = crop_box(final, [int(v) for v in box["xyxy"]])
                if crop.size:
                    with INFERENCE_LOCK:
                        box_vae_errors.append(run_vae(vae_model, crop, torch_device)["scalar_error"])
                else:
                    box_vae_errors.append(None)

            frame_records.append({
                "record": record,
                "final": final,
                "boxes": boxes,
                "box_vae_errors": box_vae_errors,
                "whole_image_error": vae_result["scalar_error"],
                "vae_panel_dir": str(frame_out_dir),
            })

        # Whole-image anomaly percentile within THIS log's own frames (same
        # convention as claude/phase1-baseline-error-analysis.md: 0 = most
        # anomalous/largest reconstruction error, 1 = least anomalous).
        # ranks[i]=0 for the smallest error (least anomalous); percentile
        # flips that so smallest error -> 1.0, largest error -> 0.0.
        errors = np.array([fr["whole_image_error"] for fr in frame_records])
        if len(errors) > 1:
            ranks = errors.argsort().argsort()
            percentiles = 1.0 - (ranks / (len(errors) - 1))
        else:
            # can't rank against a population of one frame -- neutral, not "most anomalous"
            percentiles = np.full_like(errors, 0.5)

        _emit(progress_cb, log_id=log_id, stage="confidence", message="Scoring confidence + geolocating detections")
        n_detections = 0
        with db.get_connection(db_path) as conn:
            for frame_idx, fr in enumerate(frame_records):
                record = fr["record"]
                final = fr["final"]
                percentile = float(percentiles[frame_idx])
                db.insert_frame_analysis(conn, {
                    "log_id": log_id,
                    "frame_index": frame_idx,
                    "frame_record_id": record.record_id,
                    "whole_image_error": fr["whole_image_error"],
                    "percentile": percentile,
                    "vae_panel_dir": fr["vae_panel_dir"],
                })
                for box_idx, box in enumerate(fr["boxes"]):
                    conf_result = score_detection(
                        yolo_conf=box["conf"], class_name=box["class_name"], xyxy=box["xyxy"],
                        gray_image=final, vae_whole_image_percentile=percentile,
                    )
                    nav_fix = _nav_fix_for_detection(record, box["xyxy"], final.shape[:2])
                    geo = geolocate_detection(
                        xyxy=box["xyxy"], image_width_px=final.shape[1], image_height_px=final.shape[0],
                        nav_fix=nav_fix, pixels_to_meters=pixels_to_meters, port_is_left=port_is_left,
                    )
                    det = {
                        "id": str(uuid.uuid4()),
                        "log_id": log_id,
                        "frame_index": frame_idx,
                        "frame_record_id": record.record_id,
                        "frame_image_path": str(record.source_path),
                        "class_name": box["class_name"],
                        "yolo_conf": box["conf"],
                        "bbox": box["xyxy"],
                        "confidence_score": conf_result.score,
                        "confidence_label": conf_result.label,
                        "confidence_breakdown": conf_result.breakdown,
                        "vae_box_error": fr["box_vae_errors"][box_idx],
                        "vae_whole_image_error": fr["whole_image_error"],
                        "vae_whole_image_percentile": percentile,
                        "lat": geo.lat if geo.method == "nav_fix" else None,
                        "lon": geo.lon if geo.method == "nav_fix" else None,
                        "geo_method": geo.method,
                        "depth_m": geo.depth_m,  # None unless the nav sidecar/XTF actually carried one
                        # Debris footprint (4-corner polygon, not just the center point above) --
                        # None whenever geo.footprint is None (placeholder detections, same rule as lat/lon).
                        # Stored pre-flipped to GeoJSON's [lon, lat] coordinate order so
                        # geojson_export.py can drop it straight into a Polygon with no reformatting.
                        "footprint_geojson": json.dumps([[lon, lat] for lat, lon in geo.footprint]) if geo.footprint else None,
                        "vae_panel_dir": fr["vae_panel_dir"],
                        "created_at": _now_iso(),
                    }
                    db.insert_detection(conn, det)
                    n_detections += 1
            db.update_log_counts(conn, log_id, n_frames, n_detections)
            db.update_log_performance(conn, log_id, detector_inference_ms, n_frames)
            db.update_log_status(conn, log_id, "done", completed_at=_now_iso())

        _write_reports(db_path, log_id, out_dir)
        _emit(progress_cb, log_id=log_id, stage="done", n_frames_total=n_frames,
              message=f"Done: {n_frames} frames, {n_detections} detections")

    except Exception as exc:
        logger.exception("process_log failed for %s", log_id)
        with db.get_connection(db_path) as conn:
            db.update_log_status(conn, log_id, "error", error_message=str(exc), completed_at=_now_iso())
        _emit(progress_cb, log_id=log_id, stage="error", message=str(exc))
        raise


def process_log(*args, **kwargs) -> None:
    """Serialize complete local jobs so cached models cannot be loaded or
    used concurrently by competing upload threads."""
    with JOB_LOCK:
        _process_log_unlocked(*args, **kwargs)


def _write_reports(db_path: Path, log_id: str, out_dir: Path) -> None:
    """Per the challenge brief: 'a structured report (JSON or CSV format)
    ... detail the exact location (latitude/longitude), bounding dimensions,
    and classification of each detected hazard.' Also writes report.geojson
    via the geolocation engine's own GeoJSON output method (src.geolocation.
    geojson_export) -- the same FeatureCollection GET /logs/{id}/map serves
    live, saved to disk so a log's geolocation results can be opened
    directly in a GIS tool without going through the API."""
    with db.get_connection(db_path) as conn:
        detections = db.list_detections(conn, log_id)

    geojson = write_geojson(out_dir / "report.geojson", detections, only_geolocated=True)
    logger.info("Wrote %s (%d geolocated feature(s) of %d total detection(s))",
                out_dir / "report.geojson", len(geojson["features"]), len(detections))

    report_rows = [{
        "detection_id": d["id"],
        "frame": d["frame_record_id"],
        "class": d["class_name"],
        "confidence_pct": d["confidence_score"],
        "confidence_label": d["confidence_label"],
        "bbox_px": [d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"]],
        "lat": d["lat"],
        "lon": d["lon"],
        "location_known": d["geo_method"] == "nav_fix",
    } for d in detections]

    with open(out_dir / "report.json", "w") as f:
        json.dump({"log_id": log_id, "n_detections": len(report_rows), "detections": report_rows}, f, indent=2)

    with open(out_dir / "report.csv", "w", newline="") as f:
        fieldnames = ["detection_id", "frame", "class", "confidence_pct", "confidence_label",
                      "bbox_px", "lat", "lon", "location_known"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

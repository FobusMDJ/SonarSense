"""FastAPI service -- the API a frontend attaches to.

Run with:
    uvicorn src.backend.main:app --reload --port 8000

Endpoints (all under this one app):
  POST   /logs/upload                          upload a single sonar log file, starts processing
  POST   /logs/upload_dir                      upload a whole folder of frames (+ optional nav
                                                 sidecar) as one multipart request, starts processing
  POST   /logs/upload_zip                      upload images + metadata CSV as one ZIP archive
  POST   /logs/geolocate_csv                    geolocate detections.csv (from an external/offline
                                                 detector) against nav.csv -- no image/YOLO/VAE stage
  POST   /logs/ingest_local                    process a folder that already exists on the SERVER's
                                                 own filesystem -- no upload at all (dev/local use)
  GET    /health                                service + XTF-reader status
  GET    /model/metadata                         released checkpoint + training metadata
  GET    /logs                                  list all processed/processing logs
  GET    /logs/{log_id}                         one log's status/summary
  GET    /logs/{log_id}/detections              detection records (the "records" area)
  GET    /logs/{log_id}/stats                   aggregate YOLO stats ("model output stats" area)
  GET    /logs/{log_id}/vae_stats               VAE anomaly-error stats ("VAE stats" area)
  GET    /logs/{log_id}/map                     GeoJSON of geolocated detections ("GPS map" area);
                                                 ?geometry_mode=auto|point_only|footprint_only
  GET    /logs/{log_id}/report.json             downloadable JSON report
  GET    /logs/{log_id}/report.csv              downloadable CSV report
  GET    /logs/{log_id}/report.geojson          downloadable GeoJSON (same content as /map)
  GET    /logs/{log_id}/report.sql              downloadable PostGIS SQL dump of this log's detections
  GET    /logs/{log_id}/frames/{frame}/vae/{f}  one of the 7 VAE output PNGs for one frame
  WS     /ws/logs/{log_id}                      live progress events while a log processes
"""

from __future__ import annotations

import queue
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from src.backend import class_taxonomy, db_backend as db
from src.backend.api_routes import router as frontend_api_router
from src.backend.archive_ingestion import extract_survey_zip
from src.backend.pipeline_runner import _write_reports, process_log
from src.geolocation.csv_engine import run_csv_pipeline_to_detections
from src.backend.model_release import model_metadata, validate_present_checkpoints
from src.backend.schemas import Detection, LocalIngestRequest, LogSummary, ModelOutputStats, UploadResponse, VaeStats
from src.geolocation.geojson_export import build_geojson, write_geojson
from src.geolocation.xtf_reader import xtf_reader_status
from src.utils.config import get_logger, load_config

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_cfg = load_config(str(PROJECT_ROOT / "config" / "backend.yaml"))

DB_PATH = db.configure(_cfg, project_root=PROJECT_ROOT)  # Path (sqlite) or DSN string (postgres) --
# see db_backend.py's module docstring. Everything below that already threads DB_PATH through
# db.init_db(DB_PATH) / db.get_connection(DB_PATH) / run_pipeline(..., db_path=DB_PATH, ...)
# keeps working unmodified regardless of which backend is active.
UPLOAD_DIR = PROJECT_ROOT / _cfg.get("upload_dir", "backend_uploads")
OUTPUT_DIR = PROJECT_ROOT / _cfg.get("output_dir", "backend_outputs")
YOLO_WEIGHTS = PROJECT_ROOT / _cfg.get("yolo_weights_path", "best.pt")
VAE_WEIGHTS = PROJECT_ROOT / _cfg.get("vae_weights_path", "src/vae/vae_epoch100.pth")
B2U_WEIGHTS = PROJECT_ROOT / _cfg.get("b2u_weights_path", "models/B2Ueph2.pth")
DEFAULT_YOLO_CONF = float(_cfg.get("default_yolo_conf", 0.25))
DEFAULT_LOW_CONF_THRESHOLD = float(_cfg.get("default_low_confidence_threshold", 40.0))
DEFAULT_PIXELS_TO_METERS = float(_cfg.get("default_pixels_to_meters", 0.15))
PORT_IS_LEFT = bool(_cfg.get("port_is_left", True))
# "none"/"none" matches what best.pt/the VAE were actually trained on -- see
# pipeline_runner.py's module docstring for why. "lee" and "clahe" are this
# project's other fully-implemented options; "blind2unblind" needs
# B2U_WEIGHTS to exist (checked in /health) and is a deliberate
# train/serve-mismatch tradeoff, not a drop-in upgrade -- see that docstring.
DEFAULT_DENOISE_METHOD = str(_cfg.get("default_denoise_method", "none"))
DEFAULT_CONTRAST_METHOD = str(_cfg.get("default_contrast_method", "none"))

VAE_OUTPUT_FILENAMES = {
    "01_original.png", "02_reconstruction.png", "03_anomaly_overlay.png",
    "04_difference_heatmap.png", "05_edge_contour_map.png",
    "06_difference_map_legend.png", "07_3d_anomaly_surface.png", "summary.json",
}

app = FastAPI(title="SonarSense Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Frontend-shaped /api/* routes (FobusMDJ/SonarSense React app) -- additive,
# alongside the /logs/* API above (see api_routes.py's module docstring).
app.include_router(frontend_api_router)

_progress_queues: dict[str, "queue.Queue[dict]"] = {}
MODEL_PATHS = {"detector": YOLO_WEIGHTS, "vae": VAE_WEIGHTS, "denoiser": B2U_WEIGHTS}


@app.on_event("startup")
def _startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db(DB_PATH)
    # Background inference runs in this web-service process. If the worker
    # restarts, those threads no longer exist; do not leave their rows looking
    # active forever in the frontend.
    with db.get_connection(DB_PATH) as conn:
        for log in db.list_logs(conn):
            if log["status"] in {"uploaded", "processing"}:
                db.update_log_status(
                    conn,
                    log["id"],
                    "error",
                    error_message="Stopped because the processing worker restarted.",
                    completed_at=_now_iso(),
                )
    # Missing files remain visible through /health; a present but corrupted
    # or substituted release file is unsafe and stops startup immediately.
    validate_present_checkpoints(MODEL_PATHS)
    if not YOLO_WEIGHTS.exists():
        logger.warning("YOLO weights not found at %s -- set yolo_weights_path in config/backend.yaml", YOLO_WEIGHTS)
    if not VAE_WEIGHTS.exists():
        logger.warning("VAE weights not found at %s -- set vae_weights_path in config/backend.yaml", VAE_WEIGHTS)
    if not B2U_WEIGHTS.exists():
        logger.info("B2U (Blind2Unblind) weights not found at %s -- denoise_method='blind2unblind' will fall back "
                     "to the Lee filter. Not fatal: 'none' (the default) and 'lee' don't need this file.", B2U_WEIGHTS)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_processing_params(denoise_method: str, contrast_method: str) -> None:
    """Shared by every ingestion entrypoint (/logs/upload, /logs/upload_dir,
    /logs/ingest_local) so the three don't drift out of sync on what counts
    as a valid choice."""
    if not YOLO_WEIGHTS.exists() or not VAE_WEIGHTS.exists():
        raise HTTPException(500, "Model weights not configured -- check config/backend.yaml")
    if denoise_method not in ("none", "lee", "blind2unblind"):
        raise HTTPException(422, f"denoise_method must be one of: none, lee, blind2unblind (got '{denoise_method}')")
    if contrast_method not in ("none", "clahe", "histeq"):
        raise HTTPException(422, f"contrast_method must be one of: none, clahe, histeq (got '{contrast_method}')")
    if denoise_method == "blind2unblind" and not B2U_WEIGHTS.exists():
        logger.warning("denoise_method=blind2unblind requested but %s doesn't exist -- will fall back to Lee filter "
                        "for this log (see denoising.py's fallback_on_missing_weights).", B2U_WEIGHTS)


def _start_log(source_path: Path, nav_sidecar_path: Optional[Path], filename: str, source_format: str,
                pixels_to_meters: float, yolo_conf: float, denoise_method: str, contrast_method: str) -> UploadResponse:
    """Shared tail end of every ingestion entrypoint: create the DB row,
    open the progress queue, kick off the background thread, return the
    standard UploadResponse."""
    log_id = str(uuid.uuid4())
    with db.get_connection(DB_PATH) as conn:
        db.create_log(conn, log_id, filename, source_format, _now_iso(), pixels_to_meters,
                       denoise_method=denoise_method, contrast_method=contrast_method,
                       yolo_confidence_threshold=yolo_conf)

    _progress_queues[log_id] = queue.Queue()
    _run_in_background(log_id, source_path, nav_sidecar_path, pixels_to_meters, yolo_conf,
                        denoise_method, contrast_method)

    return UploadResponse(log_id=log_id, status="processing",
                           message="Processing started. Connect to /ws/logs/{log_id} for progress.")


def _progress_cb_for(log_id: str):
    def _cb(event: dict) -> None:
        q = _progress_queues.get(log_id)
        if q is not None:
            q.put(event)
    return _cb


def _run_in_background(log_id: str, source_path: Path, nav_sidecar_path: Optional[Path],
                        pixels_to_meters: float, yolo_conf: float,
                        denoise_method: str, contrast_method: str) -> None:
    def _target():
        try:
            process_log(
                log_id=log_id, source_path=source_path, yolo_weights=YOLO_WEIGHTS, vae_weights=VAE_WEIGHTS,
                db_path=DB_PATH, output_dir=OUTPUT_DIR, pixels_to_meters=pixels_to_meters,
                nav_sidecar_path=nav_sidecar_path, yolo_conf=yolo_conf, port_is_left=PORT_IS_LEFT,
                denoise_method=denoise_method, denoise_weights_path=(B2U_WEIGHTS if denoise_method == "blind2unblind" else None),
                contrast_method=contrast_method,
                progress_cb=_progress_cb_for(log_id),
            )
        finally:
            q = _progress_queues.get(log_id)
            if q is not None:
                q.put({"log_id": log_id, "stage": "closed", "message": "processing thread finished"})

    threading.Thread(target=_target, daemon=True).start()


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    checkpoint_states = validate_present_checkpoints(MODEL_PATHS)
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": {
            "yolo_best_pt": checkpoint_states["detector"],
            "vae": checkpoint_states["vae"],
            "b2u_blind2unblind": {**checkpoint_states["denoiser"],
                                   "note": "optional -- only used when a log is uploaded with denoise_method=blind2unblind"},
            "lee_filter": {"found": True, "note": "classical, no checkpoint needed -- always available"},
        },
        "defaults": {"denoise_method": DEFAULT_DENOISE_METHOD, "contrast_method": DEFAULT_CONTRAST_METHOD},
        "xtf": xtf_reader_status(),
    }


@app.get("/model/metadata")
def get_model_metadata() -> dict:
    """Immutable training facts plus local checkpoint/device status."""
    return model_metadata(MODEL_PATHS)


# --------------------------------------------------------------------------
# upload / logs
# --------------------------------------------------------------------------

@app.post("/logs/upload", response_model=UploadResponse)
async def upload_log(
    file: UploadFile = File(..., description="The sonar log file: .xtf, or an image (PNG/JPG/TIFF/...)."),
    nav_sidecar: Optional[UploadFile] = File(
        None, description="Optional nav CSV (frame_index,lat,lon,heading_deg[,altitude_m,timestamp]) "
                           "for the image+sidecar path. Ignored for .xtf uploads (nav comes from the log itself)."),
    pixels_to_meters: float = DEFAULT_PIXELS_TO_METERS,
    yolo_conf: float = DEFAULT_YOLO_CONF,
    denoise_method: str = DEFAULT_DENOISE_METHOD,
    contrast_method: str = DEFAULT_CONTRAST_METHOD,
) -> UploadResponse:
    _validate_processing_params(denoise_method, contrast_method)

    log_id_dir = uuid.uuid4().hex  # just a scratch dirname, not the log_id used in the DB
    log_dir = UPLOAD_DIR / log_id_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    source_path = log_dir / file.filename
    with open(source_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    nav_sidecar_path = None
    if nav_sidecar is not None:
        nav_sidecar_path = log_dir / nav_sidecar.filename
        with open(nav_sidecar_path, "wb") as f:
            shutil.copyfileobj(nav_sidecar.file, f)

    if source_path.suffix.lower() == ".xtf":
        source_format = "xtf"
    elif nav_sidecar_path is not None:
        source_format = "image+sidecar"
    else:
        source_format = "image"

    return _start_log(source_path, nav_sidecar_path, file.filename, source_format,
                       pixels_to_meters, yolo_conf, denoise_method, contrast_method)


@app.post("/logs/upload_dir", response_model=UploadResponse)
async def upload_log_directory(
    files: list[UploadFile] = File(
        ..., description="Every image file in the sonar log folder (PNG/JPG/TIFF/...). Select the whole "
                          "folder's contents -- selection order doesn't matter, files are re-sorted by "
                          "filename server-side to match the nav sidecar's row order (same sort "
                          "src.utils.io_utils.list_image_files uses)."),
    nav_sidecar: Optional[UploadFile] = File(
        None, description="Optional nav CSV (frame_index,lat,lon,heading_deg[,altitude_m,timestamp]), "
                           "one row per frame in filename-sorted order."),
    pixels_to_meters: float = DEFAULT_PIXELS_TO_METERS,
    yolo_conf: float = DEFAULT_YOLO_CONF,
    denoise_method: str = DEFAULT_DENOISE_METHOD,
    contrast_method: str = DEFAULT_CONTRAST_METHOD,
) -> UploadResponse:
    """Upload an entire folder of frames (e.g. a browser <input webkitdirectory>
    selection, or any client that can attach many files to one multipart
    request) in a single call, instead of the one-file-at-a-time /logs/upload.
    Every file lands in a fresh per-log directory and is then ingested via
    ingest_directory_with_nav -- the same directory-ingestion code path
    process_log() already used for local/offline batches, just reachable
    over HTTP now.

    Practical note: this pushes every byte through the HTTP multipart body,
    which is fine for tens-to-hundreds of frames but gets slow/heavy for
    thousands (e.g. this project's own data/processed/Sonar_Log/Frames has
    several thousand tiles). For a folder that already lives on the SAME
    machine as this server, POST /logs/ingest_local skips the upload
    entirely and is much faster for exactly that case.
    """
    _validate_processing_params(denoise_method, contrast_method)
    if not files:
        raise HTTPException(422, "No files provided -- select at least one image file.")

    log_id_dir = uuid.uuid4().hex
    log_dir = UPLOAD_DIR / log_id_dir
    frames_dir = log_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for f in files:
        # Strip any subdirectory components from webkitdirectory-style
        # filenames (e.g. "Frames/subpipe_hf__...png") -- ingestion expects
        # a FLAT directory (ingest_directory_with_nav uses Path.iterdir(),
        # not a recursive walk), and flattening also closes off any
        # path-traversal attempt via a crafted filename.
        dest = frames_dir / Path(f.filename or f"file_{n_saved}").name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        n_saved += 1

    nav_sidecar_path = None
    if nav_sidecar is not None:
        nav_sidecar_path = log_dir / Path(nav_sidecar.filename).name
        with open(nav_sidecar_path, "wb") as out:
            shutil.copyfileobj(nav_sidecar.file, out)

    source_format = "directory+sidecar" if nav_sidecar_path is not None else "directory"
    display_name = f"{n_saved} file(s)"

    return _start_log(frames_dir, nav_sidecar_path, display_name, source_format,
                       pixels_to_meters, yolo_conf, denoise_method, contrast_method)


@app.post("/logs/upload_zip", response_model=UploadResponse)
async def upload_log_zip(
    archive: UploadFile = File(..., description="ZIP containing sonar images and one navigation metadata CSV."),
    pixels_to_meters: float = DEFAULT_PIXELS_TO_METERS,
    yolo_conf: float = DEFAULT_YOLO_CONF,
    denoise_method: str = DEFAULT_DENOISE_METHOD,
    contrast_method: str = DEFAULT_CONTRAST_METHOD,
) -> UploadResponse:
    """Safely unpack and process a complete exported survey archive."""
    _validate_processing_params(denoise_method, contrast_method)
    if Path(archive.filename or "").suffix.lower() != ".zip":
        raise HTTPException(422, "Complete survey upload must be a .zip file.")

    log_dir = UPLOAD_DIR / uuid.uuid4().hex
    log_dir.mkdir(parents=True, exist_ok=True)
    archive_path = log_dir / "survey.zip"
    try:
        with archive_path.open("wb") as output:
            shutil.copyfileobj(archive.file, output)
        frames_dir, metadata_path, _image_count = extract_survey_zip(archive_path, log_dir / "extracted")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        shutil.rmtree(log_dir, ignore_errors=True)
        raise HTTPException(422, str(exc)) from exc

    return _start_log(
        frames_dir, metadata_path, archive.filename or "survey.zip", "zip+metadata",
        pixels_to_meters, yolo_conf, denoise_method, contrast_method,
    )


@app.post("/logs/geolocate_csv", response_model=UploadResponse)
async def geolocate_csv(
    detections_csv: UploadFile = File(..., description="Detections CSV: ping_id,pixel_x,bbox_w,bbox_h,class,confidence "
                                                          "(column-name aliases width/w, height/h also accepted)."),
    nav_csv: UploadFile = File(..., description="Nav CSV: ping_id,lat,lon,heading,altitude_m,depth_m,timestamp "
                                                  "(ship_lat/ship_lon/heading_deg/altitude/depth/timestamp_utc "
                                                  "aliases also accepted)."),
    image_width: int = 2000,
    range_per_pixel: float = DEFAULT_PIXELS_TO_METERS,
) -> UploadResponse:
    """Second ingestion path alongside /logs/upload et al.: geolocates
    detections that were already produced by an EXTERNAL/offline detector
    (a Roboflow-hosted model, a prior batch run, ...) rather than running
    YOLO/VAE here. See src/geolocation/csv_engine.py's module docstring for
    the full math and its one intentional fix (object depth = nav depth +
    altitude) versus the standalone script this was adapted from.

    Synchronous (no background thread/WebSocket progress) -- this is pure
    arithmetic over two CSVs, not an ML pipeline, so it's fast enough to
    finish within the request. Results are written into the SAME
    `detections` table as image-pipeline logs (source_format=
    "csv_detections"), so they show up in GET /logs/{id}/map,
    /logs/{id}/detections, and the downloadable reports identically to
    pipeline-produced detections -- one unified store regardless of source.
    A ping_id in detections_csv with no matching row in nav_csv is skipped
    (not stored as a placeholder), same join-and-skip behavior as the
    original standalone script; the response's `message` reports how many
    were skipped."""
    log_id = str(uuid.uuid4())
    log_dir = UPLOAD_DIR / log_id
    log_dir.mkdir(parents=True, exist_ok=True)

    detections_path = log_dir / (detections_csv.filename or "detections.csv")
    with open(detections_path, "wb") as f:
        shutil.copyfileobj(detections_csv.file, f)
    nav_path = log_dir / (nav_csv.filename or "nav.csv")
    with open(nav_path, "wb") as f:
        shutil.copyfileobj(nav_csv.file, f)

    created_at = _now_iso()
    with db.get_connection(DB_PATH) as conn:
        db.create_log(conn, log_id, detections_csv.filename or "detections.csv", "csv_detections", created_at,
                       pixels_to_meters=range_per_pixel)

    try:
        rows, n_skipped = run_csv_pipeline_to_detections(
            detections_path, nav_path, image_width, range_per_pixel, log_id, created_at,
        )
        with db.get_connection(DB_PATH) as conn:
            for row in rows:
                db.insert_detection(conn, row)
            db.update_log_counts(conn, log_id, len(rows), len(rows))
            db.update_log_status(conn, log_id, "done", completed_at=_now_iso())

        out_dir = OUTPUT_DIR / log_id
        _write_reports(DB_PATH, log_id, out_dir)
    except Exception as exc:
        with db.get_connection(DB_PATH) as conn:
            db.update_log_status(conn, log_id, "error", error_message=str(exc), completed_at=_now_iso())
        raise HTTPException(422, f"CSV geolocation failed: {exc}") from exc

    message = f"Geolocated {len(rows)} detection(s)."
    if n_skipped:
        message += f" Skipped {n_skipped} with no matching nav row."
    return UploadResponse(log_id=log_id, status="done", message=message)


@app.post("/logs/ingest_local", response_model=UploadResponse)
def ingest_local_directory(req: LocalIngestRequest) -> UploadResponse:
    """Process a folder that ALREADY exists on the SERVER's own filesystem --
    no upload at all. Built for this project's own dev workflow: testing the
    geolocation engine against data/processed/Sonar_Log/Frames (thousands of
    tiles) without pushing them through HTTP multipart first, since in local
    development the backend and that data live on the same machine.

    source_dir / nav_sidecar_path may be absolute, or relative to the
    project root (the directory config/backend.yaml lives under).

    DEV-ONLY, NO AUTH: this endpoint reads any path the server process has
    permission to read. That matches this API's existing no-auth,
    CORS-wide-open posture (see claude/backend-pipeline-plan.md's "Known v1
    limitations") -- appropriate for local/dev use, NOT something to expose
    on a network-reachable deployment without adding auth first.
    """
    denoise_method = req.denoise_method or DEFAULT_DENOISE_METHOD
    contrast_method = req.contrast_method or DEFAULT_CONTRAST_METHOD
    pixels_to_meters = req.pixels_to_meters if req.pixels_to_meters is not None else DEFAULT_PIXELS_TO_METERS
    yolo_conf = req.yolo_conf if req.yolo_conf is not None else DEFAULT_YOLO_CONF
    _validate_processing_params(denoise_method, contrast_method)

    source_path = Path(req.source_dir)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    if not source_path.exists():
        raise HTTPException(404, f"source_dir does not exist on the server: {source_path}")
    if not source_path.is_dir():
        raise HTTPException(422, f"source_dir must be a directory (got a file): {source_path}")

    nav_sidecar_path = None
    if req.nav_sidecar_path:
        nav_sidecar_path = Path(req.nav_sidecar_path)
        if not nav_sidecar_path.is_absolute():
            nav_sidecar_path = PROJECT_ROOT / nav_sidecar_path
        if not nav_sidecar_path.exists():
            raise HTTPException(404, f"nav_sidecar_path does not exist on the server: {nav_sidecar_path}")

    source_format = "local_directory+sidecar" if nav_sidecar_path is not None else "local_directory"

    return _start_log(source_path, nav_sidecar_path, source_path.name, source_format,
                       pixels_to_meters, yolo_conf, denoise_method, contrast_method)


@app.get("/logs", response_model=list[LogSummary])
def list_logs() -> list[dict]:
    with db.get_connection(DB_PATH) as conn:
        return db.list_logs(conn)


@app.get("/logs/{log_id}", response_model=LogSummary)
def get_log(log_id: str) -> dict:
    with db.get_connection(DB_PATH) as conn:
        row = db.get_log(conn, log_id)
    if row is None:
        raise HTTPException(404, f"Log {log_id} not found")
    return row


# --------------------------------------------------------------------------
# detections / records area
# --------------------------------------------------------------------------

@app.get("/logs/{log_id}/detections", response_model=list[Detection])
def get_detections(log_id: str, min_confidence: Optional[float] = None) -> list[dict]:
    log = _require_log(log_id)
    pixels_to_meters = log.get("pixels_to_meters")
    with db.get_connection(DB_PATH) as conn:
        rows = db.list_detections(conn, log_id, min_confidence=min_confidence)
    for d in rows:
        # resolved_dimensions_m prefers a detection's OWN stored length_m/width_m (set at
        # insert time by the CSV geolocation engine's real slant-range/ground-range math --
        # see csv_engine.geolocate_csv_detection) over the generic bbox * pixels_to_meters
        # guess, which only applies to image/YOLO-pipeline detections that never had real
        # geometry computed for them. height is always class_taxonomy.ESTIMATED_HEIGHT_M's
        # placeholder either way -- neither ingestion path measures a height axis.
        # dimensions_estimated is true only when length/width themselves are unknown.
        dims = class_taxonomy.resolved_dimensions_m(d, pixels_to_meters)
        d["length_m"] = dims["length"]
        d["width_m"] = dims["width"]
        d["height_m"] = dims["height"]
        d["dimensions_estimated"] = dims["length"] is None or dims["width"] is None
    return rows


def _require_log(log_id: str) -> dict:
    with db.get_connection(DB_PATH) as conn:
        row = db.get_log(conn, log_id)
    if row is None:
        raise HTTPException(404, f"Log {log_id} not found")
    return row


# --------------------------------------------------------------------------
# model output stats area
# --------------------------------------------------------------------------

@app.get("/logs/{log_id}/stats", response_model=ModelOutputStats)
def get_model_stats(log_id: str) -> ModelOutputStats:
    log = _require_log(log_id)
    with db.get_connection(DB_PATH) as conn:
        detections = db.list_detections(conn, log_id)

    counts: dict[str, int] = {}
    conf_sums: dict[str, float] = {}
    n_low = 0
    for d in detections:
        cls = d["class_name"]
        counts[cls] = counts.get(cls, 0) + 1
        conf_sums[cls] = conf_sums.get(cls, 0.0) + d["confidence_score"]
        if d["confidence_score"] < DEFAULT_LOW_CONF_THRESHOLD:
            n_low += 1
    mean_conf = {cls: round(conf_sums[cls] / counts[cls], 1) for cls in counts}

    return ModelOutputStats(
        log_id=log_id, n_detections=len(detections), counts_by_class=counts,
        raw_counts_by_class=counts,
        mean_confidence_by_class=mean_conf, n_low_confidence=n_low,
        low_confidence_threshold=DEFAULT_LOW_CONF_THRESHOLD,
        confidence_threshold=log.get("yolo_confidence_threshold"),
        mean_inference_ms=(round(log["detector_inference_ms"] / log["detector_frames"], 2)
                           if log.get("detector_inference_ms") is not None and log.get("detector_frames") else None),
        fps=(round(1000.0 / (log["detector_inference_ms"] / log["detector_frames"]), 2)
             if log.get("detector_inference_ms") and log.get("detector_frames") else None),
    )


# --------------------------------------------------------------------------
# VAE stats area
# --------------------------------------------------------------------------

@app.get("/logs/{log_id}/vae_stats", response_model=VaeStats)
def get_vae_stats(log_id: str) -> VaeStats:
    _require_log(log_id)
    with db.get_connection(DB_PATH) as conn:
        stored_frames = db.list_frame_analyses(conn, log_id)

    frames = [{
        "frame_record_id": frame["frame_record_id"],
        "whole_image_error": frame["whole_image_error"],
        "percentile": frame["percentile"],
    } for frame in stored_frames]
    if not frames:
        return VaeStats(log_id=log_id, n_frames_analyzed=0, mean_whole_image_error=0.0,
                         min_whole_image_error=0.0, max_whole_image_error=0.0, most_anomalous_frames=[])

    errors = [f["whole_image_error"] for f in frames if f["whole_image_error"] is not None]
    most_anomalous = sorted(frames, key=lambda f: (f["percentile"] if f["percentile"] is not None else 1.0))[:10]

    return VaeStats(
        log_id=log_id, n_frames_analyzed=len(frames),
        mean_whole_image_error=round(sum(errors) / len(errors), 6) if errors else 0.0,
        min_whole_image_error=round(min(errors), 6) if errors else 0.0,
        max_whole_image_error=round(max(errors), 6) if errors else 0.0,
        most_anomalous_frames=most_anomalous,
    )


@app.get("/logs/{log_id}/frames/{frame_record_id}/vae/surface")
def get_vae_surface_data(log_id: str, frame_record_id: str) -> dict:
    """Compact normalized reconstruction-error grid for the interactive 3D viewer."""
    _require_log(log_id)
    frame_dir = OUTPUT_DIR / log_id / "vae" / frame_record_id
    original = cv2.imread(str(frame_dir / "01_original.png"), cv2.IMREAD_GRAYSCALE)
    reconstruction = cv2.imread(str(frame_dir / "02_reconstruction.png"), cv2.IMREAD_GRAYSCALE)
    if original is None or reconstruction is None:
        raise HTTPException(404, f"VAE source panels not found for frame {frame_record_id}")
    size = 56
    original = cv2.resize(original, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    reconstruction = cv2.resize(reconstruction, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    difference = np.abs(original - reconstruction) / 255.0
    scale = max(float(np.percentile(difference, 98)), 0.001)
    normalized = np.clip(difference / scale, 0.0, 1.0)
    return {"size": size, "values": np.round(normalized, 5).reshape(-1).tolist()}


@app.get("/logs/{log_id}/frames/{frame_record_id}/vae/{filename}")
def get_vae_output_file(log_id: str, frame_record_id: str, filename: str) -> FileResponse:
    if filename not in VAE_OUTPUT_FILENAMES:
        raise HTTPException(400, f"Unknown VAE output file: {filename}")
    path = OUTPUT_DIR / log_id / "vae" / frame_record_id / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} not found for frame {frame_record_id}")
    return FileResponse(str(path))


# --------------------------------------------------------------------------
# GPS map area
# --------------------------------------------------------------------------

@app.get("/logs/{log_id}/map")
def get_map_geojson(log_id: str, geometry_mode: str = "auto") -> dict:
    """GeoJSON FeatureCollection -- only detections with a real nav-derived
    location (geo_method == 'nav_fix'). Detections with no nav data are
    excluded here (not plotted as fake points); use /detections for those.
    Built by the geolocation engine's own GeoJSON output method (src.
    geolocation.geojson_export.build_geojson) -- the same function
    pipeline_runner.py uses to write report.geojson to disk, so this live
    endpoint and that downloadable file are always identical in shape.

    `geometry_mode` (query param, default "auto") is passed straight
    through to build_geojson -- see geojson_export.py's GEOMETRY_MODES for
    "auto" | "point_only" | "footprint_only". An unrecognized value is a
    client error (400), not a silent fallback to "auto"."""
    _require_log(log_id)
    with db.get_connection(DB_PATH) as conn:
        detections = db.list_detections(conn, log_id)
    try:
        return build_geojson(detections, only_geolocated=True, geometry_mode=geometry_mode)
    except ValueError as e:
        raise HTTPException(400, str(e))


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------

@app.get("/logs/{log_id}/report.json")
def get_report_json(log_id: str) -> FileResponse:
    _require_log(log_id)
    path = OUTPUT_DIR / log_id / "report.json"
    if not path.exists():
        raise HTTPException(404, "Report not ready yet -- log may still be processing.")
    return FileResponse(str(path), media_type="application/json", filename=f"{log_id}_report.json")


@app.get("/logs/{log_id}/report.pdf")
def get_report_pdf(log_id: str) -> Response:
    from src.backend.reports import build_report_pdf
    log = _require_log(log_id)
    with db.get_connection(DB_PATH) as conn:
        detections = db.list_detections(conn, log_id)
    pdf_bytes = build_report_pdf(log, detections)
    return Response(pdf_bytes, media_type="application/pdf",
                     headers={"Content-Disposition": f'attachment; filename="{log_id}_report.pdf"'})


@app.get("/logs/{log_id}/map.png")
def get_map_png(log_id: str) -> Response:
    from src.backend.reports import build_map_png
    log = _require_log(log_id)
    with db.get_connection(DB_PATH) as conn:
        detections = db.list_detections(conn, log_id)
    png_bytes = build_map_png(log, detections)
    return Response(png_bytes, media_type="image/png",
                     headers={"Content-Disposition": f'attachment; filename="{log_id}_map.png"'})


@app.get("/logs/{log_id}/report.csv")
def get_report_csv(log_id: str) -> FileResponse:
    _require_log(log_id)
    path = OUTPUT_DIR / log_id / "report.csv"
    if not path.exists():
        raise HTTPException(404, "Report not ready yet -- log may still be processing.")
    return FileResponse(str(path), media_type="text/csv", filename=f"{log_id}_report.csv")


@app.get("/logs/{log_id}/report.geojson")
def get_report_geojson(log_id: str) -> FileResponse:
    """Same content as GET /map, served as a downloadable .geojson file
    (written to disk by pipeline_runner.py._write_reports at the end of
    processing) -- for opening a log's results directly in QGIS/geojson.io/
    etc. rather than going through the live API."""
    _require_log(log_id)
    path = OUTPUT_DIR / log_id / "report.geojson"
    if not path.exists():
        raise HTTPException(404, "Report not ready yet -- log may still be processing.")
    return FileResponse(str(path), media_type="application/geo+json", filename=f"{log_id}_report.geojson")


@app.get("/logs/{log_id}/report.sql")
def get_report_sql(log_id: str) -> FileResponse:
    """Standalone PostGIS-loadable dump of this log's detections (src.
    geolocation.postgis_export), written to disk by _write_reports at the
    end of processing -- for the "GeoJSON -> PostGIS" step in the reference
    pitch deck as an actual downloadable artifact, without needing a live
    Postgres connection just to inspect it (works regardless of whether
    this deployment's own db_backend is sqlite or postgres)."""
    _require_log(log_id)
    path = OUTPUT_DIR / log_id / "report.sql"
    if not path.exists():
        raise HTTPException(404, "Report not ready yet -- log may still be processing.")
    return FileResponse(str(path), media_type="application/sql", filename=f"{log_id}_report.sql")


# --------------------------------------------------------------------------
# live progress
# --------------------------------------------------------------------------

@app.websocket("/ws/logs/{log_id}")
async def ws_progress(websocket: WebSocket, log_id: str) -> None:
    await websocket.accept()
    q = _progress_queues.get(log_id)
    if q is None:
        await websocket.send_json({"log_id": log_id, "stage": "error", "message": "No active processing for this log_id (already finished, or never started)."})
        await websocket.close()
        return
    try:
        while True:
            event = await _await_queue_get(q)
            await websocket.send_json(event)
            if event.get("stage") in ("done", "error", "closed"):
                break
    except WebSocketDisconnect:
        logger.info("Client disconnected from progress stream for log %s", log_id)
    finally:
        await websocket.close()


async def _await_queue_get(q: "queue.Queue[dict]") -> dict:
    import asyncio
    return await asyncio.to_thread(q.get)

"""FastAPI service -- the API a frontend attaches to.

Run with:
    uvicorn src.backend.main:app --reload --port 8000

Endpoints (all under this one app):
  POST   /logs/upload                          upload a single sonar log file, starts processing
  POST   /logs/upload_dir                      upload a whole folder of frames (+ optional nav
                                                 sidecar) as one multipart request, starts processing
  POST   /logs/ingest_local                    process a folder that already exists on the SERVER's
                                                 own filesystem -- no upload at all (dev/local use)
  GET    /health                                service + XTF-reader status
  GET    /logs                                  list all processed/processing logs
  GET    /logs/{log_id}                         one log's status/summary
  GET    /logs/{log_id}/detections              detection records (the "records" area)
  GET    /logs/{log_id}/stats                   aggregate YOLO stats ("model output stats" area)
  GET    /logs/{log_id}/vae_stats               VAE anomaly-error stats ("VAE stats" area)
  GET    /logs/{log_id}/map                     GeoJSON of geolocated detections ("GPS map" area)
  GET    /logs/{log_id}/report.json             downloadable JSON report
  GET    /logs/{log_id}/report.csv              downloadable CSV report
  GET    /logs/{log_id}/report.geojson          downloadable GeoJSON (same content as /map)
  GET    /logs/{log_id}/frames/{frame}/vae/{f}  one of the 7 VAE output PNGs for one frame
  WS     /ws/logs/{log_id}                      live progress events while a log processes
"""

from __future__ import annotations

import queue
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.backend import db
from src.backend.pipeline_runner import process_log
from src.backend.schemas import Detection, LocalIngestRequest, LogSummary, ModelOutputStats, UploadResponse, VaeStats
from src.geolocation.geojson_export import build_geojson, write_geojson
from src.geolocation.xtf_reader import xtf_reader_status
from src.utils.config import get_logger, load_config

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_cfg = load_config(str(PROJECT_ROOT / "config" / "backend.yaml"))

DB_PATH = PROJECT_ROOT / _cfg.get("db_path", "sonarsense.db")
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

_progress_queues: dict[str, "queue.Queue[dict]"] = {}


@app.on_event("startup")
def _startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db(DB_PATH)
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
                       denoise_method=denoise_method, contrast_method=contrast_method)

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
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": {
            "yolo_best_pt": {"path": str(YOLO_WEIGHTS), "found": YOLO_WEIGHTS.exists()},
            "vae": {"path": str(VAE_WEIGHTS), "found": VAE_WEIGHTS.exists()},
            "b2u_blind2unblind": {"path": str(B2U_WEIGHTS), "found": B2U_WEIGHTS.exists(),
                                   "note": "optional -- only used when a log is uploaded with denoise_method=blind2unblind"},
            "lee_filter": {"found": True, "note": "classical, no checkpoint needed -- always available"},
        },
        "defaults": {"denoise_method": DEFAULT_DENOISE_METHOD, "contrast_method": DEFAULT_CONTRAST_METHOD},
        "xtf": xtf_reader_status(),
    }


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
    _require_log(log_id)
    with db.get_connection(DB_PATH) as conn:
        return db.list_detections(conn, log_id, min_confidence=min_confidence)


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
    _require_log(log_id)
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
        mean_confidence_by_class=mean_conf, n_low_confidence=n_low,
        low_confidence_threshold=DEFAULT_LOW_CONF_THRESHOLD,
    )


# --------------------------------------------------------------------------
# VAE stats area
# --------------------------------------------------------------------------

@app.get("/logs/{log_id}/vae_stats", response_model=VaeStats)
def get_vae_stats(log_id: str) -> VaeStats:
    _require_log(log_id)
    with db.get_connection(DB_PATH) as conn:
        detections = db.list_detections(conn, log_id)

    # one whole_image_error per frame -- dedupe by frame_record_id
    by_frame: dict[str, dict] = {}
    for d in detections:
        by_frame.setdefault(d["frame_record_id"], {
            "frame_record_id": d["frame_record_id"],
            "whole_image_error": d["vae_whole_image_error"],
            "percentile": d["vae_whole_image_percentile"],
        })

    frames = list(by_frame.values())
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
def get_map_geojson(log_id: str) -> dict:
    """GeoJSON FeatureCollection -- only detections with a real nav-derived
    location (geo_method == 'nav_fix'). Detections with no nav data are
    excluded here (not plotted as fake points); use /detections for those.
    Built by the geolocation engine's own GeoJSON output method (src.
    geolocation.geojson_export.build_geojson) -- the same function
    pipeline_runner.py uses to write report.geojson to disk, so this live
    endpoint and that downloadable file are always identical in shape."""
    _require_log(log_id)
    with db.get_connection(DB_PATH) as conn:
        detections = db.list_detections(conn, log_id)
    return build_geojson(detections, only_geolocated=True)


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

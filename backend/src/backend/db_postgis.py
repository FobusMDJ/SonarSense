"""PostGIS persistence backend -- a drop-in alternative to db.py (SQLite)
with the exact same function signatures, so src/backend/db_backend.py can
switch between them with zero changes anywhere else in the codebase.

WHY THIS EXISTS: the reference pitch deck's step 9 ("GEOSPATIAL DATABASE")
shows GeoJSON -> PostGIS explicitly. SQLite remains the zero-setup default
(db_backend: sqlite in config/backend.yaml) for local dev/demo use -- this
module is for deployments that want a real spatial database: geometry-aware
queries (bounding-box map queries, "detections within N meters of X",
nearest-neighbor) that a plain lat/lon REAL column can't do efficiently.

Every write function takes an already-open connection (via get_connection()
as a context manager), exactly like db.py, so a request handler can wrap
several writes in one transaction and swap backends without touching call
sites.

SCHEMA PARITY WITH db.py: the `logs` and `detections` tables here have the
IDENTICAL column set to db.py's SCHEMA (including depth_m from day one --
no separate migration story needed since this module is new, unlike
db.py's _DETECTIONS_MIGRATIONS which exists only to patch pre-existing
SQLite files). A `geom geometry(Point, 4326)` column is ADDED alongside
the plain `lat`/`lon` columns (not instead of them) so existing code that
reads `lat`/`lon` off a detection dict keeps working unmodified; `geom` is
there for spatial queries/indexes, populated automatically from lat/lon
whenever they're not the geolocation placeholder (0, 0) sentinel.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import psycopg2
import psycopg2.extras

from src.utils.config import get_logger

logger = get_logger(__name__)

DEFAULT_DSN = "postgresql://sonarsense:sonarsense@localhost:5432/sonarsense"

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS logs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source_format TEXT NOT NULL,
    status TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    completed_at TEXT,
    n_frames INTEGER NOT NULL DEFAULT 0,
    n_detections INTEGER NOT NULL DEFAULT 0,
    pixels_to_meters DOUBLE PRECISION,
    denoise_method TEXT,
    contrast_method TEXT,
    yolo_confidence_threshold DOUBLE PRECISION,
    detector_inference_ms DOUBLE PRECISION,
    detector_frames INTEGER NOT NULL DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    log_id TEXT NOT NULL REFERENCES logs(id),
    frame_index INTEGER NOT NULL,
    frame_record_id TEXT NOT NULL,
    frame_image_path TEXT,
    class_name TEXT NOT NULL,
    yolo_conf DOUBLE PRECISION NOT NULL,
    bbox_x1 DOUBLE PRECISION, bbox_y1 DOUBLE PRECISION,
    bbox_x2 DOUBLE PRECISION, bbox_y2 DOUBLE PRECISION,
    confidence_score DOUBLE PRECISION NOT NULL,
    confidence_label TEXT NOT NULL,
    confidence_breakdown JSONB,
    vae_box_error DOUBLE PRECISION,
    vae_whole_image_error DOUBLE PRECISION,
    vae_whole_image_percentile DOUBLE PRECISION,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    geo_method TEXT,
    depth_m DOUBLE PRECISION,
    vae_panel_dir TEXT,
    created_at TEXT NOT NULL,
    geom geometry(Point, 4326)
);

CREATE TABLE IF NOT EXISTS frame_analyses (
    log_id TEXT NOT NULL REFERENCES logs(id),
    frame_index INTEGER NOT NULL,
    frame_record_id TEXT NOT NULL,
    whole_image_error DOUBLE PRECISION,
    percentile DOUBLE PRECISION,
    vae_panel_dir TEXT NOT NULL,
    PRIMARY KEY (log_id, frame_record_id)
);

CREATE INDEX IF NOT EXISTS idx_detections_log_id ON detections(log_id);
CREATE INDEX IF NOT EXISTS idx_frame_analyses_log_id ON frame_analyses(log_id);
CREATE INDEX IF NOT EXISTS idx_detections_geom ON detections USING GIST(geom);
ALTER TABLE logs ADD COLUMN IF NOT EXISTS yolo_confidence_threshold DOUBLE PRECISION;
ALTER TABLE logs ADD COLUMN IF NOT EXISTS detector_inference_ms DOUBLE PRECISION;
ALTER TABLE logs ADD COLUMN IF NOT EXISTS detector_frames INTEGER NOT NULL DEFAULT 0;
"""

# Everything in SCHEMA after the `CREATE EXTENSION` line -- used as a
# fallback when the connecting role can't run CREATE EXTENSION itself (see
# init_db below).
_SCHEMA_TABLES_ONLY = SCHEMA.split("\n\n", 1)[1]


def init_db(dsn: str = DEFAULT_DSN) -> None:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
        conn.commit()
        logger.info("Initialized PostGIS schema at %s", _redact_dsn(dsn))
    except psycopg2.errors.InsufficientPrivilege:
        # CREATE EXTENSION needs superuser (or a role granted CREATE on the
        # database) the first time it runs on a fresh database -- if the DBA
        # already ran `CREATE EXTENSION postgis;` once, everything after it
        # in the script still succeeds; only re-raise if the *tables* also
        # failed to materialize.
        conn.rollback()
        logger.warning(
            "Could not run CREATE EXTENSION postgis (insufficient privilege) -- "
            "assuming a DBA already enabled it on this database. Retrying table creation only."
        )
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_TABLES_ONLY)
        conn.commit()
    finally:
        conn.close()


def _redact_dsn(dsn: str) -> str:
    if "@" in dsn and "://" in dsn:
        scheme, rest = dsn.split("://", 1)
        creds, host = rest.split("@", 1)
        return f"{scheme}://***@{host}"
    return dsn


@contextmanager
def get_connection(dsn: str = DEFAULT_DSN) -> Iterator["psycopg2.extensions.connection"]:
    conn = psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _geom_expr(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    """Returns the ST_SetSRID/ST_MakePoint SQL fragment, or None if lat/lon
    are missing/the (0, 0) placeholder sentinel (see georeference.py's
    GeoResult.method='placeholder') -- a placeholder detection should not
    get a real-looking geometry point plotted at Null Island."""
    if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
        return None
    return "ST_SetSRID(ST_MakePoint(%s, %s), 4326)"


# --------------------------------------------------------------------------
# logs
# --------------------------------------------------------------------------

def create_log(conn, log_id: str, filename: str, source_format: str,
                uploaded_at: str, pixels_to_meters: Optional[float] = None,
                denoise_method: Optional[str] = None, contrast_method: Optional[str] = None,
                yolo_confidence_threshold: Optional[float] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO logs (id, filename, source_format, status, uploaded_at, pixels_to_meters, "
            "denoise_method, contrast_method, yolo_confidence_threshold) "
            "VALUES (%s, %s, %s, 'uploaded', %s, %s, %s, %s, %s)",
            (log_id, filename, source_format, uploaded_at, pixels_to_meters, denoise_method, contrast_method,
             yolo_confidence_threshold),
        )


def update_log_status(conn, log_id: str, status: str,
                       error_message: Optional[str] = None, completed_at: Optional[str] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE logs SET status = %s, error_message = COALESCE(%s, error_message), "
            "completed_at = COALESCE(%s, completed_at) WHERE id = %s",
            (status, error_message, completed_at, log_id),
        )


def update_log_counts(conn, log_id: str, n_frames: int, n_detections: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE logs SET n_frames = %s, n_detections = %s WHERE id = %s",
                     (n_frames, n_detections, log_id))


def update_log_performance(conn, log_id: str, inference_ms: float, frames: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE logs SET detector_inference_ms = %s, detector_frames = %s WHERE id = %s",
                    (inference_ms, frames, log_id))


def get_log(conn, log_id: str) -> Optional[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM logs WHERE id = %s", (log_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_logs(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM logs ORDER BY uploaded_at DESC")
        return [dict(r) for r in cur.fetchall()]


def insert_frame_analysis(conn, row: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO frame_analyses "
            "(log_id, frame_index, frame_record_id, whole_image_error, percentile, vae_panel_dir) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (log_id, frame_record_id) DO UPDATE SET "
            "whole_image_error=EXCLUDED.whole_image_error, percentile=EXCLUDED.percentile, "
            "vae_panel_dir=EXCLUDED.vae_panel_dir",
            (row["log_id"], row["frame_index"], row["frame_record_id"], row["whole_image_error"],
             row["percentile"], row["vae_panel_dir"]),
        )


def list_frame_analyses(conn, log_id: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM frame_analyses WHERE log_id = %s ORDER BY frame_index", (log_id,))
        return [dict(row) for row in cur.fetchall()]


# --------------------------------------------------------------------------
# detections
# --------------------------------------------------------------------------

def insert_detection(conn, det: dict[str, Any]) -> None:
    """Same contract as db.py's insert_detection: `det` keys match the
    `detections` table columns; `confidence_breakdown` may be a dict (JSONB
    column accepts it via psycopg2's Json adapter) or an already-encoded
    string."""
    breakdown = det.get("confidence_breakdown")
    if isinstance(breakdown, dict):
        breakdown = psycopg2.extras.Json(breakdown)
    elif isinstance(breakdown, str):
        breakdown = psycopg2.extras.Json(json.loads(breakdown))

    lat, lon = det.get("lat"), det.get("lon")
    geom_expr = _geom_expr(lat, lon)
    geom_sql = geom_expr if geom_expr else "NULL"
    geom_params = (lon, lat) if geom_expr else ()

    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO detections (
                id, log_id, frame_index, frame_record_id, frame_image_path, class_name, yolo_conf,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence_score, confidence_label,
                confidence_breakdown, vae_box_error, vae_whole_image_error, vae_whole_image_percentile,
                lat, lon, geo_method, depth_m, vae_panel_dir, created_at, geom
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, {geom_sql})""",
            (
                det["id"], det["log_id"], det["frame_index"], det["frame_record_id"],
                det.get("frame_image_path"), det["class_name"], det["yolo_conf"],
                *det["bbox"], det["confidence_score"], det["confidence_label"], breakdown,
                det.get("vae_box_error"), det.get("vae_whole_image_error"), det.get("vae_whole_image_percentile"),
                lat, lon, det.get("geo_method"), det.get("depth_m"),
                det.get("vae_panel_dir"), det["created_at"], *geom_params,
            ),
        )


def list_detections(conn, log_id: str, min_confidence: Optional[float] = None) -> list[dict]:
    query = "SELECT * FROM detections WHERE log_id = %s"
    params: list[Any] = [log_id]
    if min_confidence is not None:
        query += " AND confidence_score >= %s"
        params.append(min_confidence)
    query += " ORDER BY frame_index, id"
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d.pop("geom", None)  # internal-only; lat/lon already carry the same info for callers
        # confidence_breakdown comes back already-deserialized (JSONB -> dict via psycopg2)
        results.append(d)
    return results


def detections_within_radius(conn, lat: float, lon: float, radius_m: float,
                              log_id: Optional[str] = None) -> list[dict]:
    """PostGIS-only bonus query (no SQLite equivalent -- not part of the
    db.py-compatible surface, so db_backend.py doesn't re-export it; callers
    that want this must import db_postgis directly and accept that it's
    unavailable on the sqlite backend). Demonstrates why you'd pick this
    backend: 'debris within N meters of a point' via ST_DWithin against a
    geography cast, using the GIST index on geom."""
    query = (
        "SELECT *, ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) AS distance_m "
        "FROM detections WHERE geom IS NOT NULL "
        "AND ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)"
    )
    params: list[Any] = [lon, lat, lon, lat, radius_m]
    if log_id is not None:
        query += " AND log_id = %s"
        params.append(log_id)
    query += " ORDER BY distance_m"
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d.pop("geom", None)
        results.append(d)
    return results

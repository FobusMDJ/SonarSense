"""SQLite persistence for logs (one processed sonar log = one upload) and
detections (one row per YOLO detection, with its confidence score and
geolocation). Plain stdlib sqlite3 -- no ORM -- to match this project's
existing lean-dependency style (no ORM anywhere else in the codebase).

Every write function takes an already-open `sqlite3.Connection` (via
`get_connection()` as a context manager) rather than opening its own, so a
request handler can wrap several writes in one transaction.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from src.utils.config import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = Path("sonarsense.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source_format TEXT NOT NULL,       -- 'xtf' | 'image' | 'image+sidecar'
    status TEXT NOT NULL,              -- 'uploaded' | 'processing' | 'done' | 'error'
    uploaded_at TEXT NOT NULL,
    completed_at TEXT,
    n_frames INTEGER NOT NULL DEFAULT 0,
    n_detections INTEGER NOT NULL DEFAULT 0,
    pixels_to_meters REAL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    log_id TEXT NOT NULL REFERENCES logs(id),
    frame_index INTEGER NOT NULL,
    frame_record_id TEXT NOT NULL,
    frame_image_path TEXT,
    class_name TEXT NOT NULL,
    yolo_conf REAL NOT NULL,
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    confidence_score REAL NOT NULL,
    confidence_label TEXT NOT NULL,
    confidence_breakdown TEXT,         -- JSON
    vae_box_error REAL,
    vae_whole_image_error REAL,
    vae_whole_image_percentile REAL,
    lat REAL,
    lon REAL,
    geo_method TEXT,                   -- 'nav_fix' | 'placeholder'
    vae_panel_dir TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_detections_log_id ON detections(log_id);
"""


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(SCHEMA)
    logger.info("Initialized SQLite schema at %s", db_path)


@contextmanager
def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------
# logs
# --------------------------------------------------------------------------

def create_log(conn: sqlite3.Connection, log_id: str, filename: str, source_format: str,
                uploaded_at: str, pixels_to_meters: Optional[float] = None) -> None:
    conn.execute(
        "INSERT INTO logs (id, filename, source_format, status, uploaded_at, pixels_to_meters) "
        "VALUES (?, ?, ?, 'uploaded', ?, ?)",
        (log_id, filename, source_format, uploaded_at, pixels_to_meters),
    )


def update_log_status(conn: sqlite3.Connection, log_id: str, status: str,
                       error_message: Optional[str] = None, completed_at: Optional[str] = None) -> None:
    conn.execute(
        "UPDATE logs SET status = ?, error_message = COALESCE(?, error_message), "
        "completed_at = COALESCE(?, completed_at) WHERE id = ?",
        (status, error_message, completed_at, log_id),
    )


def update_log_counts(conn: sqlite3.Connection, log_id: str, n_frames: int, n_detections: int) -> None:
    conn.execute("UPDATE logs SET n_frames = ?, n_detections = ? WHERE id = ?", (n_frames, n_detections, log_id))


def get_log(conn: sqlite3.Connection, log_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM logs WHERE id = ?", (log_id,)).fetchone()
    return dict(row) if row else None


def list_logs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM logs ORDER BY uploaded_at DESC").fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# detections
# --------------------------------------------------------------------------

def insert_detection(conn: sqlite3.Connection, det: dict[str, Any]) -> None:
    """`det` keys must match the `detections` table columns; `confidence_breakdown`
    may be a dict (will be JSON-encoded) or an already-encoded string."""
    breakdown = det.get("confidence_breakdown")
    if isinstance(breakdown, dict):
        breakdown = json.dumps(breakdown)
    conn.execute(
        """INSERT INTO detections (
            id, log_id, frame_index, frame_record_id, frame_image_path, class_name, yolo_conf,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence_score, confidence_label,
            confidence_breakdown, vae_box_error, vae_whole_image_error, vae_whole_image_percentile,
            lat, lon, geo_method, vae_panel_dir, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            det["id"], det["log_id"], det["frame_index"], det["frame_record_id"],
            det.get("frame_image_path"), det["class_name"], det["yolo_conf"],
            *det["bbox"], det["confidence_score"], det["confidence_label"], breakdown,
            det.get("vae_box_error"), det.get("vae_whole_image_error"), det.get("vae_whole_image_percentile"),
            det.get("lat"), det.get("lon"), det.get("geo_method"), det.get("vae_panel_dir"),
            det["created_at"],
        ),
    )


def list_detections(conn: sqlite3.Connection, log_id: str,
                     min_confidence: Optional[float] = None) -> list[dict]:
    query = "SELECT * FROM detections WHERE log_id = ?"
    params: list[Any] = [log_id]
    if min_confidence is not None:
        query += " AND confidence_score >= ?"
        params.append(min_confidence)
    query += " ORDER BY frame_index, id"
    rows = conn.execute(query, params).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("confidence_breakdown"):
            try:
                d["confidence_breakdown"] = json.loads(d["confidence_breakdown"])
            except (TypeError, json.JSONDecodeError):
                pass
        results.append(d)
    return results

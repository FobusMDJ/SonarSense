"""Generates a standalone, PostGIS-loadable `.sql` dump of one log's
detections -- the downloadable artifact for GET /logs/{log_id}/report.sql.

WHY THIS EXISTS: the reference pitch deck's step 9 ("GEOSPATIAL DATABASE")
shows GeoJSON -> PostGIS explicitly, and this project's own db_postgis.py
already implements that as a LIVE connection (when config/backend.yaml sets
db_backend: postgres). But there was no way to get a `.sql` file out of a
log for someone who just wants to inspect/replay it later (`psql -f
report.sql`) without standing up a live Postgres connection first. This
module is that missing piece.

DOES NOT import db_postgis.py on purpose: db_postgis.py hard-requires
psycopg2, but SQLite (db_backend: sqlite) is this project's zero-setup
default, and report generation must keep working for every log regardless
of which backend is actually configured. The CREATE TABLE below is kept in
sync BY HAND with db_postgis.py's SCHEMA constant (detections table only)
-- if that schema changes, update both.

Every detection becomes one INSERT against the REAL `detections` table this
project actually uses (confidence breakdown, VAE fields, real-world
length_m/width_m geometry, footprint polygon, geom/geom_footprint) -- NOT
the placeholder `marine_debris` table the
originally-uploaded standalone geolocation_engine.py script produced (see
csv_engine.py's module docstring for that script's own `--load-postgis`
behavior, preserved there unchanged for CLI parity). A demo/report artifact
should show this project's real schema, not a toy one.

Values are inlined as SQL literals, not parameterized -- this is a static
file meant to be replayed later by `psql`, not executed by this process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Kept in sync BY HAND with db_postgis.py's SCHEMA (detections table only) --
# see module docstring for why this isn't just imported from there.
CREATE_DETECTIONS_TABLE_SQL = """CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    log_id TEXT NOT NULL,
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
    length_m DOUBLE PRECISION,
    width_m DOUBLE PRECISION,
    footprint_geojson JSONB,
    vae_panel_dir TEXT,
    created_at TEXT NOT NULL,
    geom geometry(Point, 4326),
    geom_footprint geometry(Polygon, 4326)
);

CREATE INDEX IF NOT EXISTS idx_detections_log_id ON detections(log_id);
CREATE INDEX IF NOT EXISTS idx_detections_geom ON detections USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_detections_geom_footprint ON detections USING GIST(geom_footprint);
"""

_DETECTIONS_COLUMNS = [
    "id", "log_id", "frame_index", "frame_record_id", "frame_image_path", "class_name", "yolo_conf",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence_score", "confidence_label",
    "confidence_breakdown", "vae_box_error", "vae_whole_image_error", "vae_whole_image_percentile",
    "lat", "lon", "geo_method", "depth_m", "length_m", "width_m", "footprint_geojson",
    "vae_panel_dir", "created_at", "geom", "geom_footprint",
]


def _sql_str(value: Any) -> str:
    """NULL for None, a single-quoted/escaped SQL string literal otherwise."""
    return "NULL" if value is None else "'" + str(value).replace("'", "''") + "'"


def _sql_num(value: Any) -> str:
    return "NULL" if value is None else repr(float(value))


def _sql_jsonb(value: Any) -> str:
    if value is None:
        return "NULL"
    payload = value if isinstance(value, str) else json.dumps(value)  # already-encoded JSON text
    # (e.g. footprint_geojson/confidence_breakdown as stored) passes through as-is; a plain
    # dict/list gets encoded here.
    return "'" + payload.replace("'", "''") + "'::jsonb"


def _geom_expr(lat: Any, lon: Any) -> str:
    if lat is None or lon is None or (float(lat) == 0.0 and float(lon) == 0.0):
        return "NULL"  # placeholder detections never get a real point -- same rule as db_postgis.py
    return f"ST_SetSRID(ST_MakePoint({float(lon)!r}, {float(lat)!r}), 4326)"


def _geom_footprint_expr(footprint_geojson: Any) -> str:
    ring = footprint_geojson
    if isinstance(ring, str):
        try:
            ring = json.loads(ring)
        except json.JSONDecodeError:
            ring = None
    if not ring:
        return "NULL"
    polygon = json.dumps({"type": "Polygon", "coordinates": [ring]})
    return "ST_SetSRID(ST_GeomFromGeoJSON('" + polygon.replace("'", "''") + "'), 4326)"


def build_postgis_sql(detections: list[dict[str, Any]]) -> str:
    """One CREATE TABLE + one INSERT per detection, matching the real
    `detections` table (see module docstring)."""
    lines = [CREATE_DETECTIONS_TABLE_SQL.rstrip(), ""]
    for d in detections:
        values = [
            _sql_str(d.get("id")), _sql_str(d.get("log_id")), _sql_num(d.get("frame_index")),
            _sql_str(d.get("frame_record_id")), _sql_str(d.get("frame_image_path")), _sql_str(d.get("class_name")),
            _sql_num(d.get("yolo_conf")),
            _sql_num(d.get("bbox_x1")), _sql_num(d.get("bbox_y1")), _sql_num(d.get("bbox_x2")), _sql_num(d.get("bbox_y2")),
            _sql_num(d.get("confidence_score")), _sql_str(d.get("confidence_label")),
            _sql_jsonb(d.get("confidence_breakdown")), _sql_num(d.get("vae_box_error")),
            _sql_num(d.get("vae_whole_image_error")), _sql_num(d.get("vae_whole_image_percentile")),
            _sql_num(d.get("lat")), _sql_num(d.get("lon")), _sql_str(d.get("geo_method")), _sql_num(d.get("depth_m")),
            _sql_num(d.get("length_m")), _sql_num(d.get("width_m")),
            _sql_jsonb(d.get("footprint_geojson")), _sql_str(d.get("vae_panel_dir")), _sql_str(d.get("created_at")),
            _geom_expr(d.get("lat"), d.get("lon")), _geom_footprint_expr(d.get("footprint_geojson")),
        ]
        lines.append(f"INSERT INTO detections ({', '.join(_DETECTIONS_COLUMNS)}) VALUES ({', '.join(values)});")
    return "\n".join(lines) + "\n"


def write_postgis_sql(path: str | Path, detections: list[dict[str, Any]]) -> str:
    """Builds the SQL dump and writes it to `path`. Returns the text
    (handy for a caller that wants both the file AND the string without
    re-reading it)."""
    sql = build_postgis_sql(detections)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sql, encoding="utf-8")
    return sql

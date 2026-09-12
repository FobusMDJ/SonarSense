"""CSV-based geolocation engine -- a second ingestion path alongside the
main image+XTF pipeline (pipeline_runner.py + georeference.py).

That pipeline runs YOLO/VAE against raw sonar frames itself. This module
instead geolocates detections that ALREADY EXIST as a CSV -- produced by an
external/offline detector (a Roboflow-hosted model, another team's
detector, a prior batch run, etc.) -- joined against a nav CSV keyed by
`ping_id`. It was originally a standalone CLI script; that CLI is preserved
below (`run_pipeline()` / `main()`) for anyone still using it that way, but
the function this backend actually calls is `run_csv_pipeline_to_detections()`,
which returns rows shaped for `src.backend.db.insert_detection()` so CSV-
sourced detections land in the SAME `detections` table as pipeline-produced
ones -- one unified store regardless of source. See `main.py`'s
`POST /logs/geolocate_csv` for the endpoint that wires this in.

INPUT FORMAT:
  detections.csv columns: ping_id, pixel_x, bbox_w (or width/w),
    bbox_h (or height/h), class, confidence
  nav.csv columns: ping_id, lat (or ship_lat), lon (or ship_lon),
    heading (or heading_deg), altitude_m (or altitude),
    depth_m (or depth), timestamp (or timestamp_utc)

MATH (unchanged from the original script -- see geolocate_csv_detection):
  1. slant range = |pixel_x - image_center| * range_per_pixel_m
  2. ground range = sqrt(slant_range^2 - altitude^2), clamped to 0 when the
     detection is close enough to nadir that this would go negative
  3. side = "port" (pixel left of center) or "starboard" (right of center);
     bearing = heading -/+ 90 degrees respectively
  4. east/north offset = ground_range * sin/cos(bearing)
  5. flat-earth offset applied to the nav fix's own lat/lon (same
     equirectangular approximation as georeference.py's offset_latlon --
     accurate to a tiny fraction of a mm at these distances)

ONE INTENTIONAL FIX vs the originally uploaded script: that script computed
`depth_m` as just the nav fix's own depth, with no altitude correction. The
detected object physically rests on/near the seafloor, which is
`altitude_m` BELOW the vehicle -- so the object's depth is
`nav_depth + altitude`, not the vehicle's own depth. Fixed here
(`object_depth_m` in `geolocate_csv_detection`). If altitude is missing/0
for a nav row, this degrades gracefully to the vehicle depth alone rather
than guessing an altitude that isn't there.

WHAT THIS MODULE DELIBERATELY DOES NOT DO (unlike georeference.py):
  - No placeholder discipline for missing nav: a ping_id with no matching
    nav row is simply skipped (see run_pipeline/run_csv_pipeline_to_detections),
    not stored as a placeholder detection. There is no "nav row present but
    some fields blank" case in practice for this CSV format the way there
    is for optional XTF/sidecar fields, so this hasn't come up -- flagged
    here in case a future caller feeds genuinely partial nav data.
  - No footprint polygon (georeference.py's 4-corner footprint needs a
    bounding-box ROW/column pair; this CSV format only carries pixel_x, not
    a pixel_y, so there's no way to place the box vertically in the source
    image). `footprint_geojson` is always None for these detections.
"""

from __future__ import annotations

import csv
import json
import math
import uuid
from pathlib import Path
from typing import Any, Optional

EARTH_RADIUS_M = 6378137.0  # WGS84 -- same constant georeference.py uses


def offset_to_latlon(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Flat-earth/equirectangular offset -- same approximation as
    georeference.py's offset_latlon, accurate to a tiny fraction of a
    millimeter at these (tens-of-meters) detection-offset distances."""
    d_lat = north_m / EARTH_RADIUS_M
    d_lon = east_m / (EARTH_RADIUS_M * math.cos(math.pi * lat / 180.0))
    return lat + math.degrees(d_lat), lon + math.degrees(d_lon)


def geolocate_csv_detection(det: dict, nav: dict, range_per_pixel_m: float, image_width: int) -> dict:
    """Runs one detection through the full slant-range -> ground-range ->
    ENU-offset -> lat/lon chain (see module docstring for the math and the
    one intentional depth fix). Returns a plain dict of derived values --
    NOT yet shaped for the `detections` table; see
    `_to_detection_row()` for that."""
    lat = float(nav.get("lat", nav.get("ship_lat", 0.0)) or 0.0)
    lon = float(nav.get("lon", nav.get("ship_lon", 0.0)) or 0.0)
    heading = float(nav.get("heading", nav.get("heading_deg", 0.0)) or 0.0)
    altitude_m = float(nav.get("altitude_m", nav.get("altitude", 0.0)) or 0.0)
    depth_m = float(nav.get("depth_m", nav.get("depth", 0.0)) or 0.0)
    timestamp = nav.get("timestamp", nav.get("timestamp_utc", None))

    pixel_x = float(det["pixel_x"])
    bbox_w_pixels = float(det.get("bbox_w", det.get("width", det.get("w", 0.0))) or 0.0)
    bbox_h_pixels = float(det.get("bbox_h", det.get("height", det.get("h", 0.0))) or 0.0)

    center_x = image_width / 2.0
    pixels_from_center = abs(pixel_x - center_x)
    slant_range_m = pixels_from_center * range_per_pixel_m

    if slant_range_m > altitude_m:
        ground_range_m = math.sqrt(slant_range_m ** 2 - altitude_m ** 2)
    else:
        ground_range_m = 0.0

    if pixel_x < center_x:
        side = "port"
        bearing_deg = heading - 90.0
    else:
        side = "starboard"
        bearing_deg = heading + 90.0

    bearing_rad = math.radians(bearing_deg)
    east_m = ground_range_m * math.sin(bearing_rad)
    north_m = ground_range_m * math.cos(bearing_rad)

    obj_lat, obj_lon = offset_to_latlon(lat, lon, east_m, north_m)

    width_m = round(bbox_w_pixels * range_per_pixel_m, 2)
    length_m = round(bbox_h_pixels * range_per_pixel_m, 2)

    # FIX (see module docstring): object depth = vehicle depth + altitude,
    # not the vehicle's own depth alone.
    object_depth_m = round(depth_m + altitude_m, 2)

    return {
        "ping_id": str(nav.get("ping_id", det.get("ping_id", ""))),
        "class_name": det.get("class", "unknown"),
        "confidence": float(det.get("confidence", 0.0) or 0.0),
        "lat": round(obj_lat, 7),
        "lon": round(obj_lon, 7),
        "depth_m": object_depth_m,
        "length_m": length_m,
        "width_m": width_m,
        "side": side,
        "ground_range_m": round(ground_range_m, 2),
        "timestamp": timestamp,
        "pixel_x": pixel_x,
        "bbox_w_px": bbox_w_pixels,
        "bbox_h_px": bbox_h_pixels,
    }


def _load_csv_rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_nav_csv(path: str | Path) -> dict[str, dict]:
    """ping_id -> nav row, for O(1) join lookups."""
    return {str(row["ping_id"]): row for row in _load_csv_rows(path)}


def _to_detection_row(geo: dict, log_id: str, frame_index: int, created_at: str) -> dict:
    """Shapes one geolocate_csv_detection() result into a dict matching
    `src.backend.db.insert_detection()`'s expected keys (same shape
    pipeline_runner.py builds for image-pipeline detections) -- this is
    what makes CSV-sourced detections show up in GET /logs/{id}/detections,
    /map, and every other endpoint identically to pipeline-produced ones.

    bbox_x1/x2 are reconstructed from pixel_x +/- half the box width so the
    frontend's existing bbox-based dimension math (bbox * pixels_to_meters)
    reproduces the same length/width this module already computed
    independently. There's no pixel_y in this CSV format at all (see module
    docstring), so bbox_y1/y2 are placed at an arbitrary but harmless
    [0, bbox_h] -- vertical position in the source frame is simply unknown
    here, unlike the image pipeline where it's real."""
    half_w = geo["bbox_w_px"] / 2.0
    from src.confidence.scoring import score_detection  # local import: avoids a hard
    # dependency on the confidence module for callers that only want the geolocation math
    conf_result = score_detection(
        yolo_conf=geo["confidence"], class_name=geo["class_name"],
        xyxy=[geo["pixel_x"] - half_w, 0.0, geo["pixel_x"] + half_w, geo["bbox_h_px"]],
        gray_image=None,  # no source frame available for a CSV-only detection
        vae_whole_image_percentile=None,  # no VAE stage ran on this detection
    )
    return {
        "id": str(uuid.uuid4()),
        "log_id": log_id,
        "frame_index": frame_index,
        "frame_record_id": geo["ping_id"],
        "frame_image_path": None,
        "class_name": geo["class_name"],
        "yolo_conf": geo["confidence"],
        "bbox": [geo["pixel_x"] - half_w, 0.0, geo["pixel_x"] + half_w, geo["bbox_h_px"]],
        "confidence_score": conf_result.score,
        "confidence_label": conf_result.label,
        "confidence_breakdown": conf_result.breakdown,
        "vae_box_error": None,
        "vae_whole_image_error": None,
        "vae_whole_image_percentile": None,
        "lat": geo["lat"],
        "lon": geo["lon"],
        "geo_method": "nav_fix",  # every row here had a matched nav fix by construction --
        # rows with no nav match are skipped before this point, see run_csv_pipeline_to_detections
        "depth_m": geo["depth_m"],
        "length_m": geo["length_m"],  # THIS engine's own real-world size, computed directly from
        "width_m": geo["width_m"],    # bbox_w_px/bbox_h_px * range_per_pixel_m in geolocate_csv_
        # detection above -- stored as-is so downstream readers (GET /logs/{id}/detections, the
        # report writers) use it instead of re-deriving a size from a reconstructed bbox (see
        # class_taxonomy.resolved_dimensions_m, which prefers these columns when they're set).
        "footprint_geojson": None,  # no pixel_y in this format -- see module docstring
        "vae_panel_dir": None,
        "created_at": created_at,
    }


def run_csv_pipeline_to_detections(
    detections_csv: str | Path, nav_csv: str | Path, image_width: int, range_per_pixel_m: float,
    log_id: str, created_at: str,
) -> tuple[list[dict], int]:
    """The backend-facing entry point: reads both CSVs, joins on ping_id,
    geolocates every matched detection, and returns
    (detection_rows_ready_for_db.insert_detection, n_skipped_for_no_nav_match).
    Mirrors run_pipeline()'s join/skip behavior below exactly, just
    returning DB-shaped rows instead of GeoJSON Features."""
    nav_by_ping = load_nav_csv(nav_csv)
    rows: list[dict] = []
    skipped = 0
    for frame_index, det in enumerate(_load_csv_rows(detections_csv)):
        ping_id = str(det["ping_id"])
        nav = nav_by_ping.get(ping_id)
        if nav is None:
            skipped += 1
            continue
        geo = geolocate_csv_detection(det, nav, range_per_pixel_m, image_width)
        rows.append(_to_detection_row(geo, log_id, frame_index, created_at))
    return rows, skipped


# --------------------------------------------------------------------------
# Original standalone-CLI surface, preserved for parity with the script this
# module was adapted from (`geolocation_engine.py`) -- same behavior, same
# output files, with the one depth fix from the module docstring applied.
# --------------------------------------------------------------------------

def build_geojson_feature(geo: dict) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [geo["lon"], geo["lat"]]},
        "properties": {
            "ping_id": geo["ping_id"],
            "class": geo["class_name"],
            "confidence": geo["confidence"],
            "depth_m": geo["depth_m"],
            "length_m": geo["length_m"],
            "width_m": geo["width_m"],
            "side": geo["side"],
            "ground_range_m": geo["ground_range_m"],
            "timestamp": geo["timestamp"],
        },
    }


def run_pipeline(detections_file, nav_file, image_width, range_per_pixel) -> list[dict]:
    """CLI-compatible entry point: reads the CSVs, matches pings, and
    returns a list of GeoJSON Features (same contract as the original
    script's run_pipeline())."""
    print("Loading navigation metadata...")
    nav_dict = load_nav_csv(nav_file)

    print("Processing AI detections...")
    results = []
    for det in _load_csv_rows(detections_file):
        ping_id = str(det["ping_id"])
        nav = nav_dict.get(ping_id)
        if nav is None:
            print(f"Warning: No navigation data found for ping_id {ping_id}")
            continue
        geo = geolocate_csv_detection(det, nav, range_per_pixel, image_width)
        results.append(build_geojson_feature(geo))
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SonarSense Geolocation Engine")
    parser.add_argument("--detections", required=True, help="Path to detections CSV")
    parser.add_argument("--nav", required=True, help="Path to navigation/telemetry CSV")
    parser.add_argument("--image-width", type=int, required=True, help="Total width of sonar image in pixels")
    parser.add_argument("--range-per-pixel", type=float, required=True, help="Resolution in meters per pixel")
    parser.add_argument("--output", default="detections.geojson", help="Output GeoJSON file path")
    parser.add_argument("--load-postgis", action="store_true", help="Also write a .sql load file")
    parser.add_argument("--db-url", help="(unused by --load-postgis, which only writes a SQL dump file -- "
                                          "kept for CLI compatibility with the original script)")
    args = parser.parse_args()

    features = run_pipeline(args.detections, args.nav, args.image_width, args.range_per_pixel)
    geojson_output = {"type": "FeatureCollection", "features": features}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(geojson_output, f, indent=2)

    print(f"Successfully geolocated {len(features)} objects.")
    print(f"Output saved to {args.output}")

    if args.load_postgis:
        sql_filename = args.output.replace(".geojson", ".sql")
        with open(sql_filename, "w") as f:
            f.write("CREATE TABLE IF NOT EXISTS marine_debris "
                     "(id SERIAL PRIMARY KEY, geom GEOMETRY(Point, 4326), properties JSONB);\n")
            for feature in features:
                geom = json.dumps(feature["geometry"])
                props = json.dumps(feature["properties"])
                f.write(f"INSERT INTO marine_debris (geom, properties) VALUES "
                        f"(ST_GeomFromGeoJSON('{geom}'), '{props}'::jsonb);\n")
        print(f"Generated PostGIS SQL file: {sql_filename}")


if __name__ == "__main__":
    main()

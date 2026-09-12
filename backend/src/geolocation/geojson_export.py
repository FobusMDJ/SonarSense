"""The geolocation engine's GeoJSON output method.

Every other geolocation concern (nav-fix modeling in nav.py, pixel->lat/lon
math in georeference.py) already lived in this package; the actual
RFC 7946 FeatureCollection serialization was, until now, built inline
inside src/backend/main.py's GET /logs/{log_id}/map handler -- meaning the
geolocation engine itself had no GeoJSON output of its own, and there was
no way to get a GeoJSON file out of a processed log (only the live /map
endpoint's in-memory dict). This module fixes both: a single reusable
builder function, used by /map AND by a new downloadable
GET /logs/{log_id}/report.geojson endpoint AND by pipeline_runner.py's
_write_reports() (so every processed log gets a report.geojson written to
disk automatically, alongside report.json/report.csv).

Only detections with a real, nav-derived location (geo_method == 'nav_fix')
are included by default -- a detection with no nav data resolves to a
(0, 0) placeholder (see georeference.py), and plotting that on a map would
misrepresent it as a real position off the coast of Africa. Pass
only_geolocated=False to include everything anyway (e.g. for a debug export
that also wants to see what got excluded and why).

GEOMETRY: each Feature carries the debris's center POINT (as before) and,
when `footprint_geojson` is present on the detection (see georeference.py's
GeoResult.footprint / pipeline_runner.py), its 4-corner footprint POLYGON
too -- the actual real-world shape of the object, not just a dot.
`geometry_mode` controls how the two combine into the one `geometry` a
GeoJSON Feature is allowed to have (RFC 7946 -- one geometry per Feature):

  "auto"           (default) GeometryCollection [Point, Polygon] when a
                    footprint exists, else plain Point. This is the
                    functional change over the old behavior.
  "point_only"     always plain Point, footprint ignored -- the exact
                    output this module produced before footprints existed,
                    for a consumer that can't handle GeometryCollection.
  "footprint_only" Polygon when available, else falls back to Point (never
                    drops a detection just because it predates footprints).

Either way, `properties.has_footprint` says which shape is actually present
without needing to inspect `geometry.type`, and `properties.footprint`
carries the raw [[lon, lat], ...] ring directly too, for a consumer that
would rather read coordinates from properties than parse GeometryCollection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GEOMETRY_MODES = ("auto", "point_only", "footprint_only")


def _footprint_ring(d: dict[str, Any]) -> list | None:
    ring = d.get("footprint_geojson")
    if isinstance(ring, str):
        try:
            ring = json.loads(ring)
        except json.JSONDecodeError:
            return None
    return ring or None


def _feature_geometry(lon: float, lat: float, ring: list | None, geometry_mode: str) -> dict[str, Any]:
    point = {"type": "Point", "coordinates": [lon, lat]}
    if geometry_mode == "point_only" or not ring:
        return point
    polygon = {"type": "Polygon", "coordinates": [ring]}
    if geometry_mode == "footprint_only":
        return polygon
    return {"type": "GeometryCollection", "geometries": [point, polygon]}  # "auto"


def build_geojson(detections: list[dict[str, Any]], only_geolocated: bool = True,
                   geometry_mode: str = "auto") -> dict[str, Any]:
    """Builds an RFC 7946 FeatureCollection from a list of detection dicts
    (as returned by src.backend.db.list_detections). Each qualifying
    detection becomes one Feature (see module docstring for `geometry_mode`);
    every value the "detection records" area already shows is carried into
    `properties` too, so a GIS tool (QGIS, geojson.io, ...) opening this
    file has the full picture without a second round-trip to the API.
    """
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(f"geometry_mode must be one of {GEOMETRY_MODES}, got {geometry_mode!r}")

    features = []
    for d in detections:
        has_location = d.get("lat") is not None and d.get("lon") is not None
        is_real = d.get("geo_method") == "nav_fix"
        if only_geolocated and not (has_location and is_real):
            continue
        if not has_location:
            continue  # can't emit a Point with no coordinates regardless of only_geolocated

        ring = _footprint_ring(d)
        features.append({
            "type": "Feature",
            "geometry": _feature_geometry(d["lon"], d["lat"], ring, geometry_mode),
            "properties": {
                "detection_id": d.get("id"),
                "log_id": d.get("log_id"),
                "class": d.get("class_name"),
                "confidence_score": d.get("confidence_score"),
                "confidence_label": d.get("confidence_label"),
                "yolo_conf": d.get("yolo_conf"),
                "frame": d.get("frame_record_id"),
                "frame_index": d.get("frame_index"),
                "bbox_px": [d.get("bbox_x1"), d.get("bbox_y1"), d.get("bbox_x2"), d.get("bbox_y2")],
                "vae_whole_image_percentile": d.get("vae_whole_image_percentile"),
                "geo_method": d.get("geo_method"),
                "created_at": d.get("created_at"),
                "has_footprint": ring is not None,
                "footprint": ring,  # [[lon, lat], ...] closed ring, or None
            },
        })

    return {"type": "FeatureCollection", "features": features}


def write_geojson(path: str | Path, detections: list[dict[str, Any]], only_geolocated: bool = True,
                   geometry_mode: str = "auto") -> dict[str, Any]:
    """Builds the FeatureCollection and writes it to `path`. Returns the
    dict that was written (handy for a caller that wants both the file AND
    the in-memory value without re-parsing it)."""
    geojson = build_geojson(detections, only_geolocated=only_geolocated, geometry_mode=geometry_mode)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
    return geojson

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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_geojson(detections: list[dict[str, Any]], only_geolocated: bool = True) -> dict[str, Any]:
    """Builds an RFC 7946 FeatureCollection from a list of detection dicts
    (as returned by src.backend.db.list_detections). Each qualifying
    detection becomes one Point Feature; every value the "detection records"
    area already shows is carried into `properties` too, so a GIS tool
    (QGIS, geojson.io, ...) opening this file has the full picture without
    a second round-trip to the API.
    """
    features = []
    for d in detections:
        has_location = d.get("lat") is not None and d.get("lon") is not None
        is_real = d.get("geo_method") == "nav_fix"
        if only_geolocated and not (has_location and is_real):
            continue
        if not has_location:
            continue  # can't emit a Point with no coordinates regardless of only_geolocated

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
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
            },
        })

    return {"type": "FeatureCollection", "features": features}


def write_geojson(path: str | Path, detections: list[dict[str, Any]], only_geolocated: bool = True) -> dict[str, Any]:
    """Builds the FeatureCollection and writes it to `path`. Returns the
    dict that was written (handy for a caller that wants both the file AND
    the in-memory value without re-parsing it)."""
    geojson = build_geojson(detections, only_geolocated=only_geolocated)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
    return geojson

"""Navigation fix data model + interpolation shared by both geolocation
input paths (raw XTF nav packets, and the image+CSV-sidecar path)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class NavFix:
    """One vehicle position fix. `frame_index` ties it to a specific
    ingested frame/tile/ping; `timestamp` (unix seconds, optional) is used
    to interpolate between fixes when frame count != nav-fix count."""

    frame_index: int
    lat: float
    lon: float
    heading_deg: float  # compass bearing, 0=N, 90=E, clockwise
    altitude_m: Optional[float] = None  # height above seafloor, if known
    depth_m: Optional[float] = None  # water depth AT the fix (below sea surface -- a
    # DIFFERENT axis than altitude_m above, which is height above the seafloor). Optional
    # because most nav sidecars built so far don't carry it; stays None rather than being
    # fabricated when absent, matching this project's standing "never fake position data"
    # convention (see georeference.py's GeoResult.method='placeholder'). API responses
    # pass this straight through as null when unset rather than inventing a number.
    timestamp: Optional[float] = None


def _parse_timestamp(value: str | None) -> Optional[float]:
    """Accept Unix seconds or an ISO-8601 timestamp from exported metadata."""
    if not value or not value.strip():
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as exc:
            raise ValueError(
                f"Invalid timestamp '{value}'. Use Unix seconds or ISO-8601, for example 2026-06-14T09:00:00Z."
            ) from exc


def load_nav_sidecar(path: str | Path) -> list[NavFix]:
    """Load a nav sidecar CSV with columns:
    frame_index,lat,lon,heading_deg[,altitude_m,depth_m,timestamp]

    This is the expected format for the "exported images + nav sidecar"
    ingestion path. One row per frame/tile, in the same order the images
    were exported.
    """
    path = Path(path)
    fixes = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"frame_index", "lat", "lon", "heading_deg"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Nav sidecar {path} must have columns {sorted(required)} "
                f"(+ optional altitude_m, depth_m, timestamp). Found: {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                heading = float(row["heading_deg"])
                if not -90 <= lat <= 90:
                    raise ValueError(f"latitude {lat} is outside -90..90")
                if not -180 <= lon <= 180:
                    raise ValueError(f"longitude {lon} is outside -180..180")
                fixes.append(NavFix(
                    frame_index=int(row["frame_index"]),
                    lat=lat,
                    lon=lon,
                    heading_deg=heading % 360.0,
                    altitude_m=float(row["altitude_m"]) if row.get("altitude_m") else None,
                    depth_m=float(row["depth_m"]) if row.get("depth_m") else None,
                    timestamp=_parse_timestamp(row.get("timestamp")),
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid navigation data in {path} at CSV row {line_number}: {exc}") from exc
    fixes.sort(key=lambda f: f.frame_index)
    return fixes


def nearest_fix(fixes: list[NavFix], frame_index: int) -> Optional[NavFix]:
    """Nearest-neighbor lookup by frame_index (fixes assumed sorted).
    Returns None if `fixes` is empty."""
    if not fixes:
        return None
    best = min(fixes, key=lambda f: abs(f.frame_index - frame_index))
    return best


def interpolate_fix(fixes: list[NavFix], frame_index: int) -> Optional[NavFix]:
    """Linear interpolation between the two bracketing fixes by frame_index
    (falls back to nearest_fix at the ends, or if fixes has <2 entries).
    Heading is interpolated the short way around the compass (handles the
    0/360 wraparound)."""
    if not fixes:
        return None
    if len(fixes) == 1 or frame_index <= fixes[0].frame_index:
        return fixes[0]
    if frame_index >= fixes[-1].frame_index:
        return fixes[-1]

    for a, b in zip(fixes, fixes[1:]):
        if a.frame_index <= frame_index <= b.frame_index:
            span = b.frame_index - a.frame_index
            t = 0.0 if span == 0 else (frame_index - a.frame_index) / span
            lat = a.lat + t * (b.lat - a.lat)
            lon = a.lon + t * (b.lon - a.lon)
            # shortest-path heading interpolation
            delta = ((b.heading_deg - a.heading_deg + 180) % 360) - 180
            heading = (a.heading_deg + t * delta) % 360
            alt = None
            if a.altitude_m is not None and b.altitude_m is not None:
                alt = a.altitude_m + t * (b.altitude_m - a.altitude_m)
            depth = None
            if a.depth_m is not None and b.depth_m is not None:
                depth = a.depth_m + t * (b.depth_m - a.depth_m)
            return NavFix(frame_index=frame_index, lat=lat, lon=lon, heading_deg=heading,
                          altitude_m=alt, depth_m=depth)
    return nearest_fix(fixes, frame_index)

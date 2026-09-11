"""Raw XTF (eXtended Triton Format) sonar log reader.

STATUS: UNVALIDATED AGAINST A REAL FIELD FILE. No sample .xtf was available
while building this -- see claude/backend-pipeline-plan.md. It's built
against the public XTF spec and the `pyxtf` library's own test fixtures,
and every field access is defensive (falls back with a logged warning
rather than crashing on an unexpected packet shape). Treat anything this
produces as provisional until it's been run against a real log and the
output visually spot-checked. `xtf_reader_status()` reports exactly this
state so the API/dashboard can surface it rather than silently pretending
this is production-ready.

Uses the `pyxtf` library (https://github.com/oysstu/pyxtf) rather than a
hand-rolled binary parser -- XTF is a well-specified but intricate packet
format (ping headers, per-channel sub-headers, multiple nav packet types
across format revisions); re-implementing that from scratch is a lot of
surface area to get subtly wrong. `pyxtf` handles the packet framing;
everything below is SonarSense-specific field extraction on top of it.

What this produces: a stream of SSSRecord tiles (matching the existing
ingestion.py contract exactly, so nothing downstream needs to change) built
by stacking consecutive sonar pings into fixed-height waterfall images, with
a per-row NavFix attached in metadata so a detection can be geolocated using
the nav fix for the EXACT ping/row it was found in, not just a tile-average.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.geolocation.nav import NavFix
from src.preprocessing.ingestion import SSSRecord
from src.utils.config import get_logger

logger = get_logger(__name__)


class XTFNotAvailable(RuntimeError):
    """Raised when pyxtf isn't installed. Message includes the install command."""


def xtf_reader_status() -> dict:
    """Reports whether the XTF path is usable at all right now, and its
    validation state. Call this from a health-check endpoint rather than
    assuming XTF ingestion works."""
    try:
        import pyxtf
        version = getattr(pyxtf, "__version__", "unknown")
        library_available = True
    except ImportError:
        version = None
        library_available = False
    return {
        "library": "pyxtf",
        "library_available": library_available,
        "library_version": version,
        "validated_against_real_file": False,
        "note": ("XTF field extraction (nav coordinates, heading, range) uses "
                 "defensive best-effort attribute lookups and has not been "
                 "confirmed against a real sonar log. Run it against a real "
                 "file and visually check the resulting waterfall tiles + "
                 "nav track before trusting geolocation output from this path."),
    }


def _require_pyxtf():
    try:
        import pyxtf
        return pyxtf
    except ImportError as exc:
        raise XTFNotAvailable(
            "pyxtf is not installed. Run: pip install pyxtf"
        ) from exc


@dataclass
class PingRecord:
    ping_index: int
    intensity: np.ndarray  # 1D, one row of the waterfall
    nav: NavFix | None


def _extract_nav_from_ping(pyxtf, ping_header, ping_index: int) -> NavFix | None:
    """Best-effort extraction of lat/lon/heading from an XTFPingHeader.
    XTF nav can be in several places/units depending on NavUnits and file
    revision -- this tries the common cases and returns None (not a guess)
    if it can't find anything usable, logging once per failure mode."""
    lat = getattr(ping_header, "SensorYcoordinate", None)
    lon = getattr(ping_header, "SensorXcoordinate", None)
    heading = getattr(ping_header, "SensorHeading", None)
    nav_units = getattr(ping_header, "NavUnits", None)

    if lat is None or lon is None:
        # some files carry ship (tow point) coordinates instead of sensor coordinates
        lat = getattr(ping_header, "ShipYcoordinate", lat)
        lon = getattr(ping_header, "ShipXcoordinate", lon)

    if lat is None or lon is None or heading is None:
        logger.warning(
            "Ping %d: missing lat/lon/heading fields on XTFPingHeader (NavUnits=%s) "
            "-- no NavFix for this ping, geolocation for it will be a placeholder.",
            ping_index, nav_units,
        )
        return None

    # NavUnits: 0 = meters (projected/local grid, NOT lat/lon), 3 = lat/lon degrees.
    # Anything other than 3 means these numbers are NOT degrees -- do not treat
    # them as such (that would silently produce nonsense coordinates).
    if nav_units is not None and int(nav_units) != 3:
        logger.warning(
            "Ping %d: NavUnits=%s (not lat/lon degrees) -- this file uses a "
            "projected coordinate system pyxtf reports as-is. Reproject to "
            "WGS84 lat/lon before trusting this NavFix; skipping for now.",
            ping_index, nav_units,
        )
        return None

    return NavFix(frame_index=ping_index, lat=float(lat), lon=float(lon), heading_deg=float(heading) % 360.0)


def read_xtf_pings(path: str | Path) -> Iterator[PingRecord]:
    """Yields one PingRecord per sonar ping in the file, port+starboard
    channels concatenated into a single row (port reversed so the image
    reads left-to-right as port-far -> nadir -> starboard-far, matching the
    waterfall convention georeference.py assumes with port_is_left=True)."""
    pyxtf = _require_pyxtf()
    path = Path(path)

    file_header, packets = pyxtf.xtf_read(str(path))
    sonar_packets = packets.get(pyxtf.XTFHeaderType.sonar, [])
    if not sonar_packets:
        logger.warning("No sonar ping packets found in %s", path)
        return

    for i, ping in enumerate(sonar_packets):
        chans = getattr(ping, "data", None)
        if chans is None or len(chans) == 0:
            logger.warning("Ping %d has no channel data, skipping", i)
            continue
        if len(chans) == 1:
            row = np.asarray(chans[0])
        else:
            port = np.asarray(chans[0])[::-1]  # reversed so far-port is leftmost
            starboard = np.asarray(chans[1])
            row = np.concatenate([port, starboard])

        nav = _extract_nav_from_ping(pyxtf, ping, i)
        yield PingRecord(ping_index=i, intensity=row, nav=nav)


def build_waterfall_tiles(
    pings: Iterator[PingRecord], source_path: Path, tile_height: int = 512,
) -> Iterator[SSSRecord]:
    """Stacks consecutive pings into tile_height-row waterfall tiles. Each
    tile's metadata carries `nav_fixes_per_row` (one NavFix-or-None per row,
    in tile-local row order) so a detection can be geolocated using the
    exact ping it came from."""
    buffer: list[PingRecord] = []
    tile_idx = 0

    def _flush(buf: list[PingRecord]) -> SSSRecord:
        widths = {len(p.intensity) for p in buf}
        max_w = max(widths)
        rows = [
            np.pad(p.intensity, (0, max_w - len(p.intensity)), mode="constant")
            if len(p.intensity) < max_w else p.intensity
            for p in buf
        ]
        image = np.clip(np.stack(rows), 0, 255).astype(np.uint8) if np.max(rows) <= 255 else \
            _to_uint8(np.stack(rows))
        return SSSRecord(
            image=image,
            source_path=source_path,
            record_id=f"{source_path.stem}_tile{tile_idx:05d}",
            metadata={
                "source_format": "xtf",
                "ping_index_start": buf[0].ping_index,
                "ping_index_end": buf[-1].ping_index,
                "nav_fixes_per_row": [p.nav for p in buf],
            },
        )

    for ping in pings:
        buffer.append(ping)
        if len(buffer) >= tile_height:
            yield _flush(buffer)
            tile_idx += 1
            buffer = []

    if buffer:
        yield _flush(buffer)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def ingest_xtf(path: str | Path, tile_height: int = 512) -> Iterator[SSSRecord]:
    """Top-level entry point -- mirrors ingestion.ingest_directory()'s
    generator contract exactly, so the rest of the pipeline (preprocessing,
    YOLO, VAE) needs zero changes to accept XTF-sourced records."""
    path = Path(path)
    pings = read_xtf_pings(path)
    yield from build_waterfall_tiles(pings, source_path=path, tile_height=tile_height)

"""Pixel -> real-world lat/lon conversion for side-scan sonar detections.

STANDARD SSS GEOREFERENCING MODEL (what this implements): a side-scan
waterfall image has one row per ping and columns running across-track --
the tow vehicle's own position (nadir) sits at (or near) the image's
horizontal center, port returns on one side, starboard on the other. A
pixel's real-world position is the vehicle's nav fix for that ping/row,
offset sideways by the pixel's across-track distance, in the direction
perpendicular to the vehicle's heading.

  across_track_m = (pixel_col - center_col) * pixels_to_meters
  bearing = heading_deg + 90                  (if image is port(left)/starboard(right))
          = heading_deg - 90                  (if image is starboard(left)/port(right))
  new_lat, new_lon = offset(lat, lon, bearing, across_track_m)

KNOWN SIMPLIFICATIONS (stated explicitly, not hidden):
  - Uses a flat-earth / equirectangular local offset, not a full geodesic
    (see `offset_latlon`). This is standard practice for SSS swaths, which
    are tens to low-hundreds of meters wide -- error at that range is a
    matter of centimeters, not something worth pulling in a geodesy library
    for. If pyproj is installed, `offset_latlon_geodesic` gives the exact
    version instead; not used by default to keep the dependency optional.
  - No towfish "layback" correction (the horizontal offset between the
    GPS antenna position, usually on the tow vessel, and the actual towfish
    position behind/below it). If the nav fix already comes from the
    towfish itself (common for AUVs) this doesn't matter; if it's surface-
    vessel GPS with a long tow cable, detections will be systematically
    offset by the layback distance. There is no data in this pipeline today
    to correct for that (no cable-out / depth sensor reading is ingested).
    Flagged here rather than silently ignored.
  - `pixels_to_meters` (across-track ground resolution) must be supplied
    per-log. For XTF logs, pyxtf-parsed ping headers carry a slant/ground
    range field this can be derived from (see xtf_reader.py); for the
    image+sidecar path there is no way to know it without either the
    sidecar specifying it or the user providing the sonar's configured
    range setting -- there is no way to recover it from pixels alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.geolocation.nav import NavFix

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class GeoResult:
    lat: float
    lon: float
    method: str  # "nav_fix" | "placeholder"
    across_track_m: float
    layback_corrected: bool = False
    depth_m: float | None = None  # passed through from the nav fix, NOT computed here --
    # this module has no way to derive water depth from pixels/geometry, so it's simply
    # relayed when the nav fix happened to carry one (see NavFix.depth_m), and stays None
    # otherwise rather than being invented.


def offset_latlon(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Flat-earth local-tangent-plane offset. Good to sub-meter accuracy at
    SSS swath-width distances (tens to low-hundreds of meters); see module
    docstring for why a full geodesic isn't used by default."""
    bearing_rad = math.radians(bearing_deg)
    lat_rad = math.radians(lat)
    dlat = (distance_m * math.cos(bearing_rad)) / EARTH_RADIUS_M
    dlon = (distance_m * math.sin(bearing_rad)) / (EARTH_RADIUS_M * math.cos(lat_rad))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def offset_latlon_geodesic(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Exact geodesic offset via pyproj, if installed. Raises ImportError
    with a clear message if not -- callers should catch and fall back to
    offset_latlon() rather than hard-depending on this."""
    from pyproj import Geod  # optional dependency, see requirements.txt

    geod = Geod(ellps="WGS84")
    lon2, lat2, _ = geod.fwd(lon, lat, bearing_deg, distance_m)
    return lat2, lon2


def box_center_across_track_px(xyxy: list[float], image_width_px: int) -> float:
    """Signed pixel offset of a detection box's horizontal center from the
    image's center column (the assumed nadir/tow-track column)."""
    x1, _, x2, _ = xyxy
    center_x = (x1 + x2) / 2.0
    return center_x - (image_width_px / 2.0)


def geolocate_detection(
    xyxy: list[float],
    image_width_px: int,
    nav_fix: NavFix | None,
    pixels_to_meters: float,
    port_is_left: bool = True,
    use_geodesic: bool = False,
) -> GeoResult:
    """Convert one detection box to a lat/lon.

    If nav_fix is None (no nav data at all for this frame), returns a
    PLACEHOLDER result centered on (0, 0) rather than silently fabricating
    a plausible-looking coordinate -- callers/UI must check `.method` and
    render placeholder results distinctly (e.g. "location unknown"), never
    plot them on the map as if they were real.
    """
    if nav_fix is None:
        return GeoResult(lat=0.0, lon=0.0, method="placeholder", across_track_m=0.0)

    offset_px = box_center_across_track_px(xyxy, image_width_px)
    across_track_m = offset_px * pixels_to_meters
    # positive offset_px = right half of image
    side_sign = 1 if port_is_left else -1
    bearing = (nav_fix.heading_deg + 90.0 * side_sign) % 360.0

    offset_fn = offset_latlon_geodesic if use_geodesic else offset_latlon
    try:
        lat, lon = offset_fn(nav_fix.lat, nav_fix.lon, bearing, across_track_m)
    except ImportError:
        lat, lon = offset_latlon(nav_fix.lat, nav_fix.lon, bearing, across_track_m)

    return GeoResult(lat=lat, lon=lon, method="nav_fix", across_track_m=across_track_m,
                      depth_m=nav_fix.depth_m)

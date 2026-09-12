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

FOOTPRINT GEOMETRY (in addition to the existing center-point output):
  `geolocate_detection` also projects all 4 bbox corners into lat/lon,
  producing a small polygon -- the debris's actual real-world footprint,
  not just a single point -- via `footprint_polygon_latlon`. This is
  ADDITIVE: `GeoResult.lat`/`.lon`/`.method`/`.depth_m` are all unchanged,
  every existing caller keeps working untouched, and the footprint is
  simply `None` whenever there's no nav fix (same placeholder discipline
  as the point).

  Two more simplifications this introduces, on top of the ones above:
  - The along-track axis (image rows, i.e. distance in the direction of
    travel) is assumed to use the SAME ground resolution as the across-
    track axis (`pixels_to_meters`) unless `pixels_to_meters_along_track`
    is given explicitly. Nothing in this pipeline measures the two
    separately today -- this is a square-pixel assumption, not a
    calibrated fact, exactly like `pixels_to_meters` itself.
  - Which end of the box (`y1` vs `y2`) is "ahead" of the vehicle depends
    on this dataset/sensor's row-time ordering, which isn't recorded
    anywhere upstream. Defaults to `along_track_sign=1` (higher row index
    = further along track); flip it to -1 if a footprint comes out
    mirrored front-to-back against known ground truth.
  - The whole detection is treated as if it were geolocated from ONE nav
    fix (the same one the center point already uses), not a separate fix
    per corner row. For a single detection box (a small fraction of a
    frame's height) this is a reasonable approximation; it is NOT
    re-deriving a per-row fix the way XTF's per-ping nav_fixes_per_row
    could in principle support -- that would be a further refinement, not
    done here.
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
    footprint: list[tuple[float, float]] | None = None  # [(lat, lon), ...] closed ring (4
    # corners + first repeated), the debris's real-world footprint -- None for a placeholder
    # result, same discipline as lat/lon=(0, 0) above: never fabricate a shape with no nav data.


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


def _corner_latlon(nav_fix: NavFix, across_track_m: float, along_track_m: float,
                    port_is_left: bool, offset_fn) -> tuple[float, float]:
    """One bbox corner -> lat/lon. Composes two single-axis flat-earth
    offsets from the nav fix: along-track first (parallel to heading), then
    across-track (perpendicular to it) from that intermediate point. At SSS
    swath-width distances this composition is equivalent to a single 2D
    vector offset (see module docstring for the along/across simplifications
    this relies on)."""
    side_sign = 1 if port_is_left else -1
    lat1, lon1 = offset_fn(nav_fix.lat, nav_fix.lon, nav_fix.heading_deg, along_track_m)
    bearing_across = (nav_fix.heading_deg + 90.0 * side_sign) % 360.0
    return offset_fn(lat1, lon1, bearing_across, across_track_m)


def footprint_polygon_latlon(
    xyxy: list[float],
    image_width_px: int,
    image_height_px: int,
    nav_fix: NavFix,
    pixels_to_meters: float,
    pixels_to_meters_along_track: float | None = None,
    port_is_left: bool = True,
    along_track_sign: int = 1,
    use_geodesic: bool = False,
) -> list[tuple[float, float]]:
    """Projects all 4 corners of a detection's bbox into lat/lon -- the
    debris's real-world footprint, not just its center point. Returns a
    closed ring: [corner0, corner1, corner2, corner3, corner0], ready to
    drop straight into a GeoJSON Polygon's coordinates (after flipping each
    (lat, lon) pair to GeoJSON's [lon, lat] order -- see geojson_export.py).
    """
    along_scale = pixels_to_meters_along_track if pixels_to_meters_along_track is not None else pixels_to_meters
    x1, y1, x2, y2 = xyxy
    center_x, center_y = image_width_px / 2.0, image_height_px / 2.0
    corners_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # top-left, top-right, bottom-right, bottom-left

    offset_fn = offset_latlon_geodesic if use_geodesic else offset_latlon
    ring = []
    for px, py in corners_px:
        across_m = (px - center_x) * pixels_to_meters
        along_m = (py - center_y) * along_scale * along_track_sign
        try:
            ring.append(_corner_latlon(nav_fix, across_m, along_m, port_is_left, offset_fn))
        except ImportError:
            ring.append(_corner_latlon(nav_fix, across_m, along_m, port_is_left, offset_latlon))
    ring.append(ring[0])
    return ring


def geolocate_detection(
    xyxy: list[float],
    image_width_px: int,
    nav_fix: NavFix | None,
    pixels_to_meters: float,
    port_is_left: bool = True,
    use_geodesic: bool = False,
    image_height_px: int | None = None,
    pixels_to_meters_along_track: float | None = None,
    along_track_sign: int = 1,
) -> GeoResult:
    """Convert one detection box to a lat/lon POINT (its center), plus --
    when `image_height_px` is given -- a footprint POLYGON from all 4 bbox
    corners (see `footprint_polygon_latlon`). `image_height_px` is optional
    and keyword-only in effect (defaults to None) so every existing caller
    that only ever wanted the point keeps working completely unchanged;
    passing it is the only thing that turns footprint generation on.

    If nav_fix is None (no nav data at all for this frame), returns a
    PLACEHOLDER result centered on (0, 0) with footprint=None rather than
    silently fabricating a plausible-looking coordinate or shape --
    callers/UI must check `.method` and render placeholder results
    distinctly (e.g. "location unknown"), never plot them on the map as if
    they were real.
    """
    if nav_fix is None:
        return GeoResult(lat=0.0, lon=0.0, method="placeholder", across_track_m=0.0, footprint=None)

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

    footprint = None
    if image_height_px is not None:
        footprint = footprint_polygon_latlon(
            xyxy, image_width_px, image_height_px, nav_fix, pixels_to_meters,
            pixels_to_meters_along_track=pixels_to_meters_along_track,
            port_is_left=port_is_left, along_track_sign=along_track_sign, use_geodesic=use_geodesic,
        )

    return GeoResult(lat=lat, lon=lon, method="nav_fix", across_track_m=across_track_m,
                      depth_m=nav_fix.depth_m, footprint=footprint)

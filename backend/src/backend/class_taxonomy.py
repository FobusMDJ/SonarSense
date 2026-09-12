"""Maps whatever raw class_name a detection model emits onto the fixed
5-category dashboard taxonomy the reference SonarSense pitch deck and the
React frontend (FobusMDJ/SonarSense) both use -- Shipwreck ("Wreck"), Pipe,
Ghost Net ("Net"), Cylinder ("Rock"), Other Debris ("Other") -- independent
of which specific detection checkpoint produced the label.

WHY THIS EXISTS: the released weights-v1 best.pt checkpoint has exactly four
raw classes (aircraft, human, ship, pipe). Rather
than hard-coding the API's classification field to one checkpoint's exact
label set, every raw class_name -- from either model, or any future one --
gets funneled through CLASS_DISPLAY_MAP into the same fixed 5-category
taxonomy the dashboard renders (donut chart, map legend, filters), while the
RAW label is preserved alongside it in every API response for anyone who
wants the specific one. A class this map hasn't seen before (e.g.
'aircraft') lands in 'Other Debris' rather than erroring -- exactly what the
reference deck's own 'Other' bucket (icon: '?') is for. Specifically,
ship maps to Shipwreck and aircraft maps to Other Debris. Ghost Net and
Cylinder remain valid historical dashboard groups but are not claimed as
classes supported by this released detector.

HUMAN DETECTIONS: 'human' is not one of the 5 dashboard debris categories
(a person in the water is not debris), but detections of it are NOT
discarded -- discarding a real person-overboard detection would be a safety
regression dressed up as a UI simplification. They're still run through the
full pipeline, scored, geolocated, and stored; is_human_class() lets callers
(src/backend/main.py's /api/* endpoints) filter them out of the normal
debris-classification list while still surfacing them via a separate
endpoint for safety review.
"""

from __future__ import annotations

from typing import Optional

DISPLAY_CLASSES = ["Shipwreck", "Pipe", "Ghost Net", "Cylinder", "Other Debris"]

# Raw model class_name (lower/snake_case, whatever a checkpoint's own class
# list uses) -> the fixed dashboard display classification. Extend this as
# new checkpoints get wired in -- it's the ONLY place that needs to change.
CLASS_DISPLAY_MAP: dict[str, str] = {
    "shipwreck": "Shipwreck",
    "wreck": "Shipwreck",
    "ship": "Shipwreck",
    "pipe": "Pipe",
    "ghost_net": "Ghost Net",
    "ghost net": "Ghost Net",
    "net": "Ghost Net",
    "cylinder": "Cylinder",
    "rock": "Cylinder",
}

HUMAN_CLASS_NAMES = {"human", "person"}


def _normalize(class_name: str) -> str:
    return class_name.strip().lower().replace("-", "_")


def is_human_class(class_name: str) -> bool:
    return _normalize(class_name) in HUMAN_CLASS_NAMES


def display_classification(class_name: str) -> str:
    """Raw model class_name -> one of DISPLAY_CLASSES. Human detections
    still get mapped (callers that need to exclude them should check
    is_human_class() first, per the module docstring) rather than this
    function raising or guessing -- 'Other Debris' is the deliberate,
    documented fallback for anything outside the 5-category taxonomy,
    including a future model's classes this map hasn't been told about yet."""
    key = _normalize(class_name)
    if key in HUMAN_CLASS_NAMES:
        return "Human"
    return CLASS_DISPLAY_MAP.get(key, "Other Debris")


# Matches classSymbols in the frontend's mapDataAdapter.ts exactly -- kept in
# sync by hand since they live in two repos; if you add a display class here,
# add its symbol there too (and vice versa).
CLASS_SYMBOLS: dict[str, str] = {
    "Ghost Net": "GN", "Pipe": "PI", "Shipwreck": "SW", "Cylinder": "CY", "Other Debris": "OD",
}

# Dashboard priority thresholds. Reuses the same confidence_score the
# existing confidence_label ('high'/'medium'/'low') already uses (see
# src/confidence/scoring.py) -- LOW_CONFIDENCE_THRESHOLD there is 40.0,
# matched here as PRIORITY_MEDIUM_THRESHOLD so "Priority: Low" in the
# dashboard and "confidence_label: low" in the raw API always agree about
# which detections they're pointing at, rather than drifting into two
# almost-but-not-quite-matching thresholds defined in two places.
PRIORITY_HIGH_THRESHOLD = 70.0
PRIORITY_MEDIUM_THRESHOLD = 40.0


def priority_from_confidence(confidence_score: float) -> str:
    if confidence_score >= PRIORITY_HIGH_THRESHOLD:
        return "High"
    if confidence_score >= PRIORITY_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


# Rough per-class height defaults (meters) -- used ONLY to fill the
# frontend's dimensions{length,width,height} type. Side-scan sonar is a 2D
# top-down projection; there is no 3rd dimension recoverable from a single
# frame the way length/width are (from the YOLO bbox * pixels_to_meters,
# both real measurements). height below is a documented placeholder, not a
# measurement -- every API response that includes it also sets
# dimensions_estimated: true so a consumer can't mistake it for real data.
ESTIMATED_HEIGHT_M: dict[str, float] = {
    "Shipwreck": 8.0, "Pipe": 0.5, "Ghost Net": 0.3, "Cylinder": 1.0, "Other Debris": 1.0, "Human": 1.7,
}


def bbox_dimensions_m(bbox_x1: float, bbox_y1: float, bbox_x2: float, bbox_y2: float,
                       pixels_to_meters: Optional[float], display_class: str) -> dict:
    """length/width are real (bbox size * pixels_to_meters, when that's
    known); height is always the ESTIMATED_HEIGHT_M placeholder -- see
    module note above. Returns None length/width (not 0.0) when
    pixels_to_meters is unavailable, so a consumer can't mistake "unknown"
    for "measured zero."""
    if pixels_to_meters is None:
        length = width = None
    else:
        length = round(abs(bbox_x2 - bbox_x1) * pixels_to_meters, 3)
        width = round(abs(bbox_y2 - bbox_y1) * pixels_to_meters, 3)
    height = ESTIMATED_HEIGHT_M.get(display_class, 1.0)
    return {"length": length, "width": width, "height": height}


def resolved_dimensions_m(detection: dict, pixels_to_meters: Optional[float]) -> dict:
    """The single place every caller (GET /logs/{id}/detections, the JSON/CSV/
    SQL report writers) should go for a detection's length/width/height --
    NOT bbox_dimensions_m directly, which is a fallback this function calls
    for you when needed.

    PREFERS a detection's own stored length_m/width_m columns, when present,
    over the generic bbox * pixels_to_meters guess below. Those columns are
    only populated at insert time by an ingestion path that computed real-
    world size itself from actual sonar geometry -- see
    src/geolocation/csv_engine.py's geolocate_csv_detection, which derives
    width from the across-track bbox extent and length from the along-track
    extent using the same slant-range math it uses for lat/lon, not a bare
    pixel-count heuristic. The image/YOLO pipeline has no equivalent physics
    -based sizing step yet, so its detections have NULL length_m/width_m at
    insert time and fall through to bbox_dimensions_m here, unchanged from
    before this function existed.

    Always returns a dict shaped like bbox_dimensions_m's ({length, width,
    height}); height has no measured equivalent in either ingestion path and
    is always the ESTIMATED_HEIGHT_M placeholder."""
    display_class = display_classification(detection["class_name"])
    stored_length = detection.get("length_m")
    stored_width = detection.get("width_m")
    if stored_length is not None and stored_width is not None:
        return {
            "length": stored_length,
            "width": stored_width,
            "height": ESTIMATED_HEIGHT_M.get(display_class, 1.0),
        }
    return bbox_dimensions_m(
        detection["bbox_x1"], detection["bbox_y1"], detection["bbox_x2"], detection["bbox_y2"],
        pixels_to_meters, display_class,
    )

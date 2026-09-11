"""Frontend-shaped REST surface: GET /api/surveys, /api/surveys/{id},
/api/surveys/{id}/detections, /api/surveys/{id}/summary,
/api/surveys/{id}/humans -- these return exactly the shapes the React
frontend (FobusMDJ/SonarSense, src/features/intelligence-map/types.ts)
expects, so Task #21's frontend wiring is a straight fetch-and-render with
no client-side reshaping.

ADDITIVE, NOT A REPLACEMENT: the existing /logs/* REST API (main.py) is
untouched -- this module is mounted alongside it under an /api prefix.
The already-delivered demo frontend (frontend/index.html) depends on the
/logs/* shapes and keeps working exactly as before; this router is for the
FobusMDJ/SonarSense React app specifically.

Uses db_backend (not db.py directly) so these endpoints work identically
against SQLite or PostGIS -- see db_backend.py's module docstring. Assumes
db_backend.configure(cfg, project_root) has already been called (main.py
does this at import time, before include_router()), so get_connection()
here is called with no target arg and picks up the configured default.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from src.backend import class_taxonomy, db_backend as db
from src.backend.schemas import ApiDetection, ApiDimensions, ApiSurvey, ApiSurveySummary

router = APIRouter(prefix="/api", tags=["frontend-api"])


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------

def _require_done_log(survey_id: str) -> dict:
    with db.get_connection() as conn:
        row = db.get_log(conn, survey_id)
    if row is None:
        raise HTTPException(404, f"Survey {survey_id} not found")
    if row["status"] != "done":
        # Survey's `status` field is the literal 'Completed' in the frontend
        # type -- an in-progress/errored log has nowhere valid to go in
        # this shape, so it 404s here rather than lying about being done.
        # Poll GET /logs/{id} (the general-purpose, non-frontend-typed
        # status endpoint) for logs that aren't finished yet.
        raise HTTPException(409, f"Survey {survey_id} is not complete yet (status={row['status']!r})")
    return row


def _row_to_api_detection(d: dict, pixels_to_meters: Optional[float]) -> ApiDetection:
    display_class = class_taxonomy.display_classification(d["class_name"])
    dims = class_taxonomy.bbox_dimensions_m(
        d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"], pixels_to_meters, display_class,
    )
    dims_estimated = dims["length"] is None or dims["width"] is None
    depth_m = d.get("depth_m")

    return ApiDetection(
        id=d["id"],
        surveyId=d["log_id"],
        classification=display_class,
        rawClassName=d["class_name"],
        priority=class_taxonomy.priority_from_confidence(d["confidence_score"]).upper(),
        confidence=round(d["confidence_score"]),
        dimensions=ApiDimensions(
            length=dims["length"] or 0.0,
            width=dims["width"] or 0.0,
            height=dims["height"],
            dimensionsEstimated=dims_estimated,
        ),
        pingId=d["frame_record_id"],
        timestamp=d["created_at"],
        depth=depth_m if depth_m is not None else 0.0,
        depthAvailable=depth_m is not None,
        coordinates=(d["lon"], d["lat"]),
    )


def _survey_track(detections: list[dict]) -> list[tuple[float, float]]:
    """Best-effort survey path from real nav-fix'd detections, in frame
    order -- see ApiSurvey.track_coordinates' docstring for why this is an
    approximation, not the true continuous tow track."""
    located = [d for d in detections if d.get("geo_method") == "nav_fix"
               and d.get("lat") is not None and d.get("lon") is not None]
    located.sort(key=lambda d: d["frame_index"])
    track: list[tuple[float, float]] = []
    for d in located:
        point = (d["lon"], d["lat"])
        if not track or track[-1] != point:
            track.append(point)
    return track


def _survey_region(track: list[tuple[float, float]]) -> str:
    if not track:
        return "Unknown (no located detections yet)"
    lons = [p[0] for p in track]
    lats = [p[1] for p in track]
    return f"{min(lats):.3f}–{max(lats):.3f}°N, {min(lons):.3f}–{max(lons):.3f}°E"


def _log_to_survey(log: dict, all_detections: list[dict]) -> ApiSurvey:
    track = _survey_track(all_detections)
    return ApiSurvey(
        id=log["id"],
        name=log["filename"],
        platform="Side-scan sonar tow system",  # generic descriptor, not measured telemetry --
        # this pipeline has no source for vehicle/platform identity (AUV vs. towed vs. ROV).
        status="Completed",
        region=_survey_region(track),
        startedAt=log["uploaded_at"],
        completedAt=log["completed_at"] or log["uploaded_at"],
        trackCoordinates=track,
    )


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

@router.get("/surveys", response_model=list[ApiSurvey])
def list_surveys() -> list[ApiSurvey]:
    with db.get_connection() as conn:
        logs = [row for row in db.list_logs(conn) if row["status"] == "done"]
        out = []
        for log in logs:
            dets = db.list_detections(conn, log["id"])
            out.append(_log_to_survey(log, dets))
        return out


@router.get("/surveys/{survey_id}", response_model=ApiSurvey)
def get_survey(survey_id: str) -> ApiSurvey:
    log = _require_done_log(survey_id)
    with db.get_connection() as conn:
        dets = db.list_detections(conn, survey_id)
    return _log_to_survey(log, dets)


@router.get("/surveys/{survey_id}/detections", response_model=list[ApiDetection])
def get_survey_detections(
    survey_id: str,
    min_confidence: Optional[float] = None,
    include_unlocated: bool = False,
) -> list[ApiDetection]:
    """Non-human detections only (see class_taxonomy.py's module docstring
    -- human detections are never silently dropped from the pipeline, just
    excluded from this normal debris list; use /humans below for those).
    Placeholder-geolocated detections (geo_method='placeholder', no real
    nav fix) are excluded by default since they'd plot at (0, 0) on a
    map -- pass include_unlocated=true to get them anyway."""
    log = _require_done_log(survey_id)
    with db.get_connection() as conn:
        rows = db.list_detections(conn, survey_id, min_confidence=min_confidence)
    pixels_to_meters = log.get("pixels_to_meters")
    out = []
    for d in rows:
        if class_taxonomy.is_human_class(d["class_name"]):
            continue
        if not include_unlocated and d.get("geo_method") != "nav_fix":
            continue
        out.append(_row_to_api_detection(d, pixels_to_meters))
    return out


@router.get("/surveys/{survey_id}/humans", response_model=list[ApiDetection])
def get_survey_human_detections(survey_id: str) -> list[ApiDetection]:
    """Human detections for this survey -- kept OUT of the normal debris
    list (see class_taxonomy.py) but never discarded; surfaced here for
    safety review. `classification` on these will read "Human", not one of
    the 5 debris display categories."""
    log = _require_done_log(survey_id)
    with db.get_connection() as conn:
        rows = db.list_detections(conn, survey_id)
    pixels_to_meters = log.get("pixels_to_meters")
    return [
        _row_to_api_detection(d, pixels_to_meters)
        for d in rows if class_taxonomy.is_human_class(d["class_name"])
    ]


@router.get("/surveys/{survey_id}/summary", response_model=ApiSurveySummary)
def get_survey_summary(survey_id: str) -> ApiSurveySummary:
    detections = get_survey_detections(survey_id)  # excludes humans/unlocated, same as the frontend's
    # own getSurveySummary(detections) in mapDataAdapter.ts -- computed the same way here so a
    # consumer that only wants the summary (e.g. the PDF report) doesn't have to fetch every detection.
    if not detections:
        return ApiSurveySummary(totalDetections=0, highPriorityCount=0, averageConfidence=0)
    total = len(detections)
    high = sum(1 for d in detections if d.priority == "HIGH")
    avg = round(sum(d.confidence for d in detections) / total)
    return ApiSurveySummary(totalDetections=total, highPriorityCount=high, averageConfidence=avg)

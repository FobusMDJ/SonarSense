"""Pydantic response/request models for the FastAPI layer -- this is the
API contract a frontend integrates against, kept in one file so it's easy
to hand to a frontend developer as the source of truth."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LogSummary(BaseModel):
    id: str
    filename: str
    source_format: str
    status: str
    uploaded_at: str
    completed_at: Optional[str] = None
    n_frames: int
    n_detections: int
    denoise_method: Optional[str] = None
    contrast_method: Optional[str] = None
    error_message: Optional[str] = None


class UploadResponse(BaseModel):
    log_id: str
    status: str
    message: str


class LocalIngestRequest(BaseModel):
    """Body for POST /logs/ingest_local -- process a directory that ALREADY
    exists on the server's own filesystem, no upload involved. Paths are
    resolved relative to the project root if not absolute."""
    source_dir: str
    nav_sidecar_path: Optional[str] = None
    pixels_to_meters: Optional[float] = None
    yolo_conf: Optional[float] = None
    denoise_method: Optional[str] = None
    contrast_method: Optional[str] = None


class Detection(BaseModel):
    id: str
    log_id: str
    frame_index: int
    frame_record_id: str
    frame_image_path: Optional[str] = None
    class_name: str
    yolo_conf: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    confidence_score: float
    confidence_label: str
    confidence_breakdown: Optional[dict] = None
    vae_box_error: Optional[float] = None
    vae_whole_image_error: Optional[float] = None
    vae_whole_image_percentile: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    geo_method: Optional[str] = None
    depth_m: Optional[float] = None  # water depth below surface, from the nav fix if it carried one
    # (see src/geolocation/nav.py's NavFix.depth_m) -- null, never fabricated, when absent.
    footprint_geojson: Optional[list[list[float]]] = None  # [[lon, lat], ...] closed ring -- the
    # debris's real-world 4-corner footprint (see src/geolocation/georeference.py's
    # GeoResult.footprint), null under the exact same rule as lat/lon: no nav fix, no shape.
    vae_panel_dir: Optional[str] = None
    created_at: str
    length_m: Optional[float] = None  # bbox size (across-track) * this log's pixels_to_meters --
    width_m: Optional[float] = None   # null (not 0.0) when pixels_to_meters is unknown for this log,
    # so a consumer can't mistake "not measured" for "measured zero" (see class_taxonomy.
    # bbox_dimensions_m, the same helper the /api/surveys/* layer's ApiDimensions already used).
    height_m: Optional[float] = None  # ALWAYS class_taxonomy.ESTIMATED_HEIGHT_M's placeholder --
    # a 2D side-scan frame has no height axis to measure directly.
    dimensions_estimated: bool = False  # true when length_m/width_m themselves are null
    # (pixels_to_meters unknown for this log) -- height_m is a placeholder either way.


class ModelOutputStats(BaseModel):
    """The "model output stats" dashboard area: aggregate YOLO detection stats for one log."""
    log_id: str
    n_detections: int
    counts_by_class: dict[str, int]
    raw_counts_by_class: dict[str, int]
    mean_confidence_by_class: dict[str, float]
    n_low_confidence: int
    low_confidence_threshold: float
    confidence_threshold: Optional[float] = None
    mean_inference_ms: Optional[float] = None
    fps: Optional[float] = None


class VaeStats(BaseModel):
    """The "VAE stats" dashboard area: whole-image anomaly-error distribution for one log."""
    log_id: str
    n_frames_analyzed: int
    mean_whole_image_error: float
    min_whole_image_error: float
    max_whole_image_error: float
    most_anomalous_frames: list[dict]  # [{frame_record_id, whole_image_error, percentile}]


class ApiDimensions(BaseModel):
    """Matches the frontend's DetectionDimensions (intelligence-map/types.ts)
    exactly: length/width/height, all required numbers. length/width come
    from the YOLO bbox * pixels_to_meters (real, when pixels_to_meters is
    known for the log); height has no equivalent measurement in a 2D
    side-scan frame and is always class_taxonomy.ESTIMATED_HEIGHT_M's
    placeholder -- see dimensionsEstimated below, which the current
    frontend type doesn't have a field for yet (added here so a future
    frontend update can surface it; harmless extra key until then)."""
    model_config = ConfigDict(populate_by_name=True)

    length: float
    width: float
    height: float
    dimensions_estimated: bool = Field(alias="dimensionsEstimated")


class ApiDetection(BaseModel):
    """Matches the frontend's Detection type (intelligence-map/types.ts)
    field-for-field via camelCase aliases, PLUS a few additive fields
    (rawClassName, depthAvailable) the current frontend type doesn't
    declare yet -- extra JSON keys are ignored by existing frontend code
    until types.ts is updated to use them (see Task #21), so this is
    forward-compatible rather than a breaking change.

    Human detections (class_taxonomy.is_human_class) are NEVER returned
    from GET /api/surveys/{id}/detections -- they're still detected,
    scored, and geolocated by the pipeline (see class_taxonomy.py's module
    docstring on why), just surfaced instead via the separate
    GET /api/surveys/{id}/humans endpoint for safety review."""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    survey_id: str = Field(alias="surveyId")
    classification: str  # one of class_taxonomy.DISPLAY_CLASSES (or "Human", only via /humans)
    raw_class_name: str = Field(alias="rawClassName")  # the exact label the detection model emitted
    priority: str  # "HIGH" | "MEDIUM" | "LOW" -- uppercase, matching the frontend's Priority type
    confidence: float
    dimensions: ApiDimensions
    ping_id: str = Field(alias="pingId")
    timestamp: str
    depth: float  # 0.0 when unknown -- see depth_available
    depth_available: bool = Field(alias="depthAvailable")  # false => `depth` above is a filler
    # zero, not a measurement (see NavFix.depth_m's "never fabricated" convention); the current
    # frontend Detection type has no field for this yet, so today's UI just shows "0.0 m" for
    # those rows until types.ts/ModelOutput-style components are updated to check this flag.
    coordinates: tuple[float, float]  # [lon, lat], matching Coordinates = [longitude, latitude]


class ApiSurvey(BaseModel):
    """Matches the frontend's Survey type (intelligence-map/types.ts).
    Only logs with status == 'done' are exposed here -- the type's
    `status` field is the literal 'Completed', so an in-progress log has
    nowhere to go in this shape yet (see GET /logs/{id} for the general-
    purpose status of any log, done or not)."""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    platform: str
    status: str = "Completed"
    region: str
    started_at: str = Field(alias="startedAt")
    completed_at: str = Field(alias="completedAt")
    track_coordinates: list[tuple[float, float]] = Field(alias="trackCoordinates")
    # ^ APPROXIMATION: this pipeline does not persist the full continuous
    # tow-track nav path per log (only per-detection nav fixes survive to
    # the DB) -- track_coordinates is built from real detections' own
    # nav-fix coordinates, in frame order, which traces the general survey
    # path but will look sparser/more angular than a true continuous GPS
    # track. Documented here rather than silently presented as more
    # precise than it is.


class ApiSurveySummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_detections: int = Field(alias="totalDetections")
    high_priority_count: int = Field(alias="highPriorityCount")
    average_confidence: float = Field(alias="averageConfidence")


class ProgressEvent(BaseModel):
    """One WebSocket message pushed while a log is being processed."""
    log_id: str
    stage: str  # 'ingest' | 'preprocess' | 'detect' | 'vae' | 'confidence' | 'geolocate' | 'persist' | 'done' | 'error'
    frame_index: Optional[int] = None
    n_frames_total: Optional[int] = None
    message: str

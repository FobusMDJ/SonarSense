"""Pydantic response/request models for the FastAPI layer -- this is the
API contract a frontend integrates against, kept in one file so it's easy
to hand to a frontend developer as the source of truth."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


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
    vae_panel_dir: Optional[str] = None
    created_at: str


class ModelOutputStats(BaseModel):
    """The "model output stats" dashboard area: aggregate YOLO detection stats for one log."""
    log_id: str
    n_detections: int
    counts_by_class: dict[str, int]
    mean_confidence_by_class: dict[str, float]
    n_low_confidence: int
    low_confidence_threshold: float


class VaeStats(BaseModel):
    """The "VAE stats" dashboard area: whole-image anomaly-error distribution for one log."""
    log_id: str
    n_frames_analyzed: int
    mean_whole_image_error: float
    min_whole_image_error: float
    max_whole_image_error: float
    most_anomalous_frames: list[dict]  # [{frame_record_id, whole_image_error, percentile}]


class ProgressEvent(BaseModel):
    """One WebSocket message pushed while a log is being processed."""
    log_id: str
    stage: str  # 'ingest' | 'preprocess' | 'detect' | 'vae' | 'confidence' | 'geolocate' | 'persist' | 'done' | 'error'
    frame_index: Optional[int] = None
    n_frames_total: Optional[int] = None
    message: str

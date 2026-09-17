// Real API client for the SonarSense FastAPI backend. Local development
// overrides this through .env.local; Vercel uses the deployed Render API
// fallback when no project-level environment variable has been configured.
const PRODUCTION_API_BASE = 'https://sonarsense-backend-ezkh.onrender.com'
const RAW_BASE = (import.meta.env.VITE_API_BASE_URL ?? PRODUCTION_API_BASE).trim()
export const API_BASE = RAW_BASE.replace(/\/+$/, '')

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`
  let res: Response
  try {
    res = await fetch(url, init)
  } catch (err) {
    throw new ApiError(0, `Could not reach the backend at ${API_BASE || '(no VITE_API_BASE_URL set)'}. ${(err as Error).message}`)
  }
  if (!res.ok) {
    let detail = ''
    try { detail = (await res.json())?.detail ?? '' } catch { /* ignore -- not JSON */ }
    throw new ApiError(res.status, detail || `${res.status} ${res.statusText} on ${path}`)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

// --------------------------------------------------------------------------
// Raw /logs/* shapes (src/backend/schemas.py) -- the internals-rich API,
// used everywhere this frontend needs pipeline-stage detail (VAE errors,
// frame ids, confidence breakdowns) that the display-safe /api/surveys/*
// layer deliberately omits.
// --------------------------------------------------------------------------

export interface LogSummary {
  id: string
  filename: string
  source_format: string
  status: 'processing' | 'done' | 'error' | string
  uploaded_at: string
  completed_at: string | null
  n_frames: number
  n_detections: number
  denoise_method: string | null
  contrast_method: string | null
  error_message: string | null
}

export interface RawDetection {
  id: string
  log_id: string
  frame_index: number
  frame_record_id: string
  frame_image_path: string | null
  class_name: string
  yolo_conf: number
  bbox_x1: number
  bbox_y1: number
  bbox_x2: number
  bbox_y2: number
  confidence_score: number
  confidence_label: 'high' | 'medium' | 'low' | string
  confidence_breakdown: Record<string, number> | null
  vae_box_error: number | null
  vae_whole_image_error: number | null
  vae_whole_image_percentile: number | null
  lat: number | null
  lon: number | null
  geo_method: string | null
  depth_m: number | null
  vae_panel_dir: string | null
  created_at: string
  length_m: number | null
  width_m: number | null
  height_m: number | null
  dimensions_estimated: boolean
}

export interface ModelOutputStats {
  log_id: string
  n_detections: number
  counts_by_class: Record<string, number>
  raw_counts_by_class: Record<string, number>
  mean_confidence_by_class: Record<string, number>
  n_low_confidence: number
  low_confidence_threshold: number
  confidence_threshold: number | null
  mean_inference_ms: number | null
  fps: number | null
}

export interface CheckpointMetadata {
  filename: string
  size_bytes: number
  sha256: string
  path: string
  found: boolean
  verified: boolean
  actual_sha256: string | null
}

export interface ModelMetadata {
  release_tag: string
  device: string
  detector: {
    name: string
    framework: string
    framework_version: string
    input_size: number
    training_epochs: number
    training_images: number | null
    raw_classes: string[]
    training_metrics: {
      precision: number
      recall: number
      f1_score: number
      map50: number
      map50_95: number
    }
    checkpoint: CheckpointMetadata
  }
  vae: {
    epoch: number
    input_size: number
    latent_dim: number
    channels: number[]
    checkpoint: CheckpointMetadata
  }
  denoiser: {
    name: string
    status: 'experimental'
    epoch: number
    base_width: number
    checkpoint: CheckpointMetadata
  }
}

export interface VaeFrameStat {
  frame_record_id: string
  whole_image_error: number | null
  percentile: number | null
}

export interface VaeStats {
  log_id: string
  n_frames_analyzed: number
  mean_whole_image_error: number
  min_whole_image_error: number
  max_whole_image_error: number
  most_anomalous_frames: VaeFrameStat[]
}

export interface UploadResponse {
  log_id: string
  status: string
  message: string
}

export interface ProgressEvent {
  log_id: string
  stage: 'ingest' | 'preprocess' | 'detect' | 'vae' | 'confidence' | 'geolocate' | 'persist' | 'done' | 'error' | 'closed' | string
  frame_index?: number
  n_frames_total?: number
  message: string
}

export const VAE_PANEL_FILES = [
  '01_original.png',
  '02_reconstruction.png',
  '03_anomaly_overlay.png',
  '04_difference_heatmap.png',
  '05_edge_contour_map.png',
  '06_difference_map_legend.png',
  '07_3d_anomaly_surface.png',
] as const

export function listLogs(): Promise<LogSummary[]> {
  return request('/logs')
}

export function getLog(logId: string): Promise<LogSummary> {
  return request(`/logs/${logId}`)
}

export function getDetections(logId: string, minConfidence?: number): Promise<RawDetection[]> {
  const qs = minConfidence != null ? `?min_confidence=${minConfidence}` : ''
  return request(`/logs/${logId}/detections${qs}`)
}

export function getModelStats(logId: string): Promise<ModelOutputStats> {
  return request(`/logs/${logId}/stats`)
}

export function getModelMetadata(): Promise<ModelMetadata> {
  return request('/model/metadata')
}

export function getVaeStats(logId: string): Promise<VaeStats> {
  return request(`/logs/${logId}/vae_stats`)
}

/** Direct (non-fetch) URL to one of the 7 saved VAE visualization PNGs for
 * one frame -- used as an <img src>, not fetched via request() above. */
export function vaeFrameImageUrl(logId: string, frameRecordId: string, filename: string): string {
  return `${API_BASE}/logs/${logId}/frames/${frameRecordId}/vae/${filename}`
}

export interface VaeSurfaceData {
  size: number
  values: number[]
}

export function getVaeSurfaceData(logId: string, frameRecordId: string): Promise<VaeSurfaceData> {
  return request(`/logs/${logId}/frames/${encodeURIComponent(frameRecordId)}/vae/surface`)
}

export function reportUrls(logId: string) {
  return {
    json: `${API_BASE}/logs/${logId}/report.json`,
    csv: `${API_BASE}/logs/${logId}/report.csv`,
    geojson: `${API_BASE}/logs/${logId}/report.geojson`,
    sql: `${API_BASE}/logs/${logId}/report.sql`,
    pdf: `${API_BASE}/logs/${logId}/report.pdf`,
    mapPng: `${API_BASE}/logs/${logId}/map.png`,
  }
}

export interface HealthResponse {
  status: string
  device: string
  models: Record<string, { path: string; found: boolean; note?: string }>
  defaults: { denoise_method: string; contrast_method: string }
  xtf: Record<string, unknown>
}

export function getHealth(): Promise<HealthResponse> {
  return request('/health')
}

export interface UploadOptions {
  pixelsToMeters?: number
  yoloConf?: number
  denoiseMethod?: 'none' | 'lee' | 'blind2unblind'
  contrastMethod?: 'none' | 'clahe' | 'histeq'
  navSidecar?: File | null
}

export async function uploadLog(file: File, opts: UploadOptions = {}): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  if (opts.navSidecar) form.append('nav_sidecar', opts.navSidecar)
  const params = new URLSearchParams()
  if (opts.pixelsToMeters != null) params.set('pixels_to_meters', String(opts.pixelsToMeters))
  if (opts.yoloConf != null) params.set('yolo_conf', String(opts.yoloConf))
  if (opts.denoiseMethod) params.set('denoise_method', opts.denoiseMethod)
  if (opts.contrastMethod) params.set('contrast_method', opts.contrastMethod)
  const qs = params.toString() ? `?${params.toString()}` : ''
  return request(`/logs/upload${qs}`, { method: 'POST', body: form })
}

export async function uploadSurveyZip(archive: File, opts: Omit<UploadOptions, 'navSidecar'> = {}): Promise<UploadResponse> {
  const form = new FormData()
  form.append('archive', archive)
  const params = new URLSearchParams()
  if (opts.pixelsToMeters != null) params.set('pixels_to_meters', String(opts.pixelsToMeters))
  if (opts.yoloConf != null) params.set('yolo_conf', String(opts.yoloConf))
  if (opts.denoiseMethod) params.set('denoise_method', opts.denoiseMethod)
  if (opts.contrastMethod) params.set('contrast_method', opts.contrastMethod)
  const qs = params.toString() ? `?${params.toString()}` : ''
  return request(`/logs/upload_zip${qs}`, { method: 'POST', body: form })
}

/** Opens the log's live progress WebSocket. Every callback is optional;
 * the socket closes itself once the backend sends stage 'done' | 'error' | 'closed'. */
export function watchProgress(logId: string, onEvent: (event: ProgressEvent) => void, onClose?: () => void): () => void {
  const wsBase = API_BASE.replace(/^http/, 'ws') || `ws://${window.location.host}`
  const ws = new WebSocket(`${wsBase}/ws/logs/${logId}`)
  ws.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data) as ProgressEvent
      onEvent(event)
      if (event.stage === 'done' || event.stage === 'error' || event.stage === 'closed') ws.close()
    } catch { /* ignore malformed frame */ }
  }
  ws.onclose = () => onClose?.()
  ws.onerror = () => onClose?.()
  return () => ws.close()
}

// --------------------------------------------------------------------------
// Display-safe /api/surveys/* shapes (src/backend/api_routes.py /
// src/backend/schemas.py's ApiSurvey/ApiDetection/ApiSurveySummary) -- used
// by the Intelligence Map, which wants the already-classified, already-
// filtered (no humans, no placeholder 0,0 coordinates) survey view.
// --------------------------------------------------------------------------

export type DebrisClass = 'Ghost Net' | 'Pipe' | 'Shipwreck' | 'Cylinder' | 'Other Debris'
export type ApiPriority = 'HIGH' | 'MEDIUM' | 'LOW'

export interface ApiSurvey {
  id: string
  name: string
  platform: string
  status: 'Completed'
  region: string
  startedAt: string
  completedAt: string
  trackCoordinates: [number, number][]
}

export interface ApiDimensions {
  length: number
  width: number
  height: number
  dimensionsEstimated: boolean
}

export interface ApiDetection {
  id: string
  surveyId: string
  classification: DebrisClass | 'Human'
  rawClassName: string
  priority: ApiPriority
  confidence: number
  dimensions: ApiDimensions
  pingId: string
  timestamp: string
  depth: number
  depthAvailable: boolean
  coordinates: [number, number]
}

export interface ApiSurveySummary {
  totalDetections: number
  highPriorityCount: number
  averageConfidence: number
}

export function listSurveys(): Promise<ApiSurvey[]> {
  return request('/api/surveys')
}

export function getSurvey(surveyId: string): Promise<ApiSurvey> {
  return request(`/api/surveys/${surveyId}`)
}

export function getSurveyDetections(surveyId: string, includeUnlocated = false): Promise<ApiDetection[]> {
  return request(`/api/surveys/${surveyId}/detections?include_unlocated=${includeUnlocated}`)
}

export function getSurveyHumans(surveyId: string): Promise<ApiDetection[]> {
  return request(`/api/surveys/${surveyId}/humans`)
}

export function getSurveySummary(surveyId: string): Promise<ApiSurveySummary> {
  return request(`/api/surveys/${surveyId}/summary`)
}

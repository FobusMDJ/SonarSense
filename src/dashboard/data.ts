import type { RawDetection, LogSummary } from '../lib/api'

export const categories = [
  { name: 'Pipe', color: '#4c80df' },
  { name: 'Ghost Net', color: '#5daa72' },
  { name: 'Shipwreck', color: '#e6635b' },
  { name: 'Cylinder', color: '#ecac48' },
  { name: 'Other Debris', color: '#8862c6' },
] as const
export type DebrisType = typeof categories[number]['name']
export type Priority = 'High' | 'Medium' | 'Low'

export interface DetectionRecord {
  id: string
  type: DebrisType
  confidence: number
  depth: number
  latitude: number
  longitude: number
  timestamp: string
  priority: Priority
  mission: string // the log this detection came from (LogSummary.filename)
  logId: string
  frameRecordId: string
  located: boolean // false when geo_method !== 'nav_fix' (no real GPS fix for this detection)
  rawClassName: string
  lengthM: number | null // across-track bbox size * this log's pixels_to_meters -- null (not 0)
  widthM: number | null  // when pixels_to_meters is unknown for this log, see dimensionsEstimated
  heightM: number | null // always a placeholder estimate -- a 2D side-scan frame has no height axis
  dimensionsEstimated: boolean // true when lengthM/widthM are null (pixels_to_meters unknown)
}

// Raw model class_name -> the fixed 5-category dashboard taxonomy. MUST be
// kept in sync by hand with src/backend/class_taxonomy.py's
// CLASS_DISPLAY_MAP -- same convention that file's own docstring already
// documents for its CLASS_SYMBOLS constant (two repos, one small lookup
// table, updated in both places together).
const CLASS_DISPLAY_MAP: Record<string, DebrisType> = {
  shipwreck: 'Shipwreck', wreck: 'Shipwreck', ship: 'Shipwreck',
  pipe: 'Pipe',
  ghost_net: 'Ghost Net', 'ghost net': 'Ghost Net', net: 'Ghost Net',
  cylinder: 'Cylinder', rock: 'Cylinder',
}
const HUMAN_CLASS_NAMES = new Set(['human', 'person'])

function normalizeClassName(name: string): string {
  return name.trim().toLowerCase().replace(/-/g, '_')
}

export function isHumanClass(className: string): boolean {
  return HUMAN_CLASS_NAMES.has(normalizeClassName(className))
}

export function displayClassification(className: string): DebrisType {
  return CLASS_DISPLAY_MAP[normalizeClassName(className)] ?? 'Other Debris'
}

function titleCasePriority(label: string): Priority {
  const lower = label.toLowerCase()
  if (lower === 'high') return 'High'
  if (lower === 'medium') return 'Medium'
  return 'Low'
}

/** Adapts one raw /logs/{id}/detections row (src.backend.schemas.Detection)
 * into the shape every existing dashboard table/chart/map component already
 * expects. Human detections are the caller's responsibility to filter out
 * first (see isHumanClass) -- kept as a separate step rather than silently
 * dropped here, so a caller that DOES want them (a future safety-review
 * view) isn't fighting this function. */
export function detectionRecordFromRaw(raw: RawDetection, missionName: string): DetectionRecord {
  return {
    id: raw.id,
    type: displayClassification(raw.class_name),
    rawClassName: raw.class_name,
    confidence: Math.round(raw.confidence_score),
    depth: raw.depth_m ?? 0,
    latitude: raw.lat ?? 0,
    longitude: raw.lon ?? 0,
    timestamp: raw.created_at,
    priority: titleCasePriority(raw.confidence_label),
    mission: missionName,
    logId: raw.log_id,
    frameRecordId: raw.frame_record_id,
    located: raw.geo_method === 'nav_fix' && raw.lat != null && raw.lon != null,
    lengthM: raw.length_m,
    widthM: raw.width_m,
    heightM: raw.height_m,
    dimensionsEstimated: raw.dimensions_estimated,
  }
}

export function missionNameFor(log: LogSummary): string {
  return log.filename || log.id.slice(0, 8)
}

export const formatTime = (value: string) => {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false }).format(date).replace(',', '')
}
export const colorForType = (type: string) => categories.find(category => category.name === type)?.color ?? '#4c80df'

export function downloadCsv(filename: string, rows: (string | number)[][]) {
  const csv = rows.map(row => row.map(value => `"${String(value).replace(/"/g, '""')}"`).join(',')).join('\r\n')
  const link = document.createElement('a')
  link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }))
  link.download = filename
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(link.href), 1000)
}
export const detectionCsv = (items: DetectionRecord[]) => [
  ['ID', 'Type', 'Confidence (%)', 'Depth (m)', 'Length (m)', 'Width (m)', 'Height (m)', 'Dimensions estimated', 'Latitude', 'Longitude', 'Timestamp', 'Priority', 'Log'],
  ...items.map(item => [item.id, item.type, item.confidence, item.depth, item.lengthM ?? '', item.widthM ?? '', item.heightM ?? '', item.dimensionsEstimated ? 'yes' : 'no', item.latitude, item.longitude, item.timestamp, item.priority, item.mission]),
]

import type { Feature, FeatureCollection, LineString, Point } from 'geojson'
import type { Detection, Survey, SurveySummaryData } from '../types'
import type { ApiDetection } from '../../../lib/api'

/** Adapts one real /api/surveys/{id}/detections row into this page's local
 * Detection shape. The backend's ApiDetection type additionally allows
 * classification: 'Human' (for the separate /humans endpoint) and an
 * `dimensionsEstimated` flag this map doesn't currently surface -- humans
 * are filtered out by the caller (getSurveyDetections defaults to excluding
 * them server-side already, this is just a type-narrowing safety net). */
export function apiDetectionToLocal(raw: ApiDetection): Detection | null {
  if (raw.classification === 'Human') return null
  return {
    id: raw.id,
    surveyId: raw.surveyId,
    classification: raw.classification,
    priority: raw.priority,
    confidence: raw.confidence,
    dimensions: { length: raw.dimensions.length, width: raw.dimensions.width, height: raw.dimensions.height },
    pingId: raw.pingId,
    timestamp: raw.timestamp,
    depth: raw.depth,
    coordinates: raw.coordinates,
  }
}

const classSymbols: Record<Detection['classification'], string> = {
  'Ghost Net': 'GN',
  Pipe: 'PI',
  Shipwreck: 'SW',
  Cylinder: 'CY',
  'Other Debris': 'OD',
}

export type DetectionFeatureProperties = {
  id: string
  classification: Detection['classification']
  priority: Detection['priority']
  confidence: number
  classSymbol: string
}

export function detectionsToGeoJSON(
  detections: Detection[],
): FeatureCollection<Point, DetectionFeatureProperties> {
  return {
    type: 'FeatureCollection',
    features: detections.map((detection) => ({
      type: 'Feature',
      id: detection.id,
      geometry: { type: 'Point', coordinates: detection.coordinates },
      properties: {
        id: detection.id,
        classification: detection.classification,
        priority: detection.priority,
        confidence: detection.confidence,
        classSymbol: classSymbols[detection.classification],
      },
    })),
  }
}

export function surveyTrackToGeoJSON(survey: Survey): Feature<LineString> {
  return {
    type: 'Feature',
    properties: { surveyId: survey.id },
    geometry: { type: 'LineString', coordinates: survey.trackCoordinates },
  }
}

export function getSurveySummary(detections: Detection[]): SurveySummaryData {
  const totalConfidence = detections.reduce((sum, detection) => sum + detection.confidence, 0)
  return {
    totalDetections: detections.length,
    highPriorityCount: detections.filter((detection) => detection.priority === 'HIGH').length,
    averageConfidence: detections.length ? Math.round(totalConfidence / detections.length) : 0,
  }
}

export function getSurveyBounds(survey: Survey, detections: Detection[]) {
  const points = [...survey.trackCoordinates, ...detections.map((detection) => detection.coordinates)]
  return points.reduce(
    (bounds, [longitude, latitude]) => ({
      west: Math.min(bounds.west, longitude),
      south: Math.min(bounds.south, latitude),
      east: Math.max(bounds.east, longitude),
      north: Math.max(bounds.north, latitude),
    }),
    { west: Infinity, south: Infinity, east: -Infinity, north: -Infinity },
  )
}

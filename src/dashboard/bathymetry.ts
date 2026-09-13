// Synthetic demo bathymetry for the map's "Bathymetry" tab.
//
// WHY SYNTHETIC: this project has no gridded depth-survey pipeline -- only
// sparse per-detection depth_m values along a survey track (see
// src/backend/pipeline_runner.py / csv_engine.py). There is no real
// seafloor DEM to contour. This module generates a plausible, NOAA-chart-
// styled seafloor surface (a colour-banded depth raster + labelled contour
// isolines, matching the reference chart look the frontend was asked to
// copy) purely for demo/POC visual purposes. It is deterministic per log
// (seeded from that log's own detection ids) so switching the map tab back
// and forth always shows the same "seafloor" for the same log, but it is
// explicitly NOT measured bathymetry -- DetectionMap.tsx renders a visible
// disclaimer alongside it for exactly that reason (same placeholder-
// discipline this project's geolocation code already follows: never present
// a fabricated number as if it were measured).
import type { Feature, FeatureCollection, LineString, Point } from 'geojson'
import type { DetectionRecord } from './data'

export interface Bounds { west: number; east: number; south: number; north: number }

export interface DepthStop { depth: number; color: string }

export interface Bathymetry {
  imageDataUrl: string
  imageCoordinates: [[number, number], [number, number], [number, number], [number, number]]
  contours: FeatureCollection<LineString, { depth: number }>
  labels: FeatureCollection<Point, { depth: number }>
  legend: DepthStop[]
  minDepth: number
  maxDepth: number
}

const GRID = 112

// Dark-navy-deep -> pale-shallow ramp, same spirit as the reference chart's
// "Depth in m" legend.
const RAMP: DepthStop[] = [
  { depth: 0, color: '#f5fbfd' }, { depth: 5, color: '#dcedf6' }, { depth: 10, color: '#c7e2f1' },
  { depth: 15, color: '#b3d8ec' }, { depth: 20, color: '#9fcde6' }, { depth: 25, color: '#8bc2e1' },
  { depth: 30, color: '#77b7db' }, { depth: 40, color: '#5ba0cd' }, { depth: 50, color: '#4790c0' },
  { depth: 60, color: '#3c7cac' }, { depth: 70, color: '#356c98' }, { depth: 80, color: '#2e5c85' },
  { depth: 90, color: '#274d73' }, { depth: 100, color: '#213f62' }, { depth: 120, color: '#1a3252' },
  { depth: 150, color: '#152943' }, { depth: 200, color: '#0f1f35' }, { depth: 300, color: '#0a1628' },
]

function hashSeed(input: string): number {
  let h = 2166136261
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return (h >>> 0) || 1
}

function mulberry32(seed: number) {
  let a = seed
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function boundsFor(items: DetectionRecord[]): Bounds {
  const located = items.filter(item => item.located)
  if (located.length === 0) {
    // Fallback: the demo survey's own Arabian Sea start point (see the
    // demo zip's generate_demo_survey.py), so the tab still shows
    // something sensible before any GPS-fixed log is selected.
    return { west: 69.98, east: 70.02, south: 16.985, north: 17.015 }
  }
  const lons = located.map(d => d.longitude), lats = located.map(d => d.latitude)
  const west = Math.min(...lons), east = Math.max(...lons)
  const south = Math.min(...lats), north = Math.max(...lats)
  const padLon = Math.max((east - west) * 0.6, 0.01)
  const padLat = Math.max((north - south) * 0.6, 0.008)
  return { west: west - padLon, east: east + padLon, south: south - padLat, north: north + padLat }
}

function colorFor(depth: number): string {
  if (depth <= RAMP[0].depth) return RAMP[0].color
  for (let i = 1; i < RAMP.length; i++) {
    if (depth <= RAMP[i].depth) {
      const a = RAMP[i - 1], b = RAMP[i]
      const t = (depth - a.depth) / (b.depth - a.depth)
      return lerpColor(a.color, b.color, t)
    }
  }
  return RAMP[RAMP.length - 1].color
}

function lerpColor(hexA: string, hexB: string, t: number): string {
  const a = parseInt(hexA.slice(1), 16), b = parseInt(hexB.slice(1), 16)
  const ar = (a >> 16) & 255, ag = (a >> 8) & 255, ab = a & 255
  const br = (b >> 16) & 255, bg = (b >> 8) & 255, bb = b & 255
  const r = Math.round(ar + (br - ar) * t), g = Math.round(ag + (bg - ag) * t), bl = Math.round(ab + (bb - ab) * t)
  return `rgb(${r},${g},${bl})`
}

function buildDepthField(bounds: Bounds, seed: number): { values: Float64Array; min: number; max: number } {
  const rng = mulberry32(seed)
  const values = new Float64Array(GRID * GRID)

  const shallowX = rng() > 0.5 ? 0 : 1
  const shallowY = rng() > 0.5 ? 0 : 1
  const minDepth = 6 + rng() * 12          // 6-18m at the shallow corner
  const deepSpread = 70 + rng() * 110      // 70-180m at the far corner

  // A handful of irregular sine terms stand in for real seafloor texture
  // (fractal-ish undulation) without needing a full Perlin/simplex noise
  // implementation -- good enough for a demo chart's wiggly contour lines.
  const octaves = Array.from({ length: 5 }, (_, k) => ({
    fx: 1.5 + rng() * 5 + k,
    fy: 1.5 + rng() * 5 + k,
    phase: rng() * Math.PI * 2,
    amp: (10 + rng() * 8) / (k + 1.4),
  }))

  // A diagonal canyon/trough, echoing the reference chart's submarine
  // canyon spine.
  const canyonAngle = (20 + rng() * 100) * (Math.PI / 180)
  const canyonOffset = 0.25 + rng() * 0.5
  const canyonWidth = 0.05 + rng() * 0.05
  const canyonDepthBonus = 60 + rng() * 90

  const dx = Math.cos(canyonAngle), dy = Math.sin(canyonAngle)

  let min = Infinity, max = -Infinity
  for (let j = 0; j < GRID; j++) {
    const y = j / (GRID - 1)
    for (let i = 0; i < GRID; i++) {
      const x = i / (GRID - 1)
      const distToShallow = Math.hypot(x - shallowX, y - shallowY) / Math.SQRT2
      let depth = minDepth + deepSpread * Math.min(1, distToShallow * 1.15)

      for (const o of octaves) {
        depth += o.amp * Math.sin(x * o.fx * Math.PI * 2 + o.phase) * Math.cos(y * o.fy * Math.PI * 2 - o.phase)
      }

      // signed perpendicular distance from the canyon's centre line
      const px = x - canyonOffset * dy, py = y - canyonOffset * dx
      const perp = px * -dy + py * dx
      depth += canyonDepthBonus * Math.exp(-(perp * perp) / (2 * canyonWidth * canyonWidth))

      depth = Math.max(2, depth)
      values[j * GRID + i] = depth
      if (depth < min) min = depth
      if (depth > max) max = depth
    }
  }
  return { values, min, max }
}

function depthFieldToImageDataUrl(values: Float64Array): string {
  const canvas = document.createElement('canvas')
  canvas.width = GRID; canvas.height = GRID
  const ctx = canvas.getContext('2d')!
  const img = ctx.createImageData(GRID, GRID)
  for (let p = 0; p < GRID * GRID; p++) {
    const color = colorFor(values[p])
    const [r, g, b] = color.startsWith('rgb') ? color.slice(4, -1).split(',').map(Number) : [0, 0, 0]
    img.data[p * 4] = r; img.data[p * 4 + 1] = g; img.data[p * 4 + 2] = b; img.data[p * 4 + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  return canvas.toDataURL('image/png')
}

function pickLevels(min: number, max: number): number[] {
  const levels: number[] = []
  const push = (v: number) => { if (v > min + 1 && v < max - 1) levels.push(v) }
  for (let v = 5; v <= 100; v += 5) push(v)
  for (let v = 110; v <= 200; v += 10) push(v)
  for (let v = 220; v <= 400; v += 20) push(v)
  return levels
}

type Edge = 'N' | 'E' | 'S' | 'W'
const CASE_EDGES: Record<number, Edge[][]> = {
  1: [['S', 'W']], 2: [['S', 'E']], 3: [['W', 'E']], 4: [['N', 'E']],
  5: [['N', 'W'], ['S', 'E']], 6: [['N', 'S']], 7: [['N', 'W']], 8: [['N', 'W']],
  9: [['N', 'S']], 10: [['N', 'E'], ['S', 'W']], 11: [['N', 'E']], 12: [['W', 'E']],
  13: [['S', 'E']], 14: [['S', 'W']],
}

function marchingSquares(values: Float64Array, bounds: Bounds, level: number): [number, number][][] {
  const lon = (i: number) => bounds.west + (i / (GRID - 1)) * (bounds.east - bounds.west)
  const lat = (j: number) => bounds.north - (j / (GRID - 1)) * (bounds.north - bounds.south)
  const at = (i: number, j: number) => values[j * GRID + i]

  const edgePoint = (edge: Edge, i: number, j: number): [number, number] => {
    let a: number, b: number, p0: [number, number], p1: [number, number]
    switch (edge) {
      case 'N': a = at(i, j); b = at(i + 1, j); p0 = [lon(i), lat(j)]; p1 = [lon(i + 1), lat(j)]; break
      case 'E': a = at(i + 1, j); b = at(i + 1, j + 1); p0 = [lon(i + 1), lat(j)]; p1 = [lon(i + 1), lat(j + 1)]; break
      case 'S': a = at(i, j + 1); b = at(i + 1, j + 1); p0 = [lon(i), lat(j + 1)]; p1 = [lon(i + 1), lat(j + 1)]; break
      case 'W': default: a = at(i, j); b = at(i, j + 1); p0 = [lon(i), lat(j)]; p1 = [lon(i), lat(j + 1)]; break
    }
    const t = b === a ? 0.5 : (level - a) / (b - a)
    return [p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t]
  }

  const segments: [number, number][][] = []
  for (let j = 0; j < GRID - 1; j++) {
    for (let i = 0; i < GRID - 1; i++) {
      const tl = at(i, j), tr = at(i + 1, j), br = at(i + 1, j + 1), bl = at(i, j + 1)
      const c = (tl > level ? 8 : 0) | (tr > level ? 4 : 0) | (br > level ? 2 : 0) | (bl > level ? 1 : 0)
      const pairs = CASE_EDGES[c]
      if (!pairs) continue
      for (const [e0, e1] of pairs) segments.push([edgePoint(e0, i, j), edgePoint(e1, i, j)])
    }
  }
  return segments
}

export function generateBathymetry(items: DetectionRecord[]): Bathymetry {
  const seedKey = items.filter(d => d.located).map(d => d.id).sort().join('|') || 'sonarsense-demo-default'
  const seed = hashSeed(seedKey)
  const bounds = boundsFor(items)
  const { values, min, max } = buildDepthField(bounds, seed)

  const contourFeatures: Feature<LineString, { depth: number }>[] = []
  const labelFeatures: Feature<Point, { depth: number }>[] = []
  for (const level of pickLevels(min, max)) {
    const segments = marchingSquares(values, bounds, level)
    segments.forEach((seg, idx) => {
      contourFeatures.push({ type: 'Feature', properties: { depth: level }, geometry: { type: 'LineString', coordinates: seg } })
      if (idx % 55 === 0) {
        const mid: [number, number] = [(seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2]
        labelFeatures.push({ type: 'Feature', properties: { depth: level }, geometry: { type: 'Point', coordinates: mid } })
      }
    })
  }

  const legend = RAMP.filter(stop => stop.depth <= Math.ceil(max / 10) * 10 + 10)

  return {
    imageDataUrl: depthFieldToImageDataUrl(values),
    imageCoordinates: [[bounds.west, bounds.north], [bounds.east, bounds.north], [bounds.east, bounds.south], [bounds.west, bounds.south]],
    contours: { type: 'FeatureCollection', features: contourFeatures },
    labels: { type: 'FeatureCollection', features: labelFeatures },
    legend,
    minDepth: min,
    maxDepth: max,
  }
}

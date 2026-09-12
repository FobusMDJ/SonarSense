import { useEffect, useRef, useState } from 'react'
import maplibregl, { type GeoJSONSource, type Map as LibreMap } from 'maplibre-gl'
import { Minus, Plus } from 'lucide-react'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { DetectionRecord } from './data'

// Fallback view (no real GPS-tagged detections yet) -- an arbitrary wide
// zoomed-out view, NOT a stand-in for any real survey location.
const WORLD_FALLBACK: [[number, number], [number, number]] = [[-40, -30], [110, 60]]

export function DetectionMap({ items, onSelect, full = false }: { items: DetectionRecord[]; onSelect: (item: DetectionRecord) => void; full?: boolean }) {
  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<LibreMap | null>(null)
  const callback = useRef(onSelect)
  const dataRef = useRef(items)
  const [ready, setReady] = useState(false)
  const [style, setStyle] = useState('Map')
  const [failed, setFailed] = useState(false)
  callback.current = onSelect; dataRef.current = items

  const located = (records: DetectionRecord[]) => records.filter(r => r.located)

  const geoJSON = (records: DetectionRecord[]): GeoJSON.FeatureCollection => ({
    type: 'FeatureCollection', features: located(records).map(record => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [record.longitude, record.latitude] },
      properties: { id: record.id, confidence: record.confidence },
    })),
  })

  const fit = () => {
    const points = located(dataRef.current)
    if (points.length === 0) { mapRef.current?.fitBounds(WORLD_FALLBACK, { duration: 0 }); return }
    const lons = points.map(p => p.longitude), lats = points.map(p => p.latitude)
    const west = Math.min(...lons), east = Math.max(...lons), south = Math.min(...lats), north = Math.max(...lats)
    if (west === east && south === north) {
      mapRef.current?.flyTo({ center: [west, south], zoom: 10, duration: 500 })
    } else {
      mapRef.current?.fitBounds([[west, south], [east, north]], { padding: 45, duration: 500 })
    }
  }

  useEffect(() => {
    if (!container.current) return
    let map: LibreMap
    try {
      map = new maplibregl.Map({ container: container.current, center: [0, 20], zoom: 1.5, attributionControl: false, style: {
        version: 8, sources: {
          imagery: { type: 'raster', tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'], tileSize: 256, attribution: 'Imagery © Esri & contributors' },
          labels: { type: 'raster', tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'], tileSize: 256 },
        }, layers: [
          { id: 'imagery', type: 'raster', source: 'imagery', paint: { 'raster-saturation': -.15, 'raster-brightness-max': .77 } },
          { id: 'labels', type: 'raster', source: 'labels', paint: { 'raster-opacity': .8 } },
        ],
      } })
    } catch { setFailed(true); return }
    mapRef.current = map
    map.on('load', () => {
      map.addSource('points', { type: 'geojson', data: geoJSON(dataRef.current) })
      map.addLayer({ id: 'points', source: 'points', type: 'circle', paint: {
        'circle-radius': 4.5,
        'circle-color': ['step', ['get', 'confidence'], '#bf586a', 20, '#dd663c', 40, '#e5a942', 60, '#219ad0', 80, '#4eaf6b'],
        'circle-stroke-color': '#b0d2d2', 'circle-stroke-width': 1, 'circle-stroke-opacity': .5,
      } })
      map.addLayer({ id: 'point-centers', source: 'points', type: 'circle', paint: { 'circle-radius': 1.1, 'circle-color': '#d6f1fb' } })
      map.on('click', 'points', event => {
        const record = dataRef.current.find(item => item.id === event.features?.[0].properties?.id)
        if (record) callback.current(record)
      })
      map.on('mouseenter', 'points', () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mouseleave', 'points', () => { map.getCanvas().style.cursor = '' })
      map.addControl(new maplibregl.ScaleControl({ maxWidth: 62, unit: 'metric' }), 'bottom-left')
      map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right')
      setReady(true); fit()
    })
    const resize = new ResizeObserver(() => map.resize())
    resize.observe(container.current)
    return () => { resize.disconnect(); map.remove(); mapRef.current = null; setReady(false) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  useEffect(() => { if (ready) { (mapRef.current?.getSource('points') as GeoJSONSource)?.setData(geoJSON(items)); fit() } }, [items, ready])
  const toggleStyle = (next: string) => { setStyle(next); if (ready) mapRef.current?.setPaintProperty('labels', 'raster-opacity', next === 'Map' ? .8 : 0) }
  const unlocatedCount = items.length - located(items).length
  return <div className={`ds-map ${full ? 'ds-map-full' : ''}`}>
    <div className="ds-map-mount" ref={container} role="region" aria-label="Detection map" />
    {failed && <div className="ds-map-unavailable">WebGL is unavailable. Detection coordinates are available in the Detections view.</div>}
    <div className="ds-map-tabs">{['Map', 'Satellite'].map(label => <button key={label} className={style === label ? 'selected' : ''} onClick={() => toggleStyle(label)} aria-pressed={style === label}>{label}</button>)}</div>
    <div className="ds-map-zoom"><button aria-label="Zoom in" title="Zoom in" onClick={() => mapRef.current?.zoomIn()}><Plus size={19}/></button><button aria-label="Zoom out" title="Zoom out" onClick={() => mapRef.current?.zoomOut()}><Minus size={19}/></button></div>
    {unlocatedCount > 0 && <div className="ds-map-note">{unlocatedCount} detection(s) have no GPS fix and aren't shown here.</div>}
    <div className="ds-map-key"><strong>Confidence</strong>{[['#4eaf6b', '80–100%'], ['#219ad0', '60–80%'], ['#e5a942', '40–60%'], ['#dd663c', '20–40%'], ['#bf586a', '0–20%']].map(([color, label]) => <span key={label}><i style={{ background: color }}/>{label}</span>)}</div>
  </div>
}

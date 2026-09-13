import { useEffect, useMemo, useRef, useState } from 'react'
import maplibregl, { type GeoJSONSource, type Map as LibreMap } from 'maplibre-gl'
import { Compass, Minus, Plus } from 'lucide-react'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { DetectionRecord } from './data'
import { boundsFor, generateBathymetry } from './bathymetry'

const BATHY_LAYER_IDS = ['bathy-image-layer', 'bathy-contour-lines', 'bathy-contour-label-text']

// Fallback view (no real GPS-tagged detections yet) -- an arbitrary wide
// zoomed-out view, NOT a stand-in for any real survey location.
const WORLD_FALLBACK: [[number, number], [number, number]] = [[-40, -30], [110, 60]]

export function DetectionMap({ items, onSelect, full = false }: { items: DetectionRecord[]; onSelect: (item: DetectionRecord) => void; full?: boolean }) {
  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<LibreMap | null>(null)
  const callback = useRef(onSelect)
  const dataRef = useRef(items)
  const [ready, setReady] = useState(false)
  const [style, setStyle] = useState<'Map' | 'Satellite' | 'Bathymetry'>('Map')
  const [failed, setFailed] = useState(false)
  callback.current = onSelect; dataRef.current = items

  const located = (records: DetectionRecord[]) => records.filter(r => r.located)

  // Synthetic demo seafloor (see bathymetry.ts's module docstring for why
  // it's synthetic, not measured) -- recomputed only when the actual set of
  // located detections changes, not on every unrelated re-render, and
  // deterministic per that set so toggling the tab never redraws a
  // different-looking seafloor for the same log.
  const bathymetryKey = located(items).map(d => d.id).sort().join('|')
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const bathymetry = useMemo(() => generateBathymetry(items), [bathymetryKey])

  const applyBathymetryLayers = (map: LibreMap) => {
    BATHY_LAYER_IDS.forEach(id => { if (map.getLayer(id)) map.removeLayer(id) })
    if (map.getSource('bathy-image')) map.removeSource('bathy-image')
    if (map.getSource('bathy-contours')) map.removeSource('bathy-contours')
    if (map.getSource('bathy-contour-labels')) map.removeSource('bathy-contour-labels')

    map.addSource('bathy-image', { type: 'image', url: bathymetry.imageDataUrl, coordinates: bathymetry.imageCoordinates })
    map.addLayer({ id: 'bathy-image-layer', type: 'raster', source: 'bathy-image', paint: { 'raster-opacity': .85 }, layout: { visibility: 'none' } }, 'points')
    map.addSource('bathy-contours', { type: 'geojson', data: bathymetry.contours })
    map.addLayer({ id: 'bathy-contour-lines', type: 'line', source: 'bathy-contours', paint: { 'line-color': '#17324a', 'line-width': 1, 'line-opacity': .85 }, layout: { visibility: 'none' } }, 'points')
    map.addSource('bathy-contour-labels', { type: 'geojson', data: bathymetry.labels })
    map.addLayer({ id: 'bathy-contour-label-text', type: 'symbol', source: 'bathy-contour-labels', layout: {
      'text-field': ['concat', ['to-string', ['get', 'depth']], 'm'], 'text-font': ['Open Sans Bold'], 'text-size': 10,
      'text-allow-overlap': false, visibility: 'none',
    }, paint: { 'text-color': '#0d1f2c', 'text-halo-color': '#eaf3f8', 'text-halo-width': 1.4 } }, 'points')

    const visibility = style === 'Bathymetry' ? 'visible' : 'none'
    BATHY_LAYER_IDS.forEach(id => map.setLayoutProperty(id, 'visibility', visibility))
  }

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

  // The Bathymetry tab's seafloor covers a much wider, padded area than the
  // tight detection-point bounding box `fit()` normally zooms to (see
  // bathymetry.ts's boundsFor -- a tight debris cluster still gets a >=1km
  // box so the generated contours have somewhere to run) -- without this,
  // switching tabs on a tightly-clustered log leaves the map zoomed in so
  // far that only a single, near-uniform-colour patch of the seafloor is
  // ever in view, and the "chart" context (contour bands, canyon shape) an
  // operator would actually want never appears.
  const fitBathymetry = () => {
    const b = boundsFor(dataRef.current)
    mapRef.current?.fitBounds([[b.west, b.south], [b.east, b.north]], { padding: 40, duration: 500 })
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
      applyBathymetryLayers(map)
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
  // Re-render the seafloor layers when the underlying detection set changes
  // (e.g. a different log is selected) while the Bathymetry tab may already
  // be mounted -- applyBathymetryLayers() itself is idempotent (tears down
  // and rebuilds its own sources/layers), so calling it again is safe.
  useEffect(() => { if (ready && mapRef.current) applyBathymetryLayers(mapRef.current) }, [bathymetry, ready])
  // Keep layer visibility/opacity in sync with `style` regardless of WHEN
  // `ready` flips true relative to the tab click -- e.g. a user tapping
  // "Bathymetry" before the map's first 'load' event (slow tile fetch,
  // slow device) must still see the seafloor once loading finishes,
  // instead of the click's one-shot handler having already bailed out on
  // `!ready` and never being told to re-apply it.
  const prevStyleRef = useRef(style)
  useEffect(() => {
    if (!ready || !mapRef.current) return
    const map = mapRef.current
    map.setPaintProperty('labels', 'raster-opacity', style === 'Map' ? .8 : style === 'Satellite' ? 0 : .25)
    map.setPaintProperty('imagery', 'raster-opacity', style === 'Bathymetry' ? .5 : 1)
    const visibility = style === 'Bathymetry' ? 'visible' : 'none'
    BATHY_LAYER_IDS.forEach(id => { if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visibility) })
    if (style === 'Bathymetry' && prevStyleRef.current !== 'Bathymetry') fitBathymetry()
    else if (style !== 'Bathymetry' && prevStyleRef.current === 'Bathymetry') fit()
    prevStyleRef.current = style
  }, [style, ready])

  const toggleStyle = (next: 'Map' | 'Satellite' | 'Bathymetry') => setStyle(next)
  const unlocatedCount = items.length - located(items).length
  return <div className={`ds-map ${full ? 'ds-map-full' : ''}`}>
    <div className="ds-map-mount" ref={container} role="region" aria-label="Detection map" />
    {failed && <div className="ds-map-unavailable">WebGL is unavailable. Detection coordinates are available in the Detections view.</div>}
    <div className="ds-map-tabs">{(['Map', 'Satellite', 'Bathymetry'] as const).map(label => <button key={label} className={style === label ? 'selected' : ''} onClick={() => toggleStyle(label)} aria-pressed={style === label}>{label}</button>)}</div>
    <div className="ds-map-zoom"><button aria-label="Zoom in" title="Zoom in" onClick={() => mapRef.current?.zoomIn()}><Plus size={19}/></button><button aria-label="Zoom out" title="Zoom out" onClick={() => mapRef.current?.zoomOut()}><Minus size={19}/></button></div>
    {style === 'Bathymetry' && <div className="ds-map-north" title="North" aria-label="North"><Compass size={15}/><b>N</b></div>}
    {unlocatedCount > 0 && <div className="ds-map-note">{unlocatedCount} detection(s) have no GPS fix and aren't shown here.</div>}
    {style === 'Bathymetry' && <div className="ds-map-bathy-note">Illustrative seafloor relief for demo purposes -- not measured bathymetry.</div>}
    {style === 'Bathymetry'
      ? <div className="ds-map-key ds-map-depth-key"><strong>Depth in m</strong>{bathymetry.legend.map(stop => <span key={stop.depth}><i style={{ background: stop.color }}/>{stop.depth}</span>)}</div>
      : <div className="ds-map-key"><strong>Confidence</strong>{[['#4eaf6b', '80–100%'], ['#219ad0', '60–80%'], ['#e5a942', '40–60%'], ['#dd663c', '20–40%'], ['#bf586a', '0–20%']].map(([color, label]) => <span key={label}><i style={{ background: color }}/>{label}</span>)}</div>}
  </div>
}

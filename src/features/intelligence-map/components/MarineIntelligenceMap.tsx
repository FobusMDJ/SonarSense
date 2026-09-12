import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import maplibregl, { type GeoJSONSource, type Map, type MapGeoJSONFeature, type MapMouseEvent, type StyleSpecification } from 'maplibre-gl'
import type { FeatureCollection, LineString } from 'geojson'
import { Crosshair, Maximize2, Minus, Plus } from 'lucide-react'
import type { Detection, Survey } from '../types'
import { detectionsToGeoJSON, getSurveyBounds, surveyTrackToGeoJSON } from '../services/mapDataAdapter'
import { MapLegend } from './MapLegend'

interface MarineIntelligenceMapProps {
  survey: Survey
  detections: Detection[]
  allDetections: Detection[]
  onSelect: (detection: Detection) => void
}

export interface MarineIntelligenceMapHandle {
  fitSurvey: () => void
}

const mapStyle: StyleSpecification = {
  version: 8,
  glyphs: 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf',
  sources: {
    ocean: {
      type: 'raster',
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256,
      maxzoom: 16,
      attribution: 'Imagery © Esri and contributors',
    },
    oceanLabels: {
      type: 'raster',
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256,
      maxzoom: 16,
    },
  },
  layers: [{
    id: 'ocean-muted',
    type: 'raster',
    source: 'ocean',
    paint: {
      'raster-saturation': -0.55,
      'raster-contrast': 0.22,
      'raster-brightness-min': 0.02,
      'raster-brightness-max': 0.32,
      'raster-opacity': 0.94,
      'raster-hue-rotate': 155,
    },
  }, {
    id: 'ocean-reference',
    type: 'raster',
    source: 'oceanLabels',
    paint: { 'raster-opacity': 0.46, 'raster-brightness-max': 0.6, 'raster-saturation': -0.7 },
  }],
}

const createTerminalMarker = (label: string, className: string) => {
  const marker = document.createElement('div')
  marker.className = `ss-terminal-marker ${className}`
  marker.setAttribute('aria-label', `${label} of survey track`)
  const dot = document.createElement('span')
  const text = document.createElement('b')
  text.textContent = label
  marker.append(dot, text)
  return marker
}

type CoordinateGridProperties = {
  axis: 'latitude' | 'longitude'
  label: string
}

const coordinateSteps = [
  .0001, .0002, .0005, .001, .002, .005, .01, .02, .05, .1, .2, .5,
  1, 2, 5, 10, 20, 30, 45, 90,
]

function gridStepFor(map: Map) {
  const bounds = map.getBounds()
  const span = Math.max(bounds.getEast() - bounds.getWest(), bounds.getNorth() - bounds.getSouth())
  const target = Math.max(.0001, span / 8)
  return coordinateSteps.find((step) => step >= target) ?? 90
}

function coordinateLabel(value: number, axis: CoordinateGridProperties['axis'], step: number) {
  const decimals = step < .001 ? 4 : step < .01 ? 3 : step < .1 ? 2 : step < 1 ? 1 : 0
  const direction = axis === 'latitude'
    ? (value >= 0 ? 'N' : 'S')
    : (value >= 0 ? 'E' : 'W')
  return `${Math.abs(value).toFixed(decimals)}°${direction}`
}

function coordinateGridFor(map: Map): FeatureCollection<LineString, CoordinateGridProperties> {
  const bounds = map.getBounds()
  const west = Math.max(-180, bounds.getWest())
  const east = Math.min(180, bounds.getEast())
  const south = Math.max(-85, bounds.getSouth())
  const north = Math.min(85, bounds.getNorth())
  const step = gridStepFor(map)
  const features: FeatureCollection<LineString, CoordinateGridProperties>['features'] = []

  for (let longitude = Math.ceil(west / step) * step; longitude <= east; longitude += step) {
    const value = Number(longitude.toFixed(7))
    features.push({
      type: 'Feature',
      properties: { axis: 'longitude', label: coordinateLabel(value, 'longitude', step) },
      geometry: { type: 'LineString', coordinates: [[value, south], [value, north]] },
    })
  }

  for (let latitude = Math.ceil(south / step) * step; latitude <= north; latitude += step) {
    const value = Number(latitude.toFixed(7))
    features.push({
      type: 'Feature',
      properties: { axis: 'latitude', label: coordinateLabel(value, 'latitude', step) },
      geometry: { type: 'LineString', coordinates: [[west, value], [east, value]] },
    })
  }

  return { type: 'FeatureCollection', features }
}

export const MarineIntelligenceMap = forwardRef<MarineIntelligenceMapHandle, MarineIntelligenceMapProps>(
  function MarineIntelligenceMap({ survey, detections, allDetections, onSelect }, ref) {
    const containerRef = useRef<HTMLDivElement>(null)
    const mapRef = useRef<Map | null>(null)
    const detectionsRef = useRef(allDetections)
    const onSelectRef = useRef(onSelect)
    const [ready, setReady] = useState(false)
    const [fullscreenSupported] = useState(() => Boolean(document.fullscreenEnabled))

    detectionsRef.current = allDetections
    onSelectRef.current = onSelect

    const fitSurvey = () => {
      const map = mapRef.current
      if (!map) return
      const bounds = getSurveyBounds(survey, allDetections)
      const compact = map.getContainer().clientWidth < 700
      map.fitBounds([[bounds.west, bounds.south], [bounds.east, bounds.north]], {
        padding: compact
          ? { top: 84, right: 46, bottom: 86, left: 46 }
          : { top: 72, right: 82, bottom: 72, left: 340 },
        duration: 850,
        maxZoom: 12,
      })
    }

    useImperativeHandle(ref, () => ({ fitSurvey }))

    useEffect(() => {
      if (!containerRef.current || mapRef.current) return

      const map = new maplibregl.Map({
        container: containerRef.current,
        style: mapStyle,
        center: [72.73, 18.785],
        zoom: 10.8,
        minZoom: 4,
        maxZoom: 18,
        attributionControl: false,
      })
      mapRef.current = map
      const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 17, className: 'ss-map-popup' })

      const onUnclusteredClick = (event: MapMouseEvent & { features?: MapGeoJSONFeature[] }) => {
        const id = event.features?.[0]?.properties?.id
        const detection = detectionsRef.current.find((item) => item.id === id)
        if (detection) onSelectRef.current(detection)
      }

      const onClusterClick = async (event: MapMouseEvent & { features?: MapGeoJSONFeature[] }) => {
        const feature = event.features?.[0]
        if (!feature || feature.geometry.type !== 'Point') return
        const clusterId = Number(feature.properties?.cluster_id)
        const source = map.getSource('detections') as GeoJSONSource
        const zoom = await source.getClusterExpansionZoom(clusterId)
        map.easeTo({ center: feature.geometry.coordinates as [number, number], zoom })
      }

      const onHover = (event: MapMouseEvent & { features?: MapGeoJSONFeature[] }) => {
        if (!window.matchMedia('(hover: hover)').matches) return
        map.getCanvas().style.cursor = 'pointer'
        const feature = event.features?.[0]
        if (!feature || feature.geometry.type !== 'Point') return
        const node = document.createElement('div')
        const title = document.createElement('strong')
        const meta = document.createElement('span')
        title.textContent = feature.properties?.classification
        meta.textContent = `${feature.properties?.confidence}% confidence · ${feature.properties?.priority}`
        node.append(title, meta)
        popup.setLngLat(feature.geometry.coordinates as [number, number]).setDOMContent(node).addTo(map)
      }

      const onLeave = () => {
        map.getCanvas().style.cursor = ''
        popup.remove()
      }

      const updateCoordinateGrid = () => {
        const source = map.getSource('coordinate-grid') as GeoJSONSource | undefined
        source?.setData(coordinateGridFor(map))
      }

      map.on('load', () => {
        map.addSource('coordinate-grid', { type: 'geojson', data: coordinateGridFor(map) })
        map.addLayer({
          id: 'coordinate-grid-lines', type: 'line', source: 'coordinate-grid',
          paint: {
            'line-color': '#b9dce1',
            'line-width': 1,
            'line-opacity': .22,
            'line-dasharray': [2, 3],
          },
        })
        map.addLayer({
          id: 'coordinate-grid-labels', type: 'symbol', source: 'coordinate-grid',
          layout: {
            'symbol-placement': 'line',
            'symbol-spacing': 360,
            'text-field': ['get', 'label'],
            'text-font': ['Open Sans Regular'],
            'text-size': 9,
            'text-letter-spacing': .08,
            'text-allow-overlap': false,
          },
          paint: {
            'text-color': '#c6e0e2',
            'text-halo-color': '#061019',
            'text-halo-width': 1.2,
            'text-opacity': .72,
          },
        })
        map.addSource('survey-track', { type: 'geojson', data: surveyTrackToGeoJSON(survey) })
        map.addLayer({
          id: 'survey-track-casing', type: 'line', source: 'survey-track',
          paint: { 'line-color': '#061019', 'line-width': 5, 'line-opacity': 0.62 },
        })
        map.addLayer({
          id: 'survey-track-line', type: 'line', source: 'survey-track',
          paint: { 'line-color': '#7bb9bd', 'line-width': 2, 'line-opacity': 0.78, 'line-dasharray': [2, 1.5] },
        })
        map.addSource('detections', {
          type: 'geojson', data: detectionsToGeoJSON(detectionsRef.current), cluster: true, clusterMaxZoom: 15, clusterRadius: 42,
        })
        map.addLayer({
          id: 'detection-clusters', type: 'circle', source: 'detections', filter: ['has', 'point_count'],
          paint: {
            'circle-color': '#315d67',
            'circle-radius': ['step', ['get', 'point_count'], 18, 10, 22, 50, 27],
            'circle-stroke-color': '#b4d4d4', 'circle-stroke-width': 2, 'circle-opacity': 0.92,
          },
        })
        map.addLayer({
          id: 'detection-cluster-count', type: 'symbol', source: 'detections', filter: ['has', 'point_count'],
          layout: { 'text-field': ['get', 'point_count_abbreviated'], 'text-font': ['Open Sans Bold'], 'text-size': 12 },
          paint: { 'text-color': '#f2f6f3' },
        })
        map.addLayer({
          id: 'detection-points', type: 'circle', source: 'detections', filter: ['!', ['has', 'point_count']],
          paint: {
            'circle-color': ['match', ['get', 'priority'], 'HIGH', '#a85652', 'MEDIUM', '#bc8b49', '#4f8d96'],
            'circle-radius': ['match', ['get', 'priority'], 'HIGH', 17, 'MEDIUM', 14, 11],
            'circle-stroke-color': ['match', ['get', 'priority'], 'HIGH', '#f3aaa2', 'MEDIUM', '#f2c781', '#a5d2d4'],
            'circle-stroke-width': ['match', ['get', 'priority'], 'HIGH', 4, 'MEDIUM', 3, 2],
            'circle-opacity': 0.96,
          },
        })
        map.addLayer({
          id: 'detection-symbols', type: 'symbol', source: 'detections', filter: ['!', ['has', 'point_count']],
          layout: {
            'text-field': ['get', 'classSymbol'], 'text-font': ['Open Sans Bold'],
            'text-size': ['match', ['get', 'priority'], 'HIGH', 11, 'MEDIUM', 10, 9],
            'text-allow-overlap': true,
          },
          paint: { 'text-color': '#f4f6f2' },
        })

        new maplibregl.Marker({ element: createTerminalMarker('START', 'is-start'), anchor: 'center' })
          .setLngLat(survey.trackCoordinates[0]).addTo(map)
        new maplibregl.Marker({ element: createTerminalMarker('END', 'is-end'), anchor: 'center' })
          .setLngLat(survey.trackCoordinates[survey.trackCoordinates.length - 1]).addTo(map)

        map.on('click', 'detection-points', onUnclusteredClick)
        map.on('click', 'detection-clusters', onClusterClick)
        map.on('mouseenter', 'detection-points', onHover)
        map.on('mouseleave', 'detection-points', onLeave)
        map.on('mouseenter', 'detection-clusters', () => { map.getCanvas().style.cursor = 'pointer' })
        map.on('mouseleave', 'detection-clusters', onLeave)
        map.on('moveend', updateCoordinateGrid)
        map.addControl(new maplibregl.ScaleControl({ maxWidth: 130, unit: 'metric' }), 'bottom-right')
        map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right')
        setReady(true)
        window.setTimeout(fitSurvey, 80)
      })

      const onFullscreenChange = () => window.setTimeout(() => map.resize(), 50)
      document.addEventListener('fullscreenchange', onFullscreenChange)

      return () => {
        document.removeEventListener('fullscreenchange', onFullscreenChange)
        popup.remove()
        map.remove()
        mapRef.current = null
      }
    }, [survey])

    useEffect(() => {
      if (!ready) return
      const source = mapRef.current?.getSource('detections') as GeoJSONSource | undefined
      source?.setData(detectionsToGeoJSON(detections))
    }, [detections, ready])

    const toggleFullscreen = async () => {
      if (!containerRef.current || !fullscreenSupported) return
      if (document.fullscreenElement) await document.exitFullscreen()
      else await containerRef.current.requestFullscreen()
    }

    return (
      <section className="ss-map-shell" aria-label="Marine debris intelligence map">
        <div ref={containerRef} className="ss-map-canvas" />
        <div className="ss-map-grain" aria-hidden="true" />
        <div className="ss-map-controls" aria-label="Map controls">
          <button type="button" onClick={() => mapRef.current?.zoomIn()} aria-label="Zoom in" title="Zoom in"><Plus size={18} /></button>
          <button type="button" onClick={() => mapRef.current?.zoomOut()} aria-label="Zoom out" title="Zoom out"><Minus size={18} /></button>
          <span />
          <button type="button" onClick={fitSurvey} aria-label="Fit survey to view" title="Fit survey"><Crosshair size={18} /></button>
          {fullscreenSupported && <button type="button" onClick={toggleFullscreen} aria-label="Toggle fullscreen" title="Fullscreen"><Maximize2 size={18} /></button>}
        </div>
        <MapLegend />
      </section>
    )
  },
)

import { useEffect, useMemo, useRef, useState } from 'react'
import { Anchor, Crosshair, Loader2, MapPinned, RadioTower, ScanSearch, X } from 'lucide-react'
import 'maplibre-gl/dist/maplibre-gl.css'
import './intelligence-map.css'
import { ApiError, getSurvey, getSurveyDetections, getSurveySummary, listSurveys, type ApiSurvey } from '../../lib/api'
import { debrisClasses, priorityLevels, type Detection, type MapFiltersState, type Survey, type SurveySummaryData } from './types'
import { apiDetectionToLocal } from './services/mapDataAdapter'
import { SurveySummary } from './components/SurveySummary'
import { MapFilters } from './components/MapFilters'
import { DetectionDetails } from './components/DetectionDetails'
import { MarineIntelligenceMap, type MarineIntelligenceMapHandle } from './components/MarineIntelligenceMap'

const defaultFilters: MapFiltersState = {
  classes: [...debrisClasses],
  priorities: [...priorityLevels],
  minimumConfidence: 0,
}

export default function IntelligenceMapPage() {
  const [filters, setFilters] = useState<MapFiltersState>(defaultFilters)
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const mapRef = useRef<MarineIntelligenceMapHandle>(null)

  const [surveys, setSurveys] = useState<ApiSurvey[]>([])
  const [surveysLoading, setSurveysLoading] = useState(true)
  const [surveysError, setSurveysError] = useState<string | null>(null)
  const [selectedSurveyId, setSelectedSurveyId] = useState<string | null>(null)

  const [survey, setSurvey] = useState<Survey | null>(null)
  const [detections, setDetections] = useState<Detection[]>([])
  const [summary, setSummary] = useState<SurveySummaryData | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)

  useEffect(() => {
    document.title = 'Marine Debris Intelligence Map · SonarSense'
  }, [])

  useEffect(() => {
    let cancelled = false
    setSurveysLoading(true); setSurveysError(null)
    listSurveys()
      .then(all => {
        if (cancelled) return
        setSurveys(all)
        setSelectedSurveyId(prev => prev ?? all[0]?.id ?? null)
      })
      .catch(err => { if (!cancelled) setSurveysError(err instanceof ApiError ? err.message : 'Could not reach the backend.') })
      .finally(() => { if (!cancelled) setSurveysLoading(false) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!selectedSurveyId) return
    let cancelled = false
    setDetailLoading(true); setDetailError(null)
    Promise.all([getSurvey(selectedSurveyId), getSurveyDetections(selectedSurveyId), getSurveySummary(selectedSurveyId)])
      .then(([surveyData, rawDetections, summaryData]) => {
        if (cancelled) return
        setSurvey(surveyData)
        setDetections(rawDetections.map(apiDetectionToLocal).filter((d): d is Detection => d != null))
        setSummary(summaryData)
      })
      .catch(err => { if (!cancelled) setDetailError(err instanceof ApiError ? err.message : 'Could not load this survey.') })
      .finally(() => { if (!cancelled) setDetailLoading(false) })
    return () => { cancelled = true }
  }, [selectedSurveyId])

  const filteredDetections = useMemo(() => detections.filter((detection) =>
    filters.classes.includes(detection.classification)
    && filters.priorities.includes(detection.priority)
    && detection.confidence >= filters.minimumConfidence,
  ), [detections, filters])

  useEffect(() => {
    if (selectedDetection && !filteredDetections.some((item) => item.id === selectedDetection.id)) {
      setSelectedDetection(null)
    }
  }, [filteredDetections, selectedDetection])

  return (
    <div className="ss-app">
      <a className="ss-skip-link" href="#intelligence-map">Skip to intelligence map</a>
      <header className="ss-topbar">
        <a className="ss-brand" href="/intelligence-map" aria-label="SonarSense intelligence map home">
          <span className="ss-brand__mark" aria-hidden="true"><Anchor size={20} /></span>
          <span><strong>SonarSense</strong><small>Marine intelligence</small></span>
        </a>
        <div className="ss-topbar__context">
          <span className="ss-live-dot" aria-hidden="true" />
          <span>Intelligence Map</span>
          <span className="ss-topbar__divider" />
          {surveys.length > 1 ? (
            <select className="ss-survey-picker" value={selectedSurveyId ?? ''} onChange={event => setSelectedSurveyId(event.target.value)} aria-label="Select survey">
              {surveys.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          ) : <span>{survey?.region ?? ''}</span>}
        </div>
        <nav aria-label="Page utilities">
          <a href="/" className="ss-back-link">Back to dashboard</a>
        </nav>
      </header>

      {(() => {
        if (surveysLoading) {
          return <main id="intelligence-map" className="ss-main-empty"><div className="ss-empty" role="status"><Loader2 size={22} className="ds-spin" aria-hidden="true" /><strong>Loading surveys…</strong></div></main>
        }
        if (surveysError) {
          return <main id="intelligence-map" className="ss-main-empty"><div className="ss-empty" role="status"><RadioTower size={22} aria-hidden="true" /><strong>Could not reach the backend</strong><span>{surveysError}</span></div></main>
        }
        if (surveys.length === 0) {
          return <main id="intelligence-map" className="ss-main-empty"><div className="ss-empty" role="status">
            <ScanSearch size={22} aria-hidden="true" />
            <strong>No completed surveys yet</strong>
            <span>Upload and process a sonar log from the dashboard to see it appear here as a geolocated survey.</span>
            <a href="/upload">Go to Upload</a>
          </div></main>
        }
        if (detailLoading || !survey || !summary) {
          return <main id="intelligence-map" className="ss-main-empty"><div className="ss-empty" role="status"><Loader2 size={22} className="ds-spin" aria-hidden="true" /><strong>Loading survey detail…</strong></div></main>
        }
        if (detailError) {
          return <main id="intelligence-map" className="ss-main-empty"><div className="ss-empty" role="status"><RadioTower size={22} aria-hidden="true" /><strong>Could not load this survey</strong><span>{detailError}</span></div></main>
        }
        return (
          <main id="intelligence-map" className="ss-main">
            <SurveySummary survey={survey} summary={summary} />
            <div className="ss-workspace">
              <MarineIntelligenceMap
                ref={mapRef}
                survey={survey}
                detections={filteredDetections}
                allDetections={detections}
                onSelect={setSelectedDetection}
              />
              <MapFilters
                filters={filters}
                open={filtersOpen}
                onOpenChange={setFiltersOpen}
                onChange={setFilters}
                onReset={() => setFilters(defaultFilters)}
              />
              <button type="button" className="ss-fit-button" onClick={() => mapRef.current?.fitSurvey()}>
                <Crosshair size={16} /> Fit Survey
              </button>
              {filteredDetections.length === 0 && (
                <div className="ss-empty" role="status">
                  <RadioTower size={22} aria-hidden="true" />
                  <strong>No detections match these filters</strong>
                  <span>The survey track remains visible. Reset or broaden the filters to restore detections.</span>
                  <button type="button" onClick={() => setFilters(defaultFilters)}>Reset filters</button>
                </div>
              )}
              <DetectionDetails detection={selectedDetection} onClose={() => setSelectedDetection(null)} />
              {selectedDetection && <button className="ss-details-scrim" onClick={() => setSelectedDetection(null)} aria-label="Close detection details"><X /></button>}
            </div>
          </main>
        )
      })()}
      <footer className="ss-statusbar">
        <span><MapPinned size={14} /> WGS 84</span>
        <span>Coastal survey extent</span>
        <span>{survey ? `${filteredDetections.length} of ${detections.length} detections visible` : 'No survey loaded'}</span>
      </footer>
    </div>
  )
}

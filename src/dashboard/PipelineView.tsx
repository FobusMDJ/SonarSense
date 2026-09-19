import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { AlertTriangle, ChevronLeft, ChevronRight, Loader2, MapPin, ScanSearch, ShieldAlert } from 'lucide-react'
import {
  ApiError, getDetections, getModelMetadata, getModelStats, getVaeStats, vaeFrameImageUrl, VAE_PANEL_FILES,
  type ModelMetadata, type ModelOutputStats, type RawDetection, type VaeStats, type LogSummary,
} from '../lib/api'
import { colorForType, displayClassification, isHumanClass } from './data'

const AnomalySurface3D = lazy(() => import('./AnomalySurface3D'))

interface PipelineViewProps {
  logs: LogSummary[]
  selectedLogId: string | null
  onSelectLog: (id: string) => void
}

const STEP_DEFS = [
  { key: 'input', title: 'Preprocessed input frame', file: '01_original.png' as const,
    caption: 'What the models actually saw after denoising, contrast enhancement, and resize. This is not the raw sensor tile.' },
  { key: 'recon', title: 'VAE reconstruction', file: '02_reconstruction.png' as const,
    caption: 'The autoencoder’s best attempt at reproducing the input from a compressed latent representation.' },
  { key: 'overlay', title: 'Anomaly overlay', file: '03_anomaly_overlay.png' as const,
    caption: 'Reconstruction-error heatmap blended over the original frame, showing where the seafloor looks unusual in context.' },
  { key: 'diff', title: 'Difference heatmap', file: '04_difference_heatmap.png' as const,
    caption: 'The colorized |original − reconstruction| error map alone, no blending.' },
  { key: 'edges', title: 'Edge / contour map', file: '05_edge_contour_map.png' as const,
    caption: 'Boundaries of the thresholded anomalous region, useful for reading shape rather than a filled area.' },
  { key: 'legend', title: 'Difference map (calibrated)', file: '06_difference_map_legend.png' as const,
    caption: 'The same error map with a High/Low colorbar, so a value can be read off rather than just eyeballed.' },
  { key: 'surface', title: '3D anomaly surface', file: '07_3d_anomaly_surface.png' as const,
    caption: 'Reconstruction error plotted as a height field. Peaks show where the VAE was most surprised.' },
] as const

function BoundingBoxes({ detections, imageSrc }: { detections: RawDetection[]; imageSrc: string }) {
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null)
  return (
    <div className="ds-bbox-layer" style={{ position: 'relative' }}>
      <img
        src={imageSrc}
        alt="Preprocessed frame with YOLO26 detection boxes"
        onLoad={event => setNaturalSize({ w: event.currentTarget.naturalWidth, h: event.currentTarget.naturalHeight })}
      />
      {naturalSize && (
        <svg className="ds-bbox-svg" viewBox={`0 0 ${naturalSize.w} ${naturalSize.h}`} preserveAspectRatio="none"
             style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
          {detections.map(det => {
            const color = colorForType(displayClassification(det.class_name))
            const x = det.bbox_x1, y = det.bbox_y1, w = det.bbox_x2 - det.bbox_x1, h = det.bbox_y2 - det.bbox_y1
            return (
              <g key={det.id}>
                <rect x={x} y={y} width={w} height={h} fill="none" stroke={color} strokeWidth={Math.max(2, naturalSize.w / 180)} />
                <text x={x} y={Math.max(12, y - 6)} fill={color} fontSize={Math.max(14, naturalSize.w / 32)} fontWeight={700}>
                  {det.class_name} {Math.round(det.confidence_score)}%
                </text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}

function ConfidenceBreakdown({ detection }: { detection: RawDetection }) {
  const breakdown = detection.confidence_breakdown ?? {}
  const entries = Object.entries(breakdown)
  return (
    <div className="ds-confidence-card">
      <header>
        <strong>{detection.class_name}</strong>
        <span className={`ds-priority ${detection.confidence_label}`}>{detection.confidence_label} confidence</span>
      </header>
      <div className="ds-confidence-score"><b>{detection.confidence_score.toFixed(1)}</b><small>fused score</small></div>
      {entries.length > 0 ? (
        <div className="ds-evidence-bars">
          {entries.map(([label, value]) => (
            <div key={label}>
              <span><b>{label.replace(/_/g, ' ')}</b><em>{typeof value === 'number' ? value.toFixed(1) : String(value)}</em></span>
              <div className="ds-bar"><i style={{ width: `${Math.max(0, Math.min(100, Number(value)))}%`, background: colorForType(displayClassification(detection.class_name)) }} /></div>
            </div>
          ))}
        </div>
      ) : <p className="ds-panel-note">No confidence breakdown was recorded for this detection.</p>}
      {detection.vae_box_error != null && (
        <p className="ds-panel-note">VAE error for this box's crop: <b>{detection.vae_box_error.toFixed(5)}</b></p>
      )}
    </div>
  )
}

function GeolocationCard({ detection }: { detection: RawDetection }) {
  const located = detection.geo_method === 'nav_fix' && detection.lat != null && detection.lon != null
  return (
    <div className="ds-geolocation-card">
      <header><MapPin size={17} /><strong>{detection.class_name}</strong></header>
      {located ? (
        <dl>
          <div><dt>Latitude</dt><dd>{detection.lat!.toFixed(5)}°</dd></div>
          <div><dt>Longitude</dt><dd>{detection.lon!.toFixed(5)}°</dd></div>
          <div><dt>Depth</dt><dd>{detection.depth_m != null ? `${detection.depth_m.toFixed(1)} m` : 'Not available'}</dd></div>
          <div><dt>Position basis</dt><dd>Navigation fix</dd></div>
        </dl>
      ) : (
        <p className="ds-panel-note">No navigation fix was available for this frame ({detection.geo_method ?? 'unknown'}), so this detection has no real-world position. This is honestly reported rather than plotted at a placeholder coordinate.</p>
      )}
    </div>
  )
}

export function PipelineView({ logs, selectedLogId, onSelectLog }: PipelineViewProps) {
  const doneLogs = useMemo(() => logs.filter(l => l.status === 'done'), [logs])
  const [detections, setDetections] = useState<RawDetection[]>([])
  const [modelStats, setModelStats] = useState<ModelOutputStats | null>(null)
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null)
  const [humanDetections, setHumanDetections] = useState<RawDetection[]>([])
  const [vaeStats, setVaeStats] = useState<VaeStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [step, setStep] = useState(0)

  useEffect(() => {
    if (!selectedLogId) return
    let cancelled = false
    setLoading(true); setError(null)
    Promise.all([getDetections(selectedLogId), getModelStats(selectedLogId), getVaeStats(selectedLogId), getModelMetadata()])
      .then(([dets, stats, vae, modelMetadata]) => {
        if (cancelled) return
        setDetections(dets.filter(d => !isHumanClass(d.class_name)))
        setHumanDetections(dets.filter(d => isHumanClass(d.class_name)))
        setModelStats(stats); setVaeStats(vae); setMetadata(modelMetadata); setFrameIndex(0); setStep(0)
      })
      .catch(err => { if (!cancelled) setError(err instanceof ApiError ? err.message : 'Failed to load pipeline data.') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [selectedLogId])

  const frames = useMemo(() => {
    const byFrame = new Map<string, RawDetection[]>()
    for (const d of detections) {
      const list = byFrame.get(d.frame_record_id) ?? []
      list.push(d)
      byFrame.set(d.frame_record_id, list)
    }
    return Array.from(byFrame.entries()).map(([frameRecordId, dets]) => ({
      frameRecordId,
      frameIndex: dets[0].frame_index,
      detections: dets,
    })).sort((a, b) => a.frameIndex - b.frameIndex)
  }, [detections])

  if (doneLogs.length === 0) {
    return (
      <div className="ds-no-results">
        <ScanSearch size={26} />
        <strong>No processed logs yet</strong>
        <p>Upload a sonar log to see the full pipeline: preprocessing, YOLO26 detection, VAE anomaly analysis, confidence scoring, and geolocation, frame by frame.</p>
      </div>
    )
  }

  const frame = frames[frameIndex]

  return (
    <>
      <div className="ds-page-intro">
        <div><h2>Processing pipeline</h2><p>Step through exactly what happened to one frame, in order: preprocessing, VAE anomaly analysis, YOLO26 detection, confidence scoring, and geolocation.</p></div>
        <label className="ds-log-picker">
          <span>Log</span>
          <select value={selectedLogId ?? ''} onChange={event => onSelectLog(event.target.value)}>
            {doneLogs.map(log => <option key={log.id} value={log.id}>{log.filename} · {log.n_detections} detections</option>)}
          </select>
        </label>
      </div>

      {loading && <div className="ds-panel-note"><Loader2 size={16} className="ds-spin" /> Loading pipeline data…</div>}
      {error && <div className="ds-panel-note ds-error-text"><AlertTriangle size={15} /> {error}</div>}

      {!loading && !error && (
        <>
          <section className="ds-run-strip">
            <div><span>Processed log</span><strong>{doneLogs.find(l => l.id === selectedLogId)?.filename}</strong></div>
            <div><span>Frames analyzed (VAE)</span><strong>{vaeStats?.n_frames_analyzed ?? 0}</strong></div>
            <div><span>Total detections</span><strong>{modelStats?.n_detections ?? 0}</strong></div>
            <div><span>Low-confidence</span><strong>{modelStats?.n_low_confidence ?? 0} / {modelStats?.low_confidence_threshold ?? 40}%</strong></div>
          </section>

          <section className="ds-panel ds-vae-stats-panel">
            <header className="ds-panel-title"><h2>VAE anomaly statistics</h2></header>
            {vaeStats && vaeStats.n_frames_analyzed > 0 ? (
              <>
                <dl className="ds-vae-stats">
                  <div><dt>Mean whole-image error</dt><dd>{vaeStats.mean_whole_image_error.toFixed(6)}</dd></div>
                  <div><dt>Min error</dt><dd>{vaeStats.min_whole_image_error.toFixed(6)}</dd></div>
                  <div><dt>Max error</dt><dd>{vaeStats.max_whole_image_error.toFixed(6)}</dd></div>
                  <div><dt>Frames analyzed</dt><dd>{vaeStats.n_frames_analyzed}</dd></div>
                </dl>
                <p className="ds-panel-note">Most anomalous frames (lowest percentile = most unusual relative to this log's own frames):</p>
                <div className="ds-anomalous-frames">
                  {vaeStats.most_anomalous_frames.slice(0, 6).map(f => (
                    <button key={f.frame_record_id} className={frame?.frameRecordId === f.frame_record_id ? 'selected' : ''}
                            onClick={() => { const idx = frames.findIndex(fr => fr.frameRecordId === f.frame_record_id); if (idx >= 0) { setFrameIndex(idx); setStep(0) } }}>
                      <img src={vaeFrameImageUrl(selectedLogId!, f.frame_record_id, '03_anomaly_overlay.png')} alt="" />
                      <span>err {f.whole_image_error?.toFixed(4)}</span>
                    </button>
                  ))}
                </div>
              </>
            ) : <p className="ds-panel-note">No VAE analysis recorded for this log yet.</p>}
          </section>

          <section className="ds-panel ds-model-panel">
            <header className="ds-panel-title"><h2>Model output</h2></header>
            {metadata && (
              <>
                <div className="ds-metric-section-head">
                  <div><strong>Training validation</strong><span>Published checkpoint metrics, not accuracy for this uploaded log.</span></div>
                  <small>{metadata.detector.name} · {metadata.release_tag} · {metadata.device.toUpperCase()}</small>
                </div>
                <div className="ds-model-metrics">
                  {[
                    ['mAP@50', metadata.detector.training_metrics.map50],
                    ['mAP@50–95', metadata.detector.training_metrics.map50_95],
                    ['Precision', metadata.detector.training_metrics.precision],
                    ['Recall', metadata.detector.training_metrics.recall],
                    ['F1 score', metadata.detector.training_metrics.f1_score],
                  ].map(([label, value]) => (
                    <article key={String(label)}><div><span>{label}</span><small>training</small></div><strong>{(Number(value) * 100).toFixed(1)}%</strong></article>
                  ))}
                  <article><div><span>Training images</span><small>training</small></div><strong>Unavailable</strong></article>
                </div>
                <div className="ds-model-context" aria-label="Released detector classes">
                  <span>Raw detector classes</span>
                  {metadata.detector.raw_classes.map(name => <b key={name}>{name}</b>)}
                  <span className="ds-model-version">Ultralytics {metadata.detector.framework_version} · {metadata.detector.training_epochs} epochs · checkpoint {metadata.detector.checkpoint.verified ? 'verified' : 'not verified'}</span>
                </div>
              </>
            )}
            <div className="ds-metric-section-head">
              <div><strong>This processed log</strong><span>Measured during local inference.</span></div>
            </div>
            <div className="ds-model-metrics">
              <article><div><span>Confidence threshold</span><small>runtime</small></div><strong>{modelStats?.confidence_threshold != null ? `${Math.round(modelStats.confidence_threshold * 100)}%` : 'Unavailable'}</strong></article>
              <article><div><span>Mean inference</span><small>runtime</small></div><strong>{modelStats?.mean_inference_ms != null ? `${modelStats.mean_inference_ms.toFixed(1)} ms` : 'Unavailable'}</strong></article>
              <article><div><span>Throughput</span><small>runtime</small></div><strong>{modelStats?.fps != null ? `${modelStats.fps.toFixed(1)} FPS` : 'Unavailable'}</strong></article>
              <article><div><span>Number of classes</span><small>checkpoint</small></div><strong>{metadata?.detector.raw_classes.length ?? 'Unavailable'}</strong></article>
            </div>
            {modelStats && modelStats.n_detections > 0 ? (
              <div className="ds-model-metrics ds-class-counts">
                {Object.entries(modelStats.raw_counts_by_class).map(([cls, count]) => (
                  <article key={cls}>
                    <div><span>{cls}</span><small>raw class</small></div>
                    <strong>{count}</strong>
                    <p>mean fused confidence {modelStats.mean_confidence_by_class[cls]?.toFixed(1)}%</p>
                  </article>
                ))}
              </div>
            ) : <p className="ds-panel-note">No detections recorded for this log.</p>}
          </section>

          <section className="ds-panel ds-safety-panel" aria-labelledby="human-safety-title">
            <header className="ds-panel-title"><h2 id="human-safety-title"><ShieldAlert size={17} /> Human safety review</h2></header>
            {humanDetections.length ? (
              <div className="ds-safety-list" role="alert">
                <p><strong>{humanDetections.length} human detection{humanDetections.length === 1 ? '' : 's'} require operator review.</strong> Human results are never grouped with marine debris.</p>
                {humanDetections.map(det => (
                  <article key={det.id}>
                    <b>Frame {det.frame_index + 1}</b>
                    <span>{det.confidence_score.toFixed(1)}% fused confidence</span>
                    <span>{det.lat != null && det.lon != null ? `${det.lat.toFixed(5)}°, ${det.lon.toFixed(5)}°` : 'No navigation position available'}</span>
                  </article>
                ))}
              </div>
            ) : <p className="ds-panel-note">No human-class safety detections were recorded for this log.</p>}
          </section>

          {frame ? (
            <section className="ds-panel ds-pipeline-panel">
              <header className="ds-panel-title">
                <h2>Frame walkthrough</h2>
                <div className="ds-frame-nav">
                  <button className="ds-icon-button" disabled={frameIndex === 0} onClick={() => { setFrameIndex(i => i - 1); setStep(0) }}><ChevronLeft size={17} /></button>
                  <span>Frame {frameIndex + 1} / {frames.length} · {frame.detections.length} detection(s)</span>
                  <button className="ds-icon-button" disabled={frameIndex === frames.length - 1} onClick={() => { setFrameIndex(i => i + 1); setStep(0) }}><ChevronRight size={17} /></button>
                </div>
              </header>

              <div className="ds-pipeline-tabs" role="tablist" aria-label="Pipeline steps">
                {STEP_DEFS.map((s, i) => <button key={s.key} role="tab" aria-selected={step === i} onClick={() => setStep(i)}>{i + 1}. {s.title}</button>)}
                <button role="tab" aria-selected={step === STEP_DEFS.length} onClick={() => setStep(STEP_DEFS.length)}>{STEP_DEFS.length + 1}. YOLO26 detections</button>
                <button role="tab" aria-selected={step === STEP_DEFS.length + 1} onClick={() => setStep(STEP_DEFS.length + 1)}>{STEP_DEFS.length + 2}. Confidence scoring</button>
                <button role="tab" aria-selected={step === STEP_DEFS.length + 2} onClick={() => setStep(STEP_DEFS.length + 2)}>{STEP_DEFS.length + 3}. Geolocation</button>
              </div>

              <div className="ds-pipeline-stage">
                {step < STEP_DEFS.length && (
                  <>
                    {STEP_DEFS[step].key === 'surface' ? (
                      <Suspense fallback={<div className="ds-surface-loading">Loading interactive 3D renderer…</div>}>
                        <AnomalySurface3D
                          logId={selectedLogId!}
                          frameRecordId={frame.frameRecordId}
                          fallbackUrl={vaeFrameImageUrl(selectedLogId!, frame.frameRecordId, '07_3d_anomaly_surface.png')}
                        />
                      </Suspense>
                    ) : (
                      <img src={vaeFrameImageUrl(selectedLogId!, frame.frameRecordId, STEP_DEFS[step].file)} alt={STEP_DEFS[step].title} />
                    )}
                    <div className="ds-view-caption"><strong>{STEP_DEFS[step].title}</strong><span>{STEP_DEFS[step].caption}</span></div>
                  </>
                )}
                {step === STEP_DEFS.length && (
                  <>
                    <BoundingBoxes detections={frame.detections} imageSrc={vaeFrameImageUrl(selectedLogId!, frame.frameRecordId, '01_original.png')} />
                    <div className="ds-view-caption"><strong>YOLO26 detections</strong><span>Every bounding box YOLO26 produced for this frame, at the confidence threshold this log was processed with.</span></div>
                  </>
                )}
                {step === STEP_DEFS.length + 1 && (
                  <div className="ds-confidence-grid">
                    {frame.detections.map(d => <ConfidenceBreakdown key={d.id} detection={d} />)}
                  </div>
                )}
                {step === STEP_DEFS.length + 2 && (
                  <div className="ds-geolocation-grid">
                    {frame.detections.map(d => <GeolocationCard key={d.id} detection={d} />)}
                  </div>
                )}
              </div>
            </section>
          ) : (
            <div className="ds-no-results">
              <ScanSearch size={24} />
              <strong>No detections in this log to walk through</strong>
              <p>This log processed successfully, but YOLO26 found no objects above its confidence threshold. VAE statistics above still reflect every frame analyzed.</p>
            </div>
          )}
        </>
      )}
    </>
  )
}

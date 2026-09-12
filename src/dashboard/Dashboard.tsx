import { useEffect, useMemo, useRef, useState, type FormEvent, type MouseEvent, type ReactNode } from 'react'
import {
  ArrowRight, Bell, CalendarDays, Check, CircleGauge, CloudUpload, Download, FileText, Layers,
  Map as MapIcon, MapPin, Menu, PanelTop, Radar, RefreshCw, Search, Settings, Shield, ShieldCheck, Waves,
  ChartNoAxesCombined, X, AlertTriangle, Loader2,
} from 'lucide-react'
import { getHealth, reportUrls, vaeFrameImageUrl, type HealthResponse } from '../lib/api'
import { colorForType, detectionCsv, downloadCsv, formatTime, type DetectionRecord } from './data'
import { AnomalyPreview, ConfidenceChart, TimelineChart, TypeChart, type TimelinePoint } from './Charts'
import { DetectionMap } from './DetectionMap'
import { PipelineView } from './PipelineView'
import { UploadView } from './UploadView'
import { useSonarData, type SonarData } from './useSonarData'
import type { LogSummary } from '../lib/api'
import './dashboard.css'

const navigation = [
  { label: 'Overview', path: '/', icon: Shield },
  { label: 'Upload', path: '/upload', icon: CloudUpload },
  { label: 'Detections', path: '/detections', icon: PanelTop },
  { label: 'Map', path: '/map', icon: MapIcon },
  { label: 'Pipeline', path: '/analysis', icon: ChartNoAxesCombined },
  { label: 'Processed Logs', path: '/missions', icon: ShieldCheck },
  { label: 'System Health', path: '/data-quality', icon: CircleGauge },
  { label: 'Reports', path: '/reports', icon: FileText },
  { label: 'Backend Status', path: '/system-status', icon: Radar },
  { label: 'Settings', path: '/settings', icon: Settings },
]
type Navigate = (path: string, event?: MouseEvent<HTMLAnchorElement>) => void

function Panel({ title, children, className = '', action }: { title: string; children: ReactNode; className?: string; action?: ReactNode }) {
  return <section className={`ds-panel ${className}`}><header className="ds-panel-title"><h2>{title}</h2>{action}</header>{children}</section>
}
function Link({ to, children, navigate }: { to: string; children: ReactNode; navigate: Navigate }) {
  return <a href={to} className="ds-text-link" onClick={event => navigate(to, event)}>{children}<ArrowRight size={15}/></a>
}

function DetectionTable({ items, onSelect, extended = false }: { items: DetectionRecord[]; onSelect: (item: DetectionRecord) => void; extended?: boolean }) {
  return <div className="ds-table-scroll"><table className="ds-table"><thead><tr>{['ID', 'Type', 'Confidence', 'Depth', 'Location', 'Time', 'Priority', ...(extended ? ['Log'] : [])].map(title => <th key={title} scope="col">{title}</th>)}</tr></thead><tbody>{items.map(item => <tr key={item.id}>
    <td><button className="ds-id-button" onClick={() => onSelect(item)} aria-label={`Open detection ${item.id}`}>{item.id.slice(0, 8)}</button></td>
    <td><span className="ds-type-label"><i style={{ background: colorForType(item.type) }}/>{item.type}</span></td><td>{item.confidence}%</td><td>{item.depth ? `${item.depth.toFixed(1)} m` : 'n/a'}</td>
    <td>{item.located ? <>{item.latitude.toFixed(4)}° N, {item.longitude.toFixed(4)}° E</> : 'No GPS fix'}</td><td>{formatTime(item.timestamp)}</td><td><span className={`ds-priority ${item.priority.toLowerCase()}`}>{item.priority}</span></td>{extended && <td>{item.mission}</td>}
  </tr>)}</tbody></table>{!items.length && <div className="ds-no-results"><Search size={25}/><strong>No detections found</strong><p>Try a different search or reset the filters, or upload a sonar log to get started.</p></div>}</div>
}

function MetricCards({ items, logsCount, framesCount }: { items: DetectionRecord[]; logsCount: number; framesCount: number }) {
  const total = items.length
  const high = items.filter(item => item.priority === 'High').length
  const avgConfidence = total ? Math.round(items.reduce((sum, item) => sum + item.confidence, 0) / total) : 0
  const cards = [
    { title: 'Total Detections', value: String(total) },
    { title: 'High Priority', value: String(high) },
    { title: 'Avg. Confidence', value: `${avgConfidence}%` },
    { title: 'Frames Processed', value: String(framesCount) },
    { title: 'Processed Logs', value: String(logsCount) },
  ]
  return <section className="ds-metrics" aria-label="Survey metrics">{cards.map(card => <div className="ds-metric" key={card.title}>
    <span>{card.title}</span><strong>{card.value}</strong>
    <p>Live from the backend</p>
  </div>)}</section>
}

function Overview({ items, logsCount, framesCount, timeline, anomalousFrames, meanError, navigate, onSelect }: {
  items: DetectionRecord[]; logsCount: number; framesCount: number; timeline: TimelinePoint[]
  anomalousFrames: SonarData['topAnomalousFrames']; meanError: number | null
  navigate: Navigate; onSelect: (item: DetectionRecord) => void
}) {
  const recent = [...items].sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)).slice(0, 5)
  return <><MetricCards items={items} logsCount={logsCount} framesCount={framesCount}/><div className="ds-overview-grid">
    <div className="ds-chart-pair"><Panel title="Detections by Type"><TypeChart items={items}/></Panel><Panel title="Detections Over Time"><TimelineChart series={timeline}/></Panel></div>
    <Panel title="Detection Map" className="ds-overview-map"><DetectionMap items={items} onSelect={onSelect}/></Panel>
    <Panel title="Recent Detections" className="ds-recent"><DetectionTable items={recent} onSelect={onSelect}/><footer><Link to="/detections" navigate={navigate}>View all Detections</Link></footer></Panel>
    <div className="ds-bottom-pair"><Panel title="Anomaly Overview (VAE)"><AnomalyPreview frames={anomalousFrames} meanError={meanError}/></Panel><Panel title="Detection Confidence Distribution"><ConfidenceChart items={items}/></Panel></div>
    <Panel title="Processed Logs" className="ds-mission-summary"><footer><Link to="/missions" navigate={navigate}>View all processed logs</Link></footer></Panel>
  </div></>
}

function DetectionsView({ items, onSelect }: { items: DetectionRecord[]; onSelect: (item: DetectionRecord) => void }) {
  const [query, setQuery] = useState(''), [type, setType] = useState('All types'), [priority, setPriority] = useState('All priorities'), [page, setPage] = useState(0)
  useEffect(() => setPage(0), [items])
  const filtered = items.filter(item => (type === 'All types' || item.type === type) && (priority === 'All priorities' || item.priority === priority) && `${item.id} ${item.type} ${item.mission}`.toLowerCase().includes(query.toLowerCase()))
  const reset = () => { setQuery(''); setType('All types'); setPriority('All priorities'); setPage(0) }
  return <><div className="ds-page-intro"><div><h2>Detection register</h2><p>Inspect and filter the {items.length} detections found across every processed log.</p></div><button className="ds-button" onClick={() => downloadCsv('sonarsense-filtered-detections.csv', detectionCsv(filtered))}><Download size={16}/> Export CSV</button></div>
    <Panel title={`${filtered.length} detections`}><div className="ds-toolbar"><label className="ds-search"><Search size={16}/><input value={query} onChange={event => { setQuery(event.target.value); setPage(0) }} placeholder="Search ID, type, or log" aria-label="Search detections"/></label>
      <select aria-label="Filter by type" value={type} onChange={event => { setType(event.target.value); setPage(0) }}><option>All types</option>{['Pipe', 'Ghost Net', 'Shipwreck', 'Cylinder', 'Other Debris'].map(name => <option key={name}>{name}</option>)}</select>
      <select aria-label="Filter by priority" value={priority} onChange={event => { setPriority(event.target.value); setPage(0) }}><option>All priorities</option>{['High', 'Medium', 'Low'].map(level => <option key={level}>{level}</option>)}</select><button className="ds-button" onClick={reset}>Reset</button></div>
      <DetectionTable items={filtered.slice(page * 15, (page + 1) * 15)} onSelect={onSelect} extended/>
      <div className="ds-pagination"><span>{filtered.length ? page * 15 + 1 : 0}–{Math.min((page + 1) * 15, filtered.length)} of {filtered.length}</span><button disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button><button disabled={(page + 1) * 15 >= filtered.length} onClick={() => setPage(page + 1)}>Next</button></div>
    </Panel></>
}

function MapView({ items, onSelect, navigate }: { items: DetectionRecord[]; onSelect: (item: DetectionRecord) => void; navigate: Navigate }) {
  const [type, setType] = useState('All types')
  const filtered = items.filter(item => type === 'All types' || item.type === type)
  const [located, setLocated] = useState<DetectionRecord | null>(items.find(i => i.located) ?? null)
  useEffect(() => { if (!located || !filtered.some(i => i.id === located.id)) setLocated(filtered.find(i => i.located) ?? null) }, [filtered]) // eslint-disable-line react-hooks/exhaustive-deps
  const inspect = (item: DetectionRecord) => { setLocated(item); onSelect(item) }
  return <><div className="ds-page-intro"><div><h2>Survey detection map</h2><p>Real detections geolocated from navigation fixes recorded during processing. Select a marker to inspect it.</p></div></div>
    <div className="ds-map-workspace"><Panel title="GPS-tagged detections" action={<select value={type} onChange={event => setType(event.target.value)} aria-label="Map detection type"><option>All types</option>{['Pipe', 'Ghost Net', 'Shipwreck', 'Cylinder', 'Other Debris'].map(name => <option key={name}>{name}</option>)}</select>}><DetectionMap full items={filtered} onSelect={inspect}/></Panel>
      <Panel title="Geolocation result" className="ds-geolocation"><div className="ds-geo-state"><MapPin size={20}/><div><span>Position quality</span><strong>{located?.located ? 'Real navigation fix' : 'No navigation fix'}</strong></div></div>{located && <dl><div><dt>Detection</dt><dd>#{located.id.slice(0, 8)} · {located.type}</dd></div><div><dt>Latitude</dt><dd>{located.located ? `${located.latitude.toFixed(4)}° N` : 'Unavailable'}</dd></div><div><dt>Longitude</dt><dd>{located.located ? `${located.longitude.toFixed(4)}° E` : 'Unavailable'}</dd></div><div><dt>Depth</dt><dd>{located.depth ? `${located.depth.toFixed(1)} m` : 'Unavailable'}</dd></div><div><dt>Source log</dt><dd>{located.mission}</dd></div></dl>}{!filtered.some(i => i.located) && <p className="ds-panel-note">None of the current detections have a real navigation fix -- upload a log with a nav sidecar (or an XTF with nav headers) to see real positions here.</p>}<footer><Link to="/detections" navigate={navigate}>Open detection records</Link></footer></Panel>
    </div></>
}

function LogsTable({ logs }: { logs: LogSummary[] }) {
  return <div className="ds-table-scroll"><table className="ds-table ds-mission-table"><thead><tr>{['Log', 'Uploaded', 'Completed', 'Frames', 'Detections', 'Status'].map(title => <th key={title} scope="col">{title}</th>)}</tr></thead><tbody>{logs.map(log => <tr key={log.id}><td>{log.filename}</td><td>{formatTime(log.uploaded_at)}</td><td>{log.completed_at ? formatTime(log.completed_at) : '—'}</td><td>{log.n_frames}</td><td>{log.n_detections}</td><td className={log.status === 'done' ? 'ds-green' : log.status === 'error' ? 'ds-error-text' : ''}>{log.status}{log.error_message ? ` — ${log.error_message}` : ''}</td></tr>)}</tbody></table>{!logs.length && <div className="ds-no-results"><strong>No logs processed yet</strong><p>Upload a sonar log to get started.</p></div>}</div>
}

function QualityView({ logs }: { logs: LogSummary[] }) {
  const done = logs.filter(l => l.status === 'done')
  const withDenoise = done.filter(l => l.denoise_method && l.denoise_method !== 'none').length
  const totalDetections = done.reduce((sum, l) => sum + l.n_detections, 0)
  const totalFrames = done.reduce((sum, l) => sum + l.n_frames, 0)
  return <><div className="ds-page-intro"><div><h2>System &amp; data quality</h2><p>Real processing statistics across every completed log -- no simulated figures.</p></div></div>
    <div className="ds-quality-stats">
      {[
        ['Completed logs', String(done.length), `${logs.length} total submitted`],
        ['Frames processed', String(totalFrames), 'Across all completed logs'],
        ['Detections recorded', String(totalDetections), totalFrames ? `${(totalDetections / totalFrames).toFixed(2)} per frame` : 'No frames yet'],
        ['Logs using denoising', String(withDenoise), done.length ? `${Math.round(withDenoise / done.length * 100)}% of completed logs` : 'n/a'],
      ].map(([title, value, caption]) => <Panel key={title} title={title}><div className="ds-big-value">{value}</div><p className="ds-panel-note">{caption}</p></Panel>)}
    </div>
    <Panel title="Every processed log"><LogsTable logs={logs}/></Panel></>
}

function ReportsView({ logs, items }: { logs: LogSummary[]; items: DetectionRecord[] }) {
  const done = logs.filter(l => l.status === 'done')
  return <><div className="ds-page-intro"><div><h2>Survey reports</h2><p>Download the backend's real generated reports for any completed log, or export the current combined detection register as CSV.</p></div><button className="ds-button" onClick={() => downloadCsv('sonarsense-detections.csv', detectionCsv(items))}><Download size={16}/> Export all detections (CSV)</button></div>
    <div className="ds-report-list">{done.map(log => {
      const urls = reportUrls(log.id)
      return <article className="ds-report" key={log.id}><span className="ds-report-icon"><FileText size={25}/></span><div><h3>{log.filename}</h3><p>{log.n_detections} detections · {log.n_frames} frames · completed {log.completed_at ? formatTime(log.completed_at) : ''}</p><small>Generated by the backend at processing time</small></div>
        <div className="ds-report-links"><a className="ds-button" href={urls.csv}>CSV</a><a className="ds-button" href={urls.json}>JSON</a><a className="ds-button" href={urls.geojson}>GeoJSON</a><a className="ds-button" href={urls.sql}>SQL</a><a className="ds-button" href={urls.pdf}>PDF</a></div>
      </article>
    })}</div>
    {!done.length && <div className="ds-no-results"><strong>No completed logs yet</strong><p>Upload a sonar log to generate downloadable reports.</p></div>}</>
}

function SystemView() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [checkedAt, setCheckedAt] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const refresh = () => {
    setLoading(true); setError('')
    getHealth().then(h => { setHealth(h); setCheckedAt(new Date().toLocaleTimeString()) }).catch(() => setError('Could not reach the backend.')).finally(() => setLoading(false))
  }
  useEffect(refresh, [])
  const rows = health ? [
    ['YOLO26 detector', health.models.yolo_best_pt?.found ?? false, health.models.yolo_best_pt?.path ?? ''],
    ['VAE anomaly model', health.models.vae?.found ?? false, health.models.vae?.path ?? ''],
    ['Blind2Unblind denoiser', health.models.b2u_blind2unblind?.found ?? false, health.models.b2u_blind2unblind?.note ?? ''],
    ['Lee filter (classical denoise)', health.models.lee_filter?.found ?? true, 'Always available, no checkpoint needed'],
    ['XTF reader', Boolean(health.xtf?.library_available), health.xtf?.library ? String(health.xtf.library) : ''],
  ] as const : []
  return <><div className="ds-page-intro"><div><h2>Backend status</h2><p>Live component health, read directly from the backend's /health endpoint.</p></div><button className="ds-button" onClick={refresh} disabled={loading}><RefreshCw size={15}/> {loading ? 'Checking…' : 'Refresh status'}</button></div>
    {error && <p className="ds-panel-note ds-error-text"><AlertTriangle size={15}/> {error}</p>}
    {health && <Panel title={`Device: ${health.device.toUpperCase()}`}><div className="ds-service-list">{rows.map(([title, ok, note]) => <div key={title}><span className="ds-health-dot" style={{ background: ok ? '#4eaf6b' : '#bf586a' }}/><div><strong>{title}</strong><small>{note}</small></div><span className={ok ? 'ds-green' : 'ds-error-text'}>{ok ? 'Available' : 'Unavailable'}</span></div>)}</div><p className="ds-panel-note" role="status">Last checked {checkedAt}</p></Panel>}
  </>
}

function SettingsView({ announce }: { announce: (message: string) => void }) {
  const [preferences, setPreferences] = useState(() => {
    try { return { operator: 'Survey Operator', units: 'Metric', alerts: true, ...JSON.parse(localStorage.getItem('sonarsense-dashboard-preferences') || '{}') } } catch { return { operator: 'Survey Operator', units: 'Metric', alerts: true } }
  })
  const save = (event: FormEvent) => { event.preventDefault(); localStorage.setItem('sonarsense-dashboard-preferences', JSON.stringify(preferences)); announce('Preferences saved on this device') }
  return <><div className="ds-page-intro"><div><h2>Workspace settings</h2><p>Preferences are stored locally on this device. No account is required.</p></div></div><Panel title="Operator preferences" className="ds-settings-panel"><form className="ds-settings-form" onSubmit={save}><label>Operator name<input required maxLength={60} value={preferences.operator} onChange={event => setPreferences({ ...preferences, operator: event.target.value })}/></label><label>Measurement units<select value={preferences.units} onChange={event => setPreferences({ ...preferences, units: event.target.value })}><option>Metric</option></select></label><label className="ds-toggle-row"><span>High-priority alert preference<small>Saved for your own reference -- no notifications are sent by this frontend yet.</small></span><input type="checkbox" checked={preferences.alerts} onChange={event => setPreferences({ ...preferences, alerts: event.target.checked })}/></label><button className="ds-button primary" type="submit">Save preferences</button></form></Panel></>
}

function DetailDialog({ detection, onClose, onOpenPipeline }: { detection: DetectionRecord | null; onClose: () => void; onOpenPipeline: (logId: string) => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  useEffect(() => { if (detection) dialogRef.current?.showModal(); else dialogRef.current?.close() }, [detection])
  return <dialog className="ds-dialog" ref={dialogRef} onCancel={onClose} onClick={event => { if (event.target === event.currentTarget) onClose() }} aria-labelledby="detail-title"><div><header><span>DETECTION RECORD</span><button className="ds-icon-button" onClick={onClose} aria-label="Close details"><X size={19}/></button></header>{detection && <><h2 id="detail-title">{detection.type} · {detection.id.slice(0, 8)}</h2>
    <figure className="ds-anomaly-photo"><div className="ds-anomaly-photo-label">VAE anomaly overlay for this detection's frame</div><img src={vaeFrameImageUrl(detection.logId, detection.frameRecordId, '03_anomaly_overlay.png')} alt="VAE anomaly overlay for this detection's frame" /></figure>
    <dl>{[
      ['Priority', detection.priority], ['Confidence', `${detection.confidence}%`], ['Depth', detection.depth ? `${detection.depth.toFixed(1)} m` : 'Unavailable'],
      ['Location', detection.located ? `${detection.latitude.toFixed(4)}° N, ${detection.longitude.toFixed(4)}° E` : 'No GPS fix'],
      ['Length', detection.lengthM != null ? `${detection.lengthM.toFixed(2)} m` : 'Unavailable'],
      ['Width', detection.widthM != null ? `${detection.widthM.toFixed(2)} m` : 'Unavailable'],
      ['Height', detection.heightM != null ? `${detection.heightM.toFixed(2)} m (estimated)` : 'Unavailable'],
      ['Timestamp', formatTime(detection.timestamp)], ['Log', detection.mission],
    ].map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl>
    {detection.dimensionsEstimated && <p className="ds-panel-note">Length/width could not be measured for this log (no pixels-to-meters calibration was provided at processing time); height is always an estimate -- a 2D side-scan frame has no height axis.</p>}
    <button className="ds-button primary" onClick={() => onOpenPipeline(detection.logId)}>Open in pipeline viewer</button>
  </>}</div></dialog>
}

export default function Dashboard() {
  const [path, setPath] = useState(window.location.pathname.replace(/\/$/, '') || '/')
  const [collapsed, setCollapsed] = useState(false), [mobileOpen, setMobileOpen] = useState(false)
  const [selected, setSelected] = useState<DetectionRecord | null>(null)
  const [toast, setToast] = useState(''), [notifications, setNotifications] = useState(false)
  const [selectedLogId, setSelectedLogId] = useState<string | null>(null)
  const active = navigation.find(item => item.path === path)
  const { logs, doneLogs, detections, topAnomalousFrames, meanWholeImageError, loading, error, refresh } = useSonarData()

  useEffect(() => {
    if (!selectedLogId && doneLogs.length > 0) {
      setSelectedLogId([...doneLogs].sort((a, b) => Date.parse(b.completed_at ?? b.uploaded_at) - Date.parse(a.completed_at ?? a.uploaded_at))[0].id)
    }
  }, [doneLogs, selectedLogId])

  const framesCount = doneLogs.reduce((sum, l) => sum + l.n_frames, 0)
  const timeline: TimelinePoint[] = useMemo(() => {
    const byDay = new Map<string, number>()
    for (const d of detections) {
      const day = new Date(d.timestamp).toISOString().slice(0, 10)
      byDay.set(day, (byDay.get(day) ?? 0) + 1)
    }
    return Array.from(byDay.entries()).sort(([a], [b]) => a.localeCompare(b)).slice(-14)
      .map(([day, count]) => ({ label: day.slice(5).replace('-', '/'), count }))
  }, [detections])

  const navigate: Navigate = (next, event) => {
    if (event && (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)) return
    event?.preventDefault(); window.history.pushState({}, '', next); setPath(next); setMobileOpen(false); setNotifications(false); window.scrollTo(0, 0)
  }
  useEffect(() => { const pop = () => { setPath(window.location.pathname.replace(/\/$/, '') || '/'); setMobileOpen(false) }; window.addEventListener('popstate', pop); return () => window.removeEventListener('popstate', pop) }, [])
  useEffect(() => { document.title = `${active?.label ?? 'Page not found'} · SonarSense`; setSelected(null) }, [path, active?.label])
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(''), 3500); return () => window.clearTimeout(timer) }, [toast])
  const closeDetail = () => setSelected(null)
  const openPipelineFor = (logId: string) => { setSelected(null); setSelectedLogId(logId); navigate('/analysis') }

  return <div className={`ds-app ${collapsed ? 'ds-collapsed' : ''}`}>
    <a href="#dashboard-main" className="ds-skip">Skip to dashboard</a>
    <aside className={`ds-sidebar ${mobileOpen ? 'is-open' : ''}`} aria-label="Primary navigation"><a className="ds-brand" href="/" onClick={event => navigate('/', event)}><span className="ds-logo"><Layers size={15}/></span><strong>SonarSense</strong></a><button className="ds-mobile-close ds-icon-button" aria-label="Close navigation" onClick={() => setMobileOpen(false)}><X size={20}/></button>
      <nav>{navigation.map(({ path: destination, label, icon: Icon }) => <a key={label} href={destination} onClick={event => navigate(destination, event)} className={`${path === destination ? 'active' : ''} ${label === 'Settings' ? 'ds-settings-nav' : ''}`} aria-current={path === destination ? 'page' : undefined} title={collapsed ? label : undefined}><Icon size={18} strokeWidth={1.6} aria-hidden="true"/><span>{label}</span></a>)}</nav>
    </aside>
    {mobileOpen && <button className="ds-sidebar-overlay" aria-label="Close menu" onClick={() => setMobileOpen(false)}/>}
    <div className="ds-workarea"><header className="ds-topbar"><button className="ds-menu ds-icon-button" aria-label="Toggle navigation" aria-expanded={mobileOpen || !collapsed} onClick={() => window.innerWidth < 760 ? setMobileOpen(!mobileOpen) : setCollapsed(!collapsed)}><Menu size={20}/></button><h1>{path === '/' ? 'Marine Debris Intelligence Dashboard' : active?.label ?? 'Page not found'}</h1><div className="ds-top-actions">
        <button className="ds-icon-button" title="Refresh data" aria-label="Refresh data" onClick={refresh}><RefreshCw size={17}/></button>
        <div className="ds-notification-wrap"><button className="ds-notification ds-icon-button" aria-label="Notifications" aria-expanded={notifications} onClick={() => setNotifications(!notifications)}><Bell size={19}/></button>{notifications && <div className="ds-notifications"><h2>Status</h2><p><strong>{doneLogs.length} log(s) processed</strong>{detections.length} total detections recorded.</p>{logs.some(l => l.status === 'processing') && <p><strong>Processing in progress</strong>Refreshing automatically.</p>}<button onClick={() => setNotifications(false)}>Close</button></div>}</div>
        <label className="ds-export"><span className="ds-sr-only">Export report</span><select value="" onChange={event => { if (event.target.value) { downloadCsv('sonarsense-detections.csv', detectionCsv(event.target.value === 'high' ? detections.filter(item => item.priority === 'High') : detections)); setToast('Report downloaded as CSV') } }}><option value="" disabled>Export Report</option><option value="all">All detections · CSV</option><option value="high">High priority · CSV</option></select></label>
        <span className="ds-date"><CalendarDays size={16} aria-hidden="true" /> Live data</span>
      </div></header>
      <main id="dashboard-main" className="ds-main" tabIndex={-1}>
        {loading && logs.length === 0 && <div className="ds-panel-note"><Loader2 size={16} className="ds-spin" /> Loading data from the backend…</div>}
        {error && <div className="ds-panel-note ds-error-text"><AlertTriangle size={15} /> {error} — the backend may be waking up from idle (free-tier services sleep after 15 minutes); try Refresh in a moment.</div>}
        {!loading && !error && logs.length === 0 && path !== '/upload' && path !== '/settings' && (
          <div className="ds-no-results"><Waves size={26}/><strong>No sonar logs processed yet</strong><p>Upload your first side-scan sonar log to populate this dashboard with real detections, VAE anomaly analysis, and geolocation results.</p><Link to="/upload" navigate={navigate}>Go to Upload</Link></div>
        )}
        {(!loading || logs.length > 0) && !error && <>
          {path === '/' && logs.length > 0 && <Overview items={detections} logsCount={doneLogs.length} framesCount={framesCount} timeline={timeline} anomalousFrames={topAnomalousFrames} meanError={meanWholeImageError} navigate={navigate} onSelect={setSelected}/>}
          {path === '/upload' && <UploadView onProcessed={(id) => { setSelectedLogId(id); refresh(); navigate('/analysis') }}/>}
          {path === '/detections' && logs.length > 0 && <DetectionsView items={detections} onSelect={setSelected}/>}
          {path === '/map' && logs.length > 0 && <MapView items={detections} onSelect={setSelected} navigate={navigate}/>}
          {path === '/analysis' && <PipelineView logs={logs} selectedLogId={selectedLogId} onSelectLog={setSelectedLogId}/>}
          {path === '/missions' && <><div className="ds-page-intro"><div><h2>Processed logs</h2><p>Every sonar log submitted to the backend, with its real processing outcome.</p></div></div><Panel title="All logs"><LogsTable logs={logs}/></Panel></>}
          {path === '/data-quality' && <QualityView logs={logs}/>}
          {path === '/reports' && <ReportsView logs={logs} items={detections}/>}
          {path === '/system-status' && <SystemView/>}
          {path === '/settings' && <SettingsView announce={setToast}/>}
          {!active && <div className="ds-no-results"><h2>Page not found</h2><Link to="/" navigate={navigate}>Return to overview</Link></div>}
        </>}
      </main>
    </div><DetailDialog detection={selected} onClose={closeDetail} onOpenPipeline={openPipelineFor}/>{toast && <div className="ds-toast" role="status"><Check size={17}/>{toast}</div>}
  </div>
}

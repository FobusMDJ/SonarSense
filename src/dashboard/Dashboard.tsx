import { useEffect, useMemo, useRef, useState, type FormEvent, type MouseEvent, type ReactNode } from 'react'
import { ArrowRight, Bell, CalendarDays, Check, CircleGauge, Download, FileText, Layers, Map, MapPin, Menu, PanelTop, Radar, Search, Settings, Shield, ShieldCheck, X, ChartNoAxesCombined, RefreshCw } from 'lucide-react'
import { detections, missions, categories, colorForType, detectionCsv, downloadCsv, formatTime, type DetectionRecord } from './data'
import { AnomalyPreview, ConfidenceChart, Sparkline, TimelineChart, TypeChart } from './Charts'
import { DetectionMap } from './DetectionMap'
import { AnomalyImage } from './AnomalyImage'
import { ModelOutput } from './ModelOutput'
import './dashboard.css'

const navigation = [
  { label: 'Overview', path: '/', icon: Shield },
  { label: 'Detections', path: '/detections', icon: PanelTop },
  { label: 'Map', path: '/map', icon: Map },
  { label: 'Model Output', path: '/analysis', icon: ChartNoAxesCombined },
  { label: 'Missions', path: '/missions', icon: ShieldCheck },
  { label: 'Data Quality', path: '/data-quality', icon: CircleGauge },
  { label: 'Reports', path: '/reports', icon: FileText },
  { label: 'System Status', path: '/system-status', icon: Radar },
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
  return <div className="ds-table-scroll"><table className="ds-table"><thead><tr>{['ID', 'Type', 'Confidence', 'Depth', 'Location', 'Time', 'Priority', ...(extended ? ['Mission'] : [])].map(title => <th key={title} scope="col">{title}</th>)}</tr></thead><tbody>{items.map(item => <tr key={item.id}>
    <td><button className="ds-id-button" onClick={() => onSelect(item)} aria-label={`Open detection ${item.id}`}>{item.id}</button></td>
    <td><span className="ds-type-label"><i style={{ background: colorForType(item.type) }}/>{item.type}</span></td><td>{item.confidence}%</td><td>{item.depth.toFixed(1)} m</td>
    <td>{item.latitude.toFixed(4)}° N, {item.longitude.toFixed(4)}° E</td><td>{formatTime(item.timestamp)}</td><td><span className={`ds-priority ${item.priority.toLowerCase()}`}>{item.priority}</span></td>{extended && <td>{item.mission}</td>}
  </tr>)}</tbody></table>{!items.length && <div className="ds-no-results"><Search size={25}/><strong>No detections found</strong><p>Try a different search or reset the filters.</p></div>}</div>
}
function MissionTable({ limit = 7, onSelect }: { limit?: number; onSelect: (id: string) => void }) {
  return <div className="ds-table-scroll"><table className="ds-table ds-mission-table"><thead><tr>{['Mission ID', 'Date', 'Area Covered', 'Ping Count', 'Data Quality', 'Status'].map(title => <th key={title} scope="col">{title}</th>)}</tr></thead><tbody>{missions.slice(0, limit).map(mission => <tr key={mission.id}><td><button className="ds-id-button" onClick={() => onSelect(mission.id)}>{mission.id}</button></td><td>{mission.date}</td><td>{mission.area.toFixed(1)} km²</td><td>{mission.pings.toLocaleString()}</td><td>{mission.quality}%</td><td className="ds-green">{mission.status}</td></tr>)}</tbody></table></div>
}
function MetricCards({ items }: { items: DetectionRecord[] }) {
  const values = [items.length, items.filter(item => item.priority === 'High').length, `${Math.round(items.reduce((sum, item) => sum + item.confidence, 0) / (items.length || 1))}%`, '125', '7', '92%']
  const titles = ['Total Detections', 'High Priority', 'Avg. Confidence', 'Area Covered', 'Missions', 'Data Quality']
  const colors = ['#668cf0', '#ed9b48', '#6dad68', '#9b68dc', '#5784df', '#72b577']
  return <section className="ds-metrics" aria-label="Survey metrics">{titles.map((title, index) => <div className="ds-metric" key={title}>
    <span>{title}</span><strong>{values[index]}{index === 3 && <small> km²</small>}</strong>
    <p>{index < 4 ? <><b style={{ color: index === 0 ? '#d5dfe8' : colors[index] }}>↑ {['12', '5', '7', '8'][index]}%</b> vs last 7 days</> : index === 4 ? 'Completed' : 'Good'}</p>
    {index < 4 ? <Sparkline color={colors[index]} variant={index}/> : <div className="ds-progress"><i style={{ width: index === 4 ? '78%' : '92%', background: colors[index] }}/></div>}
  </div>)}</section>
}

function Overview({ items, navigate, onSelect, onMission }: { items: DetectionRecord[]; navigate: Navigate; onSelect: (item: DetectionRecord) => void; onMission: (id: string) => void }) {
  return <><MetricCards items={items}/><div className="ds-overview-grid">
    <div className="ds-chart-pair"><Panel title="Detections by Type"><TypeChart items={items}/></Panel><Panel title="Detections Over Time"><TimelineChart/></Panel></div>
    <Panel title="Detection Map" className="ds-overview-map"><DetectionMap items={items} onSelect={onSelect}/></Panel>
    <Panel title="Recent Detections" className="ds-recent"><DetectionTable items={items.slice(0, 5)} onSelect={onSelect}/><footer><Link to="/detections" navigate={navigate}>View all Detections</Link></footer></Panel>
    <div className="ds-bottom-pair"><Panel title="Anomaly Overview (VAE)"><AnomalyPreview/></Panel><Panel title="Detection Confidence Distribution"><ConfidenceChart items={items}/></Panel></div>
    <Panel title="Mission Summary" className="ds-mission-summary"><MissionTable limit={3} onSelect={onMission}/><footer><Link to="/missions" navigate={navigate}>View all Missions</Link></footer></Panel>
  </div></>
}

function DetectionsView({ items, onSelect }: { items: DetectionRecord[]; onSelect: (item: DetectionRecord) => void }) {
  const [query, setQuery] = useState(''), [type, setType] = useState('All types'), [priority, setPriority] = useState('All priorities'), [page, setPage] = useState(0)
  useEffect(() => setPage(0), [items])
  const filtered = items.filter(item => (type === 'All types' || item.type === type) && (priority === 'All priorities' || item.priority === priority) && `${item.id} ${item.type} ${item.mission}`.toLowerCase().includes(query.toLowerCase()))
  const reset = () => { setQuery(''); setType('All types'); setPriority('All priorities'); setPage(0) }
  return <><div className="ds-page-intro"><div><h2>Detection register</h2><p>Inspect and filter the {items.length} observations in this demonstration survey.</p></div><button className="ds-button" onClick={() => downloadCsv('sonarsense-filtered-detections.csv', detectionCsv(filtered))}><Download size={16}/> Export CSV</button></div>
    <Panel title={`${filtered.length} detections`}><div className="ds-toolbar"><label className="ds-search"><Search size={16}/><input value={query} onChange={event => { setQuery(event.target.value); setPage(0) }} placeholder="Search ID, type, or mission" aria-label="Search detections"/></label>
      <select aria-label="Filter by type" value={type} onChange={event => { setType(event.target.value); setPage(0) }}><option>All types</option>{categories.map(category => <option key={category.name}>{category.name}</option>)}</select>
      <select aria-label="Filter by priority" value={priority} onChange={event => { setPriority(event.target.value); setPage(0) }}><option>All priorities</option>{['High', 'Medium', 'Low'].map(level => <option key={level}>{level}</option>)}</select><button className="ds-button" onClick={reset}>Reset</button></div>
      <DetectionTable items={filtered.slice(page * 15, (page + 1) * 15)} onSelect={onSelect} extended/>
      <div className="ds-pagination"><span>{filtered.length ? page * 15 + 1 : 0}–{Math.min((page + 1) * 15, filtered.length)} of {filtered.length}</span><button disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button><button disabled={(page + 1) * 15 >= filtered.length} onClick={() => setPage(page + 1)}>Next</button></div>
    </Panel></>
}
function MapView({ items, onSelect, navigate }: { items: DetectionRecord[]; onSelect: (item: DetectionRecord) => void; navigate: Navigate }) {
  const [type, setType] = useState('All types')
  const filtered = items.filter(item => type === 'All types' || item.type === type)
  const [located, setLocated] = useState<DetectionRecord | null>(items[0] ?? null)
  const inspect = (item: DetectionRecord) => { setLocated(item); onSelect(item) }
  return <><div className="ds-page-intro"><div><h2>Survey detection map</h2><p>Mock observations along an illustrative offshore survey track. Select a marker to inspect its details and anomaly reference image.</p></div></div>
    <div className="ds-map-workspace"><Panel title="GPS-tagged detections" action={<select value={type} onChange={event => setType(event.target.value)} aria-label="Map detection type"><option>All types</option>{categories.map(category => <option key={category.name}>{category.name}</option>)}</select>}><DetectionMap full items={filtered} onSelect={inspect}/></Panel>
      <Panel title="Geolocation result" className="ds-geolocation"><div className="ds-geo-state"><MapPin size={20}/><div><span>Position quality</span><strong>Estimated · GPS synchronized</strong></div></div>{located && <><dl><div><dt>Detection</dt><dd>#{located.id} · {located.type}</dd></div><div><dt>Latitude</dt><dd>{located.latitude.toFixed(4)}° N</dd></div><div><dt>Longitude</dt><dd>{located.longitude.toFixed(4)}° E</dd></div><div><dt>Depth</dt><dd>{located.depth.toFixed(1)} m</dd></div><div><dt>Horizontal accuracy</dt><dd>± 4.8 m</dd></div><div><dt>Source ping</dt><dd>288,412</dd></div><div><dt>Survey</dt><dd>{located.mission}</dd></div><div><dt>Position basis</dt><dd>Vehicle GPS + ping timestamp</dd></div></dl><p>The sensor-to-world transform is unavailable in this mock log, so SonarSense preserves source metadata and labels this field position as an estimate.</p></>}<div className="ds-log-meta"><span>INPUT SONAR LOG</span><strong>IB-2505-17.xtf</strong><small>Navigation packets: 99.2% complete</small></div><footer><Link to="/detections" navigate={navigate}>Open detection records</Link></footer></Panel>
    </div></>
}
function AnalysisView({ items }: { items: DetectionRecord[] }) {
  return <><div className="ds-page-intro"><div><h2>Model output</h2><p>Review the mock YOLO classifications, selected VAE evidence, and evaluation statistics from one processed sonar log.</p></div><span className="ds-status-pill"><Check size={15}/> Frontend mock data</span></div><ModelOutput items={items}/></>
}
function QualityView() {
  return <><div className="ds-page-intro"><div><h2>Data quality</h2><p>Quality-control results for seven completed mock missions.</p></div><span className="ds-status-pill"><Check size={15}/> Good · 92% overall</span></div><div className="ds-quality-stats">{[['Signal-to-noise ratio', '28.4 dB', 'Above 20 dB threshold'], ['GPS completeness', '99.2%', 'Position available for nearly all pings'], ['Valid sonar pings', '97.6%', '654,107 accepted returns'], ['Coverage continuity', '96.8%', 'All primary transects complete']].map(([title, value, caption]) => <Panel key={title} title={title}><div className="ds-big-value">{value}</div><p className="ds-panel-note">{caption}</p></Panel>)}</div><Panel title="Mission quality checks"><div className="ds-quality-list">{missions.map(mission => <div key={mission.id}><strong>{mission.id}</strong><span>{mission.name}</span><div className="ds-progress"><i style={{ width: `${mission.quality}%` }}/></div><b>{mission.quality}%</b><span className="ds-green">Passed</span></div>)}</div></Panel></>
}
function ReportsView({ items, announce }: { items: DetectionRecord[]; announce: (text: string) => void }) {
  return <><div className="ds-page-intro"><div><h2>Survey reports</h2><p>Download CSV reports generated locally from the demonstration data.</p></div></div><div className="ds-report-list">{[
    { title: 'Detection inventory', description: `${items.length} detections · classification, confidence, depth, and coordinates`, filename: 'sonarsense-detection-inventory.csv', rows: detectionCsv(items) },
    { title: 'High-priority observations', description: 'All high-priority detections for follow-up planning', filename: 'sonarsense-high-priority.csv', rows: detectionCsv(items.filter(item => item.priority === 'High')) },
    { title: 'Mission performance summary', description: 'Seven completed missions · area, ping count, and data quality', filename: 'sonarsense-mission-summary.csv', rows: [['Mission', 'Date', 'Area (km²)', 'Pings', 'Quality (%)', 'Status'], ...missions.map(m => [m.id, m.date, m.area, m.pings, m.quality, m.status])] },
  ].map(report => <article className="ds-report" key={report.title}><span className="ds-report-icon"><FileText size={25}/></span><div><h3>{report.title}</h3><p>{report.description}</p><small>CSV · Demo dataset · May 15–21, 2025</small></div><button className="ds-button" onClick={() => { downloadCsv(report.filename, report.rows); announce(`${report.title} downloaded`) }}><Download size={16}/> Download</button></article>)}</div></>
}
function SystemView() {
  const [checked, setChecked] = useState('Not refreshed this session')
  return <><div className="ds-page-intro"><div><h2>System status</h2><p>Simulated component health. This frontend is not connected to a live vehicle.</p></div><button className="ds-button" onClick={() => setChecked(`Checked at ${new Date().toLocaleTimeString()}`)}><RefreshCw size={15}/> Refresh mock status</button></div><Panel title="All demonstration systems operational"><div className="ds-service-list">{[['Sonar processing', '24.8 Hz', 'Sample processing throughput'], ['Classification service', '128 ms', 'Sample inference latency'], ['Geospatial service', '42 ms', 'Sample query response'], ['Mission telemetry', '99.8%', 'Sample packet completeness'], ['Storage service', '23.4 / 100 GB', 'Sample storage allocation']].map(([title, metric, description]) => <div key={title}><span className="ds-health-dot"/><div><strong>{title}</strong><small>{description}</small></div><b>{metric}</b><span className="ds-green">Operational</span></div>)}</div><p className="ds-panel-note" role="status">{checked}</p></Panel></>
}
function SettingsView({ announce }: { announce: (message: string) => void }) {
  const [preferences, setPreferences] = useState(() => {
    try { return { operator: 'Survey Operator', units: 'Metric', alerts: true, ...JSON.parse(localStorage.getItem('sonarsense-dashboard-preferences') || '{}') } } catch { return { operator: 'Survey Operator', units: 'Metric', alerts: true } }
  })
  const save = (event: FormEvent) => { event.preventDefault(); localStorage.setItem('sonarsense-dashboard-preferences', JSON.stringify(preferences)); announce('Preferences saved on this device') }
  return <><div className="ds-page-intro"><div><h2>Workspace settings</h2><p>Preferences are stored locally on this device. No account or backend is required.</p></div></div><Panel title="Operator preferences" className="ds-settings-panel"><form className="ds-settings-form" onSubmit={save}><label>Operator name<input required maxLength={60} value={preferences.operator} onChange={event => setPreferences({ ...preferences, operator: event.target.value })}/></label><label>Measurement units<select value={preferences.units} onChange={event => setPreferences({ ...preferences, units: event.target.value })}><option>Metric</option></select><small>The demonstration dataset uses metres and square kilometres.</small></label><label className="ds-toggle-row"><span>High-priority alert preference<small>Saved for future connected surveys. Alerts shown here are mock examples.</small></span><input type="checkbox" checked={preferences.alerts} onChange={event => setPreferences({ ...preferences, alerts: event.target.checked })}/></label><button className="ds-button primary" type="submit">Save preferences</button></form></Panel></>
}

function DetailDialog({ detection, missionId, onClose }: { detection: DetectionRecord | null; missionId: string | null; onClose: () => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  const mission = missions.find(item => item.id === missionId)
  useEffect(() => { if (detection || missionId) dialogRef.current?.showModal(); else dialogRef.current?.close() }, [detection, missionId])
  return <dialog className="ds-dialog" ref={dialogRef} onCancel={onClose} onClick={event => { if (event.target === event.currentTarget) onClose() }} aria-labelledby="detail-title"><div><header><span>MOCK OBSERVATION</span><button className="ds-icon-button" onClick={onClose} aria-label="Close details"><X size={19}/></button></header><h2 id="detail-title">{detection ? `${detection.type} · ${detection.id}` : mission?.name}</h2>{detection && <AnomalyImage key={detection.id} type={detection.type}/>}<dl>{(detection ? [
    ['Priority', detection.priority], ['Confidence', `${detection.confidence}%`], ['Depth', `${detection.depth.toFixed(1)} m`], ['Location', `${detection.latitude.toFixed(4)}° N, ${detection.longitude.toFixed(4)}° E`], ['Timestamp', formatTime(detection.timestamp)], ['Mission', detection.mission],
  ] : mission ? [['Mission ID', mission.id], ['Date', mission.date], ['Area covered', `${mission.area} km²`], ['Ping count', mission.pings.toLocaleString()], ['Data quality', `${mission.quality}%`], ['Status', mission.status]] : []).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl><p>Demonstration data for frontend development. Not a verified field observation.</p></div></dialog>
}

export default function Dashboard() {
  const [path, setPath] = useState(window.location.pathname.replace(/\/$/, '') || '/')
  const [collapsed, setCollapsed] = useState(false), [mobileOpen, setMobileOpen] = useState(false)
  const [selected, setSelected] = useState<DetectionRecord | null>(null), [selectedMission, setSelectedMission] = useState<string | null>(null)
  const [period, setPeriod] = useState('7'), [toast, setToast] = useState(''), [notifications, setNotifications] = useState(false), [unread, setUnread] = useState(true)
  const active = navigation.find(item => item.path === path)
  const items = useMemo(() => detections.filter(item => period === '7' || Number(item.timestamp.slice(8, 10)) >= 19), [period])
  const navigate: Navigate = (next, event) => {
    if (event && (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)) return
    event?.preventDefault(); window.history.pushState({}, '', next); setPath(next); setMobileOpen(false); setNotifications(false); window.scrollTo(0, 0)
  }
  useEffect(() => { const pop = () => { setPath(window.location.pathname.replace(/\/$/, '') || '/'); setMobileOpen(false) }; window.addEventListener('popstate', pop); return () => window.removeEventListener('popstate', pop) }, [])
  useEffect(() => { document.title = `${active?.label ?? 'Page not found'} · SonarSense`; setSelected(null); setSelectedMission(null) }, [path, active?.label])
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(''), 3500); return () => window.clearTimeout(timer) }, [toast])
  const closeDetail = () => { setSelected(null); setSelectedMission(null) }
  return <div className={`ds-app ${collapsed ? 'ds-collapsed' : ''}`}>
    <a href="#dashboard-main" className="ds-skip">Skip to dashboard</a>
    <aside className={`ds-sidebar ${mobileOpen ? 'is-open' : ''}`} aria-label="Primary navigation"><a className="ds-brand" href="/" onClick={event => navigate('/', event)}><span className="ds-logo"><Layers size={15}/></span><strong>SonarSense</strong></a><button className="ds-mobile-close ds-icon-button" aria-label="Close navigation" onClick={() => setMobileOpen(false)}><X size={20}/></button>
      <nav>{navigation.map(({ path: destination, label, icon: Icon }) => <a key={label} href={destination} onClick={event => navigate(destination, event)} className={`${path === destination ? 'active' : ''} ${label === 'Settings' ? 'ds-settings-nav' : ''}`} aria-current={path === destination ? 'page' : undefined} title={collapsed ? label : undefined}><Icon size={18} strokeWidth={1.6} aria-hidden="true"/><span>{label}</span></a>)}</nav>
    </aside>
    {mobileOpen && <button className="ds-sidebar-overlay" aria-label="Close menu" onClick={() => setMobileOpen(false)}/>}
    <div className="ds-workarea"><header className="ds-topbar"><button className="ds-menu ds-icon-button" aria-label="Toggle navigation" aria-expanded={mobileOpen || !collapsed} onClick={() => window.innerWidth < 760 ? setMobileOpen(!mobileOpen) : setCollapsed(!collapsed)}><Menu size={20}/></button><h1>{path === '/' ? 'Marine Debris Intelligence Dashboard' : active?.label ?? 'Page not found'}</h1><div className="ds-top-actions"><div className="ds-notification-wrap"><button className="ds-notification ds-icon-button" aria-label={unread ? 'Notifications, 3 unread' : 'Notifications'} aria-expanded={notifications} onClick={() => setNotifications(!notifications)}><Bell size={19}/>{unread && <b>3</b>}</button>{notifications && <div className="ds-notifications"><h2>Survey notifications</h2><p><strong>High-priority debris detected</strong>38 observations are ready for inspection.</p><p><strong>Mission completed</strong>M-2025-051 finished its planned transects.</p><p><strong>Quality checks passed</strong>Overall dataset quality is 92%.</p><button onClick={() => { setUnread(false); setNotifications(false); setToast('All notifications marked as read') }}>Mark all as read</button></div>}</div>
        <label className="ds-export"><span className="ds-sr-only">Export report</span><select value="" onChange={event => { if (event.target.value) { downloadCsv('sonarsense-detections.csv', detectionCsv(event.target.value === 'high' ? items.filter(item => item.priority === 'High') : items)); setToast('Report downloaded as CSV') } }}><option value="" disabled>Export Report</option><option value="all">All detections · CSV</option><option value="high">High priority · CSV</option></select></label>
        <label className="ds-date"><span className="ds-sr-only">Date range</span><select value={period} onChange={event => setPeriod(event.target.value)}><option value="7">May 15 – May 21, 2025</option><option value="3">May 19 – May 21, 2025</option></select><CalendarDays size={16} aria-hidden="true"/></label>
      </div></header>
      <main id="dashboard-main" className="ds-main" tabIndex={-1}>
        {path === '/' && <Overview items={items} navigate={navigate} onSelect={setSelected} onMission={setSelectedMission}/>}
        {path === '/detections' && <DetectionsView items={items} onSelect={setSelected}/>}
        {path === '/map' && <MapView items={items} onSelect={setSelected} navigate={navigate}/>}
        {path === '/analysis' && <AnalysisView items={items}/>}
        {path === '/missions' && <><div className="ds-page-intro"><div><h2>Mission history</h2><p>Seven completed missions covering 125 km². Select a mission to inspect its summary.</p></div></div><div className="ds-quality-stats">{[['Completed missions', '7'], ['Area covered', '125 km²'], ['Recorded pings', missions.reduce((sum, item) => sum + item.pings, 0).toLocaleString()], ['Average quality', '92%']].map(([title, value]) => <Panel title={title} key={title}><div className="ds-big-value">{value}</div></Panel>)}</div><Panel title="All missions"><MissionTable onSelect={setSelectedMission}/></Panel></>}
        {path === '/data-quality' && <QualityView/>}{path === '/reports' && <ReportsView items={items} announce={setToast}/>}{path === '/system-status' && <SystemView/>}{path === '/settings' && <SettingsView announce={setToast}/>}
        {!active && <div className="ds-no-results"><h2>Page not found</h2><Link to="/" navigate={navigate}>Return to overview</Link></div>}
        <div className="ds-demo-note"><span/> Demo dataset · Frontend preview</div>
      </main>
    </div><DetailDialog detection={selected} missionId={selectedMission} onClose={closeDetail}/>{toast && <div className="ds-toast" role="status"><Check size={17}/>{toast}</div>}
  </div>
}

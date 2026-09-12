import { CalendarClock, Crosshair, Gauge, Maximize, Radio, Ruler, Waves, X } from 'lucide-react'
import type { Detection } from '../types'

interface DetectionDetailsProps {
  detection: Detection | null
  onClose: () => void
}

const formatCoordinate = (value: number, positive: string, negative: string) =>
  `${Math.abs(value).toFixed(5)}° ${value >= 0 ? positive : negative}`

export function DetectionDetails({ detection, onClose }: DetectionDetailsProps) {
  if (!detection) return null
  const date = new Intl.DateTimeFormat('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit',
  }).format(new Date(detection.timestamp))

  return (
    <aside className="ss-details" aria-label="Selected detection details">
      <div className="ss-details__topline" />
      <header>
        <div>
          <span className="ss-kicker">Detection intelligence</span>
          <h2>{detection.classification}</h2>
        </div>
        <button type="button" className="ss-icon-button" onClick={onClose} aria-label="Close details panel" title="Close details">
          <X size={18} />
        </button>
      </header>
      <div className="ss-details__badges">
        <span className={`ss-priority-pill is-${detection.priority.toLowerCase()}`}>{detection.priority} priority</span>
        <span>{detection.id}</span>
      </div>
      <dl className="ss-detail-list">
        <div><dt><Gauge size={16} />Confidence</dt><dd>{detection.confidence}%</dd></div>
        <div><dt><Ruler size={16} />Length</dt><dd>{detection.dimensions.length.toFixed(1)} m</dd></div>
        <div><dt><Maximize size={16} />Width × height</dt><dd>{detection.dimensions.width.toFixed(1)} × {detection.dimensions.height.toFixed(1)} m</dd></div>
        <div><dt><Radio size={16} />Ping ID</dt><dd>{detection.pingId}</dd></div>
        <div><dt><CalendarClock size={16} />Timestamp</dt><dd>{date}</dd></div>
        <div><dt><Waves size={16} />Depth</dt><dd>{detection.depth.toFixed(1)} m</dd></div>
        <div><dt><Crosshair size={16} />Coordinates</dt><dd>{formatCoordinate(detection.coordinates[1], 'N', 'S')}<br />{formatCoordinate(detection.coordinates[0], 'E', 'W')}</dd></div>
      </dl>
      <p className="ss-details__note">Detected and geolocated by the SonarSense backend pipeline (YOLO26 + VAE anomaly analysis + confidence scoring + navigation-fix geolocation).</p>
    </aside>
  )
}

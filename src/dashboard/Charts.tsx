import { vaeFrameImageUrl } from '../lib/api'
// (useEffect/useRef no longer needed here -- AnomalyPreview now renders real backend images.)
import { categories, type DetectionRecord } from './data'
import type { AnomalousFrame } from './useSonarData'

export function Sparkline({ color, variant = 0 }: { color: string; variant?: number }) {
  const paths = [
    'M1 26 Q10 30 17 25 T30 27 T44 26 T56 18 T70 12 T85 8 T101 9 T117 13 T132 21 T148 26 T163 24 T180 26',
    'M1 26 Q15 26 21 21 T35 19 T48 25 T63 24 T79 20 T94 17 T111 19 T127 15 T143 18 T160 21 T180 20',
    'M1 27 Q12 32 23 27 T40 27 T58 25 T76 16 T95 14 T112 17 T130 21 T147 15 T164 16 T180 19',
    'M1 28 Q12 27 22 23 T39 25 T56 27 T74 28 T88 17 T104 16 T120 18 T137 22 T153 25 T168 25 T180 26',
  ]
  return <svg className="ds-spark" viewBox="0 0 181 38" aria-hidden="true"><path d={paths[variant % paths.length]} fill="none" stroke={color} strokeWidth="2" /></svg>
}

export function TypeChart({ items }: { items: DetectionRecord[] }) {
  let offset = 0
  const total = items.length
  return <div className="ds-type-chart"><div className="ds-donut"><svg viewBox="0 0 180 180" role="img" aria-label={`Detection types, ${total} total`}>
    {[categories[1], categories[0], categories[3], categories[2], categories[4]].map(category => {
      const amount = items.filter(item => item.type === category.name).length
      const fraction = total ? amount / total : 0
      const start = offset; offset += fraction * 100
      return <circle key={category.name} cx="90" cy="90" r="66" fill="none" stroke={category.color} strokeWidth="31" pathLength="100" strokeDasharray={`${fraction * 100} ${100 - fraction * 100}`} strokeDashoffset={-start} transform="rotate(-90 90 90)"><title>{category.name}: {amount}</title></circle>
    })}
  </svg><span><strong>{total}</strong><small>Total</small></span></div><ul>{categories.map(category => {
    const count = items.filter(item => item.type === category.name).length
    return <li key={category.name}><i style={{ background: category.color }} /><span>{category.name}</span><b>{count} ({total ? Math.round(count / total * 100) : 0}%)</b></li>
  })}</ul></div>
}

export interface TimelinePoint { label: string; count: number }

export function TimelineChart({ series }: { series: TimelinePoint[] }) {
  if (series.length === 0) {
    return <div className="ds-no-results"><strong>No detections yet</strong><p>Upload a sonar log to see detections plotted over time.</p></div>
  }
  const values = series.map(s => s.count)
  const maximum = Math.max(1, ...values)
  const stepX = values.length > 1 ? 382 / (values.length - 1) : 0
  const yFor = (n: number) => 169 - (n / maximum) * 151
  const gridValues = [0, 0.25, 0.5, 0.75, 1].map(f => Math.round(maximum * f))
  return <svg className="ds-timeline" viewBox="0 0 430 204" role="img" aria-label={`Detections over time: ${series.map(s => `${s.label}, ${s.count}`).join('; ')}.`}>
    {gridValues.map(n => <g key={n}><line x1="33" x2="418" y1={yFor(n)} y2={yFor(n)} stroke="#17232e" strokeWidth=".7"/><text x="4" y={yFor(n) + 4}>{n}</text></g>)}
    <path d="M33 18V169H418" fill="none" stroke="#617181" strokeWidth=".7" />
    <polyline points={values.map((n, i) => `${36 + i * stepX},${yFor(n)}`).join(' ')} fill="none" stroke="#5c85ee" strokeWidth="1.8" />
    {values.map((n, i) => <g key={i}><circle cx={36 + i * stepX} cy={yFor(n)} r="5.5" fill="#345ccd" opacity=".38"/><circle cx={36 + i * stepX} cy={yFor(n)} r="2.8" fill="#e3efff" stroke="#7d9eff"><title>{series[i].label}: {n} detections</title></circle><text x={36 + i * stepX} y="191" textAnchor="middle">{series[i].label}</text></g>)}
  </svg>
}

export function ConfidenceChart({ items }: { items: DetectionRecord[] }) {
  const bins = Array.from({ length: 10 }, (_, index) => items.filter(item => item.confidence >= index * 10 && (index === 9 ? item.confidence <= 100 : item.confidence < (index + 1) * 10)).length)
  const maximum = Math.max(60, ...bins)
  return <svg className="ds-confidence" viewBox="0 0 370 186" role="img" aria-label="Detection confidence distribution in ten percentage bands">
    <defs><linearGradient id="confidence-bars" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#7dc9a4"/><stop offset="1" stopColor="#5f9f77"/></linearGradient></defs>
    {[0, 20, 40, 60].map(n => <g key={n}><line x1="41" x2="354" y1={143 - n / maximum * 122} y2={143 - n / maximum * 122} stroke="#1c2a33" strokeWidth=".7"/><text x="24" y={147 - n / maximum * 122} textAnchor="end">{n}</text></g>)}
    <text transform="translate(10 93) rotate(-90)" textAnchor="middle">Detections</text>
    {bins.map((value, i) => <rect key={i} x={43 + i * 31} y={143 - value / maximum * 122} width="20" height={Math.max(1, value / maximum * 122)} rx="1" fill="url(#confidence-bars)"><title>{i * 10}–{(i + 1) * 10}%: {value} detections</title></rect>)}
    <path d="M40 18V144H355" fill="none" stroke="#53606d" strokeWidth=".6"/>
    {[0, 20, 40, 60, 80, 100].map((n, i) => <text key={n} x={43 + i * 62} y="162" textAnchor="middle">{n}</text>)}<text x="199" y="182" textAnchor="middle">Confidence (%)</text>
  </svg>
}

export function AnomalyPreview({ frames, meanError }: { frames: AnomalousFrame[]; meanError: number | null }) {
  const top = frames[0]
  if (!top) {
    return <div className="ds-no-results"><strong>No VAE analysis yet</strong><p>Upload a sonar log to see real anomaly-detection output here.</p></div>
  }
  return (
    <div className="ds-anomaly ds-anomaly-real">
      <img src={vaeFrameImageUrl(top.logId, top.frameRecordId, '01_original.png')} alt="Preprocessed input frame" />
      <img src={vaeFrameImageUrl(top.logId, top.frameRecordId, '03_anomaly_overlay.png')} alt="VAE anomaly overlay" />
      <div className="ds-heat-legend"><span>Low</span><i/><span>High</span></div>
      <dl className="ds-anomaly-stats">
        <div><dt>Most anomalous frame error</dt><dd>{top.wholeImageError?.toFixed(6) ?? 'n/a'}</dd></div>
        <div><dt>Mean error (all logs)</dt><dd>{meanError != null ? meanError.toFixed(6) : 'n/a'}</dd></div>
      </dl>
    </div>
  )
}

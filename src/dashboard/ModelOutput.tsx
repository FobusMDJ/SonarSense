import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { Box, CheckCircle2, ScanSearch } from 'lucide-react'
import { type DetectionRecord } from './data'

const vaeViews = [
  ['Original SSS', 'Raw side-scan sonar intensity'],
  ['Anomaly Heatmap', 'Reconstruction error intensity'],
  ['Anomaly Overlay', 'Heatmap aligned to the source image'],
  ['Auto Bounding Box', 'Thresholded anomaly candidate'],
  ['Edge / Contour Map', 'Structural boundaries around the return'],
  ['3D Anomaly Surface', 'Spatial reconstruction-error peaks'],
  ['Difference Map', 'Absolute original–reconstruction difference'],
] as const
type VaeView = typeof vaeViews[number][0]
const AnomalySurface3D = lazy(() => import('./AnomalySurface3D'))

const modelMetrics = [
  ['mAP@50', '0.91', 'Must'], ['mAP@50–95', '0.68', 'Must'], ['Precision', '0.89', 'Must'], ['Recall', '0.84', 'Must'],
  ['F1 Score', '0.86', 'Recommended'], ['Inference Time', '42 ms', 'Recommended'], ['FPS', '23.8', 'Recommended'], ['Number of Classes', '5', ''],
  ['Training Images', '4,860', 'Optional'], ['Model Version', 'YOLO11m · v1.4.2', ''], ['Confidence Threshold', '0.35', ''], ['Model Size', '43.7 MB', 'Optional'],
] as const

function pseudoNoise(x: number, y: number) {
  const value = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453
  return value - Math.floor(value)
}

function anomalyValue(x: number, y: number) {
  const blobs = [[.58, .54, .023, 1], [.48, .61, .012, .58], [.68, .43, .009, .45]]
  return Math.min(1, blobs.reduce((sum, [cx, cy, spread, scale]) => sum + scale * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / spread), 0))
}

function heatColor(value: number): [number, number, number] {
  if (value < .25) return [2, Math.round(value * 260), 90 + Math.round(value * 500)]
  if (value < .55) return [0, Math.round(90 + value * 290), Math.round(255 - value * 170)]
  if (value < .78) return [Math.round((value - .5) * 900), 225, 25]
  return [255, Math.max(15, Math.round(225 - (value - .78) * 820)), 8]
}

function VaeCanvas({ view }: { view: Exclude<VaeView, '3D Anomaly Surface'> }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current, context = canvas?.getContext('2d')
    if (!canvas || !context) return
    const { width, height } = canvas
    const pixels = context.createImageData(width, height)
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const nx = x / width, ny = y / height, p = (y * width + x) * 4
      const nadir = Math.exp(-((nx - .5) ** 2) / .002)
      const texture = 30 + pseudoNoise(x, y) ** 3 * 105 + 48 * Math.abs(Math.sin(y * .11 + x * .018))
      const object = anomalyValue(nx, ny)
      const sonar = Math.max(5, Math.min(235, texture + object * 115 - nadir * 100))
      const anomaly = Math.min(1, object + pseudoNoise(x * .7, y * .7) * .05)
      let rgb: [number, number, number] = [sonar, sonar, sonar]
      if (view === 'Anomaly Heatmap') rgb = heatColor(anomaly)
      if (view === 'Anomaly Overlay') {
        const heat = heatColor(anomaly); const alpha = anomaly * .82
        rgb = [sonar * (1 - alpha) + heat[0] * alpha, sonar * (1 - alpha) + heat[1] * alpha, sonar * (1 - alpha) + heat[2] * alpha]
      }
      if (view === 'Edge / Contour Map') {
        const next = anomalyValue(Math.min(1, nx + .007), ny)
        const down = anomalyValue(nx, Math.min(1, ny + .01))
        const edge = Math.min(255, Math.abs(anomaly - next) * 2700 + Math.abs(anomaly - down) * 2000)
        rgb = [edge, edge, edge]
      }
      if (view === 'Difference Map') {
        const difference = Math.min(255, anomaly * 245 + pseudoNoise(x, y) * 18)
        rgb = [difference, difference, difference]
      }
      pixels.data[p] = rgb[0]; pixels.data[p + 1] = rgb[1]; pixels.data[p + 2] = rgb[2]; pixels.data[p + 3] = 255
    }
    context.putImageData(pixels, 0, 0)
    if (view === 'Auto Bounding Box') {
      context.strokeStyle = '#49e48b'; context.lineWidth = 4; context.strokeRect(width * .43, height * .28, width * .3, height * .56)
      context.fillStyle = '#49e48b'; context.fillRect(width * .43, height * .21, 155, 24)
      context.fillStyle = '#04120b'; context.font = '600 14px Arial'; context.fillText('ANOMALY  0.92', width * .43 + 7, height * .21 + 17)
    }
  }, [view])
  return <canvas ref={ref} width="720" height="360" role="img" aria-label={`${view}, simulated VAE output from a side-scan sonar tile`}/>
}

function VaeExplorer() {
  const [view, setView] = useState<VaeView>('Anomaly Overlay')
  return <section className="ds-panel ds-vae-panel"><header className="ds-panel-title"><h2>VAE anomaly evidence</h2><span className="ds-mock-label">Mock processing output</span></header>
    <div className="ds-vae-layout"><div className="ds-vae-tabs" role="tablist" aria-label="VAE output views">{vaeViews.map(([name]) => <button key={name} role="tab" aria-selected={view === name} onClick={() => setView(name)}>{name}</button>)}</div>
      <div className="ds-vae-stage">{view === '3D Anomaly Surface' ? <Suspense fallback={<div className="ds-surface-loading">Preparing 3D surface…</div>}><AnomalySurface3D/></Suspense> : <VaeCanvas view={view}/>}<div className="ds-view-caption"><strong>{view}</strong><span>{vaeViews.find(item => item[0] === view)?.[1]}</span></div></div>
      <dl className="ds-vae-stats">{[['Anomaly score', '0.92'], ['Reconstruction error', '0.087'], ['Decision threshold', '0.65'], ['Anomalous area', '14.3%'], ['Candidate regions', '3'], ['VAE processing', '18 ms']].map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl>
    </div>
  </section>
}

function YoloOutput({ items }: { items: DetectionRecord[] }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const context = ref.current?.getContext('2d'); if (!context) return
    const width = 680, height = 310, image = context.createImageData(width, height)
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const p = (y * width + x) * 4, side = Math.abs(x - width / 2), lane = Math.exp(-(side ** 2) / 1200)
      let value = 25 + pseudoNoise(x, y) ** 2.8 * 115 + 32 * Math.abs(Math.sin(y * .13 + x * .02)) - lane * 90
      value += anomalyValue(x / width, y / height) * 120
      image.data[p] = value * .82; image.data[p + 1] = value * .9; image.data[p + 2] = value; image.data[p + 3] = 255
    }
    context.putImageData(image, 0, 0)
    const boxes = [[.44,.26,.27,.56,'Shipwreck 92%','#ef6a5b'],[.16,.48,.18,.25,'Pipe 84%','#5d8fe8'],[.74,.18,.17,.28,'Debris 71%','#e8ad4d']] as const
    context.font = '600 13px Arial'
    boxes.forEach(([x,y,w,h,label,color]) => { context.strokeStyle=color; context.lineWidth=3; context.strokeRect(x*width,y*height,w*width,h*height); const tw=context.measureText(label).width+14; context.fillStyle=color; context.fillRect(x*width,y*height-23,tw,23); context.fillStyle='#06101a'; context.fillText(label,x*width+7,y*height-7) })
  }, [])
  const top = items.slice(0, 3)
  return <section className="ds-panel ds-yolo-panel"><header className="ds-panel-title"><h2>YOLO detection & classification</h2><span className="ds-mock-label">3 candidates</span></header><div className="ds-yolo-layout"><div className="ds-yolo-frame"><canvas ref={ref} width="680" height="310" role="img" aria-label="Simulated side-scan sonar frame with Shipwreck, Pipe and Debris YOLO bounding boxes"/><span><ScanSearch size={14}/> Bounding boxes · class labels · confidence</span></div><div className="ds-yolo-results">{top.map((item, index) => <article key={item.id}><i style={{ background: ['#ef6a5b','#5d8fe8','#e8ad4d'][index] }}/><div><strong>{index === 0 ? 'Shipwreck' : index === 1 ? 'Pipe' : 'Other debris'}</strong><span>Detection #{item.id}</span></div><b>{[92,84,71][index]}%</b><CheckCircle2 size={16}/></article>)}<p><Box size={15}/> Non-max suppression applied at IoU 0.45</p></div></div></section>
}

export function ModelOutput({ items }: { items: DetectionRecord[] }) {
  return <><div className="ds-run-strip"><div><span>Processed sonar log</span><strong>IB-2505-17.xtf</strong></div><div><span>Pipeline status</span><strong className="ds-green"><CheckCircle2 size={14}/> Complete</strong></div><div><span>Output</span><strong>243 detections</strong></div><div><span>GPS metadata</span><strong className="ds-green">Synchronized</strong></div><div><span>Processed</span><strong>21 May 2025 · 14:32</strong></div></div>
    <YoloOutput items={items}/><VaeExplorer/>
    <section className="ds-panel ds-model-panel"><header className="ds-panel-title"><h2>YOLO model output statistics</h2><span className="ds-mock-label">Mock evaluation · not measured</span></header><div className="ds-model-metrics">{modelMetrics.map(([name,value,level]) => <article key={name}><div><span>{name}</span>{level && <small className={level === 'Optional' ? 'optional' : ''}>{level}</small>}</div><strong>{value}</strong></article>)}</div></section>
  </>
}

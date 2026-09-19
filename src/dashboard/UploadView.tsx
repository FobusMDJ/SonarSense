import { useRef, useState } from 'react'
import { CheckCircle2, CloudUpload, FileArchive, FileCheck2, FileWarning, Loader2, UploadCloud } from 'lucide-react'
import { ApiError, uploadLog, uploadSurveyZip, watchProgress, type ProgressEvent } from '../lib/api'

const STAGE_LABELS: Record<string, string> = {
  ingest: 'Ingesting sonar log',
  preprocess: 'Preprocessing frame (denoise + contrast + resize)',
  detect: 'Running YOLO26 detection',
  vae: 'Running VAE anomaly analysis',
  confidence: 'Scoring confidence & geolocating detections',
  geolocate: 'Geolocating detections',
  persist: 'Saving to database',
  done: 'Done',
  error: 'Error',
  closed: 'Connection closed',
}

interface UploadViewProps {
  onProcessed: (logId: string) => void
}

type Phase = 'idle' | 'uploading' | 'processing' | 'done' | 'error'

export function UploadView({ onProcessed }: UploadViewProps) {
  const fileInput = useRef<HTMLInputElement>(null)
  const navInput = useRef<HTMLInputElement>(null)
  const zipInput = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [navFile, setNavFile] = useState<File | null>(null)
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [archiveError, setArchiveError] = useState('')
  const [denoiseMethod, setDenoiseMethod] = useState<'none' | 'lee' | 'blind2unblind'>('none')
  const [contrastMethod, setContrastMethod] = useState<'none' | 'clahe' | 'histeq'>('none')
  const [yoloConf, setYoloConf] = useState(0.25)
  const [pixelsToMeters, setPixelsToMeters] = useState(0.15)
  const [phase, setPhase] = useState<Phase>('idle')
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [errorMessage, setErrorMessage] = useState('')
  const [doneLogId, setDoneLogId] = useState<string | null>(null)

  const latest = events[events.length - 1]

  async function submit() {
    if (!file && !zipFile) return
    setPhase('uploading')
    setEvents([])
    setErrorMessage('')
    try {
      const options = { denoiseMethod, contrastMethod, yoloConf, pixelsToMeters }
      const response = zipFile
        ? await uploadSurveyZip(zipFile, options)
        : await uploadLog(file!, { ...options, navSidecar: navFile })
      setPhase('processing')
      let sawTerminalEvent = false
      watchProgress(
        response.log_id,
        (event) => {
          setEvents(prev => [...prev, event])
          if (event.stage === 'done') {
            sawTerminalEvent = true
            setPhase('done')
            setDoneLogId(response.log_id)
          }
          if (event.stage === 'error') {
            sawTerminalEvent = true
            setPhase('error')
            setErrorMessage(event.message)
          }
        },
        () => {
          if (!sawTerminalEvent) {
            setPhase('error')
            setErrorMessage('The processing connection closed. You can start over or check Records for the server-side result.')
          }
        },
      )
    } catch (err) {
      setPhase('error')
      setErrorMessage(err instanceof ApiError ? err.message : 'Upload failed.')
    }
  }

  function reset() {
    setFile(null); setNavFile(null); setZipFile(null); setArchiveError(''); setPhase('idle'); setEvents([]); setErrorMessage(''); setDoneLogId(null)
    if (fileInput.current) fileInput.current.value = ''
    if (navInput.current) navInput.current.value = ''
    if (zipInput.current) zipInput.current.value = ''
  }

  function selectZip(selected: File | null) {
    setPhase('idle'); setEvents([]); setErrorMessage(''); setDoneLogId(null)
    setArchiveError('')
    if (!selected) { setZipFile(null); return }
    if (!selected.name.toLowerCase().endsWith('.zip')) {
      setZipFile(null); setArchiveError('Choose a file ending in .zip.'); return
    }
    if (selected.size > 256 * 1024 * 1024) {
      setZipFile(null); setArchiveError('ZIP files must be 256 MB or smaller.'); return
    }
    setZipFile(selected); setFile(null); setNavFile(null)
    if (fileInput.current) fileInput.current.value = ''
    if (navInput.current) navInput.current.value = ''
  }

  const busy = phase === 'uploading' || phase === 'processing'

  return (
    <>
      <div className="ds-page-intro">
        <div>
          <h2>Ingest a sonar log</h2>
          <p>Upload a complete survey ZIP, side-scan sonar log, or image to run the full SonarSense processing and geolocation pipeline.</p>
        </div>
      </div>

      <section className="ds-panel ds-upload-panel">
        <header className="ds-panel-title"><h2>Upload</h2></header>
        <div className="ds-upload-grid">
          <div className="ds-archive-section">
            <div className="ds-archive-copy">
              <span className="ds-upload-kicker">Complete survey</span>
              <h3>Upload images and metadata together</h3>
              <p>The ZIP can contain nested frame folders. Include one CSV with <code>frame_index</code>, <code>lat</code>, <code>lon</code>, and <code>heading_deg</code>.</p>
              <ul aria-label="ZIP requirements">
                <li><FileCheck2 size={14} /> Sonar images: PNG, JPG, TIFF, BMP, PGM, PBM, PPM, PNM, or NPY</li>
                <li><FileCheck2 size={14} /> Metadata: metadata.csv, navigation.csv, nav.csv, or the only CSV</li>
                <li><FileCheck2 size={14} /> Maximum archive size: 256 MB</li>
              </ul>
            </div>
            <label className={`ds-dropzone ds-dropzone-zip ${zipFile ? 'has-file' : ''}`}>
              <input ref={zipInput} type="file" accept=".zip,application/zip" disabled={busy}
                     onChange={event => selectZip(event.target.files?.[0] ?? null)} />
              <FileArchive size={28} />
              <strong>{zipFile ? zipFile.name : 'Choose a complete survey ZIP'}</strong>
              <small>{zipFile ? `${(zipFile.size / 1_048_576).toFixed(2)} MB · ready to validate` : 'Frames + metadata CSV in one archive'}</small>
            </label>
            <p className="ds-inline-validation" role={archiveError ? 'alert' : undefined}>{archiveError || 'The archive structure and CSV headers are checked before processing begins.'}</p>
          </div>

          <div className="ds-upload-divider"><span>Or upload individual files</span></div>

          <label className={`ds-dropzone ${file ? 'has-file' : ''}`}>
            <input ref={fileInput} type="file" accept=".xtf,image/*" disabled={busy}
                   onChange={event => { setPhase('idle'); setEvents([]); setErrorMessage(''); setDoneLogId(null); setFile(event.target.files?.[0] ?? null); setZipFile(null); setArchiveError(''); if (zipInput.current) zipInput.current.value = '' }} />
            <UploadCloud size={26} />
            <strong>{file ? file.name : 'Choose a sonar log or image'}</strong>
            <small>{file ? `${(file.size / 1_048_576).toFixed(2)} MB` : '.xtf, .png, .jpg, .tif'}</small>
          </label>

          <label className="ds-dropzone ds-dropzone-secondary">
            <input ref={navInput} type="file" accept=".csv" disabled={busy}
                   onChange={event => setNavFile(event.target.files?.[0] ?? null)} />
            <CloudUpload size={20} />
            <strong>{navFile ? navFile.name : 'Optional: navigation CSV'}</strong>
            <small>frame_index,lat,lon,heading_deg[,altitude_m,timestamp]. This sidecar is ignored for .xtf uploads.</small>
          </label>

          <div className="ds-upload-options">
            <label><span>Denoise method</span>
              <select value={denoiseMethod} disabled={busy} onChange={event => setDenoiseMethod(event.target.value as typeof denoiseMethod)}>
                <option value="none">None (matches training distribution)</option>
                <option value="lee">Lee filter</option>
                <option value="blind2unblind">Blind2Unblind (learned)</option>
              </select>
            </label>
            <label><span>Contrast method</span>
              <select value={contrastMethod} disabled={busy} onChange={event => setContrastMethod(event.target.value as typeof contrastMethod)}>
                <option value="none">None (matches training distribution)</option>
                <option value="clahe">CLAHE</option>
                <option value="histeq">Histogram equalization</option>
              </select>
            </label>
            <label><span>YOLO confidence threshold</span>
              <input type="number" min={0.05} max={0.95} step={0.05} value={yoloConf} disabled={busy}
                     onChange={event => setYoloConf(Number(event.target.value))} />
            </label>
            <label><span>Pixels → meters (across-track)</span>
              <input type="number" min={0.01} max={5} step={0.01} value={pixelsToMeters} disabled={busy}
                     onChange={event => setPixelsToMeters(Number(event.target.value))} />
            </label>
          </div>

          <div className="ds-upload-actions">
            <button className="ds-button primary" disabled={(!file && !zipFile) || busy || Boolean(archiveError)} onClick={submit}>
              {busy ? <Loader2 size={16} className="ds-spin" /> : <CloudUpload size={16} />}
              {busy ? 'Processing…' : zipFile ? 'Upload ZIP & process' : 'Upload & process'}
            </button>
            {phase !== 'idle' && <button className="ds-button" onClick={reset} disabled={busy}>Start over</button>}
          </div>
        </div>
      </section>

      {phase !== 'idle' && (
        <section className="ds-panel ds-upload-progress">
          <header className="ds-panel-title"><h2>Processing status</h2>
            {phase === 'done' && <span className="ds-status-pill"><CheckCircle2 size={15} /> Done</span>}
            {phase === 'error' && <span className="ds-status-pill ds-status-pill-error"><FileWarning size={15} /> Error</span>}
          </header>
          {phase === 'uploading' && <p className="ds-panel-note"><Loader2 size={14} className="ds-spin" /> Uploading file to the backend…</p>}
          {latest && phase === 'processing' && (
            <p className="ds-panel-note">
              <Loader2 size={14} className="ds-spin" /> {STAGE_LABELS[latest.stage] ?? latest.stage}
              {latest.n_frames_total ? ` · frame ${((latest.frame_index ?? 0) + 1)}/${latest.n_frames_total}` : ''}
            </p>
          )}
          {phase === 'error' && <p className="ds-panel-note ds-error-text">{errorMessage}</p>}
          {events.length > 0 && (
            <ul className="ds-progress-log">
              {events.slice(-12).map((event, index) => (
                <li key={index}>
                  <b>{STAGE_LABELS[event.stage] ?? event.stage}</b>
                  <span>{event.message}</span>
                </li>
              ))}
            </ul>
          )}
          {phase === 'done' && doneLogId && (
            <div className="ds-upload-done-actions">
              <button className="ds-button primary" onClick={() => onProcessed(doneLogId)}>View pipeline results</button>
            </div>
          )}
        </section>
      )}
    </>
  )
}

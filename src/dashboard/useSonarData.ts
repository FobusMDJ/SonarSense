import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, getDetections, getVaeStats, listLogs, type LogSummary } from '../lib/api'
import { detectionRecordFromRaw, isHumanClass, missionNameFor, type DetectionRecord } from './data'

export interface AnomalousFrame {
  logId: string
  frameRecordId: string
  wholeImageError: number | null
  percentile: number | null
}

export interface SonarData {
  logs: LogSummary[]
  doneLogs: LogSummary[]
  detections: DetectionRecord[]
  topAnomalousFrames: AnomalousFrame[]
  meanWholeImageError: number | null
  loading: boolean
  error: string | null
  refresh: () => void
}

/** Single real-data source for every Dashboard view: fetches every log,
 * then every completed log's detections, and adapts them into the
 * DetectionRecord shape the existing table/chart/map components already
 * render. Polls while any log is still processing so an in-progress
 * upload's result appears without a manual refresh. Human detections are
 * dropped here (see class_taxonomy.py's module docstring on why they're
 * never discarded server-side, just kept out of the normal debris view) --
 * this frontend has no separate safety-review surface yet. */
export function useSonarData(): SonarData {
  const [logs, setLogs] = useState<LogSummary[]>([])
  const [detections, setDetections] = useState<DetectionRecord[]>([])
  const [topAnomalousFrames, setTopAnomalousFrames] = useState<AnomalousFrame[]>([])
  const [meanWholeImageError, setMeanWholeImageError] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const generation = useRef(0)

  const load = useCallback(async () => {
    const myGeneration = ++generation.current
    try {
      const allLogs = await listLogs()
      if (myGeneration !== generation.current) return
      setLogs(allLogs)
      const done = allLogs.filter(log => log.status === 'done')
      const perLog = await Promise.all(done.map(async log => {
        try {
          const raw = await getDetections(log.id)
          return raw.filter(d => !isHumanClass(d.class_name)).map(d => detectionRecordFromRaw(d, missionNameFor(log)))
        } catch {
          return [] // one bad log shouldn't blank the whole dashboard
        }
      }))
      if (myGeneration !== generation.current) return
      setDetections(perLog.flat())

      const vaeResults = await Promise.all(done.map(async log => {
        try { return { log, stats: await getVaeStats(log.id) } } catch { return null }
      }))
      if (myGeneration !== generation.current) return
      const withFrames = vaeResults.filter((r): r is NonNullable<typeof r> => r != null && r.stats.n_frames_analyzed > 0)
      const allFrames: AnomalousFrame[] = withFrames.flatMap(({ log, stats }) =>
        stats.most_anomalous_frames.map(f => ({
          logId: log.id, frameRecordId: f.frame_record_id,
          wholeImageError: f.whole_image_error, percentile: f.percentile,
        })),
      )
      allFrames.sort((a, b) => (a.percentile ?? 1) - (b.percentile ?? 1))
      setTopAnomalousFrames(allFrames.slice(0, 8))
      const errors = withFrames.map(r => r.stats.mean_whole_image_error)
      setMeanWholeImageError(errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : null)

      setError(null)
    } catch (err) {
      if (myGeneration !== generation.current) return
      setError(err instanceof ApiError ? err.message : 'Could not load data from the backend.')
    } finally {
      if (myGeneration === generation.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    const anyProcessing = logs.some(log => log.status === 'processing')
    if (!anyProcessing) return
    const interval = window.setInterval(load, 5000)
    return () => window.clearInterval(interval)
  }, [logs, load])

  return {
    logs, doneLogs: logs.filter(l => l.status === 'done'), detections,
    topAnomalousFrames, meanWholeImageError, loading, error, refresh: load,
  }
}

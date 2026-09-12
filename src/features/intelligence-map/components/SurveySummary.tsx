import { Activity, Anchor, BadgeCheck, ScanSearch } from 'lucide-react'
import type { Survey, SurveySummaryData } from '../types'

interface SurveySummaryProps {
  survey: Survey
  summary: SurveySummaryData
}

export function SurveySummary({ survey, summary }: SurveySummaryProps) {
  return (
    <section className="ss-summary" aria-label="Survey summary">
      <div className="ss-summary__identity">
        <span className="ss-kicker">{survey.id}</span>
        <strong>{survey.name}</strong>
      </div>
      <div className="ss-summary__metrics">
        <span><Anchor size={15} aria-hidden="true" />{survey.platform}</span>
        <span><BadgeCheck size={15} aria-hidden="true" />{survey.status}</span>
        <span><ScanSearch size={15} aria-hidden="true" /><b>{summary.totalDetections}</b> detections</span>
        <span className="ss-summary__high"><Activity size={15} aria-hidden="true" /><b>{summary.highPriorityCount}</b> high</span>
        <span><b>{summary.averageConfidence}%</b> avg. confidence</span>
      </div>
    </section>
  )
}

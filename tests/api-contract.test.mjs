import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const api = readFileSync(new URL('../src/lib/api.ts', import.meta.url), 'utf8')
const pipeline = readFileSync(new URL('../src/dashboard/PipelineView.tsx', import.meta.url), 'utf8')
const upload = readFileSync(new URL('../src/dashboard/UploadView.tsx', import.meta.url), 'utf8')
const anomalySurface = readFileSync(new URL('../src/dashboard/AnomalySurface3D.tsx', import.meta.url), 'utf8')
const taxonomy = readFileSync(new URL('../backend/src/backend/class_taxonomy.py', import.meta.url), 'utf8')

test('frontend consumes model metadata and runtime stats from the backend', () => {
  assert.match(api, /getModelMetadata/)
  assert.match(api, /mean_inference_ms/)
  assert.match(api, /raw_counts_by_class/)
  assert.match(pipeline, /Training validation/)
  assert.match(pipeline, /This processed log/)
  assert.doesNotMatch(pipeline, /78%|92%|243/)
})

test('released raw classes and human safety handling remain explicit', () => {
  assert.match(taxonomy, /aircraft, human, ship, pipe/)
  assert.match(taxonomy, /ship maps to Shipwreck/)
  assert.match(pipeline, /Human safety review/)
  assert.match(pipeline, /Human results are never grouped with marine debris/)
  assert.match(pipeline, /det\.class_name/)
})

test('pipeline retains the seven backend VAE panels', () => {
  for (const filename of [
    '01_original.png', '02_reconstruction.png', '03_anomaly_overlay.png',
    '04_difference_heatmap.png', '05_edge_contour_map.png',
    '06_difference_map_legend.png', '07_3d_anomaly_surface.png',
  ]) assert.match(api, new RegExp(filename.replace('.', '\\.')))
})

test('complete survey ZIP upload uses a validated dedicated API flow', () => {
  assert.match(api, /uploadSurveyZip/)
  assert.match(api, /\/logs\/upload_zip/)
  assert.match(upload, /Complete survey/)
  assert.match(upload, /metadata\.csv/)
  assert.match(upload, /256 MB/)
  assert.match(upload, /processing connection closed/i)
})

test('VAE anomaly surface is a lazy interactive 3D data view', () => {
  assert.match(api, /getVaeSurfaceData/)
  assert.match(pipeline, /lazy\(\(\) => import\('\.\/AnomalySurface3D'\)\)/)
  assert.match(anomalySurface, /<Canvas/)
  assert.match(anomalySurface, /<OrbitControls/)
  assert.match(anomalySurface, /Peak height/)
  assert.match(anomalySurface, /Reset view/)
})

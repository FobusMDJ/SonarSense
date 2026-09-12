import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useEffect, useMemo, useRef, useState, type ComponentRef } from 'react'
import { BufferAttribute, BufferGeometry, Color, DoubleSide } from 'three'
import { RotateCcw } from 'lucide-react'
import { getVaeSurfaceData } from '../lib/api'

const SURFACE_WIDTH = 6.4
const SURFACE_DEPTH = 4.2

interface SurfaceData {
  geometry: BufferGeometry
  values: Float32Array
  size: number
}

function surfaceColor(value: number): Color {
  const stops = [
    [0, '#071426'], [.24, '#075f88'], [.5, '#17b9b0'], [.74, '#f3c94f'], [1, '#ff5b42'],
  ] as const
  const upper = stops.findIndex(([position]) => position >= value)
  if (upper <= 0) return new Color(stops[0][1])
  const [aPos, aColor] = stops[upper - 1]
  const [bPos, bColor] = stops[upper]
  return new Color(aColor).lerp(new Color(bColor), (value - aPos) / (bPos - aPos))
}

function makeSurface(inputValues: number[], size: number): SurfaceData {
  const values = Float32Array.from(inputValues)
  const positions: number[] = []
  const colors: number[] = []
  const indices: number[] = []

  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      const index = row * size + column
      const value = values[index]
      positions.push(
        (column / (size - 1) - .5) * SURFACE_WIDTH,
        (row / (size - 1) - .5) * SURFACE_DEPTH,
        Math.pow(value, .82) * 1.65,
      )
      const color = surfaceColor(value)
      colors.push(color.r, color.g, color.b)
      if (row < size - 1 && column < size - 1) {
        const nextRow = index + size
        indices.push(index, nextRow, index + 1, index + 1, nextRow, nextRow + 1)
      }
    }
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3))
  geometry.setAttribute('color', new BufferAttribute(new Float32Array(colors), 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return { geometry, values, size }
}

function Surface({ data, heightScale, onProbe }: { data: SurfaceData; heightScale: number; onProbe: (value: number | null) => void }) {
  return (
    <group rotation={[-Math.PI / 2, 0, 0]} scale={[1, 1, heightScale]}>
      <mesh
        geometry={data.geometry}
        onPointerMove={event => {
          if (!event.uv) return
          const column = Math.min(data.size - 1, Math.max(0, Math.round(event.uv.x * (data.size - 1))))
          const row = Math.min(data.size - 1, Math.max(0, Math.round((1 - event.uv.y) * (data.size - 1))))
          onProbe(data.values[row * data.size + column])
        }}
        onPointerOut={() => onProbe(null)}
      >
        <meshStandardMaterial vertexColors roughness={.58} metalness={.08} side={DoubleSide} />
      </mesh>
      <mesh geometry={data.geometry} position={[0, 0, .008]}>
        <meshBasicMaterial color="#9fe8ff" wireframe transparent opacity={.085} />
      </mesh>
    </group>
  )
}

export default function AnomalySurface3D({ logId, frameRecordId, fallbackUrl }: {
  logId: string
  frameRecordId: string
  fallbackUrl: string
}) {
  const [data, setData] = useState<SurfaceData | null>(null)
  const [error, setError] = useState('')
  const [heightScale, setHeightScale] = useState(1)
  const [probe, setProbe] = useState<number | null>(null)
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null)

  useEffect(() => {
    let cancelled = false
    setData(null); setError(''); setProbe(null)
    getVaeSurfaceData(logId, frameRecordId)
      .then(surface => { if (!cancelled) setData(makeSurface(surface.values, surface.size)) })
      .catch(reason => { if (!cancelled) setError(reason instanceof Error ? reason.message : '3D surface unavailable.') })
    return () => { cancelled = true }
  }, [logId, frameRecordId])

  useEffect(() => () => data?.geometry.dispose(), [data])
  const status = useMemo(() => probe == null ? 'Move over the terrain to inspect anomaly intensity' : `Anomaly intensity ${Math.round(probe * 100)}%`, [probe])

  if (error) return (
    <div className="ds-surface-fallback">
      <img src={fallbackUrl} alt="Static 3D anomaly surface fallback" />
      <p>{error} Showing the generated static surface instead.</p>
    </div>
  )

  return (
    <div className="ds-surface-shell" role="img" aria-label="Interactive 3D reconstruction-error surface">
      <div className="ds-surface-toolbar">
        <div><strong>Interactive anomaly terrain</strong><span>Drag to orbit · scroll or pinch to zoom</span></div>
        <label htmlFor="surface-height">Peak height
          <input id="surface-height" type="range" min="0.45" max="1.8" step="0.05" value={heightScale}
                 onChange={event => setHeightScale(Number(event.target.value))} />
        </label>
        <button className="ds-button" onClick={() => controls.current?.reset()}><RotateCcw size={14} /> Reset view</button>
      </div>
      <div className="ds-surface-canvas">
        {!data && <div className="ds-surface-loading">Building reconstruction-error terrain…</div>}
        {data && (
          <Canvas
            camera={{ position: [5.2, 4.2, 5.6], fov: 39, near: .1, far: 100 }}
            dpr={[1, 1.5]}
            frameloop="demand"
            fallback={<img src={fallbackUrl} alt="Static 3D anomaly surface fallback" />}
          >
            <color attach="background" args={['#040b13']} />
            <ambientLight intensity={.9} />
            <directionalLight position={[3, 7, 4]} intensity={2.2} color="#d9f4ff" />
            <directionalLight position={[-5, 2, -2]} intensity={1.1} color="#1e91b8" />
            <gridHelper args={[8, 16, '#36566f', '#152d40']} position={[0, -.04, 0]} />
            <Surface data={data} heightScale={heightScale} onProbe={setProbe} />
            <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={.08} minDistance={4.5} maxDistance={12} minPolarAngle={.25} maxPolarAngle={Math.PI / 2.04} />
          </Canvas>
        )}
      </div>
      <div className="ds-surface-status" aria-live="polite"><span>{status}</span><i aria-hidden="true" /></div>
    </div>
  )
}

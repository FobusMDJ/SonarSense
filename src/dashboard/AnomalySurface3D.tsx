import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

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

function SurfaceMesh() {
  const geometry = useMemo(() => {
    const mesh = new THREE.PlaneGeometry(6.8, 4.2, 58, 42)
    const position = mesh.attributes.position
    const colors: number[] = []
    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i), y = position.getY(i)
      const height = anomalyValue(x / 6.8 + .5, y / 4.2 + .5) * 2.15 + pseudoNoise(x * 10, y * 10) * .055
      position.setZ(i, height)
      const [r, g, b] = heatColor(Math.min(1, height / 2.15)); colors.push(r / 255, g / 255, b / 255)
    }
    mesh.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3)); mesh.computeVertexNormals()
    return mesh
  }, [])
  return <mesh geometry={geometry} rotation={[-1.02, 0, -.13]}><meshStandardMaterial vertexColors roughness={.7} metalness={.08} side={THREE.DoubleSide}/></mesh>
}

export default function AnomalySurface3D() {
  return <div className="ds-surface" role="img" aria-label="Interactive 3D surface of simulated VAE anomaly scores; higher peaks indicate stronger reconstruction error">
    <Canvas camera={{ position: [0, 3.1, 6.2], fov: 43 }} dpr={[1, 1.5]} gl={{ antialias: true }}>
      <color attach="background" args={['#06101c']}/><ambientLight intensity={1.4}/><directionalLight position={[3, 5, 5]} intensity={2.2}/><SurfaceMesh/><gridHelper args={[8, 16, '#29425d', '#152a3e']} position={[0, -.05, 0]}/><OrbitControls enablePan={false} minDistance={4.8} maxDistance={8} minPolarAngle={.6} maxPolarAngle={1.35}/>
    </Canvas><span>Drag to inspect · scroll to zoom</span>
  </div>
}

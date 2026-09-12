# SonarSense — SIH26057

SonarSense is a local-first side-scan sonar intelligence system. It combines a Vite/React dashboard with a FastAPI inference service for sonar uploads, live pipeline progress, YOLO detections, VAE anomaly analysis, interactive 3D anomaly surfaces, GPS geolocation, map visualization, persistent records, and downloadable reports.

The released detector has four raw classes: `aircraft`, `human`, `ship`, and `pipe`. The dashboard maps `ship` to **Shipwreck** and `aircraft` to **Other Debris** where grouped labels are needed. Human detections remain in a separate safety-review category and are never counted as marine debris.

## Project structure

```text
.
├── src/                         React/Vite frontend
├── backend/
│   ├── src/backend/             FastAPI API and pipeline orchestration
│   ├── src/geolocation/         Navigation parsing and georeferencing
│   ├── src/vae/                 VAE model and anomaly processing
│   ├── src/yolo/                YOLO detection integration
│   ├── scripts/                 Weight setup and smoke-test helpers
│   └── docker-compose.local.yml Local backend configuration
└── tests/                       Frontend/API contract tests
```

## Requirements

- Docker Desktop (running)
- Node.js 20 or newer
- pnpm (`npm install --global pnpm` if it is not installed)
- [GitHub CLI](https://cli.github.com/) authenticated with access to the private [`FobusMDJ/SonarSense`](https://github.com/FobusMDJ/SonarSense) release assets

Model checkpoints are intentionally not committed to Git. The setup script downloads the three `weights-v1` assets from the source repository and verifies their fixed SHA-256 digests before installing them.

## Run locally

Clone the repository and enter it:

```bash
git clone https://github.com/FobusMDJ/The-SonarSense-Project-.git
cd The-SonarSense-Project-
```

Authenticate GitHub CLI if needed, download the verified checkpoints, and start the backend:

```bash
gh auth login
cd backend
./scripts/setup_weights.sh
docker compose -f docker-compose.local.yml up --build -d
```

Confirm that the API and models are ready:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/metadata
```

In a second terminal, start the frontend from the repository root:

```bash
cp .env.example .env.local
pnpm install
pnpm dev --host 127.0.0.1
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The frontend connects to `http://127.0.0.1:8000` through `VITE_API_BASE_URL` in `.env.local`.

If port `8000` is already occupied, start Docker on another port and update `.env.local` to match:

```bash
cd backend
SONARSENSE_PORT=8001 docker compose -f docker-compose.local.yml up --build -d
```

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8001
```

## Upload and process a survey

The Upload screen accepts either individual sonar images with an optional navigation CSV, or a complete survey ZIP. For reliable geolocation, the CSV should contain these columns:

```csv
frame_index,lat,lon,heading_deg
0,18.2345,72.3456,90
1,18.2346,72.3457,91
```

A ZIP may contain nested image folders, but image basenames must be unique. The metadata file should be named `metadata.csv`, `navigation.csv`, `nav.csv`, `nav_sidecar.csv`, or `dataset_metadata.csv`. If the archive contains only one CSV, it is used automatically.

```text
survey.zip
└── survey/
    ├── frames/
    │   ├── frame_000.png
    │   └── frame_001.png
    └── metadata.csv
```

The upload limit is 256 MB. Unsafe paths, encrypted archives, symlinks, ambiguous metadata, and oversized extraction are rejected before inference. Processing time depends on archive size, image count, CPU speed, and whether Docker has sufficient memory; progress is streamed live to the Pipeline screen.

## Stop or restart

Stop the backend while keeping its SQLite database and generated files:

```bash
docker compose -f backend/docker-compose.local.yml down
```

Restart it later with:

```bash
docker compose -f backend/docker-compose.local.yml up -d
```

Runtime data is persisted under `backend/runtime/` and is excluded from Git. Stop the frontend with `Ctrl+C` in its terminal.

## Verification

Run backend tests and deterministic smoke checks:

```bash
cd backend
python3 -m unittest discover -s tests
python3 scripts/create_smoke_fixture.py
python3 scripts/smoke_models.py
python3 scripts/smoke_api.py
```

Run frontend contract tests and the production build:

```bash
cd ..
node --test tests/*.test.mjs
pnpm build
```

The synthetic fixture validates model execution and data flow, not field accuracy. Field validation still requires a representative XTF or real sonar log with trustworthy navigation data. Default preprocessing is `denoise=none` and `contrast=none`, matching the released detector/VAE training distribution. Blind2Unblind is experimental and optional.

## Main API routes

- `GET /health` — service, model integrity, device, and XTF-reader status
- `GET /model/metadata` — training metrics, raw classes, checkpoint digests, VAE architecture, and denoiser status
- `POST /logs/upload` — upload sonar data with an optional navigation CSV
- `POST /logs/upload_zip` — validate and process nested images plus metadata from one ZIP
- `WS /ws/logs/{log_id}` — live pipeline progress
- `GET /logs/{log_id}/stats` — per-log counts, confidence threshold, inference time, and FPS
- `GET /logs/{log_id}/frames/{frame_record_id}/vae/surface` — normalized reconstruction-error grid for the interactive 3D anomaly surface
- Additional endpoints under `/logs/{log_id}` provide detections, seven-panel VAE output, maps, geolocation provenance, and JSON/CSV/GeoJSON/PDF reports.

## Notes

- Local results are stored in SQLite at `backend/runtime/sonarsense.db`.
- Uploaded files and generated outputs persist under `backend/runtime/` across container restarts.
- The backend serializes local inference work and caches loaded models to avoid duplicate model memory use.
- The source release and expected checkpoint digests are defined in `backend/scripts/setup_weights.sh`.

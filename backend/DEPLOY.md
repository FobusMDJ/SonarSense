# Deploying SonarSense (backend + frontend)

Two separate repos: this backend, and the frontend at
https://github.com/FobusMDJ/SonarSense. They talk over HTTP, so they can be
deployed on the same machine or completely different ones.

## 1. Backend

### Option A — Docker (recommended, works "anywhere")

Needs: Docker installed on the target machine, your trained weights (`best.pt`,
`src/vae/vae_epoch100.pth`, optionally `models/B2Ueph2.pth`).

```bash
git clone <your-backend-repo-url> sonarsense-backend
cd sonarsense-backend
# put your model weights at the paths docker-compose.yml expects:
#   ./best.pt
#   ./src/vae/vae_epoch100.pth
#   ./models/  (if using blind2unblind denoising)

# edit config/backend.yaml:
#   db_backend: postgres
#   postgres_dsn: postgresql://sonarsense:sonarsense@db:5432/sonarsense
#   (change the password for anything beyond local testing)

docker compose up --build -d
```

Backend is now live at `http://<machine-ip>:8000`. Postgres+PostGIS comes up
automatically as the `db` service — no manual database setup needed.

### Option B — Plain Python (any VM/cloud host)

```bash
git clone <your-backend-repo-url> sonarsense-backend
cd sonarsense-backend
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# place model weights at the paths config/backend.yaml points to
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

SQLite is the default (`db_backend: sqlite` in `config/backend.yaml`) — zero
extra setup, good for a single-machine deploy. Switch to `postgres` + set
`postgres_dsn` if you want a real spatial database (needs a reachable
Postgres+PostGIS instance and `psycopg2-binary`, already in requirements.txt).

### Backend config checklist before going live

- `config/backend.yaml`: `yolo_weights_path`, `vae_weights_path` point at your
  real trained checkpoints (not the sandbox's fake ones).
- CORS is wide open (`allow_origins=["*"]`) in `src/backend/main.py` — fine for
  a demo, tighten to your actual frontend domain before any public deployment.
- No auth on any endpoint — same caveat, especially `/logs/ingest_local` which
  reads arbitrary server-local paths.

## 2. Frontend (FobusMDJ/SonarSense)

```bash
git clone https://github.com/FobusMDJ/SonarSense frontend
cd frontend
cp .env.example .env
# edit .env — set VITE_API_BASE_URL to wherever your backend ends up, e.g.:
#   VITE_API_BASE_URL=https://sonarsense-backend.yourdomain.com
#   VITE_API_BASE_URL=http://<machine-ip>:8000   (if same network/host)
pnpm install
pnpm build      # outputs static files to dist/
```

Deploy `dist/` to Vercel (the repo already has `vercel.json`), Netlify, or any
static host. Set `VITE_API_BASE_URL` as an environment variable in your
hosting provider's dashboard too (not just the local `.env`), since the build
step bakes it in.

If the backend is unreachable, the Intelligence Map page automatically falls
back to its bundled mock data — nothing breaks, it just won't show a "Live
backend data" badge.

## 3. Quick local test (both on one machine, no Docker)

Terminal 1:
```bash
cd sonarsense-backend && source .venv/bin/activate
python -m uvicorn src.backend.main:app --port 8000
```

Terminal 2:
```bash
cd frontend
echo "VITE_API_BASE_URL=http://localhost:8000" > .env
pnpm install && pnpm dev
```

Open the frontend dev URL, go to the Intelligence Map page — once you've
processed at least one log through the backend (any of `/logs/upload`,
`/logs/upload_dir`, `/logs/ingest_local`) and it reaches `status: done`, it'll
show up as a live survey there instead of the mock one.

# Backend service image. Model weights (best.pt, vae_epoch100.pth, B2U*.pth)
# are NOT baked in here (they're large, and change independently of code).
# Two ways to supply them at runtime:
#   1. A host with a real persistent disk (e.g. docker-compose's volume
#      mounts below, or a Render paid disk): mount them at the paths
#      config/backend.yaml expects.
#   2. A host with NO disk/shell access (e.g. Render's free tier, whose
#      filesystem is wiped on every restart): docker-entrypoint.sh downloads
#      them at container startup from YOLO_WEIGHTS_URL / VAE_WEIGHTS_URL /
#      B2U_WEIGHTS_URL env vars -- see that script's header comment.

FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0: opencv-python-headless still dynamically links a
# couple of system libs at import time even in the "headless" build.
# curl: used by docker-entrypoint.sh to fetch weights on hosts with no disk.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]

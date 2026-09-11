# Backend service image. Model weights (best.pt, vae_epoch100.pth, B2U*.pth)
# are NOT baked in here (they're large, and change independently of code) --
# mount them at runtime, e.g.:
#   docker run -v ./best.pt:/app/best.pt -v ./src/vae:/app/src/vae ...
# or bake your own derived image with a COPY line if you'd rather ship them
# in the image. See docker-compose.yml for the mounted-weights setup this
# Dockerfile is meant to pair with.

FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0: opencv-python-headless still dynamically links a
# couple of system libs at import time even in the "headless" build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

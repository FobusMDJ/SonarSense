#!/bin/sh
# Downloads model weights at container startup if they're not already
# present in the image -- needed for hosts like Render's free tier where
# the filesystem is ephemeral (wiped on every restart/redeploy) and there's
# no shell access to upload files after deploy. Set these as environment
# variables on your hosting platform to a direct-download URL (a private
# GitHub Release asset, an S3/GCS signed URL, a HuggingFace file, etc.):
#
#   YOLO_WEIGHTS_URL  -> downloaded to /app/best.pt
#   VAE_WEIGHTS_URL   -> downloaded to /app/src/vae/vae_epoch100.pth
#   B2U_WEIGHTS_URL   -> downloaded to /app/models/B2Ueph2.pth (optional --
#                        only needed if you use denoise_method=blind2unblind)
#
# If a variable is unset, that weight file is skipped (matches this
# backend's existing "warn, don't crash" behavior for missing weights --
# see main.py's /health checks). Already-present files are never
# re-downloaded, so this is a no-op on a host with a real persistent disk.

set -e

fetch() {
  url="$1"; dest="$2"
  if [ -n "$url" ] && [ ! -f "$dest" ]; then
    echo "Downloading $(basename "$dest") ..."
    mkdir -p "$(dirname "$dest")"
    curl -fsSL "$url" -o "$dest"
  fi
}

fetch "$YOLO_WEIGHTS_URL" /app/best.pt
fetch "$VAE_WEIGHTS_URL" /app/src/vae/vae_epoch100.pth
fetch "$B2U_WEIGHTS_URL" /app/models/B2Ueph2.pth

exec python -m uvicorn src.backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"

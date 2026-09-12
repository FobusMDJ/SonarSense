#!/bin/sh
set -eu

repo="FobusMDJ/SonarSense"
tag="weights-v1"
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backend_dir=$(dirname "$script_dir")
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sonarsense-weights.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

command -v gh >/dev/null 2>&1 || { echo "GitHub CLI (gh) is required." >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Authenticate first with: gh auth login" >&2; exit 1; }

gh release download "$tag" --repo "$repo" --dir "$tmp_dir" \
  --pattern best.pt --pattern vae_epoch100.pth --pattern B2Ueph2.pth --clobber

verify() {
  file="$1"
  expected="$2"
  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$file" | awk '{print $1}')
  else
    actual=$(shasum -a 256 "$file" | awk '{print $1}')
  fi
  [ "$actual" = "$expected" ] || {
    echo "SHA-256 mismatch for $(basename "$file"): expected $expected, got $actual" >&2
    exit 1
  }
}

verify "$tmp_dir/best.pt" "859308954e6be6c3f4ecb3038a390973294040a5ec7dc0cd65aa996d09657d4b"
verify "$tmp_dir/vae_epoch100.pth" "62da9ae6a92d4da23aad212a8e7aecbbe4fb6a60aef02149e543bee28508fecf"
verify "$tmp_dir/B2Ueph2.pth" "b2c717580aee3b1cbd41dc27b88618726148a80f22f30ed908976a007b4e2188"

mkdir -p "$backend_dir/src/vae" "$backend_dir/models"
install -m 0644 "$tmp_dir/best.pt" "$backend_dir/best.pt"
install -m 0644 "$tmp_dir/vae_epoch100.pth" "$backend_dir/src/vae/vae_epoch100.pth"
install -m 0644 "$tmp_dir/B2Ueph2.pth" "$backend_dir/models/B2Ueph2.pth"
echo "Verified and installed weights-v1 checkpoints."

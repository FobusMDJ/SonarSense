"""Create a deterministic sonar-like 512x512 PNG, nav CSV, and survey ZIP."""

from pathlib import Path
import zipfile

import cv2
import numpy as np

out_dir = Path(__file__).resolve().parents[1] / "runtime" / "smoke"
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(26057)
y, x = np.mgrid[:512, :512]
seafloor = 42 + 18 * np.sin(x / 23.0) + 10 * np.cos(y / 31.0) + rng.normal(0, 8, (512, 512))
seafloor[:, 246:266] *= 0.18
cv2.ellipse(seafloor, (365, 290), (56, 19), -18, 0, 360, 205, -1)
cv2.line(seafloor, (315, 310), (424, 267), 245, 4)
image = np.clip(seafloor, 0, 255).astype(np.uint8)
image_path = out_dir / "sonar_fixture.png"
cv2.imwrite(str(image_path), image)
metadata_path = out_dir / "metadata.csv"
metadata_path.write_text(
    "frame_index,lat,lon,heading_deg,altitude_m,timestamp\n"
    "0,15.29932,73.96301,142.0,18.4,1789207200\n",
    encoding="utf-8",
)
archive_path = out_dir / "survey_fixture.zip"
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(image_path, "survey/frames/sonar_fixture.png")
    archive.write(metadata_path, "survey/metadata.csv")
print(archive_path)

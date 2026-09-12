"""Safe extraction and validation for an uploaded survey ZIP archive."""

from __future__ import annotations

import csv
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
MAX_MEMBERS = 5000
REQUIRED_NAV_COLUMNS = {"frame_index", "lat", "lon", "heading_deg"}
PREFERRED_CSV_NAMES = ("metadata.csv", "navigation.csv", "nav.csv", "nav_sidecar.csv", "dataset_metadata.csv")
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm",
    ".pbm", ".ppm", ".pnm", ".npy",
}


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in ZIP archive: {name}")
    return path


def _choose_metadata_csv(csv_members: list[zipfile.ZipInfo]) -> zipfile.ZipInfo:
    if not csv_members:
        raise ValueError("ZIP must include a metadata CSV with frame_index, lat, lon, and heading_deg columns.")
    for preferred in PREFERRED_CSV_NAMES:
        matches = [item for item in csv_members if PurePosixPath(item.filename).name.lower() == preferred]
        if len(matches) == 1:
            return matches[0]
    if len(csv_members) == 1:
        return csv_members[0]
    names = ", ".join(PurePosixPath(item.filename).name for item in csv_members)
    raise ValueError(f"ZIP contains multiple CSV files ({names}). Name the navigation file metadata.csv.")


def extract_survey_zip(archive_path: Path, destination: Path) -> tuple[Path, Path, int]:
    """Return (flat frames directory, metadata CSV path, image count)."""
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("ZIP is larger than the 256 MB upload limit.")
    if not zipfile.is_zipfile(archive_path):
        raise ValueError("The uploaded file is not a valid ZIP archive.")

    frames_dir = destination / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) > MAX_MEMBERS:
            raise ValueError(f"ZIP contains more than {MAX_MEMBERS} files.")
        if sum(item.file_size for item in members) > MAX_EXTRACTED_BYTES:
            raise ValueError("ZIP expands beyond the 2 GB extraction limit.")

        image_members: list[zipfile.ZipInfo] = []
        csv_members: list[zipfile.ZipInfo] = []
        for item in members:
            path = _safe_member_path(item.filename)
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic links are not allowed in ZIP archives: {item.filename}")
            if item.flag_bits & 0x1:
                raise ValueError("Encrypted ZIP archives are not supported.")
            if any(part.startswith(".") or part == "__MACOSX" for part in path.parts):
                continue
            suffix = path.suffix.lower()
            if suffix in SUPPORTED_IMAGE_EXTENSIONS:
                image_members.append(item)
            elif suffix == ".csv":
                csv_members.append(item)

        if not image_members:
            raise ValueError("ZIP does not contain any supported sonar images.")
        metadata_member = _choose_metadata_csv(csv_members)

        used_names: set[str] = set()
        for item in image_members:
            filename = PurePosixPath(item.filename).name
            key = filename.lower()
            if key in used_names:
                raise ValueError(f"ZIP contains duplicate image filename after flattening: {filename}")
            used_names.add(key)
            with archive.open(item) as source, (frames_dir / filename).open("wb") as output:
                shutil.copyfileobj(source, output)

        metadata_path = destination / "metadata.csv"
        with archive.open(metadata_member) as source, metadata_path.open("wb") as output:
            shutil.copyfileobj(source, output)

    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not REQUIRED_NAV_COLUMNS.issubset(columns):
            raise ValueError(
                f"Metadata CSV must include {sorted(REQUIRED_NAV_COLUMNS)}. Found: {reader.fieldnames or []}"
            )
        if next(reader, None) is None:
            raise ValueError("Metadata CSV contains headers but no navigation rows.")

    return frames_dir, metadata_path, len(image_members)

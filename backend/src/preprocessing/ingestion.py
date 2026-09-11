"""Data ingestion for side-scan sonar imagery.

Three source formats, one output contract (`SSSRecord`):
  - flat image files (PNG/JPEG/TIFF/BMP/PGM/.npy) -- the original scope,
    still used for pre-tiled datasets with no nav data.
  - images + a nav-sidecar CSV -- same image files, plus a per-frame
    lat/lon/heading CSV joined in by position (ingest_directory_with_nav).
  - raw .xtf sonar logs -- parsed via src.geolocation.xtf_reader.ingest_xtf
    (pyxtf-based; UNVALIDATED against a real file, see that module's
    docstring) into waterfall tiles with a per-ping NavFix attached.

`ingest_source()` is the single entry point the backend orchestrator
should call -- it dispatches on the input path so adding a 4th format later
is a one-line change there, not a change at every call site. Everything
downstream (grayscale/dropout/denoise/YOLO/VAE) only ever consumes
`SSSRecord.image`, so none of it needed to change to support these two new
source formats -- exactly the seam this module's original docstring
anticipated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np

from src.utils.io_utils import list_image_files, load_image
from src.utils.config import get_logger

logger = get_logger(__name__)


@dataclass
class SSSRecord:
    """One ingested sonar observation: the raw image plus whatever
    metadata is available for it (currently just filesystem provenance;
    ping/GPS headers will land here once a real log parser exists).
    """

    image: np.ndarray
    source_path: Path
    record_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def ingest_file(path: str | Path) -> SSSRecord:
    """Load a single sonar image file into an `SSSRecord`."""
    path = Path(path)
    image = load_image(path)
    return SSSRecord(
        image=image,
        source_path=path,
        record_id=path.stem,
        metadata={"filename": path.name, "shape": image.shape},
    )


def ingest_directory(directory: str | Path) -> Iterator[SSSRecord]:
    """Yield an `SSSRecord` for every supported image file in `directory`.

    Lazy (generator-based) so large sonar logs / mosaics directories don't
    need to fit in memory all at once.
    """
    directory = Path(directory)
    files = list_image_files(directory)
    if not files:
        logger.warning("No supported image files found in %s", directory)
    for path in files:
        try:
            yield ingest_file(path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Skipping unreadable file %s: %s", path, exc)


def ingest_directory_with_nav(directory: str | Path, nav_sidecar_path: str | Path) -> Iterator[SSSRecord]:
    """Same as ingest_directory(), but attaches a NavFix (lat/lon/heading)
    to each record's metadata["nav_fix"], read from a companion CSV (see
    src.geolocation.nav.load_nav_sidecar for the expected columns). Frames
    are matched to nav rows by POSITION in sorted file order against the
    sidecar's frame_index column -- the sidecar must be exported in the
    same order the images were.
    """
    from src.geolocation.nav import load_nav_sidecar, nearest_fix

    directory = Path(directory)
    files = list_image_files(directory)
    fixes = load_nav_sidecar(nav_sidecar_path)
    if not fixes:
        logger.warning("Nav sidecar %s loaded 0 fixes -- records will have no nav_fix.", nav_sidecar_path)

    for i, path in enumerate(files):
        try:
            record = ingest_file(path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Skipping unreadable file %s: %s", path, exc)
            continue
        record.metadata["nav_fix"] = nearest_fix(fixes, i)
        record.metadata["source_format"] = "image+sidecar"
        yield record


def ingest_source(
    path: str | Path, nav_sidecar_path: str | Path | None = None,
) -> Iterator[SSSRecord]:
    """Single entry point that dispatches on what `path` actually is:
      - a .xtf file  -> src.geolocation.xtf_reader.ingest_xtf
      - a directory  -> ingest_directory_with_nav if nav_sidecar_path is
                         given, else plain ingest_directory
      - a single image file -> yields one record from ingest_file

    This is the function the backend's upload/pipeline orchestrator should
    call rather than picking a specific ingest_* function itself, so adding
    a future source format is a one-line change here, not a change at every
    call site.
    """
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".xtf":
        from src.geolocation.xtf_reader import ingest_xtf
        yield from ingest_xtf(path)
        return
    if path.is_dir():
        if nav_sidecar_path:
            yield from ingest_directory_with_nav(path, nav_sidecar_path)
        else:
            yield from ingest_directory(path)
        return
    yield ingest_file(path)

"""Released SonarSense checkpoint manifest and cached model access."""

from __future__ import annotations

import hashlib
import threading
from functools import lru_cache
from pathlib import Path

RELEASE_TAG = "weights-v1"
ULTRALYTICS_VERSION = "8.4.147"
DETECTOR_CLASSES = ["aircraft", "human", "ship", "pipe"]
DETECTOR_METRICS = {
    "precision": 0.94122,
    "recall": 0.63333,
    "f1_score": 0.75717,
    "map50": 0.6926,
    "map50_95": 0.53874,
}

CHECKPOINTS = {
    "detector": {
        "filename": "best.pt", "size_bytes": 40568843,
        "sha256": "859308954e6be6c3f4ecb3038a390973294040a5ec7dc0cd65aa996d09657d4b",
    },
    "vae": {
        "filename": "vae_epoch100.pth", "size_bytes": 18234987,
        "sha256": "62da9ae6a92d4da23aad212a8e7aecbbe4fb6a60aef02149e543bee28508fecf",
    },
    "denoiser": {
        "filename": "B2Ueph2.pth", "size_bytes": 1974913,
        "sha256": "b2c717580aee3b1cbd41dc27b88618726148a80f22f30ed908976a007b4e2188",
    },
}

# YOLO and PyTorch modules are not treated as concurrently re-entrant. This
# also bounds local model memory to a single active inference job.
INFERENCE_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=12)
def _cached_checkpoint_status(kind: str, path_text: str, size: int, modified_ns: int) -> dict:
    expected = CHECKPOINTS[kind]
    path = Path(path_text)
    actual = sha256_file(path)
    return {
        **expected,
        "path": str(path),
        "found": True,
        "verified": actual == expected["sha256"],
        "actual_sha256": actual,
    }


def checkpoint_status(kind: str, path: Path) -> dict:
    if not path.is_file():
        return {**CHECKPOINTS[kind], "path": str(path), "found": False,
                "verified": False, "actual_sha256": None}
    stat = path.stat()
    return _cached_checkpoint_status(kind, str(path), stat.st_size, stat.st_mtime_ns)


def validate_present_checkpoints(paths: dict[str, Path]) -> dict[str, dict]:
    statuses = {kind: checkpoint_status(kind, path) for kind, path in paths.items()}
    mismatches = [kind for kind, value in statuses.items() if value["found"] and not value["verified"]]
    if mismatches:
        details = ", ".join(f"{kind} ({statuses[kind]['path']})" for kind in mismatches)
        raise RuntimeError(f"Checkpoint integrity verification failed: {details}. Run scripts/setup_weights.sh again.")
    return statuses


@lru_cache(maxsize=2)
def get_yolo_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


@lru_cache(maxsize=2)
def get_vae_model(weights_path: str, device_name: str):
    import torch
    from src.vae.test_vae import load_model
    return load_model(weights_path, torch.device(device_name))


def release_yolo_model() -> None:
    """Drop cached detector references between low-memory pipeline passes."""
    get_yolo_model.cache_clear()


def release_vae_model() -> None:
    """Drop cached VAE references after a low-memory pipeline pass."""
    get_vae_model.cache_clear()


def model_metadata(paths: dict[str, Path]) -> dict:
    import torch
    statuses = validate_present_checkpoints(paths)
    return {
        "release_tag": RELEASE_TAG,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "detector": {
            "name": "YOLO26m",
            "framework": "Ultralytics",
            "framework_version": ULTRALYTICS_VERSION,
            "input_size": 640,
            "training_epochs": 150,
            "training_images": None,
            "raw_classes": DETECTOR_CLASSES,
            "training_metrics": DETECTOR_METRICS,
            "checkpoint": statuses["detector"],
        },
        "vae": {
            "epoch": 100,
            "input_size": 512,
            "latent_dim": 128,
            "channels": [16, 32, 64, 128, 128, 128],
            "checkpoint": statuses["vae"],
        },
        "denoiser": {
            "name": "Blind2Unblind",
            "status": "experimental",
            "epoch": 2,
            "base_width": 32,
            "checkpoint": statuses["denoiser"],
        },
    }

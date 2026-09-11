"""Config loading and logging setup shared across SonarSense modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "preprocessing.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load a YAML config file.

    Falls back to `config/preprocessing.yaml` at the project root if no
    path is given. Returns an empty dict (never raises) if the file is
    missing, so callers can rely on `.get(...)` defaults.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        logging.getLogger(__name__).warning("Config file not found at %s; using defaults.", path)
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent, readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger

"""Backend-selector shim: re-exports either db.py (SQLite) or db_postgis.py
(PostGIS) under one name, based on config/backend.yaml's `db_backend` key,
so the rest of the codebase (main.py, pipeline_runner.py) can do

    from src.backend import db_backend as db

once and never again care which database is actually active. Every
function this re-exports has an IDENTICAL signature across both
implementations (see db_postgis.py's module docstring) -- the only
exception is db_postgis.detections_within_radius, a PostGIS-only bonus
query that is deliberately NOT re-exported here, since there's no SQLite
equivalent and re-exporting it would make it look like part of the
portable surface.

USAGE: call `configure(cfg, project_root)` ONCE at startup (main.py's
existing `_cfg = load_config(...)` call site) -- it picks the backend AND
returns the resolved connection target (a `Path` for sqlite, a DSN string
for postgres). Assign that return value to the same `DB_PATH` variable
main.py already threads through every `db.init_db(DB_PATH)` /
`db.get_connection(DB_PATH)` / `run_pipeline(..., db_path=DB_PATH, ...)`
call site -- none of those call sites need to change, because init_db()
and get_connection() below accept that same opaque target and forward it
to whichever backend is active under the right keyword
('db_path' for db.py, 'dsn' for db_postgis.py). Everything else
(create_log, insert_detection, list_detections, ...) takes an
already-open `conn` as its first arg on both backends, so those are
forwarded straight through via __getattr__ with no target-binding needed
at all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from src.utils.config import get_logger

logger = get_logger(__name__)

_active_module = None
_target_kwarg: Optional[str] = None
_default_target: Any = None


def configure(cfg: dict, project_root: Optional[Path] = None) -> Any:
    """Pick the backend from cfg['db_backend'] ('sqlite' | 'postgres',
    default 'sqlite' -- the zero-setup option stays the default so existing
    deployments/configs that don't set this key are completely unaffected).
    Returns the resolved connection target to pass to init_db()/
    get_connection() at every existing call site.

    The DB_BACKEND / POSTGRES_DSN environment variables, when set, override
    the corresponding config/backend.yaml keys -- this is what lets a real
    Postgres password be supplied by a hosting platform's env var UI (e.g.
    Render, via render.yaml's fromDatabase) instead of being committed to
    git in a config file. Env vars win; config/backend.yaml is the local-
    dev-friendly fallback."""
    global _active_module, _target_kwarg, _default_target
    backend = (os.environ.get("DB_BACKEND") or cfg.get("db_backend") or "sqlite").lower()

    if backend == "sqlite":
        from src.backend import db as _db
        _active_module = _db
        _target_kwarg = "db_path"
        rel = cfg.get("db_path", str(_db.DEFAULT_DB_PATH))
        _default_target = (Path(project_root) / rel) if project_root is not None else Path(rel)
    elif backend == "postgres":
        from src.backend import db_postgis as _db
        _active_module = _db
        _target_kwarg = "dsn"
        _default_target = os.environ.get("POSTGRES_DSN") or cfg.get("postgres_dsn") or _db.DEFAULT_DSN
    else:
        raise ValueError(f"Unknown db_backend {backend!r} in config/backend.yaml (expected 'sqlite' or 'postgres')")

    logger.info("db_backend configured: %s (target=%s)", backend,
                _default_target if backend == "sqlite" else "<dsn redacted>")
    return _default_target


def _require_configured():
    if _active_module is None:
        raise RuntimeError(
            "db_backend.configure(cfg, project_root) must be called once at startup before using "
            "db_backend (see src/backend/main.py's _cfg = load_config(...) call site)."
        )


def init_db(target: Any = None) -> None:
    _require_configured()
    if target is None:
        target = _default_target
    return _active_module.init_db(**{_target_kwarg: target})


def get_connection(target: Any = None):
    _require_configured()
    if target is None:
        target = _default_target
    return _active_module.get_connection(**{_target_kwarg: target})


def __getattr__(name: str):
    """Any other db.py-compatible function (create_log, update_log_status,
    update_log_counts, get_log, list_logs, insert_detection,
    list_detections, ...) is forwarded straight through -- they all take an
    already-open `conn` as their first arg on both backends (see db.py's
    module docstring), so there's nothing backend-specific left to inject
    once get_connection() above has handed out a connection."""
    _require_configured()
    attr = getattr(_active_module, name, None)
    if attr is None or not callable(attr):
        raise AttributeError(f"db_backend: active module ({_active_module.__name__}) has no callable {name!r}")
    return attr

"""Runtime bootstrap helpers for local source dependencies and repo-relative paths."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
LOCAL_SOURCE_DEPENDENCIES = ("robocasa", "robosuite")


def project_root() -> Path:
    """Return the repository root."""
    return PROJECT_ROOT


def local_source_roots() -> list[Path]:
    """Return existing editable dependency roots vendored into the repo."""
    roots = []
    for name in LOCAL_SOURCE_DEPENDENCIES:
        preferred = THIRD_PARTY_ROOT / name
        legacy = PROJECT_ROOT / name
        if preferred.exists():
            roots.append(preferred)
            continue
        if legacy.exists():
            roots.append(legacy)
    return roots


def ensure_runtime_paths() -> list[Path]:
    """Prepend repo-local dependency roots and the project root to sys.path."""
    ordered_paths = local_source_roots() + [PROJECT_ROOT]
    inserted = []
    for path in reversed(ordered_paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            inserted.append(path)
    return list(reversed(inserted))


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve relative paths against the repository root."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


"""Loader for the loose scripts under ``scripts/``. They are not installed packages, so tests that
exercise them import them by file path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_script(name: str, path: Path) -> ModuleType:
    """Import a loose script by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"could not load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def version_tuple(version: str) -> tuple[int, ...]:
    """Comparable form of a numeric release version."""
    return tuple(int(part) for part in version.split("."))

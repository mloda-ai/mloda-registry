"""Shared loader for scripts/: import a sibling scripts/ module by file path, so an arbitrary cwd
cannot break the import."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_SCRIPTS_DIR = Path(__file__).resolve().parent


def load_sibling(name: str) -> ModuleType:
    """Load scripts/<name>.py by file path."""
    path = _SCRIPTS_DIR / f"{name}.py"
    if not path.exists():
        raise ImportError(f"{path} is missing; a scripts/ verifier derives its checks from {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

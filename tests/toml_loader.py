"""TOML loading shared by the tests."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file."""
    with path.open("rb") as f:
        return tomllib.load(f)


def loads_toml(text: str) -> dict[str, Any]:
    """Parse TOML text."""
    return tomllib.loads(text)

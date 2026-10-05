"""TOML loading shared by the tests, with the Python 3.10 ``tomli`` fallback."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]


def load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file."""
    with path.open("rb") as f:
        return tomllib.load(f)


def loads_toml(text: str) -> dict[str, Any]:
    """Parse TOML text."""
    return tomllib.loads(text)

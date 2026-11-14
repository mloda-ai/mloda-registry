#!/usr/bin/env python3
"""Flag a PyPI mloda release at or above the `core_dependency` cap; stdlib only, runs on a bare runner."""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

SHARED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "shared.toml"
PYPI_URL = "https://pypi.org/pypi/mloda/json"
_UPPER = re.compile(r"<\s*(\d+(?:\.\d+)*)\s*(?:,|$)")


def cap(spec: str) -> tuple[int, ...]:
    """Return the strict upper bound of a requirement specifier."""
    match = _UPPER.search(spec)
    if match is None:
        raise ValueError(f"No strict upper bound '<X.Y.Z' in {spec!r}")
    return tuple(int(n) for n in match.group(1).split("."))


def _release(version: str) -> tuple[int, ...] | None:
    return tuple(int(n) for n in version.split(".")) if re.fullmatch(r"\d+(\.\d+)*", version) else None


def above_cap(spec: str, latest: str) -> bool:
    """True when `latest` is a final release at or above the cap."""
    release = _release(latest)
    if release is None:
        return False
    upper = cap(spec)
    width = max(len(release), len(upper))
    return release + (0,) * (width - len(release)) >= upper + (0,) * (width - len(upper))


def latest_version(payload: dict[str, Any]) -> str:
    info = payload.get("info")
    version = info.get("version") if isinstance(info, dict) else None
    if not isinstance(version, str) or not version:
        raise ValueError("PyPI payload has no info.version")
    return version


def main() -> int:
    spec = tomllib.loads(SHARED_CONFIG.read_text())["defaults"]["core_dependency"]
    with urllib.request.urlopen(PYPI_URL, timeout=30) as response:  # nosec B310
        latest = latest_version(json.load(response))
    if not above_cap(spec, latest):
        print(f"mloda {latest} on PyPI is below the cap of {spec}.")
        return 0
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as handle:
            handle.write(f"version={latest}\n")
    print(f"mloda {latest} on PyPI is at or above the cap of {spec}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

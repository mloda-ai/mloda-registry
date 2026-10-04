#!/usr/bin/env python3
"""Check that every distribution in config/packages.toml is registered on PyPI and owned by us (CI only).

Usage:
    python scripts/check_pypi_names.py
"""

from __future__ import annotations

import json
import re
import runpy
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
EXPECTED_OWNER = "tomkaltofen"

_SCRIPTS_DIR = Path(__file__).resolve().parent

_load_sibling: Callable[[str], ModuleType] = runpy.run_path(str(_SCRIPTS_DIR / "script_loader.py"))["load_sibling"]
published_packages = _load_sibling("published_packages")


class NameReport:
    """Names by problem; a plain class because tests load this script without registering the module."""

    def __init__(self) -> None:
        self.missing: list[str] = []
        self.foreign: list[str] = []
        self.unverifiable: list[str] = []


def canonical_name(name: str) -> str:
    """Return the PEP 503 normalized name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def configured_names() -> list[str]:
    """Return every configured distribution name, in config order."""
    return list(published_packages.load_packages_config())


def fetch_project(name: str) -> tuple[int, dict[str, Any] | None]:
    """Fetch the PyPI JSON for a project; HTTP errors become (code, None)."""
    url = PYPI_JSON_URL.format(name=name)
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # nosec B310
            payload: dict[str, Any] = json.load(response)
            return 200, payload
    except urllib.error.HTTPError as exc:
        return exc.code, None


def _is_owned(payload: dict[str, Any] | None) -> bool:
    roles = ((payload or {}).get("ownership") or {}).get("roles") or []
    return any(role.get("user") == EXPECTED_OWNER for role in roles)


def check_names(
    names: list[str],
    fetch: Callable[[str], tuple[int, dict[str, Any] | None]],
    retries: int = 1,
    sleep: Callable[[float], None] = time.sleep,
) -> NameReport:
    """Classify each name as ok, missing, foreign or unverifiable."""
    report = NameReport()
    for name in names:
        for attempt in range(retries + 1):
            if attempt:
                sleep(2)
            try:
                status, payload = fetch(canonical_name(name))
            except OSError:
                continue
            if status == 404:
                report.missing.append(name)
                break
            if status == 200:
                if not _is_owned(payload):
                    report.foreign.append(name)
                break
        else:
            report.unverifiable.append(name)
    return report


def main() -> int:
    names = configured_names()
    report = check_names(names, fetch_project)
    groups = [
        (
            report.missing,
            "not registered on PyPI (a project with every release deleted also reads as missing); "
            "a maintainer reserves it, see docs/packaging.md",
        ),
        (report.foreign, f"registered but not owned by {EXPECTED_OWNER}"),
        (report.unverifiable, "PyPI did not answer; rerun the job"),
    ]
    failed = False
    for group, hint in groups:
        for name in group:
            failed = True
            print(f"{name}: {hint}", file=sys.stderr)
    if failed:
        return 1
    print(f"OK: all {len(names)} configured names are registered and owned by {EXPECTED_OWNER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

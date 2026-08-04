#!/usr/bin/env python3
"""Install each published package at its released version with its in-repo dependency pinned to the declared floor.

Run: python scripts/verify_floor_installs.py <version>
Exit code: 1 if any floored pair fails to install or import, 0 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess  # nosec
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# The distribution name a PEP 508 dependency string starts with; "{core_dependency}" starts with none.
DEP_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

# Only the lower bound matters: ">=0.4.0" and " >= 0.4.0, <1" both floor at 0.4.0.
FLOOR_RE = re.compile(r">=\s*([^\s,;]+)")


class FloorPair(NamedTuple):
    """One published package with the in-repo dependency floor it declares."""

    package: str
    dependency: str
    floor: str
    module: str


def internal_floor_pairs(packages: dict[str, dict[str, Any]]) -> list[FloorPair]:
    """Pairs of every published package whose dependency names another key of the packages table."""
    pairs: list[FloorPair] = []
    for pkg_name, pkg_config in packages.items():
        if pkg_config.get("published") is not True:
            continue
        for dependency in pkg_config.get("dependencies", []):
            name_match = DEP_NAME_RE.match(dependency)
            if name_match is None or name_match.group(1) not in packages:
                continue
            floor_match = FLOOR_RE.search(dependency)
            if floor_match is None:
                raise ValueError(f"{pkg_name}: in-repo dependency {dependency!r} declares no '>=' floor")
            module = str(pkg_config["path"]).replace("/", ".")
            pairs.append(FloorPair(pkg_name, name_match.group(1), floor_match.group(1), module))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify each package imports against its declared internal floors")
    parser.add_argument("version", nargs="?", default="", help="Released version to install every package at")
    args = parser.parse_args()

    # tox renders an empty argument when MLODA_REGISTRY_VERSION is unset.
    if not args.version.strip():
        parser.error("version must not be empty (is MLODA_REGISTRY_VERSION set?)")

    # The reused published_packages helper resolves the config relative to cwd.
    os.chdir(REPO_ROOT)
    from published_packages import load_packages_config
    from verify_build_floor import venv_python

    pairs = internal_floor_pairs(load_packages_config())

    # An empty set would silently verify nothing.
    if not pairs:
        print("❌ config/packages.toml declares no in-repo dependency floors")
        return 1

    errors: list[str] = []
    for pair in pairs:
        print(f"\nInstalling {pair.package}=={args.version} with {pair.dependency}=={pair.floor}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            venv = Path(tmpdir) / "venv"
            commands = [
                ["uv", "venv", "--python", sys.executable, str(venv)],
                ["uv", "pip", "install", "--python", str(venv_python(venv))]
                + [f"{pair.dependency}=={pair.floor}", f"{pair.package}=={args.version}"],
                [str(venv_python(venv)), "-c", f"import {pair.module}"],
            ]
            for command in commands:
                # cwd is the temp dir, so the checkout cannot shadow the installed packages.
                result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
                if result.returncode != 0:
                    errors.append(
                        f"{pair.package} ({pair.dependency}=={pair.floor}): "
                        f"{' '.join(command)} failed:\n{result.stderr[-500:]}"
                    )
                    break
            else:
                print(f"  ✓ import {pair.module}")

    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"\n✅ every declared internal floor installs and imports at {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

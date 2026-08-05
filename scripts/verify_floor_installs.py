#!/usr/bin/env python3
"""Install each published package with its in-repo dependency pinned to the declared floor, probing its import surface.

The import surface is the dotted package root plus its base module when the checkout ships a base.py.

Run: python scripts/verify_floor_installs.py <version>
Exit code: 1 if any floored pair fails to install or its import surface fails to load, 0 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess  # nosec
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

def _load_sibling(name: str) -> ModuleType:
    """Load another scripts/*.py module by path so we share one import_surface definition."""
    import importlib.util

    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load sibling script {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



# The distribution name a PEP 508 dependency string starts with; "{core_dependency}" starts with none.
DEP_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

# Only the lower bound matters: ">=0.4.0" and " >= 0.4.0, <1" both floor at 0.4.0.
FLOOR_RE = re.compile(r">=\s*([^\s,;]+)")


class FloorPair(NamedTuple):
    """One published package with the in-repo dependency floor it declares."""

    package: str
    dependency: str
    floor: str
    modules: tuple[str, ...]


def _normalize(name: str) -> str:
    """PEP 503 normal form: lowercase, runs of '-', '_', '.' collapsed to '-'."""
    return re.sub(r"[-_.]+", "-", name).lower()




def internal_floor_pairs(packages: dict[str, dict[str, Any]]) -> list[FloorPair]:
    """Pairs of every published package whose dependency names another key of the packages table."""
    canonical = {_normalize(name): name for name in packages}
    import_surface: Callable[[str], tuple[str, ...]] = _load_sibling(
        "verify_published_imports"
    ).import_surface
    pairs: list[FloorPair] = []
    for pkg_name, pkg_config in packages.items():
        if pkg_config.get("published") is not True:
            continue
        for dependency in pkg_config.get("dependencies", []):
            # The '>=' inside a PEP 508 environment marker is not a version floor.
            requirement = dependency.split(";", 1)[0]
            name_match = DEP_NAME_RE.match(requirement)
            if name_match is None:
                continue
            dep_name = canonical.get(_normalize(name_match.group(1)))
            if dep_name is None:
                continue
            floor_match = FLOOR_RE.search(requirement)
            if floor_match is None:
                raise ValueError(f"{pkg_name}: in-repo dependency {dependency!r} declares no '>=' floor")
            modules = import_surface(str(pkg_config["path"]))
            pairs.append(FloorPair(pkg_name, dep_name, floor_match.group(1), modules))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify each package's import surface loads against its declared internal floors"
    )
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
            # One import statement per probe: the package root plus its base module.
            imports = "\n".join(f"import {module}" for module in pair.modules)
            commands = [
                ["uv", "venv", "--python", sys.executable, str(venv)],
                ["uv", "pip", "install", "--python", str(venv_python(venv))]
                + [f"{pair.dependency}=={pair.floor}", f"{pair.package}=={args.version}"],
                [str(venv_python(venv)), "-c", imports],
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
                print(f"  ✓ import surface loads: {', '.join(pair.modules)}")

    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"\n✅ every internal floor installs and its import surface (root plus base module) loads at {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

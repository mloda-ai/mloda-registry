#!/usr/bin/env python3
"""Install each published distribution into its own venv and probe its full import surface.

Every distribution flagged 'published = true' installs and imports independently. A bundle's
probe also covers every package nested under its path, so a payload-less bundle wheel fails.
The per-distribution install-and-probe cycles run concurrently through a bounded thread pool.

Run: python scripts/verify_independent_installs.py <version>
Exit code: 1 if any distribution fails to install or import on its own, 0 otherwise.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import runpy
import subprocess  # nosec
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Installs are network-bound; bounding the pool keeps runner memory and uv cache-lock contention in check.
MAX_WORKERS = 8

# Not a plain import: this script is also loaded by file path in tests, where scripts/ is not on sys.path.
_load_sibling: Callable[[str], ModuleType] = runpy.run_path(str(REPO_ROOT / "scripts" / "script_loader.py"))[
    "load_sibling"
]


def independent_distributions(packages: dict[str, dict[str, Any]]) -> list[str]:
    """Every distribution flagged 'published = true' in config order; each installs and imports on its own."""
    published: Callable[[dict[str, dict[str, Any]]], list[str]] = _load_sibling("published_packages").published_packages
    names: list[str] = published(packages)
    return names


def probe_modules(name: str, packages: dict[str, dict[str, Any]]) -> list[str]:
    """The distribution's own import surface (root plus base/manifest modules) plus every nested
    configured package's, in config order."""
    # The single derivation point for import surfaces lives in verify_published_imports.
    surface: Callable[[str], tuple[str, ...]] = _load_sibling("verify_published_imports").import_surface
    path = str(packages[name]["path"]).rstrip("/")
    modules = list(surface(path))
    prefix = path + "/"
    for pkg_config in packages.values():
        nested = str(pkg_config["path"]).rstrip("/")
        if nested.startswith(prefix):
            modules.extend(surface(nested))
    return modules


def verify_distribution(name: str, version: str, modules: list[str]) -> str | None:
    """Install one distribution into its own venv and probe its import surface; returns an error
    message, or None on success. Runs the whole cycle in its own temporary directory; prints nothing,
    so callers running several of these concurrently control all output themselves."""
    # Resolved by file path, not a plain import: callers may invoke this without scripts/ on sys.path.
    venv_python: Callable[[Path], Path] = _load_sibling("verify_build_floor").venv_python
    with tempfile.TemporaryDirectory() as tmpdir:
        venv = Path(tmpdir) / "venv"
        # One import statement per probed module: the distribution's own surface plus its nested ones.
        imports = "\n".join(f"import {module}" for module in modules)
        commands = [
            ["uv", "venv", "--python", sys.executable, str(venv)],
            ["uv", "pip", "install", "--python", str(venv_python(venv)), f"{name}=={version}"],
            [str(venv_python(venv)), "-c", imports],
        ]
        for command in commands:
            # cwd is the temp dir, so the checkout cannot shadow the installed packages.
            result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
            if result.returncode != 0:
                return f"{name}: {' '.join(command)} failed:\n{result.stderr[-500:]}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Install each published distribution into its own venv")
    parser.add_argument("version", nargs="?", default="", help="Released version to install every distribution at")
    args = parser.parse_args()

    # tox renders an empty argument when MLODA_REGISTRY_VERSION is unset.
    if not args.version.strip():
        parser.error("version must not be empty (is MLODA_REGISTRY_VERSION set?)")

    # The reused published_packages helper resolves the config relative to cwd.
    os.chdir(REPO_ROOT)
    from published_packages import load_packages_config

    packages = load_packages_config()
    names = independent_distributions(packages)

    # An empty set would silently verify nothing.
    if not names:
        print("❌ config/packages.toml declares no published packages")
        return 1

    modules_by_name = {name: probe_modules(name, packages) for name in names}
    workers = min(len(names), MAX_WORKERS)
    print(f"\nInstalling {len(names)} distributions at {args.version}, {workers} at a time...")

    def _verify(name: str) -> str | None:
        return verify_distribution(name, args.version, modules_by_name[name])

    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # map() preserves config order, so per-distribution output stays deterministic.
        for name, error in zip(names, executor.map(_verify, names)):
            if error is None:
                print(f"  ✓ {name}: import surface loads ({len(modules_by_name[name])} modules)")
            else:
                print(f"  ✗ {name}: failed")
                errors.append(error)

    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"\n✅ every published distribution installs and its import surface loads independently at {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

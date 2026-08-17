#!/usr/bin/env python3
"""Smoke-import every configured package inside the venv holding the released set.

Every configured package ships in some wheel, so every import surface (the dotted root plus its
base module when the checkout ships a base.py, and its manifest module when the checkout ships a
manifest.py) gets probed.

Run: python scripts/verify_published_imports.py <venv-python>
Exit code: 1 if any module fails to import, 0 otherwise.
"""

from __future__ import annotations

import argparse
import os
import subprocess  # nosec
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def import_surface(path: str) -> tuple[str, ...]:
    """The dotted package root, plus its base module when <path>/base.py exists in the checkout, and its
    manifest module when <path>/manifest.py exists in the checkout."""
    root = path.replace("/", ".")
    # Most leaf __init__.py files are empty and the cross-package imports live in the leaf's base
    # module, so importing only the root is vacuous. Resolve against the checkout, never the cwd.
    modules = [root]
    if (REPO_ROOT / path / "base.py").exists():
        modules.append(f"{root}.base")
    # manifest.py is mloda's real plugin entry-point discovery module; a package can ship one without
    # a base.py, so this check is independent of the one above.
    if (REPO_ROOT / path / "manifest.py").exists():
        modules.append(f"{root}.manifest")
    return tuple(modules)


def import_modules(packages: dict[str, dict[str, Any]]) -> list[str]:
    """The import surface of every configured package, in config order (roots plus base/manifest modules)."""
    modules: list[str] = []
    for pkg_config in packages.values():
        modules.extend(import_surface(str(pkg_config["path"])))
    return modules


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-import every configured package inside a venv")
    parser.add_argument("venv_python", nargs="?", default="", help="Interpreter of the venv holding the released set")
    args = parser.parse_args()

    # tox renders an empty argument when the interpolated value is unset.
    if not args.venv_python.strip():
        parser.error("venv_python must not be empty")

    # Anchor to the invocation cwd before the chdir below; abspath keeps the venv symlink unresolved.
    interpreter = os.path.abspath(args.venv_python)

    # The reused published_packages helper resolves the config relative to cwd.
    os.chdir(REPO_ROOT)
    from published_packages import load_packages_config

    modules = import_modules(load_packages_config())

    # An empty set would silently verify nothing.
    if not modules:
        print("❌ config/packages.toml declares no packages")
        return 1

    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for module in modules:
            command = [interpreter, "-c", f"import {module}"]
            # cwd is the temp dir, so the checkout cannot shadow the installed packages.
            result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
            if result.returncode != 0:
                errors.append(f"import {module} failed:\n{result.stderr[-500:]}")
                continue
            print(f"  ✓ {module} OK")

    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"\n✅ the full import surface ({len(modules)} modules) imports from the installed released set")
    return 0


if __name__ == "__main__":
    sys.exit(main())

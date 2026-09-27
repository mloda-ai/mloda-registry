#!/usr/bin/env python3
"""Print the distributions published to PyPI, read from the ``published`` flag in config/packages.toml.

Usage:
    python scripts/published_packages.py                # one distribution name per line
    python scripts/published_packages.py --pin 0.4.0    # each name pinned to a version
    python scripts/published_packages.py --exclude-newer-exempt  # uv flags exempting the set from exclude-newer
    python scripts/published_packages.py --wheels dist  # one matched wheel path per line
"""

from __future__ import annotations

import argparse
import re
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

CONFIG_DIR = Path("config")
PACKAGES_CONFIG = CONFIG_DIR / "packages.toml"

_SCRIPTS_DIR = Path(__file__).resolve().parent

# Not a plain import: this script is also loaded by file path in tests, where scripts/ is not on sys.path.
_load_sibling: Callable[[str], ModuleType] = runpy.run_path(str(_SCRIPTS_DIR / "script_loader.py"))["load_sibling"]
gen = _load_sibling("generate_pyproject")


def load_packages_config() -> dict[str, dict[str, Any]]:
    """Return the raw [packages] table of config/packages.toml, in config order."""
    with open(PACKAGES_CONFIG, "rb") as f:
        data = tomllib.load(f)
    packages: dict[str, dict[str, Any]] = data.get("packages", {})
    return packages


def _dependency_first_siblings(name: str, pkg_config: dict[str, Any], packages: dict[str, dict[str, Any]]) -> list[str]:
    """Published siblings ``name`` must see earlier in the published order: every sibling named in its own
    ``dependencies``, plus every sibling named in a non-dev extra entry that carries the {version} placeholder."""
    raw_deps: list[str] = list(pkg_config.get("dependencies", []))
    for extra_name, deps in pkg_config.get("optional_dependencies", {}).items():
        if extra_name == "dev":
            continue
        raw_deps.extend(dep for dep in deps if "{version}" in dep)
    siblings: list[str] = gen.sibling_dependency_names(raw_deps, packages)
    return siblings


def published_packages(packages: dict[str, dict[str, Any]]) -> list[str]:
    """Return the distributions flagged ``published = true``, in config order.

    Raises ``ValueError``, naming both packages, when a published package names a published sibling
    (in ``dependencies``, or in a non-dev extra entry carrying the {version} placeholder) that does not
    appear earlier in that order.
    """
    names: list[str] = []
    for name, pkg_config in packages.items():
        flag = pkg_config.get("published")
        if flag is None:
            continue
        if not isinstance(flag, bool):
            raise ValueError(f"{name}: 'published' must be a boolean in {PACKAGES_CONFIG}, got {flag!r}")
        if flag is True:
            names.append(name)

    published = set(names)
    seen: set[str] = set()
    for name in names:
        siblings = _dependency_first_siblings(name, packages[name], packages)
        for sibling in siblings:
            if sibling in published and sibling not in seen:
                raise ValueError(
                    f"{name} names published sibling {sibling} which does not appear earlier in the "
                    f"published order; {PACKAGES_CONFIG} must declare {sibling} before {name}"
                )
        seen.add(name)

    return names


def escape_distribution_name(name: str) -> str:
    """Escape a distribution name the way a wheel filename carries it (PEP 427/503)."""
    return re.sub(r"[-_.]+", "_", name.lower())


def find_wheels(out_dir: Path, pkg_name: str) -> list[Path]:
    """Wheels in out_dir whose distribution segment is exactly pkg_name, so prefix siblings never match."""
    escaped = escape_distribution_name(pkg_name)
    return sorted((path for path in out_dir.glob("*.whl") if path.name.split("-")[0] == escaped), key=lambda p: p.name)


def published_wheels(packages: dict[str, dict[str, Any]], out_dir: Path) -> list[Path]:
    """Exactly one wheel per published distribution, in published order, matched with find_wheels.

    Raises ``ValueError``, naming the package, when no wheel matches or when more than one matches.
    """
    wheels: list[Path] = []
    for name in published_packages(packages):
        matches = find_wheels(out_dir, name)
        if len(matches) != 1:
            raise ValueError(f"{name}: expected exactly one wheel in {out_dir}, found {[m.name for m in matches]}")
        wheels.append(matches[0])
    return wheels


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the distributions published to PyPI")
    parser.add_argument("--pin", metavar="VERSION", help="Append '==VERSION' to every distribution name")
    parser.add_argument(
        "--exclude-newer-exempt",
        action="store_true",
        help="Print '--exclude-newer-package=NAME=false' for every distribution instead of its name",
    )
    parser.add_argument("--wheels", metavar="DIR", help="Print one matched wheel path per line instead of names")
    args = parser.parse_args()

    if args.pin is not None and args.wheels is not None:
        parser.error("--pin and --wheels are mutually exclusive")

    # tox renders '--pin ' when MLODA_REGISTRY_VERSION is unset.
    if args.pin is not None and not args.pin.strip():
        parser.error("--pin needs a version, got an empty value (is MLODA_REGISTRY_VERSION set?)")

    packages = load_packages_config()

    if args.wheels is not None:
        try:
            wheels = published_wheels(packages, Path(args.wheels))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        for wheel in wheels:
            print(wheel)
        return 0

    names = published_packages(packages)

    # An empty set would silently publish, verify or scan nothing.
    if not names:
        print(f"{PACKAGES_CONFIG}: no package is flagged 'published = true'", file=sys.stderr)
        return 1

    for name in names:
        if args.exclude_newer_exempt:
            print(f"--exclude-newer-package={name}=false")
        else:
            print(f"{name}=={args.pin}" if args.pin else name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

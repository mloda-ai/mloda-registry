#!/usr/bin/env python3
"""Print the distributions published to PyPI, read from config/packages.toml.

The ``published = true`` flag in config/packages.toml is the single source of the
released set. This script is how the release workflow and the tox envs that install
from PyPI read it, so none of them re-types the set.

Usage:
    python scripts/published_packages.py                # one distribution name per line
    python scripts/published_packages.py --pin 0.4.0    # each name pinned to a version
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

CONFIG_DIR = Path("config")
PACKAGES_CONFIG = CONFIG_DIR / "packages.toml"


def load_packages_config() -> dict[str, dict[str, Any]]:
    """Return the raw [packages] table of config/packages.toml, in config order."""
    with open(PACKAGES_CONFIG, "rb") as f:
        data = tomllib.load(f)
    packages: dict[str, dict[str, Any]] = data.get("packages", {})
    return packages


def published_packages(packages: dict[str, dict[str, Any]]) -> list[str]:
    """Return the distributions flagged ``published = true``, in config order."""
    return [name for name, pkg_config in packages.items() if pkg_config.get("published")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the distributions published to PyPI")
    parser.add_argument("--pin", metavar="VERSION", help="Append '==VERSION' to every distribution name")
    args = parser.parse_args()

    names = published_packages(load_packages_config())

    # An empty set would silently publish, verify or scan nothing at all.
    if not names:
        print(f"{PACKAGES_CONFIG}: no package is flagged 'published = true'", file=sys.stderr)
        return 1

    for name in names:
        print(f"{name}=={args.pin}" if args.pin else name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

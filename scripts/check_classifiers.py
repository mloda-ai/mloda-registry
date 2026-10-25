#!/usr/bin/env python3
"""Check every generated package classifier against trove-classifiers (tox -e check-classifiers)."""

from __future__ import annotations

import runpy
import sys
import tomllib
from collections.abc import Callable, Collection
from pathlib import Path
from types import ModuleType
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent

# Not a plain import: this script is also loaded by file path in tests, where scripts/ is not on sys.path.
_load_sibling: Callable[[str], ModuleType] = runpy.run_path(str(_SCRIPTS_DIR / "script_loader.py"))["load_sibling"]
gen = _load_sibling("generate_pyproject")


def generated_classifiers(shared: dict[str, Any], all_packages: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Return the generated [project].classifiers of every configured package."""
    result: dict[str, list[str]] = {}
    for name, cfg in all_packages.items():
        data = tomllib.loads(gen.generate_pyproject(name, cfg, shared, all_packages))
        result[name] = list(data["project"].get("classifiers", []))
    return result


def unknown_classifiers(classifiers_by_package: dict[str, list[str]], known: Collection[str]) -> list[str]:
    """Return one error line per classifier missing from ``known``."""
    return [
        f"{pkg}: unknown classifier {value!r}"
        for pkg, values in classifiers_by_package.items()
        for value in values
        if value not in known
    ]


def main() -> int:
    # Absent from the gate env; installed only by the check-classifiers tox env.
    import trove_classifiers

    shared, packages_config = gen.load_configs()
    errors = unknown_classifiers(
        generated_classifiers(shared, packages_config["packages"]), trove_classifiers.classifiers
    )
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print("All generated classifiers are known trove classifiers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

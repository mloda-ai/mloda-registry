"""Helpers for building entry-point manifests resilient to missing optional backends.

A data_operations plugin package ships one concrete plugin class per compute
framework, and each backend module top-imports its framework (pandas, polars,
duckdb, pyarrow). mloda's entry-point loader skips a whole entry point if
importing its manifest raises ModuleNotFoundError, so a manifest that eagerly
imports every backend becomes undiscoverable unless every optional framework is
installed. ``load_plugin_classes`` imports each backend individually and drops
only the backends whose optional framework is absent, while still raising on any
other import error (so typos and real breakage stay loud). See issue #271.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

# Optional compute-framework roots used by data_operations backends. A missing
# import whose root is one of these means "framework not installed" -> skip that
# backend only. Any other ModuleNotFoundError is a real error and re-raised.
_OPTIONAL_BACKENDS = frozenset({"pandas", "polars", "duckdb", "pyarrow"})


def load_plugin_classes(package: str, specs: Iterable[tuple[str, str]]) -> list[type[Any]]:
    """Import ``(submodule, class_name)`` pairs under ``package``.

    Skips a backend whose optional framework dependency is not installed;
    re-raises every other import error. Order follows ``specs``.
    """
    classes: list[type[Any]] = []
    for submodule, class_name in specs:
        try:
            module = importlib.import_module(f"{package}.{submodule}")
        except ModuleNotFoundError as exc:
            root = (exc.name or "").split(".")[0]
            if root in _OPTIONAL_BACKENDS:
                continue
            raise
        classes.append(getattr(module, class_name))
    return classes

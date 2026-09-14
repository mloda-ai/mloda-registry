"""Helpers for building entry-point manifests resilient to missing optional backends.

A data_operations plugin package ships one concrete plugin class per compute
framework, and each backend module top-imports its framework (pandas, polars,
duckdb, pyarrow) plus any transitive dependency it imports directly (numpy,
which pandas_binning.py imports before pandas). mloda's entry-point loader
skips a whole entry point if importing its manifest raises ModuleNotFoundError,
so a manifest that eagerly imports every backend becomes undiscoverable unless
every optional framework is installed. ``load_plugin_classes`` imports each
backend individually and drops only the backends whose optional framework is
absent, while still raising on any other import error (so typos and real
breakage stay loud). See issue #271.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

# Optional third-party roots a data_operations backend may top-import (a framework
# or one of its transitive deps, e.g. numpy via pandas). A missing import whose
# root is one of these means "not installed" -> skip that backend only. Any
# other ModuleNotFoundError is a real error and re-raised.
_OPTIONAL_BACKENDS = frozenset({"pandas", "polars", "duckdb", "pyarrow", "numpy"})


def _traceback_blames_root(exc: ImportError, root: str) -> bool:
    """True if the innermost (deepest) frame of exc's traceback, i.e. where the failure actually
    occurred, belongs to root or a submodule of it. Reimplements core's private
    PluginLoader._traceback_blames_root locally, since that helper is core-private.
    """
    tb = exc.__traceback__
    if tb is None:
        return False
    while tb.tb_next is not None:
        tb = tb.tb_next
    module_name = tb.tb_frame.f_globals.get("__name__")
    return isinstance(module_name, str) and (module_name == root or module_name.startswith(f"{root}."))


def load_plugin_classes(package: str, specs: Iterable[tuple[str, str]]) -> list[type[Any]]:
    """Import ``(submodule, class_name)`` pairs under ``package``.

    Skips a backend whose optional framework dependency is not installed, attributed either by
    ``exc.name``'s root or, if that isn't set or doesn't match, by traceback-frame blame; re-raises
    every other import error. Order follows ``specs``.
    """
    classes: list[type[Any]] = []
    for submodule, class_name in specs:
        try:
            module = importlib.import_module(f"{package}.{submodule}")
        except ImportError as exc:
            root = (exc.name or "").split(".")[0]
            blamed = root in _OPTIONAL_BACKENDS or any(
                _traceback_blames_root(exc, candidate) for candidate in _OPTIONAL_BACKENDS
            )
            if not blamed:
                raise
            continue
        classes.append(getattr(module, class_name))
    return classes

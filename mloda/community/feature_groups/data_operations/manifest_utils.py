"""Helpers for building entry-point manifests resilient to missing optional backends.

A data_operations plugin package ships one concrete plugin class per compute
framework, and each backend module top-imports its framework (pandas, polars,
duckdb, pyarrow) plus any transitive dependency it imports directly (numpy,
which pandas_binning.py imports before pandas). mloda's entry-point loader
skips a whole entry point if importing its manifest raises ImportError,
so a manifest that eagerly imports every backend becomes undiscoverable unless
every optional framework is installed. ``load_plugin_classes`` imports each
backend individually and drops only the backends whose optional framework is
absent, while still raising on any other import error (so typos and real
breakage stay loud). See issue #271.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

# Optional third-party roots a data_operations backend may top-import (a framework
# or one of its transitive deps, e.g. numpy via pandas). A missing import whose
# root is one of these means "not installed" -> skip that backend only. Any
# other ImportError is a real error and re-raised.
_OPTIONAL_BACKENDS = frozenset({"pandas", "polars", "duckdb", "pyarrow", "numpy"})


def _innermost_traceback_module(exc: ImportError) -> str | None:
    """Module name of the innermost (deepest) frame of exc's traceback, i.e. where the failure
    actually occurred, or None if exc has no traceback.
    """
    tb = exc.__traceback__
    if tb is None:
        return None
    while tb.tb_next is not None:
        tb = tb.tb_next
    module_name = tb.tb_frame.f_globals.get("__name__")
    return module_name if isinstance(module_name, str) else None


def _blamed_optional_backend(exc: ImportError) -> str | None:
    """The ``_OPTIONAL_BACKENDS`` member blamed for exc (by ``exc.name``'s root, else the innermost
    traceback frame's module), or None. Core's equivalent is private, hence this local copy."""
    root = (exc.name or "").split(".")[0]
    if root in _OPTIONAL_BACKENDS:
        return root
    innermost = _innermost_traceback_module(exc)
    if innermost is not None:
        for candidate in _OPTIONAL_BACKENDS:
            if innermost == candidate or innermost.startswith(f"{candidate}."):
                return candidate
    return None


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
            blamed = _blamed_optional_backend(exc)
            if blamed is None:
                raise
            logger.warning(
                "Skipping backend %s.%s: missing optional dependency %s",
                package,
                submodule,
                blamed,
            )
            continue
        classes.append(getattr(module, class_name))
    return classes

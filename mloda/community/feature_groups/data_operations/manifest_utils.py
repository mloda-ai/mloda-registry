"""Helpers for building entry-point manifests resilient to missing optional backends.

A data_operations plugin package ships one concrete plugin class per compute
framework, and each backend module top-imports its framework (pandas, polars,
duckdb, pyarrow) plus any transitive dependency it imports directly (numpy,
which pandas_binning.py imports before pandas). mloda's entry-point loader
skips a whole entry point if importing its manifest raises ImportError,
so a manifest that eagerly imports every backend becomes undiscoverable unless
every optional framework is installed. ``load_plugin_classes`` imports each
backend individually and skips only the ones whose optional framework is
absent (DEBUG) or installed but broken (WARNING), while still raising on any
other import error.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

# Optional third-party roots a data_operations backend may top-import (a framework
# or one of its transitive deps, e.g. numpy via pandas). A missing import attributed
# to one of these means the backend is skipped, not fatal; any other ImportError is
# a real error and re-raised.
_OPTIONAL_BACKENDS = frozenset({"pandas", "polars", "duckdb", "pyarrow", "numpy"})

# Shared message for both skip levels; the blamed root must stay the last log arg.
_SKIP_MESSAGE = "Skipping backend %s.%s: missing optional dependency %s"


def _innermost_traceback_module(exc: ImportError) -> str | None:
    """Module name of the innermost (deepest) frame of exc's traceback, or None if exc has no traceback.
    Mirrors core's private ``_traceback_blames_root`` (mloda.core.abstract_plugins.plugin_loader.plugin_loader);
    keep in sync.
    """
    tb = exc.__traceback__
    if tb is None:
        return None
    while tb.tb_next is not None:
        tb = tb.tb_next
    module_name = tb.tb_frame.f_globals.get("__name__")
    return module_name if isinstance(module_name, str) else None


def _traceback_blamed_backend(exc: ImportError) -> str | None:
    """The ``_OPTIONAL_BACKENDS`` member blamed for exc by the innermost traceback frame, or None."""
    innermost = _innermost_traceback_module(exc)
    if innermost is None:
        return None
    for candidate in _OPTIONAL_BACKENDS:
        if innermost == candidate or innermost.startswith(f"{candidate}."):
            return candidate
    return None


def _classify_import_error(exc: ImportError) -> tuple[int, str] | None:
    """Log level and blamed ``_OPTIONAL_BACKENDS`` member for exc, or None if unattributable (caller re-raises).
    DEBUG only when ``exc.name`` exactly matches an optional backend (framework not installed); WARNING for
    any other attribution, by ``exc.name``'s root or traceback-frame blame (framework installed but broken).
    """
    if isinstance(exc, ModuleNotFoundError) and exc.name in _OPTIONAL_BACKENDS:
        return logging.DEBUG, exc.name
    root = (exc.name or "").split(".")[0]
    if root in _OPTIONAL_BACKENDS:
        return logging.WARNING, root
    blamed = _traceback_blamed_backend(exc)
    if blamed is not None:
        return logging.WARNING, blamed
    return None


def load_plugin_classes(package: str, specs: Iterable[tuple[str, str]]) -> list[type[Any]]:
    """Import ``(submodule, class_name)`` pairs under ``package``. Skips a backend whose optional
    framework is absent (DEBUG) or installed but broken (WARNING); re-raises every other import
    error. Order follows ``specs``.
    """
    classes: list[type[Any]] = []
    for submodule, class_name in specs:
        try:
            module = importlib.import_module(f"{package}.{submodule}")
        except ImportError as exc:
            classification = _classify_import_error(exc)
            if classification is None:
                raise
            level, blamed = classification
            logger.log(level, _SKIP_MESSAGE, package, submodule, blamed)
            continue
        classes.append(getattr(module, class_name))
    return classes

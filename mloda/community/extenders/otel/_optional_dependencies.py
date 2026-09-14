"""Marker consumed by PluginLoader's optional-dependency lookup: this module itself has no import of
opentelemetry, but loading it via entry_point.load() still runs the parent package's own __init__.py
first (Python always imports a submodule's parent package), which does import opentelemetry --
that's why __init__.py's own try/except ImportError guard still matters here.

Must stay import-free of opentelemetry: PluginLoader reads it only after manifest.py's own import
has already failed, so a marker living inside manifest.py could never be read.
"""

from __future__ import annotations

OPTIONAL_DEPENDENCIES: tuple[str, ...] = ("opentelemetry",)

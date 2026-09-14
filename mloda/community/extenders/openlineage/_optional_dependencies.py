"""Dependency-free marker consumed by PluginLoader's optional-dependency lookup.

Must stay import-free of openlineage: PluginLoader reads it only after manifest.py's own import has
already failed, so a marker living inside manifest.py could never be read.
"""

from __future__ import annotations

OPTIONAL_DEPENDENCIES: tuple[str, ...] = ("openlineage",)

"""Marker PluginLoader reads for this package's optional import roots. Lives beside manifest.py rather
than in it because the loader reads it only after manifest.py's own import has failed.
"""

from __future__ import annotations

OPTIONAL_DEPENDENCIES: tuple[str, ...] = ("openlineage",)

"""Dependency-free marker consumed by PluginLoader's optional-dependency lookup.

Must stay import-free of openlineage: PluginLoader reads this only inside the except-ImportError
handler for manifest.py's own entry point, i.e. after manifest.py's import already failed. A marker
living inside manifest.py would re-trigger that same failure instead of being readable.
"""

from __future__ import annotations

OPTIONAL_DEPENDENCIES: tuple[str, ...] = ("openlineage",)

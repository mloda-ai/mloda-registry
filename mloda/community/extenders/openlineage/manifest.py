"""Entry-point manifest for mloda-community-openlineage."""

from __future__ import annotations

from mloda.steward import Extender

from .openlineage_extender import OpenLineageExtender

# No try/except: PluginLoader.load_entry_points() (guided by the mloda.optional_dependencies
# marker in _optional_dependencies.py) is the sole guard tolerating a missing openlineage-python.
EXTENDERS: list[type[Extender]] = [OpenLineageExtender]

"""Entry-point manifest for mloda-community-otel."""

from __future__ import annotations

from mloda.steward import Extender

from .otel_extender import OtelExtender
from .otel_metrics_extender import OtelMetricsExtender

# No try/except: PluginLoader.load_entry_points() (guided by the mloda.optional_dependencies
# marker in _optional_dependencies.py) is the sole guard tolerating a missing opentelemetry-api.
EXTENDERS: list[type[Extender]] = [OtelExtender, OtelMetricsExtender]

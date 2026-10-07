"""mloda-community-otel: OpenTelemetry spans and metrics for mloda pipelines."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mloda.community.extenders.otel.otel_extender import OtelExtender
    from mloda.community.extenders.otel.otel_metrics_extender import OtelMetricsExtender


def _api_module_missing() -> bool:
    # opentelemetry.metrics ships in the same opentelemetry-api distribution as opentelemetry.trace.
    # Probes the module the extender imports, not the namespace-package root; ValueError means a stub with no __spec__.
    try:
        return importlib.util.find_spec("opentelemetry.trace") is None
    except (ImportError, ValueError):
        return True


__all__ = ["OtelExtender", "OtelMetricsExtender"]
# mypy only reads a plain list/tuple literal, so the extra is kept above and cleared here at runtime.
if not TYPE_CHECKING and _api_module_missing():
    __all__ = []


# Lazy on purpose: this extender ships in its own distribution, which the mloda-community[otel]
# extra pulls in, and the mloda.optional_dependencies marker (see _optional_dependencies.py) is loaded
# through this package, so nothing here may import opentelemetry at import time.
def __getattr__(name: str) -> Any:
    if name == "OtelExtender":
        from mloda.community.extenders.otel.otel_extender import OtelExtender

        return OtelExtender
    if name == "OtelMetricsExtender":
        from mloda.community.extenders.otel.otel_metrics_extender import OtelMetricsExtender

        return OtelMetricsExtender
    raise AttributeError(name)

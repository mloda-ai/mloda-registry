"""mloda-community-otel: OpenTelemetry spans for mloda pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mloda.community.extenders.otel.otel_extender import OtelExtender

__all__ = ["OtelExtender"]


# Lazy on purpose: the mloda-community bundle ships this extender behind the mloda-community[otel]
# extra, and the mloda.optional_dependencies marker (see _optional_dependencies.py) is loaded through this
# package, so nothing here may import opentelemetry at import time.
def __getattr__(name: str) -> Any:
    if name == "OtelExtender":
        from mloda.community.extenders.otel.otel_extender import OtelExtender

        return OtelExtender
    raise AttributeError(name)

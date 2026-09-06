"""Entry-point manifest for mloda-community-otel."""

from __future__ import annotations

import logging

from mloda.steward import Extender

EXTENDERS: list[type[Extender]]

_logger = logging.getLogger(__name__)

# The mloda-community bundle ships this extender behind the mloda-community[otel]
# extra; core's loader would otherwise raise on the missing opentelemetry-api dependency.
try:
    from .otel_extender import OtelExtender
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] != "opentelemetry":
        raise
    EXTENDERS = []
    _logger.info(
        "OtelExtender is unavailable: install 'opentelemetry-api' via the 'mloda-community[otel]' extra to enable it."
    )
else:
    EXTENDERS = [OtelExtender]

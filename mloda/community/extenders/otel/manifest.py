"""Entry-point manifest for mloda-community-otel."""

from __future__ import annotations

import logging

from mloda.steward import Extender

EXTENDERS: list[type[Extender]]

_logger = logging.getLogger(__name__)

_ROOT = "opentelemetry"
_DISTRIBUTION = "opentelemetry-api"
_EXTRA = "mloda-community[otel]"
_SUBJECT = "OtelExtender"

# The mloda-community bundle ships this extender behind the mloda-community[otel]
# extra; core's loader would otherwise raise on the missing opentelemetry-api dependency.
try:
    from .otel_extender import OtelExtender
except ImportError as exc:
    from ._optional_dependency import log_unavailable, reraise_unless_optional

    reraise_unless_optional(exc, _ROOT)
    log_unavailable(_logger, exc, _ROOT, _DISTRIBUTION, _EXTRA, _SUBJECT)
    EXTENDERS = []
else:
    EXTENDERS = [OtelExtender]

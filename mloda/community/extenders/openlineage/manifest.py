"""Entry-point manifest for mloda-community-openlineage."""

from __future__ import annotations

import logging

from mloda.steward import Extender

EXTENDERS: list[type[Extender]]

_logger = logging.getLogger(__name__)

_ROOT = "openlineage"
_DISTRIBUTION = "openlineage-python"
_EXTRA = "mloda-community[openlineage]"
_SUBJECT = "OpenLineageExtender"

# The mloda-community bundle ships this extender behind the mloda-community[openlineage]
# extra; core's loader would otherwise raise on the missing openlineage-python dependency.
try:
    from .openlineage_extender import OpenLineageExtender
except ImportError as exc:
    from ._optional_dependency import log_unavailable, reraise_unless_optional

    reraise_unless_optional(exc, _ROOT)
    log_unavailable(_logger, exc, _ROOT, _DISTRIBUTION, _EXTRA, _SUBJECT)
    EXTENDERS = []
else:
    EXTENDERS = [OpenLineageExtender]

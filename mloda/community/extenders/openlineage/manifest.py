"""Entry-point manifest for mloda-community-openlineage."""

from __future__ import annotations

import logging

from mloda.steward import Extender

EXTENDERS: list[type[Extender]]

_logger = logging.getLogger(__name__)

# The mloda-community bundle ships this extender behind the mloda-community[openlineage]
# extra; core's loader would otherwise raise on the missing openlineage-python dependency.
try:
    from .openlineage_extender import OpenLineageExtender
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] != "openlineage":
        raise
    EXTENDERS = []
    _logger.info("OpenLineageExtender unavailable: install 'openlineage-python' via 'mloda-community[openlineage]'.")
else:
    EXTENDERS = [OpenLineageExtender]

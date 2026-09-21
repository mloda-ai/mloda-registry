"""Entry-point manifest for mloda-enterprise-lineage."""

from __future__ import annotations

import logging

from mloda.steward import Extender

logger = logging.getLogger(__name__)

_MISSING_ROOTS = ("openlineage", "mloda.community.extenders.openlineage")

EXTENDERS: list[type[Extender]]

# Guarded here, not by a mloda.optional_dependencies marker: PluginLoader re-raises a missing module whose root
# equals the entry point's own root (mloda), so an unguarded import would break every enterprise install
# without the openlineage extra.
try:
    from .lineage_extender import LineageFacetsExtender
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if not any(missing == root or missing.startswith(f"{root}.") for root in _MISSING_ROOTS):
        raise
    logger.debug("mloda-enterprise-lineage is inactive: %s is not installed (mloda-enterprise[openlineage])", missing)
    EXTENDERS = []
else:
    EXTENDERS = [LineageFacetsExtender]

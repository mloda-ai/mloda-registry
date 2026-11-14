"""Entry-point manifest for mloda-enterprise-lineage."""

from __future__ import annotations

import logging

from mloda.steward import Extender

from .community_version import mismatch_message

logger = logging.getLogger(__name__)

# attr is listed because lineage_extender.py imports it and only openlineage-python installs it.
_MISSING_ROOTS = ("openlineage", "attr", "mloda.community.extenders.openlineage")

EXTENDERS: list[type[Extender]]

_mismatch = mismatch_message()

# Guarded here, not by a mloda.optional_dependencies marker: the mloda-enterprise bundle ships this leaf, so a
# bare install lacks the openlineage extra by design, and a marker skip would log a WARNING in every process.
# The guard keeps it at DEBUG with the install hint, like the anonymizer manifest.
if _mismatch is not None:
    logger.warning("%s", _mismatch)
    EXTENDERS = []
else:
    try:
        from .lineage_extender import LineageFacetsExtender
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if not any(missing == root or missing.startswith(f"{root}.") for root in _MISSING_ROOTS):
            raise
        logger.debug(
            "mloda-enterprise-lineage is inactive: %s is not installed (mloda-enterprise[openlineage])", missing
        )
        EXTENDERS = []
    else:
        EXTENDERS = [LineageFacetsExtender]

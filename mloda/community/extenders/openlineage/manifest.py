"""Entry-point manifest for mloda-community-openlineage."""

from __future__ import annotations

from mloda.steward import Extender

EXTENDERS: list[type[Extender]]

# The mloda-community bundle ships this extender behind the mloda-community[openlineage]
# extra; core's loader would otherwise raise on the missing openlineage-python dependency.
try:
    from .openlineage_extender import OpenLineageExtender
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] != "openlineage":
        raise
    EXTENDERS = []
else:
    EXTENDERS = [OpenLineageExtender]

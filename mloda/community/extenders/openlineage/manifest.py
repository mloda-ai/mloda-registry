"""Entry-point manifest for mloda-community-openlineage."""

from __future__ import annotations

from mloda.steward import Extender

from .openlineage_extender import OpenLineageExtender

EXTENDERS: list[type[Extender]] = [
    OpenLineageExtender,
]

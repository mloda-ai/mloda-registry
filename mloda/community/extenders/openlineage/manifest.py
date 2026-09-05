"""Entry-point manifest for mloda-community-openlineage.

Lists the concrete Extender classes that mloda discovers via the
``mloda.extenders`` entry point.
"""

from __future__ import annotations

from mloda.steward import Extender

from .openlineage_extender import OpenLineageExtender

EXTENDERS: list[type[Extender]] = [
    OpenLineageExtender,
]

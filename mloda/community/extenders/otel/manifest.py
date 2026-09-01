"""Entry-point manifest for mloda-community-otel.

Lists the concrete Extender classes that mloda discovers via the
``mloda.extenders`` entry point.
"""

from __future__ import annotations

from mloda.steward import Extender

from .otel_extender import OtelExtender

EXTENDERS: list[type[Extender]] = [
    OtelExtender,
]

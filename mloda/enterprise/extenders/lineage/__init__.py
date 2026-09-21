"""mloda-enterprise-lineage: OpenLineage facets on top of the community emitter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mloda.enterprise.extenders.lineage.lineage_extender import LineageFacetsExtender

__all__ = ["LineageFacetsExtender"]


# Lazy on purpose: the community emitter needs openlineage-python, which mloda-enterprise[openlineage] provides,
# so nothing here may import it at import time.
def __getattr__(name: str) -> Any:
    if name == "LineageFacetsExtender":
        from mloda.enterprise.extenders.lineage.lineage_extender import LineageFacetsExtender

        return LineageFacetsExtender
    raise AttributeError(name)

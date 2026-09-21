"""LineageFacetsExtender: the community OpenLineage emitter, used instead of it for enterprise lineage facets."""

from __future__ import annotations

from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender

_PRODUCER = "https://github.com/mloda-ai/mloda-registry/tree/main/mloda/enterprise/extenders/lineage"


class LineageFacetsExtender(OpenLineageExtender):
    """Drop-in superset of OpenLineageExtender: pass it instead of the community emitter."""

    producer: str = _PRODUCER

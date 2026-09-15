"""mloda-community-openlineage: OpenLineage RunEvents for mloda pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender

__all__ = ["OpenLineageExtender"]


# Lazy on purpose: the mloda-community bundle ships this extender behind the mloda-community[openlineage]
# extra, and the mloda.optional_dependencies marker (see _optional_dependencies.py) is loaded through this
# package, so nothing here may import openlineage at import time.
def __getattr__(name: str) -> Any:
    if name == "OpenLineageExtender":
        from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender

        return OpenLineageExtender
    raise AttributeError(name)

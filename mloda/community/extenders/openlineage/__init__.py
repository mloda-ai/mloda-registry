"""mloda-community-openlineage: OpenLineage RunEvents for mloda pipelines."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from mloda.community.extenders.openlineage._optional_dependencies import OPTIONAL_DEPENDENCIES

if TYPE_CHECKING:
    from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender

__all__ = ["OpenLineageExtender"]
# mypy only reads a plain list/tuple literal, so the extra is kept above and cleared here at runtime.
if not TYPE_CHECKING and importlib.util.find_spec(OPTIONAL_DEPENDENCIES[0]) is None:
    __all__ = []


# Lazy on purpose: the mloda-community bundle ships this extender behind the mloda-community[openlineage]
# extra, and the mloda.optional_dependencies marker (see _optional_dependencies.py) is loaded through this
# package, so nothing here may import openlineage at import time.
def __getattr__(name: str) -> Any:
    if name == "OpenLineageExtender":
        from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender

        return OpenLineageExtender
    raise AttributeError(name)

"""mloda-community-openlineage: OpenLineage RunEvents for mloda pipelines."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender


def _api_module_missing() -> bool:
    # Probes the module the extender imports, not the namespace-package root; ValueError means a stub with no __spec__.
    try:
        return importlib.util.find_spec("openlineage.client") is None
    except (ImportError, ValueError):
        return True


__all__ = ["OpenLineageExtender"]
# mypy only reads a plain list/tuple literal, so the extra is kept above and cleared here at runtime.
if not TYPE_CHECKING and _api_module_missing():
    __all__ = []


# Lazy on purpose: the mloda-community bundle ships this extender behind the mloda-community[openlineage]
# extra, and the mloda.optional_dependencies marker (see _optional_dependencies.py) is loaded through this
# package, so nothing here may import openlineage at import time.
def __getattr__(name: str) -> Any:
    if name == "OpenLineageExtender":
        from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender

        return OpenLineageExtender
    raise AttributeError(name)

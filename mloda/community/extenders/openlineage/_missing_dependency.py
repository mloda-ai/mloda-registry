"""Module-level __getattr__ raised when openlineage-python is not installed."""

from __future__ import annotations

from typing import Any

_DISTRIBUTION = "openlineage-python"
_EXTRA = "mloda-community[openlineage]"
_EXTENDER_NAME = "OpenLineageExtender"


def __getattr__(name: str) -> Any:
    if name == _EXTENDER_NAME:
        raise ImportError(f"{_EXTENDER_NAME} requires '{_DISTRIBUTION}'; install it via '{_EXTRA}'.")
    raise AttributeError(name)

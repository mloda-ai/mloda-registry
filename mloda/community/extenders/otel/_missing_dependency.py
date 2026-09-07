"""Module-level __getattr__ raised when opentelemetry-api is not installed."""

from __future__ import annotations

from typing import Any

_DISTRIBUTION = "opentelemetry-api"
_EXTRA = "mloda-community[otel]"
_EXTENDER_NAME = "OtelExtender"


def __getattr__(name: str) -> Any:
    if name == _EXTENDER_NAME:
        raise ImportError(f"{_EXTENDER_NAME} requires '{_DISTRIBUTION}'; install it via '{_EXTRA}'.")
    raise AttributeError(name)

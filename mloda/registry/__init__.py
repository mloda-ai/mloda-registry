"""mloda registry - plugin discovery and search."""

from typing import Any


def discover() -> list[Any]:
    """List available plugins. (Not yet implemented)"""
    return []


def search(tags: list[str] | None = None) -> list[Any]:
    """Search plugins by criteria. (Not yet implemented)"""
    return []


__all__ = ["discover", "search"]

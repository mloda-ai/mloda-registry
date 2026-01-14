"""Tests for mloda.registry package."""

from mloda.registry import discover, search


def test_discover_returns_list() -> None:
    """Verify discover() returns a list."""
    result = discover()
    assert isinstance(result, list)
    assert result == []


def test_search_returns_list() -> None:
    """Verify search() returns a list."""
    result = search()
    assert isinstance(result, list)
    assert result == []


def test_search_with_tags() -> None:
    """Verify search() accepts tags parameter."""
    result = search(tags=["community", "timeseries"])
    assert isinstance(result, list)
    assert result == []

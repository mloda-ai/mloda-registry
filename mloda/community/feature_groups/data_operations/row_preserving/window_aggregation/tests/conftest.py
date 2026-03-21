"""Shared fixtures for window_aggregation tests."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.testing.data_creator import DataOperationsTestDataCreator


@pytest.fixture
def raw_data() -> dict[str, list[Any]]:
    """Return the shared 12-row test dataset as a plain dict."""
    return DataOperationsTestDataCreator.get_raw_data()

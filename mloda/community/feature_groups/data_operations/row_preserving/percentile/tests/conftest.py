"""Shared fixtures and helpers for percentile tests."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import (
    make_feature_set,
)

# Re-export make_feature_set so existing imports from conftest still work.
__all__ = ["make_feature_set"]


@pytest.fixture
def raw_data() -> dict[str, list[Any]]:
    """Return the shared 12-row test dataset as a plain dict."""
    return DataOperationsTestDataCreator.get_raw_data()


@pytest.fixture
def sample_data() -> pa.Table:
    """Return the shared 12-row test dataset as a PyArrow Table."""
    return PyArrowDataOpsTestDataCreator.create()

"""Shared fixtures and helpers for rank tests."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    make_feature_set,
)

__all__ = ["make_feature_set"]


@pytest.fixture
def raw_data() -> dict[str, list[Any]]:
    """Return the shared 12-row test dataset as a plain dict."""
    return DataOperationsTestDataCreator.get_raw_data()


@pytest.fixture
def sample_data() -> pa.Table:
    """Return the shared 12-row test dataset as a PyArrow Table."""
    return PyArrowDataOpsTestDataCreator.create()

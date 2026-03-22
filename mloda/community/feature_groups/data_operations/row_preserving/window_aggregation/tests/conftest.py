"""Shared fixtures and helpers for window_aggregation tests."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator import DataOperationsTestDataCreator, PyArrowDataOpsTestDataCreator
from mloda.user import Feature


@pytest.fixture
def raw_data() -> dict[str, list[Any]]:
    """Return the shared 12-row test dataset as a plain dict."""
    return DataOperationsTestDataCreator.get_raw_data()


@pytest.fixture
def sample_data() -> pa.Table:
    """Return the shared 12-row test dataset as a PyArrow Table."""
    return PyArrowDataOpsTestDataCreator.create()


def make_feature_set(feature_name: str, partition_by: list[str]) -> FeatureSet:
    """Build a FeatureSet with partition_by options."""
    feature = Feature(
        feature_name,
        options=Options(context={"partition_by": partition_by}),
    )
    fs = FeatureSet()
    fs.add(feature)
    return fs

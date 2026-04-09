"""Tests for PandasScalarAggregate compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pandas_scalar_aggregate import (
    PandasScalarAggregate,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate import (
    ScalarAggregateTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin


class TestPandasScalarAggregate(PandasTestMixin, ScalarAggregateTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasScalarAggregate

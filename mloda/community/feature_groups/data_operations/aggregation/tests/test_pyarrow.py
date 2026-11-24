"""Tests for PyArrowAggregation compute implementation."""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import Options

from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
    PyArrowAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_multi_feature_set
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin

_ALL_NAN_EXTREMA_CASES = [
    pytest.param([math.nan] * 3, math.nan, math.nan, id="all-nan"),
    pytest.param([math.nan, None, math.nan], math.nan, math.nan, id="nan-and-null"),
    pytest.param([math.nan, math.inf, None], math.inf, math.inf, id="nan-and-positive-infinity"),
    pytest.param([-math.inf, math.nan, math.inf], -math.inf, math.inf, id="both-infinities-and-nan"),
    pytest.param([math.nan, 3.0, -2.0], -2.0, 3.0, id="nan-and-finite"),
    pytest.param([None] * 3, None, None, id="all-null"),
]


class TestPyArrowAggregation(CapabilityHookTestMixin, PyArrowTestMixin, AggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowAggregation

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__nunique_agg", Options()),)

    @classmethod
    def capability_unsupported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__median_agg", Options()),
            ("value__mode_agg", Options()),
        )

    @classmethod
    def merges_signed_zero_keys(cls) -> bool:
        """PyArrow's own Table.group_by() splits 0.0/-0.0, matching the reference exactly."""
        return False

    @pytest.mark.parametrize("arrow_type", [pa.float32(), pa.float64()], ids=["float32", "float64"])
    @pytest.mark.parametrize(("values", "expected_min", "expected_max"), _ALL_NAN_EXTREMA_CASES)
    def test_min_max_all_nan_group(
        self,
        arrow_type: pa.DataType,
        values: list[float | None],
        expected_min: float | None,
        expected_max: float | None,
    ) -> None:
        """An all-NaN group keeps NaN for min/max (PyArrow before 26 returns +/-inf) without touching other groups."""
        table = pa.table(
            {
                "grp": ["case"] * 3 + ["control"] * 3,
                "value": pa.array([*values, 4.0, math.nan, 1.0], type=arrow_type),
            }
        )
        names = ["value__min_agg", "value__max_agg", "value__count_agg"]
        fs = make_multi_feature_set([(name, {"partition_by": ["grp"]}) for name in names])
        result = calculate_with(PyArrowAggregation, table, fs)
        for name in ("value__min_agg", "value__max_agg"):
            assert result.column(name).type == arrow_type

        rows = {row["grp"]: [row[name] for name in names] for row in result.to_pylist()}
        expected_count = sum(v is not None for v in values)
        assert rows["case"] == pytest.approx([expected_min, expected_max, expected_count], nan_ok=True)
        assert rows["control"] == pytest.approx([1.0, 4.0, 3])

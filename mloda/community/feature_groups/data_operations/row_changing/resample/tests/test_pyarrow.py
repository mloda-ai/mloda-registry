"""Tests for PyArrowResample compute implementation."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
    PyArrowResample,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_changing.resample.resample import (
    ResampleTestBase,
)


class TestPyArrowResample(PyArrowTestMixin, ResampleTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowResample

    @classmethod
    def merges_signed_zero_keys(cls) -> bool:
        """PyArrow's own Table.group_by() splits 0.0/-0.0 partition keys."""
        return False

    @pytest.mark.parametrize("arrow_type", [pa.float32(), pa.float64()], ids=["float32", "float64"])
    @pytest.mark.parametrize(
        ("values", "expected_min", "expected_max", "expected_count"),
        [
            pytest.param([math.nan] * 4, math.nan, math.nan, 4, id="all-nan"),
            pytest.param([math.nan, None, math.nan, None], math.nan, math.nan, 2, id="nan-and-null"),
            pytest.param([math.inf, math.inf, None, None], math.inf, math.inf, 2, id="positive-infinity"),
            pytest.param([-math.inf, None, -math.inf, None], -math.inf, -math.inf, 2, id="negative-infinity"),
            pytest.param([math.nan, math.inf, None, math.nan], math.inf, math.inf, 3, id="nan-and-positive-infinity"),
            pytest.param(
                [None, math.nan, -math.inf, math.nan], -math.inf, -math.inf, 3, id="nan-and-negative-infinity"
            ),
            pytest.param([-math.inf, math.nan, math.inf, None], -math.inf, math.inf, 3, id="both-infinities-and-nan"),
            pytest.param([math.nan, 3.0, None, -2.0], -2.0, 3.0, 3, id="nan-and-finite"),
            pytest.param([None] * 4, None, None, 0, id="all-null"),
        ],
    )
    def test_min_max_nan_and_infinity_multi_aggregation(
        self,
        arrow_type: pa.DataType,
        values: list[float | None],
        expected_min: float | None,
        expected_max: float | None,
        expected_count: int,
    ) -> None:
        """All-NaN extrema are NaN without replacing real infinities or changing other aggregates."""
        table = pa.table(
            {
                "region": ["case"] * 4 + ["control"] * 4 + ["case"] * 4,
                "ts": pa.array(
                    [datetime(2024, 1, 1, hour, minute) for hour in (8, 8, 9) for minute in (5, 15, 25, 35)],
                    type=pa.timestamp("us"),
                ),
                "value": pa.array(values + [10.0, 2.0, None, 6.0, -3.0, None, 4.0, None], type=arrow_type),
                "other": pa.array(range(1, 13), type=pa.int64()),
            }
        )
        names = [
            "other__resample_1_hour_max",
            "value__resample_1_hour_min",
            "value__resample_1_hour_count",
            "other__resample_1_hour_min",
            "value__resample_1_hour_max",
            "other__resample_1_hour_sum",
        ]
        result = calculate_with(PyArrowResample, table, self._multi_feature_set(names, ["region"]))
        assert set(result.column_names) == {"region", "ts", *names}
        assert len(result.column_names) == len(names) + 2
        assert result.num_rows == 3
        for operation in ("min", "max"):
            assert result.column(f"value__resample_1_hour_{operation}").type == arrow_type
        assert result.column("value__resample_1_hour_count").type == pa.int64()

        rows = {(row["region"], row["ts"]): row for row in result.to_pylist()}
        expected: dict[tuple[str, datetime], list[float | int | None]] = {
            ("case", datetime(2024, 1, 1, 8)): [4, expected_min, expected_count, 1, expected_max, 10],
            ("control", datetime(2024, 1, 1, 8)): [8, 2.0, 3, 5, 10.0, 26],
            ("case", datetime(2024, 1, 1, 9)): [12, -3.0, 2, 9, 4.0, 42],
        }
        assert rows.keys() == expected.keys()
        for key, expected_values in expected.items():
            actual = {name: rows[key][name] for name in names}
            assert actual == pytest.approx(dict(zip(names, expected_values)), nan_ok=True), f"bucket {key!r}"

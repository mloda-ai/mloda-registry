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
from mloda.community.feature_groups.data_operations.tests.pyarrow_all_nan_cases import (
    ALL_NAN_EXTREMA_CASES,
    all_nan_case_control_table,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.helpers import (
    calculate_with,
    extract_column,
    make_multi_feature_set,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin


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
    @pytest.mark.parametrize(("values", "expected_min", "expected_max"), ALL_NAN_EXTREMA_CASES)
    def test_min_max_all_nan_group(
        self,
        arrow_type: pa.DataType,
        values: list[float | None],
        expected_min: float | None,
        expected_max: float | None,
    ) -> None:
        """An all-NaN group keeps NaN for min/max (PyArrow before 26 returns +/-inf) without touching other groups."""
        table = all_nan_case_control_table(values, arrow_type)
        names = ["value__min_agg", "value__max_agg", "value__count_agg"]
        fs = make_multi_feature_set([(name, {"partition_by": ["grp"]}) for name in names])
        result = calculate_with(PyArrowAggregation, table, fs)
        for name in ("value__min_agg", "value__max_agg"):
            assert result.column(name).type == arrow_type

        rows = {row["grp"]: [row[name] for name in names] for row in result.to_pylist()}
        expected_count = sum(v is not None for v in values)
        assert rows["case"] == pytest.approx([expected_min, expected_max, expected_count], nan_ok=True)
        assert rows["control"] == pytest.approx([1.0, 4.0, 3])

        reference = self.reference_calculate(table, fs)
        reference_rows = {
            group: dict(zip(names, values_))
            for group, *values_ in zip(
                extract_column(reference, "grp"), *(extract_column(reference, name) for name in names)
            )
        }
        assert [reference_rows["case"][name] for name in names] == pytest.approx(
            [expected_min, expected_max, expected_count], nan_ok=True
        )

    def test_min_max_repair_survives_helper_name_collision_across_sources(self) -> None:
        """Repairs of two float sources keep their own flags beside a user column named like the repair helper."""
        collision = "__mloda_agg_nan__"
        table = pa.table(
            {
                collision: ["case"] * 3 + ["control"] * 3,
                "a": pa.array([math.nan] * 3 + [1.0, math.nan, 4.0], type=pa.float64()),
                "b": pa.array([math.nan, None, math.nan, -1.0, math.nan, 5.0], type=pa.float64()),
                "i": pa.array([7, 8, 9, 3, 2, 5], type=pa.int64()),
            }
        )
        original = repr(table.to_pylist())
        names = [
            "a__min_agg",
            "b__max_agg",
            "a__max_agg",
            "b__min_agg",
            "a__count_agg",
            "i__min_agg",
            "a__last_agg",
        ]
        fs = make_multi_feature_set([(name, {"partition_by": [collision]}) for name in names])
        result = calculate_with(PyArrowAggregation, table, fs)
        assert repr(table.to_pylist()) == original
        assert set(result.column_names) == {collision, *names}
        rows = {row[collision]: [row[name] for name in names] for row in result.to_pylist()}
        assert rows.keys() == {"case", "control"}
        assert rows["case"] == pytest.approx([math.nan, math.nan, math.nan, math.nan, 3, 7, math.nan], nan_ok=True)
        assert rows["control"] == pytest.approx([1.0, 5.0, 4.0, -1.0, 3, 2, 4.0], nan_ok=True)

    @pytest.mark.parametrize(
        ("operation", "expected_case", "expected_control"),
        [("min", math.nan, 1.0), ("max", math.nan, 4.0)],
    )
    def test_min_max_all_nan_under_mask(self, operation: str, expected_case: float, expected_control: float) -> None:
        """A group whose mask-kept rows are all NaN gives NaN; a group with every row masked out stays null."""
        table = pa.table(
            {
                "grp": ["case"] * 3 + ["empty"] * 2 + ["control"] * 2,
                "flag": ["Y", "N", "Y", "N", "N", "Y", "Y"],
                "value": pa.array([math.nan, 5.0, math.nan, 2.0, 3.0, 1.0, 4.0], type=pa.float64()),
            }
        )
        name = f"value__{operation}_agg"
        fs = make_multi_feature_set([(name, {"partition_by": ["grp"], "mask": ("flag", "equal", "Y")})])
        result = calculate_with(PyArrowAggregation, table, fs)
        assert result.column(name).type == pa.float64()
        rows = {row["grp"]: row[name] for row in result.to_pylist()}
        assert rows.keys() == {"case", "empty", "control"}
        assert rows["case"] == pytest.approx(expected_case, nan_ok=True)
        assert rows["empty"] is None
        assert rows["control"] == pytest.approx(expected_control)

"""Tests for the aggregation references and their shared kernel (mloda.testing)."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import DataType, Feature, Options


class TestReferenceWindowAggregationNoFutureWarning:
    """first/last with partition_by must not raise pyarrow's null_placement FutureWarning."""

    @staticmethod
    def _agg_no_warning(agg_type: str, expected: list[Any]) -> None:
        import warnings

        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.reference import (
            reference_window_aggregation,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        fs = make_feature_set(f"value_int__{agg_type}_window", ["region"], order_by="value_int")

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = reference_window_aggregation(arrow_table, fs)

        result_col = extract_column(result, f"value_int__{agg_type}_window")
        assert result_col == expected

    def test_first_window_no_future_warning(self) -> None:
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
            EXPECTED_FIRST_BY_REGION,
        )

        self._agg_no_warning("first", EXPECTED_FIRST_BY_REGION)

    def test_last_window_no_future_warning(self) -> None:
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
            EXPECTED_LAST_BY_REGION,
        )

        self._agg_no_warning("last", EXPECTED_LAST_BY_REGION)


NAN = float("nan")


class TestAggregateKernelContract:
    """The kernel counts NaN once and 0.0/-0.0 once, and mean is avg."""

    def test_nunique_counts_nan_once(self) -> None:
        from mloda.testing.feature_groups.data_operations.aggregation_helpers import aggregate

        assert aggregate([NAN, float("nan"), 1.0, None], "nunique") == 2

    def test_nunique_counts_signed_zero_once(self) -> None:
        from mloda.testing.feature_groups.data_operations.aggregation_helpers import aggregate

        assert aggregate([0.0, -0.0, -0.0], "nunique") == 1

    def test_mean_equals_avg(self) -> None:
        from mloda.testing.feature_groups.data_operations.aggregation_helpers import aggregate

        values = [1.0, 2.0, None, 6.0]
        assert aggregate(values, "mean") == aggregate(values, "avg") == 3.0


class TestGroupRows:
    """group_rows partitions row indices by key tuple in first-occurrence order."""

    @staticmethod
    def _group(keys: list[list[Any]], num_rows: int) -> dict[tuple[Any, ...], list[int]]:
        from mloda.testing.feature_groups.data_operations.aggregation_helpers import group_rows

        return group_rows(keys, num_rows)

    def test_nan_keys_form_one_group(self) -> None:
        groups = self._group([[NAN, 1.0, float("nan")]], 3)
        assert [indices for indices in groups.values()] == [[0, 2], [1]]

    def test_none_is_distinct_from_nan(self) -> None:
        groups = self._group([[None, NAN, None, NAN]], 4)
        assert sorted(groups.values()) == [[0, 2], [1, 3]]
        assert len(groups) == 2

    def test_signed_zero_is_one_key(self) -> None:
        groups = self._group([[0.0, -0.0, 0.0]], 3)
        assert list(groups.values()) == [[0, 1, 2]]

    def test_first_occurrence_order_of_groups_and_indices(self) -> None:
        groups = self._group([["b", "a", "b", "c", "a"]], 5)
        assert list(groups.items()) == [(("b",), [0, 2]), (("a",), [1, 4]), (("c",), [3])]

    def test_multiple_key_columns(self) -> None:
        groups = self._group([["a", "a", "b"], [1, 2, 1]], 3)
        assert groups == {("a", 1): [0], ("a", 2): [1], ("b", 1): [2]}

    @pytest.mark.parametrize("num_rows", [0, 3])
    def test_no_key_columns_gives_one_whole_table_group(self, num_rows: int) -> None:
        assert self._group([], num_rows) == {(): list(range(num_rows))}


class TestOrderKey:
    """order_key sorts real values ascending, then NaN, then None, stably."""

    @staticmethod
    def _sorted(values: list[Any]) -> list[Any]:
        from mloda.testing.feature_groups.data_operations.aggregation_helpers import order_key

        return sorted(values, key=order_key)

    def test_floats_values_then_nan_then_none(self) -> None:
        result = self._sorted([None, NAN, 3.0, 1.0, None, float("nan"), 2.0])
        assert result[:3] == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) and math.isnan(v) for v in result[3:5])
        assert result[5:] == [None, None]

    def test_datetimes_with_none_last(self) -> None:
        early, late = dt.datetime(2024, 1, 1), dt.datetime(2024, 6, 1)
        assert self._sorted([None, late, early]) == [early, late, None]

    def test_sort_is_stable(self) -> None:
        from mloda.testing.feature_groups.data_operations.aggregation_helpers import order_key

        pairs = [(1.0, "a"), (NAN, "b"), (1.0, "c"), (NAN, "d"), (None, "e"), (None, "f")]
        result = sorted(pairs, key=lambda p: order_key(p[0]))
        assert [tag for _, tag in result] == ["a", "c", "b", "d", "e", "f"]


_TS_UTC = pa.timestamp("us", tz="UTC")


class TestDeclaredType:
    """declared_type(agg_type, source_type) is the schema of an aggregate column."""

    @staticmethod
    def _declared(agg_type: str, source_type: pa.DataType) -> pa.DataType:
        from mloda.testing.feature_groups.data_operations.aggregation_helpers import declared_type

        return declared_type(agg_type, source_type)

    @pytest.mark.parametrize("agg_type", ["count", "nunique"])
    @pytest.mark.parametrize("source_type", [pa.int32(), pa.float64(), pa.string()])
    def test_counting_is_int64(self, agg_type: str, source_type: pa.DataType) -> None:
        assert self._declared(agg_type, source_type) == pa.int64()

    @pytest.mark.parametrize(
        "agg_type", ["avg", "mean", "median", "std", "var", "std_pop", "std_samp", "var_pop", "var_samp"]
    )
    @pytest.mark.parametrize("source_type", [pa.int64(), pa.float32()])
    def test_statistics_are_float64(self, agg_type: str, source_type: pa.DataType) -> None:
        assert self._declared(agg_type, source_type) == pa.float64()

    @pytest.mark.parametrize(
        "source_type, expected",
        [
            (pa.int32(), pa.int64()),
            (pa.int64(), pa.int64()),
            (pa.uint8(), pa.uint64()),
            (pa.float32(), pa.float64()),
            (pa.float64(), pa.float64()),
        ],
    )
    def test_sum_widens(self, source_type: pa.DataType, expected: pa.DataType) -> None:
        assert self._declared("sum", source_type) == expected

    def test_sum_on_string_raises_naming_the_type(self) -> None:
        with pytest.raises(ValueError, match="string"):
            self._declared("sum", pa.string())

    @pytest.mark.parametrize("agg_type", ["min", "max", "first", "last", "mode"])
    @pytest.mark.parametrize("source_type", [pa.int64(), pa.float64(), pa.string(), _TS_UTC])
    def test_value_picking_keeps_source_type(self, agg_type: str, source_type: pa.DataType) -> None:
        assert self._declared(agg_type, source_type) == source_type


_DATATYPE_TO_PA = {DataType.INT64: pa.int64(), DataType.DOUBLE: pa.float64()}
_AGG_TYPES = ["count", "nunique", "sum", "avg", "min", "max", "median", "std", "var", "first", "last", "mode"]


def _rule_cases() -> list[Any]:
    from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
    from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
        FrameAggregateFeatureGroup,
    )
    from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
        ScalarAggregateFeatureGroup,
    )
    from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
        WindowAggregationFeatureGroup,
    )

    part = Options(context={"partition_by": ["region"]})
    framed = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
    families: list[tuple[Any, str, Options, list[str]]] = [
        (AggregationFeatureGroup, "value_int__{a}_agg", part, _AGG_TYPES),
        (WindowAggregationFeatureGroup, "value_int__{a}_window", part, _AGG_TYPES),
        (FrameAggregateFeatureGroup, "value_int__{a}_rolling_3", framed, [a for a in _AGG_TYPES if a != "nunique"]),
        (ScalarAggregateFeatureGroup, "value_int__{a}_scalar", Options(), [a for a in _AGG_TYPES if a != "nunique"]),
    ]
    return [
        pytest.param(group, template.format(a=agg), options, agg, id=f"{group.__name__}-{agg}")
        for group, template, options, aggs in families
        for agg in aggs
    ]


class TestDeclaredTypeAgreesWithReturnDataTypeRule:
    """Where a family's return_data_type_rule is non-None, declared_type gives the same type."""

    @pytest.mark.parametrize("group, name, options, agg_type", _rule_cases())
    def test_agrees(self, group: Any, name: str, options: Options, agg_type: str) -> None:
        from mloda.testing.feature_groups.data_operations.aggregation_helpers import declared_type

        rule = group.return_data_type_rule(Feature(name, options=options))
        if rule is None:
            pytest.skip("rule leaves the type open")
        assert declared_type(agg_type, pa.int64()) == _DATATYPE_TO_PA[rule]

    def test_counting_rule_is_exercised(self) -> None:
        """Guard against vacuity: count has a non-None rule in every family."""
        counting = [c for c in _rule_cases() if c.values[3] == "count"]
        assert len(counting) == 4
        for case in counting:
            group, name, options, _ = case.values
            assert group.return_data_type_rule(Feature(name, options=options)) == DataType.INT64


class TestReferenceScalarAggregateRejectsUnsupported:
    """The scalar reference rejects agg types outside the family's _SUPPORTED_AGG_TYPES."""

    def test_nunique_raises_value_error(self) -> None:
        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
        from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.reference import (
            reference_scalar_aggregate,
        )

        fs = make_feature_set("value_int__nunique_scalar")
        with pytest.raises(ValueError, match="nunique"):
            reference_scalar_aggregate(PyArrowDataOpsTestDataCreator.create(), fs)

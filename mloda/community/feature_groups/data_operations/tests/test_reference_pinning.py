"""Runs every pinned row of a reference against it, strictly, and guards the tables."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import DataType, Options

from mloda.community.feature_groups.data_operations.catalog import DataOperationsCatalog
from mloda.community.feature_groups.data_operations.row_preserving.binning.base import BinningFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import DateTimeFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TimeBucketizationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.string.base import StringFeatureGroup
from mloda.community.feature_groups.data_operations.tests.test_framework_support_matrix import OPERATIONS
from mloda.testing.feature_groups.data_operations.helpers import (
    arrow_table_of,
    assert_values_with_nulls,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.pinning import (
    PinningRow,
    PinningTable,
)
from mloda.testing.feature_groups.data_operations.row_preserving.arithmetic.pinning_rows import (
    POINT_ARITHMETIC_PINNING,
    SCALAR_ARITHMETIC_PINNING,
)
from mloda.testing.feature_groups.data_operations.row_preserving.arithmetic.reference import (
    reference_point_arithmetic,
    reference_scalar_arithmetic,
)
from mloda.testing.feature_groups.data_operations.row_preserving.binning.pinning_rows import BINNING_PINNING
from mloda.testing.feature_groups.data_operations.row_preserving.binning.reference import reference_binning
from mloda.testing.feature_groups.data_operations.row_preserving.datetime.pinning_rows import DATETIME_PINNING
from mloda.testing.feature_groups.data_operations.row_preserving.datetime.reference import reference_datetime
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.pinning_rows import (
    TIME_BUCKETIZATION_PINNING,
)
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.reference import (
    reference_time_bucketization,
)
from mloda.testing.feature_groups.data_operations.string.pinning_rows import STRING_PINNING
from mloda.testing.feature_groups.data_operations.string.reference import reference_string
from mloda.testing.typed_data import TypedTestData

#: Every reference that ships pinning rows: its group, its reference function and its table.
PINNED: list[tuple[type, Callable[..., Any], PinningTable]] = [
    (PointArithmeticFeatureGroup, reference_point_arithmetic, POINT_ARITHMETIC_PINNING),
    (ScalarArithmeticFeatureGroup, reference_scalar_arithmetic, SCALAR_ARITHMETIC_PINNING),
    (StringFeatureGroup, reference_string, STRING_PINNING),
    (DateTimeFeatureGroup, reference_datetime, DATETIME_PINNING),
    (BinningFeatureGroup, reference_binning, BINNING_PINNING),
    (TimeBucketizationFeatureGroup, reference_time_bucketization, TIME_BUCKETIZATION_PINNING),
]

ROW_PARAMS = [
    pytest.param(group, reference, table, row, id=f"{table.operation}-{index}-{row.case}")
    for group, reference, table in PINNED
    for index, row in enumerate(table.rows)
]
TABLE_PARAMS = [pytest.param(group, table, id=table.operation) for group, _, table in PINNED]


def _expected_arrow_type(data_type: DataType, values: tuple[Any, ...]) -> pa.DataType:
    if data_type == DataType.TIMESTAMP_MICROS_TZ:
        # No single Arrow type: the zone comes from the first non-null expected value, so every TZ expected column needs one.
        return pa.array(list(values)).type
    return DataType.to_arrow_type(data_type)


@pytest.mark.parametrize(("group", "reference", "table", "row"), ROW_PARAMS)
def test_reference_matches_pinned_row(
    group: type, reference: Callable[..., Any], table: PinningTable, row: PinningRow
) -> None:
    result = reference(group, arrow_table_of(TypedTestData(row.inputs)), make_feature_set(row.feature, **row.context))
    for name, (data_type, values) in row.expected.items():
        assert_values_with_nulls(result.column(name).to_pylist(), list(values), strict=True)
        assert result.schema.field(name).type == _expected_arrow_type(data_type, values), f"column {name} type"
    assert set(result.column_names) == set(row.inputs) | set(row.expected)


@pytest.mark.parametrize(("group", "table"), TABLE_PARAMS)
class TestTableGuards:
    def test_table_has_rows(self, group: type, table: PinningTable) -> None:
        assert table.rows

    def test_every_case_is_a_rule(self, group: type, table: PinningTable) -> None:
        assert [row.case for row in table.rows if row.case not in table.rules] == []

    def test_every_rule_is_used_by_a_table_sharing_it(self, group: type, table: PinningTable) -> None:
        used = {row.case for _, _, other in PINNED if dict(other.rules) == dict(table.rules) for row in other.rows}
        assert sorted(set(table.rules) - used) == []

    def test_every_feature_is_requestable(self, group: Any, table: PinningTable) -> None:
        bad = [
            row.feature
            for row in table.rows
            if not group.match_feature_group_criteria(row.feature, Options(context=dict(row.context)))
        ]
        assert bad == []

    def test_feature_is_an_expected_column(self, group: type, table: PinningTable) -> None:
        assert [row.feature for row in table.rows if row.feature not in row.expected] == []

    def test_expected_columns_are_valid_typed_data(self, group: type, table: PinningTable) -> None:
        for row in table.rows:
            TypedTestData(row.expected)

    def test_input_values_fit_their_declared_type(self, group: type, table: PinningTable) -> None:
        bad = [
            (row.feature, name, value)
            for row in table.rows
            for name, (data_type, values) in row.inputs.items()
            for value in values
            if not _fits(data_type, value)
        ]
        assert bad == []

    def test_operation_is_known_and_matches_the_group(self, group: Any, table: PinningTable) -> None:
        assert table.operation in OPERATIONS
        assert DataOperationsCatalog.get(table.operation).prefix_pattern == group.PREFIX_PATTERN


_FLOAT_TYPES = (DataType.FLOAT, DataType.DOUBLE)


def _fits(data_type: DataType, value: Any) -> bool:
    if value is None:
        return True
    if data_type in _FLOAT_TYPES:
        return isinstance(value, float)
    if data_type == DataType.STRING:
        return isinstance(value, str)
    if data_type == DataType.DATE:
        return isinstance(value, date) and not isinstance(value, datetime)
    if data_type in (DataType.TIMESTAMP_MICROS, DataType.TIMESTAMP_MILLIS):
        return isinstance(value, datetime) and value.tzinfo is None
    if data_type == DataType.TIMESTAMP_MICROS_TZ:
        return isinstance(value, datetime) and value.tzinfo is not None
    return isinstance(value, int) and not isinstance(value, bool)


def _typed_table(**columns: tuple[pa.DataType, list[Any]]) -> pa.Table:
    return pa.table({name: pa.array(values, type=arrow_type) for name, (arrow_type, values) in columns.items()})


_POINT = (reference_point_arithmetic, PointArithmeticFeatureGroup)
_SCALAR = (reference_scalar_arithmetic, ScalarArithmeticFeatureGroup)
_STRING = (reference_string, StringFeatureGroup)
_DATETIME = (reference_datetime, DateTimeFeatureGroup)
_BINNING = (reference_binning, BinningFeatureGroup)
_BUCKET = (reference_time_bucketization, TimeBucketizationFeatureGroup)

_REJECTION_PARAMS = [
    pytest.param(
        *_POINT,
        "x&y__bogus_point",
        {},
        {"x": (pa.int64(), [1]), "y": (pa.int64(), [2])},
        "bogus",
        id="point-unknown-op",
    ),
    pytest.param(
        *_SCALAR, "x__bogus_constant", {"constant": 1}, {"x": (pa.int64(), [1])}, "bogus", id="scalar-unknown-op"
    ),
    pytest.param(
        *_POINT,
        "x&y__add_point",
        {},
        {"x": (pa.bool_(), [True]), "y": (pa.int64(), [1])},
        "must be numeric",
        id="point-bool-source",
    ),
    pytest.param(
        *_POINT,
        "x&y__add_point",
        {},
        {"x": (pa.string(), ["1"]), "y": (pa.int64(), [1])},
        "must be numeric",
        id="point-string-source",
    ),
    pytest.param(
        *_SCALAR,
        "x__add_constant",
        {"constant": 1},
        {"x": (pa.bool_(), [True])},
        "must be numeric",
        id="scalar-bool-source",
    ),
    pytest.param(
        *_SCALAR,
        "x__add_constant",
        {"constant": 1},
        {"x": (pa.string(), ["1"])},
        "must be numeric",
        id="scalar-string-source",
    ),
    pytest.param(
        *_POINT,
        "x&y__add_point",
        {},
        {"x": (pa.int64(), [2**63 - 1]), "y": (pa.int64(), [1])},
        "outside",
        id="point-int64-overflow",
    ),
    pytest.param(
        *_SCALAR,
        "x__add_constant",
        {"constant": 1},
        {"x": (pa.int64(), [2**63 - 1])},
        "outside",
        id="scalar-int64-overflow",
    ),
    pytest.param(
        *_POINT,
        "x&y__bogus_point",
        {},
        {"x": (pa.int64(), []), "y": (pa.int64(), [])},
        "bogus",
        id="point-unknown-op-empty",
    ),
    pytest.param(
        *_SCALAR,
        "x__bogus_constant",
        {"constant": 1},
        {"x": (pa.int64(), [])},
        "bogus",
        id="scalar-unknown-op-empty",
    ),
    pytest.param(
        *_POINT,
        "x&y__add_point",
        {},
        {"x": (pa.int32(), [2**31 - 1]), "y": (pa.int32(), [1])},
        "outside",
        id="point-int32-overflow",
    ),
    pytest.param(
        *_POINT,
        "x&y__subtract_point",
        {},
        {"x": (pa.uint8(), [1]), "y": (pa.uint8(), [2])},
        "outside",
        id="point-uint8-underflow",
    ),
    pytest.param(
        *_POINT,
        "x&y__multiply_point",
        {},
        {"x": (pa.int8(), [100]), "y": (pa.int8(), [100])},
        "outside",
        id="point-int8-overflow",
    ),
    pytest.param(
        *_SCALAR,
        "x__add_constant",
        {"constant": 1},
        {"x": (pa.int32(), [2**31 - 1])},
        "outside",
        id="scalar-int32-overflow",
    ),
    pytest.param(
        *_SCALAR,
        "x__subtract_constant",
        {"constant": 2},
        {"x": (pa.uint8(), [1])},
        "outside",
        id="scalar-uint8-underflow",
    ),
    pytest.param(
        *_POINT,
        "x&y__add_point",
        {},
        {"x": (pa.decimal128(10, 2), [Decimal("1.00")]), "y": (pa.int64(), [1])},
        "must be numeric",
        id="point-decimal-source",
    ),
    pytest.param(
        *_POINT,
        "x&y__add_point",
        {},
        {"x": (pa.null(), [None]), "y": (pa.int64(), [1])},
        "must be numeric",
        id="point-null-source",
    ),
    pytest.param(
        *_SCALAR,
        "x__add_constant",
        {"constant": 1},
        {"x": (pa.decimal128(10, 2), [Decimal("1.00")])},
        "must be numeric",
        id="scalar-decimal-source",
    ),
    pytest.param(
        *_SCALAR,
        "x__add_constant",
        {"constant": 1},
        {"x": (pa.null(), [None])},
        "must be numeric",
        id="scalar-null-source",
    ),
    pytest.param(
        *_SCALAR,
        "x__divide_constant",
        {"constant": 0},
        {"x": (pa.int64(), [1])},
        "divide by zero",
        id="scalar-divide-by-zero-int",
    ),
    pytest.param(
        *_SCALAR,
        "x__divide_constant",
        {"constant": -0.0},
        {"x": (pa.float64(), [1.0])},
        "divide by zero",
        id="scalar-divide-by-negative-zero",
    ),
    pytest.param(
        *_STRING,
        "bogus_feature",
        {"string_op": "bogus", "in_features": "x"},
        {"x": (pa.string(), ["a"])},
        "bogus",
        id="string-unknown-op",
    ),
    pytest.param(
        *_STRING,
        "bogus_feature",
        {"string_op": "bogus", "in_features": "x"},
        {"x": (pa.string(), [])},
        "bogus",
        id="string-unknown-op-empty",
    ),
    pytest.param(*_STRING, "x__upper", {}, {"x": (pa.int64(), [1])}, None, id="string-int-source"),
    pytest.param(*_STRING, "x__length", {}, {"x": (pa.binary(), [b"a"])}, None, id="string-binary-source"),
    pytest.param(
        *_STRING,
        "x__upper",
        {},
        {"x": (pa.string_view(), ["a"])},
        None,
        id="string-string-view-source",
        marks=pytest.mark.skipif(not hasattr(pa, "string_view"), reason="pyarrow without string_view"),
    ),
    pytest.param(*_STRING, "x__upper", {}, {"x": (pa.null(), [None])}, None, id="string-null-source"),
    pytest.param(
        *_DATETIME,
        "bogus_feature",
        {"datetime_op": "bogus", "in_features": "x"},
        {"x": (pa.timestamp("us"), [None])},
        "bogus",
        id="datetime-unknown-op",
    ),
    pytest.param(
        *_DATETIME,
        "bogus_feature",
        {"datetime_op": "bogus", "in_features": "x"},
        {"x": (pa.timestamp("us"), [])},
        "bogus",
        id="datetime-unknown-op-empty",
    ),
    pytest.param(
        *_DATETIME, "x__hour", {}, {"x": (pa.date32(), [date(2024, 1, 1)])}, None, id="datetime-date-source-hour"
    ),
    pytest.param(
        *_DATETIME, "x__minute", {}, {"x": (pa.date32(), [date(2024, 1, 1)])}, None, id="datetime-date-source-minute"
    ),
    pytest.param(
        *_DATETIME, "x__second", {}, {"x": (pa.date32(), [date(2024, 1, 1)])}, None, id="datetime-date-source-second"
    ),
    pytest.param(*_DATETIME, "x__year", {}, {"x": (pa.string(), ["2024-01-01"])}, None, id="datetime-string-source"),
    pytest.param(*_DATETIME, "x__year", {}, {"x": (pa.int64(), [1])}, None, id="datetime-int-source"),
    pytest.param(
        *_DATETIME, "x__year", {}, {"x": (pa.timestamp("ns"), [datetime(2024, 1, 1)])}, None, id="datetime-ns-source"
    ),
    pytest.param(
        *_BINNING,
        "bogus_feature",
        {"binning_op": "bogus", "n_bins": 3, "in_features": "x"},
        {"x": (pa.int64(), [1])},
        "bogus",
        id="binning-unknown-op",
    ),
    pytest.param(*_BINNING, "x__bin_3", {}, {"x": (pa.bool_(), [True])}, None, id="binning-bool-source"),
    pytest.param(*_BINNING, "x__bin_3", {}, {"x": (pa.string(), ["1"])}, None, id="binning-string-source"),
    pytest.param(
        *_BINNING, "x__bin_3", {}, {"x": (pa.decimal128(10, 2), [Decimal("1.00")])}, None, id="binning-decimal-source"
    ),
    pytest.param(*_BINNING, "x__bin_3", {}, {"x": (pa.null(), [None])}, None, id="binning-null-source"),
    pytest.param(*_BINNING, "x__qbin_3", {}, {"x": (pa.string(), ["1"])}, None, id="qbinning-string-source"),
    pytest.param(
        *_BINNING, "x__bin_3", {}, {"x": (pa.float64(), [1.0, float("inf")])}, None, id="binning-bin-infinity"
    ),
    pytest.param(
        *_BINNING, "x__bin_3", {}, {"x": (pa.int64(), [1, 2**53 + 1])}, None, id="binning-bin-integer-beyond-2-53"
    ),
    pytest.param(
        *_BUCKET,
        "x__floor_1_day",
        {},
        {"x": (pa.timestamp("ns"), [datetime(2024, 1, 1)])},
        None,
        id="bucket-ns-source",
    ),
    pytest.param(
        *_BUCKET,
        "x__floor_1_day",
        {},
        {"x": (pa.timestamp("us", tz="Europe/Berlin"), [datetime(2024, 1, 1, tzinfo=timezone.utc)])},
        None,
        id="bucket-named-zone-source",
    ),
    pytest.param(
        *_BUCKET,
        "x__round_1_month",
        {},
        {"x": (pa.timestamp("us", tz="+05:00"), [datetime(2024, 1, 16, 12, tzinfo=timezone.utc)])},
        None,
        id="bucket-round-month-non-utc-zone",
    ),
    pytest.param(
        *_BUCKET,
        "x__round_1_year",
        {},
        {"x": (pa.timestamp("us", tz="+05:00"), [datetime(2024, 1, 16, 12, tzinfo=timezone.utc)])},
        None,
        id="bucket-round-year-non-utc-zone",
    ),
    pytest.param(*_BUCKET, "x__floor_1_day", {}, {"x": (pa.int64(), [1])}, None, id="bucket-int-source"),
    pytest.param(*_BUCKET, "x__floor_1_day", {}, {"x": (pa.string(), ["2024-01-01"])}, None, id="bucket-string-source"),
    pytest.param(
        *_BUCKET, "x__floor_1_day", {}, {"x": (pa.date32(), [date(2024, 1, 1)])}, None, id="bucket-date-source"
    ),
]


class TestReferenceRejections:
    """The reference rejects what it cannot compute instead of guessing."""

    @pytest.mark.parametrize(("reference", "group", "feature", "context", "columns", "message"), _REJECTION_PARAMS)
    def test_unsupported_input_is_rejected(
        self,
        reference: Callable[..., Any],
        group: type,
        feature: str,
        context: dict[str, Any],
        columns: dict[str, tuple[pa.DataType, list[Any]]],
        message: str | None,
    ) -> None:
        table = _typed_table(**columns)
        with pytest.raises(ValueError, match=message):
            reference(group, table, make_feature_set(feature, **context))

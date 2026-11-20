"""Runs every pinned row of a reference against it, strictly, and guards the tables."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import DataType, Options

from mloda.community.feature_groups.data_operations.catalog import DataOperationsCatalog
from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import LookupFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)
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
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.pinning_rows import LOOKUP_PINNING
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.reference import reference_lookup
from mloda.testing.typed_data import TypedTestData

#: Every reference that ships pinning rows: its group, its reference function and its table.
PINNED: list[tuple[type, Callable[..., Any], PinningTable]] = [
    (PointArithmeticFeatureGroup, reference_point_arithmetic, POINT_ARITHMETIC_PINNING),
    (ScalarArithmeticFeatureGroup, reference_scalar_arithmetic, SCALAR_ARITHMETIC_PINNING),
    (LookupFeatureGroup, reference_lookup, LOOKUP_PINNING),
]

ROW_PARAMS = [
    pytest.param(group, reference, table, row, id=f"{table.operation}-{index}-{row.case}")
    for group, reference, table in PINNED
    for index, row in enumerate(table.rows)
]
TABLE_PARAMS = [pytest.param(group, table, id=table.operation) for group, _, table in PINNED]


@pytest.mark.parametrize(("group", "reference", "table", "row"), ROW_PARAMS)
def test_reference_matches_pinned_row(
    group: type, reference: Callable[..., Any], table: PinningTable, row: PinningRow
) -> None:
    result = reference(group, arrow_table_of(TypedTestData(row.inputs)), make_feature_set(row.feature, **row.context))
    for name, (data_type, values) in row.expected.items():
        assert_values_with_nulls(result.column(name).to_pylist(), list(values), strict=True)
        assert result.schema.field(name).type == DataType.to_arrow_type(data_type), f"column {name} type"
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
    if data_type is DataType.STRING:
        return isinstance(value, str)
    return isinstance(value, int) and not isinstance(value, bool)


def _typed_table(**columns: tuple[pa.DataType, list[Any]]) -> pa.Table:
    return pa.table({name: pa.array(values, type=arrow_type) for name, (arrow_type, values) in columns.items()})


_POINT = (reference_point_arithmetic, PointArithmeticFeatureGroup)
_SCALAR = (reference_scalar_arithmetic, ScalarArithmeticFeatureGroup)
_LOOKUP = (reference_lookup, LookupFeatureGroup)
_STR_MAP = {"a": "A"}
_INT_MAP = {1: 10}

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
        *_LOOKUP,
        "x__lookup",
        {"lookup_mapping": _INT_MAP, "on_unmatched": "null"},
        {"x": (pa.float64(), [1.0, float("nan")])},
        "x",
        id="lookup-float-source-nan",
    ),
    pytest.param(
        *_LOOKUP,
        "x__lookup",
        {"lookup_mapping": _INT_MAP, "on_unmatched": "null"},
        {"x": (pa.float64(), [1.0, 2.0])},
        "x",
        id="lookup-float-source",
    ),
    pytest.param(
        *_LOOKUP,
        "x__lookup",
        {"lookup_mapping": _INT_MAP, "on_unmatched": "null"},
        {"x": (pa.bool_(), [True, False])},
        "x",
        id="lookup-bool-source",
    ),
    pytest.param(
        *_LOOKUP,
        "x__lookup",
        {"lookup_mapping": _STR_MAP, "on_unmatched": "null"},
        {"x": (pa.date32(), [datetime.date(2020, 1, 1)])},
        "x",
        id="lookup-date-source",
    ),
    pytest.param(
        *_LOOKUP,
        "x__lookup",
        {"lookup_mapping": _INT_MAP, "on_unmatched": "null"},
        {"x": (pa.uint64(), [1, 2])},
        "x",
        id="lookup-uint64-source",
    ),
    pytest.param(
        *_LOOKUP,
        "x__lookup",
        {"lookup_mapping": _STR_MAP, "on_unmatched": "null"},
        {"x": (pa.int64(), [1, 2])},
        "x",
        id="lookup-str-keys-on-int-column",
    ),
    pytest.param(
        *_LOOKUP,
        "x__lookup",
        {"lookup_mapping": _INT_MAP, "on_unmatched": "null"},
        {"x": (pa.string(), ["a"])},
        "x",
        id="lookup-int-keys-on-str-column",
    ),
    pytest.param(
        *_LOOKUP,
        "x__lookup",
        {"lookup_mapping": _STR_MAP, "on_unmatched": "null"},
        {"x": (pa.int64(), [None, None])},
        "x",
        id="lookup-str-keys-on-typed-all-null-int-column",
    ),
    pytest.param(
        *_LOOKUP,
        "x__lookup",
        {"lookup_mapping": _INT_MAP, "on_unmatched": "raise"},
        {"x": (pa.int64(), [1, 2])},
        "1 row, 1 distinct key not in the mapping",
        id="lookup-raise-unmatched",
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


class TestLookupReference:
    def test_unmatched_raise_uses_the_group_error(self) -> None:
        table = _typed_table(x=(pa.int64(), [1, 2, 3, 2, None]))
        features = make_feature_set("x__lookup", lookup_mapping=_INT_MAP, on_unmatched="raise")
        with pytest.raises(LookupFeatureGroup.UNMATCHED_ERROR) as excinfo:
            reference_lookup(LookupFeatureGroup, table, features)
        err = excinfo.value
        assert isinstance(err, LookupFeatureGroup.UNMATCHED_ERROR)
        assert (err.unmatched_rows, err.distinct_keys, err.sample) == (3, 2, (2, 3))

    @pytest.mark.parametrize(
        ("policy", "mapping", "value_type"),
        [
            pytest.param(policy, mapping, value_type, id=f"{policy}-{name}")
            for policy in ("raise", "null", "keep")
            for name, mapping, value_type in (
                ("str", _STR_MAP, pa.string()),
                ("int", _INT_MAP, pa.int64()),
                *((("float", {1: 1.5}, pa.float64()), ("bool", {1: True}, pa.bool_())) if policy != "keep" else ()),
            )
        ],
    )
    def test_null_typed_source_gives_all_null_of_the_value_type(
        self, policy: str, mapping: dict[Any, Any], value_type: pa.DataType
    ) -> None:
        table = _typed_table(x=(pa.null(), [None, None]))
        features = make_feature_set("x__lookup", lookup_mapping=mapping, on_unmatched=policy)
        result = reference_lookup(LookupFeatureGroup, table, features)
        assert result.column("x__lookup").to_pylist() == [None, None]
        assert result.schema.field("x__lookup").type == value_type

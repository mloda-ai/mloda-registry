"""Runs every pinned row of a reference against it, strictly, and guards the tables and guide 03."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import DataType, Options

from mloda.community.feature_groups.data_operations.catalog import DataOperationsCatalog
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.tests.test_framework_support_matrix import (
    OPERATIONS,
    REPO_ROOT,
    splice_into_doc,
)
from mloda.testing.feature_groups.data_operations.helpers import (
    arrow_table_of,
    assert_values_with_nulls,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.pinning import (
    PINNING_BEGIN_MARKER,
    PINNING_END_MARKER,
    PinningRow,
    PinningTable,
    render_pinning_block,
)
from mloda.testing.feature_groups.data_operations.row_preserving.arithmetic.pinning_rows import (
    POINT_ARITHMETIC_PINNING,
    SCALAR_ARITHMETIC_PINNING,
)
from mloda.testing.feature_groups.data_operations.row_preserving.arithmetic.reference import (
    reference_point_arithmetic,
    reference_scalar_arithmetic,
)
from mloda.testing.typed_data import TypedTestData

PINNING_DOC_PATH = REPO_ROOT / "docs" / "guides" / "data-operation-patterns" / "03-reference-implementation.md"

#: Every reference that ships pinning rows: its group, its reference function and its table.
PINNED: list[tuple[type, Callable[..., Any], PinningTable]] = [
    (PointArithmeticFeatureGroup, reference_point_arithmetic, POINT_ARITHMETIC_PINNING),
    (ScalarArithmeticFeatureGroup, reference_scalar_arithmetic, SCALAR_ARITHMETIC_PINNING),
]

ROW_PARAMS = [
    pytest.param(group, reference, table, row, id=f"{table.operation}-{index}-{row.case}")
    for group, reference, table in PINNED
    for index, row in enumerate(table.rows)
]
TABLE_PARAMS = [pytest.param(group, table, id=table.operation) for group, _, table in PINNED]


def typed_data_of(columns: Mapping[str, tuple[DataType, tuple[Any, ...]]]) -> TypedTestData:
    return TypedTestData({name: (data_type, list(values)) for name, (data_type, values) in columns.items()})


@pytest.mark.parametrize(("group", "reference", "table", "row"), ROW_PARAMS)
def test_reference_matches_pinned_row(
    group: type, reference: Callable[..., Any], table: PinningTable, row: PinningRow
) -> None:
    result = reference(group, arrow_table_of(typed_data_of(row.inputs)), make_feature_set(row.feature, **row.context))
    for name, (data_type, values) in row.expected.items():
        assert_values_with_nulls(result.column(name).to_pylist(), list(values), strict=True)
        assert result.schema.field(name).type == DataType.to_arrow_type(data_type), f"column {name} type"


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
            typed_data_of(row.expected)

    def test_operation_is_known_and_matches_the_group(self, group: Any, table: PinningTable) -> None:
        assert table.operation in OPERATIONS
        assert DataOperationsCatalog.get(table.operation).prefix_pattern == group.PREFIX_PATTERN


_PINNING_REGENERATION_HINT = (
    "Regenerate the block between the `BEGIN GENERATED` / `END GENERATED` markers in\n"
    f"{PINNING_DOC_PATH.relative_to(REPO_ROOT)}\n"
    "so it matches render_pinning_block(tables) from the kit, then rerun."
)


def test_pinned_cases_section_is_in_sync() -> None:
    current = PINNING_DOC_PATH.read_text(encoding="utf-8")
    generated = render_pinning_block([table for _, _, table in PINNED])
    expected = splice_into_doc(current, generated, PINNING_BEGIN_MARKER, PINNING_END_MARKER)

    assert expected == current, (
        "The pinned cases section is out of sync with the pinning rows.\n" + _PINNING_REGENERATION_HINT
    )


def _typed_table(**columns: tuple[pa.DataType, list[Any]]) -> pa.Table:
    return pa.table({name: pa.array(values, type=arrow_type) for name, (arrow_type, values) in columns.items()})


class TestArithmeticReferenceRejections:
    """The reference rejects what it cannot compute instead of guessing."""

    def test_unknown_point_op_is_rejected(self) -> None:
        table = _typed_table(x=(pa.int64(), [1]), y=(pa.int64(), [2]))
        with pytest.raises(ValueError, match="bogus"):
            reference_point_arithmetic(PointArithmeticFeatureGroup, table, make_feature_set("x&y__bogus_point"))

    def test_unknown_scalar_op_is_rejected(self) -> None:
        table = _typed_table(x=(pa.int64(), [1]))
        with pytest.raises(ValueError, match="bogus"):
            reference_scalar_arithmetic(
                ScalarArithmeticFeatureGroup, table, make_feature_set("x__bogus_constant", constant=1)
            )

    @pytest.mark.parametrize("arrow_type", [pa.bool_(), pa.string()], ids=["bool", "string"])
    def test_non_numeric_point_source_is_rejected(self, arrow_type: pa.DataType) -> None:
        values: list[Any] = [True] if arrow_type == pa.bool_() else ["1"]
        table = _typed_table(x=(arrow_type, values), y=(pa.int64(), [1]))
        with pytest.raises(ValueError, match="must be numeric"):
            reference_point_arithmetic(PointArithmeticFeatureGroup, table, make_feature_set("x&y__add_point"))

    @pytest.mark.parametrize("arrow_type", [pa.bool_(), pa.string()], ids=["bool", "string"])
    def test_non_numeric_scalar_source_is_rejected(self, arrow_type: pa.DataType) -> None:
        values: list[Any] = [True] if arrow_type == pa.bool_() else ["1"]
        table = _typed_table(x=(arrow_type, values))
        with pytest.raises(ValueError, match="must be numeric"):
            reference_scalar_arithmetic(
                ScalarArithmeticFeatureGroup, table, make_feature_set("x__add_constant", constant=1)
            )

    def test_point_int64_overflow_is_rejected(self) -> None:
        table = _typed_table(x=(pa.int64(), [2**63 - 1]), y=(pa.int64(), [1]))
        with pytest.raises(ValueError):
            reference_point_arithmetic(PointArithmeticFeatureGroup, table, make_feature_set("x&y__add_point"))

    def test_scalar_int64_overflow_is_rejected(self) -> None:
        table = _typed_table(x=(pa.int64(), [2**63 - 1]))
        with pytest.raises(ValueError):
            reference_scalar_arithmetic(
                ScalarArithmeticFeatureGroup, table, make_feature_set("x__add_constant", constant=1)
            )

"""Runs every pinned row of a reference against it, strictly, and guards the tables and guide 03."""

from __future__ import annotations

from collections.abc import Callable
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


@pytest.mark.parametrize(("group", "reference", "table", "row"), ROW_PARAMS)
def test_reference_matches_pinned_row(
    group: type, reference: Callable[..., Any], table: PinningTable, row: PinningRow
) -> None:
    result = reference(group, arrow_table_of(TypedTestData(row.inputs)), make_feature_set(row.feature, **row.context))
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
            TypedTestData(row.expected)

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


_POINT = (reference_point_arithmetic, PointArithmeticFeatureGroup)
_SCALAR = (reference_scalar_arithmetic, ScalarArithmeticFeatureGroup)

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
        None,
        id="point-int64-overflow",
    ),
    pytest.param(
        *_SCALAR, "x__add_constant", {"constant": 1}, {"x": (pa.int64(), [2**63 - 1])}, None, id="scalar-int64-overflow"
    ),
]


class TestArithmeticReferenceRejections:
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

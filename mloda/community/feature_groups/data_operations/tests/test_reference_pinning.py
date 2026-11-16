"""Pinning rows of every reference: hand-computed inputs and expected outputs, run against the reference.

The tables live next to each reference as ``pinning_rows.py`` in ``mloda.testing``; the runner builds each
row's input table, calls ``reference_<operation>`` on the group under test and compares with the strict
comparison of ``assert_values_with_nulls``. The guards keep the tables well formed and complete.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import Options

import mloda.testing.feature_groups.data_operations as kit_package
from mloda.community.feature_groups.data_operations.catalog import _OPERATION_SPECS
from mloda.community.feature_groups.data_operations.tests.test_framework_support_matrix import OPERATIONS
from mloda.testing.feature_groups.data_operations.helpers import assert_values_with_nulls, make_feature_set
from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

KIT_ROOT = Path(kit_package.__file__).resolve().parent
PINNING_SUFFIX = ".pinning_rows"

#: References that ship without rows yet; an entry is removed once its directory gets a ``pinning_rows.py``.
PENDING_REFERENCES = frozenset({"rank", "offset", "percentile"})


def discover_pinning_tables() -> list[tuple[str, PinningTable]]:
    """``(module name, PINNING)`` of every ``pinning_rows`` module in the kit, sorted by module name."""
    names = sorted(
        info.name
        for info in pkgutil.walk_packages(kit_package.__path__, kit_package.__name__ + ".")
        if info.name.endswith(PINNING_SUFFIX)
    )
    return [(name, importlib.import_module(name).PINNING) for name in names]


TABLES = discover_pinning_tables()
ROW_PARAMS = [
    pytest.param(module, table, row, id=f"{table.operation}-{index}-{row.case}")
    for module, table in TABLES
    for index, row in enumerate(table.rows)
]


def group_under_test(operation: str) -> Any:
    """The production FeatureGroup class of a catalog operation, resolved as ``catalog._build_operation`` does."""
    spec = next(spec for spec in _OPERATION_SPECS if spec.name == operation)
    return getattr(importlib.import_module(spec.package + ".base"), spec.base_class)


def reference_of(module: str, operation: str) -> Any:
    """``reference_<operation>`` from the ``reference`` module next to a ``pinning_rows`` module."""
    sibling = module.rsplit(".", 1)[0] + ".reference"
    return getattr(importlib.import_module(sibling), f"reference_{operation}")


def _column(values: tuple[Any, ...], alias: str | None) -> pa.Array:
    return pa.array(list(values), type=pa.type_for_alias(alias) if alias is not None else None)


def input_table(row: PinningRow) -> pa.Table:
    """The row's input columns as a table, typed by ``row.types`` where it names a column."""
    return pa.table({name: _column(values, row.types.get(name)) for name, values in row.inputs.items()})


@pytest.mark.parametrize(("module", "table", "row"), ROW_PARAMS)
def test_reference_matches_pinned_row(module: str, table: PinningTable, row: PinningRow) -> None:
    group = group_under_test(table.operation)
    reference = reference_of(module, table.operation)
    result = reference(group, input_table(row), make_feature_set(row.feature, **row.context))
    for name, expected in row.expected.items():
        assert_values_with_nulls(result.column(name).to_pylist(), list(expected), strict=True)
        if name in row.types:
            assert result.schema.field(name).type == pa.type_for_alias(row.types[name]), f"column {name} type"


def _equal_lengths(columns: Mapping[str, tuple[Any, ...]]) -> bool:
    return len({len(values) for values in columns.values()}) <= 1


@pytest.mark.parametrize(("module", "table"), [pytest.param(m, t, id=t.operation) for m, t in TABLES])
class TestTableGuards:
    def test_operation_is_the_directory_name_and_known(self, module: str, table: PinningTable) -> None:
        assert table.operation == module.split(".")[-2]
        assert table.operation in OPERATIONS

    def test_table_has_rows(self, module: str, table: PinningTable) -> None:
        assert table.rows

    def test_every_case_is_a_rule(self, module: str, table: PinningTable) -> None:
        assert [row.case for row in table.rows if row.case not in table.rules] == []

    def test_columns_have_equal_lengths(self, module: str, table: PinningTable) -> None:
        bad = [
            i for i, row in enumerate(table.rows) if not (_equal_lengths(row.inputs) and _equal_lengths(row.expected))
        ]
        assert bad == []

    def test_every_feature_is_requestable(self, module: str, table: PinningTable) -> None:
        group = group_under_test(table.operation)
        bad = [
            row.feature
            for row in table.rows
            if not group.match_feature_group_criteria(row.feature, Options(context=dict(row.context)))
        ]
        assert bad == []

    def test_every_rule_is_used_by_a_table_sharing_it(self, module: str, table: PinningTable) -> None:
        used = {row.case for _, other in TABLES if dict(other.rules) == dict(table.rules) for row in other.rows}
        assert sorted(set(table.rules) - used) == []


def _directories(file_name: str) -> set[str]:
    return {path.parent.name for path in KIT_ROOT.rglob(file_name) if "__pycache__" not in path.parts}


def test_every_reference_has_pinning_rows_or_is_pending() -> None:
    with_reference = _directories("reference.py")
    with_rows = _directories("pinning_rows.py")
    assert sorted(with_reference - with_rows - PENDING_REFERENCES) == []


def test_pending_references_hold_no_directory_with_rows() -> None:
    assert sorted(PENDING_REFERENCES & _directories("pinning_rows.py")) == []

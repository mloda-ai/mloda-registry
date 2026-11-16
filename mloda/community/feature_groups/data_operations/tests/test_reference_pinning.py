"""Runs every ``pinning_rows.py`` table in ``mloda.testing`` against its reference, strictly, and guards the tables."""

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
from mloda.community.feature_groups.data_operations.tests.test_framework_support_matrix import (
    OPERATIONS,
    REPO_ROOT,
    splice_into_doc,
)
from mloda.testing.feature_groups.data_operations.helpers import assert_values_with_nulls, make_feature_set
from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

KIT_ROOT = Path(kit_package.__file__).resolve().parent
PINNING_SUFFIX = ".pinning_rows"

PINNING_DOC_PATH = REPO_ROOT / "docs" / "guides" / "data-operation-patterns" / "03-reference-implementation.md"
PINNING_BEGIN_MARKER = "<!-- BEGIN GENERATED: reference-pinning -->"
PINNING_END_MARKER = "<!-- END GENERATED: reference-pinning -->"

#: References that ship without rows yet; an entry is removed once its directory gets a ``pinning_rows.py``.
PENDING_REFERENCES = frozenset({"rank", "offset", "percentile"})


def discover_pinning_tables() -> list[tuple[str, PinningTable]]:
    """``(module name, PINNING)`` of every ``pinning_rows`` module, sorted by name."""
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
    """The production FeatureGroup class of a catalog operation."""
    spec = next(spec for spec in _OPERATION_SPECS if spec.name == operation)
    return getattr(importlib.import_module(spec.package + ".base"), spec.base_class)


def reference_of(module: str, operation: str) -> Any:
    """``reference_<operation>`` from the ``reference`` module beside a ``pinning_rows`` module."""
    sibling = module.rsplit(".", 1)[0] + ".reference"
    return getattr(importlib.import_module(sibling), f"reference_{operation}")


def _column(values: tuple[Any, ...], alias: str | None) -> pa.Array:
    return pa.array(list(values), type=pa.type_for_alias(alias) if alias is not None else None)


def input_table(row: PinningRow) -> pa.Table:
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


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _value(value: Any) -> str:
    if isinstance(value, float) and value != value:
        return "NaN"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _context_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    return _value(value)


def _columns(columns: Mapping[str, tuple[Any, ...]], types: Mapping[str, str]) -> str:
    parts = []
    for name, values in columns.items():
        alias = f" ({types[name]})" if name in types else ""
        parts.append(f"{name}=[{', '.join(_value(v) for v in values)}]{alias}")
    return "; ".join(parts)


def render_pinning_block(tables: list[PinningTable]) -> str:
    """The generated block of guide 03: the rule tables, then the pinned rows of every operation."""
    for table in tables:
        if table.operation not in OPERATIONS:
            raise ValueError(f"Unknown operation {table.operation!r}; expected one of {OPERATIONS}.")
    ordered = sorted((t for t in tables if t.rows), key=lambda t: OPERATIONS.index(t.operation))
    out: list[str] = [PINNING_BEGIN_MARKER, ""]
    out.append("Rendered from each reference's `pinning_rows.py`; every row is run against its reference.")
    out.append("")
    groups: list[list[PinningTable]] = []
    for table in ordered:
        for group in groups:
            if dict(group[0].rules) == dict(table.rules):
                group.append(table)
                break
        else:
            groups.append([table])
    for group in groups:
        out.append(f"### Rules ({', '.join(t.operation for t in group)})")
        out.append("")
        out.append("| Case | Rule | Pinned in |")
        out.append("|---|---|---|")
        for label, rule in group[0].rules.items():
            users = ", ".join(t.operation for t in group if any(row.case == label for row in t.rows))
            out.append(f"| {_cell(label)} | {_cell(rule)} | {users} |")
        out.append("")
    for table in ordered:
        out.append(f"### {table.operation}")
        out.append("")
        out.append("| Case | Feature | Context | Input | Expected |")
        out.append("|---|---|---|---|---|")
        for row in table.rows:
            context = ", ".join(f"{k}={_context_value(v)}" for k, v in row.context.items()) or "-"
            cells = [
                row.case,
                f"`{row.feature}`",
                context,
                _columns(row.inputs, row.types),
                _columns(row.expected, row.types),
            ]
            out.append("| " + " | ".join(_cell(c) for c in cells) + " |")
        out.append("")
    out.append(PINNING_END_MARKER)
    return "\n".join(out) + "\n"


def test_render_pinning_block_rules_and_rows() -> None:
    rules = {"none": "Skips None.", "nan": "NaN is a value."}
    aggregation = PinningTable(
        operation="aggregation",
        rules=rules,
        rows=(
            PinningRow(
                case="none",
                feature="v__sum_aggr",
                inputs={"k": ("a|b", "c"), "v": (1, None)},
                expected={"v__sum_aggr": (1, None)},
                context={"partition_by": ["k", "j"], "ddof": 1},
                types={"v": "int32"},
            ),
            PinningRow(
                case="nan",
                feature="v__max_aggr",
                inputs={"v": (float("nan"), -0.0)},
                expected={"v__max_aggr": (float("nan"),)},
            ),
        ),
    )
    scalar = PinningTable(
        operation="scalar_aggregate",
        rules=dict(rules),
        rows=(
            PinningRow(case="nan", feature="v__min_scalar", inputs={"v": (0.0,)}, expected={"v__min_scalar": (0.0,)}),
        ),
    )
    empty = PinningTable(operation="binning", rules={"x": "unused"}, rows=())

    # Tables are passed out of OPERATIONS order; the render must reorder them.
    lines = render_pinning_block([scalar, empty, aggregation]).split("\n")

    assert lines == [
        PINNING_BEGIN_MARKER,
        "",
        "Rendered from each reference's `pinning_rows.py`; every row is run against its reference.",
        "",
        "### Rules (aggregation, scalar_aggregate)",
        "",
        "| Case | Rule | Pinned in |",
        "|---|---|---|",
        "| none | Skips None. | aggregation |",
        "| nan | NaN is a value. | aggregation, scalar_aggregate |",
        "",
        "### aggregation",
        "",
        "| Case | Feature | Context | Input | Expected |",
        "|---|---|---|---|---|",
        "| none | `v__sum_aggr` | partition_by=[k, j], ddof=1 | k=[a\\|b, c]; v=[1, None] (int32) | v__sum_aggr=[1, None] |",
        "| nan | `v__max_aggr` | - | v=[NaN, -0.0] | v__max_aggr=[NaN] |",
        "",
        "### scalar_aggregate",
        "",
        "| Case | Feature | Context | Input | Expected |",
        "|---|---|---|---|---|",
        "| nan | `v__min_scalar` | - | v=[0.0] | v__min_scalar=[0.0] |",
        "",
        PINNING_END_MARKER,
        "",
    ]


_PINNING_REGENERATION_HINT = (
    "Regenerate the block between the `BEGIN GENERATED` / `END GENERATED` markers in\n"
    f"{PINNING_DOC_PATH.relative_to(REPO_ROOT)}\n"
    "so it matches render_pinning_block(tables) from this module, then rerun."
)


def test_pinned_cases_section_is_in_sync() -> None:
    current = PINNING_DOC_PATH.read_text(encoding="utf-8")
    generated = render_pinning_block([table for _, table in TABLES])
    expected = splice_into_doc(current, generated, PINNING_BEGIN_MARKER, PINNING_END_MARKER)

    assert expected == current, (
        "The pinned cases section is out of sync with the pinning rows.\n" + _PINNING_REGENERATION_HINT
    )

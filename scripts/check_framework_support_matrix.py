#!/usr/bin/env python3
"""Generate and drift-check the framework support matrix doc.

The matrix page lists, per data operation and per framework, which subtypes
(agg_type / offset_type / rank_type / op) are supported. Its source of truth
is the ``supported_*()`` classmethods on the framework test classes under
``mloda/community/feature_groups/data_operations/**/tests/test_{framework}.py``.

Usage:
    python scripts/check_framework_support_matrix.py          # regenerate the doc
    python scripts/check_framework_support_matrix.py --check  # CI drift check

In ``--check`` mode the script exits non-zero if the generated block inside
``docs/guides/data-operation-patterns/framework-support-matrix.md`` does not
match what the current ``supported_*()`` sets produce.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "guides" / "data-operation-patterns" / "framework-support-matrix.md"

BEGIN_MARKER = "<!-- BEGIN GENERATED: framework-support-matrix -->"
END_MARKER = "<!-- END GENERATED: framework-support-matrix -->"

FRAMEWORKS: list[tuple[str, str]] = [
    ("pyarrow", "PyArrow"),
    ("pandas", "Pandas"),
    ("polars_lazy", "Polars lazy"),
    ("duckdb", "DuckDB"),
    ("sqlite", "SQLite"),
]

SUPPORT_METHODS = (
    "supported_agg_types",
    "supported_ops",
    "supported_offset_types",
    "supported_rank_types",
)


@dataclass(frozen=True)
class OperationSpec:
    """Where to find an operation's test classes and its full subtype set."""

    key: str
    display: str
    tests_pkg: str
    base_module: str
    base_class: str
    subtype_label: str
    order_hint: tuple[str, ...] = ()


OPERATIONS: list[OperationSpec] = [
    OperationSpec(
        key="aggregation",
        display="aggregation",
        tests_pkg="mloda.community.feature_groups.data_operations.aggregation.tests",
        base_module="mloda.testing.feature_groups.data_operations.aggregation.aggregation",
        base_class="AggregationTestBase",
        subtype_label="agg type",
        order_hint=(
            "sum",
            "avg",
            "mean",
            "count",
            "min",
            "max",
            "std",
            "var",
            "std_pop",
            "std_samp",
            "var_pop",
            "var_samp",
            "median",
            "mode",
            "nunique",
            "first",
            "last",
        ),
    ),
    OperationSpec(
        key="binning",
        display="binning",
        tests_pkg="mloda.community.feature_groups.data_operations.row_preserving.binning.tests",
        base_module="mloda.testing.feature_groups.data_operations.row_preserving.binning.binning",
        base_class="BinningTestBase",
        subtype_label="op",
        order_hint=("bin", "qbin"),
    ),
    OperationSpec(
        key="datetime",
        display="datetime",
        tests_pkg="mloda.community.feature_groups.data_operations.row_preserving.datetime.tests",
        base_module="mloda.testing.feature_groups.data_operations.row_preserving.datetime.datetime",
        base_class="DateTimeTestBase",
        subtype_label="op",
    ),
    OperationSpec(
        key="frame_aggregate",
        display="frame_aggregate",
        tests_pkg="mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.tests",
        base_module="mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate",
        base_class="FrameAggregateTestBase",
        subtype_label="frame type",
    ),
    OperationSpec(
        key="offset",
        display="offset",
        tests_pkg="mloda.community.feature_groups.data_operations.row_preserving.offset.tests",
        base_module="mloda.testing.feature_groups.data_operations.row_preserving.offset.offset",
        base_class="OffsetTestBase",
        subtype_label="offset type",
        order_hint=("lag", "lead", "diff", "pct_change", "first_value", "last_value"),
    ),
    OperationSpec(
        key="percentile",
        display="percentile",
        tests_pkg="mloda.community.feature_groups.data_operations.row_preserving.percentile.tests",
        base_module="mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile",
        base_class="PercentileTestBase",
        subtype_label="op",
    ),
    OperationSpec(
        key="rank",
        display="rank",
        tests_pkg="mloda.community.feature_groups.data_operations.row_preserving.rank.tests",
        base_module="mloda.testing.feature_groups.data_operations.row_preserving.rank.rank",
        base_class="RankTestBase",
        subtype_label="rank type",
        order_hint=("row_number", "rank", "dense_rank", "percent_rank"),
    ),
    OperationSpec(
        key="scalar_aggregate",
        display="scalar_aggregate",
        tests_pkg="mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.tests",
        base_module="mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate",
        base_class="ScalarAggregateTestBase",
        subtype_label="agg type",
        order_hint=(
            "sum",
            "avg",
            "mean",
            "count",
            "min",
            "max",
            "std",
            "var",
            "std_pop",
            "std_samp",
            "var_pop",
            "var_samp",
            "median",
        ),
    ),
    OperationSpec(
        key="window_aggregation",
        display="window_aggregation",
        tests_pkg="mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.tests",
        base_module="mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation",
        base_class="WindowAggregationTestBase",
        subtype_label="agg type",
        order_hint=(
            "sum",
            "avg",
            "mean",
            "count",
            "min",
            "max",
            "std",
            "var",
            "std_pop",
            "std_samp",
            "var_pop",
            "var_samp",
            "median",
            "mode",
            "nunique",
            "first",
            "last",
        ),
    ),
    OperationSpec(
        key="string",
        display="string",
        tests_pkg="mloda.community.feature_groups.data_operations.string.tests",
        base_module="mloda.testing.feature_groups.data_operations.string.string",
        base_class="StringTestBase",
        subtype_label="op",
        order_hint=("upper", "lower", "trim", "length", "reverse"),
    ),
]


def import_test_class(tests_pkg: str, framework: str, base_cls: type) -> type | None:
    """Import test_{framework}.py under *tests_pkg* and return the concrete test class.

    The concrete class is the one module-local subclass of *base_cls* (or ``None``
    if the module does not exist)."""
    mod_name = f"{tests_pkg}.test_{framework}"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        return None
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != mod.__name__:
            continue
        if not issubclass(obj, base_cls):
            continue
        if obj is base_cls:
            continue
        return obj
    return None


def supported_set(cls: type) -> tuple[str | None, set[str] | None]:
    """Return (method_name, set) for the first supported_* method defined on *cls*."""
    for attr in SUPPORT_METHODS:
        method = getattr(cls, attr, None)
        if method is None:
            continue
        try:
            return attr, set(method())
        except Exception:  # pragma: no cover - defensive
            return attr, None
    return None, None


def sort_subtypes(items: Iterable[str], order_hint: tuple[str, ...]) -> list[str]:
    items = list(items)
    index = {name: pos for pos, name in enumerate(order_hint)}
    items.sort(key=lambda name: (index.get(name, len(order_hint)), name))
    return items


def collect_operation(op: OperationSpec) -> dict[str, Any]:
    base_mod = importlib.import_module(op.base_module)
    base_cls = getattr(base_mod, op.base_class)

    # Resolve the framework-level data.
    per_framework: dict[str, dict[str, Any]] = {}
    union_subtypes: set[str] = set()
    method_name: str | None = None

    for fw_key, _fw_label in FRAMEWORKS:
        cls = import_test_class(op.tests_pkg, fw_key, base_cls)
        if cls is None:
            per_framework[fw_key] = {"present": False, "subtypes": None, "method": None}
            continue
        attr, support = supported_set(cls)
        per_framework[fw_key] = {"present": True, "subtypes": support, "method": attr}
        if support is not None:
            union_subtypes.update(support)
            if method_name is None:
                method_name = attr

    if not union_subtypes:
        # Operations with no supported_*() method (datetime, percentile,
        # frame_aggregate) report at operation-level granularity only.
        subtypes: list[str] | None = None
    else:
        subtypes = sort_subtypes(union_subtypes, op.order_hint)

    return {
        "op": op,
        "method": method_name,
        "subtypes": subtypes,
        "frameworks": per_framework,
    }


def render_summary_table(collected: list[dict[str, Any]]) -> list[str]:
    header = "| Operation | " + " | ".join(label for _, label in FRAMEWORKS) + " |"
    sep = "|" + "|".join(["---"] * (len(FRAMEWORKS) + 1)) + "|"
    lines = [header, sep]
    for item in collected:
        op = item["op"]
        subtypes = item["subtypes"]
        cells: list[str] = [op.display]
        for fw_key, _label in FRAMEWORKS:
            fw = item["frameworks"][fw_key]
            if not fw["present"]:
                cells.append("--")
            elif subtypes is None:
                cells.append("full")
            elif fw["subtypes"] is None or set(fw["subtypes"]) == set(subtypes):
                cells.append("full")
            else:
                count = len(fw["subtypes"])
                total = len(subtypes)
                cells.append(f"partial ({count}/{total})")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_detail_table(item: dict[str, Any]) -> list[str]:
    op: OperationSpec = item["op"]
    subtypes = item["subtypes"]
    header_cells = [op.subtype_label.capitalize()] + [label for _, label in FRAMEWORKS]
    header = "| " + " | ".join(header_cells) + " |"
    sep = "|" + "|".join(["---"] * len(header_cells)) + "|"
    lines = [f"### {op.display}", ""]

    if subtypes is None:
        # Single-row table: either the framework has a test class or it does not.
        row = ["(all)"]
        for fw_key, _label in FRAMEWORKS:
            fw = item["frameworks"][fw_key]
            row.append("\u2713" if fw["present"] else "\u2717")
        lines += [header, sep, "| " + " | ".join(row) + " |"]
        return lines

    lines += [header, sep]
    for subtype in subtypes:
        row = [f"`{subtype}`"]
        for fw_key, _label in FRAMEWORKS:
            fw = item["frameworks"][fw_key]
            if not fw["present"]:
                row.append("--")
                continue
            if fw["subtypes"] is None:
                row.append("\u2713")
                continue
            row.append("\u2713" if subtype in fw["subtypes"] else "\u2717")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_generated_block(collected: list[dict[str, Any]]) -> str:
    out: list[str] = [BEGIN_MARKER, ""]
    out.append("## Summary")
    out.append("")
    out.append(
        "`full` means every subtype listed in the per-operation table below is supported. "
        "`partial (k/n)` means the framework's test class restricts `supported_*()` to "
        "k of the n subtypes this operation defines. `--` means the framework has no test "
        "class for this operation (typically because no production implementation exists)."
    )
    out.append("")
    out.extend(render_summary_table(collected))
    out.append("")
    out.append("## Per-operation detail")
    out.append("")
    out.append(
        "\u2713 = the framework's test-class `supported_*()` includes this subtype. "
        "\u2717 = excluded. `--` = no test class for this framework."
    )
    out.append("")
    for item in collected:
        out.extend(render_detail_table(item))
        out.append("")
    out.append(END_MARKER)
    return "\n".join(out) + "\n"


def splice_into_doc(doc_text: str, generated: str) -> str:
    if BEGIN_MARKER not in doc_text or END_MARKER not in doc_text:
        raise RuntimeError(f"Markers missing from {DOC_PATH}. Expected both {BEGIN_MARKER!r} and {END_MARKER!r}.")
    begin = doc_text.index(BEGIN_MARKER)
    end = doc_text.index(END_MARKER) + len(END_MARKER)
    # Preserve any trailing newline right after END_MARKER.
    trailing = doc_text[end:]
    return doc_text[:begin] + generated.rstrip("\n") + trailing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the doc is out of sync with supported_*() sets.",
    )
    args = parser.parse_args()

    collected = [collect_operation(op) for op in OPERATIONS]
    generated = render_generated_block(collected)

    current = DOC_PATH.read_text()
    new_doc = splice_into_doc(current, generated)

    if args.check:
        if new_doc == current:
            return 0
        diff = difflib.unified_diff(
            current.splitlines(keepends=True),
            new_doc.splitlines(keepends=True),
            fromfile=str(DOC_PATH) + " (on disk)",
            tofile=str(DOC_PATH) + " (expected)",
        )
        sys.stderr.write("".join(diff))
        sys.stderr.write(
            "\n\nFramework support matrix is out of sync with supported_*() sets.\n"
            "Run: python scripts/check_framework_support_matrix.py\n"
        )
        return 1

    if new_doc != current:
        DOC_PATH.write_text(new_doc)
        print(f"Updated {DOC_PATH.relative_to(REPO_ROOT)}")
    else:
        print(f"{DOC_PATH.relative_to(REPO_ROOT)} already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())

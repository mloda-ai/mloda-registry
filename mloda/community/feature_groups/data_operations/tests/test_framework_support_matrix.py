"""Drift check: the framework support matrix doc must match DataOperationsCatalog.

The source of truth for ``docs/guides/data-operation-patterns/framework-support-matrix.md``
is the production capability declared by the concrete classes (``compute_framework_rule``,
the ``supports_compute_framework`` hook, and match-time restrictions), queried via
``DataOperationsCatalog``. When ``test_framework_support_matrix_is_in_sync`` fails,
regenerate the block between the ``BEGIN GENERATED`` / ``END GENERATED`` markers so it
matches what this module renders, then rerun.
"""

from __future__ import annotations

from pathlib import Path

from mloda.community.feature_groups.data_operations import DataOperationsCatalog, OperationInfo

REPO_ROOT = Path(__file__).resolve().parents[5]
DOC_PATH = REPO_ROOT / "docs" / "guides" / "data-operation-patterns" / "framework-support-matrix.md"
DATA_OPERATIONS_ROOT = REPO_ROOT / "mloda" / "community" / "feature_groups" / "data_operations"

BEGIN_MARKER = "<!-- BEGIN GENERATED: framework-support-matrix -->"
END_MARKER = "<!-- END GENERATED: framework-support-matrix -->"

#: Doc column order: (framework key, doc column label).
FRAMEWORKS: list[tuple[str, str]] = [
    ("pyarrow", "PyArrow"),
    ("pandas", "Pandas"),
    ("polars_lazy", "Polars lazy"),
    ("duckdb", "DuckDB"),
    ("sqlite", "SQLite"),
]

#: Catalog framework key (``OperationInfo.frameworks``) per framework key.
FRAMEWORK_CATALOG_KEYS: dict[str, str] = {
    "pyarrow": "PyArrowTable",
    "pandas": "PandasDataFrame",
    "polars_lazy": "PolarsLazyDataFrame",
    "duckdb": "DuckDBFramework",
    "sqlite": "SqliteFramework",
}

#: Doc display order for operations. ``DataOperationsCatalog.list()`` is
#: name-sorted; the doc keeps its historical grouping instead.
OPERATIONS: list[str] = [
    "aggregation",
    "binning",
    "datetime",
    "frame_aggregate",
    "offset",
    "percentile",
    "rank",
    "scalar_aggregate",
    "scalar_arithmetic",
    "point_arithmetic",
    "time_bucketization",
    "ffill",
    "ema",
    "sessionization",
    "window_aggregation",
    "string",
    "resample",
]


def ordered_operations() -> list[OperationInfo]:
    """Catalog entries in doc display order; fails loudly on any name mismatch."""
    infos = {info.name: info for info in DataOperationsCatalog.list()}
    missing = sorted(name for name in OPERATIONS if name not in infos)
    extra = sorted(name for name in infos if name not in OPERATIONS)
    if missing or extra:
        raise RuntimeError(f"OPERATIONS is out of sync with DataOperationsCatalog: missing={missing} extra={extra}")
    return [infos[name] for name in OPERATIONS]


def _framework_supported(info: OperationInfo, fw_key: str) -> frozenset[str] | None:
    """The catalog's supported-subtype set for *fw_key*, or None when absent or subtype-less."""
    return info.frameworks.get(FRAMEWORK_CATALOG_KEYS[fw_key])


def render_summary_table(infos: list[OperationInfo]) -> list[str]:
    header = "| Operation | " + " | ".join(label for _, label in FRAMEWORKS) + " |"
    sep = "|" + "|".join(["---"] * (len(FRAMEWORKS) + 1)) + "|"
    lines = [header, sep]
    for info in infos:
        cells: list[str] = [info.name]
        for fw_key, _label in FRAMEWORKS:
            if FRAMEWORK_CATALOG_KEYS[fw_key] not in info.frameworks:
                cells.append("--")
                continue
            supported = _framework_supported(info, fw_key)
            if info.subtypes is None or supported is None or set(supported) == set(info.subtypes):
                cells.append("full")
            else:
                cells.append(f"partial ({len(supported)}/{len(info.subtypes)})")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_detail_table(info: OperationInfo) -> list[str]:
    header_cells = [info.subtype_label.capitalize()] + [label for _, label in FRAMEWORKS]
    header = "| " + " | ".join(header_cells) + " |"
    sep = "|" + "|".join(["---"] * len(header_cells)) + "|"
    lines = [f"### {info.name}", ""]

    if info.subtypes is None:
        # Single-row table: either the framework ships an implementation or it does not.
        row = ["(all)"]
        for fw_key, _label in FRAMEWORKS:
            row.append("✓" if FRAMEWORK_CATALOG_KEYS[fw_key] in info.frameworks else "--")
        lines += [header, sep, "| " + " | ".join(row) + " |"]
        return lines

    lines += [header, sep]
    for subtype in info.subtypes:
        row = [f"`{subtype}`"]
        for fw_key, _label in FRAMEWORKS:
            if FRAMEWORK_CATALOG_KEYS[fw_key] not in info.frameworks:
                row.append("--")
                continue
            supported = _framework_supported(info, fw_key)
            row.append("✓" if supported is None or subtype in supported else "✗")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_generated_block(infos: list[OperationInfo]) -> str:
    out: list[str] = [BEGIN_MARKER, ""]
    out.append("## Summary")
    out.append("")
    out.append(
        "Cells reflect the production capability declarations (`compute_framework_rule`, the "
        "`supports_compute_framework` hook, and match-time restrictions), queryable via "
        "`DataOperationsCatalog`. `full` means the framework's production implementation declares "
        "support for every subtype this operation defines. `partial (k/n)` means it declares k of "
        "the n subtypes and rejects the rest. `--` means no implementation ships for this framework."
    )
    out.append("")
    out.extend(render_summary_table(infos))
    out.append("")
    out.append("## Per-operation detail")
    out.append("")
    out.append(
        "✓ = the framework's production implementation declares support for this subtype. "
        "✗ = the implementation rejects it. `--` = no implementation ships for this framework."
    )
    out.append("")
    for info in infos:
        out.extend(render_detail_table(info))
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


def discover_operation_dirs_on_disk() -> set[str]:
    """Operation directory names that carry per-framework twin test modules.

    A directory counts when its ``tests`` subpackage contains at least one
    ``test_<framework>.py`` file. Guards against a new data operation landing
    on disk without a catalog entry.
    """
    ops: set[str] = set()
    if not DATA_OPERATIONS_ROOT.exists():
        return ops
    for tests_dir in DATA_OPERATIONS_ROOT.rglob("tests"):
        if not tests_dir.is_dir():
            continue
        if not any((tests_dir / f"test_{fw_key}.py").exists() for fw_key, _label in FRAMEWORKS):
            continue
        ops.add(tests_dir.parent.name)
    return ops


_REGENERATION_HINT = (
    "Regenerate the block between the `BEGIN GENERATED` / `END GENERATED` markers in\n"
    f"{DOC_PATH.relative_to(REPO_ROOT)}\n"
    "so it matches render_generated_block(ordered_operations()) from this module, then rerun."
)


def test_framework_support_matrix_is_in_sync() -> None:
    generated = render_generated_block(ordered_operations())

    current = DOC_PATH.read_text()
    expected = splice_into_doc(current, generated)

    assert expected == current, (
        "framework-support-matrix.md is out of sync with DataOperationsCatalog.\n" + _REGENERATION_HINT
    )


def test_operations_list_covers_every_data_operation_on_disk() -> None:
    catalog_names = {info.name for info in DataOperationsCatalog.list()}
    assert sorted(OPERATIONS) == sorted(catalog_names), (
        f"OPERATIONS does not match DataOperationsCatalog.list() names: "
        f"missing={sorted(catalog_names - set(OPERATIONS))} extra={sorted(set(OPERATIONS) - catalog_names)}"
    )
    uncovered = sorted(discover_operation_dirs_on_disk() - catalog_names)
    assert uncovered == [], (
        "DataOperationsCatalog is missing entries for these data-operation directories "
        "(each has test_<framework>.py twin files):\n  " + "\n  ".join(uncovered)
    )

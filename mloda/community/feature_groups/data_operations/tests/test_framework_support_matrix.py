"""Drift check: the framework support matrix doc must match DataOperationsCatalog.

The source of truth for ``docs/guides/data-operation-patterns/framework-support-matrix.md``
is the production capability declared by the concrete classes (``compute_framework_rule``,
the ``supports_compute_framework`` hook, and match-time restrictions), queried via
``DataOperationsCatalog``. When ``test_framework_support_matrix_is_in_sync`` fails,
regenerate the block between the ``BEGIN GENERATED`` / ``END GENERATED`` markers so it
matches what this module renders, then rerun.

Both axes of the rendered block come from the catalog rather than from a table kept
here: the columns are ``catalog.FRAMEWORKS`` in order, and the rows are
``operations_in_declaration_order()`` (the ``FAMILY_BASE_MODULES`` registry order).
"""

from __future__ import annotations

from pathlib import Path

from mloda.community.feature_groups.data_operations import OperationInfo
from mloda.community.feature_groups.data_operations.catalog import (
    FRAMEWORKS,
    FrameworkInfo,
    operations_in_declaration_order,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DOC_PATH = REPO_ROOT / "docs" / "guides" / "data-operation-patterns" / "framework-support-matrix.md"
DATA_OPERATIONS_ROOT = REPO_ROOT / "mloda" / "community" / "feature_groups" / "data_operations"

BEGIN_MARKER = "<!-- BEGIN GENERATED: framework-support-matrix -->"
END_MARKER = "<!-- END GENERATED: framework-support-matrix -->"

#: Build artifact directory names skipped by tree walks; scripts/generate_pyproject.py excludes these too.
ARTIFACT_DIR_PARTS = frozenset({"build", "dist", ".tox", ".venv", "__pycache__"})


def is_artifact_path(rel_path: Path) -> bool:
    """True when any part of *rel_path* is a build artifact directory (or an ``*.egg-info``)."""
    return any(part in ARTIFACT_DIR_PARTS or part.endswith(".egg-info") for part in rel_path.parts)


def _framework_supported(info: OperationInfo, framework: FrameworkInfo) -> frozenset[str] | None:
    """The catalog's supported-subtype set for *framework*, or None when absent or subtype-less."""
    return info.frameworks.get(framework.catalog_key)


def render_summary_table(infos: tuple[OperationInfo, ...]) -> list[str]:
    header = "| Operation | " + " | ".join(framework.label for framework in FRAMEWORKS) + " |"
    sep = "|" + "|".join(["---"] * (len(FRAMEWORKS) + 1)) + "|"
    lines = [header, sep]
    for info in infos:
        cells: list[str] = [info.name]
        for framework in FRAMEWORKS:
            if framework.catalog_key not in info.frameworks:
                cells.append("--")
                continue
            supported = _framework_supported(info, framework)
            if info.subtypes is None or supported is None or set(supported) == set(info.subtypes):
                cells.append("full")
            else:
                cells.append(f"partial ({len(supported)}/{len(info.subtypes)})")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_detail_table(info: OperationInfo) -> list[str]:
    header_cells = [info.subtype_label.capitalize()] + [framework.label for framework in FRAMEWORKS]
    header = "| " + " | ".join(header_cells) + " |"
    sep = "|" + "|".join(["---"] * len(header_cells)) + "|"
    lines = [f"### {info.name}", ""]

    if info.subtypes is None:
        # Single-row table: either the framework ships an implementation or it does not.
        row = ["(all)"]
        for framework in FRAMEWORKS:
            row.append("✓" if framework.catalog_key in info.frameworks else "--")
        lines += [header, sep, "| " + " | ".join(row) + " |"]
        return lines

    lines += [header, sep]
    for subtype in info.subtypes:
        row = [f"`{subtype}`"]
        for framework in FRAMEWORKS:
            if framework.catalog_key not in info.frameworks:
                row.append("--")
                continue
            supported = _framework_supported(info, framework)
            row.append("✓" if supported is None or subtype in supported else "✗")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_generated_block(infos: tuple[OperationInfo, ...]) -> str:
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


def discover_operation_dirs_on_disk(root: Path = DATA_OPERATIONS_ROOT) -> set[str]:
    """Operation directory names that carry per-framework twin test modules.

    A directory counts when its ``tests`` subpackage contains at least one
    ``test_<framework>.py`` file. Guards against a new data operation landing
    on disk without a catalog entry. Build artifact copies are skipped.
    """
    ops: set[str] = set()
    if not root.exists():
        return ops
    for tests_dir in root.rglob("tests"):
        if not tests_dir.is_dir():
            continue
        if is_artifact_path(tests_dir.relative_to(root)):
            continue
        if not any((tests_dir / f"test_{framework.module_prefix}.py").exists() for framework in FRAMEWORKS):
            continue
        ops.add(tests_dir.parent.name)
    return ops


_REGENERATION_HINT = (
    "Regenerate the block between the `BEGIN GENERATED` / `END GENERATED` markers in\n"
    f"{DOC_PATH.relative_to(REPO_ROOT)}\n"
    "so it matches render_generated_block(operations_in_declaration_order()) from this module, then rerun."
)


def test_framework_support_matrix_is_in_sync() -> None:
    generated = render_generated_block(operations_in_declaration_order())

    current = DOC_PATH.read_text()
    expected = splice_into_doc(current, generated)

    assert expected == current, (
        "framework-support-matrix.md is out of sync with DataOperationsCatalog.\n" + _REGENERATION_HINT
    )


def test_catalog_covers_every_data_operation_on_disk() -> None:
    catalog_names = {info.name for info in operations_in_declaration_order()}
    on_disk = discover_operation_dirs_on_disk()
    uncovered = sorted(on_disk - catalog_names)
    assert uncovered == [], (
        "DataOperationsCatalog is missing entries for these data-operation directories "
        "(each has test_<framework>.py twin files):\n  " + "\n  ".join(uncovered)
    )
    unbacked = sorted(catalog_names - on_disk)
    assert unbacked == [], (
        "These DataOperationsCatalog entries have no twin test directories on disk (an over-broad "
        "build-artifact skip in discover_operation_dirs_on_disk would hide them):\n  " + "\n  ".join(unbacked)
    )


def test_discover_operation_dirs_on_disk_counts_planted_operation(tmp_path: Path) -> None:
    """A planted <op>/tests/test_<framework>.py twin file counts its operation directory."""
    twin = tmp_path / "my_op" / "tests" / "test_pandas.py"
    twin.parent.mkdir(parents=True)
    twin.write_text("")
    assert discover_operation_dirs_on_disk(root=tmp_path) == {"my_op"}


def test_discover_operation_dirs_on_disk_skips_build_artifact_dirs(tmp_path: Path) -> None:
    """A twin-file tree under build/lib is not counted as an operation directory."""
    copy = tmp_path / "build" / "lib" / "my_op" / "tests" / "test_pandas.py"
    copy.parent.mkdir(parents=True)
    copy.write_text("")
    assert discover_operation_dirs_on_disk(root=tmp_path) == set()

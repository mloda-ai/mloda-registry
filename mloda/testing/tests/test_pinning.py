"""Tests for the pinning format and its markdown rendering (mloda.testing)."""

from __future__ import annotations

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import (
    PINNING_BEGIN_MARKER,
    PINNING_END_MARKER,
    PinningRow,
    PinningTable,
    render_pinning_block,
)

_INTRO = "Rendered from each reference's `pinning_rows.py`; every row is run against its reference."


def test_markers_are_the_reference_pinning_comments() -> None:
    assert PINNING_BEGIN_MARKER == "<!-- BEGIN GENERATED: reference-pinning -->"
    assert PINNING_END_MARKER == "<!-- END GENERATED: reference-pinning -->"


def test_render_pinning_block_rules_and_rows() -> None:
    rules = {"none": "Skips None.", "nan": "NaN is a value."}
    first = PinningTable(
        operation="second_op",
        rules=rules,
        rows=(
            PinningRow(
                case="none",
                feature="v__sum_aggr",
                inputs={"k": (DataType.STRING, ("a|b", "c")), "v": (DataType.INT64, (1, None))},
                expected={"v__sum_aggr": (DataType.INT64, (1, None))},
                context={"partition_by": ["k", "j"], "ddof": 1},
            ),
            PinningRow(
                case="nan",
                feature="v__max_aggr",
                inputs={"v": (DataType.DOUBLE, (float("nan"), -0.0, float("inf"), float("-inf")))},
                expected={"v__max_aggr": (DataType.DOUBLE, (float("nan"),))},
            ),
        ),
    )
    second = PinningTable(
        operation="first_op",
        rules=dict(rules),
        rows=(
            PinningRow(
                case="nan",
                feature="v__min_scalar",
                inputs={"v": (DataType.DOUBLE, (0.0,))},
                expected={"v__min_scalar": (DataType.DOUBLE, (0.0,))},
            ),
        ),
    )
    empty = PinningTable(operation="skipped_op", rules={"x": "unused"}, rows=())

    # Tables render in the order given, not sorted by name; the empty table is skipped.
    lines = render_pinning_block([first, empty, second]).split("\n")

    assert lines == [
        PINNING_BEGIN_MARKER,
        "",
        _INTRO,
        "",
        "### Rules (second_op, first_op)",
        "",
        "| Case | Rule | Pinned in |",
        "|---|---|---|",
        "| none | Skips None. | second_op |",
        "| nan | NaN is a value. | second_op, first_op |",
        "",
        "### second_op",
        "",
        "| Case | Feature | Context | Input | Expected |",
        "|---|---|---|---|---|",
        "| none | `v__sum_aggr` | partition_by=['k', 'j'], ddof=1 | k=['a\\|b', 'c'] (STRING); v=[1, None] (INT64)"
        " | v__sum_aggr=[1, None] (INT64) |",
        "| nan | `v__max_aggr` | - | v=[NaN, -0.0, inf, -inf] (DOUBLE) | v__max_aggr=[NaN] (DOUBLE) |",
        "",
        "### first_op",
        "",
        "| Case | Feature | Context | Input | Expected |",
        "|---|---|---|---|---|",
        "| nan | `v__min_scalar` | - | v=[0.0] (DOUBLE) | v__min_scalar=[0.0] (DOUBLE) |",
        "",
        PINNING_END_MARKER,
        "",
    ]


def test_render_pinning_block_splits_tables_with_different_rules() -> None:
    row = PinningRow(
        case="a", feature="v__x", inputs={"v": (DataType.INT64, (1,))}, expected={"v__x": (DataType.INT64, (1,))}
    )
    one = PinningTable(operation="one", rules={"a": "Rule one."}, rows=(row,))
    two = PinningTable(operation="two", rules={"a": "Rule two."}, rows=(row,))

    text = render_pinning_block([one, two])

    assert "### Rules (one)" in text
    assert "### Rules (two)" in text
    assert "| a | Rule one. | one |" in text
    assert "| a | Rule two. | two |" in text

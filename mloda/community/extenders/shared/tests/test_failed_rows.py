"""Tests for failed_rows_summary: distinct counts, sorted columns and reasons, never positions."""

from __future__ import annotations

from mloda.provider import FailedRows

from mloda.community.extenders.shared.failed_rows import failed_rows_summary

_FG = "mloda.testing.DummyFeatureGroup"


def test_empty_input_yields_zero_and_no_columns() -> None:
    assert failed_rows_summary([]) == {"count": 0, "columns": {}}


def test_positions_are_counted_distinct_per_column() -> None:
    summary = failed_rows_summary(
        [FailedRows(_FG, "price", (1, 2), "parse_error"), FailedRows(_FG, "price", (2, 3), "no_match")]
    )

    assert summary["columns"]["price"]["count"] == 3
    assert summary["count"] == 3


def test_a_position_failing_in_two_columns_counts_once_overall() -> None:
    summary = failed_rows_summary(
        [FailedRows(_FG, "price", (1, 2), "parse_error"), FailedRows(_FG, "qty", (2, 5), "null_input")]
    )

    assert summary["count"] == 3
    assert summary["columns"]["price"]["count"] == 2
    assert summary["columns"]["qty"]["count"] == 2


def test_reasons_are_sorted_and_distinct() -> None:
    summary = failed_rows_summary(
        [
            FailedRows(_FG, "price", (1,), "parse_error"),
            FailedRows(_FG, "price", (2,), "no_match"),
            FailedRows(_FG, "price", (3,), "parse_error"),
        ]
    )

    assert summary["columns"]["price"]["reasons"] == ["no_match", "parse_error"]


def test_columns_are_in_sorted_order() -> None:
    summary = failed_rows_summary([FailedRows(_FG, "qty", (1,), "a"), FailedRows(_FG, "price", (2,), "a")])

    assert list(summary["columns"]) == ["price", "qty"]


def test_accepts_any_iterable() -> None:
    summary = failed_rows_summary(iter([FailedRows(_FG, "price", (7,), "a")]))

    assert summary["count"] == 1


def test_result_carries_no_positions() -> None:
    summary = failed_rows_summary([FailedRows(_FG, "price", (4242, 4243), "parse_error")])

    assert "4242" not in repr(summary)
    assert "4243" not in repr(summary)

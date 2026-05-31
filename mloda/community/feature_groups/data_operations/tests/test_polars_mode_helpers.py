"""Unit tests for polars_mode_helpers shared utilities."""

from __future__ import annotations

from typing import Any

import pytest

pl = pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.polars_mode_helpers import (
    ModeHelperCols,
    add_mode_helper_cols,
    drop_mode_helper_cols,
    mode_agg_expr,
    mode_window_expr,
)


def _pick(lf: Any, feature_name: str = "mode_v") -> ModeHelperCols:
    return ModeHelperCols.pick(set(lf.collect_schema().names()) | {feature_name})


class TestAddDropModeHelperCols:
    def test_add_mode_helper_cols_adds_three_columns(self) -> None:
        lf = pl.DataFrame({"r": ["a", "a", "b"], "v": [1, 2, 3]}).lazy()
        cols = _pick(lf)
        result = add_mode_helper_cols(lf, "v", ["r"], cols).collect()
        names = set(result.columns)
        assert cols.idx in names
        assert cols.cnt in names
        assert cols.first in names

    def test_drop_mode_helper_cols_removes_them(self) -> None:
        lf = pl.DataFrame({"r": ["a", "a", "b"], "v": [1, 2, 3]}).lazy()
        cols = _pick(lf)
        added = add_mode_helper_cols(lf, "v", ["r"], cols)
        dropped = drop_mode_helper_cols(added, cols).collect()
        assert set(dropped.columns) == {"r", "v"}

    def test_drop_mode_helper_cols_no_error_when_missing(self) -> None:
        """drop should use strict=False so it is safe to call even if cols are missing."""
        lf = pl.DataFrame({"r": ["a"], "v": [1]}).lazy()
        cols = _pick(lf)
        result = drop_mode_helper_cols(lf, cols).collect()
        assert set(result.columns) == {"r", "v"}

    def test_pick_avoids_collision_with_existing_helper_named_columns(self) -> None:
        """When the input already carries the default helper names, pick chooses suffixed names."""
        lf = pl.DataFrame(
            {
                "r": ["a", "a", "b"],
                "v": [1, 2, 3],
                "__mloda_mode_idx__": [10, 20, 30],
                "__mloda_mode_cnt__": [1, 1, 1],
                "__mloda_mode_first__": [0, 0, 0],
            }
        ).lazy()
        cols = _pick(lf)
        assert cols.idx == "__mloda_mode_idx___1"
        assert cols.cnt == "__mloda_mode_cnt___1"
        assert cols.first == "__mloda_mode_first___1"
        # The user columns survive the round-trip.
        result = drop_mode_helper_cols(add_mode_helper_cols(lf, "v", ["r"], cols), cols).collect()
        assert "__mloda_mode_idx__" in result.columns
        assert "__mloda_mode_cnt__" in result.columns
        assert "__mloda_mode_first__" in result.columns


class TestModeAggExpr:
    def test_mode_agg_expr_clear_winner(self) -> None:
        df = pl.DataFrame({"r": ["a", "a", "a", "b", "b"], "v": [1, 1, 2, 3, 3]}).lazy()
        cols = _pick(df)
        result = (
            add_mode_helper_cols(df, "v", ["r"], cols)
            .group_by(["r"], maintain_order=True)
            .agg(mode_agg_expr("v", "mode_v", cols))
            .sort("r")
            .collect()
        )
        out = dict(zip(result["r"].to_list(), result["mode_v"].to_list()))
        assert out["a"] == 1
        assert out["b"] == 3

    def test_mode_agg_expr_tie_breaks_by_first_seen(self) -> None:
        # In group "a": 2 appears first (idx 0), 5 appears later (idx 1). Both have count 2.
        df = pl.DataFrame({"r": ["a", "a", "a", "a"], "v": [2, 5, 2, 5]}).lazy()
        cols = _pick(df)
        result = (
            add_mode_helper_cols(df, "v", ["r"], cols)
            .group_by(["r"], maintain_order=True)
            .agg(mode_agg_expr("v", "mode_v", cols))
            .collect()
        )
        assert result["mode_v"].to_list() == [2]

    def test_mode_agg_expr_single_value(self) -> None:
        df = pl.DataFrame({"r": ["a"], "v": [42]}).lazy()
        cols = _pick(df)
        result = (
            add_mode_helper_cols(df, "v", ["r"], cols)
            .group_by(["r"], maintain_order=True)
            .agg(mode_agg_expr("v", "mode_v", cols))
            .collect()
        )
        assert result["mode_v"].to_list() == [42]

    def test_mode_agg_expr_all_nulls(self) -> None:
        df = pl.DataFrame({"r": ["a", "a"], "v": [None, None]}, schema={"r": pl.Utf8, "v": pl.Int64}).lazy()
        cols = _pick(df)
        result = (
            add_mode_helper_cols(df, "v", ["r"], cols)
            .group_by(["r"], maintain_order=True)
            .agg(mode_agg_expr("v", "mode_v", cols))
            .collect()
        )
        assert result["mode_v"].to_list() == [None]

    def test_mode_agg_expr_mixed_nulls_with_clear_winner(self) -> None:
        df = pl.DataFrame(
            {"r": ["a", "a", "a", "a", "a"], "v": [None, 7, 7, None, 9]},
            schema={"r": pl.Utf8, "v": pl.Int64},
        ).lazy()
        cols = _pick(df)
        result = (
            add_mode_helper_cols(df, "v", ["r"], cols)
            .group_by(["r"], maintain_order=True)
            .agg(mode_agg_expr("v", "mode_v", cols))
            .collect()
        )
        assert result["mode_v"].to_list() == [7]


class TestModeWindowExpr:
    def test_mode_window_expr_basic(self) -> None:
        df = pl.DataFrame({"r": ["a", "a", "a", "b", "b"], "v": [1, 1, 2, 3, 3]}).lazy()
        cols = _pick(df)
        result = (
            add_mode_helper_cols(df, "v", ["r"], cols)
            .with_columns(mode_window_expr("v", ["r"], "mode_v", cols))
            .sort(["r"])
            .collect()
        )
        # All rows in partition "a" get 1, all rows in partition "b" get 3.
        a_rows = result.filter(pl.col("r") == "a")["mode_v"].to_list()
        b_rows = result.filter(pl.col("r") == "b")["mode_v"].to_list()
        assert a_rows == [1, 1, 1]
        assert b_rows == [3, 3]

    def test_mode_window_expr_tie_breaks_by_first_seen(self) -> None:
        df = pl.DataFrame({"r": ["a", "a", "a", "a"], "v": [2, 5, 2, 5]}).lazy()
        cols = _pick(df)
        result = (
            add_mode_helper_cols(df, "v", ["r"], cols)
            .with_columns(mode_window_expr("v", ["r"], "mode_v", cols))
            .collect()
        )
        assert result["mode_v"].to_list() == [2, 2, 2, 2]

    def test_mode_window_expr_all_nulls(self) -> None:
        df = pl.DataFrame({"r": ["a", "a"], "v": [None, None]}, schema={"r": pl.Utf8, "v": pl.Int64}).lazy()
        cols = _pick(df)
        result = (
            add_mode_helper_cols(df, "v", ["r"], cols)
            .with_columns(mode_window_expr("v", ["r"], "mode_v", cols))
            .collect()
        )
        assert result["mode_v"].to_list() == [None, None]

    def test_mode_window_expr_mixed_nulls_with_clear_winner(self) -> None:
        df = pl.DataFrame(
            {"r": ["a", "a", "a", "a", "a"], "v": [None, 7, 7, None, 9]},
            schema={"r": pl.Utf8, "v": pl.Int64},
        ).lazy()
        cols = _pick(df)
        result = (
            add_mode_helper_cols(df, "v", ["r"], cols)
            .with_columns(mode_window_expr("v", ["r"], "mode_v", cols))
            .collect()
        )
        assert result["mode_v"].to_list() == [7, 7, 7, 7, 7]

    def test_mode_window_expr_multi_key_partition(self) -> None:
        df = pl.DataFrame(
            {
                "r1": ["a", "a", "a", "a", "b", "b"],
                "r2": ["x", "x", "y", "y", "x", "x"],
                "v": [1, 1, 2, 3, 4, 5],
            }
        ).lazy()
        cols = _pick(df)
        result = (
            add_mode_helper_cols(df, "v", ["r1", "r2"], cols)
            .with_columns(mode_window_expr("v", ["r1", "r2"], "mode_v", cols))
            .sort(["r1", "r2"])
            .collect()
        )
        # Partition (a, x): [1, 1] -> 1
        # Partition (a, y): [2, 3] tied count 1 each, first seen is 2
        # Partition (b, x): [4, 5] tied count 1 each, first seen is 4
        ax = result.filter((pl.col("r1") == "a") & (pl.col("r2") == "x"))["mode_v"].to_list()
        ay = result.filter((pl.col("r1") == "a") & (pl.col("r2") == "y"))["mode_v"].to_list()
        bx = result.filter((pl.col("r1") == "b") & (pl.col("r2") == "x"))["mode_v"].to_list()
        assert ax == [1, 1]
        assert ay == [2, 2]
        assert bx == [4, 4]

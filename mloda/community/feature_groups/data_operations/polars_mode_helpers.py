"""Polars Lazy helpers for mode aggregation with first-occurrence tie-breaking.

The three helper columns are chosen collision-free at runtime via
``ModeHelperCols.pick`` (pass the names already present, i.e. the input columns
plus the output feature name) and threaded through ``add_mode_helper_cols``,
``mode_agg_expr`` / ``mode_window_expr`` and ``drop_mode_helper_cols``.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import polars as pl

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name


@dataclass(frozen=True)
class ModeHelperCols:
    """Collision-free names for the three mode helper columns."""

    idx: str
    cnt: str
    first: str

    @classmethod
    def pick(cls, taken: Collection[str]) -> ModeHelperCols:
        idx = unique_helper_name("__mloda_mode_idx__", taken)
        cnt = unique_helper_name("__mloda_mode_cnt__", set(taken) | {idx})
        first = unique_helper_name("__mloda_mode_first__", set(taken) | {idx, cnt})
        return cls(idx, cnt, first)

    def as_list(self) -> list[str]:
        return [self.idx, self.cnt, self.first]


def add_mode_helper_cols(
    data: pl.LazyFrame, source_col: str, partition_by: list[str], cols: ModeHelperCols
) -> pl.LazyFrame:
    data = data.with_row_index(cols.idx)
    data = data.with_columns(
        pl.col(source_col).count().over([*partition_by, source_col]).alias(cols.cnt),
        pl.col(cols.idx).min().over([*partition_by, source_col]).alias(cols.first),
    )
    return data


def drop_mode_helper_cols(data: pl.LazyFrame, cols: ModeHelperCols) -> pl.LazyFrame:
    return data.drop(cols.as_list(), strict=False)


def mode_agg_expr(source_col: str, feature_name: str, cols: ModeHelperCols) -> pl.Expr:
    return (
        pl.col(source_col)
        .sort_by([pl.col(cols.cnt), pl.col(cols.first)], descending=[True, False], maintain_order=True)
        .first()
        .alias(feature_name)
    )


def mode_window_expr(source_col: str, partition_by: list[str], feature_name: str, cols: ModeHelperCols) -> pl.Expr:
    return (
        pl.col(source_col)
        .sort_by([pl.col(cols.cnt), pl.col(cols.first)], descending=[True, False], maintain_order=True)
        .first()
        .over(partition_by)
        .alias(feature_name)
    )

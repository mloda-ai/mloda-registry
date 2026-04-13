"""Polars Lazy helpers for mode aggregation with first-occurrence tie-breaking."""

from __future__ import annotations

import polars as pl

_MODE_IDX = "__mloda_mode_idx__"
_MODE_CNT = "__mloda_mode_cnt__"
_MODE_FIRST = "__mloda_mode_first__"

MODE_HELPER_COLS = [_MODE_IDX, _MODE_CNT, _MODE_FIRST]


def add_mode_helper_cols(data: pl.LazyFrame, source_col: str, partition_by: list[str]) -> pl.LazyFrame:
    existing = set(data.collect_schema().names())
    for reserved in MODE_HELPER_COLS:
        if reserved in existing:
            raise ValueError(
                f"Column '{reserved}' is a reserved mloda mode helper column and must not be present in the input."
            )
    data = data.with_row_index(_MODE_IDX)
    data = data.with_columns(
        pl.col(source_col).count().over([*partition_by, source_col]).alias(_MODE_CNT),
        pl.col(_MODE_IDX).min().over([*partition_by, source_col]).alias(_MODE_FIRST),
    )
    return data


def drop_mode_helper_cols(data: pl.LazyFrame) -> pl.LazyFrame:
    return data.drop(MODE_HELPER_COLS, strict=False)


def mode_agg_expr(source_col: str, feature_name: str) -> pl.Expr:
    return (
        pl.col(source_col)
        .sort_by([pl.col(_MODE_CNT), pl.col(_MODE_FIRST)], descending=[True, False], maintain_order=True)
        .first()
        .alias(feature_name)
    )


def mode_window_expr(source_col: str, partition_by: list[str], feature_name: str) -> pl.Expr:
    return (
        pl.col(source_col)
        .sort_by([pl.col(_MODE_CNT), pl.col(_MODE_FIRST)], descending=[True, False], maintain_order=True)
        .first()
        .over(partition_by)
        .alias(feature_name)
    )

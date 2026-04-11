"""Polars LazyFrame framework test mixin."""

from __future__ import annotations

from typing import Any

import pyarrow as pa


class PolarsLazyTestMixin:
    """Mixin implementing adapter methods for Polars LazyFrame.

    Requires ``polars`` to be importable. Concrete test modules should guard
    with ``pytest.importorskip("polars")``.
    """

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        import polars as pl

        df = pl.from_arrow(arrow_table)
        assert isinstance(df, pl.DataFrame)
        return df.lazy()

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        collected = result.collect()
        return list(collected[column_name].to_list())

    def get_row_count(self, result: Any) -> int:
        return int(result.collect().height)

    def get_expected_type(self) -> Any:
        import polars as pl

        return pl.LazyFrame

"""Polars Lazy test data creator for data operations plugins."""

from __future__ import annotations

from typing import Any

from mloda.testing.data_creator.base import DataOperationsTestDataCreator


class PolarsLazyDataOpsTestDataCreator(DataOperationsTestDataCreator):
    """Converts the raw test data to a Polars LazyFrame."""

    @classmethod
    def create(cls) -> Any:
        import polars as pl

        return pl.LazyFrame(cls.get_raw_data())

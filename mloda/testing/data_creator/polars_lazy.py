"""Polars Lazy test data creator for data operations plugins."""

from __future__ import annotations

from typing import Any

from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.testing.data_creator.base import DataOperationsTestDataCreator


class PolarsLazyDataOpsTestDataCreator(DataOperationsTestDataCreator):
    compute_framework = PolarsLazyDataFrame

    @classmethod
    def create(cls) -> Any:
        import polars as pl

        return pl.LazyFrame(cls.get_raw_data())

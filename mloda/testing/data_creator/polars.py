"""Polars test data creator for data operations plugins."""

from __future__ import annotations

from typing import Any

from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

from mloda.testing.data_creator.base import DataOperationsTestDataCreator


class PolarsDataOpsTestDataCreator(DataOperationsTestDataCreator):
    compute_framework = PolarsDataFrame

    @classmethod
    def create(cls) -> Any:
        import polars as pl  # polars is optional: imported only when data is created

        return pl.DataFrame(cls.get_raw_data())

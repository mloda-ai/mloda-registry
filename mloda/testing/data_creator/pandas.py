"""Pandas test data creator for data operations plugins."""

from __future__ import annotations

from typing import Any

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.testing.data_creator.base import DataOperationsTestDataCreator


class PandasDataOpsTestDataCreator(DataOperationsTestDataCreator):
    compute_framework = PandasDataFrame

    @classmethod
    def create(cls) -> Any:
        import pandas as pd  # pandas is optional: imported only when data is created

        return pd.DataFrame(cls.get_raw_data())

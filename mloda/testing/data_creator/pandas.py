"""Pandas test data creator for data operations plugins."""

from __future__ import annotations

from typing import Any

from mloda.testing.data_creator.base import DataOperationsTestDataCreator


class PandasDataOpsTestDataCreator(DataOperationsTestDataCreator):
    """Converts the raw test data to a pandas DataFrame."""

    @classmethod
    def create(cls) -> Any:
        import pandas as pd

        return pd.DataFrame(cls.get_raw_data())

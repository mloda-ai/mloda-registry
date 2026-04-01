"""Pandas framework test mixin."""

from __future__ import annotations

from typing import Any

import pyarrow as pa


class PandasTestMixin:
    """Mixin implementing adapter methods for Pandas.

    Requires ``pandas`` to be importable at class-definition time.
    Concrete test modules should guard with ``pytest.importorskip("pandas")``.
    """

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table.to_pandas()

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        import pandas as pd

        series = result[column_name]
        return [None if pd.isna(v) else v for v in series.tolist()]

    def get_row_count(self, result: Any) -> int:
        return len(result)

    def get_expected_type(self) -> Any:
        import pandas as pd

        return pd.DataFrame

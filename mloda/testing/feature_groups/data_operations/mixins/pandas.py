"""Pandas framework test mixin."""

from __future__ import annotations

from typing import Any

import pyarrow as pa


def _nullable_int_types_mapper(arrow_type: pa.DataType) -> Any:
    """Map Arrow integer types to pandas nullable integer dtypes, others to None."""
    import pandas as pd

    mapping = {
        pa.int8(): pd.Int8Dtype(),
        pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(),
        pa.int64(): pd.Int64Dtype(),
        pa.uint8(): pd.UInt8Dtype(),
        pa.uint16(): pd.UInt16Dtype(),
        pa.uint32(): pd.UInt32Dtype(),
        pa.uint64(): pd.UInt64Dtype(),
    }
    return mapping.get(arrow_type)


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

    def compute_framework_class(self) -> Any:
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        return PandasDataFrame


class PandasNullableIntTestMixin(PandasTestMixin):
    """Pandas mixin whose fixtures hold integer columns as nullable Int dtypes, keeping nulls exact."""

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table.to_pandas(types_mapper=_nullable_int_types_mapper)

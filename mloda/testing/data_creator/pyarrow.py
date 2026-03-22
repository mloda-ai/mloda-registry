"""PyArrow test data creator for data operations plugins."""

from __future__ import annotations

from typing import Any

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.testing.data_creator.base import DataOperationsTestDataCreator


class PyArrowDataOpsTestDataCreator(DataOperationsTestDataCreator):
    compute_framework = PyArrowTable

    @classmethod
    def create(cls) -> Any:
        import pyarrow as pa

        raw = cls.get_raw_data()
        arrays: list[pa.Array] = []
        names: list[str] = []
        for col_name, values in raw.items():
            names.append(col_name)
            arrays.append(pa.array(values))
        return pa.table(dict(zip(names, arrays)))

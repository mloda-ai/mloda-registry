"""PyArrow test data creator for data operations plugins."""

from __future__ import annotations

from typing import Any

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.framework_adapter import to_arrow_table


class PyArrowDataOpsTestDataCreator(DataOperationsTestDataCreator):
    compute_framework = PyArrowTable

    @classmethod
    def create(cls) -> Any:
        return to_arrow_table(cls.typed_data())

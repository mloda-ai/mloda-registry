"""PythonDict framework test mixin."""

from __future__ import annotations

from typing import Any

import pyarrow as pa


class PythonDictTestMixin:
    """Mixin implementing adapter methods for PythonDict.

    Requires ``mloda_plugins`` (core) to be importable; the PythonDict framework
    ships as part of mloda core, so no ``pytest.importorskip`` guard is needed.
    """

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result[column_name])

    def get_row_count(self, result: Any) -> int:
        from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count

        return row_count(result)

    def get_expected_type(self) -> Any:
        return dict

    def compute_framework_class(self) -> Any:
        from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
            PythonDictFramework,
        )

        return PythonDictFramework

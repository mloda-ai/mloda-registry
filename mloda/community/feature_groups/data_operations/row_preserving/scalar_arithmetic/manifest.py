"""Entry-point manifest for mloda-community-scalar-arithmetic.

Lists the operation FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)

FEATURE_GROUPS: list[type[Any]] = [
    ScalarArithmeticFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_scalar_arithmetic", "DuckdbScalarArithmetic"),
            ("pandas_scalar_arithmetic", "PandasScalarArithmetic"),
            ("polars_lazy_scalar_arithmetic", "PolarsLazyScalarArithmetic"),
            ("pyarrow_scalar_arithmetic", "PyArrowScalarArithmetic"),
            ("python_dict_scalar_arithmetic", "PythonDictScalarArithmetic"),
            ("sqlite_scalar_arithmetic", "SqliteScalarArithmetic"),
        ],
    ),
]

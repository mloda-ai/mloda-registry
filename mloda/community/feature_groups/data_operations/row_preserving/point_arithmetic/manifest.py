"""Entry-point manifest for mloda-community-point-arithmetic.

Lists the operation FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)

FEATURE_GROUPS: list[type[Any]] = [
    PointArithmeticFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_point_arithmetic", "DuckdbPointArithmetic"),
            ("pandas_point_arithmetic", "PandasPointArithmetic"),
            ("polars_lazy_point_arithmetic", "PolarsLazyPointArithmetic"),
            ("pyarrow_point_arithmetic", "PyArrowPointArithmetic"),
            ("python_dict_point_arithmetic", "PythonDictPointArithmetic"),
            ("sqlite_point_arithmetic", "SqlitePointArithmetic"),
        ],
    ),
]

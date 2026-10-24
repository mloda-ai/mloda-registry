"""Entry-point manifest for mloda-community-scalar-aggregate.

Lists the scalar aggregate FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)

FEATURE_GROUPS: list[type[Any]] = [
    ScalarAggregateFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_scalar_aggregate", "DuckdbScalarAggregate"),
            ("pandas_scalar_aggregate", "PandasScalarAggregate"),
            ("polars_lazy_scalar_aggregate", "PolarsLazyScalarAggregate"),
            ("pyarrow_scalar_aggregate", "PyArrowScalarAggregate"),
            ("python_dict_scalar_aggregate", "PythonDictScalarAggregate"),
            ("sqlite_scalar_aggregate", "SqliteScalarAggregate"),
        ],
    ),
]

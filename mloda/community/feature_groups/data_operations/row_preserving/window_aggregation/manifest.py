"""Entry-point manifest for mloda-community-window-aggregation.

Lists the window aggregation FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

FEATURE_GROUPS: list[type[Any]] = [
    WindowAggregationFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_window_aggregation", "DuckdbWindowAggregation"),
            ("pandas_window_aggregation", "PandasWindowAggregation"),
            ("polars_lazy_window_aggregation", "PolarsLazyWindowAggregation"),
            ("pyarrow_window_aggregation", "PyArrowWindowAggregation"),
            ("python_dict_window_aggregation", "PythonDictWindowAggregation"),
            ("sqlite_window_aggregation", "SqliteWindowAggregation"),
        ],
    ),
]

"""Entry-point manifest for mloda-community-aggregation.

Lists the aggregation FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes

FEATURE_GROUPS: list[type[Any]] = [
    AggregationFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_aggregation", "DuckdbAggregation"),
            ("pandas_aggregation", "PandasAggregation"),
            ("polars_lazy_aggregation", "PolarsLazyAggregation"),
            ("pyarrow_aggregation", "PyArrowAggregation"),
            ("python_dict_aggregation", "PythonDictAggregation"),
            ("sqlite_aggregation", "SqliteAggregation"),
        ],
    ),
]

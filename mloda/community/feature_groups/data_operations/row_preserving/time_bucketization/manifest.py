"""Entry-point manifest for mloda-community-time-bucketization.

Lists the FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TimeBucketizationFeatureGroup,
)

FEATURE_GROUPS: list[type[Any]] = [
    TimeBucketizationFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_time_bucketization", "DuckdbTimeBucketization"),
            ("pandas_time_bucketization", "PandasTimeBucketization"),
            ("polars_lazy_time_bucketization", "PolarsLazyTimeBucketization"),
            ("pyarrow_time_bucketization", "PyArrowTimeBucketization"),
            ("python_dict_time_bucketization", "PythonDictTimeBucketization"),
            ("sqlite_time_bucketization", "SqliteTimeBucketization"),
        ],
    ),
]

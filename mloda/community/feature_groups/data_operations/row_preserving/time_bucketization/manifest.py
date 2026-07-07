"""Entry-point manifest for mloda-community-time-bucketization.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_time_bucketization import DuckdbTimeBucketization
from .pandas_time_bucketization import PandasTimeBucketization
from .polars_lazy_time_bucketization import PolarsLazyTimeBucketization
from .pyarrow_time_bucketization import PyArrowTimeBucketization
from .sqlite_time_bucketization import SqliteTimeBucketization

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbTimeBucketization,
    PandasTimeBucketization,
    PolarsLazyTimeBucketization,
    PyArrowTimeBucketization,
    SqliteTimeBucketization,
]

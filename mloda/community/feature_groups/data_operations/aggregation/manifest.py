"""Entry-point manifest for mloda-community-aggregation.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_aggregation import DuckdbAggregation
from .pandas_aggregation import PandasAggregation
from .polars_lazy_aggregation import PolarsLazyAggregation
from .pyarrow_aggregation import PyArrowAggregation
from .sqlite_aggregation import SqliteAggregation

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbAggregation,
    PandasAggregation,
    PolarsLazyAggregation,
    PyArrowAggregation,
    SqliteAggregation,
]

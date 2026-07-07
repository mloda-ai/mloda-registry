"""Entry-point manifest for mloda-community-window-aggregation.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_window_aggregation import DuckdbWindowAggregation
from .pandas_window_aggregation import PandasWindowAggregation
from .polars_lazy_window_aggregation import PolarsLazyWindowAggregation
from .pyarrow_window_aggregation import PyArrowWindowAggregation
from .sqlite_window_aggregation import SqliteWindowAggregation

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbWindowAggregation,
    PandasWindowAggregation,
    PolarsLazyWindowAggregation,
    PyArrowWindowAggregation,
    SqliteWindowAggregation,
]

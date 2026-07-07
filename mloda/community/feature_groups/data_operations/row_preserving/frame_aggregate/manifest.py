"""Entry-point manifest for mloda-community-frame-aggregate.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_frame_aggregate import DuckdbFrameAggregate
from .pandas_frame_aggregate import PandasFrameAggregate
from .polars_lazy_frame_aggregate import PolarsLazyFrameAggregate
from .sqlite_frame_aggregate import SqliteFrameAggregate

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbFrameAggregate,
    PandasFrameAggregate,
    PolarsLazyFrameAggregate,
    SqliteFrameAggregate,
]

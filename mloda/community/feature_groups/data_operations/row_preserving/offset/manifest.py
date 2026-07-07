"""Entry-point manifest for mloda-community-offset.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_offset import DuckdbOffset
from .pandas_offset import PandasOffset
from .polars_lazy_offset import PolarsLazyOffset
from .sqlite_offset import SqliteOffset

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbOffset,
    PandasOffset,
    PolarsLazyOffset,
    SqliteOffset,
]

"""Entry-point manifest for mloda-community-rank.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_rank import DuckdbRank
from .pandas_rank import PandasRank
from .polars_lazy_rank import PolarsLazyRank
from .sqlite_rank import SqliteRank

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbRank,
    PandasRank,
    PolarsLazyRank,
    SqliteRank,
]

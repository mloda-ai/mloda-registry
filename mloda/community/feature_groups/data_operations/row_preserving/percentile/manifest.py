"""Entry-point manifest for mloda-community-percentile.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_percentile import DuckdbPercentile
from .pandas_percentile import PandasPercentile
from .polars_lazy_percentile import PolarsLazyPercentile

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbPercentile,
    PandasPercentile,
    PolarsLazyPercentile,
]

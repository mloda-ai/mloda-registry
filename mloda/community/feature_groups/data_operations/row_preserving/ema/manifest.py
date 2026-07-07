"""Entry-point manifest for mloda-community-ema.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_ema import DuckdbEma
from .pandas_ema import PandasEma
from .polars_lazy_ema import PolarsLazyEma
from .pyarrow_ema import PyArrowEma
from .sqlite_ema import SqliteEma

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbEma,
    PandasEma,
    PolarsLazyEma,
    PyArrowEma,
    SqliteEma,
]

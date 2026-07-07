"""Entry-point manifest for mloda-community-ffill.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_ffill import DuckdbFfill
from .pandas_ffill import PandasFfill
from .polars_lazy_ffill import PolarsLazyFfill
from .pyarrow_ffill import PyArrowFfill
from .sqlite_ffill import SqliteFfill

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbFfill,
    PandasFfill,
    PolarsLazyFfill,
    PyArrowFfill,
    SqliteFfill,
]

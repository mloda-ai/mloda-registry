"""Entry-point manifest for mloda-community-string.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_string import DuckdbStringOps
from .pandas_string import PandasStringOps
from .polars_lazy_string import PolarsLazyStringOps
from .pyarrow_string import PyArrowStringOps
from .sqlite_string import SqliteStringOps

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbStringOps,
    PandasStringOps,
    PolarsLazyStringOps,
    PyArrowStringOps,
    SqliteStringOps,
]

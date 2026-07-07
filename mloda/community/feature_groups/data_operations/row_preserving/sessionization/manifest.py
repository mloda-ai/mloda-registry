"""Entry-point manifest for mloda-community-sessionization.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_sessionization import DuckdbSessionization
from .pandas_sessionization import PandasSessionization
from .polars_lazy_sessionization import PolarsLazySessionization
from .pyarrow_sessionization import PyArrowSessionization
from .sqlite_sessionization import SqliteSessionization

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbSessionization,
    PandasSessionization,
    PolarsLazySessionization,
    PyArrowSessionization,
    SqliteSessionization,
]

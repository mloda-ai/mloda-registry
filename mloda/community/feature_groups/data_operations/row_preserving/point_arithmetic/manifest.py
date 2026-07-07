"""Entry-point manifest for mloda-community-point-arithmetic.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_point_arithmetic import DuckdbPointArithmetic
from .pandas_point_arithmetic import PandasPointArithmetic
from .polars_lazy_point_arithmetic import PolarsLazyPointArithmetic
from .pyarrow_point_arithmetic import PyArrowPointArithmetic
from .sqlite_point_arithmetic import SqlitePointArithmetic

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbPointArithmetic,
    PandasPointArithmetic,
    PolarsLazyPointArithmetic,
    PyArrowPointArithmetic,
    SqlitePointArithmetic,
]

"""Entry-point manifest for mloda-community-scalar-arithmetic.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_scalar_arithmetic import DuckdbScalarArithmetic
from .pandas_scalar_arithmetic import PandasScalarArithmetic
from .polars_lazy_scalar_arithmetic import PolarsLazyScalarArithmetic
from .pyarrow_scalar_arithmetic import PyArrowScalarArithmetic
from .sqlite_scalar_arithmetic import SqliteScalarArithmetic

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbScalarArithmetic,
    PandasScalarArithmetic,
    PolarsLazyScalarArithmetic,
    PyArrowScalarArithmetic,
    SqliteScalarArithmetic,
]

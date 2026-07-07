"""Entry-point manifest for mloda-community-scalar-aggregate.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_scalar_aggregate import DuckdbScalarAggregate
from .pandas_scalar_aggregate import PandasScalarAggregate
from .polars_lazy_scalar_aggregate import PolarsLazyScalarAggregate
from .pyarrow_scalar_aggregate import PyArrowScalarAggregate
from .sqlite_scalar_aggregate import SqliteScalarAggregate

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbScalarAggregate,
    PandasScalarAggregate,
    PolarsLazyScalarAggregate,
    PyArrowScalarAggregate,
    SqliteScalarAggregate,
]

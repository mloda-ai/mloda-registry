"""Window aggregation feature group (broadcast, partition_by, row-preserving)."""

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

try:  # noqa: SIM105 (optional framework import)
    from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
        PyArrowWindowAggregation,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.sqlite_window_aggregation import (
        SqliteWindowAggregation,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.duckdb_window_aggregation import (
        DuckdbWindowAggregation,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.polars_lazy_window_aggregation import (
        PolarsLazyWindowAggregation,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pandas_window_aggregation import (
        PandasWindowAggregation,
    )
except ImportError:
    pass

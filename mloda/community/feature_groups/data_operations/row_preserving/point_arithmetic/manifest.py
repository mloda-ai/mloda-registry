"""Entry-point manifest for mloda-community-point-arithmetic.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes

FEATURE_GROUPS: list[type[FeatureGroup]] = load_plugin_classes(
    __package__ or __name__.rpartition(".")[0],
    [
        ("duckdb_point_arithmetic", "DuckdbPointArithmetic"),
        ("pandas_point_arithmetic", "PandasPointArithmetic"),
        ("polars_lazy_point_arithmetic", "PolarsLazyPointArithmetic"),
        ("pyarrow_point_arithmetic", "PyArrowPointArithmetic"),
        ("sqlite_point_arithmetic", "SqlitePointArithmetic"),
    ],
)

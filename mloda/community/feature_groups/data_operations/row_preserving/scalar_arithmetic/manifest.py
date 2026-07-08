"""Entry-point manifest for mloda-community-scalar-arithmetic.

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
        ("duckdb_scalar_arithmetic", "DuckdbScalarArithmetic"),
        ("pandas_scalar_arithmetic", "PandasScalarArithmetic"),
        ("polars_lazy_scalar_arithmetic", "PolarsLazyScalarArithmetic"),
        ("pyarrow_scalar_arithmetic", "PyArrowScalarArithmetic"),
        ("sqlite_scalar_arithmetic", "SqliteScalarArithmetic"),
    ],
)

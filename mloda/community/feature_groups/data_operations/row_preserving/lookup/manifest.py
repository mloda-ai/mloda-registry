"""Entry-point manifest for mloda-community-lookup."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import LookupFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    LookupFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_lookup", "DuckdbLookup"),
            ("pandas_lookup", "PandasLookup"),
            ("polars_lazy_lookup", "PolarsLazyLookup"),
            ("pyarrow_lookup", "PyArrowLookup"),
            ("python_dict_lookup", "PythonDictLookup"),
            ("sqlite_lookup", "SqliteLookup"),
        ],
    ),
]

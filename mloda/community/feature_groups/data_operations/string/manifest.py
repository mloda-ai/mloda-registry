"""Entry-point manifest for mloda-community-string.

Lists the FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.string.base import StringFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    StringFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_string", "DuckdbStringOps"),
            ("pandas_string", "PandasStringOps"),
            ("polars_lazy_string", "PolarsLazyStringOps"),
            ("pyarrow_string", "PyArrowStringOps"),
            ("python_dict_string", "PythonDictStringOps"),
            ("sqlite_string", "SqliteStringOps"),
        ],
    ),
]

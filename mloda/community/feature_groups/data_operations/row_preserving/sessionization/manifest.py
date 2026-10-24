"""Entry-point manifest for mloda-community-sessionization.

Lists the FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import SessionizationFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    SessionizationFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_sessionization", "DuckdbSessionization"),
            ("pandas_sessionization", "PandasSessionization"),
            ("polars_lazy_sessionization", "PolarsLazySessionization"),
            ("pyarrow_sessionization", "PyArrowSessionization"),
            ("python_dict_sessionization", "PythonDictSessionization"),
            ("sqlite_sessionization", "SqliteSessionization"),
        ],
    ),
]

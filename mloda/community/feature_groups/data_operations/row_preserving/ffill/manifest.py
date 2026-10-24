"""Entry-point manifest for mloda-community-ffill.

Lists the FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    FfillFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_ffill", "DuckdbFfill"),
            ("pandas_ffill", "PandasFfill"),
            ("polars_lazy_ffill", "PolarsLazyFfill"),
            ("pyarrow_ffill", "PyArrowFfill"),
            ("python_dict_ffill", "PythonDictFfill"),
            ("sqlite_ffill", "SqliteFfill"),
        ],
    ),
]

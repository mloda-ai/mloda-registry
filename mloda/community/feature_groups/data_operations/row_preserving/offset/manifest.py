"""Entry-point manifest for mloda-community-offset.

Lists the offset FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import OffsetFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    OffsetFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_offset", "DuckdbOffset"),
            ("pandas_offset", "PandasOffset"),
            ("polars_lazy_offset", "PolarsLazyOffset"),
            ("python_dict_offset", "PythonDictOffset"),
            ("sqlite_offset", "SqliteOffset"),
        ],
    ),
]

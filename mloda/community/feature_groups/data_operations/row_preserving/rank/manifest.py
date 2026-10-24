"""Entry-point manifest for mloda-community-rank.

Lists the rank FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.rank.base import RankFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    RankFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_rank", "DuckdbRank"),
            ("pandas_rank", "PandasRank"),
            ("polars_lazy_rank", "PolarsLazyRank"),
            ("python_dict_rank", "PythonDictRank"),
            ("sqlite_rank", "SqliteRank"),
        ],
    ),
]

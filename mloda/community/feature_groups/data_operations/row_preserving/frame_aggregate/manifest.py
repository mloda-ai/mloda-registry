"""Entry-point manifest for mloda-community-frame-aggregate.

Lists the frame aggregate FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)

FEATURE_GROUPS: list[type[Any]] = [
    FrameAggregateFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_frame_aggregate", "DuckdbFrameAggregate"),
            ("pandas_frame_aggregate", "PandasFrameAggregate"),
            ("polars_lazy_frame_aggregate", "PolarsLazyFrameAggregate"),
            ("python_dict_frame_aggregate", "PythonDictFrameAggregate"),
            ("sqlite_frame_aggregate", "SqliteFrameAggregate"),
        ],
    ),
]

"""Entry-point manifest for mloda-community-percentile.

Lists the percentile FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import PercentileFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    PercentileFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_percentile", "DuckdbPercentile"),
            ("pandas_percentile", "PandasPercentile"),
            ("polars_lazy_percentile", "PolarsLazyPercentile"),
            ("python_dict_percentile", "PythonDictPercentile"),
        ],
    ),
]

"""Entry-point manifest for mloda-community-binning.

Lists the FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.binning.base import BinningFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    BinningFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_binning", "DuckdbBinning"),
            ("pandas_binning", "PandasBinning"),
            ("polars_lazy_binning", "PolarsLazyBinning"),
            ("pyarrow_binning", "PyArrowBinning"),
            ("python_dict_binning", "PythonDictBinning"),
            ("sqlite_binning", "SqliteBinning"),
        ],
    ),
]

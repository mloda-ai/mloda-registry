"""Entry-point manifest for mloda-community-resample.

Lists the FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_changing.resample.base import ResampleFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    ResampleFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_resample", "DuckdbResample"),
            ("pandas_resample", "PandasResample"),
            ("polars_lazy_resample", "PolarsLazyResample"),
            ("pyarrow_resample", "PyArrowResample"),
            ("python_dict_resample", "PythonDictResample"),
        ],
    ),
]

"""Entry-point manifest for mloda-community-ema.

Lists the FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.ema.base import EmaFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    EmaFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("pandas_ema", "PandasEma"),
            ("polars_lazy_ema", "PolarsLazyEma"),
            ("python_dict_ema", "PythonDictEma"),
        ],
    ),
]

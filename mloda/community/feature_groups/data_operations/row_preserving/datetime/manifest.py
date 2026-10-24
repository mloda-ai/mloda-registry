"""Entry-point manifest for mloda-community-datetime.

Lists the FeatureGroup and its Implementation classes that mloda discovers via the
``mloda.feature_groups`` entry point. Backends whose optional framework is not
installed are skipped so the rest still register. See issue #271.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes
from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import DateTimeFeatureGroup

FEATURE_GROUPS: list[type[Any]] = [
    DateTimeFeatureGroup,
    *load_plugin_classes(
        __package__ or __name__.rpartition(".")[0],
        [
            ("duckdb_datetime", "DuckdbDateTimeExtraction"),
            ("pandas_datetime", "PandasDateTimeExtraction"),
            ("polars_lazy_datetime", "PolarsLazyDateTimeExtraction"),
            ("pyarrow_datetime", "PyArrowDateTimeExtraction"),
            ("python_dict_datetime", "PythonDictDateTimeExtraction"),
            ("sqlite_datetime", "SqliteDateTimeExtraction"),
        ],
    ),
]

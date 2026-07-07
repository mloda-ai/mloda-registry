"""Entry-point manifest for mloda-community-datetime.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_datetime import DuckdbDateTimeExtraction
from .pandas_datetime import PandasDateTimeExtraction
from .polars_lazy_datetime import PolarsLazyDateTimeExtraction
from .pyarrow_datetime import PyArrowDateTimeExtraction
from .sqlite_datetime import SqliteDateTimeExtraction

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbDateTimeExtraction,
    PandasDateTimeExtraction,
    PolarsLazyDateTimeExtraction,
    PyArrowDateTimeExtraction,
    SqliteDateTimeExtraction,
]

"""Entry-point manifest for mloda-community-resample.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_resample import DuckdbResample
from .pandas_resample import PandasResample
from .polars_lazy_resample import PolarsLazyResample
from .pyarrow_resample import PyArrowResample

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbResample,
    PandasResample,
    PolarsLazyResample,
    PyArrowResample,
]

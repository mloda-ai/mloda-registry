"""Entry-point manifest for mloda-community-binning.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .duckdb_binning import DuckdbBinning
from .pandas_binning import PandasBinning
from .polars_lazy_binning import PolarsLazyBinning
from .pyarrow_binning import PyArrowBinning
from .sqlite_binning import SqliteBinning

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    DuckdbBinning,
    PandasBinning,
    PolarsLazyBinning,
    PyArrowBinning,
    SqliteBinning,
]

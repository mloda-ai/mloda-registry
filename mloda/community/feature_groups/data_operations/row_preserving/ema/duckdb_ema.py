"""DuckDB EMA backend: rejects EMA up-front.

DuckDB has no native exponentially weighted (EWM) aggregate, and a recursive /
Python emulation is forbidden by the CFW-backend rule. EMA is therefore
rejected at validation time with a clear ``ValueError``.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

from mloda.community.feature_groups.data_operations.row_preserving.ema.base import EmaFeatureGroup


class DuckdbEma(EmaFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise ValueError(
            "EMA (exponential moving average) is not supported on the DuckDB backend: "
            "DuckDB has no native exponentially weighted aggregate and a recursive Python "
            "emulation is forbidden. Use the pandas or polars-lazy backend instead."
        )

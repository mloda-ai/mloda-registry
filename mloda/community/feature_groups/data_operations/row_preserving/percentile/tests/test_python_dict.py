"""Tests for PythonDictPercentile compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.percentile.python_dict_percentile import (
    PythonDictPercentile,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)


class TestPythonDictPercentile(PythonDictTestMixin, PercentileTestBase):
    """All tests inherited from the base class.

    No overrides: PythonDict aims for the same full support level as pandas,
    DuckDB, and polars-lazy (PERCENTILE_CONT-style linear interpolation over
    every partition, including multi-key partitions and null skipping).
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictPercentile

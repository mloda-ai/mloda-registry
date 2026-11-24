"""Tests for PandasPercentile compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.percentile.pandas_percentile import (
    PandasPercentile,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)


class TestPandasPercentile(PandasTestMixin, PercentileTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasPercentile

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        return {
            **super().pinned_divergences(),
            # Interpolation involving infinities returns null instead of the pinned endpoint or exact median.
            "4-endpoints-v__p0_percentile": {"v__p0_percentile": [None, None, None]},
            "5-endpoints-v__p100_percentile": {"v__p100_percentile": [None, None, None]},
            "6-endpoints-v__p50_percentile": {"v__p50_percentile": [None, None, None]},
            "7-endpoints-v__p50_percentile": {"v__p50_percentile": [None, None]},
            # NaN and None partition keys merge, sharing one median.
            "16-partition_keys-v__p50_percentile": {"v__p50_percentile": [6.5, 6.5, 6.5, 6.5]},
        }


class TestPandasPercentileNullableInt(PandasNullableIntTestMixin, TestPandasPercentile):
    """Runs the percentile suite on nullable pandas int columns."""

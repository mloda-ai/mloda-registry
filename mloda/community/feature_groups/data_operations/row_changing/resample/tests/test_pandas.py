"""Tests for PandasResample compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_changing.resample.pandas_resample import (
    PandasResample,
)
from mloda.testing.feature_groups.data_operations.base import declares_nan_as_null
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_changing.resample.resample import (
    RESAMPLE_SORTED_BUCKET_DIVERGENCES,
    ResampleTestBase,
)


class TestPandasResample(PandasTestMixin, ResampleTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasResample

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        divergences: dict[str, Mapping[str, list[Any]]] = {
            **RESAMPLE_SORTED_BUCKET_DIVERGENCES,
            **super().pinned_divergences(),
        }
        nan_as_null_pins: dict[str, tuple[str, Mapping[str, list[Any]]]] = {
            "15-nan-v__resample_1_hour_sum": ("sum", {"v__resample_1_hour_sum": [1.0]}),
            "16-nan-v__resample_1_hour_mean": ("mean", {"v__resample_1_hour_mean": [1.0]}),
            "17-nan-v__resample_1_hour_count": ("count", {"v__resample_1_hour_count": [1]}),
        }
        for row_id, (subtype, columns) in nan_as_null_pins.items():
            # Pandas skips or does not count NaN.
            if declares_nan_as_null(cls.implementation_class(), frozenset({subtype})):
                divergences[row_id] = columns
        return divergences


class TestPandasResampleNullableInt(PandasNullableIntTestMixin, TestPandasResample):
    """Runs the resample suite on nullable pandas int columns."""

"""Tests for PandasFfill compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.ffill.pandas_ffill import (
    PandasFfill,
)
from mloda.testing.feature_groups.data_operations.base import declares_nan_as_null
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.ffill.ffill import (
    FfillTestBase,
)


class TestPandasFfill(PandasTestMixin, FfillTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasFfill

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        divergences: dict[str, Mapping[str, list[Any]]] = {
            # NaN and None partition keys merge, carrying a value across the reference partition boundary.
            "10-partition_keys-v__ffill": {"v__ffill": [1, 1, 1, 1]},
        }
        if declares_nan_as_null(cls.implementation_class()):
            # NaN and None tie in order_by, so ffill uses a different predecessor.
            divergences["3-order_tiers-v__ffill"] = {"v__ffill": [None, 20, None]}
            # A NaN source value is treated as a null to fill instead of as a value to carry.
            divergences["7-nan-v__ffill"] = {"v__ffill": [1.0, 1.0, 1.0]}
            divergences["8-nan-v__ffill"] = {"v__ffill": [None, None, None]}
        return divergences


class TestPandasFfillNullableInt(PandasNullableIntTestMixin, TestPandasFfill):
    """Runs the ffill suite on nullable pandas int columns."""

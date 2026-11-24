"""Tests for PandasEma compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.ema.pandas_ema import (
    PandasEma,
)
from mloda.testing.feature_groups.data_operations.base import declares_nan_as_null
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import (
    EmaTestBase,
)


class TestPandasEma(PandasTestMixin, EmaTestBase):
    """All value/semantics/error tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasEma

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        divergences: dict[str, Mapping[str, list[Any]]] = {
            # NaN and None partition keys merge, sharing the EMA state across reference partitions.
            "14-partition_keys-v__ema_3": {"v__ema_3": [2.0, 3.0, 4.5, 6.25]},
        }
        if declares_nan_as_null(cls.implementation_class()):
            # NaN and None tie in order_by, changing the EMA recurrence order.
            divergences["8-order_tiers-v__ema_3"] = {"v__ema_3": [3.0, 4.5, 4.0]}
            # Pandas treats a NaN source value as null instead of propagating NaN through the recurrence.
            divergences["12-nan-v__ema_2"] = {"v__ema_2": [10.0, None, 23.333333333333336, None, 41.11111111111111]}
        return divergences


class TestPandasEmaNullableInt(PandasNullableIntTestMixin, TestPandasEma):
    """Runs the ema suite on nullable pandas int columns."""

"""Tests for PandasSessionization compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.pandas_sessionization import (
    PandasSessionization,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestPandasSessionization(PandasTestMixin, SessionizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasSessionization

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        # NaN and None partition keys merge, changing session boundaries and IDs.
        return {"8-nan_none_keys-ts__sessionize_30_minute": {"ts__sessionize_30_minute": [1, 1, 0, 1, 1]}}


class TestPandasSessionizationNullableInt(PandasNullableIntTestMixin, TestPandasSessionization):
    """Runs the sessionization suite on nullable pandas int columns."""
